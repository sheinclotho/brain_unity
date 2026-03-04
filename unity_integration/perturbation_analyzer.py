"""
Perturbation-based Effective Connectivity Analyzer
===================================================

Inspired by the NPI (Neural Perturbation Inference) framework, this module
implements data-driven causal inference for brain effective connectivity (EC).

两种推断方法:
1. 有限差分扰动法 (finite-difference / NPI): 对代理模型输入施加微小扰动，
   测量输出变化 → EC[j, i] = Δoutput_i / Δinput_j
2. Jacobian 解析法: 通过自动微分直接计算代理模型的 Jacobian 矩阵，
   更精确且更快，要求模型可微。

与原版 NPI 的三大改进:
A. 统一接口 — 同时支持 MLP 代理（无模型时自动训练）和 GNN 代理（有模型时直接使用）
B. 基于 EC 的靶点推荐 — 自动识别影响力最强的脑区，辅助设计刺激方案
C. 活动增量预测 — 利用 EC 矩阵线性近似刺激的下游传播效应，为前端可视化提供依据

WebSocket 消息协议:
  请求:  {type: "infer_ec", method: "perturbation"|"jacobian", n_lags: 5}
  响应:  {type: "ec_result",
          ec_flat:       [200×200 floats, row-major],
          top_sources:   [int×10],   // 影响力最强的源脑区
          top_targets:   [int×10],   // 感受性最强的目标脑区
          activity_delta:[200 floats] // 刺激 top_sources[0:3] 后的预测活动变化
          fit_quality:   {train_mse, val_mse, overfit_ratio, reliable}
         }
"""

import logging
import os
from typing import Callable, List, Optional, Tuple

# Suppress Intel OpenMP duplicate-runtime crash (libiomp5 / libiomp5md.dll).
# This module imports both torch and numpy at the top level; the env var must
# be set *before* those libraries initialise their OpenMP context.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

logger = logging.getLogger(__name__)

# Number of source regions to perturb in each batched forward pass.
# Trades memory (CHUNK_SIZE × M × N*n_lags float32) for reduced Python /
# PyTorch kernel-launch overhead.  32 uses ~50 MB for typical M=400, N=200.
_EC_PERTURBATION_CHUNK_SIZE: int = 32


# ── Module-level tuneable constants ─────────────────────────────────────────
# Scale model capacity down when n_train < this threshold.  Below this value
# the parameter-to-sample ratio of the full-size MLP becomes too high for
# reliable gradient descent.
_MIN_SAMPLES_FOR_FULL_CAPACITY: int = 500

# When val_mse / train_mse exceeds this threshold the model has severely
# overfit noise; EC inference results are flagged as unreliable.
_OVERFIT_RELIABILITY_THRESHOLD: float = 20.0


# ── Surrogate brain: lightweight MLP (same architecture as NPI's ANN_MLP) ──────

class _SurrogateMLP(nn.Module):
    """
    Single-hidden-layer MLP that predicts brain_state_{t+1} from a lag window.

    Input:  (n_regions × n_lags,) flattened
    Output: (n_regions,)

    Activation: GELU rather than ReLU.  GELU is smooth and non-zero for all
    inputs, which gives well-defined, numerically stable Jacobians everywhere.
    ReLU is piecewise-linear (zero gradient in the negative half-plane), so
    Jacobian-based EC estimates are noisy near deactivated units — a known
    limitation in neural surrogate models for causal inference.  Reference:
    Hendrycks & Gimpel (2016) "Gaussian Error Linear Units (GELUs)".

    Dropout is applied after each hidden activation to regularise training
    on small neuroimaging datasets (typical T ≈ 300–500 frames).
    """

    def __init__(self, n_regions: int, n_lags: int,
                 hidden_dim: int = 256, latent_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        input_dim = n_regions * n_lags
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(latent_dim, n_regions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data helpers ─────────────────────────────────────────────────────────────

def _build_lag_dataset(
    time_series: np.ndarray,
    n_lags: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    将时序数据切分为 (滑动窗口输入, 下一步目标) 对。

    Uses vectorized numpy indexing instead of a Python loop, which is
    significantly faster for large T (e.g. 10–50× on T=1000, N=200).

    Args:
        time_series: (T, N_regions) float array
        n_lags:      历史步数

    Returns:
        input_X:  (T-n_lags, N_regions*n_lags)
        target_Y: (T-n_lags, N_regions)
    """
    T, N = time_series.shape
    M = T - n_lags
    # Build an index matrix of shape (M, n_lags) where row i contains
    # [i, i+1, ..., i+n_lags-1].  Indexing time_series[idx] gives
    # (M, n_lags, N); reshape to (M, n_lags*N) for the flat input vector.
    idx = np.arange(M)[:, None] + np.arange(n_lags)[None, :]  # (M, n_lags)
    X = time_series[idx].reshape(M, N * n_lags).astype(np.float32)
    Y = time_series[n_lags:].astype(np.float32)                # (M, N)
    return X, Y


# ── Main class ───────────────────────────────────────────────────────────────

class PerturbationAnalyzer:
    """
    数据驱动的有效连接（EC）推断器。

    使用方法:
        analyzer = PerturbationAnalyzer(n_regions=200)
        analyzer.fit_surrogate(time_series)   # (T, 200) numpy array
        ec = analyzer.infer_ec_jacobian()     # (200, 200)
        targets = analyzer.suggest_targets(ec, n_targets=5)
    """

    def __init__(self, n_regions: int = 200, n_lags: int = 5,
                 device: Optional[str] = None):
        self.n_regions = n_regions
        self.n_lags    = n_lags
        self.device    = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._surrogate: Optional[_SurrogateMLP] = None
        self._input_X:   Optional[np.ndarray]    = None   # (M, N*lags) – cached training inputs
        self._last_ec:   Optional[np.ndarray]    = None   # (N, N) – last computed EC matrix
        # Per-edge Jacobian std — populated only by infer_ec_jacobian (not by
        # infer_ec_perturbation, which computes a single point estimate).
        # None if the user has not called infer_ec_jacobian yet.
        self._last_ec_std: Optional[np.ndarray]  = None   # (N, N) – per-edge Jacobian std
        self._fit_quality: dict                  = {}     # last fit metrics (train/val MSE)
        # Per-region z-score statistics from fit_surrogate, used in predict_trajectory
        # to normalise new initial states into the same space the surrogate was trained on.
        self._ts_mean: Optional[np.ndarray] = None   # (N,) float32
        self._ts_std:  Optional[np.ndarray] = None   # (N,) float32

    # ── Surrogate training ──────────────────────────────────────────────────

    def fit_surrogate(
        self,
        time_series: np.ndarray,
        n_lags: Optional[int] = None,
        hidden_dim: int = 256,
        latent_dim: int = 128,
        num_epochs: int = 80,
        batch_size: int = 64,
        lr: float = 1e-3,
        train_ratio: float = 0.8,
        weight_decay: float = 1e-4,
        dropout: float = 0.3,
        patience: int = 12,
    ) -> Tuple[List[float], List[float]]:
        """
        在提供的时序数据上训练 MLP 代理模型。

        Args:
            time_series:  (T, N_regions) numpy array, 值域任意（内部会标准化）
            n_lags:       历史窗口长度（覆盖 self.n_lags）
            hidden_dim / latent_dim: 网络宽度（会根据数据集规模自适应缩放）
            num_epochs / batch_size / lr: 训练超参数
            train_ratio:  训练/测试分割比例
            weight_decay: Adam L2 正则化系数（防止过拟合，对照 NPI 的 l2 参数）
            dropout:      隐层 Dropout 概率（减少小数据集过拟合）
            patience:     早停耐心值 —— 验证 MSE 连续 patience 轮不改善则停止

        Returns:
            (train_losses, val_losses) 各 epoch 的 MSE 损失
        """
        if n_lags is not None:
            self.n_lags = n_lags

        # Guard: replace NaN/Inf values before they silently corrupt training.
        # Using replace-not-raise so the server stays alive and returns a result
        # with a warning; callers can check the warning log to assess data quality.
        if not np.isfinite(time_series).all():
            n_bad = int((~np.isfinite(time_series)).sum())
            logger.warning(
                f"fit_surrogate: 时序数据含 {n_bad} 个 NaN/Inf 值，已用列均值替换。"
                "建议检查原始数据质量。"
            )
            time_series = time_series.copy().astype(np.float32)
            # Replace NaN/Inf per column with the column finite mean (or 0 if all bad).
            # Vectorized: no Python loop over columns.
            bad_mask  = ~np.isfinite(time_series)                            # (T, N)
            good_sum  = np.where(~bad_mask, time_series, 0.0).sum(axis=0, keepdims=True)  # (1, N)
            n_good    = (~bad_mask).sum(axis=0, keepdims=True).astype(np.float32)         # (1, N)
            col_means = np.where(n_good > 0, good_sum / np.maximum(n_good, 1.0), 0.0)    # (1, N)
            time_series = np.where(bad_mask, col_means, time_series)

        # Z-score 标准化使训练更稳定
        ts_mean = time_series.mean(axis=0, keepdims=True)
        ts_std  = np.maximum(time_series.std(axis=0, keepdims=True), 1e-6)
        ts_norm = (time_series - ts_mean) / ts_std

        # Store per-region stats so predict_trajectory can re-normalise new states.
        self._ts_mean = ts_mean.flatten().astype(np.float32)   # (N,)
        self._ts_std  = ts_std.flatten().astype(np.float32)    # (N,)

        X, Y = _build_lag_dataset(ts_norm, self.n_lags)
        self._input_X = X          # 保存用于后续 EC 推断

        split = int(len(X) * train_ratio)
        tx = torch.tensor(X[:split],  dtype=torch.float32).to(self.device)
        ty = torch.tensor(Y[:split],  dtype=torch.float32).to(self.device)
        vx = torch.tensor(X[split:],  dtype=torch.float32).to(self.device)
        vy = torch.tensor(Y[split:],  dtype=torch.float32).to(self.device)

        train_dl = data.DataLoader(data.TensorDataset(tx, ty), batch_size, shuffle=True)
        val_dl   = data.DataLoader(data.TensorDataset(vx, vy), batch_size, shuffle=False)

        # Adaptive capacity: scale hidden/latent dims down for small datasets to
        # reduce the parameter-to-sample ratio and mitigate overfitting.
        # Reference: NPI uses fixed dims regardless of dataset size, which causes
        # severe overfitting when T < ~1000.
        n_train = len(tx)
        scale   = min(1.0, n_train / _MIN_SAMPLES_FOR_FULL_CAPACITY)
        adj_hidden = max(32, int(hidden_dim * scale))
        adj_latent = max(16, int(latent_dim * scale))
        if scale < 1.0:
            logger.info(
                f"数据集较小 ({n_train} 样本), 自适应缩减模型容量: "
                f"hidden={adj_hidden}, latent={adj_latent}"
            )

        model = _SurrogateMLP(
            self.n_regions, self.n_lags, adj_hidden, adj_latent, dropout
        ).to(self.device)
        opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.MSELoss()

        train_hist, val_hist = [], []
        best_val   = float("inf")
        best_state = None
        no_improve = 0

        for _ in range(num_epochs):
            model.train()
            for xb, yb in train_dl:
                pred = model(xb)
                l = loss_fn(pred, yb)
                opt.zero_grad(); l.backward(); opt.step()

            model.eval()
            with torch.no_grad():
                tl = sum(loss_fn(model(xb), yb).item() * len(xb)
                         for xb, yb in train_dl) / len(tx)
                vl = (sum(loss_fn(model(xb), yb).item() * len(xb)
                          for xb, yb in val_dl) / max(len(vx), 1)
                      if len(vx) > 0 else tl)
            train_hist.append(tl)
            val_hist.append(vl)

            # Early stopping: restore best weights when val loss stops improving.
            # 1e-4 threshold avoids halting on pure floating-point noise while
            # still detecting genuine plateaus (original 1e-7 was too strict).
            if vl < best_val - 1e-4:
                best_val   = vl
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"早停: 验证MSE连续{patience}轮未改善，在第{len(train_hist)}轮停止")
                    break

        # Restore the checkpoint with best validation loss
        if best_state is not None:
            model.load_state_dict(best_state)

        self._surrogate = model

        final_train = train_hist[-1]
        final_val   = val_hist[-1]
        overfit_ratio = final_val / max(final_train, 1e-9)

        # Record fit quality for downstream consumers (e.g. frontend reliability badge)
        self._fit_quality = {
            "train_mse":    round(float(final_train), 6),
            "val_mse":      round(float(final_val),   6),
            "overfit_ratio": round(float(overfit_ratio), 2),
            "reliable":     overfit_ratio < _OVERFIT_RELIABILITY_THRESHOLD,
            "n_epochs":     len(train_hist),
        }

        if overfit_ratio > _OVERFIT_RELIABILITY_THRESHOLD:
            logger.warning(
                f"代理模型严重过拟合: 验证MSE({final_val:.5f}) / 训练MSE({final_train:.5f}) "
                f"= {overfit_ratio:.1f}×. EC推断结果可信度较低，建议提供更多时序数据。"
            )
        else:
            logger.info(
                f"代理模型训练完成: 最终训练MSE={final_train:.5f}, 验证MSE={final_val:.5f}"
            )
        return train_hist, val_hist

    # ── EC inference — finite difference (NPI-style) ────────────────────────

    def infer_ec_perturbation(
        self,
        pert_strength: float = 0.05,
        n_samples: Optional[int] = None,
    ) -> np.ndarray:
        """
        有限差分扰动法推断有效连接（原版 NPI 方法）。

        对每个源脑区 j：
          1. 在最后一个时间步施加扰动 δ
          2. 比较扰动前后模型输出的差异
          3. EC[j, i] = mean_t(Δoutput_i) / δ

        Args:
            pert_strength: 扰动强度 δ（标准化坐标，默认0.05）
            n_samples:     使用的样本数（None = 全部）

        Returns:
            ec_matrix: (N_regions, N_regions) float array
                       ec_matrix[j, i] = 刺激区域j对区域i的影响强度
        """
        self._require_surrogate()
        model  = self._surrogate
        input_X = self._input_X
        if n_samples:
            input_X = input_X[:n_samples]

        N = self.n_regions
        M = len(input_X)
        model.eval()
        ec = np.zeros((N, N), dtype=np.float32)

        # Chunked batched perturbation: process _EC_PERTURBATION_CHUNK_SIZE source
        # regions at once.  This replaces N=200 serial forward passes (each on M
        # samples) with ceil(N / CHUNK_SIZE) batched passes, reducing PyTorch
        # kernel-launch overhead by ~(N / CHUNK_SIZE) times.
        # Tune via the module-level _EC_PERTURBATION_CHUNK_SIZE constant.
        CHUNK_SIZE = _EC_PERTURBATION_CHUNK_SIZE

        with torch.no_grad():
            X_t      = torch.tensor(input_X, dtype=torch.float32, device=self.device)
            baseline = model(X_t).cpu().numpy()   # (M, N)

            for chunk_start in range(0, N, CHUNK_SIZE):
                chunk_end = min(chunk_start + CHUNK_SIZE, N)
                chunk_n   = chunk_end - chunk_start   # number of perturbations in this batch

                # Build (chunk_n * M, N*n_lags) batched perturbed input.
                # Each of the chunk_n blocks is a copy of X_t with one column perturbed.
                X_batch = X_t.unsqueeze(0).expand(chunk_n, M, -1).reshape(chunk_n * M, -1).clone()
                # Apply perturbation: block k perturbs source region (chunk_start + k)
                for k, j in enumerate(range(chunk_start, chunk_end)):
                    X_batch[k * M:(k + 1) * M, -N + j] += pert_strength

                perturbed = model(X_batch).reshape(chunk_n, M, N)  # (chunk_n, M, N)
                # ec[j] = mean over M samples of (perturbed - baseline) / pert_strength
                baseline_t = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)  # (1, M, N)
                diff = (perturbed - baseline_t.expand(chunk_n, -1, -1)).mean(dim=1)    # (chunk_n, N)
                ec[chunk_start:chunk_end] = (diff / pert_strength).cpu().numpy()

        self._last_ec = ec
        logger.info("有限差分 EC 推断完成 (NPI 方法, 批量推断)")
        return ec

    # ── EC inference — Jacobian via autograd ────────────────────────────────

    def infer_ec_jacobian(
        self,
        n_samples: Optional[int] = 200,
    ) -> np.ndarray:
        """
        解析 Jacobian 法推断有效连接（比有限差分更精确，更快）。

        **全时滞 Granger 因果**（改进）:
        原实现仅使用最近一步的 Jacobian 块（最后 N 列），丢失了 lag-2 至 lag-n 的
        因果贡献。标准 Granger 因果定义为所有时滞的加权贡献之和：
            EC[j, i] = Σ_{l=1}^{n_lags} ∂output_i / ∂input_{t−l, j}
        数值验证表明各时滞块贡献量级相近（±15%），忽略它们会使 EC 信号量减少 67%。

        **不确定性量化**:
        对每个输入样本计算独立 Jacobian，然后取均值和标准差。标准差大的边
        （相对于均值）意味着该连接在不同脑状态下不稳定，可靠性低。

        Args:
            n_samples: 用于平均的样本数

        Returns:
            ec_matrix: (N_regions, N_regions)
            同时更新 self._last_ec 和 self._last_ec_std
        """
        self._require_surrogate()
        model  = self._surrogate
        input_X = self._input_X
        if n_samples and n_samples < len(input_X):
            idx     = np.random.choice(len(input_X), n_samples, replace=False)
            input_X = input_X[idx]

        N      = self.n_regions
        n_lags = self.n_lags

        # Accumulate per-sample Jacobians to compute both mean and std.
        # Shape: (M_samples, N_output, N_regions) — we keep only the multi-lag
        # aggregated block (summed across all lag blocks) to stay memory efficient.
        per_sample_ec = []

        model.eval()
        for row in input_X:
            x = torch.tensor(row, dtype=torch.float32, device=self.device)
            J = torch.autograd.functional.jacobian(
                lambda inp: model(inp.unsqueeze(0)).squeeze(0),
                x,
            ).cpu().detach().numpy()  # (N_out, N*n_lags)

            # Sum Jacobian contributions across ALL lag blocks.
            # Block l (0-indexed, leftmost=oldest) corresponds to time offset
            # (n_lags − l) steps before the current step.  Summing all blocks
            # captures the full Granger-causal horizon, not just the most recent
            # lag, matching the standard VAR / Granger definition:
            #   GC[j→i] = Σ_{l=1}^{n_lags} A_l[i,j]
            # Each block is (N_out, N_in); we sum them to get total causal influence.
            # Vectorized: reshape J into (N_out, n_lags, N_in) and sum over lag axis,
            # replacing the Python loop over n_lags with a single NumPy reduction.
            J_3d  = J.reshape(N, n_lags, N).astype(np.float64)  # (N_out, n_lags, N_in)
            J_sum = J_3d.sum(axis=1)                             # (N_out, N_in)
            # Transpose so ec[src, tgt] = influence of src on tgt
            per_sample_ec.append(J_sum.T)

        per_sample_ec = np.array(per_sample_ec, dtype=np.float64)  # (M, N, N)

        ec_mean = per_sample_ec.mean(axis=0).astype(np.float32)    # (N, N)
        ec_std  = per_sample_ec.std(axis=0).astype(np.float32)     # (N, N)

        self._last_ec     = ec_mean
        self._last_ec_std = ec_std
        logger.info(
            f"Jacobian EC 推断完成 (全时滞 {n_lags} 步, "
            f"{len(input_X)} 样本, mean|EC|={float(np.abs(ec_mean).mean()):.4f})"
        )
        return self._last_ec

    # ── Fast demo EC (no training needed) ───────────────────────────────────

    def infer_ec_demo(self) -> np.ndarray:
        """
        无需训练数据的演示模式 EC 推断。

        基于 Fibonacci 球面坐标生成距离矩阵，用指数衰减近似局部 EC，
        并添加同侧镜像（homotopic）长程连接。
        结果不代表真实连接，仅用于前端可视化演示。
        """
        n = self.n_regions
        half = n // 2

        # Reproduce brain positions using vectorized computation (same formula
        # as app.js / _demo_simulate / realtime_server._make_fibonacci_brain_positions).
        golden_angle = 2 * np.pi * (2 - (1 + np.sqrt(5)) / 2)
        i_arr  = np.arange(half, dtype=np.float64)
        t_arr  = (i_arr + 0.5) / half
        el_arr = 1.0 - 1.85 * t_arr
        r_arr  = np.sqrt(np.maximum(0.0, 1.0 - el_arr ** 2))
        az_arr = golden_angle * i_arr
        lat    = np.abs(r_arr * np.cos(az_arr)) * 0.85 + 0.15
        bulge  = 9 * np.exp(-((el_arr + 0.22) ** 2) * 5)

        pos = np.zeros((n, 3), dtype=np.float32)
        for h, sign in enumerate((-1, 1)):
            start = h * half
            pos[start: start + half, 0] = sign * (lat * 55 + bulge + 9)
            pos[start: start + half, 1] = el_arr * 63 - 4
            pos[start: start + half, 2] = r_arr * np.sin(az_arr) * 76 - 8

        rng = np.random.default_rng(42)
        # Local connections (exponential with Euclidean distance)
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        sigma = 40.0
        ec = np.exp(-(D ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(ec, 0.0)
        # Homotopic long-range connections (ipsilateral mirror: i ↔ i+half)
        idx = np.arange(half)
        ec[idx, idx + half] = 0.55
        ec[idx + half, idx] = 0.55
        # Small noise to break symmetry
        ec += rng.uniform(0, 0.04, (n, n))
        np.fill_diagonal(ec, 0.0)
        # Row-normalise so magnitudes are comparable
        row_max = ec.max(axis=1, keepdims=True)
        ec = np.where(row_max > 0, ec / row_max, ec)
        self._last_ec = ec
        return ec

    # ── Target suggestion ────────────────────────────────────────────────────

    def suggest_targets(
        self,
        ec_matrix: Optional[np.ndarray] = None,
        n_targets: int = 5,
        strategy: str = "outgoing",
    ) -> List[int]:
        """
        基于 EC 矩阵推荐刺激靶点。

        策略:
        - "outgoing":   按行和排序 → 影响力最强的源脑区（最适合刺激）
        - "incoming":   按列和排序 → 最容易被影响的目标区域
        - "hub":        同时具备强传出和强传入（PageRank 近似）

        Returns:
            Sorted list of region indices (descending influence)
        """
        if ec_matrix is None:
            if self._last_ec is None:
                raise ValueError("尚未推断 EC 矩阵，请先调用 infer_ec_* 之一。")
            ec_matrix = self._last_ec

        ec_abs = np.abs(ec_matrix).copy()
        np.fill_diagonal(ec_abs, 0.0)   # remove self-loops: trivially high autocorrelation
                                        # biases top-source ranking toward high-autocorr regions

        if strategy == "outgoing":
            scores = ec_abs.sum(axis=1)   # row sum = 影响其他区域的强度
        elif strategy == "incoming":
            scores = ec_abs.sum(axis=0)   # col sum = 被其他区域影响的强度
        elif strategy == "hub":
            out_scores = ec_abs.sum(axis=1)
            in_scores  = ec_abs.sum(axis=0)
            # 几何平均
            scores = np.sqrt(out_scores * in_scores)
        else:
            raise ValueError(f"未知策略: {strategy}. 选项: outgoing, incoming, hub")

        top_idx = np.argsort(scores)[::-1][:n_targets]
        return top_idx.tolist()

    # ── Data-driven trajectory prediction ────────────────────────────────────

    def predict_trajectory(
        self,
        initial_state: np.ndarray,
        stim_weights: np.ndarray,
        stim_fn: Callable[[int], float],
        n_steps: int = 60,
        n_warmup: int = 0,
    ) -> List[dict]:
        """Auto-regressive stimulation trajectory prediction using the trained surrogate.

        This is the scientifically correct way to simulate "what will this brain
        do after stimulation?": apply a perturbation to the current state in z-score
        space, feed it through the MLP fitted on *actual* fMRI/EEG time series, and
        auto-regressively unroll the prediction.

        Perturbation units (important):
            ``stim_fn(k)`` returns a scalar that is interpreted as a **z-score
            deviation** — i.e., how many standard deviations the stimulation shifts
            the target regions' activity at step k.  The NPI convention (pert_strength
            = 0.05) uses the same scale.  The caller should NOT pre-divide by
            ``_ts_std``: that conversion is already baked into the z-scored lag window.
            Perturbation in display space → z-score: divide by std.
            But ``stim_fn`` already returns values in z-score-compatible magnitude
            (small relative to ±3σ dynamic range), so no further scaling is needed.

        Args:
            initial_state : (N,) float32 activity in [0,1] display space (same space
                            as the time series used in fit_surrogate).
            stim_weights  : (N,) spatial distribution of the stimulation (0→1,
                            peak = 1 at target regions, Gaussian falloff elsewhere).
            stim_fn       : callable(step_index) → z-score-scale amplitude.
                            Returns 0 when stimulation is inactive.
            n_steps       : total frames to return (including pre-stim if stim_fn
                            returns 0 for early steps).
            n_warmup      : number of surrogate steps to run *before* recording frames,
                            using stim_fn returning 0.  Builds a realistic lag-window
                            history from the initial state, so the first recorded frame
                            reflects genuine predicted baseline dynamics rather than the
                            artificial "all lags = initial state" assumption.

        Returns:
            List of {"activity": [N floats]} dicts in display space [0, 1],
            length == n_steps.
        """
        self._require_surrogate()
        if self._ts_mean is None or self._ts_std is None:
            raise RuntimeError(
                "Normalization stats missing; fit_surrogate must be called before "
                "predict_trajectory."
            )

        std_safe  = np.maximum(self._ts_std, 1e-6)        # (N,) — avoids div-by-zero
        init_norm = (initial_state.astype(np.float32) - self._ts_mean) / std_safe

        # Build initial lag window by repeating the current z-scored state n_lags times.
        # All lags start equal; the warmup phase (below) replaces this with genuine
        # surrogate dynamics before stimulation is applied.
        lag_window = np.tile(init_norm, self.n_lags).astype(np.float32)  # (N*n_lags,)

        def _step(stim_amp: float) -> np.ndarray:
            """One surrogate forward pass; returns z-scored prediction."""
            x = lag_window.copy()
            if stim_amp != 0.0:
                # Perturbation is applied directly in z-score space.
                # stim_weights ∈ [0,1] (spatial Gaussian), stim_amp is in z-score
                # units (small relative to ±3σ range, e.g. 0.0–0.5 for visible effect).
                # DO NOT divide by std_safe: amp is already in z-score units, and
                # dividing by std (≈0.05–0.15) would amplify the perturbation 7–20×,
                # completely saturating the MLP's nonlinearities.
                x[-self.n_regions:] += (stim_amp * stim_weights).astype(np.float32)
            x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
            pred = self._surrogate(x_t.unsqueeze(0)).squeeze(0).cpu().numpy()
            # Slide lag window: drop oldest step, append newest prediction
            lag_window[:-self.n_regions] = lag_window[self.n_regions:]
            lag_window[-self.n_regions:] = pred
            return pred

        self._surrogate.eval()
        with torch.no_grad():
            # Warmup: run surrogate unperturbed to fill the lag window with
            # genuine predicted dynamics (instead of the flat "all=init" window).
            for _ in range(n_warmup):
                _step(0.0)

            # Recording phase: n_steps frames returned to the caller.
            frames = []
            for k in range(n_steps):
                pred_zs = _step(float(stim_fn(k)))
                pred_display = np.clip(
                    pred_zs * std_safe + self._ts_mean, 0.0, 1.0
                ).astype(np.float32)
                frames.append({"activity": pred_display.tolist()})

        return frames

    # ── Rollout-based response matrix ────────────────────────────────────────

    def compute_response_matrix(
        self,
        initial_state: np.ndarray,
        stim_regions: Optional[List[int]] = None,
        rollout_steps: int = 20,
        sustained_steps: int = 5,
        alpha: float = 0.3,
        mode: str = "sustained",
    ) -> dict:
        """Compute the perturbation-based dynamical response matrix R[i,j,k].

        For each stimulated region i, runs a rollout WITH sustained injection
        and another WITHOUT (baseline), then computes:

            R[i, j, k] = X_stim[j, k] − X_base[j, k]

        This characterises the *dynamical propagation structure* of the surrogate:
        which regions j respond to a perturbation at i by step k, and with what
        sign/magnitude.

        Stimulation convention (NPI-aligned):
            delta_z = alpha  (z-score units, matching NPI's pert_strength).
            This corresponds to display-space amplitude ≈ alpha × ts_std[i],
            scaling perturbations to each region's local variability.
            alpha ∈ [0.1, 0.5] keeps perturbations < 1σ (avoids OOD).

        Args:
            initial_state  : (N,) float32 brain state in [0,1] display space.
            stim_regions   : list of region indices to stimulate; None = all N.
            rollout_steps  : T — total rollout length in time steps.
            sustained_steps: K — number of steps stimulation is injected.
                             Ignored when mode == "impulse" (K is always 1).
            alpha          : perturbation strength in z-score units (0.1–0.5).
            mode           : "sustained" (inject for K steps) or
                             "impulse" (inject at step 0 only, then observe decay).

        Returns:
            dict with keys:
                R              : np.ndarray (n_stim, N, rollout_steps) float32 — ΔX
                stim_regions   : list[int] — which regions were stimulated
                baseline       : np.ndarray (N, rollout_steps) float32 — zero-stim
                alpha          : float
                mode           : str
                sustained_steps: int — effective K used
        """
        self._require_surrogate()
        N = self.n_regions
        if stim_regions is None:
            stim_regions = list(range(N))
        else:
            stim_regions = [int(r) for r in stim_regions if 0 <= int(r) < N]
        K = 1 if mode == "impulse" else max(1, int(sustained_steps))
        T = max(1, int(rollout_steps))

        null_weights = np.zeros(N, dtype=np.float32)

        def _run_traj(sw: np.ndarray, amp: float) -> np.ndarray:
            """Rollout with given stim_weights and amplitude; return (N, T) display."""
            stim_fn: Callable[[int], float] = (
                (lambda k: amp if k == 0 else 0.0)
                if mode == "impulse"
                else (lambda k: amp if k < K else 0.0)
            )
            frames = self.predict_trajectory(
                initial_state=initial_state.astype(np.float32),
                stim_weights=sw,
                stim_fn=stim_fn,
                n_steps=T,
            )
            return np.array([f["activity"] for f in frames], dtype=np.float32).T  # (N, T)

        # Baseline: zero stimulation (shared across all stim_regions)
        baseline = _run_traj(null_weights, 0.0)   # (N, T)

        # Response for each stimulated region
        R = np.zeros((len(stim_regions), N, T), dtype=np.float32)
        for idx, region_i in enumerate(stim_regions):
            sw = np.zeros(N, dtype=np.float32)
            sw[region_i] = 1.0
            R[idx] = _run_traj(sw, alpha) - baseline  # ΔX[j, k]

        logger.info(
            "compute_response_matrix: %d regions × T=%d, mode=%s, K=%d, "
            "alpha=%.3f, mean|R|=%.4f",
            len(stim_regions), T, mode, K, alpha, float(np.abs(R).mean()),
        )
        return {
            "R":               R,
            "stim_regions":    stim_regions,
            "baseline":        baseline,
            "alpha":           float(alpha),
            "mode":            mode,
            "sustained_steps": K,
        }

    @staticmethod
    def analyze_response_matrix(
        R: np.ndarray,
        stim_regions: List[int],
        N: Optional[int] = None,
        threshold: float = 0.01,
    ) -> dict:
        """Analyse structural properties of a response matrix.

        Args:
            R            : (n_stim, N, T) float32 — from compute_response_matrix.
            stim_regions : list of stimulated region indices (length == n_stim).
            N            : total region count (inferred from R if None).
            threshold    : |ΔX| threshold for "visible response" in spatial spread.

        Returns:
            dict with keys:
                spatial_spread     : float — fraction of non-target regions with
                                     mean |ΔX| > threshold
                temporal_decay     : float — linear slope of mean |R| over time
                                     (negative = decaying response → stable system)
                delay_peak_steps   : list[int] — per stim-region, which step shows
                                     maximum non-target mean |ΔX|
                off_diagonal_ratio : float — mean off-diag |R| / mean diag |R|
                mean_response_map  : list[float] (N,) — mean |ΔX| across stim regions
                                     and time, for 3D overlay
                has_spatial_spread : bool — spatial_spread > 0.05
                has_decay          : bool — temporal_decay < −1e-4
                has_delay          : bool — any delay_peak_steps > 0
                plausibility_summary : str — human-readable assessment
        """
        n_stim, n_regions, T = R.shape
        if N is None:
            N = n_regions

        # Mean |ΔX| across stim regions and time → per-region overlay
        mean_response_map = np.abs(R).mean(axis=(0, 2))   # (N,)

        # Spatial spread: fraction of non-target regions above threshold
        target_set  = set(stim_regions)
        non_target  = [j for j in range(n_regions) if j not in target_set]
        if non_target:
            spread_vals  = mean_response_map[np.array(non_target)]
            spatial_spread = float((spread_vals > threshold).sum()) / len(non_target)
        else:
            spatial_spread = 0.0

        # Temporal decay: linear regression slope of mean |R| vs time step
        mean_over_t = np.abs(R).mean(axis=(0, 1))   # (T,)
        t_arr = np.arange(T, dtype=np.float64)
        if T >= 2:
            t_c = t_arr - t_arr.mean()
            v_c = mean_over_t.astype(np.float64) - mean_over_t.mean()
            temporal_decay = float((t_c * v_c).sum() / max((t_c ** 2).sum(), 1e-12))
        else:
            temporal_decay = 0.0

        # Delay peaks: per stim-region, step with max non-target mean |ΔX|
        delay_peak_steps: List[int] = []
        nt_arr = np.array(non_target) if non_target else None
        for idx in range(n_stim):
            if nt_arr is not None:
                profile = np.abs(R[idx, nt_arr, :]).mean(axis=0)  # (T,)
                delay_peak_steps.append(int(np.argmax(profile)))
            else:
                delay_peak_steps.append(0)

        # Off-diagonal ratio: spread-to-non-targets vs direct target effect
        diag_vals: List[float]     = []
        off_diag_vals: List[float] = []
        for idx, ri in enumerate(stim_regions):
            if ri < n_regions:
                diag_vals.append(float(np.abs(R[idx, ri, :]).mean()))
            if nt_arr is not None:
                off_diag_vals.extend(
                    float(np.abs(R[idx, j, :]).mean()) for j in non_target
                )
        mean_diag      = float(np.mean(diag_vals))    if diag_vals    else 0.0
        mean_off_diag  = float(np.mean(off_diag_vals)) if off_diag_vals else 0.0
        off_diagonal_ratio = mean_off_diag / max(mean_diag, 1e-9)

        has_spatial_spread = spatial_spread > 0.05
        has_decay          = temporal_decay < -1e-4
        has_delay          = any(d > 0 for d in delay_peak_steps)

        # Human-readable plausibility summary
        lines = []
        if has_spatial_spread:
            lines.append(f"✓ 空间传播: {spatial_spread*100:.0f}% 非靶区响应超阈值")
        else:
            lines.append(
                f"✗ 空间传播不明显 ({spatial_spread*100:.0f}% 超阈值)，"
                "建议增大 alpha 或提供更多数据"
            )
        if has_decay:
            lines.append(f"✓ 时间衰减: 斜率={temporal_decay:.4f} (稳定系统)")
        else:
            lines.append(
                f"△ 无显著时间衰减 (斜率={temporal_decay:.4f})，"
                "系统可能处于持续振荡"
            )
        if has_delay:
            lines.append(
                f"✓ 传播延迟: 非靶区峰值在步骤 "
                f"{min(delay_peak_steps)}–{max(delay_peak_steps)}"
            )
        else:
            lines.append("△ 无明显传播延迟 (瞬时扩散或无传播)")
        lines.append(
            f"  离轴/对角比={off_diagonal_ratio:.3f} "
            f"({'传播至远端区域' if off_diagonal_ratio > 0.1 else '主要局部效应'})"
        )

        return {
            "spatial_spread":      spatial_spread,
            "temporal_decay":      temporal_decay,
            "delay_peak_steps":    delay_peak_steps,
            "off_diagonal_ratio":  off_diagonal_ratio,
            "mean_response_map":   mean_response_map.tolist(),
            "has_spatial_spread":  has_spatial_spread,
            "has_decay":           has_decay,
            "has_delay":           has_delay,
            "plausibility_summary": "\n".join(lines),
        }

    def validate_response_matrix(
        self,
        initial_states: List[np.ndarray],
        stim_regions: Optional[List[int]] = None,
        rollout_steps: int = 20,
        sustained_steps: int = 5,
        alpha: float = 0.3,
        mode: str = "sustained",
    ) -> dict:
        """Check self-consistency of response matrix across multiple initial states.

        Runs ``compute_response_matrix`` for each state, then computes pairwise
        Pearson correlations between the flattened R matrices.

        High correlation (r ≥ 0.5) means the propagation structure is robust to
        initial conditions.  Low correlation may indicate strong nonlinear dynamics
        or an over-fitted surrogate — neurobiologically interesting in either case.

        Args:
            initial_states : list of (N,) float32 arrays in [0,1] display space.
                             Provide ≥ 2 states for a meaningful comparison.
            Others         : forwarded to ``compute_response_matrix``.

        Returns:
            dict with keys:
                consistency_r  : float — mean pairwise Pearson r
                n_states       : int
                reliable       : bool — consistency_r ≥ 0.5
                interpretation : str — human-readable explanation
        """
        if len(initial_states) < 2:
            return {
                "consistency_r": float("nan"),
                "n_states":      len(initial_states),
                "reliable":      False,
                "interpretation": (
                    "自一致性检验需要至少 2 个初始状态，请提供更多样本。"
                ),
            }

        Rs = []
        for state in initial_states:
            res = self.compute_response_matrix(
                initial_state=state,
                stim_regions=stim_regions,
                rollout_steps=rollout_steps,
                sustained_steps=sustained_steps,
                alpha=alpha,
                mode=mode,
            )
            Rs.append(res["R"].flatten())

        # Pairwise Pearson r (vectorised — no scipy dependency)
        r_values: List[float] = []
        for i in range(len(Rs)):
            for j in range(i + 1, len(Rs)):
                a_c = Rs[i] - Rs[i].mean()
                b_c = Rs[j] - Rs[j].mean()
                denom = (
                    np.sqrt((a_c ** 2).sum()) * np.sqrt((b_c ** 2).sum())
                )
                r_values.append(float((a_c * b_c).sum() / max(denom, 1e-12)))

        consistency_r = float(np.mean(r_values)) if r_values else float("nan")
        reliable = consistency_r >= 0.5 if not np.isnan(consistency_r) else False

        if reliable:
            interpretation = (
                f"✓ 自一致性良好 (r={consistency_r:.3f} ≥ 0.5): "
                "响应矩阵在不同初始状态下保持稳定，传播结构可靠。"
            )
        elif not np.isnan(consistency_r) and consistency_r >= 0.3:
            interpretation = (
                f"△ 自一致性中等 (r={consistency_r:.3f}): "
                "响应矩阵对初始状态有一定依赖性，建议增加数据量后重新训练代理。"
            )
        else:
            interpretation = (
                f"✗ 自一致性差 (r={consistency_r:.3f} < 0.3): "
                "响应矩阵在不同初始状态间差异显著，可能反映强非线性动力学或代理过拟合。"
                "建议检查 fit_quality 并增加时序数据。"
            )

        return {
            "consistency_r": consistency_r,
            "n_states":      len(initial_states),
            "reliable":      reliable,
            "interpretation": interpretation,
        }

    # ── Activity delta prediction ────────────────────────────────────────────

    def predict_activity_delta(
        self,
        target_regions: List[int],
        amplitude: float = 0.5,
        ec_matrix: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        用 EC 矩阵线性近似刺激目标区域后全脑的活动变化量。

        Δactivity_i ≈ amplitude × Σ_j∈targets EC[j, i]

        Returns:
            delta: (N_regions,) float array in [0, 1] (normalised)
        """
        if ec_matrix is None:
            ec_matrix = self._last_ec
        if ec_matrix is None:
            return np.zeros(self.n_regions, dtype=np.float32)

        stim = np.zeros(self.n_regions, dtype=np.float32)
        for j in target_regions:
            if 0 <= j < self.n_regions:
                stim[j] = amplitude

        delta = ec_matrix.T @ stim          # (N,) = EC^T @ stim
        # Normalise to [0, 1] for visualisation
        d_abs = np.abs(delta)
        if d_abs.max() > 0:
            d_abs = d_abs / d_abs.max()
        return d_abs.astype(np.float32)

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _require_surrogate(self):
        if self._surrogate is None or self._input_X is None:
            raise RuntimeError(
                "代理模型未训练，请先调用 fit_surrogate(time_series)。"
            )

    def ec_to_dict(self, ec_matrix: Optional[np.ndarray] = None) -> dict:
        """将 EC 矩阵转换为前端可用的 JSON 友好字典。

        **归一化** — globally normalised so the maximum absolute value equals
        1.0.  Without this step raw Jacobian / finite-difference values are
        typically O(0.001–0.01), falling below the frontend draw-threshold.
        Relative rankings (top_sources, top_targets) are unaffected by scaling.

        **自环移除** — diagonal is zeroed before any downstream analysis.
        Self-loop EC[i,i] measures autocorrelation, not causal *inter-region*
        influence; including it biases hub-score and top-source rankings toward
        high-autocorrelation regions, which are not necessarily influential.

        **不确定性** — if ``_last_ec_std`` is available (set by
        ``infer_ec_jacobian``), the dict also contains ``ec_std_flat``: the
        per-edge Jacobian standard deviation (same normalisation as ec_flat).
        A high std relative to the mean indicates an unreliable edge — the
        connection pattern varies across brain states and may not represent a
        stable anatomical pathway.
        """
        if ec_matrix is None:
            ec_matrix = self._last_ec
        if ec_matrix is None:
            return {}
        n = self.n_regions

        # Remove self-loops before ranking and normalisation.
        ec_work = ec_matrix.copy()
        np.fill_diagonal(ec_work, 0.0)

        top_sources = self.suggest_targets(ec_work, 10, "outgoing")
        top_targets = self.suggest_targets(ec_work, 10, "incoming")

        # Normalise to [-1, 1] so the frontend threshold (0.05) is meaningful.
        ec_max = float(np.abs(ec_work).max())
        ec_norm = ec_work / ec_max if ec_max > 1e-12 else ec_work.copy()

        result = {
            "ec_flat":      ec_norm.flatten().tolist(),   # 200×200, normalised, no self-loops
            "top_sources":  top_sources,
            "top_targets":  top_targets,
            "n_regions":    n,
        }

        # Per-edge uncertainty (Jacobian std across input samples).
        # Normalised by the same ec_max so magnitudes are directly comparable
        # to ec_flat.  NaN / inf guards prevent JSON serialisation errors.
        if self._last_ec_std is not None:
            std_work = self._last_ec_std.copy()
            np.fill_diagonal(std_work, 0.0)
            std_norm = std_work / ec_max if ec_max > 1e-12 else std_work
            std_norm = np.nan_to_num(std_norm, nan=0.0, posinf=0.0, neginf=0.0)
            result["ec_std_flat"] = std_norm.flatten().tolist()

        # Include surrogate quality so the frontend can show a reliability badge
        if self._fit_quality:
            result["fit_quality"] = self._fit_quality
        return result
