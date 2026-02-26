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
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
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

    Args:
        time_series: (T, N_regions) float array
        n_lags:      历史步数

    Returns:
        input_X:  (T-n_lags, N_regions*n_lags)
        target_Y: (T-n_lags, N_regions)
    """
    T, N = time_series.shape
    X = np.zeros((T - n_lags, N * n_lags), dtype=np.float32)
    Y = np.zeros((T - n_lags, N), dtype=np.float32)
    for i in range(T - n_lags):
        X[i] = time_series[i: i + n_lags].flatten()
        Y[i] = time_series[i + n_lags]
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
        model.eval()
        ec = np.zeros((N, N), dtype=np.float32)

        with torch.no_grad():
            X_t = torch.tensor(input_X, dtype=torch.float32, device=self.device)
            baseline = model(X_t).cpu().numpy()  # (M, N)

            for j in range(N):
                # 仅扰动最近一步中区域 j 的值（对应 input 的最后 N 列中的列 j）
                X_pert = X_t.clone()
                X_pert[:, -N + j] += pert_strength
                perturbed = model(X_pert).cpu().numpy()   # (M, N)
                ec[j] = (perturbed - baseline).mean(axis=0) / pert_strength

        self._last_ec = ec
        logger.info("有限差分 EC 推断完成 (NPI 方法)")
        return ec

    # ── EC inference — Jacobian via autograd ────────────────────────────────

    def infer_ec_jacobian(
        self,
        n_samples: Optional[int] = 200,
    ) -> np.ndarray:
        """
        解析 Jacobian 法推断有效连接（比有限差分更精确，更快）。

        对每个输入样本 x 计算 ∂output / ∂input，取最近一步的块。
        EC[j, i] = mean_x(∂output_i / ∂input_{t-1, j})

        Args:
            n_samples: 用于平均的样本数

        Returns:
            ec_matrix: (N_regions, N_regions)
        """
        self._require_surrogate()
        model  = self._surrogate
        input_X = self._input_X
        if n_samples and n_samples < len(input_X):
            # Use a fixed-seed local RNG so the same surrogate always produces the
            # same EC matrix (deterministic subset selection).  Calling np.random
            # without a seed made top_sources and fc_vs_ec vary across repeated
            # calls on the same dataset (AGENTS.md 2026-02-26 bug report).
            rng_ec = np.random.default_rng(42)
            idx    = rng_ec.choice(len(input_X), n_samples, replace=False)
            input_X = input_X[idx]

        N = self.n_regions
        jacobian = np.zeros((N, N), dtype=np.float64)

        # Use eval mode: _SurrogateMLP has no dropout/batchnorm so eval() is
        # equivalent to train() for forward pass, but it is best practice.
        model.eval()
        for row in input_X:
            x = torch.tensor(row, dtype=torch.float32, device=self.device)
            # torch.autograd.functional.jacobian creates its own grad-enabled copy
            # of the input; the caller does not need requires_grad=True.
            J = torch.autograd.functional.jacobian(
                lambda inp: model(inp.unsqueeze(0)).squeeze(0),
                x,
            ).cpu().detach().numpy()  # (N, N*n_lags)
            # Keep only the last-step block (columns -N:)
            jacobian += J[:, -N:]

        model.eval()
        ec = (jacobian / len(input_X)).T  # (N_source, N_target)
        self._last_ec = ec.astype(np.float32)
        logger.info("Jacobian EC 推断完成")
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
        # Reproduce brain positions (same formula as app.js / _demo_simulate).
        # golden_angle: angular increment from the golden ratio for Fibonacci sphere.
        golden_angle = 2 * np.pi * (2 - (1 + np.sqrt(5)) / 2)
        pos = np.zeros((n, 3), dtype=np.float32)
        for h in range(2):
            sign = -1 if h == 0 else 1
            for i in range(100):
                t_  = (i + 0.5) / 100.0
                el  = 1.0 - 1.85 * t_
                r   = np.sqrt(max(0.0, 1 - el * el))
                az  = golden_angle * i
                lat = abs(r * np.cos(az)) * 0.85 + 0.15
                bulge = 9 * np.exp(-((el + 0.22) ** 2) * 5)
                ri  = h * 100 + i
                pos[ri] = [sign * (lat * 55 + bulge + 9), el * 63 - 4,
                           r * np.sin(az) * 76 - 8]

        rng = np.random.default_rng(42)
        ec  = np.zeros((n, n), dtype=np.float32)
        # Local connections (exponential with Euclidean distance)
        D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        sigma = 40.0
        ec = np.exp(-(D ** 2) / (2 * sigma ** 2))
        np.fill_diagonal(ec, 0.0)
        # Homotopic long-range connections (ipsilateral mirror: i ↔ i+100)
        for i in range(100):
            ec[i, i + 100] = ec[i + 100, i] = 0.55
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

        ec_abs = np.abs(ec_matrix)

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

        The returned ``ec_flat`` is globally normalised so that the maximum
        absolute value equals 1.0.  Without this step the raw Jacobian /
        finite-difference values are typically O(0.001–0.01), which fall below
        the 0.05 draw-threshold in the frontend and result in zero visible
        connection lines.  Relative rankings (top_sources, top_targets) are
        unaffected by scaling.
        """
        if ec_matrix is None:
            ec_matrix = self._last_ec
        if ec_matrix is None:
            return {}
        n = self.n_regions
        top_sources = self.suggest_targets(ec_matrix, 10, "outgoing")
        top_targets = self.suggest_targets(ec_matrix, 10, "incoming")
        # Normalise to [-1, 1] so the frontend threshold (0.05) is meaningful.
        # Use a small epsilon guard and explicitly copy to avoid returning a
        # reference to the original matrix.
        ec_max = float(np.abs(ec_matrix).max())
        ec_norm = ec_matrix / ec_max if ec_max > 1e-12 else ec_matrix.copy()
        result = {
            "ec_flat":      ec_norm.flatten().tolist(),   # 200×200, normalised
            "top_sources":  top_sources,
            "top_targets":  top_targets,
            "n_regions":    n,
        }
        # Include surrogate quality so the frontend can show a reliability badge
        if self._fit_quality:
            result["fit_quality"] = self._fit_quality
        return result
