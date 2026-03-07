"""
Surrogate Test
==============

验证真实系统的非线性动力学是否显著超过线性基准（surrogate data test）。

三种代替序列（Schreiber & Schmitz 2000）：

1. **phase_randomize**（AAFT/FFT 相位随机化）
   - 保留功率谱密度 + 振幅分布，破坏高阶非线性相关
   - 原假设 H₀: 数据来自线性高斯过程（经振幅变换后）
   - 参考: Theiler et al. (1992) Physica D 58:77–94

2. **shuffle**（直接打乱时序）
   - 保留振幅分布，破坏所有时序结构（线性 + 非线性）
   - 原假设 H₀: 数据为独立同分布（IID）
   - 参考: Theiler & Prichard (1996)

3. **ar_surrogate**（AR(p) 拟合代替）
   - 拟合最优阶数 AR 模型，从模型残差生成代替序列
   - 原假设 H₀: 数据来自 AR(p) 线性过程
   - 参考: Small (2005) Applied Nonlinear Time Series Analysis

验证准则：
  - 若 LLE_real >> LLE_surrogate（95% 置信区间上界），拒绝 H₀，
    说明数据具有真实非线性（不可被线性模型解释）。
  - LLE_real 与 surrogate 无显著差异表示混沌可能是线性噪声假象。

输出文件：
  outputs/surrogate_test.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Surrogate generation
# ─────────────────────────────────────────────────────────────────────────────

def phase_randomize_surrogate(
    ts: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    FFT 相位随机化代替序列（Theiler et al. 1992）。

    对每个通道独立随机化傅里叶相位，保留功率谱密度（PSD）和振幅分布。
    等效于：振幅调整傅里叶变换代替序列（AAFT 的简化版）。

    算法：
      1. FFT → 复数频谱
      2. 为每个正频率分量添加均匀随机相位 φ ∈ [0, 2π)
      3. 保持共轭对称（确保逆变换为实数）
      4. IFFT

    Args:
        ts:  shape (T,) 或 (T, N)，原始时序数组。
        rng: 随机数生成器（None → np.random.default_rng()）。

    Returns:
        surrogate: shape 与 ts 相同，相位随机化代替序列。
    """
    if rng is None:
        rng = np.random.default_rng()

    ts = np.asarray(ts, dtype=np.float64)
    scalar_input = ts.ndim == 1
    if scalar_input:
        ts = ts[:, None]

    T, N = ts.shape
    surrogate = np.empty_like(ts)

    for i in range(N):
        x = ts[:, i]
        fx = np.fft.rfft(x)

        # Number of unique positive-frequency bins (excluding DC and Nyquist)
        n_freq = len(fx)
        # Allocate phases for all bins; DC (index 0) is forced to 0 to preserve
        # the mean, and the Nyquist bin (last bin for even-length signals) is
        # also forced to 0 so that irfft produces a real-valued output.
        phases = rng.uniform(0.0, 2.0 * np.pi, size=n_freq)
        phases[0] = 0.0  # preserve mean (DC component must be real)
        if T % 2 == 0:
            phases[-1] = 0.0  # Nyquist component must be real

        randomized = fx * np.exp(1j * phases)
        surrogate[:, i] = np.fft.irfft(randomized, n=T)

    return surrogate[:, 0] if scalar_input else surrogate


def shuffle_surrogate(
    ts: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    时序打乱代替序列。

    对每个通道独立随机置换时间步，保留振幅分布，
    破坏所有时序相关结构（线性 + 非线性）。

    Args:
        ts:  shape (T,) 或 (T, N)，原始时序数组。
        rng: 随机数生成器（None → np.random.default_rng()）。

    Returns:
        surrogate: shape 与 ts 相同，打乱的代替序列。
    """
    if rng is None:
        rng = np.random.default_rng()

    ts = np.asarray(ts, dtype=np.float64)
    scalar_input = ts.ndim == 1
    if scalar_input:
        ts = ts[:, None]

    T, N = ts.shape
    surrogate = ts.copy()
    for i in range(N):
        rng.shuffle(surrogate[:, i])

    return surrogate[:, 0] if scalar_input else surrogate


def ar_surrogate(
    ts: np.ndarray,
    order: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    AR(p) 代替序列（Small 2005）。

    对每个通道独立拟合 AR(p) 模型，然后用模型 + 白噪声（残差重采样）
    生成代替序列。

    算法：
      1. 利用最小二乘法拟合 AR(p): x(t) = Σ a_k x(t-k) + ε(t)
      2. 对拟合残差 ε 做自举重采样（block bootstrap 保持残差分布）
      3. 用 AR 系数 + 重采样残差前向迭代生成代替序列

    Args:
        ts:    shape (T,) 或 (T, N)，原始时序数组。
        order: AR 模型阶数（默认 1；对于 fMRI 使用 1–3）。
        rng:   随机数生成器（None → np.random.default_rng()）。

    Returns:
        surrogate: shape 与 ts 相同，AR 代替序列。
    """
    if rng is None:
        rng = np.random.default_rng()

    ts = np.asarray(ts, dtype=np.float64)
    scalar_input = ts.ndim == 1
    if scalar_input:
        ts = ts[:, None]

    T, N = ts.shape
    surrogate = np.empty_like(ts)
    p = max(1, order)

    for i in range(N):
        x = ts[:, i]
        surrogate[:, i] = _ar_surrogate_1d(x, p, rng)

    return surrogate[:, 0] if scalar_input else surrogate


def _ar_surrogate_1d(
    x: np.ndarray,
    p: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """AR(p) 代替序列生成（单通道）。"""
    T = len(x)
    # Build regression matrix X_lag ∈ ℝ^{(T-p) × p} and target y
    X_lag = np.column_stack([x[p - 1 - k: T - 1 - k] for k in range(p)])
    y = x[p:]

    # Least-squares fit: a = (XᵀX)⁻¹Xᵀy
    coeffs, _, _, _ = np.linalg.lstsq(X_lag, y, rcond=None)

    # Compute residuals
    residuals = y - X_lag @ coeffs

    # Bootstrap residuals: sample with replacement
    n_res = len(residuals)
    sampled_res = residuals[rng.integers(0, n_res, size=T - p)]

    # Forward-simulate AR process
    out = np.zeros(T)
    out[:p] = x[:p]  # seed with original initial values
    for t in range(p, T):
        out[t] = float(coeffs @ out[t - p: t][::-1]) + sampled_res[t - p]

    return out


# ─────────────────────────────────────────────────────────────────────────────
# LLE computation for surrogate (uses Rosenstein method on trajectory)
# ─────────────────────────────────────────────────────────────────────────────

def _lle_from_surrogate_trajectory(
    surrogate_ts: np.ndarray,
    max_lag: int = 30,
    min_temporal_sep: int = 10,
) -> float:
    """
    在代替序列上运行 Rosenstein LLE 估计。

    Args:
        surrogate_ts:    shape (T, N)，代替序列轨迹。
        max_lag:         Rosenstein 追踪分离的最大步数。
        min_temporal_sep: 最小时间间距（避免平凡近邻）。

    Returns:
        lle: 最大 Lyapunov 指数（每步单位）。
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
    except ImportError:
        from lyapunov import rosenstein_lyapunov

    lle, _ = rosenstein_lyapunov(
        surrogate_ts,
        max_lag=max_lag,
        min_temporal_sep=min_temporal_sep,
    )
    return float(lle)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_surrogate_test(
    trajectories: np.ndarray,
    n_surrogates: int = 19,
    surrogate_types: Optional[List[str]] = None,
    ar_order: int = 1,
    rosenstein_max_lag: int = 30,
    rosenstein_min_sep: int = 10,
    n_traj_sample: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    代替数据检验（surrogate data test）。

    对轨迹数据运行三种代替序列的 Lyapunov 指数比较：
      - 相位随机化（phase_randomize）
      - 时序打乱（shuffle）
      - AR(p) 代替（ar_surrogate）

    判断准则（单侧 Theiler et al. 1992 秩检验）：
      - 若真实 LLE > 所有 n_surrogates 个代替序列的 LLE，
        则 p < 1/(n_surrogates+1) ≈ 0.05（n=19）
      - 以 z 分数 (real_lle - mean_surr) / std_surr 量化显著性

    Args:
        trajectories:       shape (n_init, steps, n_regions)。
        n_surrogates:       每种代替类型的代替序列数量（默认 19）。
        surrogate_types:    代替类型列表（默认 ["phase_randomize","shuffle","ar"]）。
        ar_order:           AR 代替模型阶数（默认 1）。
        rosenstein_max_lag: Rosenstein LLE 追踪延迟上限。
        rosenstein_min_sep: Rosenstein NN 最小时间间距。
        n_traj_sample:      使用前 n 条轨迹（节省计算量）。
        seed:               随机数种子。
        output_dir:         保存 surrogate_test.json；None → 不保存。

    Returns:
        results: {
            "real_lle":           float — 真实轨迹 Rosenstein LLE（均值）
            "surrogate_results":  Dict[str, {
                "lle_values":  List[float],   — 各代替序列 LLE
                "mean":        float,
                "std":         float,
                "z_score":     float,         — (real - mean) / std
                "significant": bool,          — real > all surrogates
                "p_value_upper_bound": float, — 1/(n_surrogates+1)
            }]
            "summary":            str         — 人类可读摘要
        }
    """
    if surrogate_types is None:
        surrogate_types = ["phase_randomize", "shuffle", "ar"]

    rng = np.random.default_rng(seed)

    n_init, steps, n_regions = trajectories.shape
    use_n = min(n_traj_sample, n_init)
    traj_sample = trajectories[:use_n]  # (use_n, steps, n_regions)

    logger.info(
        "代替数据检验: %d 条轨迹 × %d 步 × %d 脑区, 每类 %d 个代替序列",
        use_n, steps, n_regions, n_surrogates,
    )

    # ── Compute real LLE (mean over sampled trajectories) ─────────────────────
    real_lles: List[float] = []
    for k in range(use_n):
        lle_k = _lle_from_surrogate_trajectory(
            traj_sample[k],
            max_lag=rosenstein_max_lag,
            min_temporal_sep=rosenstein_min_sep,
        )
        if np.isfinite(lle_k):
            real_lles.append(lle_k)

    if not real_lles:
        logger.warning("  无法计算真实轨迹的 LLE（轨迹过短）。代替检验已跳过。")
        return {"real_lle": float("nan"), "surrogate_results": {}, "summary": "轨迹过短，代替检验失败。"}

    real_lle = float(np.mean(real_lles))
    logger.info("  真实系统 LLE = %.5f (±%.5f, n=%d)", real_lle, float(np.std(real_lles)), len(real_lles))

    surrogate_results: Dict = {}

    # ── Per surrogate type ────────────────────────────────────────────────────
    _generators = {
        "phase_randomize": lambda ts, r: phase_randomize_surrogate(ts, rng=r),
        "shuffle":         lambda ts, r: shuffle_surrogate(ts, rng=r),
        "ar":              lambda ts, r: ar_surrogate(ts, order=ar_order, rng=r),
    }

    for surr_type in surrogate_types:
        if surr_type not in _generators:
            logger.warning("  未知代替类型 '%s'，跳过。", surr_type)
            continue

        gen_fn = _generators[surr_type]
        surr_lles: List[float] = []

        # Deterministic per-type seed offset — avoids hash() variability across
        # Python sessions (PYTHONHASHSEED randomises str hashes by default).
        _TYPE_SEED_OFFSET = {"phase_randomize": 0, "shuffle": 1, "ar": 2}
        type_seed = _TYPE_SEED_OFFSET.get(surr_type, 3)

        for s_idx in range(n_surrogates):
            s_rng = np.random.default_rng([seed, type_seed, s_idx])
            # Average LLE over sampled trajectories for this surrogate instance
            s_lles_traj: List[float] = []
            for k in range(use_n):
                ts_orig = traj_sample[k]  # (steps, n_regions)
                # Generate surrogate for each channel independently
                ts_surr = gen_fn(ts_orig, s_rng)
                lle_s = _lle_from_surrogate_trajectory(
                    ts_surr,
                    max_lag=rosenstein_max_lag,
                    min_temporal_sep=rosenstein_min_sep,
                )
                if np.isfinite(lle_s):
                    s_lles_traj.append(lle_s)

            if s_lles_traj:
                surr_lles.append(float(np.mean(s_lles_traj)))

        if not surr_lles:
            logger.warning("  代替类型 '%s' 无有效 LLE，跳过。", surr_type)
            continue

        mean_surr = float(np.mean(surr_lles))
        std_surr = float(np.std(surr_lles))
        z_score = (real_lle - mean_surr) / max(std_surr, 1e-10)
        significant = bool(real_lle > max(surr_lles))
        p_upper = 1.0 / (n_surrogates + 1)

        logger.info(
            "  [%s] 代替LLE均值=%.5f±%.5f  z=%.2f  显著=%s (p<%.3f)",
            surr_type, mean_surr, std_surr, z_score,
            "是" if significant else "否", p_upper,
        )

        surrogate_results[surr_type] = {
            "lle_values": surr_lles,
            "mean": mean_surr,
            "std": std_surr,
            "z_score": z_score,
            "significant": significant,
            "p_value_upper_bound": p_upper,
        }

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = _build_summary(real_lle, surrogate_results)
    logger.info("  代替检验摘要: %s", summary)

    results: Dict = {
        "real_lle": real_lle,
        "real_lle_values": real_lles,
        "surrogate_results": surrogate_results,
        "summary": summary,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "surrogate_test.json"
        # Convert numpy types for JSON serialisation
        _safe = {
            "real_lle": real_lle,
            "real_lle_values": [float(v) for v in real_lles],
            "surrogate_results": {
                k: {
                    "lle_values": [float(v) for v in vd["lle_values"]],
                    "mean": float(vd["mean"]),
                    "std": float(vd["std"]),
                    "z_score": float(vd["z_score"]),
                    "significant": bool(vd["significant"]),
                    "p_value_upper_bound": float(vd["p_value_upper_bound"]),
                }
                for k, vd in surrogate_results.items()
            },
            "summary": summary,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(_safe, fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s", out_path)

    return results


def _build_summary(real_lle: float, surrogate_results: Dict) -> str:
    """构建人类可读摘要字符串。"""
    if not surrogate_results:
        return "代替检验无有效结果。"

    sig_types = [k for k, v in surrogate_results.items() if v["significant"]]
    total_types = list(surrogate_results.keys())

    if len(sig_types) == len(total_types) and total_types:
        verdict = "✓ 系统具有显著非线性动力学（真实 LLE 超过所有代替序列）"
    elif sig_types:
        not_sig = [k for k in total_types if k not in sig_types]
        verdict = (
            f"△ 部分代替检验显著（{sig_types}），"
            f"不显著：{not_sig}。动力学非线性程度中等。"
        )
    else:
        verdict = (
            "✗ 真实 LLE 未超过代替序列（未能拒绝线性原假设）。"
            "检测到的混沌可能是线性噪声的假象，请增加数据量后重试。"
        )

    parts = [f"真实LLE={real_lle:.5f}"]
    for k, v in surrogate_results.items():
        parts.append(f"{k}: z={v['z_score']:.2f}")
    return verdict + "  |  " + ", ".join(parts)
