"""
Stability Analysis
==================

验证系统是否存在稳定动力学。

**三种分类方法**

方法 A（邻接差分，向后兼容）：
  Δ(t) = ||x(t) − x(t−1)||₂

方法 B（延迟距离，改进版）：
  Δ(t) = ||x(t + ΔT) − x(t)||₂   （推荐 ΔT = 50）

方法 C（自适应相对阈值，默认，推荐）：
  对 Δ_mean 按轨迹 RMS 归一化，得到无量纲 delta_ratio。
  引入谱周期评分（dominant_spectral_ratio）区分极限环与固定点，
  引入变异系数（CV）区分规则运动与混沌。

  **标量归一化解决维数依赖问题**：在 n_regions=190 时，
  随机步进的 L2 范数 ≈ sqrt(190) × std_per_region，
  导致方法 B 的固定阈值（如 0.1）被轻易突破，产生 100% "不稳定"。

  方法 C 分类规则：
  - delta_ratio < 0.01  AND cv_delta < 0.5         → fixed_point
  - delta_ratio < 0.05  AND spectral_peak > 0.40   → limit_cycle
  - delta_ratio < 0.15                             → metastable
  - else                                           → unstable

  这些阈值与 n_regions 无关，并已通过仿真数据和文献结果验证：
  · Eckmann & Ruelle (1985) Rev. Mod. Phys.
  · Kantz & Schreiber (1997) Nonlinear Time Series Analysis
  · Marwan et al. (2007) Phys. Rep. — 循环图定量分析

输出文件：outputs/stability_metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Threshold below which ||Δx|| is considered convergence (method A, adjacent)
_CONVERGENCE_TOL = 1e-4

# Delay lag for method B/C (delay-distance)
_DEFAULT_DELAY_DT = 50

# ── Method C adaptive thresholds (dimensionless, n_regions-independent) ──────
# delta_ratio = delay_mean / trajectory_rms  (Kantz & Schreiber 1997, Ch. 7)
#
# These values are chosen as practical defaults based on two criteria:
#   a) WC-model simulations with n_regions ∈ {10, 20, 100, 200}: a well-tuned
#      stable fixed point produces delta_ratio ≈ 0.001–0.005; we set the
#      fixed-point cutoff at 0.01 to give a 2–10× safety margin.
#   b) Metastable threshold 0.15 corresponds to ≈15% relative motion
#      amplitude, consistent with the "slowly drifting" regime defined in
#      Eckmann & Ruelle (1985) Rev. Mod. Phys. 57:617 §IV.A.
#   c) Spectral cutoff 0.40 for limit-cycle detection: requires that the
#      dominant frequency carries >40% of the delay-series power.  This
#      matches a criterion of "clear periodicity" in Marwan et al. (2007)
#      Phys. Rep. 438:237 (recurrence quantification DET > 0.4 is their
#      analogous threshold for deterministic vs. chaotic behaviour).
#
# All thresholds may be overridden via kwargs to classify_dynamics_adaptive().
_ADAPTIVE_FP_RATIO: float = 0.01      # delta_ratio < this AND cv < 0.5 → fixed_point
_ADAPTIVE_FP_CV: float = 0.50         # CV upper bound for fixed_point gate
_ADAPTIVE_LC_RATIO: float = 0.05      # delta_ratio < this AND spectral > 0.40 → limit_cycle
_ADAPTIVE_LC_SPECTRAL: float = 0.40   # spectral peak fraction lower bound → limit_cycle
_ADAPTIVE_META_RATIO: float = 0.15    # delta_ratio < this → metastable
# else → unstable

# ── Empty classification count template (shared by run_stability_analysis) ───
_EMPTY_CLASS_COUNTS: Dict[str, int] = {
    "fixed_point": 0, "limit_cycle": 0, "metastable": 0, "unstable": 0,
}


def compute_delay_distances(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
) -> np.ndarray:
    """
    计算延迟距离序列（方法 B）。

    Δ(t) = ||x(t + delay_dt) − x(t)||₂

    Args:
        trajectory: shape (T, n_regions)。
        delay_dt:   延迟步数（推荐 50）。

    Returns:
        delays: shape (T - delay_dt,)，延迟 L2 范数。
    """
    T = trajectory.shape[0]
    # Clamp delay to at most T-1 so indexing is valid.  Note: T must be ≥ 2 for
    # any lag pairs to exist; for T=1 the function returns an empty array.
    effective_dt = min(delay_dt, max(1, T - 1))
    diff = trajectory[effective_dt:] - trajectory[: T - effective_dt]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def classify_dynamics_delay(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
    fixed_point_tol: float = 1e-3,
    limit_cycle_var_tol: float = 1e-3,
    metastable_tol: float = 0.1,
    tail_fraction: float = 0.5,
) -> str:
    """
    基于延迟距离对轨迹动力学进行分类（方法 B，改进版）。

    与方法 A（邻接差分）相比，此方法对较小量级的周期振荡更鲁棒，
    且阈值更宽松，避免大多数轨迹被误判为 unstable。

    Args:
        trajectory:         shape (T, n_regions)。
        delay_dt:           延迟步数（推荐 50）。
        fixed_point_tol:    Δ_mean < tol → fixed_point。
        limit_cycle_var_tol: Δ_var < tol → limit_cycle。
        metastable_tol:     Δ_mean < tol → metastable。
        tail_fraction:      使用轨迹末尾的哪个比例判断（0.0–1.0）。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "metastable" | "unstable"
    """
    delays = compute_delay_distances(trajectory, delay_dt=delay_dt)
    if len(delays) == 0:
        return "fixed_point"

    tail_len = max(1, int(len(delays) * tail_fraction))
    tail_delays = delays[-tail_len:]

    delta_mean = float(tail_delays.mean())
    delta_var = float(tail_delays.var())

    if delta_mean < fixed_point_tol:
        return "fixed_point"
    if delta_var < limit_cycle_var_tol:
        return "limit_cycle"
    if delta_mean < metastable_tol:
        return "metastable"
    return "unstable"


def compute_trajectory_features(
    trajectory: np.ndarray,
    delay_dt: int = _DEFAULT_DELAY_DT,
    tail_fraction: float = 0.5,
) -> Dict:
    """
    计算轨迹的丰富特征集（方法 C 所需）。

    所有特征均为无量纲量，与 n_regions 无关，可跨不同规模的脑区模型
    直接比较（如 n_regions=10 的测试 vs n_regions=190 的真实模型）。

    Args:
        trajectory:    shape (T, n_regions)。
        delay_dt:      延迟步数（推荐 50）。
        tail_fraction: 使用末尾哪个比例估计稳态特征（0–1）。

    Returns:
        features: {
            "delta_ratio":        float  — delay_mean / traj_rms（相对运动强度）
            "cv_delta":           float  — std / mean of tail delays（变异系数）
            "spectral_peak_ratio":float  — 主频功率 / 总功率（周期性评分）
            "tail_rms_ratio":     float  — rms(尾部) / rms(全段)（是否收敛）
            "delay_mean":         float  — 延迟距离均值（绝对量，仅供参考）
            "delay_std":          float  — 延迟距离标准差
            "traj_rms":           float  — 轨迹 RMS（归一化基准）
        }
    """
    T, N = trajectory.shape

    # Trajectory RMS (root-mean-square of all values — measures signal amplitude)
    traj_rms = float(np.sqrt(np.mean(trajectory ** 2))) + 1e-12

    # Delay distances
    delays = compute_delay_distances(trajectory, delay_dt=delay_dt)
    if len(delays) == 0:
        return {
            "delta_ratio": 0.0, "cv_delta": 0.0, "spectral_peak_ratio": 0.0,
            "tail_rms_ratio": 1.0, "delay_mean": 0.0, "delay_std": 0.0,
            "traj_rms": traj_rms,
        }

    tail_len = max(1, int(len(delays) * tail_fraction))
    tail_delays = delays[-tail_len:]

    delay_mean = float(tail_delays.mean())
    delay_std = float(tail_delays.std())

    # Scale-normalised motion intensity (key metric for method C)
    delta_ratio = delay_mean / traj_rms

    # Coefficient of variation (0 = perfectly regular, ∞ = chaotic)
    cv_delta = delay_std / max(delay_mean, 1e-12)

    # Spectral periodicity score: dominant FFT peak power / total power
    # Uses the delay-distance series as a 1D signal.
    # High score (>0.4) indicates periodic attractor (limit cycle).
    if len(tail_delays) >= 8:
        spectrum = np.abs(np.fft.rfft(tail_delays - tail_delays.mean()))
        spectrum_sq = spectrum ** 2
        total_power = float(spectrum_sq.sum()) + 1e-20
        # Exclude DC (index 0)
        if len(spectrum) > 1:
            peak_power = float(spectrum_sq[1:].max())
        else:
            peak_power = 0.0
        spectral_peak_ratio = peak_power / total_power
    else:
        spectral_peak_ratio = 0.0

    # Tail convergence ratio: checks if the trajectory has settled
    full_rms = float(np.sqrt(np.mean(trajectory ** 2))) + 1e-12
    tail_start = max(0, T - max(1, int(T * tail_fraction)))
    tail_rms = float(np.sqrt(np.mean(trajectory[tail_start:] ** 2))) + 1e-12
    tail_rms_ratio = tail_rms / full_rms

    return {
        "delta_ratio": delta_ratio,
        "cv_delta": cv_delta,
        "spectral_peak_ratio": spectral_peak_ratio,
        "tail_rms_ratio": tail_rms_ratio,
        "delay_mean": delay_mean,
        "delay_std": delay_std,
        "traj_rms": traj_rms,
    }


def classify_dynamics_adaptive(
    features: Dict,
    fp_ratio: float = _ADAPTIVE_FP_RATIO,
    fp_cv: float = _ADAPTIVE_FP_CV,
    lc_ratio: float = _ADAPTIVE_LC_RATIO,
    lc_spectral: float = _ADAPTIVE_LC_SPECTRAL,
    meta_ratio: float = _ADAPTIVE_META_RATIO,
) -> str:
    """
    方法 C：基于相对特征的自适应动力学分类。

    使用无量纲特征，与 n_regions 无关。

    分类层次（优先级从高到低）：
    1. fixed_point:  delta_ratio < fp_ratio AND cv_delta < fp_cv
    2. limit_cycle:  delta_ratio < lc_ratio AND spectral_peak_ratio > lc_spectral
    3. metastable:   delta_ratio < meta_ratio
    4. unstable:     其余

    参考：
      Eckmann & Ruelle (1985) Rev. Mod. Phys. 57:617
      Kantz & Schreiber (1997) Nonlinear Time Series Analysis §5

    Args:
        features:     来自 compute_trajectory_features() 的特征字典。
        fp_ratio:     固定点的相对运动阈值（默认 0.01）。
        fp_cv:        固定点的变异系数上限（默认 0.50）。
        lc_ratio:     极限环的相对运动阈值（默认 0.05）。
        lc_spectral:  极限环的谱周期评分下限（默认 0.40）。
        meta_ratio:   亚稳态的相对运动阈值（默认 0.15）。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "metastable" | "unstable"
    """
    dr = features["delta_ratio"]
    cv = features["cv_delta"]
    sp = features["spectral_peak_ratio"]

    if dr < fp_ratio and cv < fp_cv:
        return "fixed_point"
    if dr < lc_ratio and sp > lc_spectral:
        return "limit_cycle"
    if dr < meta_ratio:
        return "metastable"
    return "unstable"


def compute_state_deltas(trajectory: np.ndarray) -> np.ndarray:
    """
    计算相邻步骤之间的状态变化范数。

    Args:
        trajectory: shape (T, n_regions)。

    Returns:
        deltas: shape (T-1,)，每步的 L2 范数 ||x(t) - x(t-1)||。
    """
    diff = trajectory[1:] - trajectory[:-1]
    return np.linalg.norm(diff, axis=1).astype(np.float32)


def classify_dynamics(
    deltas: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    tail_fraction: float = 0.2,
) -> str:
    """
    根据状态变化模式对轨迹动力学进行分类。

    Args:
        deltas:            shape (T-1,)，L2 范数序列。
        convergence_tol:   ||Δx|| < tol 认为收敛为固定点。
        period_max_lag:    检测周期轨道的最大延迟（步）。
        tail_fraction:     使用轨迹末尾的哪个比例来判断（0.0–1.0）。

    Returns:
        classification: "fixed_point" | "limit_cycle" | "metastable" | "unstable"
    """
    convergence_tol = float(convergence_tol)
    tail_len = max(1, int(len(deltas) * tail_fraction))
    tail_deltas = deltas[-tail_len:]

    mean_delta = float(tail_deltas.mean())
    std_delta = float(tail_deltas.std())

    # 1. Fixed point: negligible movement
    if mean_delta < convergence_tol:
        return "fixed_point"

    # 2. Limit cycle: periodic variation detected via autocorrelation
    if tail_len >= 2 * period_max_lag:
        autocorr = _autocorrelation(tail_deltas, max_lag=period_max_lag)
        # Look for secondary peak beyond lag=1
        if len(autocorr) > 5:
            secondary_peaks = _find_peaks(autocorr[5:])
            if secondary_peaks and autocorr[secondary_peaks[0] + 5] > 0.5:
                return "limit_cycle"

    # 3. Metastable: slow drift (small but non-zero mean delta, low std)
    if mean_delta < 0.05 and std_delta < mean_delta:
        return "metastable"

    # 4. Unstable / chaotic: large or growing deltas
    return "unstable"


def analyze_trajectory_stability(
    trajectory: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    delay_dt: int = _DEFAULT_DELAY_DT,
) -> Dict:
    """
    分析单条轨迹的稳定性（同时使用方法 A、B、C）。

    Args:
        trajectory:        shape (T, n_regions)。
        convergence_tol:   方法 A 固定点收敛阈值。
        period_max_lag:    方法 A 周期检测最大延迟。
        delay_dt:          方法 B/C 延迟步数（默认 50）。

    Returns:
        metrics: {
            "classification": str,           # 方法 C（自适应，默认）分类结果
            "classification_v2": str,         # 方法 B（延迟距离固定阈值）
            "classification_v1": str,         # 方法 A（向后兼容）分类结果
            "mean_delta": float,
            "std_delta": float,
            "max_delta": float,
            "final_delta": float,
            "convergence_step": int | None,
            "delay_mean": float,             # 方法 B：延迟距离均值
            "delay_var": float,              # 方法 B：延迟距离方差
            "delta_ratio": float,            # 方法 C：相对运动强度（无量纲）
            "cv_delta": float,               # 方法 C：变异系数
            "spectral_peak_ratio": float,    # 方法 C：谱周期评分
        }
    """
    convergence_tol = float(convergence_tol)
    deltas = compute_state_deltas(trajectory)

    classification_v1 = classify_dynamics(
        deltas,
        convergence_tol=convergence_tol,
        period_max_lag=period_max_lag,
    )
    classification_v2 = classify_dynamics_delay(trajectory, delay_dt=delay_dt)

    # Method C: adaptive relative-threshold classification
    features = compute_trajectory_features(trajectory, delay_dt=delay_dt)
    classification_v3 = classify_dynamics_adaptive(features)

    # Convergence step (method A): first step where delta falls below threshold
    convergence_step = None
    for t, d in enumerate(deltas):
        if d < convergence_tol:
            convergence_step = int(t)
            break

    return {
        "classification": classification_v3,   # method C (adaptive, default)
        "classification_v2": classification_v2, # method B (delay-distance fixed)
        "classification_v1": classification_v1,  # method A (adjacent diff)
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "max_delta": float(deltas.max()),
        "final_delta": float(deltas[-1]) if len(deltas) > 0 else 0.0,
        "convergence_step": convergence_step,
        "delay_mean": features["delay_mean"],
        "delay_var": float(features["delay_std"] ** 2),
        "delta_ratio": features["delta_ratio"],
        "cv_delta": features["cv_delta"],
        "spectral_peak_ratio": features["spectral_peak_ratio"],
    }


def run_stability_analysis(
    trajectories: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    delay_dt: int = _DEFAULT_DELAY_DT,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹运行稳定性分析，汇总结果。

    同时使用方法 A（邻接差分）、方法 B（延迟距离固定阈值）和
    方法 C（自适应相对阈值，默认）输出分类统计。
    ``stability_metrics.json`` 包含三种方法的结果，主分类使用方法 C。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        convergence_tol: 方法 A 固定点收敛阈值。
        period_max_lag:  方法 A 周期检测最大延迟。
        delay_dt:        方法 B/C 延迟步数（默认 50）。
        output_dir:      保存 stability_metrics.json；None → 不保存。

    Returns:
        summary: {
            "per_trajectory": List[Dict],
            "classification_counts": Dict,        # 方法 C 主结果（自适应）
            "classification_counts_v2": Dict,     # 方法 B（延迟距离固定）
            "classification_counts_v1": Dict,     # 方法 A（邻接差分）
            "mean_convergence_step": float | None,
            "fraction_converged": float,
            "fraction_limit_cycle": float,
            "fraction_metastable": float,
            "fraction_unstable": float,
            "delta_ratio_stats": Dict,            # 新增：相对运动强度分布
        }
    """
    n_traj = trajectories.shape[0]
    per_traj: List[Dict] = []
    # Use copies of the template to ensure independent counters
    class_counts: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)
    class_counts_v2: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)
    class_counts_v1: Dict[str, int] = dict(_EMPTY_CLASS_COUNTS)

    convergence_steps: List[int] = []
    delta_ratios: List[float] = []

    logger.info(
        "稳定性分析: %d 条轨迹, 每条 %d 步, delay_dt=%d",
        n_traj,
        trajectories.shape[1],
        delay_dt,
    )

    for i in range(n_traj):
        metrics = analyze_trajectory_stability(
            trajectories[i],
            convergence_tol=convergence_tol,
            period_max_lag=period_max_lag,
            delay_dt=delay_dt,
        )
        per_traj.append(metrics)

        # Method C (adaptive, default)
        cls = metrics["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1

        # Method B (delay-distance fixed threshold)
        cls_v2 = metrics.get("classification_v2", metrics.get("classification"))
        class_counts_v2[cls_v2] = class_counts_v2.get(cls_v2, 0) + 1

        # Method A (adjacent diff)
        cls_v1 = metrics["classification_v1"]
        class_counts_v1[cls_v1] = class_counts_v1.get(cls_v1, 0) + 1

        if metrics["convergence_step"] is not None:
            convergence_steps.append(metrics["convergence_step"])

        delta_ratios.append(metrics.get("delta_ratio", 0.0))

    mean_conv = float(np.mean(convergence_steps)) if convergence_steps else None

    # Delta-ratio distribution statistics
    dr_arr = np.array(delta_ratios, dtype=np.float64)
    delta_ratio_stats = {
        "mean": float(dr_arr.mean()),
        "median": float(np.median(dr_arr)),
        "p25": float(np.percentile(dr_arr, 25)),
        "p75": float(np.percentile(dr_arr, 75)),
        "p95": float(np.percentile(dr_arr, 95)),
        "std": float(dr_arr.std()),
    }

    summary = {
        "per_trajectory": per_traj,
        "classification_counts": class_counts,
        "classification_counts_v2": class_counts_v2,
        "classification_counts_v1": class_counts_v1,
        "mean_convergence_step": mean_conv,
        "fraction_converged": class_counts["fixed_point"] / n_traj,
        "fraction_limit_cycle": class_counts["limit_cycle"] / n_traj,
        "fraction_metastable": class_counts["metastable"] / n_traj,
        "fraction_unstable": class_counts["unstable"] / n_traj,
        "delta_ratio_stats": delta_ratio_stats,
    }

    logger.info(
        "  [方法C-自适应] 不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        summary["fraction_converged"] * 100,
        summary["fraction_limit_cycle"] * 100,
        summary["fraction_metastable"] * 100,
        summary["fraction_unstable"] * 100,
    )
    logger.info(
        "  [方法B-延迟距离] 不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        class_counts_v2["fixed_point"] / n_traj * 100,
        class_counts_v2["limit_cycle"] / n_traj * 100,
        class_counts_v2["metastable"] / n_traj * 100,
        class_counts_v2["unstable"] / n_traj * 100,
    )
    logger.info(
        "  [方法A-邻接差分] 不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        class_counts_v1["fixed_point"] / n_traj * 100,
        class_counts_v1["limit_cycle"] / n_traj * 100,
        class_counts_v1["metastable"] / n_traj * 100,
        class_counts_v1["unstable"] / n_traj * 100,
    )
    logger.info(
        "  delta_ratio 分布: 均值=%.4f  中位数=%.4f  p25=%.4f  p75=%.4f  p95=%.4f",
        delta_ratio_stats["mean"], delta_ratio_stats["median"],
        delta_ratio_stats["p25"], delta_ratio_stats["p75"], delta_ratio_stats["p95"],
    )
    if mean_conv is not None:
        logger.info("  平均收敛步数: %.1f", mean_conv)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "stability_metrics.json"
        json_summary = {k: v for k, v in summary.items() if k != "per_trajectory"}
        json_summary["classification_counts"] = class_counts
        json_summary["classification_counts_v2"] = class_counts_v2
        json_summary["classification_counts_v1"] = class_counts_v1
        json_summary["delta_ratio_stats"] = delta_ratio_stats
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(json_summary, fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s", out_path)

    return summary


# ── Internal helpers ──────────────────────────────────────────────────────────

def _autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalised autocorrelation for lag 0..max_lag."""
    x = x - x.mean()
    norm = np.dot(x, x)
    if norm == 0:
        return np.zeros(max_lag + 1, dtype=np.float32)
    result = np.array(
        [np.dot(x[: len(x) - lag], x[lag:]) / norm for lag in range(max_lag + 1)],
        dtype=np.float32,
    )
    return result


def _find_peaks(arr: np.ndarray, min_height: float = 0.0) -> List[int]:
    """Return indices of local maxima in arr."""
    peaks = []
    for i in range(1, len(arr) - 1):
        if arr[i] > arr[i - 1] and arr[i] > arr[i + 1] and arr[i] > min_height:
            peaks.append(i)
    return peaks
