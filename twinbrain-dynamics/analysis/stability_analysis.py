"""
Stability Analysis
==================

验证系统是否存在稳定动力学。

**两种分类方法**

方法 A（邻接差分，向后兼容）：
  Δ(t) = ||x(t) − x(t−1)||₂

方法 B（延迟距离，改进版，默认）：
  Δ(t) = ||x(t + ΔT) − x(t)||₂   （推荐 ΔT = 50）

  分类规则（方法 B）：
  - Δ_mean < 1e-3            → fixed_point（不动点）
  - Δ_var  < 1e-3            → limit_cycle（极限环）
  - Δ_mean < 0.1             → metastable（亚稳态）
  - else                     → unstable（混沌 / 不稳定）

  方法 B 使用更宽松的阈值，避免大多数轨迹被误判为 unstable。

输出文件：outputs/stability_metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Threshold below which ||Δx|| is considered convergence (method A, adjacent)
_CONVERGENCE_TOL = 1e-4

# Delay lag for method B (delay-distance)
_DEFAULT_DELAY_DT = 50


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
    分析单条轨迹的稳定性（同时使用方法 A 和方法 B）。

    Args:
        trajectory:        shape (T, n_regions)。
        convergence_tol:   方法 A 固定点收敛阈值。
        period_max_lag:    方法 A 周期检测最大延迟。
        delay_dt:          方法 B 延迟步数（默认 50）。

    Returns:
        metrics: {
            "classification": str,          # 方法 B（改进版）分类结果
            "classification_v1": str,        # 方法 A（向后兼容）分类结果
            "mean_delta": float,
            "std_delta": float,
            "max_delta": float,
            "final_delta": float,
            "convergence_step": int | None,
            "delay_mean": float,             # 方法 B：延迟距离均值
            "delay_var": float,              # 方法 B：延迟距离方差
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

    # Convergence step (method A): first step where delta falls below threshold
    convergence_step = None
    for t, d in enumerate(deltas):
        if d < convergence_tol:
            convergence_step = int(t)
            break

    # Delay-distance stats for reporting
    delays = compute_delay_distances(trajectory, delay_dt=delay_dt)
    delay_mean = float(delays.mean()) if len(delays) > 0 else 0.0
    delay_var = float(delays.var()) if len(delays) > 0 else 0.0

    return {
        "classification": classification_v2,   # improved method (default)
        "classification_v1": classification_v1,  # original method
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "max_delta": float(deltas.max()),
        "final_delta": float(deltas[-1]) if len(deltas) > 0 else 0.0,
        "convergence_step": convergence_step,
        "delay_mean": delay_mean,
        "delay_var": delay_var,
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

    同时使用方法 A（邻接差分）和方法 B（延迟距离，改进版）输出分类统计。
    ``stability_metrics.json`` 包含两种方法的结果，主分类使用方法 B。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        convergence_tol: 方法 A 固定点收敛阈值。
        period_max_lag:  方法 A 周期检测最大延迟。
        delay_dt:        方法 B 延迟步数（默认 50）。
        output_dir:      保存 stability_metrics.json；None → 不保存。

    Returns:
        summary: {
            "per_trajectory": List[Dict],
            "classification_counts": Dict,       # 方法 B 主结果
            "classification_counts_v1": Dict,    # 方法 A 参考结果
            "mean_convergence_step": float | None,
            "fraction_converged": float,
            "fraction_limit_cycle": float,
            "fraction_metastable": float,
            "fraction_unstable": float,
        }
    """
    n_traj = trajectories.shape[0]
    per_traj: List[Dict] = []
    class_counts: Dict[str, int] = {
        "fixed_point": 0,
        "limit_cycle": 0,
        "metastable": 0,
        "unstable": 0,
    }
    class_counts_v1: Dict[str, int] = {
        "fixed_point": 0,
        "limit_cycle": 0,
        "metastable": 0,
        "unstable": 0,
    }

    convergence_steps: List[int] = []

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
        cls = metrics["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        cls_v1 = metrics["classification_v1"]
        class_counts_v1[cls_v1] = class_counts_v1.get(cls_v1, 0) + 1
        if metrics["convergence_step"] is not None:
            convergence_steps.append(metrics["convergence_step"])

    mean_conv = float(np.mean(convergence_steps)) if convergence_steps else None

    summary = {
        "per_trajectory": per_traj,
        "classification_counts": class_counts,
        "classification_counts_v1": class_counts_v1,
        "mean_convergence_step": mean_conv,
        "fraction_converged": class_counts["fixed_point"] / n_traj,
        "fraction_limit_cycle": class_counts["limit_cycle"] / n_traj,
        "fraction_metastable": class_counts["metastable"] / n_traj,
        "fraction_unstable": class_counts["unstable"] / n_traj,
    }

    logger.info(
        "  [方法B-延迟距离] 不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        summary["fraction_converged"] * 100,
        summary["fraction_limit_cycle"] * 100,
        summary["fraction_metastable"] * 100,
        summary["fraction_unstable"] * 100,
    )
    logger.info(
        "  [方法A-邻接差分] 不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        class_counts_v1["fixed_point"] / n_traj * 100,
        class_counts_v1["limit_cycle"] / n_traj * 100,
        class_counts_v1["metastable"] / n_traj * 100,
        class_counts_v1["unstable"] / n_traj * 100,
    )
    if mean_conv is not None:
        logger.info("  平均收敛步数: %.1f", mean_conv)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "stability_metrics.json"
        json_summary = {k: v for k, v in summary.items() if k != "per_trajectory"}
        json_summary["classification_counts"] = class_counts
        json_summary["classification_counts_v1"] = class_counts_v1
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
