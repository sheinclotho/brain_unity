"""
Stability Analysis
==================

验证系统是否存在稳定动力学。

方法：
  Δ(t) = ||x(t) − x(t−1)||₂

  - Δ → 0          : fixed point（不动点）
  - 周期变化         : limit cycle（极限环）
  - 缓慢漂移         : metastable state（亚稳态）
  - 持续大幅波动      : chaotic / unstable

输出文件：outputs/stability_metrics.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Threshold below which ||Δx|| is considered convergence
_CONVERGENCE_TOL = 1e-4


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
) -> Dict:
    """
    分析单条轨迹的稳定性。

    Args:
        trajectory:        shape (T, n_regions)。
        convergence_tol:   固定点收敛阈值。
        period_max_lag:    周期检测最大延迟。

    Returns:
        metrics: {
            "classification": str,
            "mean_delta": float,
            "std_delta": float,
            "max_delta": float,
            "final_delta": float,
            "convergence_step": int | None,
        }
    """
    convergence_tol = float(convergence_tol)
    deltas = compute_state_deltas(trajectory)

    classification = classify_dynamics(
        deltas,
        convergence_tol=convergence_tol,
        period_max_lag=period_max_lag,
    )

    # Convergence step: first step where delta falls below threshold
    convergence_step = None
    for t, d in enumerate(deltas):
        if d < convergence_tol:
            convergence_step = int(t)
            break

    return {
        "classification": classification,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "max_delta": float(deltas.max()),
        "final_delta": float(deltas[-1]) if len(deltas) > 0 else 0.0,
        "convergence_step": convergence_step,
    }


def run_stability_analysis(
    trajectories: np.ndarray,
    convergence_tol: float = _CONVERGENCE_TOL,
    period_max_lag: int = 100,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹运行稳定性分析，汇总结果。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        convergence_tol: 固定点收敛阈值。
        period_max_lag:  周期检测最大延迟。
        output_dir:      保存 stability_metrics.json；None → 不保存。

    Returns:
        summary: {
            "per_trajectory": List[Dict],   # per-trajectory metrics
            "classification_counts": Dict,  # {class: count}
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

    convergence_steps: List[int] = []

    logger.info(
        "稳定性分析: %d 条轨迹, 每条 %d 步", n_traj, trajectories.shape[1]
    )

    for i in range(n_traj):
        metrics = analyze_trajectory_stability(
            trajectories[i],
            convergence_tol=convergence_tol,
            period_max_lag=period_max_lag,
        )
        per_traj.append(metrics)
        cls = metrics["classification"]
        class_counts[cls] = class_counts.get(cls, 0) + 1
        if metrics["convergence_step"] is not None:
            convergence_steps.append(metrics["convergence_step"])

    mean_conv = float(np.mean(convergence_steps)) if convergence_steps else None

    summary = {
        "per_trajectory": per_traj,
        "classification_counts": class_counts,
        "mean_convergence_step": mean_conv,
        "fraction_converged": class_counts["fixed_point"] / n_traj,
        "fraction_limit_cycle": class_counts["limit_cycle"] / n_traj,
        "fraction_metastable": class_counts["metastable"] / n_traj,
        "fraction_unstable": class_counts["unstable"] / n_traj,
    }

    logger.info(
        "  不动点: %.1f%%  极限环: %.1f%%  亚稳态: %.1f%%  不稳定: %.1f%%",
        summary["fraction_converged"] * 100,
        summary["fraction_limit_cycle"] * 100,
        summary["fraction_metastable"] * 100,
        summary["fraction_unstable"] * 100,
    )
    if mean_conv is not None:
        logger.info("  平均收敛步数: %.1f", mean_conv)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "stability_metrics.json"
        # Serialize without per_trajectory (too large) for the JSON file
        json_summary = {k: v for k, v in summary.items() if k != "per_trajectory"}
        # Add aggregate stats per trajectory class
        json_summary["classification_counts"] = class_counts
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
