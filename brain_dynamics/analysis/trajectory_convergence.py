"""
Trajectory Convergence Analysis
================================

检测系统是否存在 **吸引子结构**（多条轨迹是否随时间相互靠近）。

方法：
  随机抽取 n_pairs 对轨迹 (x_a, x_b)，计算距离随时间的演化：

    D(t) = || x_a(t) − x_b(t) ||₂

  取所有轨迹对的均值：

    D_mean(t) = mean_pairs D(t)

解释：
  D(t) ↓  → 轨迹收敛，存在吸引子
  D(t) ≈ const → 无结构（随机动力学）
  D(t) ↑  → 轨迹发散，可能混沌

输出：
  distance_curve.npy              — shape (steps,)，均值距离曲线
  plots/trajectory_convergence.png
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_pairwise_distances(
    trajectories: np.ndarray,
    n_pairs: int = 50,
    seed: int = 0,
) -> np.ndarray:
    """
    随机抽取轨迹对，计算每步的均值 L2 距离。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        n_pairs:      随机抽取的轨迹对数量（默认 50）。
        seed:         随机种子。

    Returns:
        mean_distances: shape (steps,)，所有轨迹对距离的均值。
    """
    n_init, steps, n_regions = trajectories.shape
    rng = np.random.default_rng(seed)

    # Clamp n_pairs to the maximum number of unique pairs
    max_pairs = n_init * (n_init - 1) // 2
    n_pairs = min(n_pairs, max_pairs)

    if n_pairs <= 0 or n_init < 2:
        logger.warning("轨迹对不足（n_init=%d），跳过收敛分析。", n_init)
        return np.zeros(steps, dtype=np.float32)

    # Collect exactly n_pairs unique (a, b) pairs with a ≠ b.
    # We use a while-loop with a 20× attempt budget to guarantee the exact count
    # regardless of n_init (the old 3× oversampling heuristic was not guaranteed
    # to produce enough valid pairs when n_init is small).
    pairs_a: List[int] = []
    pairs_b: List[int] = []
    seen = set()
    attempts = 0
    max_attempts = n_pairs * 20
    while len(pairs_a) < n_pairs and attempts < max_attempts:
        a = int(rng.integers(0, n_init))
        b = int(rng.integers(0, n_init))
        if a != b and (a, b) not in seen:
            pairs_a.append(a)
            pairs_b.append(b)
            seen.add((a, b))
        attempts += 1

    if not pairs_a:
        return np.zeros(steps, dtype=np.float32)

    pa = np.array(pairs_a)
    pb = np.array(pairs_b)

    # Compute pairwise distances: shape (n_pairs_actual, steps)
    diff = trajectories[pa] - trajectories[pb]  # (n_pairs_actual, steps, n_regions)
    distances = np.linalg.norm(diff, axis=2)     # (n_pairs_actual, steps)

    return distances.mean(axis=0).astype(np.float32)


def _convergence_label(mean_distances: np.ndarray) -> str:
    """Return a human-readable convergence classification."""
    if len(mean_distances) < 2:
        return "insufficient_data"
    early = float(mean_distances[: len(mean_distances) // 4].mean())
    late = float(mean_distances[-len(mean_distances) // 4 :].mean())
    ratio = late / (early + 1e-12)
    if ratio < 0.7:
        return "converging"
    if ratio > 1.3:
        return "diverging"
    return "no_structure"


def run_trajectory_convergence(
    trajectories: np.ndarray,
    n_pairs: int = 50,
    seed: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行轨迹收敛分析。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        n_pairs:      随机轨迹对数量（默认 50）。
        seed:         随机种子。
        output_dir:   保存 distance_curve.npy；None → 不保存。

    Returns:
        results: {
            "mean_distances":         np.ndarray (steps,)   均值距离曲线
            "initial_mean_distance":  float                 起始均值距离
            "final_mean_distance":    float                 终止均值距离
            "distance_ratio":         float                 final/initial
            "convergence_label":      str                   "converging" / "diverging" / "no_structure"
        }
    """
    n_init, steps, n_regions = trajectories.shape
    logger.info(
        "轨迹收敛分析: n_traj=%d, steps=%d, n_pairs=%d",
        n_init,
        steps,
        n_pairs,
    )

    mean_dist = compute_pairwise_distances(trajectories, n_pairs=n_pairs, seed=seed)

    initial_d = float(mean_dist[:max(1, steps // 10)].mean())
    final_d = float(mean_dist[-max(1, steps // 10) :].mean())
    ratio = final_d / (initial_d + 1e-12)
    label = _convergence_label(mean_dist)

    logger.info(
        "  初始距离=%.4f  终止距离=%.4f  比率=%.3f  → %s",
        initial_d,
        final_d,
        ratio,
        label,
    )
    if label == "converging":
        logger.info("  → 检测到吸引子结构（轨迹收敛）")
    elif label == "diverging":
        logger.info("  → 轨迹发散（可能混沌）")
    else:
        logger.info("  → 无明显收敛或发散结构")

    results = {
        "mean_distances": mean_dist,
        "initial_mean_distance": initial_d,
        "final_mean_distance": final_d,
        "distance_ratio": ratio,
        "convergence_label": label,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "distance_curve.npy", mean_dist)
        logger.info("  → 已保存: %s/distance_curve.npy", output_dir)

    return results
