"""
Lyapunov Exponent Estimation
==============================

估计系统最大 Lyapunov 指数，用于判断系统是否混沌。

方法（有限时间 Lyapunov 指数，FTLE）：

  对每条轨迹 x(t)：
  1. 创建微小扰动  x'(0) = x(0) + ε，ε ≈ 1e-5
  2. 同时演化原始和扰动轨迹
  3. 计算距离  δ(t) = ||x'(t) − x(t)||
  4. 用线性回归估计  λ ≈ slope(log δ(t))

解释：
  λ < 0  → 收敛系统（轨迹相互靠近）
  λ ≈ 0  → 边缘稳定
  λ > 0  → 混沌系统（轨迹指数发散）

输出：
  lyapunov_values.npy        — 每条轨迹的 Lyapunov 指数
  stability_metrics.json     — 追加 mean_lyapunov / median_lyapunov
  plots/lyapunov_histogram.png
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default small perturbation magnitude
_DEFAULT_EPSILON = 1e-5

# Fraction of trajectory to use for linear regression (skip transient)
_FIT_SKIP_FRACTION = 0.1


def estimate_lyapunov_exponent(
    trajectory: np.ndarray,
    simulator,
    epsilon: float = _DEFAULT_EPSILON,
    skip_fraction: float = _FIT_SKIP_FRACTION,
) -> float:
    """
    估计单条轨迹的最大 Lyapunov 指数。

    Args:
        trajectory:     shape (T, n_regions)，已经计算好的轨迹。
        simulator:      ``BrainDynamicsSimulator`` 实例（WC 模式）。
        epsilon:        初始扰动幅度（默认 1e-5）。
        skip_fraction:  跳过轨迹前段的比例（避免瞬态效应）。

    Returns:
        lambda_est: 估计的最大 Lyapunov 指数（每步）。
    """
    T, n_regions = trajectory.shape
    x0 = trajectory[0].copy()

    # Perturbed initial state: random unit vector scaled by epsilon
    rng = np.random.default_rng(int(abs(x0.sum() * 1e6)) % (2**31))
    perturb = rng.random(n_regions).astype(np.float32)
    perturb /= (np.linalg.norm(perturb) + 1e-12)
    x0_perturbed = np.clip(x0 + epsilon * perturb, 0.0, 1.0)

    # Roll out both trajectories with the simulator
    traj_orig, _ = simulator.rollout(x0=x0, steps=T, stimulus=None)
    traj_pert, _ = simulator.rollout(x0=x0_perturbed, steps=T, stimulus=None)

    # Compute log-distance over time; skip initial transient
    dist = np.linalg.norm(traj_pert - traj_orig, axis=1)  # (T,)
    dist = np.maximum(dist, 1e-20)  # avoid log(0)
    log_dist = np.log(dist)

    skip = max(1, int(T * skip_fraction))
    t_vals = np.arange(T - skip, dtype=np.float64)
    y_vals = log_dist[skip:].astype(np.float64)

    if len(t_vals) < 2:
        return 0.0

    # Linear regression: log(δ(t)) = λ·t + const
    coeffs = np.polyfit(t_vals, y_vals, deg=1)
    return float(coeffs[0])


def run_lyapunov_analysis(
    trajectories: np.ndarray,
    simulator,
    epsilon: float = _DEFAULT_EPSILON,
    skip_fraction: float = _FIT_SKIP_FRACTION,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹估计最大 Lyapunov 指数，汇总结果。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        simulator:      ``BrainDynamicsSimulator`` 实例（需支持 rollout）。
        epsilon:        初始扰动幅度（默认 1e-5）。
        skip_fraction:  跳过轨迹前段比例。
        output_dir:     保存 lyapunov_values.npy；None → 不保存。

    Returns:
        results: {
            "lyapunov_values": np.ndarray,   shape (n_init,)
            "mean_lyapunov":   float,
            "median_lyapunov": float,
            "std_lyapunov":    float,
            "fraction_positive": float,      fraction with λ > 0 (chaotic)
            "fraction_negative": float,      fraction with λ < 0 (convergent)
        }
    """
    n_traj = trajectories.shape[0]
    logger.info(
        "Lyapunov 指数估计: %d 条轨迹, 每条 %d 步, ε=%.2e",
        n_traj,
        trajectories.shape[1],
        epsilon,
    )

    values = np.zeros(n_traj, dtype=np.float32)
    log_interval = max(1, n_traj // 5)

    for i in range(n_traj):
        values[i] = estimate_lyapunov_exponent(
            trajectories[i],
            simulator=simulator,
            epsilon=epsilon,
            skip_fraction=skip_fraction,
        )
        if (i + 1) % log_interval == 0:
            logger.info("  %d/%d 轨迹完成", i + 1, n_traj)

    mean_lam = float(np.mean(values))
    median_lam = float(np.median(values))
    std_lam = float(np.std(values))
    frac_pos = float((values > 0).mean())
    frac_neg = float((values < 0).mean())

    logger.info(
        "  均值 λ=%.4f  中位数 λ=%.4f  std=%.4f  混沌比例=%.1f%%  收敛比例=%.1f%%",
        mean_lam,
        median_lam,
        std_lam,
        frac_pos * 100,
        frac_neg * 100,
    )

    if mean_lam < 0:
        logger.info("  → 系统整体收敛（λ < 0）")
    elif mean_lam < 0.01:
        logger.info("  → 系统边缘稳定（λ ≈ 0）")
    else:
        logger.info("  → 系统存在混沌成分（λ > 0）")

    results = {
        "lyapunov_values": values,
        "mean_lyapunov": mean_lam,
        "median_lyapunov": median_lam,
        "std_lyapunov": std_lam,
        "fraction_positive": frac_pos,
        "fraction_negative": frac_neg,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "lyapunov_values.npy", values)
        logger.info("  → 已保存: %s/lyapunov_values.npy", output_dir)

    return results
