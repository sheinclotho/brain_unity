"""
Trajectory Plot
===============

可视化自由动力学轨迹：
  - 多条轨迹的状态范数随时间变化
  - 轨迹汇聚 / 发散示意图（降维到 PCA 前两维）
  - 逐脑区活动热图
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib 未安装，轨迹可视化功能不可用。")


def plot_trajectory_norms(
    trajectories: np.ndarray,
    times: Optional[np.ndarray] = None,
    max_show: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制多条轨迹的 L2 范数随时间变化。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        times:        shape (steps,)，时间轴（秒）；None → 用步数。
        max_show:     最多显示的轨迹数。
        save_path:    保存路径；None → 显示到屏幕。
    """
    if not _MPL_AVAILABLE:
        return

    n_traj, steps, _ = trajectories.shape
    norms = np.linalg.norm(trajectories, axis=2)  # (n_init, steps)

    if times is None:
        times = np.arange(steps, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))

    indices = np.linspace(0, n_traj - 1, min(max_show, n_traj), dtype=int)
    cmap = plt.get_cmap("viridis", len(indices))

    for k, idx in enumerate(indices):
        ax.plot(times, norms[idx], alpha=0.6, color=cmap(k), linewidth=0.8)

    ax.set_xlabel("Time (s)" if times[1] < 1.0 else "Step")
    ax.set_ylabel("State L₂ norm")
    ax.set_title(f"Free Dynamics — Trajectory Norms ({len(indices)} of {n_traj})")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_pca_trajectories(
    trajectories: np.ndarray,
    max_show: int = 50,
    save_path: Optional[Path] = None,
) -> None:
    """
    用 PCA 将高维轨迹投影到 2D，并绘制轨迹路径。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        max_show:     最多显示轨迹数。
        save_path:    保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:  # pragma: no cover
        logger.warning("scikit-learn 未安装，跳过 PCA 轨迹图。")
        return

    n_traj, steps, n_regions = trajectories.shape
    all_states = trajectories.reshape(-1, n_regions)  # (n_init*steps, n_regions)

    pca = PCA(n_components=2)
    pca.fit(all_states)
    var_ratio = pca.explained_variance_ratio_

    indices = np.linspace(0, n_traj - 1, min(max_show, n_traj), dtype=int)
    cmap = plt.get_cmap("plasma", len(indices))

    fig, ax = plt.subplots(figsize=(8, 7))

    for k, idx in enumerate(indices):
        traj_2d = pca.transform(trajectories[idx])  # (steps, 2)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.5, color=cmap(k), linewidth=0.7)
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=cmap(k), marker="o", s=10, zorder=5)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=cmap(k), marker="*", s=20, zorder=5)

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    ax.set_title("Free Dynamics in PCA Space (○ start, ★ end)")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_region_heatmap(
    trajectory: np.ndarray,
    times: Optional[np.ndarray] = None,
    title: str = "Region Activity Heatmap",
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制单条轨迹的逐脑区活动热图（时间 × 脑区）。

    Args:
        trajectory: shape (steps, n_regions)。
        times:      shape (steps,)，时间轴。
        title:      图标题。
        save_path:  保存路径。
    """
    if not _MPL_AVAILABLE:
        return

    steps, n_regions = trajectory.shape
    if times is None:
        times = np.arange(steps, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        trajectory.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[times[0], times[-1], 0, n_regions],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="Activity (normalised)")
    ax.set_xlabel("Time (s)" if times[1] < 1.0 else "Step")
    ax.set_ylabel("Region index")
    ax.set_title(title)

    _save_or_show(fig, save_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_or_show(fig: "plt.Figure", save_path: Optional[Path]) -> None:
    """Save figure to file or display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("  → 图表已保存: %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_trajectory_convergence(
    mean_distances: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制轨迹对均值距离随时间的变化（收敛分析图）。

    Args:
        mean_distances: shape (steps,)，均值 L2 距离。
        save_path:      保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    steps = len(mean_distances)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(steps), mean_distances, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title("Trajectory Convergence — Mean Pairwise Distance over Time\n"
                 "(↓ converging, → no structure, ↑ diverging)")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_lyapunov_histogram(
    lyapunov_values: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制 Lyapunov 指数分布直方图。

    Args:
        lyapunov_values: shape (n_init,)，每条轨迹的 Lyapunov 指数。
        save_path:       保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    mean_lam = float(np.mean(lyapunov_values))
    median_lam = float(np.median(lyapunov_values))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lyapunov_values, bins=min(30, max(10, len(lyapunov_values) // 3)),
            color="salmon", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="λ=0")
    ax.axvline(mean_lam, color="red", linestyle="-", linewidth=1.2,
               label=f"mean={mean_lam:.4f}")
    ax.axvline(median_lam, color="orange", linestyle=":", linewidth=1.2,
               label=f"median={median_lam:.4f}")
    ax.set_xlabel("Lyapunov Exponent λ (per step)")
    ax.set_ylabel("Count")
    ax.set_title("Lyapunov Exponent Distribution\n"
                 "(λ<0: convergent, λ≈0: marginal, λ>0: chaotic)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_basin_sizes(
    basin_distribution: dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制吸引子 basin 大小分布条形图。

    Args:
        basin_distribution: {attractor_id: fraction} 字典（int 键或 str 键均可）。
        save_path:          保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    labels_abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ids = sorted(basin_distribution.keys())
    fractions = [basin_distribution[k] for k in ids]
    bar_labels = [
        (f"Attractor {labels_abc[k]}" if isinstance(k, int) and k < len(labels_abc)
         else str(k))
        for k in ids
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(ids) * 1.5), 4))
    cmap = plt.get_cmap("tab10", len(ids))
    bars = ax.bar(bar_labels, fractions, color=[cmap(i) for i in range(len(ids))],
                  edgecolor="white")
    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{frac:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, min(1.15, max(fractions) + 0.15))
    ax.set_ylabel("Basin Size (fraction of trajectories)")
    ax.set_title("Attractor Basin Size Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    _save_or_show(fig, save_path)
