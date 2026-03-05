"""
Response Plot
=============

可视化刺激响应矩阵和单次刺激实验结果：
  - 响应矩阵热图（N×N heatmap）
  - 单节点刺激响应时间曲线
  - 响应强度条形图（top-K 受影响脑区）
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib 未安装，响应可视化功能不可用。")


def plot_response_matrix(
    R: np.ndarray,
    title: str = "Stimulation Response Matrix R[i,j]",
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制响应矩阵热图。

    R[i,j] = response of node j when node i is stimulated.

    Args:
        R:         shape (n_nodes, n_regions)。
        title:     图标题。
        save_path: 保存路径；None → 显示到屏幕。
    """
    if not _MPL_AVAILABLE:
        return

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = float(np.abs(R).max()) or 1.0
    im = ax.imshow(
        R,
        aspect="auto",
        cmap="RdBu_r",
        origin="lower",
        vmin=-vmax,
        vmax=vmax,
    )
    plt.colorbar(im, ax=ax, label="Response (stim − baseline)")
    ax.set_xlabel("Target region j")
    ax.set_ylabel("Stimulated region i")
    ax.set_title(title)

    _save_or_show(fig, save_path)


def plot_stimulation_response(
    pre_traj: np.ndarray,
    stim_traj: np.ndarray,
    post_traj: np.ndarray,
    target_node: int,
    dt: float = 0.004,
    top_k: int = 5,
    title: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制单次刺激实验的三阶段时间曲线。

    Args:
        pre_traj:    shape (pre_steps, n_regions)。
        stim_traj:   shape (stim_steps, n_regions)。
        post_traj:   shape (post_steps, n_regions)。
        target_node: 被刺激的节点索引。
        dt:          时间步长（秒）。
        top_k:       额外绘制响应最大的 K 个节点。
        title:       图标题。
        save_path:   保存路径。
    """
    if not _MPL_AVAILABLE:
        return

    full_traj = np.concatenate([pre_traj, stim_traj, post_traj], axis=0)
    steps, n_regions = full_traj.shape
    times = np.arange(steps) * dt
    pre_end = len(pre_traj)
    stim_end = pre_end + len(stim_traj)

    # Find top-K responsive nodes (excluding target)
    mean_response = stim_traj.mean(axis=0) - pre_traj.mean(axis=0)
    other_nodes = [i for i in range(n_regions) if i != target_node]
    top_nodes = sorted(other_nodes, key=lambda j: abs(mean_response[j]), reverse=True)[:top_k]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Target node (red)
    ax.plot(
        times,
        full_traj[:, target_node],
        color="red",
        linewidth=2,
        label=f"Stimulated node {target_node}",
    )

    # Top responding nodes
    cmap = plt.get_cmap("tab10", top_k)
    for k, node in enumerate(top_nodes):
        ax.plot(
            times,
            full_traj[:, node],
            alpha=0.7,
            color=cmap(k),
            linewidth=1,
            label=f"Node {node} (Δ={mean_response[node]:.3f})",
        )

    # Phase boundaries
    ax.axvline(times[pre_end], color="gray", linestyle="--", alpha=0.7, label="Stim on")
    if stim_end < steps:
        ax.axvline(times[stim_end], color="black", linestyle="--", alpha=0.7, label="Stim off")

    ax.axvspan(times[pre_end], times[min(stim_end, steps - 1)], alpha=0.08, color="yellow")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Activity (normalised)")
    ax.set_title(title or f"Stimulation Response — Node {target_node}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_top_response_bar(
    response: np.ndarray,
    target_node: int,
    k: int = 20,
    title: str = "",
    save_path: Optional[Path] = None,
) -> None:
    """
    条形图：响应最强的 K 个脑区（按 |response| 排序）。

    Args:
        response:    shape (n_regions,)，每个脑区的响应量（Δactivity）。
        target_node: 被刺激节点（高亮显示）。
        k:           显示 top-K 节点。
        title:       图标题。
        save_path:   保存路径。
    """
    if not _MPL_AVAILABLE:
        return

    n = len(response)
    order = np.argsort(np.abs(response))[::-1][:k]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["red" if i == target_node else "steelblue" for i in order]
    ax.bar(range(len(order)), response[order], color=colors)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(i) for i in order], rotation=45, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Region index")
    ax.set_ylabel("Response (stim − baseline)")
    ax.set_title(title or f"Top-{k} Responding Regions (red = stimulated)")
    ax.grid(True, alpha=0.3, axis="y")

    _save_or_show(fig, save_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_or_show(fig: "plt.Figure", save_path: Optional[Path]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("  → 图表已保存: %s", save_path)
    else:
        plt.show()
    plt.close(fig)
