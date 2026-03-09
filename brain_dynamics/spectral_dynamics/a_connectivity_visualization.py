"""
Module A: Connectivity Matrix Visualization
=============================================

直接可视化连接矩阵，检测是否存在 block / hierarchical 结构。
这是整个分析管线的第一步——眼睛看到的模式常常比数字更有说服力。

A1 原始连接矩阵热图
--------------------
  - 绘制原始 W（含正负值）
  - 绘制 |W| 的对数尺度版本（突出弱连接）
  - 输出：connectivity_matrix_raw.png

A2 社区排序后的矩阵热图
------------------------
  - 按社区归属对节点重排序
  - 重绘矩阵：若存在模块结构，应出现沿对角线的 block 模式
  - 输出：connectivity_matrix_reordered.png

批判性说明
----------
1. 有效连接矩阵（响应矩阵 R）含负值（抑制性连接）。可视化时须使用
   diverging colormap（如 RdBu_r），而非仅显示绝对值。
2. "block pattern" 只是一个定性判断；Module C 的 modularity Q 提供定量依据。
3. 若矩阵维度 N >> 200，热图像素会模糊；建议同时绘制低秩近似版本。
4. 节点排序完全取决于社区检测算法——不同算法给出的"block"外观不同。
   可视化结果不能脱离社区检测结果单独解读。

输出文件
--------
  connectivity_matrix_raw.png            — 原始矩阵热图
  connectivity_matrix_log_abs.png        — log(|W|) 热图
  connectivity_matrix_reordered.png      — 按社区排序后的热图
  connectivity_matrix_sorted.png         — 兼容名（同 reordered）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# A1: Raw connectivity matrix heatmap
# ─────────────────────────────────────────────────────────────────────────────

def plot_connectivity_raw(
    W: np.ndarray,
    output_dir: Path,
    label: str = "matrix",
) -> None:
    """
    绘制原始连接矩阵热图（A1）。

    生成两张图：
    1. 原始值（diverging colormap，中心为零）
    2. log₁₀(|W| + ε)（揭示弱连接分布）

    Args:
        W:          连接矩阵 (N, N)。
        output_dir: 输出目录。
        label:      文件名标签。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        logger.warning("matplotlib not installed; skipping A1 plot.")
        return

    N = W.shape[0]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Plot 1: Raw values with diverging colormap
    vabs = float(np.abs(W).max())
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vabs, vmax=vabs, aspect="auto",
                   interpolation="none" if N <= 300 else "bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Connection Weight")
    ax.set_title(f"Connectivity Matrix (raw)  N={N}  [{label}]")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")
    fig.tight_layout()
    path1 = out / f"connectivity_matrix_raw_{label}.png"
    fig.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", path1)

    # Plot 2: log|W|
    log_abs = np.log10(np.abs(W) + 1e-10)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(log_abs, cmap="viridis", aspect="auto",
                   interpolation="none" if N <= 300 else "bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="log10(|W|)")
    ax.set_title(f"Connectivity Matrix (log|W|)  N={N}  [{label}]")
    ax.set_xlabel("Target Node")
    ax.set_ylabel("Source Node")
    fig.tight_layout()
    path2 = out / f"connectivity_matrix_log_abs_{label}.png"
    fig.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", path2)


# ─────────────────────────────────────────────────────────────────────────────
# A2: Community-reordered connectivity matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_connectivity_reordered(
    W: np.ndarray,
    community_labels: np.ndarray,
    output_dir: Path,
    label: str = "matrix",
) -> None:
    """
    按社区归属重排节点，重绘连接矩阵热图（A2）。

    视觉效果：若有模块结构，对角线上会出现密集的 block 模式。

    Args:
        W:                 连接矩阵 (N, N)。
        community_labels:  shape (N,)，每个节点所属的社区编号（0-indexed）。
        output_dir:        输出目录。
        label:             文件名标签。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    N = W.shape[0]
    # Sort nodes by community, then by node index within community
    idx_sorted = np.argsort(community_labels, kind="stable")
    W_reordered = W[np.ix_(idx_sorted, idx_sorted)]
    labels_sorted = community_labels[idx_sorted]

    # Find community boundaries
    boundaries = []
    current = labels_sorted[0]
    start = 0
    for i in range(1, N):
        if labels_sorted[i] != current:
            boundaries.append((start, i))
            start = i
            current = labels_sorted[i]
    boundaries.append((start, N))

    vabs = float(np.abs(W_reordered).max())
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(W_reordered, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                   aspect="auto", interpolation="none" if N <= 300 else "bilinear")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Connection Weight")

    # Draw community boundary boxes
    n_communities = len(boundaries)
    cmap_c = plt.cm.tab10(np.linspace(0, 0.9, min(n_communities, 10)))
    for ci, (s, e) in enumerate(boundaries):
        size = e - s
        color = cmap_c[ci % len(cmap_c)]
        rect = Rectangle((s - 0.5, s - 0.5), size, size,
                          linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.8)
        ax.add_patch(rect)

    ax.set_title(
        f"Connectivity Matrix ({n_communities} communities)  N={N}  [{label}]\n"
        f"colored boxes = community boundaries"
    )
    ax.set_xlabel("Node (community order)")
    ax.set_ylabel("Node (community order)")
    fig.tight_layout()
    path = Path(output_dir) / f"connectivity_matrix_reordered_{label}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", path)

    # Also save a combined figure (raw vs reordered)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_r, mat, title in [
        (axes[0], W, "Original order"),
        (axes[1], W_reordered, f"Community order ({n_communities} communities)"),
    ]:
        im_r = ax_r.imshow(mat, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
                           aspect="auto",
                           interpolation="none" if N <= 300 else "bilinear")
        ax_r.set_title(title)
        ax_r.set_xlabel("Node")
        ax_r.set_ylabel("Node")
        plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

    # Add community boxes to reordered panel
    for ci, (s, e) in enumerate(boundaries):
        size = e - s
        color = cmap_c[ci % len(cmap_c)]
        rect = Rectangle((s - 0.5, s - 0.5), size, size,
                          linewidth=1.5, edgecolor=color, facecolor="none", alpha=0.8)
        axes[1].add_patch(rect)

    fig.suptitle(f"Connectivity Visualization  [{label}]", fontsize=12)
    fig.tight_layout()
    combined_path = Path(output_dir) / f"connectivity_matrix_sorted_{label}.png"
    fig.savefig(combined_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", combined_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_connectivity_visualization(
    W: np.ndarray,
    community_labels: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> dict:
    """
    运行模块 A：连接矩阵可视化。

    Args:
        W:                 连接矩阵 (N, N)。
        community_labels:  若提供，生成按社区排序的热图（A2）；否则仅生成 A1。
        output_dir:        输出目录。
        label:             文件名标签。

    Returns:
        summary dict（矩阵统计信息）。
    """
    W = np.asarray(W, dtype=np.float32)
    N = W.shape[0]

    result = {
        "n_regions": N,
        "weight_max": float(np.abs(W).max()),
        "weight_mean_nonzero": float(np.abs(W[W != 0]).mean()) if (W != 0).any() else 0.0,
        "density": float((W != 0).sum()) / (N * N),
        "has_negative_weights": bool((W < 0).any()),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        plot_connectivity_raw(W, out, label=label)
        if community_labels is not None:
            plot_connectivity_reordered(W, np.asarray(community_labels), out, label=label)

    return result
