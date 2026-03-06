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

    关闭 matplotlib 默认的双线性插值（interpolation='none'），
    确保每个像素对应矩阵中的单个条目，避免插值模糊产生虚假竖条/横条。

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
        interpolation="none",   # disable bilinear interp; every pixel = one matrix entry
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


def plot_response_column_stats(
    R: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制响应矩阵的列统计图，用于区分 hub 节点和过度扩散。

    包含三个子图（基于 GPT 建议，经科学验证）：

    **子图 1 — 列均值（column mean）**
    ``column_mean[j] = mean(|R[:,j]|)``
    检测哪些节点 j 对任何刺激都容易响应（潜在 hub）。
    若少数节点的列均值显著高于其他节点 → heavy-tail 分布 → hub 节点，
    竖条是真实网络结构，不是模型问题。

    **子图 2 — 刺激特异性（stimulus specificity）**
    ``S(i) = std_j(R[i,j])``
    检测刺激节点 i 的响应是否具有选择性（大 std）或均匀扩散（小 std）。
    若 S(i) 接近 0 → 所有节点响应差不多 → 全局扩散，模型可能过度耦合。
    若 S(i) 较大 → 刺激具有空间选择性 → 模型行为正常。

    **子图 3 — 列均值分布直方图**
    显示列均值的分布形状，heavy-tail（右偏）提示 hub 存在。

    Args:
        R:         shape (n_nodes, n_regions) 响应矩阵。
        save_path: 保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    n_nodes, n_regions = R.shape
    col_mean = np.abs(R).mean(axis=0)          # (n_regions,)
    stim_specificity = R.std(axis=1)           # (n_nodes,)  S(i) = std_j(R[i,j])

    # Hub score: how many std-devs above the mean is each column
    col_mean_z = (col_mean - col_mean.mean()) / (col_mean.std() + 1e-12)
    top_hub_k = min(10, n_regions)
    top_hubs = np.argsort(col_mean)[::-1][:top_hub_k]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # ── Subplot 1: column mean bar chart ──────────────────────────────────────
    ax = axes[0]
    colors = ["tomato" if i in top_hubs else "steelblue" for i in range(n_regions)]
    ax.bar(np.arange(n_regions), col_mean, color=colors, width=1.0, edgecolor="none")
    ax.axhline(col_mean.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={col_mean.mean():.4f}")
    ax.axhline(col_mean.mean() + 2 * col_mean.std(), color="red", linewidth=0.8,
               linestyle=":", label="+2σ")
    ax.set_xlabel("Target region j")
    ax.set_ylabel("mean|R[:,j]|")
    ax.set_title(f"Column Mean (Hub Detection)\nred = top-{top_hub_k} hubs")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Subplot 2: stimulus specificity per stimulated node ───────────────────
    ax = axes[1]
    low_spec = stim_specificity < stim_specificity.mean() * 0.5
    spec_colors = ["orange" if low_spec[i] else "steelblue" for i in range(n_nodes)]
    ax.bar(np.arange(n_nodes), stim_specificity, color=spec_colors, width=1.0, edgecolor="none")
    ax.axhline(stim_specificity.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={stim_specificity.mean():.4f}")
    ax.set_xlabel("Stimulated region i")
    ax.set_ylabel("S(i) = std_j(R[i,j])")
    ax.set_title("Stimulus Specificity\n(orange = low specificity / diffuse)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Subplot 3: distribution of column means ───────────────────────────────
    ax = axes[2]
    ax.hist(col_mean, bins=max(2, min(40, n_regions // 3 + 1)),
            color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(col_mean.mean(), color="black", linewidth=1.0, linestyle="--",
               label=f"mean={col_mean.mean():.4f}")
    # Mark 2σ threshold
    ax.axvline(col_mean.mean() + 2 * col_mean.std(), color="red", linewidth=0.8,
               linestyle=":", label="+2σ (hub threshold)")
    ax.set_xlabel("mean|R[:,j]|")
    ax.set_ylabel("Count")
    ax.set_title("Column Mean Distribution\n(right-skew → hub nodes present)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Summary annotation
    n_low_spec = int(low_spec.sum())
    frac_low = n_low_spec / n_nodes if n_nodes > 0 else 0
    n_hubs_2sigma = int((col_mean_z > 2).sum())
    fig.suptitle(
        f"Response Matrix Structure  |  "
        f"Hubs (>2σ): {n_hubs_2sigma}  |  "
        f"Low-specificity stim nodes: {n_low_spec}/{n_nodes} ({frac_low:.0%})",
        fontsize=10, y=1.02,
    )
    plt.tight_layout()
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
