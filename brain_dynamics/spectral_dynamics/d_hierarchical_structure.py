"""
Module D: Hierarchical Structure Detection
============================================

检测连接矩阵中的"模块中的模块"（modules of modules）层级结构。

科学意义
--------
大脑连接组具有多尺度层级结构（Meunier et al. 2009 *PLoS Comput Biol*；
Bassett et al. 2010 *PNAS*）：

  全脑
  → 叶（额叶/顶叶/颞叶/枕叶）
    → 功能网络（DMN/DAN/SMN/VIS）
      → 局部皮层区
        → 皮层柱（更细尺度）

这种层级结构在连接矩阵中表现为**嵌套的 block 模式**，
可以通过层级聚类（Dendrogram）直接可视化。

**层级结构与低维动力学的联系**：
  - 层级组织 → 连接矩阵有多个特征值尺度 → 谱分布分层
  - 不同尺度的模块对应不同频率的振荡模态
  - 这解释了为什么 PR 值往往比纯随机网络低很多

D1 层级聚类（Dendrogram）
--------------------------
  - 方法：Ward linkage（最小方差，对群簇形状假设最少）
  - 距离：1 - |W_sym_norm| （归一化绝对权重的互补值）
  - 输出：dendrogram.png

D2 层级统计
-----------
  - 在 5 个截断高度下统计簇数量和簇规模
  - 计算"层级指数"：不同尺度的簇数量的对数斜率

批判性注意事项
--------------
1. **Ward linkage 假设球形簇**：对高度异质（非球形）的神经模块
   可能产生错误合并。Single/complete linkage 保留异质性但树形图难以解读。
   建议对比 Ward vs Average linkage 结果一致性。
2. **非对称矩阵的处理**：Ward linkage 需要对称距离矩阵。
   对 EC 矩阵，我们对称化后聚类（丢失方向信息）。
   有向层级结构需要特殊算法（Caggiano et al. 2020 超出本模块范围）。
3. **Dendrogram 的截断高度选择**：没有"正确"的截断点。
   本模块提供 5 个固定百分位截断（0.2, 0.4, 0.6, 0.8, 1.0 × max_height），
   并绘制完整树形图供人工判断。

输出文件
--------
  dendrogram_{label}.png         — 完整层级聚类树形图
  hierarchical_stats_{label}.json — 层级统计数据
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

_TRUNCATE_LEVELS = 5    # Number of levels to show in truncated dendrogram
_HEIGHT_FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]


# ─────────────────────────────────────────────────────────────────────────────
# Distance matrix computation
# ─────────────────────────────────────────────────────────────────────────────

def _weight_to_distance(W: np.ndarray) -> np.ndarray:
    """
    将连接权重矩阵转换为距离矩阵用于层级聚类。

    策略：
    1. 对称化：W_sym = (|W| + |W|ᵀ) / 2
    2. 归一化到 [0, 1]
    3. 距离 = 1 - W_norm（强连接 → 小距离 → 早合并）

    对角线设为 0（零距离，与 squareform 兼容）。

    Returns:
        D: shape (N, N)，对称，非负，对角为 0。
    """
    W_abs = np.abs(W).astype(np.float64)
    W_sym = (W_abs + W_abs.T) / 2.0
    np.fill_diagonal(W_sym, 0.0)
    max_w = W_sym.max()
    if max_w < 1e-30:
        logger.warning("连接矩阵权重全部为零，距离矩阵将全为 1。")
        return np.ones_like(W_sym)
    W_norm = W_sym / max_w
    D = 1.0 - W_norm
    np.fill_diagonal(D, 0.0)
    return D.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Hierarchical clustering
# ─────────────────────────────────────────────────────────────────────────────

def compute_linkage(
    W: np.ndarray,
    method: str = "ward",
) -> np.ndarray:
    """
    计算层级聚类的 linkage 矩阵。

    Args:
        W:      连接矩阵 (N, N)。
        method: "ward" / "average" / "complete" / "single"。

    Returns:
        Z: linkage 矩阵，shape (N-1, 4)，scipy 格式。
    """
    D = _weight_to_distance(W)
    # squareform expects condensed distance vector
    condensed = squareform(D.astype(np.float64), checks=False)
    Z = hierarchy.linkage(condensed, method=method, optimal_ordering=True)
    return Z


def compute_cluster_stats_at_levels(
    Z: np.ndarray,
    N: int,
    height_fractions: Optional[List[float]] = None,
) -> List[Dict]:
    """
    在不同截断高度处统计簇数量和簇规模分布（D2）。

    Args:
        Z:                 linkage 矩阵 (N-1, 4)。
        N:                 样本数量。
        height_fractions:  截断高度分位列表（0-1）；默认 [0.2, 0.4, 0.6, 0.8, 1.0]。

    Returns:
        list of dict，每个截断高度对应一个记录：
          {height, n_clusters, mean_size, max_size, min_size, sizes}
    """
    if height_fractions is None:
        height_fractions = _HEIGHT_FRACTIONS

    max_height = float(Z[:, 2].max())
    results = []
    for frac in height_fractions:
        h = frac * max_height
        labels = hierarchy.fcluster(Z, t=h, criterion="distance")
        unique, counts = np.unique(labels, return_counts=True)
        results.append({
            "height_fraction": frac,
            "height_value": round(h, 4),
            "n_clusters": int(len(unique)),
            "mean_cluster_size": round(float(counts.mean()), 1),
            "max_cluster_size": int(counts.max()),
            "min_cluster_size": int(counts.min()),
            "cluster_sizes": counts.tolist(),
        })
    return results


def compute_hierarchy_index(cluster_stats: List[Dict]) -> float:
    """
    层级指数：不同截断高度下簇数量的对数斜率。

    若斜率很大（绝对值）→ 层级结构明显（少量截断高度变化就改变簇数大量）
    若斜率接近 0 → 层级结构平坦（截断高度对簇数影响小）

    Returns:
        slope: 对数-线性空间中 log(n_clusters) vs height_fraction 的斜率。
               负值（高截断高度时簇数多，低高度时少，实际上符合直觉：
               低截断 = 少簇，高截断 = 多簇……）
               实际意义：|slope| 越大，层级越陡峭。
    """
    heights = np.array([s["height_fraction"] for s in cluster_stats])
    n_clusters = np.array([s["n_clusters"] for s in cluster_stats], dtype=float)
    n_clusters = np.maximum(n_clusters, 1)  # avoid log(0)

    if len(heights) < 2:
        return 0.0
    # Linear fit of log(n_clusters) vs height
    slope, _ = np.polyfit(heights, np.log(n_clusters), 1)
    return float(slope)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_dendrogram(
    Z: np.ndarray,
    N: int,
    output_path: Path,
    label: str,
    method: str,
) -> None:
    """绘制完整树形图 + 截断版（D1）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster import hierarchy as sch
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Full dendrogram (truncated to last 50 merges if N > 100)
    p_val = min(N, 50)
    ax = axes[0]
    sch.dendrogram(Z, ax=ax, truncate_mode="lastp", p=p_val,
                   leaf_rotation=90, leaf_font_size=7,
                   color_threshold=None)
    ax.set_title(f"层级聚类树形图（显示最后 {p_val} 次合并）\n方法={method}  [{label}]")
    ax.set_xlabel("节点/簇")
    ax.set_ylabel("距离（1 - 归一化|W|）")

    # Show cluster counts at several heights
    max_h = float(Z[:, 2].max())
    for frac, color in zip([0.2, 0.4, 0.6, 0.8], ["blue", "green", "orange", "red"]):
        h = frac * max_h
        n_c = len(np.unique(hierarchy.fcluster(Z, t=h, criterion="distance")))
        ax.axhline(h, ls="--", color=color, lw=0.8, alpha=0.7,
                   label=f"h={h:.2f} → {n_c} 簇")
    ax.legend(fontsize=7)

    # n_clusters vs height curve
    h_values = np.linspace(0.01 * max_h, max_h, 50)
    n_clusters_curve = [
        len(np.unique(hierarchy.fcluster(Z, t=h, criterion="distance")))
        for h in h_values
    ]
    ax2 = axes[1]
    ax2.plot(h_values / max_h, n_clusters_curve, "o-", ms=3, lw=1.5)
    ax2.set_xlabel("截断高度（归一化）")
    ax2.set_ylabel("簇数量")
    ax2.set_title(f"簇数量 vs 截断高度  [{label}]")
    ax2.set_yscale("log")
    # Mark "elbow" regions
    for frac, color in zip([0.2, 0.4, 0.6, 0.8], ["blue", "green", "orange", "red"]):
        ax2.axvline(frac, ls="--", color=color, lw=0.8, alpha=0.6)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_hierarchical_structure(
    W: np.ndarray,
    method: str = "ward",
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    运行模块 D：层级结构检测。

    Args:
        W:          连接矩阵 (N, N)。
        method:     层级聚类方法，"ward"（推荐）或 "average" / "complete"。
        output_dir: 输出目录。
        label:      文件名标签。

    Returns:
        dict 包含:
          linkage_method:      聚类方法
          max_linkage_height:  最大合并高度
          cluster_stats:       不同截断高度下的簇统计
          hierarchy_index:     层级陡峭度指数
          is_hierarchical:     bool，是否存在明显层级结构
    """
    N = W.shape[0]
    logger.info("D 层级结构检测: N=%d, method=%s", N, method)

    Z = compute_linkage(W, method=method)
    cluster_stats = compute_cluster_stats_at_levels(Z, N)
    h_index = compute_hierarchy_index(cluster_stats)

    # Simple heuristic for "is hierarchical"
    # If cluster count roughly quadruples between 20%-80% height fraction,
    # the hierarchy is meaningful
    n_20 = cluster_stats[0]["n_clusters"]
    n_80 = cluster_stats[3]["n_clusters"]
    is_hierarchical = bool(n_80 > 2 * n_20)

    result: Dict = {
        "linkage_method": method,
        "n_nodes": N,
        "max_linkage_height": round(float(Z[:, 2].max()), 5),
        "cluster_stats": cluster_stats,
        "hierarchy_index": round(h_index, 4),
        "is_hierarchical": is_hierarchical,
        "n_clusters_at_20pct_height": n_20,
        "n_clusters_at_80pct_height": n_80,
    }

    logger.info(
        "D 结果: hierarchy_index=%.3f, n_clusters@20%%h=%d, @80%%h=%d, is_hierarchical=%s",
        h_index, n_20, n_80, is_hierarchical,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save JSON (without cluster_stats sizes lists for readability)
        json_result = {k: v for k, v in result.items() if k != "cluster_stats"}
        json_result["cluster_stats_summary"] = [
            {kk: vv for kk, vv in s.items() if kk != "cluster_sizes"}
            for s in cluster_stats
        ]
        json_path = out / f"hierarchical_stats_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        # Save linkage matrix
        np.save(out / f"linkage_matrix_{label}.npy", Z)

        _try_plot_dendrogram(
            Z, N,
            out / f"dendrogram_{label}.png",
            label, method,
        )

    return result
