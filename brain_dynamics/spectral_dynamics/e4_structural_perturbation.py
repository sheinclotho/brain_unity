"""
E4: Structural Perturbation Experiments
========================================

验证低维动力学是否依赖真实网络结构（假设 H1+H2 的验证实验）。

核心思路
--------
若"低维动力学是结构必然"，则对连接矩阵进行破坏性扰动后，
低维性质（谱有效维度 PR、主导模态数量、谱半径）应发生显著变化。

三种扰动方式
------------
1. **边权重随机化（weight_shuffle）**
   保留网络拓扑（edge_index 不变），随机打乱所有边的权重。

2. **保度随机重连（degree_preserving_rewire）**
   随机交换边对（Configuration Model 方法），保留节点度序列。

3. **低秩截断（low_rank_truncation）**
   保留 SVD 前 k 个奇异值，截断剩余 N-k 个模态。

指标计算
--------
对每种扰动后的矩阵，重新计算纯谱指标：
  - 谱有效维度（参与率 PR）
  - 主导特征值数量（n_dominant）
  - 谱半径（spectral_radius）
  - 谱间隔比（spectral_gap_ratio）

LLE 估计需要运行模型（由 twinbrain-dynamics 管线负责）。
本模块只做纯矩阵谱分析，不模拟任何动力学。

输出文件
--------
  perturbation_summary.json     — 所有扰动的指标对比
  perturbation_comparison.png   — PR / n_dominant / spectral_radius 对比柱状图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .e1_spectral_analysis import compute_spectral_metrics

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Perturbation generators
# ─────────────────────────────────────────────────────────────────────────────

def weight_shuffle(W: np.ndarray, seed: int = 0) -> np.ndarray:
    """
    保拓扑权重随机化：随机打乱所有非零元素的权重。

    仅对非零元素操作，保留零元素的位置（稀疏拓扑）。
    对稠密矩阵，等价于权重值随机排列。

    Args:
        W:    连接矩阵 (N, N)。
        seed: 随机种子。

    Returns:
        W_shuffled: 同形状，非零元素权重随机打乱。
    """
    rng = np.random.default_rng(seed)
    W_out = W.copy()
    nonzero_mask = W != 0
    vals = W[nonzero_mask]
    rng.shuffle(vals)
    W_out[nonzero_mask] = vals
    return W_out.astype(np.float32)


def degree_preserving_rewire(
    W: np.ndarray,
    n_swaps_factor: int = 10,
    seed: int = 0,
) -> np.ndarray:
    """
    保度随机重连：随机交换边对，保留节点度（连接权重之和）序列。

    算法（Maslov & Sneppen 2002 Configuration Model）：
      重复 n_swaps 次：
        1. 随机选两条边 (i→j) 和 (k→l)
        2. 交换目标：变为 (i→l) 和 (k→j)
        3. 若结果产生自环或重边则跳过

    注意：对有权矩阵，本函数交换**权重值**（而非仅重连边），
    因此严格保留了所有边的权重幅值集合和度序列。

    Args:
        W:               连接矩阵 (N, N)。
        n_swaps_factor:  实际交换次数 = n_swaps_factor × nnz
        seed:            随机种子。

    Returns:
        W_rewired: 同形状，边随机重连。
    """
    rng = np.random.default_rng(seed)
    W_out = W.copy().astype(np.float32)
    rows, cols = np.where(W_out != 0)
    n_edges = len(rows)
    if n_edges < 4:
        logger.warning("边数过少 (%d)，无法有效重连。", n_edges)
        return W_out

    n_swaps = n_swaps_factor * n_edges
    for _ in range(n_swaps):
        # Pick two distinct edges
        idx = rng.integers(0, n_edges, size=2)
        if idx[0] == idx[1]:
            continue
        i, j = int(rows[idx[0]]), int(cols[idx[0]])
        k, l = int(rows[idx[1]]), int(cols[idx[1]])
        # Avoid self-loops and duplicate edges after swap
        if i == l or k == j:
            continue
        if W_out[i, l] != 0 or W_out[k, j] != 0:
            continue
        # Perform swap
        wij = W_out[i, j]
        wkl = W_out[k, l]
        W_out[i, j] = 0.0
        W_out[k, l] = 0.0
        W_out[i, l] = wij
        W_out[k, j] = wkl
        # Update edge index tracker
        rows[idx[0]], cols[idx[0]] = i, l
        rows[idx[1]], cols[idx[1]] = k, j

    return W_out


def low_rank_truncation(W: np.ndarray, k: int) -> np.ndarray:
    """
    低秩截断：保留前 k 个 SVD 模态，截断其余。

    W ≈ U_k Σ_k V_k^T

    Args:
        W: 连接矩阵 (N, N)。
        k: 保留的奇异值数量（rank of approximation）。

    Returns:
        W_lowrank: 同形状，秩为 k 的近似矩阵。
    """
    k = min(k, min(W.shape))
    U, sv, Vt = np.linalg.svd(W.astype(np.float64), full_matrices=False)
    W_lr = (U[:, :k] * sv[:k]) @ Vt[:k, :]
    return W_lr.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_structural_perturbation(
    W: np.ndarray,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    k_values: Optional[List[int]] = None,
    seed: int = 42,
) -> Dict:
    """
    运行 E4 结构扰动实验：对三类扰动比较谱指标。

    本函数只做纯矩阵谱分析（PR、谱半径、n_dominant、谱间隔比），
    不模拟任何动力学。LLE 估计由 twinbrain-dynamics 管线负责。

    Args:
        W:          原始连接矩阵 (N, N)。
        output_dir: 结果保存目录。
        label:      矩阵标签（用于文件名）。
        k_values:   低秩截断的 k 列表；默认 [1, 2, 5, 10, N//10]。
        seed:       随机种子。

    Returns:
        summary dict（可序列化为 JSON）。
    """
    N = W.shape[0]
    if k_values is None:
        k_values = sorted({1, 2, 5, 10, max(1, N // 10)})

    def _metrics(mat: np.ndarray, tag: str) -> Dict:
        spec = compute_spectral_metrics(mat, symmetric=False)
        return {
            "tag": tag,
            "spectral_radius": spec["spectral_radius"],
            "participation_ratio": spec["participation_ratio"],
            "n_dominant": spec["n_dominant"],
            "spectral_gap_ratio": spec["spectral_gap_ratio"],
        }

    results: Dict = {}

    logger.info("E4: 计算原始矩阵指标 (%s)...", label)
    results["original"] = _metrics(W, "original")

    logger.info("E4: weight_shuffle 扰动...")
    results["weight_shuffle"] = _metrics(weight_shuffle(W, seed=seed), "weight_shuffle")

    logger.info("E4: degree_preserving_rewire 扰动...")
    results["degree_preserving_rewire"] = _metrics(
        degree_preserving_rewire(W, seed=seed), "degree_preserving_rewire"
    )

    logger.info("E4: low_rank_truncation 扰动 (k=%s)...", k_values)
    results["low_rank_truncation"] = {}
    for k in k_values:
        if k < N:
            results["low_rank_truncation"][f"k={k}"] = _metrics(
                low_rank_truncation(W, k), f"low_rank_k{k}"
            )

    # Delta from original
    orig_pr = results["original"]["participation_ratio"]

    def _delta_summary(r: Dict) -> Dict:
        return {**r, "delta_pr": round(r["participation_ratio"] - orig_pr, 3)}

    results["weight_shuffle"] = _delta_summary(results["weight_shuffle"])
    results["degree_preserving_rewire"] = _delta_summary(
        results["degree_preserving_rewire"]
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"perturbation_summary_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("保存扰动摘要: %s", json_path)
        _try_plot_perturbation(results, out / f"perturbation_comparison_{label}.png", label)

    return results


def _try_plot_perturbation(results: Dict, output_path: Path, label: str) -> None:
    """Bar chart comparing perturbation conditions: PR / n_dominant / spectral_radius."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    conditions = ["original", "weight_shuffle", "degree_preserving_rewire"]
    existing = [c for c in conditions if c in results]

    metrics_to_plot = ["participation_ratio", "n_dominant", "spectral_radius"]
    metric_labels = ["Effective dim. PR", "Dominant eigenvalues n_dom", "Spectral radius ρ(W)"]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(13, 4))
    x = np.arange(len(existing))
    width = 0.6
    colors = ["steelblue", "darkorange", "forestgreen"][:len(existing)]

    for ax, metric, ml in zip(axes, metrics_to_plot, metric_labels):
        vals = [results[c].get(metric, float("nan")) for c in existing]
        bars = ax.bar(x, vals, width=width, color=colors, alpha=0.8, edgecolor="k", lw=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(existing, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(ml)
        ax.set_title(ml)
        for bar, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle(f"Structural Perturbation Experiments  [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)
