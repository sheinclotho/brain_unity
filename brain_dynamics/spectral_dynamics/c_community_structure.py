"""
Module C: Community Structure Analysis
========================================

量化连接矩阵的模块结构，判断是否存在统计显著的社区组织。

科学意义
--------
大脑的层级模块结构（Meunier et al. 2010 *Front Neuroinform*；
Bullmore & Sporns 2012 *Nat Rev Neurosci*）是神经网络的重要特性：

- 模块内高度连接 → 高效局部信息处理
- 模块间稀疏连接 → 低代谢成本跨区域通信
- **谱低秩性往往是模块结构的谱域表现**：k 个模块
  → 连接矩阵 Laplacian 有 k 个接近零的特征值
  → 谱有效维度 PR ≈ k（模块数量）

实现算法（优先级顺序）
----------------------
1. **Louvain 算法**（若 python-louvain/networkx 可用）
   - 最优模块度最大化，O(N log N)
   - 非确定性：多次运行取最优 Q

2. **谱聚类**（sklearn，纯 numpy fallback）
   - 基于归一化 Laplacian 的前 k 个特征向量
   - 对 k 的选择敏感，但结果确定且数值稳定
   - 连续型谱聚类比 Louvain 更适合稠密矩阵（如 FC、R）

3. **模块度 Q 计算（纯 numpy）**
   - Newman-Girvan (2004) 公式，适用于有权无向图的对称版本
   - 对有向矩阵（如 R）：先对称化再计算 Q（方向信息丢失）

批判性注意事项
--------------
1. **Louvain 在稠密加权矩阵上可能过分分割（resolution limit）**
   应同时报告 k=4, 6, 8 等固定 k 值的谱聚类结果。
2. **Q > 0.4 阈值来自 Newman (2004)，针对稀疏图**。
   对稠密 FC 矩阵（几乎全非零），Q 值系统性偏低。
   建议同时对比随机矩阵的 Q（零假设下的基准线）。
3. **有向 EC 矩阵的社区检测**：将 R 对称化（(R+Rᵀ)/2）后检测
   无向社区，再计算有向模块度 Q_directed（若需方向性社区）。

输出文件
--------
  community_structure_{label}.json     — 社区检测结果与 Q 值
  community_size_histogram_{label}.png — 社区规模直方图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from scipy import sparse

logger = logging.getLogger(__name__)

_DEFAULT_K_RANGE = [3, 4, 5, 6, 7, 8]


# ─────────────────────────────────────────────────────────────────────────────
# Modularity Q computation (pure numpy, directed or undirected)
# ─────────────────────────────────────────────────────────────────────────────

def compute_modularity_q(
    W: np.ndarray,
    community_labels: np.ndarray,
    directed: bool = False,
) -> float:
    """
    计算 Newman-Girvan 模块度 Q。

    对无向图（或对称化后的有向图）：
      Q = (1/2m) Σ_{ij} [W_{ij} - k_i k_j / 2m] δ(c_i, c_j)

    对有向图（Arenas et al. 2007）：
      Q = (1/m) Σ_{ij} [W_{ij} - k_i^{out} k_j^{in} / m] δ(c_i, c_j)

    Args:
        W:                 连接矩阵 (N, N)（可含负值）。
        community_labels:  shape (N,)，0-indexed 社区归属。
        directed:          若 True 使用有向模块度公式。

    Returns:
        Q: float，模块度值（理想范围 [0, 1]，负值可能）。
    """
    W = np.asarray(W, dtype=np.float64)
    # Use absolute values to handle negative weights (standard practice)
    W_pos = np.abs(W)

    if not directed:
        W_sym = (W_pos + W_pos.T) / 2.0
        m = W_sym.sum() / 2.0
        if m < 1e-30:
            return 0.0
        k = W_sym.sum(axis=1)  # degree
        N = len(k)
        Q = 0.0
        labels = np.asarray(community_labels)
        for c in np.unique(labels):
            in_c = np.where(labels == c)[0]
            # Sum of W[i,j] for i,j in same community
            L_c = W_sym[np.ix_(in_c, in_c)].sum()
            d_c = k[in_c].sum()
            Q += L_c / (2.0 * m) - (d_c / (2.0 * m)) ** 2
    else:
        m = W_pos.sum()
        if m < 1e-30:
            return 0.0
        k_out = W_pos.sum(axis=1)  # out-degree
        k_in = W_pos.sum(axis=0)   # in-degree
        labels = np.asarray(community_labels)
        Q = 0.0
        for c in np.unique(labels):
            in_c = np.where(labels == c)[0]
            L_c = W_pos[np.ix_(in_c, in_c)].sum()
            d_out_c = k_out[in_c].sum()
            d_in_c = k_in[in_c].sum()
            Q += L_c / m - (d_out_c * d_in_c) / (m ** 2)

    return float(Q)


# ─────────────────────────────────────────────────────────────────────────────
# Spectral community detection (primary fallback — sklearn + numpy)
# ─────────────────────────────────────────────────────────────────────────────

def spectral_community_detection(
    W: np.ndarray,
    k: int,
    symmetric: bool = True,
    seed: int = 42,
) -> np.ndarray:
    """
    基于归一化 Laplacian 的谱聚类社区检测。

    算法（Ng, Jordan & Weiss 2002 / Shi & Malik 2000）：
    1. 对称化 W_sym = (|W| + |W|ᵀ) / 2
    2. 计算度矩阵 D = diag(W_sym.sum(axis=1))
    3. 归一化 Laplacian L = D^{-1/2} W_sym D^{-1/2}（相似变换）
    4. 计算前 k 个特征向量（最大特征值对应模块内高连接性）
    5. 在特征向量空间运行 k-means

    与 Louvain 的区别：
    - 确定性（给定种子）
    - 适合稠密矩阵（Louvain 对稠密图有 resolution limit）
    - 直接控制社区数量 k

    Args:
        W:         连接矩阵 (N, N)。
        k:         社区数量。
        symmetric: 是否对称化（对非对称 EC 矩阵应为 True）。
        seed:      k-means 随机种子。

    Returns:
        labels: shape (N,)，0-indexed 社区归属。
    """
    from sklearn.cluster import KMeans

    N = W.shape[0]
    W_abs = np.abs(W).astype(np.float64)
    if symmetric:
        W_abs = (W_abs + W_abs.T) / 2.0

    # Degree and normalized Laplacian (ratio cut form)
    d = W_abs.sum(axis=1)
    d_safe = np.where(d < 1e-12, 1.0, d)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d_safe))
    L_sym = D_inv_sqrt @ W_abs @ D_inv_sqrt  # (N, N) symmetric

    # Top-k eigenvectors of L_sym (largest eigenvalues = most connected communities)
    k_actual = min(k, N - 1)
    try:
        eigvals, eigvecs = np.linalg.eigh(L_sym)
        # eigh returns ascending order → take last k
        V = eigvecs[:, -k_actual:]  # (N, k)
    except np.linalg.LinAlgError:
        logger.warning("特征值分解失败，回退到随机初始化。")
        V = np.random.default_rng(seed).random((N, k_actual))

    # Row-normalize V (standard spectral clustering step)
    row_norms = np.linalg.norm(V, axis=1, keepdims=True)
    V = V / np.where(row_norms < 1e-12, 1.0, row_norms)

    km = KMeans(n_clusters=k_actual, random_state=seed, n_init=10)
    labels = km.fit_predict(V)
    return labels.astype(np.int32)


def _try_louvain(W: np.ndarray, seed: int = 42) -> Optional[Tuple[np.ndarray, float]]:
    """
    尝试使用 Louvain 算法进行社区检测。

    优先级：
    1. python-louvain (``community`` 包，pip install python-louvain)
    2. networkx >= 2.7 内置 Louvain (``networkx.algorithms.community.louvain_communities``)

    Returns:
        (labels, Q) 若成功；None 若两者均不可用。
    """
    try:
        import networkx as nx
    except ImportError:
        return None

    N = W.shape[0]
    W_abs = np.abs(W)
    W_sym = (W_abs + W_abs.T) / 2.0

    G = nx.from_numpy_array(W_sym)

    # ── 尝试 python-louvain (pip install python-louvain) ─────────────────────
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G, random_state=seed)
        labels = np.array([partition[i] for i in range(N)], dtype=np.int32)
        Q = community_louvain.modularity(partition, G)
        return labels, float(Q)
    except ImportError:
        pass

    # ── 尝试 networkx >= 2.7 内置 Louvain ────────────────────────────────────
    try:
        from networkx.algorithms.community import louvain_communities, modularity
        communities = louvain_communities(G, seed=seed)
        labels = np.zeros(N, dtype=np.int32)
        for k, comm in enumerate(communities):
            for node in comm:
                labels[node] = k
        Q = modularity(G, communities)
        return labels, float(Q)
    except (AttributeError, ImportError):
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Optimal k selection via modularity
# ─────────────────────────────────────────────────────────────────────────────

def find_optimal_k(
    W: np.ndarray,
    k_range: Optional[list] = None,
    seed: int = 42,
) -> Tuple[int, np.ndarray, float]:
    """
    在 k_range 中搜索使模块度 Q 最大的社区数 k。

    对每个 k，运行谱聚类，计算 Q；返回最优 (k, labels, Q)。

    批判性注意：Q 的最大化不等于"正确"社区结构；
    Q 对 k 不单调，一般在 k ≈ sqrt(N/2) 附近达到峰值。

    Args:
        W:       连接矩阵 (N, N)。
        k_range: 候选 k 列表；默认 [3,4,5,6,7,8]。
        seed:    随机种子。

    Returns:
        (best_k, best_labels, best_Q)
    """
    if k_range is None:
        k_range = _DEFAULT_K_RANGE
    N = W.shape[0]
    k_range = [k for k in k_range if 2 <= k < N]

    best_k, best_labels, best_Q = k_range[0], None, -np.inf
    results = []
    for k in k_range:
        labels = spectral_community_detection(W, k=k, seed=seed)
        Q = compute_modularity_q(W, labels)
        results.append((k, Q))
        if Q > best_Q:
            best_Q = Q
            best_k = k
            best_labels = labels
        logger.debug("  k=%d: Q=%.4f", k, Q)

    logger.info("最优 k=%d (Q=%.4f)", best_k, best_Q)
    if best_labels is None:
        best_labels = spectral_community_detection(W, k=best_k, seed=seed)
    return best_k, best_labels, best_Q


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_community_histogram(
    community_labels: np.ndarray,
    Q: float,
    output_path: Path,
    label: str,
) -> None:
    """Community size histogram (C3)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    unique, counts = np.unique(community_labels, return_counts=True)
    n_communities = len(unique)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram of community sizes
    ax1.bar(range(n_communities), counts, color="steelblue", alpha=0.8, edgecolor="k", lw=0.5)
    ax1.set_xlabel("Community ID")
    ax1.set_ylabel("Node Count")
    ax1.set_title(f"Community Size Distribution (k={n_communities}, Q={Q:.3f})  [{label}]")
    ax1.axhline(counts.mean(), ls="--", color="red", lw=1, label=f"mean={counts.mean():.1f}")
    ax1.legend()

    # Pie chart
    ax2.pie(counts, labels=[f"C{i+1}" for i in range(n_communities)],
            autopct="%1.0f%%", startangle=90,
            colors=plt.cm.tab10(np.linspace(0, 0.9, n_communities)))
    ax2.set_title("Community Size Proportion")

    _mod_str = "strong" if Q > 0.4 else ("med" if Q > 0.2 else "weak")
    fig.suptitle(f"Community Structure (k={n_communities}, Q={Q:.3f}, "
                 f"{_mod_str} modularity)  [{label}]",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


def _try_plot_q_vs_k(
    k_q_pairs: list,
    output_path: Path,
    label: str,
) -> None:
    """Plot Q vs k curve to aid optimal community count selection."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    ks = [p[0] for p in k_q_pairs]
    qs = [p[1] for p in k_q_pairs]
    best_k = ks[int(np.argmax(qs))]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, qs, "o-", ms=6, lw=1.8, color="steelblue")
    ax.axvline(best_k, ls="--", color="red", lw=1, label=f"best k={best_k}")
    ax.axhline(0.4, ls=":", color="orange", lw=1, label="Q=0.4 (strong mod.)")
    ax.axhline(0.2, ls=":", color="gray", lw=0.8, label="Q=0.2")
    ax.set_xlabel("Number of communities k")
    ax.set_ylabel("Modularity Q")
    ax.set_title(f"Modularity vs Number of Communities  [{label}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_community_structure(
    W: np.ndarray,
    k_range: Optional[list] = None,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    seed: int = 42,
) -> Dict:
    """
    运行模块 C：社区结构分析。

    步骤：
    1. 尝试 Louvain（若 networkx/community 可用）
    2. 否则：谱聚类搜索最优 k（模块度最大化）
    3. 计算模块度 Q
    4. 生成社区规模直方图和 Q vs k 曲线

    Args:
        W:          连接矩阵 (N, N)。
        k_range:    候选社区数列表（谱聚类模式）；默认 [3,4,5,6,7,8]。
        output_dir: 输出目录。
        label:      文件名标签。
        seed:       随机种子。

    Returns:
        dict 包含:
          method:              "louvain" 或 "spectral_clustering"
          n_communities:       最终社区数量
          community_labels:    np.ndarray (N,)，社区归属
          modularity_q:        Q 值
          community_sizes:     list，各社区节点数
          q_interpretation:    "strong" / "moderate" / "weak"
          k_q_pairs:           list of (k, Q) 供绘图
    """
    N = W.shape[0]
    logger.info("C 社区检测: N=%d", N)

    # Try Louvain first
    louvain_result = _try_louvain(W, seed=seed)
    if louvain_result is not None:
        labels, Q = louvain_result
        method = "louvain"
        best_k = int(np.max(labels) + 1)
        logger.info("Louvain: k=%d, Q=%.4f", best_k, Q)
        k_q_pairs = [(best_k, Q)]
    else:
        logger.info("Louvain 不可用（需要 python-louvain 或 networkx>=2.7），使用谱聚类。")
        method = "spectral_clustering"
        if k_range is None:
            k_range = _DEFAULT_K_RANGE
        k_range = [k for k in k_range if 2 <= k < N]

        k_q_pairs = []
        best_k, labels, Q = -1, None, -np.inf
        for k in k_range:
            lbl = spectral_community_detection(W, k=k, seed=seed)
            q = compute_modularity_q(W, lbl)
            k_q_pairs.append((k, q))
            if q > Q:
                Q = q
                best_k = k
                labels = lbl
        logger.info("谱聚类最优: k=%d, Q=%.4f", best_k, Q)

    unique_labels, counts = np.unique(labels, return_counts=True)

    q_interpretation = (
        "strong" if Q > 0.4 else
        "moderate" if Q > 0.2 else
        "weak"
    )

    result: Dict = {
        "method": method,
        "n_communities": int(best_k),
        "modularity_q": round(float(Q), 5),
        "q_interpretation": q_interpretation,
        "community_sizes": counts.tolist(),
        "community_labels": labels.tolist(),
        "k_q_pairs": k_q_pairs,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save JSON (without large arrays)
        json_result = {k: v for k, v in result.items()
                       if k not in ("community_labels",)}
        json_path = out / f"community_structure_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        # Save labels array
        np.save(out / f"community_labels_{label}.npy", labels)

        # Plots
        _try_plot_community_histogram(
            labels, Q,
            out / f"community_size_histogram_{label}.png",
            label,
        )
        if len(k_q_pairs) > 1:
            _try_plot_q_vs_k(
                k_q_pairs,
                out / f"community_q_vs_k_{label}.png",
                label,
            )

    return result
