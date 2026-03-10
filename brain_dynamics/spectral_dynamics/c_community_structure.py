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
from typing import Dict, List, Optional, Tuple

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
# Null Q distribution (degree-sequence-preserving random networks)
# ─────────────────────────────────────────────────────────────────────────────

def _degree_preserving_rewire_sym(
    W_sym: np.ndarray,
    n_swaps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Maslov-Sneppen rewiring for a symmetric weighted matrix.

    Performs ``n_swaps`` attempted edge-pair swaps that preserve the
    strength sequence (row/column sums) while randomising the wiring
    pattern.  Only the upper triangle is operated on; the result is
    symmetrised before returning.

    **Important**: this algorithm is designed for *sparse* binary/weighted
    networks.  For dense matrices (most/all pairs have non-zero weights) the
    "skip if proposed edge already exists" guard will reject virtually every
    swap attempt, leaving the matrix unchanged and producing a null
    distribution identical to the observed Q (z=0, p=1).  Use
    :func:`_weight_permutation_null` instead for dense matrices.

    Args:
        W_sym:   Symmetric (N, N) non-negative weight matrix.
        n_swaps: Number of swap attempts.
        rng:     ``np.random.Generator`` instance.

    Returns:
        Rewired symmetric matrix with the same shape and strength sequence.
    """
    # Work on upper triangle only to keep symmetry
    W_out = W_sym.copy().astype(np.float64)
    rows, cols = np.where(np.triu(W_out, k=1) != 0)
    n_edges = len(rows)
    if n_edges < 4:
        return W_out

    for _ in range(n_swaps):
        idx = rng.choice(n_edges, size=2, replace=False)
        i, j = int(rows[idx[0]]), int(cols[idx[0]])
        k, l = int(rows[idx[1]]), int(cols[idx[1]])
        # Proposed new edges: (i, l) and (k, j)
        # Skip if proposed edge already exists or if swap would create self-loop
        if i == l or k == j or i == k or j == l:
            continue
        if W_out[i, l] != 0 or W_out[k, j] != 0:
            continue
        # Swap weights
        wij = W_out[i, j]
        wkl = W_out[k, l]
        W_out[i, j] = 0.0
        W_out[j, i] = 0.0
        W_out[k, l] = 0.0
        W_out[l, k] = 0.0
        W_out[i, l] = wij
        W_out[l, i] = wij
        W_out[k, j] = wkl
        W_out[j, k] = wkl
        rows[idx[0]], cols[idx[0]] = i, l
        rows[idx[1]], cols[idx[1]] = k, j

    return W_out


def _weight_permutation_null(
    W_sym: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Weight-permutation null model for *dense* symmetric matrices.

    Randomly permutes all off-diagonal weights while preserving the diagonal
    and the overall weight distribution.  This tests: "what Q would we expect
    if the same set of edge weights were assigned randomly?"

    Unlike Maslov-Sneppen rewiring, weight permutation does NOT preserve the
    strength (degree) sequence.  However, for fully-connected FC matrices
    (where every pair has a non-zero weight) Maslov-Sneppen cannot rewire at
    all — so weight permutation is the only practical alternative.

    Scientific rationale: for dense FC matrices derived from Pearson
    correlations, nearly all off-diagonal entries are non-zero.  The
    Maslov-Sneppen "skip if edge exists" guard rejects every proposed swap,
    leaving the null matrix identical to the input → null Q = observed Q,
    z = 0, p = 1 (degenerate result).  Weight permutation breaks the
    community wiring pattern while keeping the magnitude distribution intact,
    producing a meaningful null baseline.  See Watts & Strogatz (1998) and
    Fornito et al. (2016, *Fundamentals of Brain Network Analysis*), §6.4.

    Args:
        W_sym:   Symmetric (N, N) non-negative weight matrix.
        rng:     ``np.random.Generator`` instance.

    Returns:
        Permuted symmetric matrix with the same diagonal and weight
        distribution as *W_sym*.
    """
    N = W_sym.shape[0]
    W_out = W_sym.copy().astype(np.float64)
    # Extract all upper-triangle indices (off-diagonal weights)
    iu = np.triu_indices(N, k=1)
    weights = W_out[iu].copy()
    rng.shuffle(weights)
    W_out[iu] = weights
    # Reflect to lower triangle to maintain symmetry
    W_out[(iu[1], iu[0])] = weights
    return W_out


# Fraction of upper-triangle non-zero entries above which a matrix is
# considered "dense" and weight-permutation replaces Maslov-Sneppen rewiring.
_DENSE_THRESHOLD = 0.50


def compute_q_null_distribution(
    W_sym: np.ndarray,
    n_null: int = 100,
    seed: int = 42,
    louvain_seed: int = 42,
) -> Tuple[List[float], str]:
    """Generate a null Q distribution for modularity significance testing.

    Selects the appropriate null-model strategy based on matrix density:

    * **Sparse** matrices (≤50 % non-zero upper-triangle entries):
      Maslov-Sneppen degree-preserving rewiring (Maslov & Sneppen, 2002).
      Preserves the strength sequence.  Answers: "what Q if FC had the same
      degree sequence but random wiring?"

    * **Dense** matrices (>50 % non-zero, typical for brain FC):
      Weight-permutation null model.  Preserves the weight magnitude
      distribution but randomises which node-pairs carry which weights.
      Answers: "what Q if the same weights were randomly reassigned?"

    For each of ``n_null`` realisations the same community-detection algorithm
    used on the real network (Louvain if available, otherwise best spectral-
    clustering k) is run and the resulting Q is recorded.

    Args:
        W_sym:        Symmetric (N, N) FC matrix (non-negative weights).
        n_null:       Number of null networks to generate.
        seed:         RNG seed for rewiring / permutation.
        louvain_seed: Seed passed to Louvain / spectral clustering.

    Returns:
        Tuple ``(null_qs, null_model_type)`` where *null_qs* is a list of
        Q values (length ``n_null``) and *null_model_type* is either
        ``"maslov_sneppen"`` or ``"weight_permutation"``.
    """
    rng = np.random.default_rng(seed)
    N = W_sym.shape[0]

    # Determine null-model strategy from matrix density
    n_upper = N * (N - 1) // 2
    n_nonzero = int(np.sum(np.triu(W_sym, k=1) != 0))
    density = n_nonzero / max(n_upper, 1)

    # n_swaps is only used by maslov_sneppen; initialise to 0 to satisfy
    # static-analysis tools that flag potentially-undefined references.
    n_swaps: int = 0

    if density > _DENSE_THRESHOLD:
        null_model_type = "weight_permutation"
        logger.info(
            "  Null model: weight permutation (density=%.1f%% > %.0f%% threshold). "
            "Maslov-Sneppen rewiring is inapplicable for dense FC matrices — "
            "virtually all swap attempts are rejected because every pair already "
            "has a non-zero weight. Weight permutation randomises the wiring "
            "while preserving the overall weight distribution.",
            density * 100, _DENSE_THRESHOLD * 100,
        )
    else:
        null_model_type = "maslov_sneppen"
        # Use 10×|E| swap attempts — standard practice for thorough rewiring
        n_swaps = max(n_nonzero * 10, 100)
        logger.info(
            "  Null model: Maslov-Sneppen rewiring (density=%.1f%% ≤ %.0f%% threshold, "
            "n_swaps=%d).",
            density * 100, _DENSE_THRESHOLD * 100, n_swaps,
        )

    null_qs: List[float] = []
    for i in range(n_null):
        if null_model_type == "weight_permutation":
            W_rand = _weight_permutation_null(W_sym, rng)
        else:
            # n_swaps is defined in the maslov_sneppen branch above
            W_rand = _degree_preserving_rewire_sym(W_sym, n_swaps, rng)

        # Use a different community-detection seed for each null realisation.
        # A single fixed seed would give identical community assignments on
        # nearly-identical rewired matrices, collapsing the null distribution.
        # louvain_seed + i guarantees independent Louvain runs while remaining
        # deterministic and reproducible given the same louvain_seed argument.
        louvain_result = _try_louvain(W_rand, seed=louvain_seed + i)
        if louvain_result is not None:
            _, q_rand = louvain_result
        else:
            # Spectral clustering fallback: sweep k ∈ [3, 4, 5, 6, 7, 8]
            k_range = [k for k in _DEFAULT_K_RANGE if 2 <= k < N]
            q_rand = -np.inf
            for k in k_range:
                lbl = spectral_community_detection(W_rand, k=k, seed=louvain_seed + i)
                q = compute_modularity_q(W_rand, lbl)
                if q > q_rand:
                    q_rand = q
        null_qs.append(float(q_rand))
        if (i + 1) % 10 == 0:
            logger.debug("  Q null %d/%d: q_rand=%.4f", i + 1, n_null, q_rand)

    return null_qs, null_model_type


def q_significance_test(
    observed_q: float,
    null_qs: List[float],
) -> Dict:
    """Compute z-score and p-value of the observed Q against a null distribution.

    Args:
        observed_q: Q from the real network.
        null_qs:    List of Q values from null (rewired) networks.

    Returns:
        dict with keys:
            null_mean   float  — mean of null distribution
            null_std    float  — std  of null distribution
            z_score     float  — (observed_q - null_mean) / null_std
            p_value     float  — fraction of null Qs ≥ observed_q (one-sided)
            significant bool   — True when z_score > 1.96  (p < 0.05, approx.)
            n_null      int    — number of null realisations
    """
    arr = np.array(null_qs, dtype=np.float64)
    null_mean = float(np.mean(arr))
    null_std = float(np.std(arr))
    if null_std < 1e-12:
        z_score = 0.0
    else:
        z_score = (observed_q - null_mean) / null_std
    # One-sided empirical p-value: P(Q_null ≥ Q_observed)
    p_value = float(np.mean(arr >= observed_q))

    # ── Paper-quality p-value string ─────────────────────────────────────────
    # The empirical p_value resolves only to 1/n_null (e.g. 0.01 for n=100).
    # When no null sample exceeds the observed Q, report the upper bound and
    # supplement with the Gaussian normal-tail p-value derived from z_score.
    n_null = len(null_qs)
    if p_value == 0.0:
        # Exact empirical bound: p < 1/n_null
        p_bound = 1.0 / n_null
        # Gaussian tail approximation (valid when z >> 1)
        try:
            import math
            # Use log-space for large z to avoid underflow: log10(p) ≈ -z²/2/ln(10)
            if z_score > 37:  # scipy.stats.norm.sf would underflow to 0.0
                log10_p = -(z_score ** 2) / (2.0 * math.log(10))
                # Format: p < X.XXe-NN
                exp = int(math.floor(log10_p))
                mantissa = 10 ** (log10_p - exp)
                p_gaussian_str = f"{mantissa:.2f}×10^{exp}"
            else:
                from scipy.stats import norm as _norm
                p_gauss = float(_norm.sf(z_score))
                if p_gauss < 1e-4:
                    p_gaussian_str = f"{p_gauss:.2e}"
                else:
                    p_gaussian_str = f"{p_gauss:.4f}"
        except Exception:
            p_gaussian_str = "N/A"
        p_formatted = f"p < {p_bound:.2e} (empirical); p_Gauss = {p_gaussian_str}"
    else:
        p_formatted = f"p = {p_value:.4f}"

    return {
        "null_mean": round(null_mean, 5),
        "null_std": round(null_std, 5),
        "z_score": round(z_score, 4),
        "p_value": round(p_value, 4),
        "p_formatted": p_formatted,
        "significant": bool(z_score > 1.96),
        "n_null": n_null,
    }


def _try_plot_q_null_distribution(
    null_qs: List[float],
    observed_q: float,
    sig: Dict,
    output_path: Path,
    label: str,
    null_model_type: str = "unknown",
) -> None:
    """Histogram of null Q distribution with observed Q marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null_qs, bins=20, color="steelblue", alpha=0.75, edgecolor="k",
            lw=0.5, label=f"Null Q (n={len(null_qs)})")
    ax.axvline(observed_q, color="red", lw=2,
               label=f"Observed Q={observed_q:.4f}")
    ax.axvline(sig.get("null_mean", float("nan")), color="gray",
               lw=1, ls="--", label=f"Null mean={sig.get('null_mean', 0):.4f}")

    z = sig.get("z_score", 0)
    p_display = sig.get("p_formatted") or f"p={sig.get('p_value', 1):.4f}"
    sig_txt = "significant (p<0.05)" if sig.get("significant") else "not significant (p>=0.05)"
    ax.set_xlabel("Modularity Q")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Q Null Distribution ({null_model_type})  [{label}]\n"
        f"z={z:.2f}, {p_display} — {sig_txt}"
    )
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
    n_null: int = 100,
) -> Dict:
    """
    运行模块 C：社区结构分析（含 Q 显著性检验）。

    步骤：
    1. 尝试 Louvain（若 networkx/community 可用）
    2. 否则：谱聚类搜索最优 k（模块度最大化）
    3. 计算模块度 Q
    4. 若 n_null > 0：生成度序列保持的随机网络（配置模型，Maslov-Sneppen），
       对每个随机网络运行相同社区检测，得到 Q 零分布，计算 z-score / p 值。
    5. 生成社区规模直方图和 Q vs k 曲线；若运行了显著性检验，
       额外保存 Q 零分布直方图。

    Args:
        W:          连接矩阵 (N, N)。
        k_range:    候选社区数列表（谱聚类模式）；默认 [3,4,5,6,7,8]。
        output_dir: 输出目录。
        label:      文件名标签。
        seed:       随机种子。
        n_null:     生成的随机网络数量（默认 100）。设为 0 跳过显著性检验。

    Returns:
        dict 包含:
          method:              "louvain" 或 "spectral_clustering"
          n_communities:       最终社区数量
          community_labels:    list (N,)，社区归属
          modularity_q:        Q 值
          community_sizes:     list，各社区节点数
          q_interpretation:    "strong" / "moderate" / "weak"
          k_q_pairs:           list of (k, Q) 供绘图
          q_significance:      dict（仅当 n_null > 0）
            null_mean, null_std, z_score, p_value, significant, n_null
    """
    N = W.shape[0]
    logger.info("C 社区检测: N=%d", N)

    # ── Symmetrise for community detection (FC is already symmetric, but
    #    guard against floating-point asymmetry) ──────────────────────────────
    W_abs = np.abs(W)
    W_sym = (W_abs + W_abs.T) / 2.0

    # Try Louvain first
    louvain_result = _try_louvain(W_sym, seed=seed)
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
            lbl = spectral_community_detection(W_sym, k=k, seed=seed)
            q = compute_modularity_q(W_sym, lbl)
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

    # ── Q significance test (null distribution) ──────────────────────────────
    null_qs: Optional[List[float]] = None
    null_model_type: Optional[str] = None
    if n_null > 0:
        logger.info(
            "  Q 显著性检验: 生成 %d 个随机网络零分布…", n_null
        )
        null_qs, null_model_type = compute_q_null_distribution(W_sym, n_null=n_null, seed=seed)
        sig = q_significance_test(float(Q), null_qs)
        sig["null_model"] = null_model_type
        result["q_significance"] = sig
        sig_str = "显著 ✓" if sig["significant"] else "不显著"
        logger.info(
            "  Q 显著性 [%s]: Q=%.4f, null mean=%.4f±%.4f, z=%.2f, %s (%s)",
            null_model_type,
            Q, sig["null_mean"], sig["null_std"],
            sig["z_score"], sig.get("p_formatted", f"p={sig['p_value']:.4f}"),
            sig_str,
        )

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
        if null_qs is not None:
            _try_plot_q_null_distribution(
                null_qs, float(Q),
                result.get("q_significance", {}),
                out / f"community_q_null_distribution_{label}.png",
                label,
                null_model_type=null_model_type or "unknown",
            )

    return result
