"""
Network Controllability Analysis
==================================

基于线性动力系统理论，计算脑网络的 **可控性（Controllability）** 指标，
量化哪些脑区是最优的刺激靶点。

理论背景
--------
给定线性时不变系统：

  x(t+1) = A·x(t) + B·u(t)

其中 A 为状态转移矩阵（有效连接矩阵），B 为输入矩阵（刺激模式），
**可控性 Gramian** 定义为：

  W_c = Σ_{t=0}^{∞} A^t · B · B^T · (A^T)^t

W_c 描述了单位能量输入能够驱动到哪些状态空间方向。

本模块实现三种可控性指标（Gu et al. 2015 Nature Commun.）：

1. **Average Controllability（平均可控性）**
   ``AC(i) = trace(W_c^{(i)})``
   度量从节点 i 输入时，系统的平均可达性。
   AC 大的脑区是"泛用型"驱动器，适合以较少能量驱动系统到大量不同状态。
   神经科学解释：前额叶、后扣带皮层等默认模式网络核心区通常 AC 最高，
   反映其作为广泛影响网络的枢纽角色。

2. **Modal Controllability（模态可控性）**
   ``MC(i) = Σ_k (1 − λ_k²) · v_{ki}²``
   加权所有模态（特征向量），权重为 1 − λ_k²（λ_k 为第 k 个特征值）。
   MC 大的脑区擅长驱动"难以到达"的高能量状态（与 AC 互补）。
   神经科学解释：初级感觉皮层、下丘脑等通常 MC 较高，能够转换动力学状态。

3. **Boundary Controllability（边界可控性）**
   基于图的社区检测，量化节点在社区间通信中的边界控制能力。
   BC(i) = 跨越社区边界的边的权重之和。

**稳定性保证**：
若 A 的最大特征值绝对值 ≥ 1（不稳定系统），Gramian 级数发散。
本模块对 A 进行谱归一化（除以 max|λ| + margin），确保级数收敛。
此操作不改变特征向量方向，仅缩放特征值幅度，保留相对可控性排序。

**与响应矩阵的关系**：
响应矩阵 R[i,j] 是数值模拟的"因果效应"矩阵，可近似作为 A 的估计。
可控性分析在 R 上进行，提供刺激靶点的最优选择理论依据。

科学参考
--------
  Gu S et al (2015) Nat Commun 6:8414
    — 脑网络可控性的第一篇系统研究
  Tang E & Bassett DS (2018) Rev Mod Phys 90:031003
    — 神经科学中的网络控制论综述
  Muldoon SF et al (2016) PLOS Comput Biol 12:e1005071
    — 节点选择对多频段控制的影响
  Kim JZ et al (2018) Nat Phys 14:91-98
    — 认知任务与可控性地形图的对应关系

输出文件
--------
  controllability_report.json  — 汇总统计 + 可控性排名
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of terms in the truncated Gramian series Σ A^t B B^T (A^T)^t.
# Theory requires Σ_{t=0}^∞ but for stable A (ρ(A) < 1) the series converges
# geometrically.  100 terms give ‖W_c − W_c_∞‖ < ρ(A)^100 ‖W_c_∞‖ ≤ 10^{-10}
# for ρ(A) ≤ 0.8 and ≈ 10^{-5} for ρ(A) = 0.9 — sufficient precision.
_GRAMIAN_TERMS: int = 100

# Spectral radius shrinkage: after normalising A so ρ(A) < 1, we further
# shrink to this target ρ to guarantee numeric convergence of the Gramian sum.
_TARGET_SPECTRAL_RADIUS: float = 0.9


# ══════════════════════════════════════════════════════════════════════════════
# Matrix normalisation
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_adjacency(A: np.ndarray, target_rho: float = _TARGET_SPECTRAL_RADIUS) -> np.ndarray:
    """
    谱归一化：将 A 的谱半径缩放到 target_rho，保留特征结构。

    Args:
        A:          方阵 (N, N)。
        target_rho: 目标谱半径（< 1 保证 Gramian 收敛）。

    Returns:
        A_norm: 归一化后的矩阵，形状同 A。
    """
    eigvals = np.linalg.eigvals(A)
    rho = float(np.abs(eigvals).max())
    if rho < 1e-10:
        return A.copy()
    return A * (target_rho / rho)


# ══════════════════════════════════════════════════════════════════════════════
# Controllability Gramian (per-node, diagonal only)
# ══════════════════════════════════════════════════════════════════════════════

def _compute_gramian_diagonal(
    A: np.ndarray,
    node_idx: int,
    n_terms: int = _GRAMIAN_TERMS,
) -> float:
    """
    计算节点 ``node_idx`` 作为单一输入时 Gramian 的迹（= 平均可控性）。

    数学等价于：
      W_c^{(i)} = Σ_{t=0}^{n_terms-1}  A^t · e_i · e_i^T · (A^T)^t
      AC(i) = trace(W_c^{(i)}) = Σ_t ‖A^t e_i‖²

    实现：逐步计算 A^t · e_i，累积其 L2 范数的平方。

    Args:
        A:         谱归一化后的状态矩阵 (N, N)。
        node_idx:  输入节点索引 i。
        n_terms:   级数截断项数。

    Returns:
        trace_value: AC(i) = Σ_{t=0}^{n_terms-1} ‖A^t e_i‖²。
    """
    N = A.shape[0]
    v = np.zeros(N, dtype=np.float64)
    v[node_idx] = 1.0

    gramian_trace = 0.0
    for _ in range(n_terms):
        gramian_trace += float(np.dot(v, v))  # ‖A^t e_i‖²
        v = A @ v
        if np.linalg.norm(v) < 1e-15:
            break
    return gramian_trace


def compute_average_controllability(
    A: np.ndarray,
    n_terms: int = _GRAMIAN_TERMS,
) -> np.ndarray:
    """
    计算所有节点的 **平均可控性（Average Controllability）**。

    AC(i) = trace(W_c^{(i)}) = Σ_{t=0}^{T-1} ‖A^t e_i‖²

    AC 大的节点能以较小的控制能量将系统驱动到大量不同状态（"易于控制"方向）。

    Args:
        A:       谱归一化后的状态矩阵 (N, N)。
        n_terms: Gramian 级数截断项数（默认 100）。

    Returns:
        ac: shape (N,)，各节点平均可控性（未归一化）。
    """
    N = A.shape[0]
    ac = np.array(
        [_compute_gramian_diagonal(A, i, n_terms) for i in range(N)],
        dtype=np.float64,
    )
    return ac


def compute_modal_controllability(A: np.ndarray) -> np.ndarray:
    """
    计算所有节点的 **模态可控性（Modal Controllability）**。

    MC(i) = Σ_k (1 − |λ_k|²) · |v_{ki}|²

    其中 λ_k 是 A 的第 k 个特征值，v_{ki} 是对应特征向量的第 i 个分量。

    模态可控性大的节点擅长驱动"难以激活"的慢模式（通常是高能量模式）。
    在神经网络中，这类节点往往与认知灵活性和状态切换相关。

    参考：Gu et al. (2015) 等式 (4)，Tang & Bassett (2018) §II.A。

    Args:
        A: 谱归一化后的状态矩阵 (N, N)，ρ(A) < 1 保证 1 − |λ_k|² > 0。

    Returns:
        mc: shape (N,)，各节点模态可控性（未归一化）。
    """
    eigvals, eigvecs = np.linalg.eig(A)
    # |v_{ki}|² for each node i and mode k
    # eigvecs columns are eigenvectors: eigvecs[i, k] = v_{ki}
    v_sq = np.abs(eigvecs) ** 2  # (N, N)
    weights = np.maximum(0.0, 1.0 - np.abs(eigvals) ** 2)  # (N,)
    mc = (v_sq * weights[np.newaxis, :]).sum(axis=1)  # (N,)
    return mc.real.astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Boundary controllability (community-based)
# ══════════════════════════════════════════════════════════════════════════════

def _detect_communities_simple(A: np.ndarray, n_communities: int = 6) -> np.ndarray:
    """
    简单的谱聚类社区检测（作为可控性计算的辅助）。

    使用绝对值对称化矩阵的前 n_communities 个特征向量，
    用 K-means 聚类分配社区标签。

    Args:
        A:              状态矩阵 (N, N)。
        n_communities:  目标社区数。

    Returns:
        labels: shape (N,)，整数社区标签 [0, n_communities)。
    """
    try:
        from sklearn.cluster import KMeans
    except ImportError:
        # Fallback: 均等分配
        N = A.shape[0]
        return np.arange(N, dtype=int) % n_communities

    A_sym = 0.5 * (np.abs(A) + np.abs(A).T)
    eigvals, eigvecs = np.linalg.eigh(A_sym)
    k = min(n_communities, A.shape[0])
    # Use the k eigenvectors corresponding to the k largest eigenvalues
    embedding = eigvecs[:, -k:].real

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=0)
    return kmeans.fit_predict(embedding)


def compute_boundary_controllability(
    A: np.ndarray,
    community_labels: Optional[np.ndarray] = None,
    n_communities: int = 6,
) -> np.ndarray:
    """
    计算 **边界可控性（Boundary Controllability）**。

    BC(i) = Σ_{j: label[j] ≠ label[i]} |A[i,j]| + |A[j,i]|

    边界可控性大的节点位于社区交界处，是社区间信息传递和状态切换的关键桥节点。

    Args:
        A:                状态矩阵 (N, N)。
        community_labels: 预计算的社区标签 (N,)；None → 自动检测。
        n_communities:    自动检测时的社区数。

    Returns:
        bc: shape (N,)，各节点边界可控性。
    """
    if community_labels is None:
        community_labels = _detect_communities_simple(A, n_communities)

    N = A.shape[0]
    A_abs = np.abs(A)
    bc = np.zeros(N, dtype=np.float64)
    for i in range(N):
        cross_mask = community_labels != community_labels[i]
        bc[i] = float((A_abs[i, cross_mask] + A_abs[cross_mask, i]).sum())
    return bc


# ══════════════════════════════════════════════════════════════════════════════
# Public run function
# ══════════════════════════════════════════════════════════════════════════════

def run_controllability_analysis(
    response_matrix: np.ndarray,
    n_communities: int = 6,
    n_gramian_terms: int = _GRAMIAN_TERMS,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行完整的网络可控性分析。

    使用响应矩阵 R（即经验有效连接矩阵 A ≈ R）作为线性系统的状态转移矩阵，
    计算三种可控性指标，并生成刺激靶点排名。

    Args:
        response_matrix:  shape (N, N)，响应矩阵 R[i,j]（TwinBrain 或 WC 模式）。
        n_communities:    边界可控性中的社区数量（默认 6）。
        n_gramian_terms:  Gramian 级数截断项（默认 100）。
        output_dir:       输出目录；None → 不保存文件。

    Returns:
        results: {
            "average_controllability":   np.ndarray (N,)
            "modal_controllability":     np.ndarray (N,)
            "boundary_controllability":  np.ndarray (N,)
            "community_labels":          np.ndarray (N,)
            "top_ac_regions":            List[int]  — AC 最高的前 5 个脑区
            "top_mc_regions":            List[int]  — MC 最高的前 5 个脑区
            "top_bc_regions":            List[int]  — BC 最高的前 5 个脑区
            "report":                    dict  — JSON-serialisable summary
        }
    """
    output_dir = Path(output_dir) if output_dir is not None else None

    R = response_matrix.astype(np.float64)
    N = R.shape[0]

    # Symmetrise for controllability: A ≈ (R + R.T) / 2 preserves spectral
    # properties while eliminating asymmetric noise (Tang & Bassett 2018 §II.A).
    A_raw = 0.5 * (R + R.T)

    # Normalise spectral radius to _TARGET_SPECTRAL_RADIUS for Gramian convergence.
    A = _normalise_adjacency(A_raw, _TARGET_SPECTRAL_RADIUS)
    logger.info(
        "可控性分析：A 矩阵大小 %d×%d，谱半径已归一化到 %.2f",
        N, N, _TARGET_SPECTRAL_RADIUS,
    )

    # ── Average Controllability ───────────────────────────────────────────────
    logger.info("  计算平均可控性（Average Controllability）...")
    ac = compute_average_controllability(A, n_terms=n_gramian_terms)

    # ── Modal Controllability ─────────────────────────────────────────────────
    logger.info("  计算模态可控性（Modal Controllability）...")
    mc = compute_modal_controllability(A)

    # ── Community detection + Boundary Controllability ─────────────────────
    logger.info("  检测社区结构并计算边界可控性...")
    community_labels = _detect_communities_simple(A, n_communities)
    bc = compute_boundary_controllability(A, community_labels)

    # ── Rankings ──────────────────────────────────────────────────────────────
    k = min(5, N)
    top_ac = list(map(int, np.argsort(ac)[-k:][::-1]))
    top_mc = list(map(int, np.argsort(mc)[-k:][::-1]))
    top_bc = list(map(int, np.argsort(bc)[-k:][::-1]))

    # AC−MC correlation (negative expected: AC and MC are theoretically anti-correlated
    # for generic networks — Gu et al. 2015 Fig. 2)
    if ac.std() > 1e-10 and mc.std() > 1e-10:
        ac_mc_corr = float(np.corrcoef(ac, mc)[0, 1])
    else:
        ac_mc_corr = float("nan")

    report = {
        "n_regions":         N,
        "n_communities":     n_communities,
        "n_gramian_terms":   n_gramian_terms,
        "ac_mc_correlation": ac_mc_corr,
        "top_ac_regions":    top_ac,
        "top_mc_regions":    top_mc,
        "top_bc_regions":    top_bc,
        "ac_mean":           float(ac.mean()),
        "ac_std":            float(ac.std()),
        "mc_mean":           float(mc.mean()),
        "mc_std":            float(mc.std()),
        "bc_mean":           float(bc.mean()),
        "bc_std":            float(bc.std()),
        "community_sizes":   {
            str(lbl): int((community_labels == lbl).sum())
            for lbl in range(n_communities)
        },
        "interpretation_zh": (
            f"网络可控性分析完成（N={N} 脑区，{n_communities} 个社区）。"
            f"AC-MC 相关性 = {ac_mc_corr:.3f}"
            f"{'（预期负相关，与 Gu 2015 一致）' if (np.isfinite(ac_mc_corr) and ac_mc_corr < 0) else '（无效值 (NaN)，数据不足）' if not np.isfinite(ac_mc_corr) else '（正相关，不典型）'}。"
            f"平均可控性最高脑区（易于广泛驱动）：{top_ac[:3]}；"
            f"模态可控性最高脑区（能量-efficient 状态切换）：{top_mc[:3]}；"
            f"边界可控性最高脑区（社区间协调）：{top_bc[:3]}。"
            "建议以 AC 最高脑区作为广播型刺激靶点，MC 最高脑区作为状态切换型刺激靶点。"
        ),
    }

    if output_dir is not None:
        report_path = output_dir / "controllability_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  可控性报告已保存: %s", report_path)

    logger.info(
        "可控性分析完成: AC 最高=%s, MC 最高=%s, BC 最高=%s",
        top_ac[:3], top_mc[:3], top_bc[:3],
    )

    return {
        "average_controllability":   ac,
        "modal_controllability":     mc,
        "boundary_controllability":  bc,
        "community_labels":          community_labels,
        "top_ac_regions":            top_ac,
        "top_mc_regions":            top_mc,
        "top_bc_regions":            top_bc,
        "report":                    report,
    }
