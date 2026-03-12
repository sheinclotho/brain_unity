#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Task PCA Projection — High-Dimensional Cognitive Dynamics Manifold Analysis
==================================================================================

**核心问题**：不同任务（或不同被试、不同 run）之间的动力学差异，
是真实的动力学结构差异，还是仅仅是同一个高维吸引子上的相位/位置差异？

**背景**：最新研究（Stringer et al. 2019; Saxena & Cunningham 2019;
Gallego et al. 2020）表明神经动力学流形通常为**10–20 维**。
因此，仅在 PC1–PC2（2D 投影）上评估流形结构（如环形评分）是不可靠的：
一个真实的高维环形/环面可能在 2D 投影中看起来完全弥散。

**本模块采用维度感知（dimension-aware）方法**：

方法：
  1. 从 graph 文件夹读取所有图缓存 (``.pt`` 文件）
  2. 对**每个**图文件，使用 GNN 模型生成自由动力学轨迹。
     每个任务只使用**自身**的 BOLD 历史作为初始上下文（不混入其它任务的数据）。
  3. 将所有任务的轨迹**合并后**拟合统一联合 PCA（保证投影空间一致），
     计算足够多的主成分（默认 20 个，覆盖 10D+ 流形）。
  4. 高维流形分析（核心新增）：
     (a) **参与率 PR**（Participation Ratio）—— 从 PCA 特征值估计内在维度：
         PR = (Σλ_i)² / Σλ_i²；完美 k 维子空间的 PR = k。
     (b) **Grassmannian 距离**（主角度）—— 使用 SVD 计算任务特异子空间之间的
         主角度（principal angles），量化子空间的几何距离。
         小角度 → 任务共享同一流形；大角度 → 任务各占独立子空间。
     (c) **高维 k-NN 任务纯度**（k-NN Task Purity）—— 在完整高维 PCA 空间中，
         计算每个点 k 个最近邻中来自同一任务的比例，与随机基准（1/n_tasks）比较。
         接近随机 → 任务良好混合（共享流形）；远超随机 → 任务分离。
     (d) **高维组间/组内方差比**（Between/Within Variance Ratio）——
         在高维 PCA 空间中计算组间方差 vs 组内方差。
  5. 输出多 PC 对可视化（PC1-2、PC3-4、PC5-6、PC7-8 方格图）和 JSON 指标文件

**如何解读结果**：

- **k-NN 纯度比 < 1.5 且主角度 < 30°** → 共享高维流形：
  不同任务轨迹在高维 PCA 空间中高度混合，使用相同的底层吸引子几何。

- **k-NN 纯度比 > 2.5 或主角度 > 60°** → 独立高维流形：
  不同任务轨迹占据高维空间的不同区域，具有任务特异性的动力学子空间。

- **PR（参与率）≈ 10** 证实流形确为高维（与近期文献一致），2D 环形评分不再有效。

**用法**::

    python -m analysis.cross_task_pca \\
        --model outputs/exp/best_model.pt \\
        --graphs outputs/graph_cache/ \\
        --output outputs/cross_task_pca/ \\
        [--modality fmri] \\
        [--n-init 30] \\
        [--steps 200] \\
        [--seed 42] \\
        [--n-components 20]

**命令行参数**：

--model         训练好的模型检查点路径（``best_model.pt``）
--graphs        包含图缓存 .pt 文件的文件夹路径
--output        输出目录
--modality      分析模态：fmri / eeg / joint（默认 fmri）
--n-init        每个图文件的初始轨迹数（默认 20）
--steps         每条轨迹的预测步数（默认 200）
--seed          随机种子（默认 42）
--pca-burnin    丢弃的初始帧数，避免瞬态污染 PCA（默认自动 = max(10, steps//10)）
--n-components  计算的 PCA 主成分数（默认 20，覆盖 10D+ 流形）
--max-pts       每条轨迹采样点数上限，防止图表过密（默认 300）
--no-density    关闭密度热图（默认开启）

**参考文献**：
  Stringer et al. (2019) Science — ~10D visual cortex manifold.
  Saxena & Cunningham (2019) Nature Neuroscience — dimensionality in neural data.
  Gallego et al. (2020) Nature Neuroscience — motor cortex manifold.
  Bjorck & Golub (1973) Math. Comp. — principal angles between subspaces.
  Williams et al. (2021) NeurIPS — shape metrics on neural representations.
  Cunningham & Yu (2014) Nature Neuroscience — dimensionality reduction.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_path() -> None:
    """Add the brain_dynamics root to sys.path so submodules resolve."""
    _here = Path(__file__).resolve().parent.parent  # brain_dynamics/
    if str(_here) not in sys.path:
        sys.path.insert(0, str(_here))


_ensure_path()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint / graph-cache file filtering
# ─────────────────────────────────────────────────────────────────────────────

_CHECKPOINT_STEMS: frozenset = frozenset({"best_model", "swa_model"})
_CHECKPOINT_PREFIX: str = "checkpoint_epoch_"


def _find_graph_pts(folder: Path) -> List[Path]:
    """Return sorted list of graph-cache ``.pt`` files (training checkpoints excluded)."""
    result: List[Path] = []
    for f in sorted(folder.glob("*.pt")):
        stem = f.stem
        if stem not in _CHECKPOINT_STEMS and not stem.startswith(_CHECKPOINT_PREFIX):
            result.append(f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Core function: generate trajectories for one graph
# ─────────────────────────────────────────────────────────────────────────────

def _trajectories_for_graph(
    graph_path: Path,
    model,
    modality: str,
    n_init: int,
    steps: int,
    seed: int,
    device: str,
) -> Optional[np.ndarray]:
    """Load one graph, build a simulator, run free dynamics using this graph's context only.

    Parameters
    ----------
    graph_path:
        Path to the graph cache file representing one task / run / subject.
        Its BOLD history is used **exclusively** as the initial context for
        all ``n_init`` rollouts.

    Design rationale
    ----------------
    For cross-task PCA the goal is to compare whether different tasks produce
    trajectories that occupy the same manifold.  This comparison is only
    meaningful when each task's trajectories are generated from **that task's
    own BOLD history** — not from a pool of other tasks' BOLD data.

    Mixing another task's BOLD as initial context would contaminate the
    per-task signal: trajectories for task A initialised with task B's context
    no longer represent task A's intrinsic dynamics and cannot be used to
    characterise task A's attractor geometry.

    Therefore ``graph_paths=None`` is passed to ``run_free_dynamics``,
    enforcing single-graph (task-specific) context throughout.

    Returns ``trajectories`` of shape ``(n_init, steps, n_regions)`` or
    ``None`` if loading / simulation failed.
    """
    try:
        from loader.load_model import load_graph_for_inference
        from simulator.brain_dynamics_simulator import BrainDynamicsSimulator
        from experiments.free_dynamics import run_free_dynamics

        graph = load_graph_for_inference(graph_path, device=device)
        sim = BrainDynamicsSimulator(model, graph, modality=modality, device=device)

        # Single-graph mode (graph_paths=None): use only this task's own BOLD
        # history as initial context.  No cross-task context mixing.
        trajs = run_free_dynamics(sim, n_init=n_init, steps=steps, seed=seed,
                                  graph_paths=None)
        logger.info(
            "  [%s] shape=%s, init_std=%.4f, final_std=%.4f",
            graph_path.name, trajs.shape,
            float(np.std(trajs[:, 0, :], axis=0).mean()),
            float(np.std(trajs[:, -1, :], axis=0).mean()),
        )
        return trajs
    except Exception as exc:
        logger.warning("  [%s] 跳过（%s）", graph_path.name, exc, exc_info=True)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Joint PCA helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_joint_pca(
    all_trajs: np.ndarray,
    burnin: int,
    n_components: int = 20,
) -> Tuple["PCA", np.ndarray, np.ndarray]:
    """Fit PCA on **all** trajectories (all tasks pooled).

    Parameters
    ----------
    all_trajs:
        Shape ``(n_traj_total, steps, n_regions)``.
    burnin:
        Number of initial frames to discard from each trajectory.
    n_components:
        PCA components to compute.  Default is 20 to properly cover a
        10-dimensional neural manifold (Stringer et al. 2019; Saxena &
        Cunningham 2019).  Using only 10 components would undercount the
        true manifold dimensionality, making participation-ratio and
        principal-angle estimates biased.

    Returns
    -------
    pca:       Fitted sklearn.PCA object.
    X_pca:     Projected data ``(n_samples, n_components)``.
    traj_idx:  Index identifying which trajectory each sample belongs to.
    """
    from sklearn.decomposition import PCA as SkPCA

    n_traj, T, N = all_trajs.shape
    # Remove burnin
    data = all_trajs[:, burnin:, :]          # (n_traj, T_after, N)
    T_after = data.shape[1]
    X = data.reshape(n_traj * T_after, N).astype(np.float64)
    X -= X.mean(axis=0)                       # centre once across all tasks

    n_comp = min(n_components, X.shape[0] - 1, X.shape[1] - 1)
    if n_comp < 2:
        n_samples = X.shape[0]
        n_features = X.shape[1]
        raise ValueError(
            f"PCA 拟合失败：可用主成分数 n_comp={n_comp} < 2。\n"
            f"  n_samples = n_traj({n_traj}) × (T({T}) - burnin({burnin})) = {n_samples}\n"
            f"  n_features = n_regions = {n_features}\n"
            f"  n_comp = min(n_components={n_components}, n_samples-1={n_samples-1}, "
            f"n_features-1={n_features-1}) = {n_comp}\n"
            "请增加 n_init 或减少 pca_burnin，以确保 PCA 可用至少 2 个主成分。"
        )
    pca = SkPCA(n_components=n_comp, random_state=42)
    X_pca = pca.fit_transform(X)             # (n_traj*T_after, n_comp)

    traj_idx = np.repeat(np.arange(n_traj), T_after)  # which trajectory each row belongs to
    return pca, X_pca, traj_idx


# ─────────────────────────────────────────────────────────────────────────────
# Manifold analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ring_score(X2d: np.ndarray) -> float:
    """Heuristic 'ring score': ratio of distance-from-centre std / mean.

    A perfect ring has points at nearly constant distance from its centroid
    (score ≈ 0).  A filled disk has high std (score ≈ 0.3–0.5).
    A 1-D line has score ≈ 0.5+.

    Returns
    -------
    float in [0, ∞); lower = more ring-like.
    """
    centroid = X2d.mean(axis=0)
    dists = np.linalg.norm(X2d - centroid, axis=1)
    return float(dists.std() / (dists.mean() + 1e-10))


def _phase_offset(X2d_a: np.ndarray, X2d_b: np.ndarray) -> float:
    """Estimate the mean angle between two point clouds on the PC1–PC2 plane.

    Both clouds are centred at the joint centroid.  The 'phase offset' is the
    difference in their mean polar angles (in degrees).  If both tasks lie on
    the same ring, they should have a well-defined offset ≠ 0.

    .. note::
        This is a **2D-only** metric, valid only when most manifold variance
        is in PC1–PC2.  For high-dimensional (≥10D) manifolds it is
        unreliable; prefer the principal-angles and k-NN purity metrics
        computed in the full high-D PCA space.

    Returns
    -------
    float: mean angular difference in degrees ∈ [0, 180].
    """
    # Require at least 2 dimensions for angle calculation.
    if X2d_a.ndim < 2 or X2d_a.shape[1] < 2 or X2d_b.ndim < 2 or X2d_b.shape[1] < 2:
        return float("nan")
    if len(X2d_a) == 0 or len(X2d_b) == 0:
        return float("nan")
    joint = np.vstack([X2d_a, X2d_b])
    c = joint.mean(axis=0)
    # Compute mean direction vector for each task cloud (relative to joint centroid).
    # np.arctan2(y, x) expects (y, x); mean(axis=0) gives [pc1_mean, pc2_mean].
    # We reverse to [pc2_mean, pc1_mean] so the unpack gives arctan2(y=pc2, x=pc1).
    mean_a = (X2d_a - c).mean(axis=0)   # [pc1_mean, pc2_mean]
    mean_b = (X2d_b - c).mean(axis=0)
    angle_a = float(np.arctan2(mean_a[1], mean_a[0]) * 180 / np.pi)  # y=PC2, x=PC1
    angle_b = float(np.arctan2(mean_b[1], mean_b[0]) * 180 / np.pi)
    diff = abs(angle_a - angle_b)
    return min(diff, 360.0 - diff)


# ─────────────────────────────────────────────────────────────────────────────
# High-dimensional manifold analysis helpers (10D+ aware)
# ─────────────────────────────────────────────────────────────────────────────

def _participation_ratio(eigenvalues: np.ndarray) -> float:
    """Participation Ratio (PR) — robust intrinsic dimensionality estimate.

    PR = (Σλ_i)² / Σλ_i²

    For a perfectly k-dimensional subspace, PR = k.
    For a neural manifold with PR ≈ 10, the effective dimensionality is ~10.
    This is a well-calibrated estimator that does not require choosing a
    variance threshold and is unaffected by the number of components computed.

    References
    ----------
    Cunningham & Yu (2014) Nature Neuroscience — dimensionality in neural data.
    Litwin-Kumar et al. (2017) Neuron — synaptic connectivity and dimensionality.
    Abbott et al. (2011) Current Opinion — neural manifolds.
    """
    ev = np.asarray(eigenvalues, dtype=np.float64)
    ev = ev[ev > 0]
    if len(ev) == 0:
        return 0.0
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def _task_pca_basis(trajs: np.ndarray, burnin: int, n_components: int) -> Optional[np.ndarray]:
    """Fit PCA on a single task's trajectories; return orthonormal basis (n_comp × N).

    Parameters
    ----------
    trajs:        shape ``(n_traj, T, N)``.
    burnin:       initial frames to discard.
    n_components: number of principal components to retain.

    Returns
    -------
    basis : ndarray of shape ``(n_comp, N)`` or ``None``.
        Returns ``None`` when ``T - burnin <= 0`` (no frames remain after burnin).
        When effective samples < ``n_components + 1``, returns a basis with
        fewer components than requested (capped at ``min(samples-1, N-1)``).
    """
    from sklearn.decomposition import PCA as SkPCA

    n_traj, T, N = trajs.shape
    data = trajs[:, burnin:, :].reshape(-1, N).astype(np.float64)
    if data.shape[0] == 0:
        return None
    data -= data.mean(axis=0)
    nc = min(n_components, data.shape[0] - 1, data.shape[1] - 1)
    if nc < 1:
        return None
    pca = SkPCA(n_components=nc, random_state=42)
    pca.fit(data)
    return pca.components_  # (nc, N) orthonormal rows


def _grassmannian_distance(basis_a: np.ndarray, basis_b: np.ndarray) -> Dict:
    """Grassmannian distance between two subspaces via principal angles.

    Principal angles θ_k satisfy cos(θ_k) = σ_k(A @ B.T) where A, B are
    orthonormal bases and σ_k are singular values.

    - All angles ≈ 0° → subspaces are identical (tasks share the same manifold).
    - All angles ≈ 90° → subspaces are orthogonal (fully task-specific manifolds).
    - Geodesic distance = sqrt(Σ θ_k²) (Grassmannian RMS).

    This is the geometrically correct way to compare manifolds in 10D+;
    it captures differences in *all* principal directions simultaneously.

    References
    ----------
    Bjorck & Golub (1973) Math. Comp. — numerical principal angles.
    Williams et al. (2021) NeurIPS — shape metrics on neural representations.
    Gallego et al. (2018) Nature Neuroscience — motor cortex manifold comparison.
    """
    nc = min(basis_a.shape[0], basis_b.shape[0])
    A = basis_a[:nc]  # (nc, N)
    B = basis_b[:nc]  # (nc, N)
    M = A @ B.T       # (nc, nc)
    sv = np.linalg.svd(M, compute_uv=False)
    sv = np.clip(sv, -1.0, 1.0)      # numerical safety
    angles_deg = np.degrees(np.arccos(sv))
    return {
        "principal_angles_deg": angles_deg.tolist(),
        "mean_principal_angle_deg": float(angles_deg.mean()),
        "max_principal_angle_deg": float(angles_deg.max()),
        # RMS distance on the Grassmannian: sqrt(mean(θ²))
        # This is the chordal distance normalised by the number of components.
        "grassmannian_distance_deg": float(np.sqrt((angles_deg ** 2).mean())),
        "n_components": nc,
    }


def _knn_task_purity(
    X_pca: np.ndarray,
    sample_task: np.ndarray,
    k: int = 15,
    n_dims: Optional[int] = None,
) -> Dict:
    """k-NN task purity in high-dimensional PCA space.

    For each point, computes the fraction of its k nearest neighbours
    belonging to the **same** task.  Compares against the chance level
    (1 / n_tasks).

    - Purity ≈ chance  → tasks are well-mixed on a shared manifold.
    - Purity >> chance → tasks occupy distinct regions (separate manifolds).

    This metric is *dimension-aware*: it operates in the top-``n_dims``
    PCA dimensions, not just PC1–PC2.  For a 10D manifold, ``n_dims``
    should be ≥ 10 (typically set to n90 or the participation ratio).

    References
    ----------
    Kriegeskorte et al. (2008) Neuron — representational geometry.
    Kornblith et al. (2019) ICML — CKA and nearest-neighbour metrics.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        logger.warning("sklearn not available; skipping knn_task_purity.")
        return {}

    n_tasks = int(sample_task.max()) + 1 if len(sample_task) > 0 else 1
    chance = 1.0 / max(n_tasks, 1)
    dims = n_dims if n_dims is not None else X_pca.shape[1]
    dims = min(dims, X_pca.shape[1])
    X = X_pca[:, :dims].astype(np.float64)

    # Subsample for speed
    n_max = 2000
    rng = np.random.default_rng(0)
    if len(X) > n_max:
        idx = rng.choice(len(X), n_max, replace=False)
        X_sub = X[idx]
        task_sub = sample_task[idx]
    else:
        X_sub = X
        task_sub = sample_task

    n = len(X_sub)
    k_eff = min(k, n - 1)
    if k_eff < 1:
        return {}

    nn = NearestNeighbors(n_neighbors=k_eff + 1, algorithm="auto", metric="euclidean")
    nn.fit(X_sub)
    indices = nn.kneighbors(X_sub, return_distance=False)[:, 1:]  # exclude self

    same_frac = np.array([
        float(np.mean(task_sub[indices[i]] == task_sub[i]))
        for i in range(n)
    ])
    mean_purity = float(same_frac.mean())
    purity_ratio = float(mean_purity / max(chance, 1e-9))
    return {
        "mean_purity": mean_purity,
        "chance_purity": float(chance),
        "purity_ratio": purity_ratio,
        "n_dims_used": dims,
        "k": k_eff,
        "interpretation": (
            "purity_ratio < 1.5 → tasks well-mixed on shared manifold; "
            "1.5–2.5 → partially shared; >2.5 → distinct task manifolds."
        ),
    }


def _between_within_variance(
    X_pca: np.ndarray,
    sample_task: np.ndarray,
    n_dims: Optional[int] = None,
) -> Dict:
    """Between-task vs. within-task variance ratio in high-D PCA space.

    A high ratio → tasks occupy distinct regions (separate manifolds).
    A low ratio  → tasks share the same manifold region.

    Uses the first ``n_dims`` PCA dimensions (reflecting the true manifold
    dimensionality, not just PC1–PC2).

    References
    ----------
    Bishop (2006) PRML — between/within-class scatter analysis.
    """
    dims = n_dims if n_dims is not None else X_pca.shape[1]
    dims = min(dims, X_pca.shape[1])
    X = X_pca[:, :dims].astype(np.float64)

    task_ids = np.unique(sample_task)
    n_tasks = len(task_ids)
    if n_tasks < 2:
        return {}

    global_mean = X.mean(axis=0)

    task_centroids = np.array([
        X[sample_task == t].mean(axis=0) for t in task_ids
    ])
    # Between-task: mean squared distance of task centroids from global centroid
    between_var = float(np.mean(
        np.sum((task_centroids - global_mean) ** 2, axis=1)
    ))
    # Within-task: mean squared distance of points from their task centroid
    within_var = float(np.mean([
        np.mean(np.sum(
            (X[sample_task == t] - X[sample_task == t].mean(axis=0)) ** 2, axis=1
        ))
        for t in task_ids
    ]))

    ratio = float(between_var / (within_var + 1e-10))
    return {
        "between_var": between_var,
        "within_var": within_var,
        "between_within_ratio": ratio,
        "n_dims_used": dims,
        "interpretation": (
            "ratio < 0.1 → tasks strongly overlapping (shared manifold); "
            "0.1–0.5 → partial overlap; >0.5 → well-separated tasks."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _plot_joint_pca(
    X_pca: np.ndarray,
    traj_idx: np.ndarray,
    task_labels: List[str],
    pca,
    output_path: Path,
    max_pts: int = 300,
    show_density: bool = True,
) -> None:
    """Two-panel figure: density heat-map + per-task scatter in PC1-PC2 space."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except Exception:
            pass
    except ImportError:
        logger.warning("matplotlib not available; skipping PCA plot.")
        return

    cmap_list = plt.get_cmap("tab20").colors  # 20 distinct colours
    n_tasks = len(task_labels)

    ev_ratio = pca.explained_variance_ratio_
    pc1_var = float(ev_ratio[0] * 100) if len(ev_ratio) > 0 else 0.0
    pc2_var = float(ev_ratio[1] * 100) if len(ev_ratio) > 1 else 0.0

    n_panels = 4 if show_density else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle(
        "Cross-Task PCA — High-Dimensional Manifold Analysis\n"
        f"PC1={pc1_var:.1f}%  PC2={pc2_var:.1f}%  "
        "(joint PCA across all tasks; see pc_grid.png for PC3–PC8)",
        fontsize=10,
    )

    ax_scatter  = axes[0]
    ax_pc34     = axes[1]
    ax_time     = axes[2]
    ax_density  = axes[3] if show_density else None

    def _scatter_tasks(ax, col_a, col_b, xlabel, ylabel, title):
        """Helper: scatter all tasks using columns col_a, col_b of X_pca."""
        if X_pca.shape[1] <= max(col_a, col_b):
            ax.text(0.5, 0.5, "Not enough components",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        for task_i in range(n_tasks):
            mask = traj_idx == task_i
            pts = X_pca[mask][:, [col_a, col_b]]
            if len(pts) == 0:
                continue
            if len(pts) > max_pts:
                rng_idx = np.random.default_rng(42 + task_i).choice(
                    len(pts), max_pts, replace=False)
                pts = pts[rng_idx]
            color = cmap_list[task_i % len(cmap_list)]
            ax.scatter(
                pts[:, 0], pts[:, 1],
                s=4, alpha=0.35, color=color,
                label=task_labels[task_i] if len(task_labels[task_i]) <= 20
                else task_labels[task_i][:18] + "..",
                rasterized=True,
            )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax.axvline(0, color="k", lw=0.5, alpha=0.3)
        ax.grid(True, alpha=0.2)

    # ── Panel 0: PC1–PC2 scatter, all tasks overlaid ────────────────────────
    _scatter_tasks(
        ax_scatter, 0, 1,
        xlabel=f"PC1 ({pc1_var:.1f}%)",
        ylabel=f"PC2 ({pc2_var:.1f}%)",
        title="PC1 vs PC2 (all tasks)\n(Note: high-D manifold may look diffuse in 2D)",
    )
    ax_scatter.legend(markerscale=3, fontsize=7, ncol=max(1, n_tasks // 8),
                      loc="best", framealpha=0.7)

    # ── Panel 1: PC3–PC4 scatter ─────────────────────────────────────────────
    pc3_var = float(ev_ratio[2] * 100) if len(ev_ratio) > 2 else 0.0
    pc4_var = float(ev_ratio[3] * 100) if len(ev_ratio) > 3 else 0.0
    _scatter_tasks(
        ax_pc34, 2, 3,
        xlabel=f"PC3 ({pc3_var:.1f}%)",
        ylabel=f"PC4 ({pc4_var:.1f}%)",
        title="PC3 vs PC4 (all tasks)\n(high-D structure: compare with PC1-PC2 panel)",
    )

    # ── Panel 2: time-coloured scatter ───────────────────────────────────────
    sc = None
    for task_i in range(min(n_tasks, 5)):
        mask = traj_idx == task_i
        pts = X_pca[mask, :2]
        T_pts = len(pts)
        if T_pts == 0 or pts.shape[1] < 2:
            continue
        sc = ax_time.scatter(
            pts[:, 0], pts[:, 1],
            c=np.linspace(0, 1, T_pts),
            cmap="viridis", s=3, alpha=0.25,
            rasterized=True,
        )
    if n_tasks > 0 and sc is not None:
        plt.colorbar(sc, ax=ax_time, label="Time (0=start, 1=end)")
    ax_time.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax_time.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax_time.set_title(
        "Trajectories coloured by time (PC1-PC2)\n(first 5 tasks; circular = ring structure)"
    )
    ax_time.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_time.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax_time.grid(True, alpha=0.2)

    # ── Panel 3: joint density heat-map ──────────────────────────────────────
    if ax_density is not None:
        all_pts = X_pca[:, :2]
        ax_density.hexbin(
            all_pts[:, 0], all_pts[:, 1],
            gridsize=40, cmap="YlOrRd", mincnt=1,
        )
        ax_density.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        ax_density.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
        ax_density.set_title(
            "Joint density heat-map (all tasks, PC1-PC2)\n"
            "(ring-shaped = circular attractor in this projection)"
        )
        ax_density.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_density.axvline(0, color="k", lw=0.5, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → PCA figure saved: %s", output_path)


def _plot_pc_grid(
    X_pca: np.ndarray,
    traj_idx: np.ndarray,
    task_labels: List[str],
    pca,
    output_path: Path,
    max_pts: int = 300,
) -> None:
    """2×4 grid of PC-pair scatter plots: (PC1-2), (PC3-4), (PC5-6), (PC7-8).

    For a high-dimensional (10D+) manifold, structure that is invisible in
    PC1–PC2 alone can be clearly seen in the higher PC pairs.  This plot
    lets the analyst judge whether task separation / manifold geometry is
    consistent across many dimensions.

    References
    ----------
    Stringer et al. (2019) Science — Fig. 1 multi-PC visualisation.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except Exception:
            pass
    except ImportError:
        logger.warning("matplotlib not available; skipping PC grid plot.")
        return

    n_comp = X_pca.shape[1]
    cmap_list = plt.get_cmap("tab20").colors
    n_tasks = len(task_labels)
    ev_ratio = pca.explained_variance_ratio_

    # PC pairs to show: (0,1), (2,3), (4,5), (6,7) — up to available components
    pairs = [(2 * i, 2 * i + 1) for i in range(4) if 2 * i + 1 < n_comp]
    n_pairs = len(pairs)
    if n_pairs == 0:
        logger.warning("Not enough PCA components for PC grid plot; skipping.")
        return

    n_cols = min(n_pairs, 4)
    n_rows = 2  # row 0: task-coloured; row 1: time-coloured

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                              squeeze=False)
    fig.suptitle(
        "Multi-PC-Pair View — High-Dimensional Manifold Structure\n"
        "(row 1: coloured by task; row 2: coloured by time — first task shown)",
        fontsize=10,
    )

    for col, (ca, cb) in enumerate(pairs):
        pct_a = float(ev_ratio[ca] * 100) if ca < len(ev_ratio) else 0.0
        pct_b = float(ev_ratio[cb] * 100) if cb < len(ev_ratio) else 0.0
        xlabel = f"PC{ca + 1} ({pct_a:.1f}%)"
        ylabel = f"PC{cb + 1} ({pct_b:.1f}%)"

        ax_top = axes[0][col]
        ax_bot = axes[1][col]

        # Row 0: task-coloured scatter
        for task_i in range(n_tasks):
            mask = traj_idx == task_i
            pts = X_pca[mask][:, [ca, cb]]
            if len(pts) == 0:
                continue
            if len(pts) > max_pts:
                idx = np.random.default_rng(42 + task_i + col).choice(
                    len(pts), max_pts, replace=False)
                pts = pts[idx]
            color = cmap_list[task_i % len(cmap_list)]
            lbl = task_labels[task_i] if col == 0 else "_nolegend_"
            lbl = (lbl if len(lbl) <= 18 else lbl[:16] + "..") if lbl != "_nolegend_" else lbl
            ax_top.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.3,
                           color=color, label=lbl, rasterized=True)
        ax_top.set_xlabel(xlabel)
        ax_top.set_ylabel(ylabel)
        ax_top.set_title(f"PC{ca+1} vs PC{cb+1} (tasks)")
        ax_top.grid(True, alpha=0.2)
        if col == 0:
            ax_top.legend(markerscale=3, fontsize=6, ncol=max(1, n_tasks // 6),
                          loc="best", framealpha=0.6)

        # Row 1: time-coloured (first task only to keep legible)
        sc = None
        mask0 = traj_idx == 0
        pts0 = X_pca[mask0][:, [ca, cb]]
        if len(pts0) > 0:
            sc = ax_bot.scatter(
                pts0[:, 0], pts0[:, 1],
                c=np.linspace(0, 1, len(pts0)),
                cmap="viridis", s=3, alpha=0.3, rasterized=True,
            )
            if col == n_cols - 1 and sc is not None:
                plt.colorbar(sc, ax=ax_bot, label="Time")
        ax_bot.set_xlabel(xlabel)
        ax_bot.set_ylabel(ylabel)
        ax_bot.set_title(f"PC{ca+1} vs PC{cb+1} (time; task 0)")
        ax_bot.grid(True, alpha=0.2)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → PC grid figure saved: %s", output_path)


def _plot_variance_curve(pca, output_path: Path, participation_ratio: float = 0.0) -> None:
    """Scree plot of cumulative explained variance with Participation Ratio annotation."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    evr = pca.explained_variance_ratio_
    cumvar = np.cumsum(evr)
    n = len(evr)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # ── Left: per-PC variance bar chart with PR vertical line ─────────────────
    axes[0].bar(np.arange(1, n + 1), evr * 100, color="steelblue", alpha=0.7)
    if participation_ratio > 0:
        axes[0].axvline(participation_ratio, color="crimson", linestyle="--",
                        linewidth=1.5, label=f"PR = {participation_ratio:.1f} (intrinsic dim)")
        axes[0].legend(fontsize=8)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Per-PC variance (joint PCA)\nCrimson line = Participation Ratio")
    axes[0].grid(True, axis="y", alpha=0.3)

    # ── Right: cumulative variance with PR and threshold lines ────────────────
    axes[1].plot(np.arange(1, n + 1), cumvar * 100, "o-", color="steelblue")
    axes[1].axhline(90, color="salmon", linestyle="--", label="90%")
    axes[1].axhline(50, color="orange", linestyle="--", label="50%")
    if participation_ratio > 0:
        axes[1].axvline(participation_ratio, color="crimson", linestyle="--",
                        linewidth=1.5, label=f"PR = {participation_ratio:.1f}")
    axes[1].set_xlabel("Principal components (top k)")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title(
        "Cumulative explained variance\n"
        f"Participation Ratio = {participation_ratio:.1f} (effective manifold dim)"
        if participation_ratio > 0 else "Cumulative explained variance"
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → Variance curve saved: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_task_pca(
    model_path: Path,
    graph_folder: Path,
    output_dir: Path,
    modality: str = "fmri",
    n_init: int = 20,
    steps: int = 200,
    seed: int = 42,
    pca_burnin: Optional[int] = None,
    n_components: int = 20,
    max_pts: int = 300,
    show_density: bool = True,
    device: str = "cpu",
) -> Dict:
    """Run cross-task PCA projection analysis with high-dimensional manifold metrics.

    Parameters
    ----------
    model_path:
        Path to the trained model checkpoint (``best_model.pt``).
    graph_folder:
        Folder containing graph-cache ``.pt`` files.
    output_dir:
        Where to save outputs (figures, JSON, ``.npy`` trajectory caches).
    modality:
        Modality to analyse (``'fmri'``, ``'eeg'``, or ``'joint'``).
    n_init:
        Number of free-dynamics trajectories per graph file.
    steps:
        Prediction steps per trajectory.
    seed:
        Random seed.
    pca_burnin:
        Frames to discard from each trajectory before PCA.  ``None`` →
        ``max(10, steps // 10)``.
    n_components:
        Number of PCA components to compute.  Default is 20 to cover the
        ~10-dimensional neural manifold documented in recent literature
        (Stringer et al. 2019; Saxena & Cunningham 2019).  With only 10
        components the participation-ratio estimate is biased low.
    max_pts:
        Maximum number of scatter points per task in the figure.
    show_density:
        Whether to include the joint density heat-map panel.
    device:
        Compute device (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    dict with keys ``tasks``, ``pca_metrics``, ``manifold_analysis``.

    Notes
    -----
    **Why the old PC1-PC2 ring score is unreliable for 10D+ manifolds**

    If the neural manifold occupies ~10 dimensions, PC1 and PC2 together
    typically explain only 20–40% of the variance.  A circular attractor in
    10D will project to a filled disk in PC1–PC2 (diffuse scatter), not a
    ring.  Conversely, a ring-shaped projection can arise from many non-ring
    geometries in higher dimensions.  Therefore the ring-score heuristic is
    retained as a *legacy* 2D indicator but is no longer the primary metric.

    The primary high-D metrics are:
      - **Participation Ratio (PR)**: intrinsic dimensionality from eigenvalues.
      - **Grassmannian principal angles**: geometric distance between
        per-task PCA subspaces (gold standard for subspace comparison).
      - **k-NN task purity**: fraction of same-task nearest neighbours in
        the full high-D PCA space.
      - **Between/within variance ratio**: ANOVA-like separation in high-D.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graph_folder = Path(graph_folder)
    graph_paths = _find_graph_pts(graph_folder)
    if not graph_paths:
        raise FileNotFoundError(
            f"在 {graph_folder} 中未找到图缓存 .pt 文件。\n"
            "请确认路径正确，且文件不是训练检查点（best_model.pt 等）。"
        )
    logger.info("发现 %d 个图缓存文件：%s", len(graph_paths), [p.name for p in graph_paths])

    # Load model
    from loader.load_model import load_trained_model
    logger.info("加载模型: %s", model_path)
    model = load_trained_model(model_path, device=device)
    # Note: load_trained_model already returns the twin with model.eval() applied
    # internally (TwinBrainDigitalTwin.__init__ calls self.model.eval()).
    # Calling .eval() on the TwinBrainDigitalTwin wrapper itself is a no-op error
    # because TwinBrainDigitalTwin is not an nn.Module.

    if pca_burnin is None:
        pca_burnin = max(10, steps // 10)

    # ── Generate trajectories for each graph ──────────────────────────────────
    task_trajs: Dict[str, np.ndarray] = {}
    task_labels: List[str] = []

    for gp in graph_paths:
        label = gp.stem
        trajs = _trajectories_for_graph(
            gp, model, modality, n_init, steps, seed, device,
        )
        if trajs is None:
            continue
        task_trajs[label] = trajs
        task_labels.append(label)

        # Cache trajectory array
        traj_path = output_dir / f"trajectories_{label}.npy"
        np.save(str(traj_path), trajs)
        logger.debug("  Saved %s", traj_path)

    if not task_trajs:
        raise RuntimeError("所有图文件的轨迹生成均失败，请检查上方日志。")

    logger.info("成功生成轨迹的任务数: %d / %d", len(task_trajs), len(graph_paths))

    # ── n_regions consistency check ───────────────────────────────────────────
    # Different graph files may have different numbers of nodes (e.g. some
    # subjects have fewer parcellated regions after QC).  Trajectories with
    # mismatched n_regions cannot be concatenated; filter them out here so that
    # np.concatenate below never raises a shape-mismatch ValueError.
    if len(task_trajs) > 1:
        ref_label = task_labels[0]
        ref_n = task_trajs[ref_label].shape[2]
        mismatched = [lbl for lbl in task_labels if task_trajs[lbl].shape[2] != ref_n]
        if mismatched:
            mismatch_info = {lbl: task_trajs[lbl].shape[2] for lbl in mismatched}
            logger.warning(
                "%d 个任务的 n_regions 与参考任务 '%s'（n_regions=%d）不一致，已跳过：%s",
                len(mismatched), ref_label, ref_n,
                mismatch_info,
            )
            for lbl in mismatched:
                task_trajs.pop(lbl)
                task_labels.remove(lbl)
            if not task_trajs:
                raise RuntimeError(
                    "过滤 n_regions 不一致的任务后无任务剩余。"
                    "请检查图缓存文件是否使用了同一套脑区图谱（atlas）。"
                )

    # ── Joint PCA ─────────────────────────────────────────────────────────────
    # Stack all trajectories into a single pool for joint PCA fitting.
    # Order: task_labels[0] traj 0..n_init-1, task_labels[1] traj 0..n_init-1, …
    all_trajs_list = [task_trajs[lbl] for lbl in task_labels]
    all_trajs = np.concatenate(all_trajs_list, axis=0)   # (n_tasks*n_init, steps, N)
    n_trajs_per_task = [len(task_trajs[lbl]) for lbl in task_labels]

    logger.info(
        "合并轨迹形状: %s  (pca_burnin=%d，有效帧数=%d，n_components=%d)",
        all_trajs.shape, pca_burnin, max(0, steps - pca_burnin), n_components,
    )

    pca, X_pca, traj_idx = _fit_joint_pca(
        all_trajs, burnin=pca_burnin, n_components=n_components,
    )

    # Rebuild task-level traj_idx (tasks are stacked in order)
    traj_to_task = np.empty(sum(n_trajs_per_task), dtype=np.int32)
    start = 0
    for ti, cnt in enumerate(n_trajs_per_task):
        traj_to_task[start: start + cnt] = ti
        start += cnt

    # Expand traj_idx (over trajectories) to sample-level task assignment
    T_after = max(0, steps - pca_burnin)
    sample_task = np.repeat(traj_to_task, T_after) if T_after > 0 else np.array([], dtype=np.int32)

    # ── PCA metrics ──────────────────────────────────────────────────────────
    evr = pca.explained_variance_ratio_
    ev  = pca.explained_variance_          # per-component variance (eigenvalues of covariance matrix)
    cumvar = np.cumsum(evr)
    n50 = int(np.searchsorted(cumvar, 0.50) + 1)
    n80 = int(np.searchsorted(cumvar, 0.80) + 1)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)

    # Participation Ratio — primary intrinsic dimensionality estimate
    pr = _participation_ratio(ev)

    pca_metrics = {
        "n_tasks": len(task_labels),
        "n_traj_per_task": n_init,
        "steps": steps,
        "pca_burnin": pca_burnin,
        "n_components_computed": len(evr),
        "variance_pc1": float(evr[0]) if len(evr) > 0 else None,
        "variance_pc2": float(evr[1]) if len(evr) > 1 else None,
        "variance_top2": float(cumvar[1]) if len(cumvar) > 1 else None,
        "n_components_50pct": n50,
        "n_components_80pct": n80,
        "n_components_90pct": n90,
        # High-D metrics (new)
        "participation_ratio": float(pr),
        "note_pr": (
            "PR = (Σλ)²/Σλ² = effective intrinsic dimensionality. "
            "PR≈10 confirms a ~10D manifold (Cunningham & Yu 2014). "
            "Variance-threshold counts (n50/n80/n90) are sensitive to noise; "
            "PR is robust."
        ),
    }
    logger.info(
        "PCA 维度分析：PR=%.1f (内在维度)  n90=%d  PC1=%.1f%%  PC2=%.1f%%",
        pr, n90,
        float(evr[0] * 100) if len(evr) > 0 else 0.0,
        float(evr[1] * 100) if len(evr) > 1 else 0.0,
    )

    # ── High-D manifold analysis ──────────────────────────────────────────────
    # Number of dimensions to use for high-D metrics.
    # We use n90 (number of components explaining 90% variance) rather than
    # int(PR) because: (a) n90 is directly interpretable as "the subspace
    # that accounts for 90% of the dynamics"; (b) PR can be noisy when the
    # eigenvalue spectrum is not perfectly flat within the manifold; (c) n90
    # always satisfies n90 >= PR in practice, so it is a conservative upper
    # bound that includes all informative dimensions without adding pure noise.
    # The 90% threshold (vs 80% or 95%) is chosen as the standard "elbow" in
    # scree-plot analysis; it captures most signal directions while excluding
    # the long tail of noise PCs.  It is configurable via the data indirectly
    # (larger n_components covers more of the spectrum, shifting n90 up).
    hd_dims = min(n90, X_pca.shape[1])

    # k-NN task purity in high-D PCA space
    knn_purity = _knn_task_purity(
        X_pca, sample_task, k=15, n_dims=hd_dims,
    ) if T_after > 0 else {}

    # Between/within variance ratio in high-D PCA space
    bw_var = _between_within_variance(
        X_pca, sample_task, n_dims=hd_dims,
    ) if T_after > 0 else {}

    # Per-task and pairwise analyses (legacy 2D + new high-D subspace)
    task_analysis = {}
    for ti, lbl in enumerate(task_labels):
        mask_t = sample_task == ti
        if mask_t.sum() == 0:
            continue
        pts2d = X_pca[mask_t, :2]
        ring_s = _ring_score(pts2d)
        centroid = pts2d.mean(axis=0).tolist()
        task_analysis[lbl] = {
            # Legacy 2D ring score (unreliable for 10D+ manifolds; kept for reference)
            "ring_score_pc12_legacy": float(ring_s),
            "centroid_pc1": float(centroid[0]) if len(centroid) > 0 else float("nan"),
            "centroid_pc2": float(centroid[1]) if len(centroid) > 1 else float("nan"),
            "n_samples": int(mask_t.sum()),
        }

    # Phase offsets between task pairs (PC1-PC2 only — legacy metric)
    phase_offsets = {}
    lbls = list(task_analysis.keys())
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            la, lb = lbls[i], lbls[j]
            pts_a = X_pca[sample_task == task_labels.index(la), :2]
            pts_b = X_pca[sample_task == task_labels.index(lb), :2]
            offset = _phase_offset(pts_a, pts_b)
            phase_offsets[f"{la}_vs_{lb}"] = float(offset)

    # Principal angles between task subspaces (high-D, new primary metric)
    principal_angles: Dict[str, Dict] = {}
    if len(task_labels) >= 2:
        pa_n_comp = min(hd_dims, n_components)
        for i in range(len(task_labels)):
            for j in range(i + 1, len(task_labels)):
                la, lb = task_labels[i], task_labels[j]
                basis_a = _task_pca_basis(task_trajs[la], pca_burnin, pa_n_comp)
                basis_b = _task_pca_basis(task_trajs[lb], pca_burnin, pa_n_comp)
                if basis_a is not None and basis_b is not None:
                    pa = _grassmannian_distance(basis_a, basis_b)
                    principal_angles[f"{la}_vs_{lb}"] = pa
                    logger.info(
                        "  %s vs %s: Grassmannian dist=%.1f°  mean_angle=%.1f°",
                        la, lb,
                        pa["grassmannian_distance_deg"],
                        pa["mean_principal_angle_deg"],
                    )

    # ── Compute mean principal angle across all pairs ─────────────────────────
    all_mean_angles = [
        v["mean_principal_angle_deg"]
        for v in principal_angles.values()
        if "mean_principal_angle_deg" in v
    ]
    mean_pa = float(np.mean(all_mean_angles)) if all_mean_angles else float("nan")

    # ── Manifold verdict (high-D aware) ───────────────────────────────────────
    # Global ring score (legacy, PC1-PC2 only — unreliable for 10D+ manifolds)
    all_pts2d = X_pca[:, :2]
    global_ring_score = _ring_score(all_pts2d)
    avg_task_ring = float(np.mean([
        v["ring_score_pc12_legacy"] for v in task_analysis.values()
    ])) if task_analysis else float("nan")

    knn_ratio  = knn_purity.get("purity_ratio", float("nan")) if knn_purity else float("nan")
    bw_ratio   = bw_var.get("between_within_ratio", float("nan")) if bw_var else float("nan")

    # Primary verdict based on high-D metrics (k-NN purity + principal angles)
    # -------------------------------------------------------------------
    # Thresholds are empirically motivated by k-NN geometry and subspace
    # alignment studies (Kriegeskorte et al. 2008 Neuron; Williams et al.
    # 2021 NeurIPS) but not taken verbatim from those papers:
    #   knn_ratio < 1.5 AND mean_pa < 30° → strong evidence for shared manifold
    #   knn_ratio > 2.5 OR  mean_pa > 60° → strong evidence for distinct manifolds
    #   otherwise                          → partially shared / ambiguous
    # For datasets with fewer trajectories, increase n_init before drawing
    # firm conclusions from these thresholds.
    # -------------------------------------------------------------------
    knn_ok  = (not np.isnan(knn_ratio)) and knn_ratio < 1.5
    pa_ok   = (not np.isnan(mean_pa))   and mean_pa < 30.0
    knn_sep = (not np.isnan(knn_ratio)) and knn_ratio > 2.5
    pa_sep  = (not np.isnan(mean_pa))   and mean_pa > 60.0
    # single-task edge-case
    only_one_task = len(task_labels) <= 1

    if only_one_task:
        manifold_verdict = (
            "SINGLE TASK — 仅有一个任务，无跨任务比较。\n"
            f"  内在维度（PR）= {pr:.1f}  n90={n90}  (应 >= 5 才有意义)\n"
            "  增加更多任务的图缓存文件以启用跨任务流形分析。"
        )
    elif knn_ok and (pa_ok or np.isnan(mean_pa)):
        manifold_verdict = (
            "SHARED HIGH-D MANIFOLD — 共享高维流形：\n"
            "  k-NN 纯度接近随机基准（任务轨迹在高维 PCA 空间中高度混合），\n"
            "  任务特异子空间夹角较小（Grassmannian 距离小）。\n"
            "  所有任务使用同一高维吸引子；差异仅为流形上的位置/相位偏移。\n"
            f"  内在维度（PR）= {pr:.1f}  k-NN 纯度比 = {knn_ratio:.2f}  "
            f"平均主角度 = {mean_pa:.1f}°"
        )
    elif knn_sep or pa_sep:
        manifold_verdict = (
            "DISTINCT HIGH-D MANIFOLDS — 独立高维流形：\n"
            "  k-NN 纯度显著高于随机基准（任务轨迹在高维空间中聚集），\n"
            "  或任务特异子空间夹角较大（Grassmannian 距离大）。\n"
            "  不同任务对应不同的吸引子子空间，而非同一流形上的不同点。\n"
            f"  内在维度（PR）= {pr:.1f}  k-NN 纯度比 = {knn_ratio:.2f}  "
            f"平均主角度 = {mean_pa:.1f}°"
        )
    else:
        manifold_verdict = (
            "PARTIALLY SHARED HIGH-D MANIFOLD — 部分共享高维流形：\n"
            "  k-NN 纯度和主角度均处于中间范围，证据不一致或不充分。\n"
            "  可能：(a) 任务共享流形但各占不同区域；\n"
            "         (b) 数据量不足，高维指标估计有噪声；\n"
            "         (c) 多任务中部分任务对共享而其他任务对不共享。\n"
            f"  内在维度（PR）= {pr:.1f}  k-NN 纯度比 = {knn_ratio:.2f}  "
            f"平均主角度 = {mean_pa:.1f}°\n"
            "  建议：增加 n_init 和 steps 以获得更可靠的高维估计。"
        )

    manifold_analysis = {
        # ── High-D metrics (primary, new) ─────────────────────────────────────
        "participation_ratio": float(pr),
        "hd_dims_used": hd_dims,
        "knn_task_purity": knn_purity,
        "between_within_variance": bw_var,
        "principal_angles": principal_angles,
        "mean_principal_angle_deg": float(mean_pa),
        # ── Verdict ───────────────────────────────────────────────────────────
        "verdict": manifold_verdict,
        "interpretation": (
            "Primary metrics (high-D): knn_purity_ratio < 1.5 AND mean_pa < 30° "
            "→ shared manifold; knn_purity_ratio > 2.5 OR mean_pa > 60° "
            "→ distinct manifolds; otherwise → partially shared. "
            "Participation Ratio (PR) estimates intrinsic dimensionality. "
            "Legacy PC1-PC2 ring score is kept for reference but is unreliable "
            "for manifolds with PR > 3 (Stringer et al. 2019)."
        ),
        # ── Legacy 2D metrics (retained for reference) ────────────────────────
        "global_ring_score_pc12_legacy": float(global_ring_score),
        "avg_task_ring_score_pc12_legacy": avg_task_ring,
        "task_metrics": task_analysis,
        "phase_offsets_deg_pc12_legacy": phase_offsets,
    }

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_joint_pca(
        X_pca, sample_task, task_labels, pca,
        output_dir / "cross_task_pca.png",
        max_pts=max_pts,
        show_density=show_density,
    )
    _plot_variance_curve(pca, output_dir / "pca_variance_curve.png",
                         participation_ratio=pr)
    _plot_pc_grid(
        X_pca, sample_task, task_labels, pca,
        output_dir / "pc_grid.png",
        max_pts=max_pts,
    )

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "tasks": task_labels,
        "pca_metrics": pca_metrics,
        "manifold_analysis": manifold_analysis,
        "modality": modality,
        "device": device,
        "model_path": str(model_path),
        "graph_folder": str(graph_folder),
    }

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_json_safe(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            f = float(obj)
            return f if np.isfinite(f) else None
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        return obj

    results_path = output_dir / "cross_task_pca_results.json"
    with open(results_path, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(results), fh, indent=2, ensure_ascii=False)
    logger.info("  → Results saved: %s", results_path)

    # ── Summary log ──────────────────────────────────────────────────────────
    def _fmt(val: float, fmt: str = ".2f", na: str = "N/A") -> str:
        return na if np.isnan(val) else format(val, fmt)

    logger.info(
        "\n══════════════════════════════════════════════\n"
        "跨任务 PCA 分析完成 (%d 个任务)\n"
        "  PC1=%.1f%%  PC2=%.1f%%  合计=%.1f%%\n"
        "  参与率 PR=%.1f（内在维度）  n90=%d\n"
        "  k-NN 纯度比=%s  平均主角度=%s°\n"
        "  组间/组内方差比=%s\n"
        "  结论: %s\n"
        "══════════════════════════════════════════════",
        len(task_labels),
        pca_metrics["variance_pc1"] * 100 if pca_metrics["variance_pc1"] else 0,
        pca_metrics["variance_pc2"] * 100 if pca_metrics["variance_pc2"] else 0,
        (pca_metrics["variance_top2"] or 0) * 100,
        pr, n90,
        _fmt(knn_ratio), _fmt(mean_pa, ".1f"), _fmt(bw_ratio, ".3f"),
        manifold_verdict.split("\n")[0],  # first line only for concise log
    )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.cross_task_pca",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--model", required=True, type=Path,
                   help="训练好的模型检查点路径（best_model.pt）")
    p.add_argument("--graphs", required=True, type=Path,
                   help="包含图缓存 .pt 文件的文件夹路径")
    p.add_argument("--output", required=True, type=Path,
                   help="输出目录（不存在时自动创建）")
    p.add_argument("--modality", default="fmri",
                   choices=["fmri", "eeg", "joint"],
                   help="分析模态（默认 fmri）")
    p.add_argument("--n-init", type=int, default=20,
                   help="每个图文件的初始轨迹数（默认 20）")
    p.add_argument("--steps", type=int, default=200,
                   help="每条轨迹的预测步数（默认 200）")
    p.add_argument("--seed", type=int, default=42,
                   help="随机种子（默认 42）")
    p.add_argument("--pca-burnin", type=int, default=None,
                   help="丢弃的初始帧数（默认 max(10, steps//10)）")
    p.add_argument("--n-components", type=int, default=20,
                   help=(
                       "计算的 PCA 主成分数（默认 20，覆盖 ~10D 神经流形）。"
                       "建议设置为预期流形维度的 2× 以确保参与率 (PR) 估计准确。"
                   ))
    p.add_argument("--max-pts", type=int, default=300,
                   help="每任务散点图的最大点数（默认 300）")
    p.add_argument("--no-density", action="store_true",
                   help="关闭密度热图面板（输出图更紧凑）")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                   help="计算设备（默认 cpu）")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="显示 DEBUG 级别日志")
    return p


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    run_cross_task_pca(
        model_path=args.model,
        graph_folder=args.graphs,
        output_dir=args.output,
        modality=args.modality,
        n_init=args.n_init,
        steps=args.steps,
        seed=args.seed,
        pca_burnin=args.pca_burnin,
        n_components=args.n_components,
        max_pts=args.max_pts,
        show_density=not args.no_density,
        device=args.device,
    )


if __name__ == "__main__":
    main()
