#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-Task PCA Projection — Unified Cognitive Dynamics Manifold Analysis
==========================================================================

**核心问题**：不同任务（或不同被试、不同 run）之间的动力学差异，
是真实的动力学结构差异，还是仅仅是同一个环形/环面吸引子上的"相位偏移"？

**方法**：
  1. 从 graph 文件夹读取所有图缓存 (``.pt`` 文件）
  2. 对**每个**图文件，使用 GNN 模型生成自由动力学轨迹。
     每个任务只使用**自身**的 BOLD 历史作为初始上下文，不混入其它任务的数据。
     这是关键设计原则：轨迹必须反映该任务自身的内在动力学，
     才能在 PCA 空间中对不同任务进行有意义的比较。
  3. 将所有任务的轨迹**合并后**拟合一个统一 PCA（确保投影空间相同）
  4. 将每个任务的轨迹投影到 PC1-PC2 平面，用不同颜色标注任务
  5. 输出可视化图和 JSON 指标文件

**如何解读结果**：

- **同一环/环面，不同位置** → 统一认知动力流形（unified cognitive manifold）：
  模型发现了跨任务通用的动力学吸引子，不同任务仅是该吸引子上的不同相位状态。
  这是一个重要的神经科学发现：认知动力学存在任务不变的底层流形。

- **不同的环/环面，分离的簇** → 任务特异性吸引子：
  每个任务有自己的稳定动力学，任务切换对应吸引子切换而非相位漂移。

**用法**::

    python -m analysis.cross_task_pca \\
        --model outputs/exp/best_model.pt \\
        --graphs outputs/graph_cache/ \\
        --output outputs/cross_task_pca/ \\
        [--modality fmri] \\
        [--n-init 30] \\
        [--steps 200] \\
        [--seed 42]

**命令行参数**：

--model         训练好的模型检查点路径（``best_model.pt``）
--graphs        包含图缓存 .pt 文件的文件夹路径
--output        输出目录
--modality      分析模态：fmri / eeg / joint（默认 fmri）
--n-init        每个图文件的初始轨迹数（默认 20）
--steps         每条轨迹的预测步数（默认 200）
--seed          随机种子（默认 42）
--pca-burnin    丢弃的初始帧数，避免瞬态污染 PCA（默认自动 = max(10, steps//10)）
--max-pts       每条轨迹采样点数上限，防止图表过密（默认 300）
--no-density    关闭密度热图（默认开启）
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
    n_components: int = 10,
) -> Tuple["PCA", np.ndarray, np.ndarray]:
    """Fit PCA on **all** trajectories (all tasks pooled).

    Parameters
    ----------
    all_trajs:
        Shape ``(n_traj_total, steps, n_regions)``.
    burnin:
        Number of initial frames to discard from each trajectory.
    n_components:
        PCA components to compute.

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

    n_panels = 3 if show_density else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    fig.suptitle(
        "Cross-Task PCA Projection — Unified Cognitive Dynamics Manifold?\n"
        f"PC1={pc1_var:.1f}%  PC2={pc2_var:.1f}%  (joint PCA across all tasks)",
        fontsize=10,
    )

    ax_scatter = axes[0]
    ax_time    = axes[1]
    ax_density = axes[2] if show_density else None

    # ── Panel 0: scatter, all tasks overlaid ─────────────────────────────────
    for task_i in range(n_tasks):
        mask = traj_idx == task_i
        pts = X_pca[mask, :2]
        if len(pts) == 0:
            continue
        # Downsample if too many points
        if len(pts) > max_pts:
            idx = np.random.default_rng(42 + task_i).choice(len(pts), max_pts, replace=False)
            pts = pts[idx]
        color = cmap_list[task_i % len(cmap_list)]
        ax_scatter.scatter(
            pts[:, 0], pts[:, 1],
            s=4, alpha=0.35, color=color,
            label=task_labels[task_i] if len(task_labels[task_i]) <= 20
            else task_labels[task_i][:18] + "..",
            rasterized=True,
        )
    ax_scatter.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
    ax_scatter.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
    ax_scatter.set_title(
        "All tasks in joint PCA space\n"
        "(same ring/torus = phase-shift only; separate clusters = distinct attractors)"
    )
    ax_scatter.legend(markerscale=3, fontsize=7, ncol=max(1, n_tasks // 8),
                      loc="best", framealpha=0.7)
    ax_scatter.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_scatter.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax_scatter.grid(True, alpha=0.2)

    # ── Panel 1: time-coloured scatter (single task or mean trajectory) ────────
    # Show all tasks, each trajectory coloured by time to reveal orbits
    sc = None  # initialise before loop so the colorbar reference is always valid
    for task_i in range(min(n_tasks, 5)):  # limit to first 5 to keep legible
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
        "Trajectories coloured by time\n(first 5 tasks shown; circular = ring attractor)"
    )
    ax_time.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_time.axvline(0, color="k", lw=0.5, alpha=0.3)
    ax_time.grid(True, alpha=0.2)

    # ── Panel 2: joint density heat-map ──────────────────────────────────────
    if ax_density is not None:
        all_pts = X_pca[:, :2]
        ax_density.hexbin(
            all_pts[:, 0], all_pts[:, 1],
            gridsize=40, cmap="YlOrRd", mincnt=1,
        )
        ax_density.set_xlabel(f"PC1 ({pc1_var:.1f}%)")
        ax_density.set_ylabel(f"PC2 ({pc2_var:.1f}%)")
        ax_density.set_title(
            "Joint density heat-map (all tasks)\n"
            "(ring-shaped density = unified circular attractor)"
        )
        ax_density.axhline(0, color="k", lw=0.5, alpha=0.3)
        ax_density.axvline(0, color="k", lw=0.5, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  → PCA figure saved: %s", output_path)


def _plot_variance_curve(pca, output_path: Path) -> None:
    """Scree plot of cumulative explained variance."""
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
    axes[0].bar(np.arange(1, n + 1), evr * 100, color="steelblue", alpha=0.7)
    axes[0].set_xlabel("Principal component")
    axes[0].set_ylabel("Explained variance (%)")
    axes[0].set_title("Per-PC variance (joint PCA)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].plot(np.arange(1, n + 1), cumvar * 100, "o-", color="steelblue")
    axes[1].axhline(90, color="salmon", linestyle="--", label="90%")
    axes[1].axhline(50, color="orange", linestyle="--", label="50%")
    axes[1].set_xlabel("Principal components (top k)")
    axes[1].set_ylabel("Cumulative variance (%)")
    axes[1].set_title("Cumulative explained variance")
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
    max_pts: int = 300,
    show_density: bool = True,
    device: str = "cpu",
) -> Dict:
    """Run cross-task PCA projection analysis.

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
    max_pts:
        Maximum number of scatter points per task in the figure.
    show_density:
        Whether to include the joint density heat-map panel.
    device:
        Compute device (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    dict with keys ``tasks``, ``pca_metrics``, ``manifold_analysis``.
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
        "合并轨迹形状: %s  (pca_burnin=%d，有效帧数=%d)",
        all_trajs.shape, pca_burnin, max(0, steps - pca_burnin),
    )

    pca, X_pca, traj_idx = _fit_joint_pca(all_trajs, burnin=pca_burnin)

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
    cumvar = np.cumsum(evr)
    n50 = int(np.searchsorted(cumvar, 0.50) + 1)
    n80 = int(np.searchsorted(cumvar, 0.80) + 1)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)

    pca_metrics = {
        "n_tasks": len(task_labels),
        "n_traj_per_task": n_init,
        "steps": steps,
        "pca_burnin": pca_burnin,
        "variance_pc1": float(evr[0]) if len(evr) > 0 else None,
        "variance_pc2": float(evr[1]) if len(evr) > 1 else None,
        "variance_top2": float(cumvar[1]) if len(cumvar) > 1 else None,
        "n_components_50pct": n50,
        "n_components_80pct": n80,
        "n_components_90pct": n90,
    }

    # ── Per-task manifold analysis ────────────────────────────────────────────
    task_analysis = {}
    for ti, lbl in enumerate(task_labels):
        mask_t = sample_task == ti
        if mask_t.sum() == 0:
            continue
        pts = X_pca[mask_t, :2]
        ring_s = _ring_score(pts)
        # Centroid in PC space (guard against < 2 components)
        centroid = pts.mean(axis=0).tolist()
        task_analysis[lbl] = {
            "ring_score": float(ring_s),
            "centroid_pc1": float(centroid[0]) if len(centroid) > 0 else float("nan"),
            "centroid_pc2": float(centroid[1]) if len(centroid) > 1 else float("nan"),
            "n_samples": int(mask_t.sum()),
        }

    # Phase offsets between task pairs
    phase_offsets = {}
    lbls = list(task_analysis.keys())
    for i in range(len(lbls)):
        for j in range(i + 1, len(lbls)):
            la, lb = lbls[i], lbls[j]
            pts_a = X_pca[sample_task == task_labels.index(la), :2]
            pts_b = X_pca[sample_task == task_labels.index(lb), :2]
            offset = _phase_offset(pts_a, pts_b)
            phase_offsets[f"{la}_vs_{lb}"] = float(offset)

    # Global ring score (all tasks pooled)
    all_pts = X_pca[:, :2]
    global_ring_score = _ring_score(all_pts)

    # Manifold interpretation
    avg_task_ring = float(np.mean([v["ring_score"] for v in task_analysis.values()])) \
        if task_analysis else float("nan")
    if global_ring_score < 0.25:
        manifold_verdict = (
            "RING / TORUS — 统一认知动力流形：所有任务轨迹集中在同一环形吸引子上。"
            "任务差异表现为相位偏移，而非不同吸引子。"
        )
    elif global_ring_score < 0.45:
        manifold_verdict = (
            "PARTIAL RING — 准环形吸引子：轨迹大致呈环形但存在任务间扩散。"
            "可能是多个略微不同的环形吸引子叠加，或存在任务特异性轨道变形。"
        )
    else:
        manifold_verdict = (
            "DIFFUSE / MULTI-ATTRACTOR — 弥散分布或多吸引子："
            "不同任务轨迹分散分布，不共享明显的环形结构。"
            "任务差异可能来自不同的吸引子盆地，而非相位偏移。"
        )

    manifold_analysis = {
        "global_ring_score": float(global_ring_score),
        "avg_task_ring_score": avg_task_ring,
        "verdict": manifold_verdict,
        "interpretation": (
            "ring_score < 0.25 → ring/torus attractor (phase shift hypothesis supported); "
            "0.25–0.45 → partial ring; >0.45 → diffuse/multi-attractor. "
            "Phase offsets between task pairs (in degrees) are also reported."
        ),
        "task_metrics": task_analysis,
        "phase_offsets_deg": phase_offsets,
    }

    # ── Figures ───────────────────────────────────────────────────────────────
    _plot_joint_pca(
        X_pca, sample_task, task_labels, pca,
        output_dir / "cross_task_pca.png",
        max_pts=max_pts,
        show_density=show_density,
    )
    _plot_variance_curve(pca, output_dir / "pca_variance_curve.png")

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
    logger.info(
        "\n══════════════════════════════════════════════\n"
        "跨任务 PCA 分析完成 (%d 个任务)\n"
        "  PC1=%.1f%%  PC2=%.1f%%  合计=%.1f%%\n"
        "  全局环形评分: %.3f\n"
        "  结论: %s\n"
        "══════════════════════════════════════════════",
        len(task_labels),
        pca_metrics["variance_pc1"] * 100 if pca_metrics["variance_pc1"] else 0,
        pca_metrics["variance_pc2"] * 100 if pca_metrics["variance_pc2"] else 0,
        (pca_metrics["variance_top2"] or 0) * 100,
        global_ring_score,
        manifold_verdict,
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
        max_pts=args.max_pts,
        show_density=not args.no_density,
        device=args.device,
    )


if __name__ == "__main__":
    main()
