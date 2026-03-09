"""
task_comparison — Cross-task dynamics comparison
=================================================

Answers four key structural questions about multi-task brain dynamics:

  Q1  Does task change the attractor, or only the trajectory on it?
      → shared-PCA phase portrait coloured by task (task_phase_portrait.png)

  Q2  Is the task difference only a transient perturbation?
      → early-window comparison t=0–50 vs t=0–200 (task_early_dynamics.png)

  Q3  Are hub nodes structurally fixed or task-specific?
      → binary hub-mask heatmap (N regions × n_tasks) (task_hub_stability.png)

  Q4  Does task only rotate the dominant modal directions?
      → principal-angle matrix between per-task top-2 PCA eigenvectors
        (task_modal_rotation.png)

All four analyses are computed from the **trajectory arrays alone** — no
additional model calls are made.

Usage::

    from analysis.task_comparison import run_task_comparison

    task_trajs = {
        "rest":   np.load("rest_trajectories.npy"),    # (n_traj, T, N)
        "nback":  np.load("nback_trajectories.npy"),
        "emotion": np.load("emotion_trajectories.npy"),
    }
    results = run_task_comparison(task_trajs, output_dir=Path("outputs/tasks"))

Inputs
------
``task_trajectories`` — ``Dict[str, np.ndarray]``, each array ``(n_traj, T, N)``.
All tasks must share the **same N** (number of brain regions).  T and n_traj
may differ between tasks.

Outputs
-------
task_phase_portrait.png     — PC1 vs PC2, all tasks, colour = task
task_early_dynamics.png     — two panels: t=0–early_window vs t=0–T
task_hub_stability.png      — N×n_tasks binary heatmap (hub mask)
task_modal_rotation.png     — principal-angle matrix (degrees)
task_comparison.json        — serialisable numerical summary
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from spectral_dynamics.plot_utils import write_fallback_png as _write_fallback_png
except ImportError:
    import struct
    import zlib

    def _write_fallback_png(path: "Path") -> None:  # type: ignore[misc]
        """Minimal fallback when spectral_dynamics is not on sys.path."""
        def _chunk(tag: bytes, data: bytes) -> bytes:
            c = struct.pack(">I", len(data)) + tag + data
            return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

        raw_row = b"\x00\xcc\xcc"
        idat_data = zlib.compress(raw_row * 2, level=1)
        png = (
            b"\x89PNG\r\n\x1a\n"
            + _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0))
            + _chunk(b"IDAT", idat_data)
            + _chunk(b"IEND", b"")
        )
        Path(path).write_bytes(png)

logger = logging.getLogger(__name__)

# Default early-dynamics window (steps to show in the "transient" panel).
_DEFAULT_EARLY_WINDOW: int = 50

# Hub detection: regions with mean activity > global_mean + hub_sigma * std.
_DEFAULT_HUB_SIGMA: float = 2.0

# Maximum number of trajectories per task shown in phase-portrait plots.
_MAX_TRAJ_SHOW: int = 8

# Number of PCA components used for modal rotation analysis.
_N_MODAL_COMPONENTS: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_task_comparison(
    task_trajectories: Dict[str, np.ndarray],
    output_dir: Optional[Path] = None,
    burnin: int = 0,
    early_window: int = _DEFAULT_EARLY_WINDOW,
    hub_sigma: float = _DEFAULT_HUB_SIGMA,
) -> Dict:
    """
    Run all four cross-task comparison analyses from trajectory data alone.

    Args:
        task_trajectories:  Mapping ``task_name → trajectories``.
                            Each value has shape ``(n_traj, T, N)``.
                            All tasks must share the same N.
        output_dir:         Directory for plots and JSON; None → no files saved.
        burnin:             Steps to skip at the start of each trajectory
                            before analysis (removes context-injection transient).
        early_window:       Number of post-burnin steps shown in the
                            "early dynamics" panel (Q2).  Default 50.
        hub_sigma:          Threshold multiplier for hub detection (Q3).
                            hub = region_mean > global_mean + hub_sigma × std.

    Returns:
        dict with keys:
          shared_pca_variance_top2_pct:  float — % variance captured by PC1+PC2
          hub_overlap_fraction:      float — fraction of regions that are hub
                                     in ALL tasks (structural hubs)
          modal_angles_deg:          Dict[str, float] — pairwise principal
                                     angles between tasks' top-2 PCA directions
          n_tasks:                   int
          task_names:                list[str]
          attractor_verdict:         str — "same_attractor" | "different_attractors"
                                          | "insufficient_data"
          transient_verdict:         str — "task_is_perturbation" | "task_shifts_attractor"
                                          | "insufficient_data"
          hub_verdict:               str — "hubs_structural" | "hubs_task_specific"
                                          | "insufficient_data"
          modal_verdict:             str — "modal_rotation" | "modal_restructuring"
                                          | "insufficient_data"
    """
    tasks = list(task_trajectories.keys())
    n_tasks = len(tasks)

    if n_tasks < 2:
        logger.warning(
            "run_task_comparison requires >= 2 tasks; got %d.  "
            "Returning empty result.",
            n_tasks,
        )
        return {
            "n_tasks": n_tasks,
            "task_names": tasks,
            "attractor_verdict": "insufficient_data",
            "transient_verdict": "insufficient_data",
            "hub_verdict": "insufficient_data",
            "modal_verdict": "insufficient_data",
        }

    # Validate N consistency
    N_values = [arr.shape[2] for arr in task_trajectories.values()]
    if len(set(N_values)) > 1:
        raise ValueError(
            f"All tasks must have the same number of brain regions N; "
            f"got {dict(zip(tasks, N_values))}"
        )
    N = N_values[0]
    logger.info(
        "Task comparison: %d tasks, N=%d regions. tasks=%s",
        n_tasks, N, tasks,
    )

    # ── Shared PCA (fit on ALL tasks combined) ─────────────────────────────
    shared_pca, X_all_dict, _ = _fit_shared_pca(
        task_trajectories, burnin=burnin
    )

    # ── Analysis 1: Phase portrait in shared PCA ──────────────────────────
    attractor_verdict = _compute_attractor_verdict(X_all_dict)

    # ── Analysis 2: Early vs full dynamics ────────────────────────────────
    transient_verdict = _compute_transient_verdict(
        task_trajectories, shared_pca, burnin, early_window
    )

    # ── Analysis 3: Hub stability ──────────────────────────────────────────
    hub_results = _compute_hub_stability(task_trajectories, burnin, hub_sigma)

    # ── Analysis 4: Modal rotation ─────────────────────────────────────────
    modal_results = _compute_modal_rotation(task_trajectories, burnin)

    # ── Assemble result dict ───────────────────────────────────────────────
    evr = shared_pca.explained_variance_ratio_
    shared_var_top2 = float(evr[:2].sum()) * 100 if len(evr) >= 2 else 0.0

    results: Dict = {
        "n_tasks": n_tasks,
        "task_names": tasks,
        "shared_pca_variance_top2_pct": round(shared_var_top2, 2),
        "hub_overlap_fraction": hub_results["overlap_fraction"],
        "modal_angles_deg": modal_results["pairwise_angles_deg"],
        "attractor_verdict": attractor_verdict,
        "transient_verdict": transient_verdict,
        "hub_verdict": hub_results["verdict"],
        "modal_verdict": modal_results["verdict"],
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Plot 1: task phase portrait
        _plot_task_phase_portrait(
            X_all_dict, shared_pca,
            task_trajectories=task_trajectories,
            burnin=burnin,
            output_path=out / "task_phase_portrait.png",
        )

        # Plot 2: early dynamics comparison
        _plot_early_dynamics(
            task_trajectories, shared_pca,
            burnin=burnin,
            early_window=early_window,
            output_path=out / "task_early_dynamics.png",
        )

        # Plot 3: hub stability heatmap
        _plot_hub_stability(
            hub_results,
            output_path=out / "task_hub_stability.png",
        )

        # Plot 4: modal rotation matrix
        _plot_modal_rotation(
            modal_results,
            output_path=out / "task_modal_rotation.png",
        )

        # JSON summary
        with open(out / "task_comparison.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info("  → Saved task comparison summary: %s", out / "task_comparison.json")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_shared_pca(
    task_trajectories: Dict[str, np.ndarray],
    burnin: int,
    n_components: int = 3,
):
    """
    Fit a single PCA on all task trajectories combined.

    Returns:
        pca:         fitted sklearn PCA object
        X_all_dict:  Dict[task → np.ndarray (n_traj*T_eff, n_components)]
        X_all:       np.ndarray (total_samples, N) — concatenated states
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn is required for task_comparison") from e

    all_states: List[np.ndarray] = []
    task_samples: Dict[str, int] = {}

    for task, trajs in task_trajectories.items():
        _n, T, N = trajs.shape
        b = min(burnin, T - 1)
        states = trajs[:, b:, :].reshape(-1, N).astype(np.float64)
        all_states.append(states)
        task_samples[task] = states.shape[0]

    X_all = np.concatenate(all_states, axis=0)  # (total_samples, N)

    n_comp = min(n_components, X_all.shape[1], X_all.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=42, svd_solver="full")
    pca.fit(X_all)

    # Project each task's states
    X_all_dict: Dict[str, np.ndarray] = {}
    cursor = 0
    for task in task_trajectories:
        ns = task_samples[task]
        X_all_dict[task] = pca.transform(X_all[cursor:cursor + ns])
        cursor += ns

    return pca, X_all_dict, X_all


def _compute_attractor_verdict(X_all_dict: Dict[str, np.ndarray]) -> str:
    """
    Q1: Same attractor or different attractors?

    Heuristic: compute the centroid of PC1/PC2 for each task.
    If the maximum pairwise centroid distance < 1× the typical within-task
    spread, tasks occupy the same PC region → "same_attractor".
    """
    centroids: List[np.ndarray] = []
    spreads: List[float] = []

    for X in X_all_dict.values():
        pc2 = X[:, :2]
        centroids.append(pc2.mean(axis=0))
        spreads.append(float(np.sqrt((pc2.var(axis=0)).sum())))

    if len(centroids) < 2:
        return "insufficient_data"

    avg_spread = float(np.mean(spreads)) if spreads else 1.0
    max_dist = max(
        float(np.linalg.norm(centroids[i] - centroids[j]))
        for i in range(len(centroids))
        for j in range(i + 1, len(centroids))
    )
    ratio = max_dist / max(avg_spread, 1e-9)
    return "same_attractor" if ratio < 1.0 else "different_attractors"


def _compute_transient_verdict(
    task_trajectories: Dict[str, np.ndarray],
    shared_pca,
    burnin: int,
    early_window: int,
) -> str:
    """
    Q2: Task difference only in transient?

    Computes centroid separation at the early window vs the full trajectory
    in shared PC1/PC2 space.  If early separation > 2× late separation,
    the task difference is mainly transient → "task_is_perturbation".
    """
    early_seps: List[float] = []
    late_seps: List[float] = []
    tasks = list(task_trajectories.keys())

    for i, t1 in enumerate(tasks):
        for t2 in tasks[i + 1:]:
            trajs1 = task_trajectories[t1]
            trajs2 = task_trajectories[t2]
            N = trajs1.shape[2]
            b = min(burnin, trajs1.shape[1] - 1)

            # Early window PC centroids
            ew1 = min(early_window, trajs1.shape[1] - b)
            ew2 = min(early_window, trajs2.shape[1] - b)
            early1 = trajs1[:, b:b + ew1, :].reshape(-1, N).astype(np.float64)
            early2 = trajs2[:, b:b + ew2, :].reshape(-1, N).astype(np.float64)
            c_early1 = shared_pca.transform(early1)[:, :2].mean(axis=0)
            c_early2 = shared_pca.transform(early2)[:, :2].mean(axis=0)
            early_seps.append(float(np.linalg.norm(c_early1 - c_early2)))

            # Late (full) PC centroids
            late1 = trajs1[:, b:, :].reshape(-1, N).astype(np.float64)
            late2 = trajs2[:, b:, :].reshape(-1, N).astype(np.float64)
            c_late1 = shared_pca.transform(late1)[:, :2].mean(axis=0)
            c_late2 = shared_pca.transform(late2)[:, :2].mean(axis=0)
            late_seps.append(float(np.linalg.norm(c_late1 - c_late2)))

    if not early_seps:
        return "insufficient_data"

    mean_early = float(np.mean(early_seps))
    mean_late = float(np.mean(late_seps))
    ratio = mean_early / max(mean_late, 1e-9)
    return "task_is_perturbation" if ratio > 2.0 else "task_shifts_attractor"


def _compute_hub_stability(
    task_trajectories: Dict[str, np.ndarray],
    burnin: int,
    hub_sigma: float,
) -> Dict:
    """
    Q3: Hub nodes stable across tasks?

    For each task: mean activity per region, then hub = mean > global_mean + sigma*std.
    Returns hub_masks (N × n_tasks bool), overlap_fraction, and verdict.
    """
    tasks = list(task_trajectories.keys())
    n_tasks = len(tasks)
    N = list(task_trajectories.values())[0].shape[2]

    hub_masks = np.zeros((N, n_tasks), dtype=bool)

    for j, task in enumerate(tasks):
        trajs = task_trajectories[task]
        b = min(burnin, trajs.shape[1] - 1)
        states = trajs[:, b:, :].reshape(-1, N).astype(np.float64)
        region_mean = states.mean(axis=0)   # (N,)
        g_mean = float(region_mean.mean())
        g_std = float(region_mean.std())
        hub_masks[:, j] = region_mean > g_mean + hub_sigma * g_std

    # Fraction of regions that are hub in ALL tasks
    always_hub = hub_masks.all(axis=1)
    overlap_fraction = round(float(always_hub.mean()), 4)

    # Fraction of regions that are hub in ANY task
    any_hub = hub_masks.any(axis=1)
    any_fraction = round(float(any_hub.mean()), 4)

    # Jaccard similarity between all task pairs
    jaccard_pairs: Dict[str, float] = {}
    for i, t1 in enumerate(tasks):
        for k2, t2 in enumerate(tasks):
            if k2 <= i:
                continue
            m1 = hub_masks[:, i]
            m2 = hub_masks[:, k2]
            inter = float((m1 & m2).sum())
            union = float((m1 | m2).sum())
            jaccard = inter / max(union, 1.0)
            jaccard_pairs[f"{t1}_vs_{t2}"] = round(jaccard, 4)

    mean_jaccard = round(float(np.mean(list(jaccard_pairs.values()))), 4) if jaccard_pairs else 0.0
    verdict = "hubs_structural" if mean_jaccard > 0.5 else "hubs_task_specific"

    return {
        "hub_masks": hub_masks,          # (N, n_tasks) bool
        "tasks": tasks,
        "overlap_fraction": overlap_fraction,
        "any_hub_fraction": any_fraction,
        "jaccard_pairs": jaccard_pairs,
        "mean_jaccard": mean_jaccard,
        "verdict": verdict,
    }


def _compute_modal_rotation(
    task_trajectories: Dict[str, np.ndarray],
    burnin: int,
) -> Dict:
    """
    Q4: Does task only rotate modal directions?

    For each task independently, fit PCA on that task's trajectories and
    extract the top-2 eigenvectors.  Compute the principal angle (in degrees)
    between the 2D subspaces of all task pairs.

    Principal angle between subspaces A and B (each spanned by 2 vectors):
        θ = arccos(σ_min(A.T @ B))   where σ_min is the smallest singular value.
    Using σ_min gives the *largest* principal angle — i.e. the most-different
    pair of directions between the two subspaces.  This is the right quantity
    for detecting modal restructuring: if even the hardest-to-align direction
    stays < 10°, the subspaces are essentially the same.
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        return {"pairwise_angles_deg": {}, "verdict": "insufficient_data"}

    tasks = list(task_trajectories.keys())
    eigvec_dict: Dict[str, np.ndarray] = {}  # task → (N, n_modal)

    for task, trajs in task_trajectories.items():
        _n, T, N = trajs.shape
        b = min(burnin, T - 1)
        states = trajs[:, b:, :].reshape(-1, N).astype(np.float64)
        n_comp = min(_N_MODAL_COMPONENTS, N, states.shape[0] - 1)
        if n_comp < 1:
            continue
        pca_t = PCA(n_components=n_comp, random_state=42, svd_solver="full")
        pca_t.fit(states)
        eigvec_dict[task] = pca_t.components_.T  # (N, n_comp)

    if len(eigvec_dict) < 2:
        return {"pairwise_angles_deg": {}, "verdict": "insufficient_data"}

    pairwise_angles: Dict[str, float] = {}
    task_list = list(eigvec_dict.keys())
    for i, t1 in enumerate(task_list):
        for t2 in task_list[i + 1:]:
            A = eigvec_dict[t1]   # (N, k)
            B = eigvec_dict[t2]   # (N, k)
            # Principal angle: min singular value of A.T @ B gives cos(θ_max)
            # Use SVD of cross-product matrix
            M = A.T @ B           # (k, k)
            sv = np.linalg.svd(M, compute_uv=False)
            # The principal angle is arccos of the smallest singular value
            # (most different pair of directions)
            cos_angle = float(np.clip(sv.min(), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_angle)))
            pairwise_angles[f"{t1}_vs_{t2}"] = round(angle_deg, 2)

    mean_angle = float(np.mean(list(pairwise_angles.values()))) if pairwise_angles else 0.0

    # Threshold: < 10° → same dynamical core (only rotation)
    verdict = "modal_rotation" if mean_angle < 10.0 else "modal_restructuring"

    return {
        "pairwise_angles_deg": pairwise_angles,
        "mean_angle_deg": round(mean_angle, 2),
        "verdict": verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────

# Colour cycle for tasks (matplotlib default tab10 is overridden for clarity)
_TASK_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990",
]


def _task_color(task_idx: int) -> str:
    return _TASK_COLORS[task_idx % len(_TASK_COLORS)]


def _plot_task_phase_portrait(
    X_all_dict: Dict[str, np.ndarray],
    shared_pca,
    task_trajectories: Dict[str, np.ndarray],
    burnin: int,
    output_path: Path,
) -> None:
    """
    Plot 1: PC1 vs PC2 for all tasks in the shared PCA space.

    Shows the full trajectory rollout for each task, coloured by task label.
    Up to _MAX_TRAJ_SHOW trajectories per task are drawn.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except ImportError:
            pass
    except ImportError:
        _write_fallback_png(output_path)
        return

    tasks = list(task_trajectories.keys())
    evr = shared_pca.explained_variance_ratio_
    var1 = evr[0] * 100 if len(evr) >= 1 else 0
    var2 = evr[1] * 100 if len(evr) >= 2 else 0

    fig, ax = plt.subplots(figsize=(8, 7))

    for ti, task in enumerate(tasks):
        trajs = task_trajectories[task]
        n_traj, T, N = trajs.shape
        b = min(burnin, T - 1)
        T_eff = T - b
        color = _task_color(ti)
        n_show = min(_MAX_TRAJ_SHOW, n_traj)
        idx_show = np.linspace(0, n_traj - 1, n_show, dtype=int)
        first = True
        for i in idx_show:
            states = trajs[i, b:, :].astype(np.float64)
            proj = shared_pca.transform(states)    # (T_eff, n_comp)
            ax.plot(proj[:, 0], proj[:, 1],
                    color=color, lw=0.9, alpha=0.55,
                    label=task if first else None)
            ax.scatter(proj[0, 0], proj[0, 1],
                       color=color, s=20, zorder=4, alpha=0.8)
            first = False

    ax.set_xlabel(f"PC1  ({var1:.1f}% var)")
    ax.set_ylabel(f"PC2  ({var2:.1f}% var)")
    ax.set_title(
        "Task Phase Portrait — Shared PCA Space\n"
        "Same region / same orbit → same attractor; "
        "separate regions → different attractors"
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved task phase portrait: %s", output_path)


def _plot_early_dynamics(
    task_trajectories: Dict[str, np.ndarray],
    shared_pca,
    burnin: int,
    early_window: int,
    output_path: Path,
) -> None:
    """
    Plot 2: Early (t=0–early_window) vs full trajectory comparison.

    Two panels side-by-side in shared PC1/PC2 space, coloured by task.
    If early separation > full separation, task effect is mainly transient.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except ImportError:
            pass
    except ImportError:
        _write_fallback_png(output_path)
        return

    tasks = list(task_trajectories.keys())
    evr = shared_pca.explained_variance_ratio_
    var1 = evr[0] * 100 if len(evr) >= 1 else 0
    var2 = evr[1] * 100 if len(evr) >= 2 else 0

    fig, (ax_early, ax_full) = plt.subplots(1, 2, figsize=(14, 6))

    for ax, title_suffix, use_early in [
        (ax_early, f"Early (t=0–{early_window})", True),
        (ax_full, "Full rollout", False),
    ]:
        for ti, task in enumerate(tasks):
            trajs = task_trajectories[task]
            n_traj, T, N = trajs.shape
            b = min(burnin, T - 1)
            color = _task_color(ti)
            n_show = min(_MAX_TRAJ_SHOW, n_traj)
            idx_show = np.linspace(0, n_traj - 1, n_show, dtype=int)
            first = True
            for i in idx_show:
                states = trajs[i, b:, :].astype(np.float64)
                if use_early:
                    states = states[:min(early_window, len(states))]
                proj = shared_pca.transform(states)
                ax.plot(proj[:, 0], proj[:, 1],
                        color=color, lw=0.9, alpha=0.55,
                        label=task if first else None)
                ax.scatter(proj[0, 0], proj[0, 1],
                           color=color, s=18, zorder=4, alpha=0.8)
                first = False

        ax.set_xlabel(f"PC1  ({var1:.1f}% var)")
        ax.set_ylabel(f"PC2  ({var2:.1f}% var)")
        ax.set_title(title_suffix)
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.2)

    fig.suptitle(
        "Early Dynamics Comparison\n"
        "Early fork + late convergence → task is perturbation; "
        "persistent separation → task shifts attractor",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved early dynamics comparison: %s", output_path)


def _plot_hub_stability(hub_results: Dict, output_path: Path) -> None:
    """
    Plot 3: Binary hub-mask heatmap (N regions × n_tasks).

    Rows = brain regions, columns = tasks.
    White = hub, dark = non-hub.
    A column of identical white cells → task-invariant structural hub.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except ImportError:
            pass
    except ImportError:
        _write_fallback_png(output_path)
        return

    hub_masks = hub_results["hub_masks"].astype(float)  # (N, n_tasks)
    tasks = hub_results["tasks"]
    n_tasks = len(tasks)
    N = hub_masks.shape[0]
    overlap = hub_results["overlap_fraction"]
    mean_j = hub_results["mean_jaccard"]
    verdict = hub_results["verdict"]

    # Sort rows: always-hub regions first
    sort_key = hub_masks.sum(axis=1)
    row_order = np.argsort(-sort_key)
    sorted_masks = hub_masks[row_order]

    fig, ax = plt.subplots(figsize=(max(5, n_tasks * 1.5 + 3), min(12, N // 8 + 4)))
    im = ax.imshow(sorted_masks, aspect="auto", cmap="RdYlGn",
                   vmin=0, vmax=1, interpolation="nearest")

    ax.set_xticks(np.arange(n_tasks))
    ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=9)
    ax.set_yticks([])
    ax.set_ylabel(f"Brain Regions (N={N}, sorted by hub frequency)")
    ax.set_xlabel("Task")
    ax.set_title(
        f"Hub Node Stability  [{verdict}]\n"
        f"Always-hub: {overlap * 100:.1f}%  |  "
        f"Mean Jaccard: {mean_j:.2f}  "
        f"(>0.5 = structural,  <0.5 = task-specific)"
    )
    plt.colorbar(im, ax=ax, label="Hub (1=yes, 0=no)", shrink=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved hub stability heatmap: %s", output_path)


def _plot_modal_rotation(modal_results: Dict, output_path: Path) -> None:
    """
    Plot 4: Principal-angle matrix between task pairs (degrees).

    Each cell shows the principal angle between the 2D PCA subspaces
    of two tasks.  < 10° → same dynamical core (only rotation).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except ImportError:
            pass
    except ImportError:
        _write_fallback_png(output_path)
        return

    pairwise = modal_results.get("pairwise_angles_deg", {})
    mean_angle = modal_results.get("mean_angle_deg", float("nan"))
    verdict = modal_results.get("verdict", "insufficient_data")

    if not pairwise:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.text(0.5, 0.5, "Insufficient data\n(need ≥ 2 tasks)",
                ha="center", va="center", transform=ax.transAxes)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return

    # Build square angle matrix
    tasks_seen: List[str] = []
    for key in pairwise:
        t1, t2 = key.split("_vs_", 1)
        if t1 not in tasks_seen:
            tasks_seen.append(t1)
        if t2 not in tasks_seen:
            tasks_seen.append(t2)
    n = len(tasks_seen)
    angle_matrix = np.zeros((n, n), dtype=float)
    for key, ang in pairwise.items():
        t1, t2 = key.split("_vs_", 1)
        i = tasks_seen.index(t1)
        j = tasks_seen.index(t2)
        angle_matrix[i, j] = ang
        angle_matrix[j, i] = ang

    fig, ax = plt.subplots(figsize=(max(5, n + 2), max(4, n + 2)))
    im = ax.imshow(angle_matrix, aspect="equal", cmap="YlOrRd",
                   vmin=0, vmax=90, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Principal Angle (degrees)")

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(tasks_seen, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(tasks_seen, fontsize=9)

    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{angle_matrix[i, j]:.1f}°",
                    ha="center", va="center", fontsize=8,
                    color="black" if angle_matrix[i, j] < 45 else "white")

    ax.set_title(
        f"Modal Direction Rotation  [{verdict}]\n"
        f"Mean angle = {mean_angle:.1f}°  "
        f"(<10° = same dynamical core,  >10° = modal restructuring)"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → Saved modal rotation matrix: %s", output_path)
