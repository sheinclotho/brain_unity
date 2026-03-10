"""
node_ablation.py
================
TASK 4: Node Ablation Experiment
TASK 5: Dynamic Lesion Experiment

Both experiments operate purely on the GNN-generated trajectories from
Phase 1 — no additional model inference, no surrogate dynamics model.

Node Ablation (TASK 4)
----------------------
For each candidate node *i*, its activity is masked out of the observed
trajectories (set to per-trajectory mean) to measure how much the
dynamical manifold depends on that node::

    trajs_ablated[:, :, i] = trajs[:, :, i].mean()

Then the following change metrics are computed relative to the intact baseline:

  ΔLLE           — change in Largest Lyapunov Exponent
  ΔD2            — change in correlation dimension
  Δρ (%)         — percentage change in spectral radius of the FC matrix
  Procrustes dist — Procrustes distance between baseline and ablated PCA manifolds

Nodes are ranked by |ΔLLE|; the output CSV lists every tested node.

Outputs
-------
  node_importance.csv
  node_importance_hist.png

Dynamic Lesion (TASK 5)
-----------------------
Using the top-k ablation-ranked nodes plus a matched set of random control
nodes, we split each trajectory at ``lesion_step``::

    pre-lesion:  trajs[:, :lesion_step, :]
    post-lesion: trajs[:, lesion_step:, :] with node activity clamped
                 to its pre-lesion mean from t = lesion_step onwards.

Pre- and post-lesion metrics (LLE, D2, attractor escape rate, PCA
manifold shift) are then compared between the top vs. control groups.

Outputs
-------
  lesion_effects.csv
  lesion_examples.png
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STD_GUARD = 1e-8


# ---------------------------------------------------------------------------
# Shared metric helpers (all trajectory-based)
# ---------------------------------------------------------------------------

def _lle(trajs: np.ndarray) -> float:
    """Estimate LLE by averaging Rosenstein estimates across all trajectories.

    Calls ``rosenstein_lyapunov`` directly (no simulator required) on each
    individual trajectory (shape T × N) and returns the mean, skipping NaN.
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        vals = []
        for traj in trajs:               # traj: (T, N)
            lle, _ = rosenstein_lyapunov(
                traj.astype(np.float64),
                max_lag=50,
                min_temporal_sep=20,
            )
            if np.isfinite(lle):
                vals.append(lle)
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:
        return float("nan")


def _d2(trajs: np.ndarray) -> float:
    """Estimate correlation dimension D2 on a single representative trajectory.

    Uses the first trajectory (shape T × N) so that ``correlation_dimension``
    receives the expected 2-D array.  The old code incorrectly flattened the
    array to 1-D and passed unsupported keyword arguments.
    """
    try:
        from analysis.embedding_dimension import correlation_dimension
        traj = trajs[0].astype(np.float64)   # (T, N)
        result = correlation_dimension(traj, max_points=1000)
        return float(result["D2"])
    except Exception:
        return float("nan")


def _pca_dim(trajs: np.ndarray, var_threshold: float = 0.90) -> int:
    X = trajs.reshape(-1, trajs.shape[-1])
    X = X - X.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        v = s ** 2
        return int(np.searchsorted(np.cumsum(v) / (v.sum() + _STD_GUARD), var_threshold) + 1)
    except Exception:
        return -1


def _fc_spectral_radius(trajs: np.ndarray) -> float:
    """Spectral radius of the Pearson FC matrix computed from trajectories."""
    try:
        X = trajs.reshape(-1, trajs.shape[-1])
        X = X - X.mean(axis=0)
        std = X.std(axis=0)
        std[std < _STD_GUARD] = _STD_GUARD
        X /= std
        FC = (X.T @ X) / (len(X) - 1)
        np.fill_diagonal(FC, 0.0)
        return float(np.abs(np.linalg.eigvals(FC)).max())
    except Exception:
        return float("nan")


def _procrustes_dist(trajs_a: np.ndarray, trajs_b: np.ndarray,
                     n_comp: int = 10) -> float:
    """Normalised Procrustes distance between two trajectory manifolds."""
    Xa = trajs_a.reshape(-1, trajs_a.shape[-1])
    Xb = trajs_b.reshape(-1, trajs_b.shape[-1])
    joint = np.vstack([Xa, Xb])
    mu = joint.mean(axis=0)
    joint_c = joint - mu
    try:
        _, _, Vt = np.linalg.svd(joint_c, full_matrices=False)
        comps = Vt[:n_comp]
    except np.linalg.LinAlgError:
        comps = np.eye(min(n_comp, Xa.shape[1]))
    Pa = (Xa - mu) @ comps.T
    Pb = (Xb - mu) @ comps.T
    n = min(len(Pa), len(Pb))
    Pa, Pb = Pa[:n], Pb[:n]
    na, nb = np.linalg.norm(Pa), np.linalg.norm(Pb)
    if na < 1e-12 or nb < 1e-12:
        return float("nan")
    Pa /= na
    Pb /= nb
    try:
        from scipy.spatial import procrustes as _sp
        _, _, d = _sp(Pa, Pb)
        return float(d)
    except Exception:
        M = Pb.T @ Pa
        _, s, _ = np.linalg.svd(M)
        s = np.clip(s, -1, 1)
        return float(np.mean(np.arccos(s)))


# ---------------------------------------------------------------------------
# TASK 4: Node ablation (trajectory-space masking)
# ---------------------------------------------------------------------------

def run_node_ablation(
    trajectories: np.ndarray,
    nodes_to_test: Optional[Sequence[int]] = None,
    n_top_variance: int = 50,
    n_random: int = 50,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Node ablation experiment on GNN-generated trajectories.

    For each candidate node *i*, the node's activity is masked (replaced by
    its per-trajectory mean) and the change in manifold structure is
    measured.

    Parameters
    ----------
    trajectories:
        GNN trajectories from Phase 1, shape ``(n_traj, T, N)``.
    nodes_to_test:
        Explicit list of node indices to ablate.  ``None`` → auto-select
        the top ``n_top_variance`` high-variance nodes plus ``n_random``
        random nodes.
    n_top_variance:
        Number of high-variance nodes to include in the auto-selection.
    n_random:
        Number of random nodes to include as a control sample.
    seed:
        Random seed for the random-node selection.
    output_dir:
        Directory to save outputs.

    Returns
    -------
    dict with keys ``'baseline'``, ``'ablation'`` (sorted by |ΔLLE|),
    ``'top_nodes'`` (sorted node list).
    """
    trajs = np.asarray(trajectories, dtype=np.float32)
    n_traj, T, N = trajs.shape
    rng = np.random.default_rng(seed)

    # Auto-select nodes
    if nodes_to_test is None:
        node_var = trajs.reshape(-1, N).var(axis=0)  # (N,)
        top_high = list(np.argsort(node_var)[-n_top_variance:])
        remaining = [i for i in range(N) if i not in set(top_high)]
        rand_sample = list(rng.choice(remaining,
                                      size=min(n_random, len(remaining)),
                                      replace=False))
        nodes_to_test = sorted(set(top_high + rand_sample))

    logger.info("Node ablation: testing %d nodes on %d trajectories (T=%d, N=%d)",
                len(nodes_to_test), n_traj, T, N)

    # Baseline metrics
    base_lle = _lle(trajs)
    base_d2 = _d2(trajs)
    base_rho = _fc_spectral_radius(trajs)
    logger.info("  Baseline: LLE=%.4f  D2=%.2f  FC_rho=%.3f", base_lle, base_d2, base_rho)
    baseline = {"lle": base_lle, "d2": base_d2, "fc_spectral_radius": base_rho}

    ablation_rows = []
    for k, node in enumerate(nodes_to_test):
        # Mask node: replace each trajectory's node activity with its per-trajectory mean.
        # node_mean shape: (n_traj, 1) broadcasts to (n_traj, T) for the column assignment.
        ablated = trajs.copy()
        node_mean = ablated[:, :, node].mean(axis=1, keepdims=True)  # (n_traj, 1)
        ablated[:, :, node] = node_mean  # broadcast: (n_traj, 1) → (n_traj, T)

        abl_lle = _lle(ablated)
        abl_d2 = _d2(ablated)
        abl_rho = _fc_spectral_radius(ablated)
        proc = _procrustes_dist(trajs, ablated)

        delta_lle = float(abl_lle - base_lle) if not (np.isnan(abl_lle) or np.isnan(base_lle)) else float("nan")
        delta_d2 = float(abl_d2 - base_d2) if not (np.isnan(abl_d2) or np.isnan(base_d2)) else float("nan")
        delta_rho_pct = (
            (abl_rho - base_rho) / (abs(base_rho) + _STD_GUARD) * 100.0
            if not (np.isnan(abl_rho) or np.isnan(base_rho)) else float("nan")
        )

        row = {
            "node": int(node),
            "delta_lle": delta_lle,
            "delta_d2": delta_d2,
            "delta_rho_pct": delta_rho_pct,
            "procrustes_dist": float(proc),
            "ablated_lle": float(abl_lle),
            "ablated_d2": float(abl_d2),
            "ablated_fc_rho": float(abl_rho),
        }
        ablation_rows.append(row)

        if k % 20 == 0:
            logger.info(
                "  [%d/%d] Node %d: ΔLLE=%.4f  ΔD2=%.2f  Δρ=%.1f%%  Procrustes=%.4f",
                k + 1, len(nodes_to_test), node,
                delta_lle, delta_d2, delta_rho_pct, proc,
            )

    # Sort by |ΔLLE| descending (NaN treated as 0 so they sort last)
    def _sort_key(r):
        v = r["delta_lle"]
        return abs(v) if not np.isnan(v) else 0.0

    ablation_rows.sort(key=_sort_key, reverse=True)
    top_nodes = [r["node"] for r in ablation_rows]

    results = {
        "baseline": baseline,
        "ablation": ablation_rows,
        "top_nodes": top_nodes,
        "n_tested": len(ablation_rows),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_ablation_csv(ablation_rows, output_dir / "node_importance.csv")
        _save_ablation_hist(ablation_rows, output_dir / "node_importance_hist.png")

    return results


# ---------------------------------------------------------------------------
# TASK 5: Dynamic lesion (trajectory-space clamping)
# ---------------------------------------------------------------------------

def run_lesion_dynamics(
    trajectories: np.ndarray,
    top_nodes: Sequence[int],
    n_control: int = 10,
    n_lesion_nodes: int = 10,
    lesion_step: int = 500,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Dynamic lesion experiment on GNN-generated trajectories.

    Splits trajectories at ``lesion_step``; from that point the lesion
    node's activity is clamped to its pre-lesion mean.  Pre- and
    post-lesion dynamics metrics are compared for top-importance vs
    control nodes.

    Parameters
    ----------
    trajectories:
        GNN trajectories from Phase 1, shape ``(n_traj, T, N)``.
    top_nodes:
        Nodes ranked by importance (output of :func:`run_node_ablation`).
    n_control:
        Number of random control nodes.
    n_lesion_nodes:
        Number of top-importance nodes to lesion.
    lesion_step:
        Time step at which the lesion is applied.
    seed:
        Random seed for control-node sampling.
    output_dir:
        Directory to save outputs.

    Returns
    -------
    dict with ``'lesion_results'``, ``'lesion_step'``.
    """
    trajs = np.asarray(trajectories, dtype=np.float32)
    n_traj, T, N = trajs.shape
    rng = np.random.default_rng(seed)

    if lesion_step >= T:
        logger.warning("lesion_step (%d) >= T (%d); clamping to T//2", lesion_step, T)
        lesion_step = T // 2

    lesion_top = list(top_nodes[:n_lesion_nodes])
    used = set(lesion_top)
    control_pool = [i for i in range(N) if i not in used]
    control_nodes = list(rng.choice(control_pool,
                                    size=min(n_control, len(control_pool)),
                                    replace=False))
    all_nodes = [(n, "top") for n in lesion_top] + [(n, "control") for n in control_nodes]

    logger.info("Dynamic lesion: %d top + %d control nodes, lesion_step=%d",
                len(lesion_top), len(control_nodes), lesion_step)

    rows = []
    for node, node_type in all_nodes:
        pre_trajs, post_trajs = _apply_lesion(trajs, node, lesion_step)

        pre_lle = _lle(pre_trajs)
        pre_d2 = _d2(pre_trajs)
        post_lle = _lle(post_trajs)
        post_d2 = _d2(post_trajs)
        escape = _attractor_escape_rate(pre_trajs, post_trajs)
        pca_shift = _procrustes_dist(pre_trajs, post_trajs)

        row = {
            "node": int(node),
            "node_type": node_type,
            "pre_lle": float(pre_lle),
            "post_lle": float(post_lle),
            "delta_lle": float(post_lle - pre_lle)
                if not (np.isnan(post_lle) or np.isnan(pre_lle)) else float("nan"),
            "pre_d2": float(pre_d2),
            "post_d2": float(post_d2),
            "delta_d2": float(post_d2 - pre_d2)
                if not (np.isnan(post_d2) or np.isnan(pre_d2)) else float("nan"),
            "attractor_escape_rate": float(escape),
            "pca_manifold_shift": float(pca_shift),
        }
        rows.append(row)
        logger.info(
            "  Node %d (%s): ΔLLE=%.4f  ΔD2=%.2f  escape=%.4f  shift=%.4f",
            node, node_type,
            row["delta_lle"] if not np.isnan(row["delta_lle"]) else float("nan"),
            row["delta_d2"] if not np.isnan(row["delta_d2"]) else float("nan"),
            escape, pca_shift,
        )

    results = {"lesion_results": rows, "lesion_step": lesion_step}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_lesion_csv(rows, output_dir / "lesion_effects.csv")
        _save_lesion_plot(rows, output_dir / "lesion_examples.png")

    return results


# ---------------------------------------------------------------------------
# Lesion helpers
# ---------------------------------------------------------------------------

def _apply_lesion(
    trajs: np.ndarray,
    node: int,
    lesion_step: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (pre, post) trajectory segments with post-lesion node clamped."""
    pre_trajs = trajs[:, :lesion_step, :]  # (n_traj, lesion_step, N)

    post_trajs = trajs[:, lesion_step:, :].copy()   # (n_traj, T-lesion_step, N)
    # Clamp the lesion node to its pre-lesion per-trajectory mean
    pre_mean = pre_trajs[:, :, node].mean(axis=1, keepdims=True)  # (n_traj, 1)
    post_trajs[:, :, node] = pre_mean

    return pre_trajs, post_trajs


def _attractor_escape_rate(pre_trajs: np.ndarray, post_trajs: np.ndarray) -> float:
    """Fraction of post-lesion frames outside the pre-lesion attractor radius.

    Radius = 2 × mean per-channel std of pre-lesion trajectories.
    """
    pre_flat = pre_trajs.reshape(-1, pre_trajs.shape[-1])
    post_flat = post_trajs.reshape(-1, post_trajs.shape[-1])
    mu = pre_flat.mean(axis=0)
    sigma = pre_flat.std(axis=0).mean() * 2.0 + _STD_GUARD
    dists = np.linalg.norm(post_flat - mu, axis=1)
    return float((dists > sigma).mean())


# ---------------------------------------------------------------------------
# CSV / PNG savers
# ---------------------------------------------------------------------------

def _save_ablation_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = ["node", "delta_lle", "delta_d2", "delta_rho_pct",
                  "procrustes_dist", "ablated_lle", "ablated_d2", "ablated_fc_rho"]
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _save_ablation_hist(rows: List[Dict], path: Path) -> None:
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
        return

    nodes = [r["node"] for r in rows]
    delta_lle = [r["delta_lle"] if not (r["delta_lle"] != r["delta_lle"]) else 0.0
                 for r in rows]
    proc = [r["procrustes_dist"] if not (r["procrustes_dist"] != r["procrustes_dist"]) else 0.0
            for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Node Ablation: Importance Ranking", fontsize=12)

    top_n = min(40, len(nodes))
    axes[0].barh(range(top_n), delta_lle[:top_n], color="steelblue")
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels([str(n) for n in nodes[:top_n]], fontsize=6)
    axes[0].set_xlabel("delta_LLE")
    axes[0].set_title(f"Top-{top_n} nodes by |ΔLLE|")
    axes[0].axvline(0, color="k", linewidth=0.8)

    axes[1].hist(delta_lle, bins=20, color="salmon", edgecolor="white")
    axes[1].axvline(0, color="k", linewidth=0.8)
    axes[1].set_xlabel("delta_LLE")
    axes[1].set_title("Distribution of ΔLLE")
    axes[1].set_ylabel("Count")

    axes[2].hist(proc, bins=20, color="green", alpha=0.7, edgecolor="white")
    axes[2].set_xlabel("Procrustes distance")
    axes[2].set_title("Manifold deformation per node")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)


def _save_lesion_csv(rows: List[Dict], path: Path) -> None:
    fieldnames = ["node", "node_type", "pre_lle", "post_lle", "delta_lle",
                  "pre_d2", "post_d2", "delta_d2",
                  "attractor_escape_rate", "pca_manifold_shift"]
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _save_lesion_plot(rows: List[Dict], path: Path) -> None:
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
        return

    top_rows = [r for r in rows if r["node_type"] == "top"]
    ctrl_rows = [r for r in rows if r["node_type"] == "control"]

    metrics = ["delta_lle", "delta_d2", "attractor_escape_rate", "pca_manifold_shift"]
    labels_m = ["delta_LLE", "delta_D2", "Escape rate", "PCA shift"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5))
    fig.suptitle("Dynamic Lesion: Top nodes vs. controls", fontsize=12)

    for ax, met, lab in zip(axes, metrics, labels_m):
        top_vals = [r[met] for r in top_rows if not np.isnan(r[met])]
        ctrl_vals = [r[met] for r in ctrl_rows if not np.isnan(r[met])]
        if top_vals and ctrl_vals:
            parts = ax.violinplot([top_vals, ctrl_vals], positions=[0, 1],
                                  showmedians=True)
            for pc, c in zip(parts["bodies"], ["steelblue", "salmon"]):
                pc.set_facecolor(c)
                pc.set_alpha(0.7)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Top nodes", "Controls"])
        ax.set_ylabel(lab)
        ax.set_title(lab)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)
