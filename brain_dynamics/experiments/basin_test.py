"""
Attractor Basin Test — Experiment 2
=====================================

Tests whether the system has a **single global attractor** by initialising
trajectories from three diverse starting distributions:

  1. **Task states** — from the data (natural initial conditions, narrow distribution)
  2. **Gaussian random** — N(0, σ) states (wide isotropic distribution)
  3. **Uniform random** — uniform in state bounds (maximally diverse)

All three populations should converge to the same attractor region if a
single global attractor exists.

Method:
  1. Generate ``n_diverse`` trajectories per init type.
  2. Run ``T`` prediction steps.
  3. Compute tail-state means (last ``tail_steps`` steps).
  4. PCA-project to 3D; save ``basin_pca_projection.npy``.
  5. K-means cluster all tail states together; check if one cluster dominates.

Judgment:
  - ``single_attractor``: dominant cluster fraction > ``dominant_thresh`` (default 0.75)
  - This is assessed separately for each init type AND for the pooled population.

Outputs
-------
basin_pca_projection.npy    — shape (n_traj_total, 3) PCA projections
basin_cluster_report.json   — cluster fractions + judgment

Note: This experiment requires additional GNN model calls
(n_diverse × 3 init types × T steps).  It is **disabled by default** in
config.yaml.  Enable with ``dynamics.basin_test.enabled: true``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_DOMINANT_THRESH = 0.75   # Fraction of trajectories in dominant cluster


def _run_trajectories(
    simulator,
    init_states: np.ndarray,
    T: int,
    device: str = "cpu",
) -> np.ndarray:
    """
    Roll out the simulator from a batch of initial states.

    Parameters
    ----------
    simulator:
        BrainDynamicsSimulator instance.
    init_states:
        shape (n_traj, N) — initial states.
    T:
        Number of prediction steps.
    device:
        Unused and deprecated.  The compute device is fixed at simulator
        construction time (``BrainDynamicsSimulator.device``).  The parameter
        is kept for backward-compatible call sites only; passing it has no
        effect and will raise a warning.

    Returns
    -------
    trajectories: shape (n_traj, T, N).
    """
    import warnings
    if device != "cpu":
        warnings.warn(
            "_run_trajectories: the `device` parameter is unused — "
            "the compute device is fixed at BrainDynamicsSimulator construction "
            "time.  Remove `device=` from your call site.",
            DeprecationWarning,
            stacklevel=2,
        )
    from experiments.free_dynamics import run_free_dynamics
    n_traj, N = init_states.shape
    trajs = np.zeros((n_traj, T, N), dtype=np.float32)
    for i in range(n_traj):
        x0 = init_states[i]
        # rollout() returns (trajectory, times); device is set at simulator init.
        traj, _ = simulator.rollout(x0=x0, steps=T)
        trajs[i] = traj
    return trajs


def _generate_init_states(
    simulator,
    n: int,
    init_type: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate n initial states of a given type.

    Parameters
    ----------
    simulator:
        BrainDynamicsSimulator (used for task-state sampling and state_bounds).
    n:
        Number of states to generate.
    init_type:
        One of: "task", "gaussian", "uniform".
    rng:
        Random number generator.

    Returns
    -------
    states: shape (n, N).
    """
    N = simulator.n_regions
    bounds = simulator.state_bounds  # None for z-score, (0,1) for bounded

    if init_type == "task":
        states = np.stack([
            simulator.sample_random_state(rng, from_data=True)
            for _ in range(n)
        ], axis=0)

    elif init_type == "gaussian":
        # N(0, 1) in z-score space or N(0.5, 0.2) in [0,1] space
        if bounds is None:
            states = rng.standard_normal((n, N)).astype(np.float32)
        else:
            lo, hi = bounds
            states = rng.standard_normal((n, N)).astype(np.float32) * 0.2 + (lo + hi) / 2
            states = np.clip(states, lo, hi)

    elif init_type == "uniform":
        if bounds is None:
            # Uniform in [-3σ, +3σ] range for z-score
            states = (rng.random((n, N)).astype(np.float32) * 6.0 - 3.0)
        else:
            lo, hi = bounds
            states = (rng.random((n, N)).astype(np.float32) * (hi - lo) + lo)

    else:
        raise ValueError(f"Unknown init_type: {init_type!r}")

    return states


def run_basin_test(
    simulator,
    n_diverse: int = 50,
    T: int = 300,
    tail_steps: int = 50,
    dominant_thresh: float = _DOMINANT_THRESH,
    seed: int = 42,
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Attractor Basin Test (Experiment 2).

    Tests for single global attractor by using diverse initial conditions.

    Parameters
    ----------
    simulator:
        BrainDynamicsSimulator instance.
    n_diverse:
        Number of trajectories per initialisation type (task / Gaussian / uniform).
    T:
        Prediction steps per trajectory.
    tail_steps:
        Steps used for tail-state extraction (must be ≤ T).
    dominant_thresh:
        Fraction threshold for "single attractor" verdict.
    seed:
        Random seed.
    device:
        Compute device.
    output_dir:
        Directory for outputs.

    Returns
    -------
    Dict with keys:
        per_type_cluster_fractions  dict per init type
        pooled_dominant_fraction    float fraction in largest cluster
        n_trajectories_total        int
        single_attractor            bool
        judgment                    str
        pca_projection              np.ndarray (n_total, 3)
    """
    rng = np.random.default_rng(seed)
    tail_steps = min(tail_steps, T)
    init_types = ["task", "gaussian", "uniform"]
    N = simulator.n_regions

    all_trajs: List[np.ndarray] = []
    type_labels: List[str] = []

    for init_type in init_types:
        logger.info("  Basin test: generating %d '%s' trajectories (T=%d)...",
                    n_diverse, init_type, T)
        try:
            init_states = _generate_init_states(simulator, n_diverse, init_type, rng)
            trajs = _run_trajectories(simulator, init_states, T, device=device)
            all_trajs.append(trajs)
            type_labels.extend([init_type] * n_diverse)
        except Exception as e:
            logger.warning("  Basin test: %s init failed: %s", init_type, e)

    if not all_trajs:
        return {"error": "all_init_types_failed"}

    all_trajs_arr = np.concatenate(all_trajs, axis=0)   # (n_total, T, N)
    n_total = all_trajs_arr.shape[0]

    # Extract tail states
    tail_states = all_trajs_arr[:, -tail_steps:, :].mean(axis=1)  # (n_total, N)

    # PCA projection (3D)
    try:
        from sklearn.decomposition import PCA as _PCA
        pca = _PCA(n_components=min(3, N, n_total - 1))
        pca_proj = pca.fit_transform(tail_states)  # (n_total, 3)
    except ImportError:
        # Manual SVD fallback
        flat = tail_states - tail_states.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        k = min(3, Vt.shape[0])
        pca_proj = flat @ Vt[:k].T
    except Exception as e:
        logger.warning("  Basin test PCA failed: %s", e)
        pca_proj = tail_states[:, :3].copy()

    # K-means clustering on tail states
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        best_k = 1
        best_sil = -1.0
        best_labels = np.zeros(n_total, dtype=int)
        for k_test in [2, 3, 4, 5]:
            if n_total <= k_test:
                continue
            km = KMeans(n_clusters=k_test, n_init=10, random_state=seed)
            labels = km.fit_predict(tail_states)
            if len(np.unique(labels)) < 2:
                continue
            sil = float(silhouette_score(tail_states, labels))
            if sil > best_sil:
                best_sil = sil
                best_k = k_test
                best_labels = labels
        cluster_labels = best_labels
        n_clusters = best_k
    except ImportError:
        cluster_labels = np.zeros(n_total, dtype=int)
        n_clusters = 1
        best_sil = float("nan")

    # Cluster fractions (pooled)
    unique_clusters, counts = np.unique(cluster_labels, return_counts=True)
    cluster_fractions = {
        int(c): round(float(cnt / n_total), 3)
        for c, cnt in zip(unique_clusters, counts)
    }
    dominant_fraction = float(counts.max() / n_total)

    # Per-type cluster fractions
    type_arr = np.array(type_labels)
    per_type: Dict[str, Any] = {}
    for it in init_types:
        mask = type_arr == it
        if mask.sum() == 0:
            continue
        t_labels = cluster_labels[mask]
        t_unique, t_counts = np.unique(t_labels, return_counts=True)
        per_type[it] = {
            "n_trajectories": int(mask.sum()),
            "cluster_fractions": {
                int(c): round(float(cnt / mask.sum()), 3)
                for c, cnt in zip(t_unique, t_counts)
            },
            "dominant_fraction": round(float(t_counts.max() / mask.sum()), 3),
        }

    # Single attractor judgment
    # Both pooled dominant fraction AND per-type dominant fractions
    type_dom_fracs = [v["dominant_fraction"] for v in per_type.values()]
    min_type_dom = min(type_dom_fracs) if type_dom_fracs else 0.0
    single_attractor = bool(dominant_fraction > dominant_thresh
                            and min_type_dom > dominant_thresh * 0.8)

    if single_attractor:
        judgment = (
            f"single global attractor (pooled dominant={dominant_fraction:.1%}, "
            f"min_per_type={min_type_dom:.1%} > threshold={dominant_thresh:.0%})"
        )
    else:
        judgment = (
            f"possible multi-attractor or slow mixing "
            f"(pooled dominant={dominant_fraction:.1%} < threshold={dominant_thresh:.0%})"
        )

    logger.info(
        "  Basin test: n=%d, n_clusters=%d, dominant=%.1f%%, single=%s",
        n_total, n_clusters, dominant_fraction * 100, single_attractor,
    )

    result = {
        "n_trajectories_total": n_total,
        "n_trajectories_per_type": n_diverse,
        "T": T,
        "tail_steps": tail_steps,
        "n_clusters": n_clusters,
        "silhouette_score": float(best_sil) if np.isfinite(best_sil) else None,
        "pooled_cluster_fractions": cluster_fractions,
        "pooled_dominant_fraction": round(dominant_fraction, 3),
        "per_type_results": per_type,
        "single_attractor": single_attractor,
        "judgment": judgment,
        "pca_projection": pca_proj,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "basin_pca_projection.npy", pca_proj.astype(np.float32))

        report = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        with open(output_dir / "basin_cluster_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info("  Basin test outputs saved to: %s", output_dir)

    return result
