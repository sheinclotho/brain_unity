"""
input_dimension_control.py
==========================
TASK 2: Input Dimension Control Experiment

Tests whether low-dimensional dynamics arise from the **network structure**
rather than from properties of the input drive.

All three conditions operate on the GNN-generated trajectories already
produced in Phase 1 of the pipeline — no additional model inference is
required.

Three conditions
----------------
A  No input (autonomous)
    Raw GNN trajectories from Phase 1.

B  High-dimensional noise
    GNN trajectories with independent Gaussian noise added to every
    channel at every step (σ = noise_sigma).  Mimics a system driven by
    an unstructured full-rank external input.

C  Low-dimensional (3-D) structured drive
    GNN trajectories projected through a random rank-``low_dim_k``
    matrix and re-added.  Mimics a system driven by a structured
    low-rank input.

For each condition the following metrics are estimated:
    D2       — Grassberger–Procaccia correlation dimension
    PCA dim  — fewest PCA components explaining 90 % variance
    LLE      — Largest Lyapunov Exponent (Rosenstein)

Outputs
-------
  input_dimension_results.csv
  input_dimension_plot.png
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_NORM_GUARD = 1e-8   # prevent division by zero when normalising projection columns
_CONDITIONS = ["no_input", "high_dim_noise", "low_dim_drive"]
_CONDITION_LABELS = ["A: No input", "B: High-dim noise", "C: 3-D low-dim drive"]


# ---------------------------------------------------------------------------
# Trajectory modification helpers (no dynamics model needed)
# ---------------------------------------------------------------------------

def _add_high_dim_noise(
    trajectories: np.ndarray,
    sigma: float,
    seed: int,
) -> np.ndarray:
    """Add i.i.d. Gaussian noise N(0, sigma²) to every channel at every step."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(trajectories.shape).astype(np.float32) * sigma
    return (trajectories + noise).astype(np.float32)


def _add_low_dim_drive(
    trajectories: np.ndarray,
    low_dim_k: int,
    seed: int,
) -> np.ndarray:
    """Mix trajectories with a structured rank-k signal.

    Constructs a random projection P ∈ ℝ^{N×k}, projects all activity
    down to k dimensions, then projects back up to N.  The result is
    added to the original trajectories, so the observable dimensionality
    is dominated by the structured k-dimensional component.
    """
    rng = np.random.default_rng(seed)
    N = trajectories.shape[-1]
    # Random projection matrix (each column unit-normalised)
    P = rng.standard_normal((N, low_dim_k)).astype(np.float32)
    P /= np.linalg.norm(P, axis=0, keepdims=True) + _NORM_GUARD  # (N, k)

    # Project all frames: (n_traj, T, N) → (n_traj, T, k) → (n_traj, T, N)
    flat = trajectories.reshape(-1, N)          # (n_traj*T, N)
    low = flat @ P                              # (n_traj*T, k)
    drive = (low @ P.T).reshape(trajectories.shape)  # (n_traj, T, N)
    return (trajectories + drive).astype(np.float32)


# ---------------------------------------------------------------------------
# Metric helpers (trajectory-based, no model calls)
# ---------------------------------------------------------------------------

def _compute_d2(trajs: np.ndarray) -> float:
    """Estimate correlation dimension D2.

    Passes the first trajectory as a 2-D array (T × N) to
    ``correlation_dimension``, which requires a 2-D input.
    """
    try:
        from analysis.embedding_dimension import correlation_dimension
        traj = trajs[0].astype(np.float64)   # (T, N)
        result = correlation_dimension(traj, max_points=2000)
        return float(result["D2"])
    except (ImportError, KeyError, ValueError) as exc:
        logger.debug("D2 failed: %s", exc)
        return float("nan")
    except Exception as exc:
        logger.debug("D2 unexpected error: %s", exc)
        return float("nan")


def _compute_pca_dim(trajs: np.ndarray, var_threshold: float = 0.90) -> int:
    X = trajs.reshape(-1, trajs.shape[-1])
    X = X - X.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        v = s ** 2
        return int(np.searchsorted(np.cumsum(v) / (v.sum() + 1e-30), var_threshold) + 1)
    except Exception:
        return -1


def _compute_lle(trajs: np.ndarray) -> float:
    """Estimate LLE using shared Rosenstein helper from random_comparison."""
    from analysis.random_comparison import avg_rosenstein_lle
    lle = avg_rosenstein_lle(trajs)
    if np.isnan(lle):
        logger.debug("LLE returned NaN for this condition.")
    return lle
# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_input_dimension_control(
    trajectories: np.ndarray,
    noise_sigma: float = 0.5,
    low_dim_k: int = 3,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Run the input-dimension control experiment on GNN-generated trajectories.

    Parameters
    ----------
    trajectories:
        GNN-generated trajectories from Phase 1, shape ``(n_traj, T, N)``.
    noise_sigma:
        Noise amplitude for condition B (high-dimensional noise).
    low_dim_k:
        Dimensionality of the structured drive added in condition C.
    seed:
        Random seed for noise/projection generation.
    output_dir:
        Directory to save CSV and PNG.

    Returns
    -------
    dict  — keys ``'no_input'``, ``'high_dim_noise'``, ``'low_dim_drive'``,
    each containing ``{'d2', 'pca_dim', 'lle'}``.
    """
    trajs = np.asarray(trajectories, dtype=np.float32)
    n_traj, T, N = trajs.shape
    logger.info("Input dimension control: n_traj=%d, T=%d, N=%d", n_traj, T, N)

    modified: Dict[str, np.ndarray] = {
        "no_input": trajs,
        "high_dim_noise": _add_high_dim_noise(trajs, sigma=noise_sigma, seed=seed),
        "low_dim_drive": _add_low_dim_drive(trajs, low_dim_k=low_dim_k, seed=seed + 1),
    }

    results: Dict[str, Dict] = {}
    for cond, label in zip(_CONDITIONS, _CONDITION_LABELS):
        t = modified[cond]
        d2 = _compute_d2(t)
        pca_dim = _compute_pca_dim(t)
        lle = _compute_lle(t)
        logger.info("  %s: D2=%.2f  PCA_dim=%d  LLE=%.5f", label, d2, pca_dim, lle)
        results[cond] = {"condition": label, "d2": d2, "pca_dim": pca_dim, "lle": lle}

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(results, output_dir / "input_dimension_results.csv")
        _save_plot(results, output_dir / "input_dimension_plot.png")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(results: Dict, path: Path) -> None:
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "d2", "pca_dim", "lle"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(results.values())
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _save_plot(results: Dict, path: Path) -> None:
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

    labels = [r["condition"] for r in results.values()]
    lle_vals = [r["lle"] for r in results.values()]
    d2_vals = [r["d2"] for r in results.values()]
    pca_vals = [r["pca_dim"] for r in results.values()]
    colors = ["steelblue", "salmon", "green"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Input Dimension Control Experiment (GNN trajectories)", fontsize=11)

    for ax, vals, ylabel, title in [
        (axes[0], lle_vals, "LLE", "Largest Lyapunov Exponent"),
        (axes[1], d2_vals, "D2", "Correlation Dimension D2"),
        (axes[2], pca_vals, "PCA dim (90% var)", "PCA Intrinsic Dimension"),
    ]:
        ax.bar(x, vals, color=colors, width=0.5)
        if ax is axes[0]:
            ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)
