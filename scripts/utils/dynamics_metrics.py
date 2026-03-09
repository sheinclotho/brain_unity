"""
dynamics_metrics.py
-------------------
Thin wrappers around brain_dynamics.analysis functions for computing
core dynamics metrics.  Import from here in experiment scripts so that
you don't need to know the internal module layout.

Functions
---------
compute_LLE(trajectories, ...)
    Largest Lyapunov Exponent (Rosenstein or Wolf method).

compute_correlation_dimension(trajectories, ...)
    Grassberger–Procaccia correlation dimension D₂.

compute_intrinsic_dim_PCA(trajectories, ...)
    PCA intrinsic dimension: fewest components that explain >= var_threshold
    of total variance.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "brain_dynamics"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Largest Lyapunov Exponent
# ---------------------------------------------------------------------------

def compute_LLE(
    trajectories: np.ndarray,
    method: str = "rosenstein",
    max_lag: int = 50,
    min_sep: int = 20,
    n_segments: int = 1,
) -> float:
    """Compute the Largest Lyapunov Exponent from trajectories.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)`` (single trajectory).
    method:
        ``'rosenstein'`` (default), ``'wolf'``, or ``'ftle'``.
    max_lag:
        Maximum divergence lag for Rosenstein / Wolf method.
    min_sep:
        Minimum neighbour separation (Rosenstein).
    n_segments:
        Number of trajectory segments; averages LLE across segments.

    Returns
    -------
    float
        Mean LLE across all trajectories.  ``nan`` on failure.
    """
    from analysis.lyapunov import run_lyapunov_analysis

    trajs = np.asarray(trajectories, dtype=float)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]
    result = run_lyapunov_analysis(
        trajectories=trajs,
        method=method,
        max_lag=max_lag,
        min_sep=min_sep,
        n_segments=n_segments,
    )
    return float(result.get("primary_mean", float("nan")))


# ---------------------------------------------------------------------------
# Correlation Dimension D₂
# ---------------------------------------------------------------------------

def compute_correlation_dimension(
    trajectories: np.ndarray,
    max_dim: int = 20,
    n_points: int = 2000,
    channel: Optional[int] = None,
) -> float:
    """Estimate the Grassberger–Procaccia correlation dimension D₂.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)``.
    max_dim:
        Maximum embedding dimension for the delay-embedding.
    n_points:
        Number of points sampled from the concatenated trajectory (for speed).
    channel:
        Which spatial channel to use (default: 0).

    Returns
    -------
    float
        D₂ estimate.  ``nan`` on failure.
    """
    from analysis.embedding_dimension import correlation_dimension

    trajs = np.asarray(trajectories, dtype=float)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]
    ch = channel if channel is not None else 0
    flat = trajs[:, :, ch].reshape(-1)
    try:
        return float(correlation_dimension(flat, max_dim=max_dim, n_points=n_points))
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# PCA Intrinsic Dimension
# ---------------------------------------------------------------------------

def compute_intrinsic_dim_PCA(
    trajectories: np.ndarray,
    var_threshold: float = 0.90,
    burnin: int = 0,
) -> int:
    """Number of PCA components needed to explain ``var_threshold`` variance.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)``.
    var_threshold:
        Cumulative variance threshold, e.g. ``0.90`` for 90 %.
    burnin:
        Discard the first ``burnin`` time-steps from each trajectory.

    Returns
    -------
    int
        PCA intrinsic dimension.  ``-1`` on failure.
    """
    trajs = np.asarray(trajectories, dtype=float)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]
    if burnin > 0:
        trajs = trajs[:, burnin:, :]
    X = trajs.reshape(-1, trajs.shape[-1])
    X = X - X.mean(axis=0)
    try:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        var = s ** 2
        cumvar = np.cumsum(var) / (var.sum() + 1e-30)
        return int(np.searchsorted(cumvar, var_threshold) + 1)
    except Exception:
        return -1
