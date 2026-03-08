"""
Wilson-Cowan Matrix Dynamics Utilities
=======================================

This module belongs to **spectral_dynamics** — the matrix-driven analysis
layer that is completely independent of the TwinBrain GNN model.

Background
----------
The ``spectral_dynamics`` experiments (E4, E5, I) use Wilson-Cowan (WC)
dynamics as a *matrix-proxy* to study how properties of the connectivity
matrix affect brain-like dynamics without requiring the full GNN model:

* **E4** — structural perturbation: does WC-LLE change when the matrix is
  shuffled / rewired / truncated?
* **E5** — phase diagram: scan coupling strength ``g`` to find the edge-of-chaos
  transition point.
* **I** — energy constraint: how does an L1 energy budget alter WC dynamics?

These experiments are distinct from twinbrain-dynamics which uses the actual
``GraphNativeBrainModel`` (ST-GCN encoder + GNN propagator + decoder). WC is
the simple proxy for matrix analysis; the GNN is the real model.

Architecture note
-----------------
This file lives in ``spectral_dynamics/`` because WC dynamics are exclusively
a tool of the spectral analysis layer. The ``twinbrain-dynamics/`` package
deals only with the GNN-based model and must not contain WC code.

The Rosenstein LLE helper calls ``analysis.lyapunov.rosenstein_lyapunov`` from
``twinbrain-dynamics`` when it is available on ``sys.path`` (which
``spectral_dynamics/__init__._ensure_twinbrain_path()`` arranges at package
import time), and falls back to :func:`wolf_benettin_lle` otherwise.

Previously duplicated code (now consolidated here)
---------------------------------------------------
+------------------------------------------+---------------------------------+
| 旧位置（已删除）                          | 新统一位置                        |
+==========================================+=================================+
| e4._wc_step                              | wc_dynamics.wc_step             |
| e5._wc_trajectories                      | wc_dynamics.wc_simulate         |
| e4._rosenstein_lle_on_wc / _simple_lle   | wc_dynamics.rosenstein_lle_on_wc|
| e5._rosenstein_from_twinbrain /          |                                 |
|   _simple_rosenstein                     | wc_dynamics.rosenstein_lle_on_wc|
| i._rosenstein                            | wc_dynamics.rosenstein_lle_on_wc|
| i._project_energy_wc                     | wc_dynamics.project_energy_l1_bounded |
+------------------------------------------+---------------------------------+

Public API
----------
``wc_step(x, W, g)``                   — single WC step
``wc_simulate(W, ...)``                 — generate trajectories
``rosenstein_lle_on_wc(W, ...)``        — LLE via Rosenstein method
``project_energy_l1_bounded(y, E)``     — L1-ball energy projection (bounded)
``wolf_benettin_lle(traj)``             — Wolf-Benettin fallback (rarely needed)
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "wc_step",
    "wc_simulate",
    "rosenstein_lle_on_wc",
    "project_energy_l1_bounded",
    "wolf_benettin_lle",
]


# ─────────────────────────────────────────────────────────────────────────────
# Core WC dynamics
# ─────────────────────────────────────────────────────────────────────────────

def wc_step(x: np.ndarray, W: np.ndarray, g: float = 1.0) -> np.ndarray:
    """
    Single Wilson-Cowan step: ``clip(tanh(g * W @ x), 0, 1)``.

    Args:
        x:  State vector (N,), float32 or float64.
        W:  Connectivity matrix (N, N).
        g:  Coupling strength scalar (default 1.0).

    Returns:
        x_next: (N,) float64, clipped to [0, 1].
    """
    return np.clip(np.tanh(g * (W @ x)), 0.0, 1.0)


def wc_simulate(
    W: np.ndarray,
    n_traj: int = 20,
    steps: int = 300,
    g: float = 1.0,
    warmup: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    Simulate Wilson-Cowan trajectories from random initial states.

    Args:
        W:       Connectivity matrix (N, N).
        n_traj:  Number of trajectories.
        steps:   Recording steps per trajectory (after warmup).
        g:       Coupling strength.
        warmup:  Warm-up steps (not recorded, used to reach attractor).
        seed:    Random seed.

    Returns:
        trajs: shape (n_traj, steps, N), float32.
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    gW = (g * W).astype(np.float64)
    trajs = np.empty((n_traj, steps, N), dtype=np.float32)

    for i in range(n_traj):
        x = rng.random(N)
        for _ in range(warmup):
            x = np.clip(np.tanh(gW @ x), 0.0, 1.0)
        for t in range(steps):
            trajs[i, t] = x.astype(np.float32)
            x = np.clip(np.tanh(gW @ x), 0.0, 1.0)

    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# LLE on WC trajectories
# ─────────────────────────────────────────────────────────────────────────────

def rosenstein_lle_on_wc(
    W: np.ndarray,
    n_traj: int = 30,
    steps: int = 300,
    g: float = 0.9,
    max_lag: int = 30,
    min_sep: int = 10,
    warmup: int = 50,
    seed: int = 42,
) -> float:
    """
    Estimate the maximum Lyapunov exponent for a WC system via Rosenstein's
    method.

    Generates ``n_traj`` trajectories with :func:`wc_simulate` and calls the
    canonical ``rosenstein_lyapunov`` from
    ``twinbrain-dynamics/analysis/lyapunov`` (available when
    ``spectral_dynamics._ensure_twinbrain_path()`` has been called).  Falls
    back to :func:`wolf_benettin_lle` otherwise.

    Args:
        W:        Connectivity matrix (N, N).
        n_traj:   Number of trajectories for averaging.
        steps:    Steps per trajectory.
        g:        WC coupling strength.
        max_lag:  Rosenstein tracking lag.
        min_sep:  Minimum temporal separation for nearest-neighbour search.
        warmup:   Warm-up steps (not recorded).
        seed:     Random seed.

    Returns:
        Mean LLE over valid trajectories; ``float("nan")`` if estimation fails.
    """
    trajs = wc_simulate(W, n_traj=n_traj, steps=steps, g=g,
                        warmup=warmup, seed=seed)

    try:
        from analysis.lyapunov import rosenstein_lyapunov
        lles = [
            rosenstein_lyapunov(trajs[i], max_lag=max_lag,
                                min_temporal_sep=min_sep)[0]
            for i in range(n_traj)
        ]
    except ImportError:
        logger.warning(
            "wc_dynamics: rosenstein_lyapunov not importable; "
            "using Wolf-Benettin fallback."
        )
        lles = [wolf_benettin_lle(trajs[i]) for i in range(n_traj)]

    valid = [v for v in lles if np.isfinite(v)]
    return float(np.mean(valid)) if valid else float("nan")


def wolf_benettin_lle(
    traj: np.ndarray,
    renorm_steps: int = 20,
    eps: float = 1e-6,
) -> float:
    """
    Lightweight Wolf-Benettin LLE on a pre-computed WC trajectory.

    Used only when ``analysis.lyapunov.rosenstein_lyapunov`` is not available.
    This is the single canonical fallback for WC-based LLE estimation; the
    duplicate copies that previously lived in e4, e5, and i have been removed.

    Args:
        traj:          shape (T, N), float32/64.
        renorm_steps:  Re-normalisation interval.
        eps:           Initial perturbation size.

    Returns:
        LLE estimate (per step).
    """
    T, N = traj.shape
    log_growths: list[float] = []

    rng = np.random.default_rng(0)
    x = traj[0].astype(np.float64)
    d = rng.standard_normal(N)
    d /= np.linalg.norm(d) + 1e-30
    x_p = x + eps * d

    # Replay the trajectory, treating it as the reference orbit.
    for t in range(1, T):
        # Advance perturbed state using finite difference:
        # x_p → x_p + (traj[t] - traj[t-1])  (orbit-parallel transport)
        delta_traj = traj[t].astype(np.float64) - traj[t - 1].astype(np.float64)
        x_p = x_p + delta_traj

        if (t % renorm_steps) == 0:
            delta = x_p - traj[t].astype(np.float64)
            r = np.linalg.norm(delta)
            if r > 1e-30:
                log_growths.append(np.log(r / eps))
                x_p = traj[t].astype(np.float64) + eps * delta / r

    if not log_growths:
        return float("nan")
    return float(np.mean(log_growths)) / renorm_steps


# ─────────────────────────────────────────────────────────────────────────────
# L1-ball energy projection (bounded WC state space [0, 1]^N)
# ─────────────────────────────────────────────────────────────────────────────

def project_energy_l1_bounded(
    y: np.ndarray,
    E_budget: float,
) -> Tuple[np.ndarray, bool]:
    """
    Project WC output ``y ∈ [0, 1]^N`` onto ``{z : mean(z) ≤ E_budget}``.

    This is the bounded (non-negative) specialisation used by the WC energy
    constraint experiment (``spectral_dynamics/i_energy_constraint.py``).

    The general version — which also handles unbounded z-score joint states —
    lives in ``twinbrain-dynamics/experiments/energy_constraint._project_energy``
    and should be used when the state space may be negative.

    Projection solution (soft-threshold, 50-iteration bisection):
      ``x_i = max(y_i - λ*, 0)``  where ``λ*`` makes ``mean(x) = E_budget``.

    Args:
        y:         WC output, shape (N,), values in [0, 1].
        E_budget:  Energy budget (L1 mean upper bound).

    Returns:
        (x_projected, constraint_was_active):
            x_projected:           shape (N,), float32.
            constraint_was_active: True if the constraint was binding.

    Cost: O(50 × N) per call — < 0.1 ms for N = 190.
    """
    current = float(y.mean())
    if current <= E_budget:
        return y.astype(np.float32), False

    lo, hi = 0.0, float(y.max())
    for _ in range(50):
        lam = (lo + hi) * 0.5
        proj = np.maximum(y - lam, 0.0)
        if float(proj.mean()) <= E_budget:
            hi = lam
        else:
            lo = lam

    result = np.clip(np.maximum(y - (lo + hi) * 0.5, 0.0), 0.0, 1.0)
    return result.astype(np.float32), True
