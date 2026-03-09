"""
manifold_metrics.py
-------------------
Geometric distances between dynamical manifolds (PCA projections of
brain-activity trajectories).

Functions
---------
procrustes_distance(X, Y, n_components)
    Normalised Procrustes disparity between two trajectory clouds.

hausdorff_distance(X, Y)
    Symmetric Hausdorff distance between two finite point sets.
"""
from __future__ import annotations

import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Procrustes distance
# ---------------------------------------------------------------------------

def procrustes_distance(
    X: np.ndarray,
    Y: np.ndarray,
    n_components: int = 10,
) -> float:
    """Normalised Procrustes distance between two trajectory sets in PCA space.

    Both ``X`` and ``Y`` are projected to a joint PCA basis of dimension
    ``n_components``, unit-normalised by Frobenius norm, and then optimally
    rotated/reflected via SciPy's Procrustes.  The returned disparity is in
    [0, 1] (0 = identical manifolds).

    Parameters
    ----------
    X, Y:
        Shape ``(n_traj, T, N)`` or ``(n_frames, N)``.
    n_components:
        Dimension of the shared PCA embedding.

    Returns
    -------
    float
        Procrustes disparity (lower = more similar manifolds).
    """
    def _flatten(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr

    Xf = _flatten(X)
    Yf = _flatten(Y)

    # Joint PCA basis
    joint = np.vstack([Xf, Yf])
    joint_mean = joint.mean(axis=0)
    joint_centered = joint - joint_mean
    try:
        _, _, Vt = np.linalg.svd(joint_centered, full_matrices=False)
        components = Vt[:n_components]
    except np.linalg.LinAlgError:
        components = np.eye(min(n_components, joint_centered.shape[1]))

    Xp = (Xf - joint_mean) @ components.T
    Yp = (Yf - joint_mean) @ components.T

    # Match lengths
    n = min(len(Xp), len(Yp))
    Xp = Xp[:n]
    Yp = Yp[:n]

    # Try scipy first, fallback to manual
    try:
        from scipy.spatial import procrustes as _scipy_procrustes
        _, _, disparity = _scipy_procrustes(Xp, Yp)
        return float(disparity)
    except Exception:
        pass

    # Manual: normalise both to unit Frobenius norm
    nx = np.linalg.norm(Xp)
    ny = np.linalg.norm(Yp)
    if nx < 1e-12 or ny < 1e-12:
        return float("nan")
    Xn = Xp / nx
    Yn = Yp / ny

    # Optimal rotation via SVD of cross-covariance
    M = Yn.T @ Xn
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    Yr = Yn @ R.T

    disparity = float(np.linalg.norm(Xn - Yr) ** 2)
    return disparity


# ---------------------------------------------------------------------------
# Hausdorff distance
# ---------------------------------------------------------------------------

def hausdorff_distance(
    X: np.ndarray,
    Y: np.ndarray,
) -> float:
    """Symmetric Hausdorff distance between two point clouds.

    Parameters
    ----------
    X, Y:
        Shape ``(n_x, d)`` and ``(n_y, d)``.

    Returns
    -------
    float
        max(directed_hausdorff(X→Y), directed_hausdorff(Y→X)).
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    try:
        from scipy.spatial.distance import directed_hausdorff
        dxy = directed_hausdorff(X, Y)[0]
        dyx = directed_hausdorff(Y, X)[0]
        return float(max(dxy, dyx))
    except ImportError:
        pass

    # Pure-numpy fallback (O(n²) – only for small arrays)
    d_xy = np.array([np.min(np.linalg.norm(Y - x, axis=1)) for x in X])
    d_yx = np.array([np.min(np.linalg.norm(X - y, axis=1)) for y in Y])
    return float(max(d_xy.max(), d_yx.max()))
