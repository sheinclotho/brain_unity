"""
intrinsic_dimension — Local Intrinsic Dimension Estimation
===========================================================

Implements the **TwoNN (Two Nearest Neighbours)** estimator of local intrinsic
dimension (Facco et al., 2017, *Scientific Reports* 7:12140) and a thin
wrapper for integration with the brain dynamics pipeline.

The key insight of TwoNN is that the ratio ``μ = r₂ / r₁`` (second-nearest
to first-nearest neighbour distance) follows a Pareto distribution whose
shape parameter *d* is the local intrinsic dimension.  By fitting the
empirical CDF of ``μ`` over a log-log range, we obtain a robust, parameter-
free estimate that is accurate even for high-dimensional embeddings and small
sample sizes.

Public API
----------
* :func:`twonn_dimension`            — Main estimator.
* :func:`run_intrinsic_dimension`    — Pipeline-style wrapper.

References
----------
Facco E., d'Errico M., Rodriguez A., Laio A. (2017).
Estimating the intrinsic dimension of datasets by a minimal neighborhood
information. *Sci Rep* **7**, 12140. https://doi.org/10.1038/s41598-017-11873-y
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["twonn_dimension", "run_intrinsic_dimension"]

# ──────────────────────────────────────────────────────────────────────────────
#  Core estimator
# ──────────────────────────────────────────────────────────────────────────────

def twonn_dimension(
    X: np.ndarray,
    *,
    fraction: float = 0.90,
    max_points: int = 5_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    TwoNN intrinsic dimension estimator (Facco et al., 2017).

    Parameters
    ----------
    X : ndarray, shape (T, N)
        Point cloud (e.g. brain-state trajectory; T time points, N features).
    fraction : float
        Fraction of the empirical CDF used for Pareto regression
        (the low end is discarded to avoid boundary effects).
    max_points : int
        Sub-sample if T > max_points to keep O(T²) kNN tractable.
    seed : int
        Random seed for reproducible sub-sampling.

    Returns
    -------
    dict with keys:
        ``d_twonn``   — Estimated intrinsic dimension (float).
        ``d_upper``   — Upper confidence bound (±1σ from slope fit).
        ``d_lower``   — Lower confidence bound.
        ``fit_r2``    — R² of the Pareto log-log linear fit.
        ``n_used``    — Number of points used after sub-sampling.
        ``mu_values`` — Raw µ ratios (list of float).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    T, N = X.shape

    # Sub-sample if too many points
    if T > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(T, size=max_points, replace=False)
        X = X[idx]
        T = max_points

    # Compute pairwise distances and find two nearest neighbours
    # Using vectorised broadcasting for moderate T; for large T use scipy.
    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        dists, _ = tree.query(X, k=3)  # k=3 because first is self (dist=0)
        r1 = dists[:, 1]
        r2 = dists[:, 2]
    except ImportError:
        # Fallback: O(T²) brute force
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (T, T, N)
        dist_mat = np.sqrt((diff ** 2).sum(axis=-1))      # (T, T)
        np.fill_diagonal(dist_mat, np.inf)
        sorted_d = np.sort(dist_mat, axis=1)
        r1 = sorted_d[:, 0]
        r2 = sorted_d[:, 1]

    # Exclude degenerate points (r1 ≈ 0)
    valid = r1 > 1e-15
    r1 = r1[valid]
    r2 = r2[valid]
    mu = r2 / np.maximum(r1, 1e-30)

    # Filter mu ≤ 1 (can happen with duplicate points)
    mu = mu[mu > 1.0]
    n_used = len(mu)

    if n_used < 10:
        return {
            "d_twonn": float("nan"),
            "d_upper": float("nan"),
            "d_lower": float("nan"),
            "fit_r2": float("nan"),
            "n_used": n_used,
            "mu_values": [],
            "error": "Too few valid points for TwoNN fit.",
        }

    # Empirical CDF of log(µ)
    mu_sorted = np.sort(mu)
    n = len(mu_sorted)
    # Empirical Pareto CDF: F(µ) = 1 - µ^(-d)  →  log(1-F) = -d·log(µ)
    # Use the middle `fraction` of the sorted data (skip extremes)
    lo = int((1.0 - fraction) * n)
    hi = int(fraction * n)
    if hi <= lo + 2:
        hi = n
        lo = 0

    x_fit = np.log(mu_sorted[lo:hi])
    y_fit = np.log(1.0 - np.arange(lo, hi) / n)  # log(1 - F_emp)

    # Ordinary least-squares: y = -d * x  (intercept ≈ 0)
    # Use pinv for robustness
    A = x_fit[:, np.newaxis]
    slope, _, _, _ = np.linalg.lstsq(A, y_fit, rcond=None)
    d_hat = float(-slope[0])

    # Residual for R² and confidence
    y_pred = slope[0] * x_fit
    ss_res = float(np.sum((y_fit - y_pred) ** 2))
    ss_tot = float(np.sum((y_fit - y_fit.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-30 else float("nan")

    # 1σ uncertainty from regression variance
    n_fit = len(x_fit)
    if n_fit > 2 and ss_res > 0:
        sigma2 = ss_res / (n_fit - 1)
        xx = float(np.sum(x_fit ** 2))
        d_std = float(np.sqrt(sigma2 / xx)) if xx > 1e-30 else 0.0
    else:
        d_std = 0.0

    return {
        "d_twonn": round(d_hat, 3),
        "d_upper": round(d_hat + d_std, 3),
        "d_lower": round(max(0.0, d_hat - d_std), 3),
        "fit_r2": round(r2, 4) if np.isfinite(r2) else float("nan"),
        "n_used": n_used,
        "mu_values": mu_sorted.tolist(),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Ensemble over multiple trajectories
# ──────────────────────────────────────────────────────────────────────────────

def _twonn_ensemble(
    trajectories: np.ndarray,
    k_traj: int = 10,
    burnin_fraction: float = 0.10,
    max_points: int = 2_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Run TwoNN on *k_traj* randomly selected trajectories and aggregate.

    Returns a dict with ``d_mean``, ``d_std``, ``d_median``, ``d_iqr``,
    ``d_min``, ``d_max``, ``n_valid``, ``fail_rate``, and per-sample
    ``d_samples`` list.
    """
    rng = np.random.default_rng(seed)
    n_traj, T, N = trajectories.shape
    k = min(k_traj, n_traj)
    indices = rng.choice(n_traj, size=k, replace=False)

    burnin = max(0, int(T * burnin_fraction))
    d_samples: List[float] = []
    failures: List[str] = []

    for i in indices:
        seg = trajectories[i, burnin:, :]
        try:
            out = twonn_dimension(seg, max_points=max_points, seed=int(rng.integers(0, 2**31)))
            d = out["d_twonn"]
            if np.isfinite(d):
                d_samples.append(d)
            else:
                failures.append(out.get("error", "nan"))
        except Exception as exc:
            failures.append(str(exc)[:80])

    n_valid = len(d_samples)
    n_total = k
    fail_rate = (n_total - n_valid) / max(n_total, 1)

    if n_valid == 0:
        return {
            "d_mean": float("nan"),
            "d_std": float("nan"),
            "d_median": float("nan"),
            "d_iqr": float("nan"),
            "d_min": float("nan"),
            "d_max": float("nan"),
            "n_valid": 0,
            "n_total": n_total,
            "fail_rate": 1.0,
            "d_samples": [],
        }

    arr = np.array(d_samples)
    return {
        "d_mean": round(float(arr.mean()), 3),
        "d_std": round(float(arr.std()), 3),
        "d_median": round(float(np.median(arr)), 3),
        "d_iqr": round(float(np.percentile(arr, 75) - np.percentile(arr, 25)), 3),
        "d_min": round(float(arr.min()), 3),
        "d_max": round(float(arr.max()), 3),
        "n_valid": n_valid,
        "n_total": n_total,
        "fail_rate": round(fail_rate, 3),
        "d_samples": d_samples,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Pipeline entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_intrinsic_dimension(
    trajectories: np.ndarray,
    k_traj: int = 10,
    burnin_fraction: float = 0.10,
    max_points: int = 2_000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Estimate local intrinsic dimension of brain dynamics using TwoNN.

    This function is designed for integration with the unified
    ``dynamics_pipeline``.  It samples *k_traj* trajectories from the
    provided ensemble and returns distribution statistics.

    Parameters
    ----------
    trajectories : ndarray, shape (n_traj, T, N)
        Free-dynamics trajectory ensemble from Phase 1.
    k_traj : int
        Number of trajectories to subsample.
    burnin_fraction : float
        Fraction of each trajectory to discard as burn-in.
    max_points : int
        Maximum points per trajectory segment for kNN search.
    seed : int
        Random seed.
    output_dir : Path, optional
        If provided, a JSON summary and PNG figure are saved here.

    Returns
    -------
    dict with keys:
        ``d_mean``, ``d_std``, ``d_median``, ``d_iqr``,
        ``d_min``, ``d_max``, ``n_valid``, ``n_total``,
        ``fail_rate``, ``d_samples``, ``method``.
    """
    logger.info(
        "  TwoNN intrinsic dimension: n_traj=%d, k_traj=%d, T=%d, N=%d",
        trajectories.shape[0], k_traj, trajectories.shape[1], trajectories.shape[2],
    )

    result = _twonn_ensemble(
        trajectories,
        k_traj=k_traj,
        burnin_fraction=burnin_fraction,
        max_points=max_points,
        seed=seed,
    )
    result["method"] = "twonn"

    if result["n_valid"] > 0:
        logger.info(
            "  TwoNN: d=%.2f±%.2f (n=%d/%d, fail=%.0f%%)",
            result["d_mean"], result["d_std"],
            result["n_valid"], result["n_total"],
            result["fail_rate"] * 100,
        )
    else:
        logger.warning("  TwoNN: all samples failed.")

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        import json
        json_path = out / "intrinsic_dimension_twonn.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("  TwoNN: saved %s", json_path)

        _try_plot(result, out / "intrinsic_dimension_twonn.png")

    return result


def _try_plot(result: Dict[str, Any], output_path: Path) -> None:
    """Histogram of TwoNN d estimates across trajectory samples."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # Try to configure CJK-safe font (harmless no-op if module unavailable)
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except ImportError:
            pass
    except ImportError:
        return

    d_samples = result.get("d_samples", [])
    if not d_samples:
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(d_samples, bins=max(5, len(d_samples) // 3),
            color="steelblue", alpha=0.75, edgecolor="k", lw=0.5)
    d_mean = result.get("d_mean", float("nan"))
    d_std = result.get("d_std", float("nan"))
    if np.isfinite(d_mean):
        ax.axvline(d_mean, ls="--", color="red", lw=1.5, label=f"mean={d_mean:.2f}")
    ax.set_xlabel("Intrinsic dimension d (TwoNN)")
    ax.set_ylabel("Count")
    n_valid = result.get("n_valid", 0)
    n_total = result.get("n_total", 0)
    fail_rate = result.get("fail_rate", 0.0)
    ax.set_title(
        f"TwoNN Local Intrinsic Dimension\n"
        f"d={d_mean:.2f}±{d_std:.2f}  "
        f"(n={n_valid}/{n_total}, fail={fail_rate:.0%})"
    )
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.debug("TwoNN plot saved: %s", output_path)
