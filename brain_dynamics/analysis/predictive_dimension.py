"""
predictive_dimension.py
=======================
TASK 7: Optimal Predictive Dimension (Information-Theoretic Validation)

Tests the hypothesis that the intrinsic attractor dimension D₂ ≈ optimal
predictive dimension (the fewest PCA dimensions needed for accurate one-step
prediction).

Method
------
For m = 1 … max_dim:
  1. Project all trajectories to the first m PCA components.
  2. Train a VAR(1) predictor on the first ``train_fraction`` of each
     trajectory.
  3. Evaluate MSE on the held-out portion.
  4. Compute AIC and BIC using the VAR(1) log-likelihood.

Find the optimal m = argmin AIC (or BIC).

The AIC elbow ideally coincides with D₂ from the Grassberger–Procaccia
correlation dimension, validating that the attractor has that intrinsic
dimension.

Outputs
-------
  predictive_dimension.csv   — m, MSE, AIC, BIC columns
  aic_curve.png              — AIC / BIC vs m plot
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# VAR(1) helpers
# ---------------------------------------------------------------------------

def _fit_var1(X: np.ndarray) -> np.ndarray:
    """Fit a VAR(1) model X(t+1) = A @ X(t) + c via OLS.

    Parameters
    ----------
    X: shape (T, m) — m-dimensional PCA projections over T time steps.

    Returns
    -------
    A: shape (m, m), c: shape (m,)
    """
    T, m = X.shape
    if T < m + 2:
        return np.zeros((m, m)), np.zeros(m)

    Y = X[1:]          # (T-1, m)  targets
    Z = np.hstack([X[:-1], np.ones((T - 1, 1))])  # (T-1, m+1) design

    try:
        coef, _, _, _ = np.linalg.lstsq(Z, Y, rcond=None)
        A = coef[:m].T       # (m, m)
        c = coef[m]          # (m,)
        return A, c
    except np.linalg.LinAlgError:
        return np.zeros((m, m)), np.zeros(m)


def _var1_mse_loglik(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple:
    """Return (MSE_test, log_likelihood_train) for VAR(1) fit on X_train."""
    A, c = _fit_var1(X_train)

    # Test MSE
    Y_pred = X_test[:-1] @ A.T + c
    Y_true = X_test[1:]
    mse = float(np.mean((Y_pred - Y_true) ** 2))

    # Training log-likelihood (Gaussian assumption)
    T_tr = len(X_train) - 1
    m = X_train.shape[1]
    Y_hat_tr = X_train[:-1] @ A.T + c
    resid_tr = X_train[1:] - Y_hat_tr
    sigma2 = float(np.mean(resid_tr ** 2)) + 1e-12
    log_lik = -0.5 * T_tr * m * (np.log(2 * np.pi * sigma2) + 1.0)

    return mse, log_lik, T_tr, m


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_predictive_dimension(
    trajectories: np.ndarray,
    max_dim: int = 10,
    train_fraction: float = 0.70,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Find the optimal predictive dimension via VAR(1) MSE, AIC, BIC.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)``.
    max_dim:
        Maximum PCA dimension to evaluate.
    train_fraction:
        Fraction of each trajectory used for training.
    seed:
        Random seed (unused currently, reserved for future subsampling).
    output_dir:
        Directory to save CSV and PNG.

    Returns
    -------
    dict with keys 'results' (list of per-m dicts), 'optimal_m_aic',
    'optimal_m_bic', 'optimal_m_mse'.
    """
    trajs = np.asarray(trajectories, dtype=np.float64)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]

    n_traj, T, N = trajs.shape
    max_dim = min(max_dim, N, T - 2)
    logger.info("Predictive dimension: n_traj=%d, T=%d, N=%d, max_dim=%d",
                n_traj, T, N, max_dim)

    # Compute joint PCA basis on all training data
    t_split = int(T * train_fraction)
    X_all_train = trajs[:, :t_split, :].reshape(-1, N)
    X_all_train -= X_all_train.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(X_all_train, full_matrices=False)
        pca_components = Vt[:max_dim]  # (max_dim, N)
    except np.linalg.LinAlgError:
        logger.warning("SVD failed; using identity PCA components.")
        pca_components = np.eye(min(max_dim, N))

    mu = trajs[:, :t_split, :].reshape(-1, N).mean(axis=0)

    rows = []
    for m in range(1, max_dim + 1):
        comps = pca_components[:m]  # (m, N)

        mse_list, loglik_list, n_obs_list, n_params_list = [], [], [], []
        for traj in trajs:
            Xc = traj - mu                     # (T, N)
            P = Xc @ comps.T                   # (T, m)
            Ptr = P[:t_split]
            Pte = P[t_split:]
            if len(Ptr) < m + 2 or len(Pte) < 2:
                continue
            mse, log_lik, T_tr, m_ = _var1_mse_loglik(Ptr, Pte)
            n_params = m * m + m  # A (m×m) + c (m)
            mse_list.append(mse)
            loglik_list.append(log_lik)
            n_obs_list.append(T_tr * m_)
            n_params_list.append(n_params)

        if not mse_list:
            continue

        mean_mse = float(np.mean(mse_list))
        mean_loglik = float(np.mean(loglik_list))
        mean_n = float(np.mean(n_obs_list))
        k_params = float(np.mean(n_params_list))

        aic = -2.0 * mean_loglik + 2.0 * k_params
        bic = -2.0 * mean_loglik + k_params * np.log(max(mean_n, 1))

        rows.append({
            "m": m,
            "mse": mean_mse,
            "aic": float(aic),
            "bic": float(bic),
            "log_lik": mean_loglik,
        })
        logger.info("  m=%2d: MSE=%.6f  AIC=%.2f  BIC=%.2f", m, mean_mse, aic, bic)

    if not rows:
        return {"results": [], "optimal_m_aic": -1, "optimal_m_bic": -1, "optimal_m_mse": -1}

    # Find optima
    aic_vals = np.array([r["aic"] for r in rows])
    bic_vals = np.array([r["bic"] for r in rows])
    mse_vals = np.array([r["mse"] for r in rows])
    m_vals = np.array([r["m"] for r in rows])

    opt_m_aic = int(m_vals[np.argmin(aic_vals)])
    opt_m_bic = int(m_vals[np.argmin(bic_vals)])
    opt_m_mse = int(m_vals[np.argmin(mse_vals)])

    logger.info("Optimal m: AIC=%d  BIC=%d  MSE=%d", opt_m_aic, opt_m_bic, opt_m_mse)

    result = {
        "results": rows,
        "optimal_m_aic": opt_m_aic,
        "optimal_m_bic": opt_m_bic,
        "optimal_m_mse": opt_m_mse,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        _save_csv(rows, output_dir / "predictive_dimension.csv")
        _save_plot(rows, opt_m_aic, opt_m_bic, output_dir / "aic_curve.png")

    return result


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _save_csv(rows, path: Path) -> None:
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["m", "mse", "aic", "bic", "log_lik"],
                               extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _save_plot(rows, opt_aic: int, opt_bic: int, path: Path) -> None:
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

    m_vals = [r["m"] for r in rows]
    aic_vals = [r["aic"] for r in rows]
    bic_vals = [r["bic"] for r in rows]
    mse_vals = [r["mse"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Optimal Predictive Dimension (VAR(1))", fontsize=12)

    axes[0].plot(m_vals, aic_vals, "b-o", label="AIC", markersize=5)
    axes[0].plot(m_vals, bic_vals, "r-s", label="BIC", markersize=5)
    axes[0].axvline(opt_aic, color="blue", linestyle="--", alpha=0.7,
                    label=f"AIC opt m={opt_aic}")
    axes[0].axvline(opt_bic, color="red", linestyle=":", alpha=0.7,
                    label=f"BIC opt m={opt_bic}")
    axes[0].set_xlabel("PCA dimension m")
    axes[0].set_ylabel("Information criterion")
    axes[0].set_title("AIC / BIC vs PCA dimension")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(m_vals, mse_vals, "g-^", markersize=5)
    axes[1].set_xlabel("PCA dimension m")
    axes[1].set_ylabel("Test MSE")
    axes[1].set_title("Prediction error vs PCA dimension")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("  Saved %s", path)
