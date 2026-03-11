"""
Attractor Topology Analysis
============================

Rigorous, multi-evidence identification of attractor type from free-dynamics
trajectory data.  Unlike ``line_attractor.py`` (which *assumes* a 2-D slow
manifold and only tests specific properties of it), this module tests all five
primary attractor hypotheses without prior assumption.

Hypotheses tested
-----------------
  FP — Fixed point:          trajectory converges to a single steady state.
  LC — Limit cycle:          periodic 1-D orbit (one dominant frequency).
  QP — Quasi-periodic/torus: ≥2 incommensurate frequencies.
  CA — Continuous attractor: manifold of marginal states (near-zero Jacobian
                             eigenvalue, neutral direction).
  SA — Strange/chaotic attr: sensitive dependence, positive Rosenstein LLE,
                             broadband spectrum.

Five convergent evidence streams
---------------------------------
  E1  Frequency decomposition   — FFT of PC1-PC3 time series.
  E2  Phase velocity field      — Mean velocity quiver in PC1-PC2 space;
                                  Jacobian eigenvalue analysis.
  E3  Recurrence Quantification — RR, DET, LAM from the recurrence matrix.
  E4  Local dimensionality      — neighbourhood PCA at multiple locations.
  E5  Hypothesis scorecard      — weighted evidence → ranked hypotheses.

Inputs
------
Can be called directly with trajectory arrays, or reuse pipeline-generated
files (``trajectories.npy``, ``pca_projections_trajectories.npy``).

Outputs
-------
  attractor_topology_report.json   — full metrics + ranked hypotheses
  freq_spectrum_pcs.png            — FFT of PC1-PC3 with peak annotations
  velocity_field_pca.png           — velocity quiver + density background
  recurrence_plot.png              — recurrence matrix + RQA overlay
  hypothesis_scorecard.png         — ranked bar-chart of hypothesis scores

CLI
---
  python -m analysis.attractor_topology \\
      --trajectories  outputs/dynamics/trajectories.npy \\
      --output        outputs/dynamics/attractor_topology \\
      [--burnin 50]  [--dt 1.0]
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

# ── module-level path fix (standalone usage) ────────────────────────────────
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── constants ────────────────────────────────────────────────────────────────
_N_PCS: int = 6                    # number of PCs to use internally
_SPECTRAL_PEAK_SNR: float = 5.0    # peak must be > SNR * median(power) to count
_GRID_N: int = 20                  # velocity field grid resolution
_RQA_EPS_PERCENTILE: float = 10.0  # recurrence threshold as % of distance range
_RQA_LMIN: int = 2                 # minimum diagonal / vertical line length
_LOCAL_DIM_K: int = 30             # neighbourhood size for local PCA


# ─────────────────────────────────────────────────────────────────────────────
# 0. Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_pca(
    trajectories: np.ndarray,
    burnin: int,
    n_pcs: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z, evr).

    Z   — shape (n_traj, T_eff, n_pcs): PCA projections.
    evr — shape (n_pcs,): explained variance ratio per component.
    """
    from sklearn.decomposition import PCA

    n_traj, T, N = trajectories.shape
    burnin = min(burnin, T - 1)
    T_eff = T - burnin
    X = trajectories[:, burnin:, :].reshape(-1, N)
    nc = min(n_pcs, min(X.shape))
    pca = PCA(n_components=nc, random_state=42)
    Z_flat = pca.fit_transform(X)                      # (n_traj*T_eff, nc)
    Z = Z_flat.reshape(n_traj, T_eff, nc)
    return Z, pca.explained_variance_ratio_


# ─────────────────────────────────────────────────────────────────────────────
# E1: Frequency decomposition
# ─────────────────────────────────────────────────────────────────────────────

def _spectral_analysis(Z: np.ndarray, dt: float = 1.0) -> Dict:
    """FFT of top-3 PC time series.

    Uses Hann window + linear detrend to minimise spectral leakage.  Peaks
    are detected against a SNR threshold.  The frequency classification (FP /
    LC / QP / SA) is based solely on the number and commensurability of peaks.

    Returns a serialisable dict with per-PC metrics and an overall
    ``freq_classification`` key.
    """
    n_traj, T, n_pcs = Z.shape

    try:
        from scipy.signal import detrend as _detrend, find_peaks
        _has_scipy = True
    except ImportError:
        _has_scipy = False

    # Average over trajectories to reduce noise
    mean_Z = Z.mean(axis=0)   # (T, n_pcs)

    # Hann window
    window = np.hanning(T)

    per_pc: List[Dict] = []
    all_peak_freqs: List[float] = []

    for pc_i in range(min(3, n_pcs)):
        ts = mean_Z[:, pc_i].astype(np.float64)
        if _has_scipy:
            ts = _detrend(ts, type="linear")
        else:
            # Fallback: manual linear detrend
            tt = np.arange(T, dtype=np.float64)
            slope = np.polyfit(tt, ts, 1)
            ts = ts - np.polyval(slope, tt)

        ts_w = ts * window
        fft_vals = np.fft.rfft(ts_w)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(T, d=dt)

        # Noise floor: median of power (excluding DC)
        noise_floor = float(np.median(power[1:]))
        peak_threshold = _SPECTRAL_PEAK_SNR * (noise_floor + 1e-30)

        if _has_scipy:
            raw_peaks, _ = find_peaks(power[1:], height=peak_threshold,
                                      prominence=noise_floor * 0.5)
            peak_idx = raw_peaks + 1  # restore DC offset
        else:
            # Simple threshold-based peak finding
            above = np.where(power[1:] > peak_threshold)[0] + 1
            # Keep only local maxima among those above threshold
            peak_idx_list: List[int] = []
            for k in above:
                left = power[k - 1] if k > 0 else 0.0
                right = power[k + 1] if k < len(power) - 1 else 0.0
                if power[k] >= left and power[k] >= right:
                    peak_idx_list.append(k)
            peak_idx = np.array(peak_idx_list, dtype=int)

        # Sort peaks by power (descending)
        if len(peak_idx):
            order = np.argsort(power[peak_idx])[::-1]
            peak_idx = peak_idx[order]

        pc_peaks = freqs[peak_idx[:5]].tolist()
        pc_powers = power[peak_idx[:5]].tolist()

        per_pc.append({
            "n_peaks": int(len(peak_idx)),
            "dominant_freq": float(pc_peaks[0]) if pc_peaks else 0.0,
            "peak_freqs": pc_peaks,
            "peak_powers": pc_powers,
            "noise_floor": round(float(noise_floor), 6),
            "power": power.tolist(),
            "freqs": freqs.tolist(),
        })
        all_peak_freqs.extend(pc_peaks[:3])

    # ── Aggregate and classify ──────────────────────────────────────────────
    n_total_peaks = sum(p["n_peaks"] for p in per_pc)

    # Unique non-zero peaks (rounded to 4 decimals)
    unique_freqs = sorted({round(f, 4) for f in all_peak_freqs if f > 0})

    def _commensurate(f1: float, f2: float, tol: float = 0.06) -> bool:
        """Return True if f2/f1 is close to a ratio n/m with n,m ≤ 8.

        Uses a precomputed sorted lookup of all n/m ratios so that we only
        need one binary-search pass per (f1, f2) pair instead of a nested
        loop, keeping the cost well below O(n²×64) for many peaks.
        """
        if f1 < 1e-10:
            return False
        ratio = f2 / f1
        # Precomputed sorted lookup of all n/m for 1 ≤ n,m ≤ 8
        # (evaluated once per call; small list, negligible cost)
        _ratios = sorted({n / m for n in range(1, 9) for m in range(1, 9)})
        import bisect
        pos = bisect.bisect_left(_ratios, ratio)
        for idx in (pos - 1, pos):
            if 0 <= idx < len(_ratios) and abs(ratio - _ratios[idx]) < tol:
                return True
        return False

    if n_total_peaks == 0:
        freq_class = "fixed_point_or_noise"
    elif len(unique_freqs) == 1:
        freq_class = "limit_cycle"
    elif len(unique_freqs) >= 2:
        # Check if all pairs are commensurate (harmonics of one base frequency)
        base = unique_freqs[0]
        all_comm = all(_commensurate(base, f) for f in unique_freqs[1:3])
        if all_comm:
            freq_class = "limit_cycle_with_harmonics"
        else:
            freq_class = "quasi_periodic"
    else:
        freq_class = "limit_cycle"

    return {
        "per_pc": per_pc,
        "n_total_peaks": n_total_peaks,
        "unique_peak_freqs": unique_freqs[:8],
        "freq_classification": freq_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# E2: Phase velocity field
# ─────────────────────────────────────────────────────────────────────────────

def _velocity_field_analysis(Z: np.ndarray, n_grid: int = _GRID_N) -> Dict:
    """Compute mean velocity field on a regular PC1-PC2 grid.

    Also fits a global linear model (Jacobian approximation) to detect:
      - neutral directions (|Re(λ)| ≈ 0) → continuous attractor
      - rotational structure (Im(λ) ≠ 0) → oscillatory / limit-cycle
      - overall contraction (Re(λ) < 0) → stable

    Returns a serialisable dict plus internal arrays for plotting.
    """
    n_traj, T, n_pcs = Z.shape

    pos = Z[:, :-1, :2].reshape(-1, 2).astype(np.float64)  # (n_traj*(T-1), 2)
    vel = (Z[:, 1:, :2] - Z[:, :-1, :2]).reshape(-1, 2).astype(np.float64)

    # Grid extents
    x_min, x_max = pos[:, 0].min(), pos[:, 0].max()
    y_min, y_max = pos[:, 1].min(), pos[:, 1].max()
    pad_x = 0.05 * (x_max - x_min + 1e-9)
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    x_edges = np.linspace(x_min - pad_x, x_max + pad_x, n_grid + 1)
    y_edges = np.linspace(y_min - pad_y, y_max + pad_y, n_grid + 1)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    xi = np.clip(np.digitize(pos[:, 0], x_edges) - 1, 0, n_grid - 1)
    yi = np.clip(np.digitize(pos[:, 1], y_edges) - 1, 0, n_grid - 1)

    u_sum = np.zeros((n_grid, n_grid))
    v_sum = np.zeros((n_grid, n_grid))
    cnt = np.zeros((n_grid, n_grid), dtype=int)
    np.add.at(u_sum, (xi, yi), vel[:, 0])
    np.add.at(v_sum, (xi, yi), vel[:, 1])
    np.add.at(cnt, (xi, yi), 1)

    occ = cnt > 0
    u_grid = np.where(occ, u_sum / np.where(occ, cnt, 1), np.nan)
    v_grid = np.where(occ, v_sum / np.where(occ, cnt, 1), np.nan)

    # ── Global linear fit (Jacobian approximation) ─────────────────────────
    eigenvalues_info: List[Dict] = []
    min_abs_re = float("nan")
    rotation_score = float("nan")

    try:
        from sklearn.linear_model import Ridge

        A_model = Ridge(alpha=1e-3).fit(pos, vel)
        A = A_model.coef_          # shape (2, 2)
        evals = np.linalg.eigvals(A)
        evals_sorted = sorted(evals, key=lambda e: e.real)
        eigenvalues_info = [
            {"real": round(float(e.real), 6), "imag": round(float(e.imag), 6)}
            for e in evals_sorted
        ]
        min_abs_re = float(min(abs(e.real) for e in evals_sorted))
        # Rotation score: how dominant is the imaginary part?
        rotation_score = float(
            max(abs(e.imag) for e in evals_sorted) /
            (abs(evals_sorted[-1].real) + abs(evals_sorted[-1].imag) + 1e-10)
        )
    except Exception as exc:
        logger.debug("Jacobian fit failed: %s", exc)

    # ── Curl / divergence from the gridded field ───────────────────────────
    u_filled = np.where(occ, u_grid, 0.0)
    v_filled = np.where(occ, v_grid, 0.0)
    du_dx = np.gradient(u_filled, axis=0)
    dv_dy = np.gradient(v_filled, axis=1)
    dv_dx = np.gradient(v_filled, axis=0)
    du_dy = np.gradient(u_filled, axis=1)
    curl = dv_dx - du_dy
    div  = du_dx + dv_dy
    mean_curl = float(np.abs(curl[occ]).mean()) if occ.any() else 0.0
    mean_div  = float(np.abs(div[occ]).mean())  if occ.any() else 0.0

    result: Dict = {
        "mean_curl": round(mean_curl, 8),
        "mean_divergence": round(mean_div, 8),
        "rotation_index": round(
            mean_curl / (mean_curl + mean_div + 1e-15), 4),
        "jacobian_eigenvalues": eigenvalues_info,
        "min_eigenvalue_real": round(min_abs_re, 6) if np.isfinite(min_abs_re) else None,
        "has_neutral_direction": (min_abs_re < 0.02) if np.isfinite(min_abs_re) else False,
        "rotation_score": round(rotation_score, 4) if np.isfinite(rotation_score) else None,
        # Internal arrays (not serialised to JSON, used for plotting)
        "_x_centers": x_centers,
        "_y_centers": y_centers,
        "_u_grid": u_grid,
        "_v_grid": v_grid,
        "_occ": occ,
        "_pos": pos,
        "_vel": vel,
    }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# E3: Recurrence Quantification Analysis
# ─────────────────────────────────────────────────────────────────────────────

def _recurrence_analysis(
    Z: np.ndarray,
    eps_percentile: float = _RQA_EPS_PERCENTILE,
    lmin: int = _RQA_LMIN,
    max_T: int = 500,
) -> Dict:
    """Recurrence Quantification Analysis (RQA).

    Computes the recurrence matrix R[i,j] = 1 iff ||z_i - z_j|| < ε, where
    ε is set at the ``eps_percentile``-th percentile of pairwise distances.

    The ``max_T`` parameter limits analysis to the first ``max_T`` steps to
    keep memory and runtime manageable: computing the full distance matrix
    requires O(T²) space and time, which becomes expensive for long
    trajectories (e.g., T=1000 needs ~24 MB for float64).  The default of
    500 steps captures enough temporal structure for typical brain dynamics
    data (fMRI TR ≈ 2 s, 500 × 2 s = 1000 s of observation) while staying
    below ~6 MB per analysis.

    Key RQA measures:
      RR  — recurrence rate (fraction of recurrent points): general activity.
      DET — determinism (fraction of recurrence points on diagonal lines ≥
            lmin): high ↔ periodic / quasi-periodic.
      LAM — laminarity (fraction on vertical lines ≥ lmin): high ↔ laminar
            intermittency.
      TT  — trapping time (mean vertical line length): higher ↔ the system
            stays near a state for longer.

    Interpretation guide
    --------------------
      DET > 0.90 → strongly periodic (LC or QP)
      DET 0.60-0.90 → weakly periodic (QP or mild chaos)
      DET < 0.40 → chaotic or noise-driven (SA)
      LAM >> DET → intermittency / laminar phases (not purely periodic)
      LAM ≈ DET → uniform dynamics
    """
    n_traj, T_full, n_pcs = Z.shape

    # Use first trajectory, truncated to max_T
    T = min(T_full, max_T)
    z = Z[0, :T, :min(n_pcs, 3)].astype(np.float64)  # (T, 3)

    # Pairwise distances
    diffs = z[:, None, :] - z[None, :, :]             # (T, T, 3)
    dist = np.sqrt((diffs ** 2).sum(axis=-1))          # (T, T)

    # Recurrence threshold
    flat_dist = dist[np.triu_indices(T, k=1)]
    eps = float(np.percentile(flat_dist, eps_percentile))

    R = (dist < eps).astype(np.uint8)

    # ── RR ──────────────────────────────────────────────────────────────────
    n_recur = int(R.sum()) - T           # exclude identity diagonal
    rr = n_recur / max(T * (T - 1), 1)  # normalise to off-diagonal

    # ── DET: diagonal lines ─────────────────────────────────────────────────
    det_points = 0
    total_diag_points = n_recur  # all off-diagonal recurrence points
    for diag in range(-(T - lmin), T - lmin + 1):
        if diag == 0:
            continue
        d = np.diagonal(R, offset=diag)
        # Count points in diagonal lines of length ≥ lmin
        run = 0
        for bit in d:
            if bit:
                run += 1
            else:
                if run >= lmin:
                    det_points += run
                run = 0
        if run >= lmin:
            det_points += run

    det = det_points / max(total_diag_points, 1)

    # ── LAM: vertical lines ─────────────────────────────────────────────────
    lam_points = 0
    for col in range(T):
        run = 0
        for row in range(T):
            if row == col:
                run = 0
                continue
            if R[row, col]:
                run += 1
            else:
                if run >= lmin:
                    lam_points += run
                run = 0
        if run >= lmin:
            lam_points += run

    lam = lam_points / max(total_diag_points, 1)

    # ── TT: mean vertical line length ───────────────────────────────────────
    vline_lengths: List[int] = []
    for col in range(T):
        run = 0
        for row in range(T):
            if row == col:
                if run >= lmin:
                    vline_lengths.append(run)
                run = 0
                continue
            if R[row, col]:
                run += 1
            else:
                if run >= lmin:
                    vline_lengths.append(run)
                run = 0
        if run >= lmin:
            vline_lengths.append(run)

    tt = float(np.mean(vline_lengths)) if vline_lengths else 0.0

    # ── Interpret DET + LAM ─────────────────────────────────────────────────
    if det > 0.90:
        rqa_class = "periodic"
    elif det > 0.60:
        rqa_class = "weakly_periodic"
    elif det > 0.40:
        rqa_class = "mixed"
    else:
        rqa_class = "chaotic_or_noise"

    return {
        "epsilon": round(eps, 6),
        "T_used": int(T),
        "RR": round(float(rr), 4),
        "DET": round(float(det), 4),
        "LAM": round(float(lam), 4),
        "TT": round(float(tt), 4),
        "rqa_classification": rqa_class,
        "_R": R,   # internal — used for plotting
    }


# ─────────────────────────────────────────────────────────────────────────────
# E4: Local dimensionality
# ─────────────────────────────────────────────────────────────────────────────

def _local_dimensionality(
    Z: np.ndarray,
    k: int = _LOCAL_DIM_K,
    n_samples: int = 200,
    var_thresh: float = 0.90,
    seed: int = 42,
) -> Dict:
    """Estimate intrinsic dimensionality by neighbourhood PCA.

    At ``n_samples`` randomly drawn points in the trajectory, find the
    ``k``-nearest neighbours and fit PCA.  The *local* effective dimension is
    the number of PCs needed to explain ≥ ``var_thresh`` of variance in the
    neighbourhood.

    The mean and distribution of local dimensions provide strong evidence:
      mean ≈ 1 → line attractor / 1-D manifold
      mean ≈ 2 → limit cycle / 2-D surface
      mean ≈ 2-3 → torus / 3-D surface
      large spread → heterogeneous / chaotic
    """
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    n_traj, T, n_pcs = Z.shape
    rng = np.random.default_rng(seed)

    # Flatten all trajectory points: (n_traj*T, n_pcs)
    Z_flat = Z.reshape(-1, n_pcs).astype(np.float64)
    N_pts = Z_flat.shape[0]

    if N_pts < k + 1:
        return {"local_dim_mean": float("nan"), "local_dim_std": float("nan"),
                "error": "too_few_points"}

    # Random sample of query points
    n_samples = min(n_samples, N_pts)
    query_idx = rng.choice(N_pts, size=n_samples, replace=False)

    # Fit k-NN on the full cloud
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto",
                            metric="euclidean").fit(Z_flat)
    _, nn_idx = nbrs.kneighbors(Z_flat[query_idx])

    local_dims: List[int] = []
    for qi, nbr_ids in zip(query_idx, nn_idx):
        # Neighbourhood cloud (k+1 includes the query itself)
        cloud = Z_flat[nbr_ids]
        cloud -= cloud.mean(axis=0)
        if cloud.shape[0] < 2:
            continue
        nc = min(cloud.shape[0], cloud.shape[1])
        if nc < 1:
            continue
        pca_loc = PCA(n_components=nc).fit(cloud)
        cum_var = np.cumsum(pca_loc.explained_variance_ratio_)
        # Effective dimension: smallest d s.t. cumvar ≥ var_thresh
        dims_above = np.where(cum_var >= var_thresh)[0]
        ld = int(dims_above[0] + 1) if len(dims_above) > 0 else nc
        local_dims.append(ld)

    if not local_dims:
        return {"local_dim_mean": float("nan"), "local_dim_std": float("nan"),
                "error": "no_valid_neighbourhoods"}

    arr = np.array(local_dims, dtype=float)
    counts = {int(d): int((arr == d).sum()) for d in np.unique(arr)}

    return {
        "local_dim_mean": round(float(arr.mean()), 3),
        "local_dim_std": round(float(arr.std()), 3),
        "local_dim_median": int(np.median(arr)),
        "local_dim_mode": int(arr[np.bincount(arr.astype(int)).argmax()]),
        "local_dim_histogram": counts,
        "k_neighbours": k,
        "var_threshold": var_thresh,
        "n_samples": len(local_dims),
    }



# ─────────────────────────────────────────────────────────────────────────────
# E6: Period stability & phase drift analysis (Hilbert transform)
# ─────────────────────────────────────────────────────────────────────────────

def _period_stability_analysis(Z: np.ndarray, dt: float = 1.0) -> Dict:
    """Measure period stability and phase drift of the dominant oscillation.

    Uses the Hilbert transform of PC1 to extract instantaneous amplitude and
    phase.  Three metrics are reported:

    phase_drift_rate  (float)
        Linear phase drift per step estimated by OLS.  ≈ 0 → no secular drift
        (strict limit cycle); nonzero → slow-manifold oscillation or drift.

    period_cv  (float)
        Coefficient of variation of inter-crossing periods (std/mean).
        < 0.05 → very stable (limit cycle); > 0.20 → irregular / drifting.

    phase_variance  (float)
        Variance of the detrended (residual) instantaneous phase.  Small →
        regular orbit; large → stochastic / irregular phase noise.

    slow_envelope_variation  (float)
        Std / mean of the Hilbert envelope (|analytic signal|) over the
        trajectory.  < 0.10 → constant amplitude (limit cycle); > 0.30 →
        slow modulation (slow-manifold / inhomogeneous oscillation).

    Parameters
    ----------
    Z   : shape (n_traj, T_eff, n_pcs) — PCA projections
    dt  : time step (for frequency axis labelling, not used numerically here)
    """
    n_traj, T_eff, n_pcs = Z.shape
    if T_eff < 20:
        return {"error": "trajectory_too_short"}

    # Use PC1 concatenated across all trajectories (or a subset if many)
    n_use = min(n_traj, 10)
    pc1_all = Z[:n_use, :, 0].reshape(-1)  # (n_use * T_eff,)

    # --- Hilbert transform ---
    try:
        from scipy.signal import hilbert as scipy_hilbert
        analytic = scipy_hilbert(pc1_all)
    except ImportError:
        # Pure NumPy fallback: construct analytic signal via two-sided FFT.
        # H(ω) = 2 for ω>0, 1 for ω=0 and ω=Nyquist, 0 for ω<0.
        N = len(pc1_all)
        F = np.fft.fft(pc1_all)
        H = np.zeros(N, dtype=complex)
        if N % 2 == 0:
            H[0] = 1.0
            H[1: N // 2] = 2.0
            H[N // 2] = 1.0
        else:
            H[0] = 1.0
            H[1: (N + 1) // 2] = 2.0
        analytic = np.fft.ifft(F * H)

    envelope = np.abs(analytic)
    inst_phase = np.unwrap(np.angle(analytic))

    # --- Phase drift (linear trend in phase) ---
    t = np.arange(len(inst_phase), dtype=float)
    # OLS: phase ≈ ω*t + φ₀ + residual
    A = np.column_stack([t, np.ones_like(t)])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, inst_phase, rcond=None)
        omega_est = float(coeffs[0])    # rad/step (mean angular frequency)
        phase_residual = inst_phase - (omega_est * t + float(coeffs[1]))
    except Exception:
        omega_est = 0.0
        phase_residual = inst_phase.copy()

    # Drift rate as fraction of mean angular frequency (dimensionless)
    # Positive phase drift means the phase is accelerating (slower return = drifting)
    phase_variance = float(np.var(phase_residual))

    # --- Inter-crossing period variance (upward zero-crossings of PC1) ---
    pc1 = pc1_all
    cross_mask = (pc1[:-1] < 0) & (pc1[1:] >= 0)
    crossing_times = np.where(cross_mask)[0].astype(float)
    # Refine crossing time by linear interpolation
    for i, ct in enumerate(crossing_times):
        ct_i = int(ct)
        dpc = float(pc1[ct_i + 1]) - float(pc1[ct_i])
        if abs(dpc) > 1e-15:
            crossing_times[i] = ct_i - float(pc1[ct_i]) / dpc

    if len(crossing_times) >= 3:
        periods = np.diff(crossing_times)
        period_mean = float(periods.mean())
        period_std  = float(periods.std())
        period_cv   = period_std / (period_mean + 1e-15)
    else:
        period_mean = float("nan")
        period_std  = float("nan")
        period_cv   = float("nan")

    # --- Envelope variation ---
    env_mean = float(envelope.mean())
    env_std  = float(envelope.std())
    slow_envelope_variation = env_std / (env_mean + 1e-15)

    # --- Summary classification ---
    # Heuristic thresholds (dimensionless)
    if (not np.isnan(period_cv)) and period_cv < 0.05 and slow_envelope_variation < 0.15:
        stability_class = "stable_limit_cycle"
    elif (not np.isnan(period_cv)) and period_cv < 0.15 and slow_envelope_variation < 0.30:
        stability_class = "weakly_stable_oscillation"
    elif (not np.isnan(period_cv)) and period_cv < 0.35:
        stability_class = "slow_manifold_oscillation"
    elif np.isnan(period_cv):
        stability_class = "insufficient_crossings"
    else:
        stability_class = "irregular_oscillation"

    return {
        "n_crossings": len(crossing_times),
        "period_mean_steps": round(period_mean, 2) if not np.isnan(period_mean) else None,
        "period_cv": round(period_cv, 4) if not np.isnan(period_cv) else None,
        "phase_variance": round(phase_variance, 6),
        "slow_envelope_variation": round(slow_envelope_variation, 4),
        "stability_class": stability_class,
    }


# ─────────────────────────────────────────────────────────────────────────────
# E5: Hypothesis scoring
# ─────────────────────────────────────────────────────────────────────────────

def _score_hypotheses(
    spectral: Dict,
    velocity: Dict,
    rqa: Dict,
    local_dim: Dict,
    rosenstein_lle: Optional[float] = None,
    n_pca_90: Optional[int] = None,
    period_stability: Optional[Dict] = None,
    kmeans_uniform_suspect: bool = False,
    dmd_n_hopf_pairs: Optional[int] = None,
) -> Dict:
    """Combine evidence streams into a ranked scorecard.

    Each hypothesis is scored in [0, 1].  Scoring is additive (not
    multiplicative) so that absent evidence neither confirms nor denies.
    Scores are normalised to sum to 1 at the end so they can be read as
    rough posterior probabilities.

    Evidence channels used
    ----------------------
    spectral          → freq_classification (FP / LC / LC+harm / QP / broadband)
    velocity          → rotation_index, has_neutral_direction, rotation_score
    rqa               → DET, LAM
    local_dim         → local_dim_mean
    external          → rosenstein_lle, n_pca_90
    period_stability  → stability_class, period_cv, slow_envelope_variation
    kmeans_uniform_suspect → uniform KMeans (ring/continuous attractor flag)
    dmd_n_hopf_pairs  → number of Hopf pairs in DMD linearised spectrum
    """
    # ── raw scores (accumulators) ──────────────────────────────────────────
    scores: Dict[str, float] = {
        "FP": 0.0,   # fixed point
        "LC": 0.0,   # limit cycle
        "QP": 0.0,   # quasi-periodic / torus
        "CA": 0.0,   # continuous attractor
        "SA": 0.0,   # strange / chaotic attractor
        "SM": 0.0,   # slow manifold + Hopf oscillation
    }
    evidence_notes: Dict[str, List[str]] = {h: [] for h in scores}

    # ── E1: spectral ──────────────────────────────────────────────────────
    fc = spectral.get("freq_classification", "")
    n_peaks = spectral.get("n_total_peaks", 0)

    if fc == "fixed_point_or_noise":
        scores["FP"] += 2.0
        scores["SA"] += 0.5   # broadband = no distinct peaks too
        evidence_notes["FP"].append("no spectral peaks")
        evidence_notes["SA"].append("flat spectrum (possible broadband)")
    elif fc in ("limit_cycle", "limit_cycle_with_harmonics"):
        scores["LC"] += 2.5
        evidence_notes["LC"].append(f"1 dominant frequency ({fc})")
    elif fc == "quasi_periodic":
        scores["QP"] += 2.5
        evidence_notes["QP"].append("≥2 incommensurate frequencies")
    # Many peaks can also suggest broadband chaos
    if n_peaks >= 5:
        scores["SA"] += 1.0
        evidence_notes["SA"].append(f"many peaks (n={n_peaks}) → broadband")

    # ── E2: velocity field ────────────────────────────────────────────────
    ri = velocity.get("rotation_index", 0.5)
    has_neut = velocity.get("has_neutral_direction", False)
    rs = velocity.get("rotation_score")

    if ri > 0.60:
        scores["LC"] += 1.5
        scores["QP"] += 1.0
        evidence_notes["LC"].append(f"high rotation index ({ri:.2f})")
        evidence_notes["QP"].append(f"rotational flow ({ri:.2f})")
    elif ri < 0.25:
        scores["FP"] += 1.0
        scores["SA"] += 0.5
        evidence_notes["FP"].append(f"low rotation (divergent/convergent flow, ri={ri:.2f})")

    if has_neut:
        scores["CA"] += 2.5
        evidence_notes["CA"].append("near-zero Jacobian eigenvalue → neutral direction")
    # Absence of a neutral direction is *neutral* evidence (not negative) — the
    # global linear Jacobian may miss local neutrality, so we do not penalise CA.

    if rs is not None and rs > 1.5:
        scores["LC"] += 0.8
        scores["QP"] += 0.8
        evidence_notes["LC"].append(f"complex Jacobian eigenvalues (rotation_score={rs:.2f})")

    # ── E3: RQA ───────────────────────────────────────────────────────────
    det = rqa.get("DET", 0.5)
    lam = rqa.get("LAM", 0.5)

    if det > 0.90:
        scores["LC"] += 2.0
        scores["QP"] += 1.5
        evidence_notes["LC"].append(f"very high DET={det:.3f}")
        evidence_notes["QP"].append(f"high DET={det:.3f}")
    elif det > 0.60:
        scores["QP"] += 1.0
        scores["LC"] += 0.7
        evidence_notes["QP"].append(f"moderate DET={det:.3f}")
    elif det < 0.30:
        scores["SA"] += 2.0
        evidence_notes["SA"].append(f"very low DET={det:.3f} → irregular")
        scores["FP"] += 0.5

    if lam > det + 0.15:
        scores["CA"] += 0.8
        evidence_notes["CA"].append(f"LAM ({lam:.3f}) > DET ({det:.3f}) → laminar phases")

    # ── E4: local dimensionality ─────────────────────────────────────────
    ld = local_dim.get("local_dim_mean")
    if ld is not None and np.isfinite(ld):
        if ld < 1.3:
            scores["FP"] += 1.0
            scores["CA"] += 1.5
            evidence_notes["FP"].append(f"local dim ≈ 1 ({ld:.2f})")
            evidence_notes["CA"].append(f"local dim ≈ 1 ({ld:.2f}) → 1-D manifold")
        elif ld < 2.3:
            scores["LC"] += 1.5
            scores["QP"] += 1.0
            evidence_notes["LC"].append(f"local dim ≈ 2 ({ld:.2f}) → 2-D orbit")
            evidence_notes["QP"].append(f"local dim ≈ 2 ({ld:.2f})")
        elif ld < 3.5:
            scores["QP"] += 1.5
            scores["SA"] += 1.0
            evidence_notes["QP"].append(f"local dim ≈ 3 ({ld:.2f}) → possible torus")
            evidence_notes["SA"].append(f"local dim ≈ 3 ({ld:.2f}) → moderate dim")
        else:
            scores["SA"] += 2.0
            evidence_notes["SA"].append(f"high local dim ({ld:.2f}) → high-D / chaotic")

    # ── External: Rosenstein LLE ──────────────────────────────────────────
    if rosenstein_lle is not None:
        if rosenstein_lle > 0.01:
            scores["SA"] += 2.5
            evidence_notes["SA"].append(f"Rosenstein LLE > 0 ({rosenstein_lle:.5f})")
            # Positive LLE is also consistent with CA (marginal stability)
            scores["CA"] += 0.5
        elif rosenstein_lle < -0.01:
            scores["FP"] += 1.5
            scores["LC"] += 1.0
            evidence_notes["FP"].append(f"Rosenstein LLE < 0 ({rosenstein_lle:.5f})")
            evidence_notes["LC"].append(f"Rosenstein LLE < 0 ({rosenstein_lle:.5f})")
        else:
            # Near-zero: marginal stability
            scores["CA"] += 1.5
            scores["LC"] += 0.5
            evidence_notes["CA"].append(
                f"Rosenstein LLE ≈ 0 ({rosenstein_lle:.5f}) → marginal stability")

    # ── External: PCA n@90% ──────────────────────────────────────────────
    if n_pca_90 is not None:
        if n_pca_90 <= 2:
            scores["FP"] += 0.5
            scores["LC"] += 0.5
            scores["CA"] += 0.5
            evidence_notes["LC"].append(f"PCA n@90%={n_pca_90} (low-D)")
        elif n_pca_90 <= 5:
            scores["QP"] += 0.5
            evidence_notes["QP"].append(f"PCA n@90%={n_pca_90} (moderate-D)")

    # ── E6: Period stability (Hilbert transform) ─────────────────────────
    if period_stability and "error" not in period_stability:
        sc = period_stability.get("stability_class", "")
        pcv = period_stability.get("period_cv")
        sev = period_stability.get("slow_envelope_variation")

        if sc == "stable_limit_cycle":
            scores["LC"] += 3.0
            evidence_notes["LC"].append(
                f"stable limit cycle: period_cv={pcv:.3f}, "
                f"slow_envelope_variation={sev:.3f}"
            )
        elif sc == "weakly_stable_oscillation":
            scores["LC"] += 1.5
            scores["CA"] += 0.5
            evidence_notes["LC"].append(
                f"weakly stable oscillation: period_cv={pcv:.3f}, sev={sev:.3f}"
            )
        elif sc == "slow_manifold_oscillation":
            # Now that SM hypothesis exists, slow_manifold_oscillation is
            # direct evidence for SM — NOT primarily CA.  This block adds a
            # partial +1.5 credit (E6 period-stability evidence).  The main
            # SM-specific scoring block below adds a further +3.0 (from the
            # SM hypothesis section), giving a cumulative total of +4.5 for SM
            # when period_stability == "slow_manifold_oscillation".  This
            # higher total is intentional: two independent evidence streams
            # (E6 period-stability + SM hypothesis scoring) both confirm the
            # same classification, so the evidence should accumulate.
            # The final score is normalised by max_possible_score, so the
            # absolute magnitude of these numbers does not matter — only
            # their relative ordering across hypotheses.
            scores["SM"] += 1.5   # E6 period-stability partial credit
            scores["CA"] += 0.5   # reduced from 1.5; SM is the better fit
            scores["QP"] += 0.5   # reduced from 1.0
            evidence_notes["SM"].append(
                "slow-manifold oscillation (E6 preview): "
                f"period_cv={f'{pcv:.3f}' if pcv is not None else '?'}, "
                f"envelope variation={f'{sev:.3f}' if sev is not None else '?'}"
            )
            evidence_notes["CA"].append(
                "slow-manifold oscillation (CA component): neutral drift direction"
            )
            evidence_notes["QP"].append(
                f"two-timescale: period_cv={f'{pcv:.3f}' if pcv is not None else '?'}"
            )
        elif sc == "irregular_oscillation":
            scores["SA"] += 1.0
            scores["QP"] += 0.5
            evidence_notes["SA"].append(
                f"irregular oscillation: period_cv={pcv:.3f}"
            )
        # insufficient_crossings: no update (no crossings detected)

    # ── External: KMeans uniform-cluster signature ───────────────────────
    # When KMeans(K) gives perfectly equal cluster sizes + high silhouette,
    # K-means is likely cutting through a continuous (ring) attractor, not
    # detecting discrete basins.  This is evidence for CA (continuous attractor)
    # and weak evidence AGAINST SA (strange attractor requires folding).
    if kmeans_uniform_suspect:
        scores["CA"] += 2.0
        scores["SA"] -= 1.0   # uniform ring is inconsistent with strange attractor
        evidence_notes["CA"].append(
            "uniform KMeans cluster sizes + high silhouette → ring/continuous attractor"
        )

    # ── SM: Slow Manifold + Hopf Oscillation ─────────────────────────────
    # This hypothesis captures the "canonical near-critical brain pattern":
    #   modular network → low-rank coupling → ρ≈1 → mode decay → slow manifold
    #   + Hopf oscillations riding on the manifold.
    # Evidence accumulates from:
    #   1. Period stability = "slow_manifold_oscillation" (direct evidence: two
    #      timescales — slow drift + faster oscillation on the manifold surface)
    #   2. Slow envelope variation (amplitude modulation by the slow mode)
    #   3. Hopf pairs in DMD linearised spectrum
    #   4. Local dimensionality ≈ 1 (thin manifold)
    #   5. Neutral direction in velocity field (manifold's flat direction)
    #   6. LLE near zero or slightly positive (near-critical, not deeply stable)
    if period_stability and "error" not in period_stability:
        sc = period_stability.get("stability_class", "")
        pcv = period_stability.get("period_cv")
        sev = period_stability.get("slow_envelope_variation", 0.0)

        if sc == "slow_manifold_oscillation":
            # Strongest single evidence for SM hypothesis.
            scores["SM"] += 3.0
            pcv_str = f"{pcv:.3f}" if pcv is not None else "?"
            evidence_notes["SM"].append(
                f"slow-manifold oscillation confirmed by Hilbert: "
                f"period_cv={pcv_str}, "
                f"slow_envelope_variation={sev:.3f}"
            )
        elif sc in ("weakly_stable_oscillation",):
            # Partial evidence: oscillation present but less irregular drift
            scores["SM"] += 1.0
            evidence_notes["SM"].append(
                f"weakly stable oscillation ({sc}): possible slow manifold"
            )

        # Slow envelope variation: amplitude modulated by slow drift
        # (distinguishes SM from pure LC where sev < 0.15)
        if sev > 0.25:
            scores["SM"] += 1.0
            # Limit cycle is characterised by constant amplitude; slow amplitude
            # modulation (sev > 0.25) is inconsistent with that assumption.
            # Weight -0.5 (half the SM reward) so a single signal does not
            # dominate the comparison.
            scores["LC"] -= 0.5
            evidence_notes["SM"].append(
                f"slow amplitude modulation (sev={sev:.3f}>0.25) → slow drift on manifold"
            )
            if "LC" in evidence_notes:
                evidence_notes["LC"].append(
                    f"slow amplitude modulation (sev={sev:.3f}) reduces LC likelihood"
                )

    # DMD Hopf pairs: oscillatory modes present in linearised dynamics
    if dmd_n_hopf_pairs is not None and dmd_n_hopf_pairs > 0:
        scores["SM"] += 1.5
        scores["LC"] += 0.5   # Hopf is also consistent with LC
        evidence_notes["SM"].append(
            f"DMD has {dmd_n_hopf_pairs} Hopf pair(s) — linearised oscillatory modes"
        )

    # Neutral direction in velocity field: slow manifold's flat direction
    has_neut = velocity.get("has_neutral_direction", False)
    if has_neut:
        # Already scored CA, but also consistent with SM (which has a neutral
        # slow-drift direction in addition to oscillations)
        scores["SM"] += 1.0
        evidence_notes["SM"].append(
            "neutral velocity field direction → slow manifold flat direction"
        )

    # Local dimension ≈ 1: very thin manifold (canonical SM signature)
    ld = local_dim.get("local_dim_mean")
    if ld is not None and np.isfinite(ld) and ld < 1.5:
        scores["SM"] += 1.5
        evidence_notes["SM"].append(
            f"local dim ≈ 1 ({ld:.2f}) — very thin attractor, consistent with SM"
        )

    # LLE near zero or slightly positive: near-critical dynamics on the manifold
    # (not strongly stable = FP; not strongly chaotic = SA)
    if rosenstein_lle is not None:
        if -0.02 <= rosenstein_lle <= 0.05:
            scores["SM"] += 0.5
            evidence_notes["SM"].append(
                f"Rosenstein LLE ≈ 0 ({rosenstein_lle:.5f}) → near-critical SM"
            )

    # ── Normalise scores (clip to 0 first) ──────────────────────────────
    for h in scores:
        scores[h] = max(0.0, scores[h])
    total = sum(scores.values()) + 1e-15
    norm_scores = {h: round(v / total, 4) for h, v in scores.items()}

    # ── Rank hypotheses ──────────────────────────────────────────────────
    ranked = sorted(norm_scores.items(), key=lambda kv: -kv[1])
    top_hyp, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    # Confidence: ratio of top score to second
    confidence_ratio = top_score / (second_score + 1e-15)
    if confidence_ratio > 3.0 and top_score > 0.40:
        confidence = "high"
    elif confidence_ratio > 1.5 and top_score > 0.25:
        confidence = "moderate"
    else:
        confidence = "low"

    _FULL_NAMES = {
        "FP": "Fixed Point",
        "LC": "Limit Cycle",
        "QP": "Quasi-Periodic / Torus",
        "CA": "Continuous Attractor",
        "SA": "Strange / Chaotic Attractor",
        "SM": "Slow Manifold + Hopf Oscillation",
    }

    hypothesis_ranking = [
        {
            "code": h,
            "name": _FULL_NAMES[h],
            "score": norm_scores[h],
            "evidence": evidence_notes[h],
        }
        for h, _ in ranked
    ]

    return {
        "scores": norm_scores,
        "hypothesis_ranking": hypothesis_ranking,
        "top_hypothesis": top_hyp,
        "top_hypothesis_name": _FULL_NAMES[top_hyp],
        "top_score": top_score,
        "confidence": confidence,
        "confidence_ratio": round(float(confidence_ratio), 3),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_freq_spectrum(spectral: Dict, output_path: Path) -> None:
    """Plot FFT power spectra for PC1-PC3 with annotated peaks."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _safe_fallback_png(output_path)
        return

    per_pc = spectral.get("per_pc", [])
    n_pcs = len(per_pc)
    if n_pcs == 0:
        _safe_fallback_png(output_path)
        return

    fig, axes = plt.subplots(1, n_pcs, figsize=(5 * n_pcs, 4), sharey=False)
    if n_pcs == 1:
        axes = [axes]

    for i, (pc_data, ax) in enumerate(zip(per_pc, axes)):
        freqs = np.array(pc_data["freqs"])
        power = np.array(pc_data["power"])
        noise_floor = pc_data.get("noise_floor", 0.0)

        ax.semilogy(freqs[1:], power[1:], color="steelblue", lw=1.2, label="Power")
        ax.axhline(noise_floor * _SPECTRAL_PEAK_SNR, ls="--", color="red",
                   lw=1.0, alpha=0.8, label=f"Threshold ({_SPECTRAL_PEAK_SNR}x noise)")

        peak_freqs = pc_data.get("peak_freqs", [])
        peak_powers = pc_data.get("peak_powers", [])
        for pf, pp in zip(peak_freqs, peak_powers):
            ax.axvline(pf, ls=":", color="orange", lw=1.2, alpha=0.9)
            ax.annotate(f"{pf:.4f}", xy=(pf, pp), xytext=(4, 4),
                        textcoords="offset points", fontsize=7, color="darkorange")

        ax.set_xlabel("Frequency (1/step)")
        ax.set_ylabel("Power")
        ax.set_title(f"PC{i+1} spectrum  (n_peaks={len(peak_freqs)})")
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.2)

    fc = spectral.get("freq_classification", "?")
    fig.suptitle(
        f"Frequency Decomposition: PC1-PC{n_pcs}\n"
        f"Classification: {fc}  |  total peaks: {spectral.get('n_total_peaks', 0)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_velocity_field(velocity: Dict, Z: np.ndarray, output_path: Path) -> None:
    """Density background + velocity quiver in PC1-PC2 space."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _safe_fallback_png(output_path)
        return

    x_centers = velocity.get("_x_centers")
    y_centers = velocity.get("_y_centers")
    u_grid    = velocity.get("_u_grid")
    v_grid    = velocity.get("_v_grid")
    occ       = velocity.get("_occ")
    pos       = velocity.get("_pos")

    if x_centers is None:
        _safe_fallback_png(output_path)
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    # Density background
    if pos is not None and len(pos) > 0:
        ax.hexbin(pos[:, 0], pos[:, 1], gridsize=40, cmap="Blues",
                  mincnt=1, alpha=0.65, zorder=1)

    # Velocity quiver (only occupied cells)
    XX, YY = np.meshgrid(x_centers, y_centers, indexing="ij")
    u_plot = np.where(occ, u_grid, np.nan)
    v_plot = np.where(occ, v_grid, np.nan)
    speed = np.sqrt(u_plot**2 + v_plot**2)
    speed_max = np.nanmax(speed)
    if speed_max > 0:
        u_plot /= speed_max
        v_plot /= speed_max

    ax.quiver(XX, YY, u_plot, v_plot,
              speed, cmap="hot_r", alpha=0.9, scale=30,
              width=0.004, zorder=2)

    # Eigenvalue annotation
    evals = velocity.get("jacobian_eigenvalues", [])
    eval_txt = "  ".join(
        f"λ{i+1}={e['real']:.3f}{'+' if e['imag']>=0 else ''}{e['imag']:.3f}i"
        for i, e in enumerate(evals)
    )
    ri = velocity.get("rotation_index", float("nan"))
    has_neut = velocity.get("has_neutral_direction", False)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(
        "Phase Velocity Field (PC1-PC2)\n"
        f"Rotation index={ri:.2f}  |  neutral direction={'YES' if has_neut else 'NO'}\n"
        f"{eval_txt}"
    )
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_recurrence(rqa: Dict, output_path: Path) -> None:
    """Recurrence matrix heatmap with RQA metrics annotated."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _safe_fallback_png(output_path)
        return

    R = rqa.get("_R")
    if R is None:
        _safe_fallback_png(output_path)
        return

    T = R.shape[0]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(R, origin="lower", cmap="binary", aspect="auto",
              interpolation="nearest")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Time step")

    det = rqa.get("DET", float("nan"))
    lam = rqa.get("LAM", float("nan"))
    rr  = rqa.get("RR",  float("nan"))
    tt  = rqa.get("TT",  float("nan"))
    cls = rqa.get("rqa_classification", "?")

    ax.set_title(
        f"Recurrence Plot  (T={T}, ε={rqa.get('epsilon', 0):.4f})\n"
        f"RR={rr:.3f}  DET={det:.3f}  LAM={lam:.3f}  TT={tt:.1f}  → {cls}"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_hypothesis_scorecard(scoring: Dict, output_path: Path) -> None:
    """Horizontal bar chart of normalised hypothesis scores."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _safe_fallback_png(output_path)
        return

    ranking = scoring.get("hypothesis_ranking", [])
    if not ranking:
        _safe_fallback_png(output_path)
        return

    names  = [r["name"] for r in ranking]
    scores = [r["score"] for r in ranking]
    # Reverse for bottom-up display
    names  = names[::-1]
    scores = scores[::-1]

    colours = plt.cm.RdYlGn(np.array(scores) / (max(scores) + 1e-9))

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(names, scores, color=colours, edgecolor="k", linewidth=0.5)

    # Annotate evidence on top hypothesis bar
    top_evidence = ranking[0].get("evidence", [])
    if top_evidence:
        top_name = ranking[0]["name"]
        # The bar for the top hypothesis is the last one (reversed list)
        ev_txt = "; ".join(top_evidence[:3])
        ax.text(scores[-1] + 0.002, len(names) - 1, ev_txt,
                va="center", ha="left", fontsize=7, color="k",
                wrap=True)

    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Normalised Score (0-1)")
    top_hyp  = scoring.get("top_hypothesis_name", "?")
    conf     = scoring.get("confidence", "?")
    conf_r   = scoring.get("confidence_ratio", float("nan"))
    ax.set_title(
        f"Attractor Topology Hypothesis Scorecard\n"
        f"Top: {top_hyp}  ({conf} confidence, ratio={conf_r:.2f})"
    )
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _safe_fallback_png(path: Path) -> None:
    """Write minimal fallback PNG if matplotlib is unavailable."""
    try:
        from spectral_dynamics.plot_utils import write_fallback_png
        write_fallback_png(path)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_attractor_topology(
    trajectories: np.ndarray,
    output_dir: Optional[Path] = None,
    burnin: int = 0,
    dt: float = 1.0,
    n_pcs: int = _N_PCS,
    rosenstein_lle: Optional[float] = None,
    n_pca_90: Optional[int] = None,
    seed: int = 42,
    rqa_max_T: int = 500,
    kmeans_uniform_suspect: bool = False,
    dmd_n_hopf_pairs: Optional[int] = None,
) -> Dict:
    """Run the complete attractor topology analysis.

    Args:
        trajectories:           shape (n_traj, T, N) — free-dynamics trajectories.
        output_dir:             directory to write plots and JSON.  None → no files.
        burnin:                 steps to skip at the start of each trajectory.
        dt:                     time step size (for frequency axis labelling).
        n_pcs:                  number of PCs to use internally (default 6).
        rosenstein_lle:         pre-computed Rosenstein LLE from pipeline Phase 3d
                                (improves hypothesis scoring).
        n_pca_90:               pre-computed n@90% PCA threshold from pipeline Phase
                                3g (used as an additional dimension evidence channel).
        seed:                   random seed for local dimensionality sampling.
        rqa_max_T:              truncate to this many steps for RQA (avoids O(T²)
                                cost for very long trajectories).
        kmeans_uniform_suspect: True if Phase 3b KMeans detected nearly equal-sized
                                clusters + high silhouette (ring/continuous attractor
                                signature from attractor_analysis.py).
        dmd_n_hopf_pairs:       number of Hopf pairs from Phase 3e DMD spectrum
                                analysis (improves SM hypothesis scoring; None if
                                DMD was not run or failed).

    Returns:
        Serialisable dict with keys:
          spectral, velocity_field, recurrence, local_dimensionality,
          period_stability, scoring, top_hypothesis, confidence.
    """
    n_traj, T, N = trajectories.shape
    burnin = min(burnin, T - 1)
    T_eff  = T - burnin
    logger.info(
        "Attractor topology: n_traj=%d, T=%d (burnin=%d, T_eff=%d), N=%d",
        n_traj, T, burnin, T_eff, N,
    )

    # ── Step 0: PCA projections ─────────────────────────────────────────────
    try:
        Z, evr = _fit_pca(trajectories, burnin=burnin, n_pcs=n_pcs)
        logger.info("  PCA: top2_var=%.1f%%, top5_var=%.1f%%",
                    float(evr[:2].sum()) * 100, float(evr[:5].sum()) * 100)
    except Exception as e:
        logger.error("PCA failed in attractor_topology: %s", e)
        return {"error": f"PCA failed: {e}"}

    # ── E1: Frequency analysis ──────────────────────────────────────────────
    try:
        spectral = _spectral_analysis(Z, dt=dt)
        logger.info(
            "  Spectral: freq_class=%s, n_peaks=%d",
            spectral["freq_classification"], spectral["n_total_peaks"],
        )
    except Exception as e:
        logger.warning("  Spectral analysis failed: %s", e)
        spectral = {"freq_classification": "error", "n_total_peaks": 0, "per_pc": []}

    # ── E2: Velocity field ──────────────────────────────────────────────────
    try:
        velocity = _velocity_field_analysis(Z)
        logger.info(
            "  Velocity: rot_idx=%.3f, neutral=%s, min_eig_re=%s",
            velocity.get("rotation_index", float("nan")),
            velocity.get("has_neutral_direction", "?"),
            velocity.get("min_eigenvalue_real", "?"),
        )
    except Exception as e:
        logger.warning("  Velocity field analysis failed: %s", e)
        velocity = {}

    # ── E3: RQA ─────────────────────────────────────────────────────────────
    try:
        rqa = _recurrence_analysis(Z, max_T=rqa_max_T)
        logger.info(
            "  RQA: RR=%.3f, DET=%.3f, LAM=%.3f → %s",
            rqa.get("RR", float("nan")), rqa.get("DET", float("nan")),
            rqa.get("LAM", float("nan")), rqa.get("rqa_classification", "?"),
        )
    except Exception as e:
        logger.warning("  RQA failed: %s", e)
        rqa = {}

    # ── E4: Local dimensionality ─────────────────────────────────────────────
    try:
        local_dim = _local_dimensionality(Z, seed=seed)
        logger.info(
            "  Local dim: mean=%.2f±%.2f, median=%s",
            local_dim.get("local_dim_mean", float("nan")),
            local_dim.get("local_dim_std", float("nan")),
            local_dim.get("local_dim_median", "?"),
        )
    except Exception as e:
        logger.warning("  Local dimensionality failed: %s", e)
        local_dim = {}

    # ── E6: Period stability (Hilbert transform) ─────────────────────────────
    try:
        period_stability = _period_stability_analysis(Z, dt=dt)
        if "error" not in period_stability:
            logger.info(
                "  Period stability: class=%s, period_cv=%s, envelope_var=%.3f",
                period_stability.get("stability_class", "?"),
                period_stability.get("period_cv"),
                period_stability.get("slow_envelope_variation", float("nan")),
            )
    except Exception as e:
        logger.warning("  Period stability analysis failed: %s", e)
        period_stability = {"error": str(e)}

    # ── E5: Hypothesis scoring ──────────────────────────────────────────────
    try:
        scoring = _score_hypotheses(
            spectral=spectral,
            velocity=velocity,
            rqa=rqa,
            local_dim=local_dim,
            rosenstein_lle=rosenstein_lle,
            n_pca_90=n_pca_90,
            period_stability=period_stability,
            kmeans_uniform_suspect=kmeans_uniform_suspect,
            dmd_n_hopf_pairs=dmd_n_hopf_pairs,
        )
        logger.info(
            "  Top hypothesis: %s (score=%.3f, %s confidence)",
            scoring.get("top_hypothesis_name", "?"),
            scoring.get("top_score", float("nan")),
            scoring.get("confidence", "?"),
        )
    except Exception as e:
        logger.warning("  Hypothesis scoring failed: %s", e)
        scoring = {}

    # ── Build serialisable result ───────────────────────────────────────────
    def _strip_internal(d: Dict) -> Dict:
        """Remove numpy-array internal keys (prefixed with '_')."""
        return {k: v for k, v in d.items() if not k.startswith("_")}

    result: Dict = {
        "spectral":             _strip_internal(spectral),
        "velocity_field":       _strip_internal(velocity),
        "recurrence":           _strip_internal(rqa),
        "local_dimensionality": local_dim,
        "period_stability":     period_stability,
        "scoring":              scoring,
        "top_hypothesis":       scoring.get("top_hypothesis"),
        "top_hypothesis_name":  scoring.get("top_hypothesis_name"),
        "confidence":           scoring.get("confidence"),
        "n_traj": n_traj, "T_eff": T_eff, "N": N, "n_pcs_used": Z.shape[2],
    }

    # ── Write outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # JSON report
        report_path = out / "attractor_topology_report.json"
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                def _json_default(o):
                    if isinstance(o, (np.floating, np.float32, np.float64)):
                        return float(o)
                    if isinstance(o, (np.integer, np.int32, np.int64)):
                        return int(o)
                    if isinstance(o, np.ndarray):
                        return o.tolist()
                    return str(o)  # last resort: str() is valid JSON string
                json.dump(result, f, indent=2, ensure_ascii=False,
                          default=_json_default)
            logger.info("Saved: %s", report_path)
        except Exception as e:
            logger.warning("Could not save JSON report: %s", e)

        # Plots
        _plot_freq_spectrum(spectral, out / "freq_spectrum_pcs.png")
        _plot_velocity_field(velocity, Z, out / "velocity_field_pca.png")
        _plot_recurrence(rqa, out / "recurrence_plot.png")
        _plot_hypothesis_scorecard(scoring, out / "hypothesis_scorecard.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s",
    )
    ap = argparse.ArgumentParser(
        description="Attractor Topology Analysis — rigorous hypothesis testing",
    )
    ap.add_argument("--trajectories", required=True,
                    help="Path to trajectories.npy  (n_traj, T, N)")
    ap.add_argument("--output", default=None,
                    help="Output directory (default: same folder as trajectories)")
    ap.add_argument("--burnin", type=int, default=0,
                    help="Burnin steps to skip per trajectory (default 0)")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="Time step size for frequency axis (default 1.0)")
    ap.add_argument("--n-pcs", type=int, default=_N_PCS,
                    help=f"Number of PCA components to use (default {_N_PCS})")
    ap.add_argument("--rosenstein-lle", type=float, default=None,
                    help="Pre-computed Rosenstein LLE from main pipeline")
    ap.add_argument("--n-pca-90", type=int, default=None,
                    help="Pre-computed PCA n@90%% from main pipeline")
    args = ap.parse_args()

    traj_path = Path(args.trajectories)
    if not traj_path.exists():
        logger.error("trajectories file not found: %s", traj_path)
        sys.exit(1)

    trajectories = np.load(str(traj_path))
    out_dir = Path(args.output) if args.output else traj_path.parent / "attractor_topology"

    result = run_attractor_topology(
        trajectories=trajectories,
        output_dir=out_dir,
        burnin=args.burnin,
        dt=args.dt,
        n_pcs=args.n_pcs,
        rosenstein_lle=args.rosenstein_lle,
        n_pca_90=args.n_pca_90,
    )

    print("\n=== Attractor Topology Summary ===")
    print(f"  Top hypothesis : {result.get('top_hypothesis_name', '?')}")
    print(f"  Confidence     : {result.get('confidence', '?')}")
    print(f"  Freq class     : {result.get('spectral', {}).get('freq_classification', '?')}")
    rqa_r = result.get("recurrence", {})
    print(f"  RQA DET / LAM  : {rqa_r.get('DET', '?')} / {rqa_r.get('LAM', '?')}")
    ld_r = result.get("local_dimensionality", {})
    print(f"  Local dim      : {ld_r.get('local_dim_mean', '?')} "
          f"± {ld_r.get('local_dim_std', '?')}")
    print(f"\nOutputs written to: {out_dir}")


if __name__ == "__main__":
    _cli()
