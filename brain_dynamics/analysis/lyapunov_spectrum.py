"""
Lyapunov Spectrum Analysis — Experiment 1
==========================================

Extracts the **full linearized Lyapunov spectrum** from the DMD operator
computed in Phase 3e and assesses whether the system exhibits a
**strange attractor** (chaotic, low-dimensional, dissipative).

Scientific basis
----------------
DMD fits the best linear operator A: x(t+1) ≈ A·x(t) from trajectory data.
Its eigenvalues μ_i yield the linearised Lyapunov exponents:

    λᵢ = ln|μᵢ|    (nats/step)

For a **strange attractor** the descending spectrum satisfies:

    λ₁ > 0     — at least one positive exponent  (exponential divergence / chaos)
    λ₂ ≈ 0     — neutral direction along the flow
    λ₃ < 0     — first contracting direction
    ...
    λₙ ≪ 0    — strong contraction (dissipativity)

The Kaplan–Yorke dimension  D_KY = j + Σᵢ₌₁ʲ λᵢ / |λⱼ₊₁|  (j = largest index
with non-negative partial sum) estimates the fractal dimension of the attractor.

Important caveats
-----------------
* This is a **linearised estimate**.  The true nonlinear λ₁ from Rosenstein is
  the authoritative chaos indicator (Phase 3d).
* DMD cannot capture non-linear folding / stretching, so the spectrum is an
  approximation.  Use it for dimensional structure, not absolute Lyapunov values.
* If DMD was not run (Phase 3e disabled), this module returns an empty dict.

Outputs
-------
lyapunov_spectrum.npy           — shape (N,), descending linearised spectrum
lyapunov_spectrum_report.json   — strange-attractor assessment
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Thresholds for strange attractor classification
_POSITIVE_THRESH = 0.001   # λ₁ > this → positive (chaotic)
_NEUTRAL_THRESH  = 0.005   # |λ₂| < this → neutral direction
_NEGATIVE_THRESH = -0.005  # λ₃ < this → contracting


def _kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """Kaplan–Yorke dimension from a descending Lyapunov spectrum."""
    s = np.asarray(spectrum, dtype=np.float64)
    if len(s) == 0 or s[0] < 0:
        return 0.0
    cs = np.cumsum(s)
    j = int(np.sum(cs >= 0))
    if j >= len(s):
        return float(len(s))
    if j == 0:
        return 0.0
    denom = abs(float(s[j]))
    return float(j) + float(cs[j - 1]) / denom if denom > 1e-20 else float(j)


def assess_strange_attractor(
    spectrum: np.ndarray,
    rosenstein_lle: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Assess whether the spectrum is consistent with a strange attractor.

    Parameters
    ----------
    spectrum:
        Descending linearised Lyapunov spectrum (from DMD eigenvalues).
    rosenstein_lle:
        Non-linear λ₁ from Rosenstein method.  Used as the authoritative
        chaos indicator when available.

    Returns
    -------
    assessment dict with keys:
        is_chaotic         bool  — λ₁ (or Rosenstein) > 0
        has_neutral        bool  — |λ₂| < _NEUTRAL_THRESH
        is_dissipative     bool  — sum(spectrum) < 0
        n_positive         int   — number of positive exponents
        n_neutral          int   — number of near-zero exponents
        n_negative         int   — number of negative exponents
        strange_attractor  str   — "likely" / "possible" / "unlikely"
        ky_dimension       float — Kaplan–Yorke dimension
        evidence           list  — bullet-point evidence summary
    """
    s = np.asarray(spectrum, dtype=np.float64)
    N = len(s)
    if N == 0:
        return {"strange_attractor": "unknown", "error": "empty spectrum"}

    lam1 = float(s[0])
    lam2 = float(s[1]) if N > 1 else float("nan")
    lam3 = float(s[2]) if N > 2 else float("nan")

    # Use Rosenstein as authoritative chaos indicator if available
    is_chaotic_lin = lam1 > _POSITIVE_THRESH
    if rosenstein_lle is not None:
        is_chaotic = rosenstein_lle > _POSITIVE_THRESH
        chaos_source = "Rosenstein"
    else:
        is_chaotic = is_chaotic_lin
        chaos_source = "DMD linearised"

    has_neutral = np.isfinite(lam2) and abs(lam2) < _NEUTRAL_THRESH
    is_dissipative = float(np.sum(s)) < 0.0

    n_positive = int(np.sum(s > _POSITIVE_THRESH))
    n_neutral = int(np.sum(np.abs(s) <= _NEUTRAL_THRESH))
    n_negative = int(np.sum(s < _NEGATIVE_THRESH))

    ky_dim = _kaplan_yorke_dimension(s)

    evidence: List[str] = []

    # Chaos criterion
    if is_chaotic:
        evidence.append(
            f"✓ λ₁={rosenstein_lle if rosenstein_lle is not None else lam1:.5f} > 0 "
            f"({chaos_source}) — exponential divergence / chaos"
        )
    else:
        evidence.append(
            f"✗ λ₁={rosenstein_lle if rosenstein_lle is not None else lam1:.5f} ≤ 0 "
            f"— no exponential divergence"
        )

    # Neutral direction
    if np.isfinite(lam2):
        if has_neutral:
            evidence.append(f"✓ λ₂={lam2:.5f} ≈ 0 — neutral direction along flow")
        else:
            evidence.append(f"○ λ₂={lam2:.5f} not near-zero — expected for strange attractor")

    # Contraction
    if is_dissipative:
        evidence.append(
            f"✓ Σλ={np.sum(s):.4f} < 0 — dissipative system "
            f"({n_negative}/{N} negative exponents)"
        )
    else:
        evidence.append(f"✗ Σλ={np.sum(s):.4f} ≥ 0 — not dissipative")

    # Dimensional structure
    evidence.append(
        f"K-Y dimension = {ky_dim:.2f} "
        f"(n_positive={n_positive}, n_neutral={n_neutral}, n_negative={n_negative})"
    )

    if np.isfinite(lam3):
        evidence.append(
            f"λ₁={lam1:.5f}, λ₂={lam2:.5f}, λ₃={lam3:.5f} "
            f"[DMD linearised top-3]"
        )

    # Overall verdict
    if is_chaotic and is_dissipative and n_negative > n_positive:
        verdict = "likely"
    elif is_chaotic and is_dissipative:
        verdict = "possible"
    elif is_chaotic:
        verdict = "possible (non-dissipative chaos)"
    else:
        verdict = "unlikely"

    return {
        "is_chaotic": bool(is_chaotic),
        "is_chaotic_linearised": bool(is_chaotic_lin),
        "has_neutral_direction": bool(has_neutral),
        "is_dissipative": bool(is_dissipative),
        "n_positive": n_positive,
        "n_neutral": n_neutral,
        "n_negative": n_negative,
        "strange_attractor": verdict,
        "lambda_1_dmd": float(lam1),
        "lambda_2_dmd": float(lam2) if np.isfinite(lam2) else None,
        "lambda_3_dmd": float(lam3) if np.isfinite(lam3) else None,
        "lambda_sum": float(np.sum(s)),
        "ky_dimension": float(ky_dim),
        "chaos_source": chaos_source,
        "evidence": evidence,
    }


def run_lyapunov_spectrum_analysis(
    dmd_spectrum: Dict[str, Any],
    trajectories: np.ndarray,
    rosenstein_lle: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full Lyapunov spectrum analysis (Experiment 1).

    Extracts the complete linearised Lyapunov spectrum from DMD eigenvalues
    and assesses whether the system exhibits strange attractor dynamics.

    Parameters
    ----------
    dmd_spectrum:
        Result dict from ``run_jacobian_analysis`` (Phase 3e).  Must contain
        at least ``linearised_lyapunov_spectrum``.
    trajectories:
        Free-dynamics trajectories, shape (n_traj, T, N).  Used to compute
        supplementary statistics (trajectory variance, mean norm).
    rosenstein_lle:
        Scalar λ₁ from Rosenstein analysis (Phase 3d).  Used as authoritative
        chaos indicator.
    output_dir:
        Directory to save outputs.  If None, nothing is saved.

    Returns
    -------
    Dict with keys:
        spectrum            np.ndarray  (N,) descending linearised spectrum
        assessment          dict        strange attractor assessment
        ky_dimension        float       Kaplan–Yorke dimension
        n_positive          int
        n_negative          int
        trajectory_stats    dict        mean_norm, mean_var from trajectories
    """
    result: Dict[str, Any] = {}

    # ── Extract linearised spectrum from DMD ──────────────────────────────────
    lin_spec = dmd_spectrum.get("linearised_lyapunov_spectrum")
    if lin_spec is None:
        # Try to compute from eigenvalues directly
        eigvals = dmd_spectrum.get("eigenvalues")
        if eigvals is not None:
            eigvals = np.asarray(eigvals)
            lin_spec = np.sort(np.log(np.maximum(np.abs(eigvals), 1e-30)))[::-1]
        else:
            logger.warning(
                "Lyapunov spectrum: DMD has no eigenvalues. "
                "Cannot compute spectrum. Is Phase 3e enabled?"
            )
            return {"error": "no_dmd_eigenvalues"}

    spectrum = np.asarray(lin_spec, dtype=np.float64)
    result["spectrum"] = spectrum
    result["ky_dimension"] = float(_kaplan_yorke_dimension(spectrum))
    result["n_positive"] = int(np.sum(spectrum > _POSITIVE_THRESH))
    result["n_neutral"] = int(np.sum(np.abs(spectrum) <= _NEUTRAL_THRESH))
    result["n_negative"] = int(np.sum(spectrum < _NEGATIVE_THRESH))
    result["lambda_sum"] = float(np.sum(spectrum))
    result["lambda_max_dmd"] = float(spectrum[0]) if len(spectrum) > 0 else float("nan")

    # ── Assess strange attractor ──────────────────────────────────────────────
    assessment = assess_strange_attractor(spectrum, rosenstein_lle=rosenstein_lle)
    result["assessment"] = assessment
    result["strange_attractor"] = assessment["strange_attractor"]

    # ── Trajectory statistics ─────────────────────────────────────────────────
    if trajectories is not None and trajectories.ndim == 3:
        norms = np.linalg.norm(trajectories.reshape(-1, trajectories.shape[-1]), axis=1)
        result["trajectory_stats"] = {
            "mean_norm": float(norms.mean()),
            "std_norm": float(norms.std()),
            "mean_var": float(trajectories.var(axis=(1, 2)).mean()),
        }

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        np.save(output_dir / "lyapunov_spectrum.npy", spectrum)

        report = {
            "ky_dimension": result["ky_dimension"],
            "n_positive": result["n_positive"],
            "n_neutral": result["n_neutral"],
            "n_negative": result["n_negative"],
            "lambda_sum": result["lambda_sum"],
            "lambda_max_dmd": result["lambda_max_dmd"],
            "rosenstein_lle": float(rosenstein_lle) if rosenstein_lle is not None else None,
            "strange_attractor": result["strange_attractor"],
            "assessment": {
                k: v for k, v in assessment.items()
                if not isinstance(v, np.ndarray)
            },
            "top10_spectrum": spectrum[:10].tolist(),
            "trajectory_stats": result.get("trajectory_stats"),
        }
        with open(output_dir / "lyapunov_spectrum_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(
            "  Lyapunov spectrum (DMD linearised): K-Y_lin=%.2f, n_pos=%d, "
            "n_neg=%d, strange=%s  "
            "[NOTE: linearised exponents from DMD; n_pos=0 is expected when "
            "DMD spectral radius ≤ 1 — does NOT contradict a positive "
            "Rosenstein LLE (which measures true nonlinear divergence)]",
            result["ky_dimension"], result["n_positive"],
            result["n_negative"], result["strange_attractor"],
        )

    return result
