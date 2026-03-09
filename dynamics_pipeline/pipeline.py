#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics_pipeline — Unified Brain Dynamics Analysis Pipeline
============================================================

Orchestrates the ``brain_dynamics`` support library through a single,
config-driven pipeline.  ``brain_dynamics`` is the consolidated package that
merges:
  - the former ``twinbrain-dynamics`` (model-driven GNN analysis: simulator,
    loader, analysis algorithms, experiments)
  - the former ``spectral_dynamics`` (matrix-driven spectral analysis:
    connectivity, community, modal projection, phase diagrams, etc.),
    now available as ``brain_dynamics.spectral_dynamics``.

Phases:
  1  Data generation   — free dynamics & response matrix (model calls)
  2  Network structure  — spectral, community, hierarchy, modal energy
  3  Dynamics regime    — stability, LLE, DMD spectrum, PSD, PCA
  4  Validation         — surrogate test, random comparison, embedding dim
  5  Advanced           — stimulation, energy, controllability, TE, CSD
  6  Synthesis          — hypothesis evaluation & consistency checks

Usage::

    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt
    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

_REPO_ROOT = Path(__file__).resolve().parent.parent
_BD_DIR = _REPO_ROOT / "brain_dynamics"

for _p in (_REPO_ROOT, _BD_DIR):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np

logger = logging.getLogger("dynamics_pipeline")

# ── Constants ─────────────────────────────────────────────────────────────────
_PS_BURNIN_MIN = 20
_PS_BURNIN_FRACTION = 10

# Regimes consistent with near-critical brain hypothesis (H3).
# Must match classifications from analysis.lyapunov.run_lyapunov_analysis.
_NEAR_CRITICAL_REGIMES = frozenset({
    "edge_of_chaos", "marginal_stable", "weakly_chaotic",
})


def _kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """Kaplan-Yorke dimension from a descending Lyapunov spectrum.

    D_KY = j + (Σᵢ₌₁ʲ λ_i) / |λ_{j+1}|
    where j is the largest index with non-negative partial sum.
    """
    s = np.sort(np.asarray(spectrum, dtype=np.float64))[::-1]
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


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase runners
# ═══════════════════════════════════════════════════════════════════════════════

def _step(phase: int, name: str, total: int = 6) -> None:
    logger.info("=" * 60)
    logger.info("Phase %d/%d  %s", phase, total, name)


def _get_square_connectivity(
    results: Dict[str, Any],
    trajs: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    """Return a square (N×N) connectivity matrix for Phase 5 analyses.

    Priority:
      1. Response matrix R if it is already square (N×N).
      2. DMD operator matrix (dmd_A) from Phase 3e — always N×N.
      3. Functional connectivity (Pearson) derived from free-dynamics
         trajectories — always N×N.

    Returns None when no square matrix can be constructed.
    """
    R = results.get("response_matrix")
    if R is not None and R.ndim == 2 and R.shape[0] == R.shape[1]:
        return R

    # Non-square response matrix (n_nodes × N_regions): prefer DMD operator.
    if R is not None and R.ndim == 2 and R.shape[0] != R.shape[1]:
        logger.info(
            "  Response matrix is non-square (%s); falling back to "
            "square connectivity for Phase 5 analyses.",
            "×".join(str(d) for d in R.shape),
        )

    # Try DMD operator from Phase 3e (always N×N)
    dmd = results.get("dmd_spectrum", {})
    dmd_A = dmd.get("dmd_A")
    if dmd_A is not None and dmd_A.ndim == 2 and dmd_A.shape[0] == dmd_A.shape[1]:
        logger.info("  Using DMD operator (N=%d) as square connectivity.", dmd_A.shape[0])
        return dmd_A

    # Fall back to FC from trajectories
    if trajs is not None:
        stacked = trajs.reshape(-1, trajs.shape[-1])
        W_fc = np.corrcoef(stacked.T)
        W_fc = np.nan_to_num(W_fc, nan=0.0)
        logger.info("  Using FC from trajectories (N=%d) as square connectivity.", W_fc.shape[0])
        return W_fc

    return None


def run_phase1_data(cfg: dict, simulator, output_dir: Path,
                    device: str) -> Dict[str, Any]:
    """Phase 1: Generate free-dynamics trajectories and response matrix."""
    _step(1, "Data Generation")
    results: Dict[str, Any] = {}
    dg = cfg["data_generation"]

    from experiments.free_dynamics import run_free_dynamics
    trajectories = run_free_dynamics(
        simulator=simulator,
        n_init=dg.get("n_init", 100),
        steps=dg.get("steps", 500),
        seed=dg.get("seed", 42),
        output_dir=output_dir if cfg["output"].get("save_trajectories") else None,
        device=device,
        n_temporal_windows=dg.get("n_temporal_windows", None),
    )
    results["trajectories"] = trajectories
    logger.info("  → trajectories shape: %s", trajectories.shape)

    # Response matrix (Phase 5 / advanced, but generated here for Phase 2)
    adv = cfg.get("advanced", {})
    rm_cfg = adv.get("response_matrix", {})
    if rm_cfg.get("enabled", True):
        from analysis.response_matrix import compute_response_matrix
        n_nodes_cfg = rm_cfg.get("n_nodes", None)
        if n_nodes_cfg is None:
            n_nodes = simulator.n_regions
        else:
            try:
                n_nodes = min(int(n_nodes_cfg), simulator.n_regions)
            except (TypeError, ValueError):
                logger.warning(
                    "advanced.response_matrix.n_nodes=%r is not a valid integer; "
                    "falling back to all %d regions.",
                    n_nodes_cfg, simulator.n_regions,
                )
                n_nodes = simulator.n_regions
        response_matrix = compute_response_matrix(
            simulator=simulator,
            n_nodes=n_nodes,
            stim_amplitude=rm_cfg.get("stim_amplitude", 0.5),
            output_dir=output_dir,
        )
        results["response_matrix"] = response_matrix
        logger.info("  → response_matrix shape: %s", response_matrix.shape)

    return results


def run_phase2_structure(cfg: dict, results: Dict[str, Any],
                         output_dir: Path) -> None:
    """Phase 2: Network structure analysis from connectivity matrices."""
    ns_cfg = cfg.get("network_structure", {})
    if not ns_cfg.get("enabled", True):
        logger.info("Phase 2 (Network Structure) disabled, skipping.")
        return
    _step(2, "Network Structure")

    R = results.get("response_matrix")
    trajs = results.get("trajectories")

    # Determine connectivity matrix W
    # Phase 2 analyses (spectral decomposition, community detection, modal
    # projection) all require a SQUARE (N×N) matrix.  The response matrix R
    # may have shape (n_stimulated_nodes, N) when n_nodes < N (e.g. n_nodes=10
    # in the default config).  In that case we fall back to the functional
    # connectivity matrix derived from the free-dynamics trajectories.
    W = None
    w_label = "response_matrix"
    # P0-4: evidence grading
    #   A: response_matrix (N×N causal/interventional, strongest)
    #   B: graph_cache structural connectivity (physical meaning, if present)
    #   C: FC (Pearson correlation, statistical only — weakest)
    connectivity_source: Optional[str] = None
    evidence_grade: Optional[str] = None
    if R is not None and R.ndim == 2 and R.shape[0] == R.shape[1]:
        W = R
        connectivity_source = "response_matrix"
        evidence_grade = "A"
    elif trajs is not None:
        # Compute functional connectivity from trajectories
        stacked = trajs.reshape(-1, trajs.shape[-1])
        W = np.corrcoef(stacked.T)
        W = np.nan_to_num(W, nan=0.0)
        w_label = "fc"
        connectivity_source = "fc"
        evidence_grade = "C"
        if R is not None:
            logger.info(
                "  Response matrix is non-square (%s); using FC from "
                "trajectories for Phase 2 structure analysis.",
                "×".join(str(d) for d in R.shape),
            )
        else:
            logger.info("  No response matrix; using FC from trajectories.")

    if W is None:
        logger.warning("  No connectivity matrix available, skipping Phase 2.")
        return

    # Store evidence grade for Phase 6 reporting
    results["connectivity_source"] = connectivity_source
    results["evidence_grade"] = evidence_grade
    if evidence_grade == "C":
        logger.info(
            "  Evidence grade: C (FC only). Spectral radius/eigenvalues "
            "cannot be interpreted as Jacobian/causal stability boundary."
        )
    elif evidence_grade == "A":
        logger.info("  Evidence grade: A (response_matrix, N×N causal).")

    struct_dir = output_dir / "structure"
    struct_dir.mkdir(exist_ok=True)

    # 2a: Spectral decomposition
    if ns_cfg.get("spectral", {}).get("enabled", True):
        try:
            from spectral_dynamics.e1_spectral_analysis import run_spectral_analysis
            spec = run_spectral_analysis(W, label=w_label, output_dir=struct_dir)
            results["spectral"] = spec
            logger.info(
                "  Spectral: ρ=%.4f, PR=%.2f, n_dominant=%d, gap=%.4f",
                spec["spectral_radius"], spec["participation_ratio"],
                spec["n_dominant"], spec["spectral_gap_ratio"],
            )
        except Exception as e:
            logger.warning("  Spectral analysis failed: %s", e)

    # 2b: Community detection
    comm_cfg = ns_cfg.get("community", {})
    if comm_cfg.get("enabled", True):
        try:
            from spectral_dynamics.c_community_structure import run_community_structure
            comm = run_community_structure(
                W, k_range=comm_cfg.get("k_range", [3, 4, 5, 6, 7, 8]),
                label=w_label, output_dir=struct_dir,
            )
            results["community"] = comm
            logger.info(
                "  Community: Q=%.4f, k=%d, method=%s",
                comm["modularity_q"], comm["n_communities"],
                comm.get("method", "unknown"),
            )
        except Exception as e:
            logger.warning("  Community detection failed: %s", e)

    # 2c: Hierarchy
    if ns_cfg.get("hierarchy", {}).get("enabled", False):
        try:
            from spectral_dynamics.d_hierarchical_structure import run_hierarchical_structure
            hier = run_hierarchical_structure(W, label=w_label, output_dir=struct_dir)
            results["hierarchy"] = hier
            logger.info("  Hierarchy index: %.4f", hier.get("hierarchy_index", 0))
        except Exception as e:
            logger.warning("  Hierarchical analysis failed: %s", e)

    # 2d: Modal energy
    if ns_cfg.get("modal_energy", {}).get("enabled", True) and trajs is not None:
        try:
            from spectral_dynamics.e2_e3_modal_projection import run_modal_projection
            modal = run_modal_projection(
                trajs, W, label=w_label, output_dir=struct_dir,
            )
            results["modal_energy"] = modal
            logger.info(
                "  Modal energy: top5=%.1f%%, n@90%%=%d",
                modal.get("energy_top5_pct", 0),
                modal.get("n_modes_90pct", 0),
            )
        except Exception as e:
            logger.warning("  Modal projection failed: %s", e)

    # 2e: Visualization
    if ns_cfg.get("visualization", {}).get("enabled", True):
        try:
            from spectral_dynamics.a_connectivity_visualization import run_connectivity_visualization
            comm_labels = results.get("community", {}).get("labels")
            run_connectivity_visualization(
                W, community_labels=comm_labels,
                label=w_label, output_dir=struct_dir,
            )
            logger.info("  Connectivity plots saved.")
        except Exception as e:
            logger.warning("  Connectivity visualization failed: %s", e)


def run_phase3_dynamics(cfg: dict, results: Dict[str, Any],
                        simulator, output_dir: Path,
                        modality: str = "fmri") -> None:
    """Phase 3: Dynamics characterisation from trajectory data."""
    dyn_cfg = cfg.get("dynamics", {})
    trajs = results.get("trajectories")
    if trajs is None:
        logger.warning("Phase 3 requires trajectories, skipping.")
        return
    _step(3, "Dynamics Characterisation")

    dyn_dir = output_dir / "dynamics"
    dyn_dir.mkdir(exist_ok=True)

    # 3a: Stability classification
    if dyn_cfg.get("stability", {}).get("enabled", True):
        try:
            from analysis.stability_analysis import run_stability_analysis
            stab = run_stability_analysis(
                trajectories=trajs,
                delay_dt=dyn_cfg.get("stability", {}).get("delay_dt", 50),
                output_dir=dyn_dir,
            )
            results["stability"] = stab
            v2 = stab.get("classification_counts_v2", {})
            logger.info("  Stability (Method C): %s", v2)
        except Exception as e:
            logger.warning("  Stability analysis failed: %s", e)

    # 3b: Attractor analysis
    att_cfg = dyn_cfg.get("attractor", {})
    if att_cfg.get("enabled", True):
        try:
            from experiments.attractor_analysis import run_attractor_analysis
            att = run_attractor_analysis(
                trajectories=trajs,
                tail_steps=att_cfg.get("tail_steps", 50),
                k_candidates=att_cfg.get("k_candidates", [2, 3, 4, 5, 6]),
                output_dir=dyn_dir,
            )
            results["attractor"] = att
            logger.info(
                "  Attractor: k=%d, basins=%s",
                att.get("k_best", 0),
                att.get("basin_distribution", {}),
            )
        except Exception as e:
            logger.warning("  Attractor analysis failed: %s", e)

    # 3c: Trajectory convergence
    conv_cfg = dyn_cfg.get("convergence", {})
    if conv_cfg.get("enabled", True):
        try:
            from analysis.trajectory_convergence import run_trajectory_convergence
            conv = run_trajectory_convergence(
                trajectories=trajs,
                n_pairs=conv_cfg.get("n_pairs", 50),
                output_dir=dyn_dir,
            )
            results["convergence"] = conv
            logger.info(
                "  Convergence: ratio=%.4f, label=%s",
                conv.get("distance_ratio", float("nan")),
                conv.get("label", "unknown"),
            )
        except Exception as e:
            logger.warning("  Trajectory convergence failed: %s", e)

    # 3d: Lyapunov exponent
    # Default method: "rosenstein" (zero extra model calls, no context-dilution bias).
    # For joint mode, Wolf/FTLE would clip z-scored states to [0,1] producing
    # spurious attractors — force rosenstein automatically.
    lya_cfg = dyn_cfg.get("lyapunov", {})
    if lya_cfg.get("enabled", True):
        try:
            from analysis.lyapunov import run_lyapunov_analysis
            lya_method = lya_cfg.get("method", "rosenstein")
            if modality == "joint" and lya_method != "rosenstein":
                logger.info(
                    "  joint modality: auto-switching Lyapunov method '%s' → "
                    "'rosenstein'. Wolf/FTLE relies on [0,1] state-space clipping; "
                    "the joint z-score state space is unbounded so clipping would "
                    "introduce spurious attractors.",
                    lya_method,
                )
                lya_method = "rosenstein"
            lya = run_lyapunov_analysis(
                trajectories=trajs,
                simulator=simulator,
                method=lya_method,
                convergence_result=results.get("convergence"),
                convergence_threshold=lya_cfg.get("convergence_threshold", 0.05),
                n_segments=lya_cfg.get("n_segments", 3),
                rosenstein_max_lag=lya_cfg.get("max_lag", 50),
                rosenstein_min_sep=lya_cfg.get("min_sep", 20),
                rosenstein_delay_embed_dim=lya_cfg.get("delay_embed_dim", 0),
                rosenstein_delay_embed_tau=lya_cfg.get("delay_embed_tau", 1),
                n_workers=lya_cfg.get("n_workers", 1),
                output_dir=dyn_dir,
            )
            results["lyapunov"] = lya
            regime = lya["chaos_regime"]["regime"]
            mean_lya = lya.get("mean_lyapunov", float("nan"))
            logger.info(
                "  Lyapunov (%s): λ=%.5f, regime=%s", lya_method, mean_lya, regime,
            )
        except Exception as e:
            logger.warning("  Lyapunov analysis failed: %s", e)

    # 3e: Linearised spectral analysis (DMD)
    # DMD fits the best linear operator A: x_{t+1} ≈ A·x_t from trajectory data.
    # Its eigenvalues give the LINEARISED dynamics — not true nonlinear Lyapunov
    # exponents.  Useful for: spectral radius ρ, slow modes, Hopf oscillations,
    # and linearised K-Y dimension.  See README §3e for scientific framing.
    dmd_cfg = dyn_cfg.get("dmd_spectrum", {})
    if dmd_cfg.get("enabled", True):
        try:
            from analysis.jacobian_analysis import run_jacobian_analysis
            dmd = run_jacobian_analysis(
                simulator=simulator,
                trajectories=trajs,
                tail_steps=dmd_cfg.get("tail_steps", 20),
                n_states=dmd_cfg.get("n_states", 3),
            )
            # Derive linearised Lyapunov spectrum from DMD eigenvalues:
            #   λ_DMD_i = ln|μ_i|  (nats/step, discrete-time convention)
            eigvals = dmd.get("eigenvalues")
            if eigvals is not None:
                eigvals = np.asarray(eigvals)
                lin_spectrum = np.sort(
                    np.log(np.maximum(np.abs(eigvals), 1e-30))
                )[::-1]
                dmd["linearised_lyapunov_spectrum"] = lin_spectrum
                dmd["linearised_ky_dim"] = float(
                    _kaplan_yorke_dimension(lin_spectrum)
                )
                logger.info(
                    "  DMD (linearised): ρ=%.4f, n_slow=%d, n_Hopf=%d, "
                    "K-Y_lin=%.2f",
                    dmd["spectral_radius"], dmd["n_slow_modes"],
                    dmd["n_hopf_pairs"], dmd["linearised_ky_dim"],
                )
            else:
                logger.info(
                    "  DMD: ρ=%.4f, n_slow=%d, n_Hopf=%d",
                    dmd["spectral_radius"], dmd["n_slow_modes"],
                    dmd["n_hopf_pairs"],
                )
            results["dmd_spectrum"] = dmd
        except Exception as e:
            logger.warning("  DMD analysis failed: %s", e)

    # 3f: Power spectrum
    if dyn_cfg.get("power_spectrum", {}).get("enabled", True):
        try:
            from analysis.power_spectrum import run_power_spectrum_analysis
            _T = trajs.shape[1] if trajs.ndim >= 2 else 0
            burnin = max(_PS_BURNIN_MIN, _T // _PS_BURNIN_FRACTION)
            psd = run_power_spectrum_analysis(
                trajectories=trajs,
                dt=simulator.dt,
                burnin=burnin,
                output_dir=dyn_dir if cfg["output"].get("save_plots") else None,
            )
            results["power_spectrum"] = psd
            ba = psd.get("band_analysis", {})
            logger.info(
                "  Power spectrum: f_dom=%.4f Hz [%s]",
                ba.get("dominant_freq_hz", 0),
                ba.get("dominant_freq_band", "?"),
            )
        except Exception as e:
            logger.warning("  Power spectrum failed: %s", e)

    # 3g: PCA dimensionality
    if dyn_cfg.get("pca", {}).get("enabled", True):
        try:
            from spectral_dynamics.f_pca_attractor import run_pca_attractor
            # With context-end-aligned x0 (from_data=True, step_idx=context_end),
            # the initial state IS the natural BOLD continuation of the context.
            # No large correction transient → small burnin (T//10) is sufficient.
            _T = trajs.shape[1] if trajs.ndim >= 2 else 0
            burnin = _pca_burnin(_T)
            pca = run_pca_attractor(
                trajectories=trajs,
                burnin=burnin,
                output_dir=dyn_dir,
            )
            results["pca"] = pca
            logger.info(
                "  PCA: var_top5=%.1f%%, n@90%%=%d",
                pca.get("variance_top5_pct", 0),
                pca.get("n_components_90pct", 0),
            )
        except Exception as e:
            logger.warning("  PCA analysis failed: %s", e)

    # 3h: Attractor dimension
    # Two complementary estimates:
    #   D₂ (correlation dimension): nonlinear, from Grassberger-Procaccia on trajectories
    #   K-Y_linear: from DMD linearised Lyapunov spectrum (already computed in 3e)
    ad_cfg = dyn_cfg.get("attractor_dimension", {})
    if ad_cfg.get("enabled", True):
        dim_results: Dict[str, Any] = {}

        # ── D₂ robustness: multi-trajectory / multi-window distribution ───────
        # P0-1: instead of a single trajectory, sample K trajectories × M windows
        # and report distribution statistics (mean/std/fail_rate/r2 quality).
        d2_cfg = ad_cfg.get("d2", {})
        if d2_cfg.get("enabled", True):
            try:
                from analysis.embedding_dimension import correlation_dimension
                rng_d2 = np.random.default_rng(d2_cfg.get("seed", 42))
                k_traj = min(d2_cfg.get("k_traj", 10), trajs.shape[0])
                m_windows = d2_cfg.get("m_windows", 5)
                window_len = d2_cfg.get("window_len", 300)
                burnin_frac = d2_cfg.get("burnin_fraction", 0.1)
                min_r2 = d2_cfg.get("min_r2", 0.90)

                T = trajs.shape[1]
                burnin_steps = max(0, int(T * burnin_frac))
                eff_T = T - burnin_steps
                window_len = min(window_len, eff_T)

                traj_indices = rng_d2.choice(trajs.shape[0], size=k_traj, replace=False)
                d2_values: List[float] = []
                r2_values: List[float] = []
                d2_n_total = 0
                d2_failures: List[str] = []

                for ti in traj_indices:
                    traj_i = trajs[ti, burnin_steps:, :]  # (eff_T, N)
                    if eff_T < window_len:
                        # Not enough steps: use full trajectory
                        window_starts = [0]
                    else:
                        # Sample m_windows non-overlapping start positions
                        possible = np.arange(0, eff_T - window_len + 1)
                        if len(possible) < m_windows:
                            window_starts = possible.tolist()
                        else:
                            window_starts = rng_d2.choice(
                                possible, size=m_windows, replace=False
                            ).tolist()

                    for ws in window_starts:
                        d2_n_total += 1
                        seg = traj_i[ws:ws + window_len]
                        try:
                            out = correlation_dimension(seg)
                            d2_val = out.get("D2", float("nan"))
                            r2_val = out.get("fit_r2", float("nan"))
                            if not np.isfinite(d2_val):
                                d2_failures.append("non-finite D2")
                                continue
                            if np.isfinite(r2_val) and r2_val < min_r2:
                                d2_failures.append(f"low_r2={r2_val:.3f}")
                                continue
                            d2_values.append(float(d2_val))
                            r2_values.append(float(r2_val) if np.isfinite(r2_val) else float("nan"))
                        except Exception as exc:
                            d2_failures.append(str(exc)[:200])

                d2_n_valid = len(d2_values)
                fail_rate = (d2_n_total - d2_n_valid) / max(d2_n_total, 1)

                if d2_n_valid > 0:
                    arr = np.array(d2_values)
                    r2_arr = np.array([v for v in r2_values if np.isfinite(v)])
                    dim_results["D2_mean"] = round(float(arr.mean()), 3)
                    dim_results["D2_std"] = round(float(arr.std()), 3)
                    dim_results["D2_median"] = round(float(np.median(arr)), 3)
                    dim_results["D2_iqr"] = round(
                        float(np.percentile(arr, 75) - np.percentile(arr, 25)), 3
                    )
                    dim_results["D2_n_total"] = d2_n_total
                    dim_results["D2_n_valid"] = d2_n_valid
                    dim_results["D2_fail_rate"] = round(fail_rate, 3)
                    dim_results["D2_r2_mean"] = round(float(r2_arr.mean()), 3) if len(r2_arr) > 0 else None
                    dim_results["D2_r2_min"] = round(float(r2_arr.min()), 3) if len(r2_arr) > 0 else None
                    # Backward-compat scalar for H2 evaluation
                    dim_results["D2"] = dim_results["D2_mean"]
                    dim_results["D2_fit_r2"] = dim_results["D2_r2_mean"]
                    logger.info(
                        "  Attractor dim: D₂=%.2f±%.2f (n=%d/%d, fail=%.0f%%, R²_mean=%.3f)",
                        dim_results["D2_mean"], dim_results["D2_std"],
                        d2_n_valid, d2_n_total,
                        fail_rate * 100,
                        dim_results["D2_r2_mean"] or float("nan"),
                    )
                    if fail_rate > 0.5:
                        logger.warning(
                            "  D₂ fail_rate=%.0f%% > 50%%: H2 evidence grade downgraded.",
                            fail_rate * 100,
                        )
                        dim_results["D2_h2_reliable"] = False
                    else:
                        dim_results["D2_h2_reliable"] = True
                else:
                    logger.warning(
                        "  D₂: all %d windows failed (%s …). H2 D₂ evidence unavailable.",
                        d2_n_total,
                        (d2_failures[0] if d2_failures else "unknown"),
                    )
                    dim_results["D2_n_total"] = d2_n_total
                    dim_results["D2_n_valid"] = 0
                    dim_results["D2_fail_rate"] = 1.0
                    dim_results["D2_h2_reliable"] = False
            except Exception as e:
                logger.warning("  D₂ distribution computation failed: %s", e)

        # K-Y from linearised DMD spectrum (already in dmd_spectrum)
        dmd = results.get("dmd_spectrum", {})
        ky_lin = dmd.get("linearised_ky_dim")
        if ky_lin is not None:
            dim_results["KY_linearised"] = ky_lin
            logger.info("  Attractor dim: K-Y_linear=%.2f (from DMD)", ky_lin)

        # PCA effective dimension as upper bound
        pca = results.get("pca", {})
        n90 = pca.get("n_components_90")
        if n90 is not None:
            dim_results["PCA_n90"] = n90
            logger.info("  Attractor dim: PCA n@90%%=%d (upper bound)", n90)

        if dim_results:
            results["attractor_dimension"] = dim_results

    # 3i: Full Lyapunov Spectrum (Experiment 1)
    # Extracts the complete linearised Lyapunov spectrum from DMD eigenvalues
    # and assesses whether the system exhibits strange attractor dynamics.
    # Zero extra model calls — uses DMD from 3e (if available).
    lsp_cfg = dyn_cfg.get("lyapunov_spectrum", {})
    if lsp_cfg.get("enabled", True):
        try:
            from analysis.lyapunov_spectrum import run_lyapunov_spectrum_analysis
            dmd = results.get("dmd_spectrum", {})
            lya = results.get("lyapunov", {})
            lsp = run_lyapunov_spectrum_analysis(
                dmd_spectrum=dmd,
                trajectories=trajs,
                rosenstein_lle=lya.get("mean_lyapunov"),
                output_dir=dyn_dir,
            )
            if "error" not in lsp:
                results["lyapunov_spectrum"] = lsp
                logger.info(
                    "  Lyapunov spectrum: K-Y=%.2f, n_pos=%d, n_neg=%d, "
                    "strange_attractor=%s",
                    lsp.get("ky_dimension", float("nan")),
                    lsp.get("n_positive", 0),
                    lsp.get("n_negative", 0),
                    lsp.get("strange_attractor", "?"),
                )
        except Exception as e:
            logger.warning("  Lyapunov spectrum analysis failed: %s", e)

    # 3j: Node Contribution Analysis (Experiment 3)
    # Identifies which brain regions are the primary drivers of the
    # low-dimensional dynamical manifold via PCA, DMD, and response matrix.
    # Zero extra model calls.
    nc_cfg = dyn_cfg.get("node_contribution", {})
    if nc_cfg.get("enabled", True):
        try:
            from analysis.node_contribution import run_node_contribution
            nc = run_node_contribution(
                trajectories=trajs,
                dmd_spectrum=results.get("dmd_spectrum"),
                response_matrix=results.get("response_matrix"),
                top_k=nc_cfg.get("top_k", 20),
                n_pca_components=nc_cfg.get("n_pca_components", 10),
                output_dir=dyn_dir,
            )
            results["node_contribution"] = nc
            logger.info(
                "  Node contribution: core_size=%d, top10=%.1f%%, has_core=%s",
                nc.get("dynamical_core_size", 0),
                nc.get("top10_contribution_pct", 0.0),
                nc.get("has_core", False),
            )
        except Exception as e:
            logger.warning("  Node contribution analysis failed: %s", e)

    # 3k: Attractor Basin Test (Experiment 2) — OPTIONAL, expensive
    # Tests for a single global attractor by generating trajectories from
    # diverse initial conditions (task / Gaussian / uniform).
    bt_cfg = dyn_cfg.get("basin_test", {})
    if bt_cfg.get("enabled", False):
        try:
            from experiments.basin_test import run_basin_test
            basin = run_basin_test(
                simulator=simulator,
                n_diverse=bt_cfg.get("n_diverse", 50),
                T=bt_cfg.get("T", 300),
                tail_steps=bt_cfg.get("tail_steps", 50),
                dominant_thresh=bt_cfg.get("dominant_thresh", 0.75),
                seed=cfg["data_generation"].get("seed", 42),
                device=simulator.device if hasattr(simulator, "device") else "cpu",
                output_dir=dyn_dir,
            )
            results["basin_test"] = basin
            logger.info(
                "  Basin test: single_attractor=%s, dominant=%.1f%%",
                basin.get("single_attractor", "?"),
                basin.get("pooled_dominant_fraction", 0.0) * 100,
            )
        except Exception as e:
            logger.warning("  Basin test failed: %s", e)


def run_phase4_validation(cfg: dict, results: Dict[str, Any],
                          output_dir: Path) -> None:
    """Phase 4: Statistical validation of dynamics characterisation."""
    val_cfg = cfg.get("validation", {})
    trajs = results.get("trajectories")
    if trajs is None:
        logger.warning("Phase 4 requires trajectories, skipping.")
        return
    _step(4, "Statistical Validation")

    val_dir = output_dir / "validation"
    val_dir.mkdir(exist_ok=True)
    seed = cfg["data_generation"].get("seed", 42)

    # 4a: Surrogate test
    st_cfg = val_cfg.get("surrogate_test", {})
    if st_cfg.get("enabled", True):
        try:
            from analysis.surrogate_test import run_surrogate_test
            surr = run_surrogate_test(
                trajectories=trajs,
                n_surrogates=st_cfg.get("n_surrogates", 19),
                surrogate_types=st_cfg.get("types"),
                n_traj_sample=st_cfg.get("n_traj_sample", 5),
                seed=seed,
                output_dir=val_dir,
            )
            results["surrogate_test"] = surr
            logger.info(
                "  Surrogate test: real_LLE=%.5f, nonlinear=%s",
                surr.get("real_lle", float("nan")),
                surr.get("is_nonlinear", "?"),
            )
        except Exception as e:
            logger.warning("  Surrogate test failed: %s", e)

    # 4b: Random comparison
    rc_cfg = val_cfg.get("random_comparison", {})
    if rc_cfg.get("enabled", True):
        try:
            from analysis.random_comparison import run_random_model_comparison
            comp = run_random_model_comparison(
                trajectories=trajs,
                attractor_results=results.get("attractor"),
                lyapunov_results=results.get("lyapunov"),
                response_matrix=results.get("response_matrix"),
                spectral_radii=rc_cfg.get("spectral_radii", [0.9, 1.5, 2.0]),
                n_seeds=rc_cfg.get("n_seeds", 5),
                seed=seed,
                output_dir=val_dir,
            )
            results["random_comparison"] = comp
            logger.info("  Random comparison: done (see report).")
        except Exception as e:
            logger.warning("  Random comparison failed: %s", e)

    # 4c: Embedding dimension
    ed_cfg = val_cfg.get("embedding_dimension", {})
    if ed_cfg.get("enabled", True):
        try:
            from analysis.embedding_dimension import run_embedding_dimension_analysis
            emb = run_embedding_dimension_analysis(
                trajectories=trajs,
                fnn_max_dim=ed_cfg.get("fnn_max_dim", 8),
                corr_dim=ed_cfg.get("corr_dim", True),
                check_leakage=ed_cfg.get("check_leakage", True),
                train_fraction=ed_cfg.get("train_fraction", 0.7),
                output_dir=val_dir,
            )
            results["embedding_dimension"] = emb
            logger.info(
                "  Embedding: FNN_dim=%s, D₂=%.2f",
                emb.get("fnn_min_dim", "?"),
                emb.get("correlation_dimension", float("nan")),
            )
        except Exception as e:
            logger.warning("  Embedding dimension failed: %s", e)

    # 4d: Structural perturbation
    if val_cfg.get("perturbation", {}).get("enabled", False):
        R = results.get("response_matrix")
        if R is not None:
            try:
                from spectral_dynamics.e4_structural_perturbation import run_structural_perturbation
                pert = run_structural_perturbation(R, output_dir=val_dir)
                results["perturbation"] = pert
                logger.info("  Structural perturbation: done (see report).")
            except Exception as e:
                logger.warning("  Structural perturbation failed: %s", e)

    # 4e: Local intrinsic dimension (TwoNN, P1-1)
    id_cfg = val_cfg.get("intrinsic_dimension", {})
    if id_cfg.get("enabled", True):
        try:
            from analysis.intrinsic_dimension import run_intrinsic_dimension
            id_result = run_intrinsic_dimension(
                trajectories=trajs,
                k_traj=id_cfg.get("k_traj", 10),
                burnin_fraction=id_cfg.get("burnin_fraction", 0.10),
                max_points=id_cfg.get("max_points", 2000),
                seed=id_cfg.get("seed", seed),
                output_dir=val_dir,
            )
            results["intrinsic_dimension"] = id_result
            logger.info(
                "  TwoNN intrinsic dim: d=%.2f±%.2f (n=%d/%d)",
                id_result.get("d_mean", float("nan")),
                id_result.get("d_std", float("nan")),
                id_result.get("n_valid", 0),
                id_result.get("n_total", 0),
            )
        except Exception as e:
            logger.warning("  Intrinsic dimension (TwoNN) failed: %s", e)

    # ── Network perturbation experiments (Experiments 4, 5, 6, 7) ─────────────
    # These operate on the connectivity matrix W (response matrix or FC).
    # No additional GNN model calls are needed.
    W_pert = _get_square_connectivity(results, trajs)

    # 4f: Hub Perturbation (Experiment 5) — OPTIONAL
    hp_cfg = val_cfg.get("hub_perturbation", {})
    if hp_cfg.get("enabled", False) and W_pert is not None:
        try:
            from analysis.network_perturbation import run_hub_perturbation
            hp = run_hub_perturbation(
                W=W_pert,
                top_k_list=hp_cfg.get("top_k_list", [5, 10]),
                output_dir=val_dir,
            )
            results["hub_perturbation"] = hp
            logger.info("  Hub perturbation: %s", hp.get("judgment", "?"))
        except Exception as e:
            logger.warning("  Hub perturbation failed: %s", e)
    elif hp_cfg.get("enabled", False) and W_pert is None:
        logger.warning("  Hub perturbation skipped: no square connectivity matrix available.")

    # 4g: Weight Randomisation (Experiment 6) — OPTIONAL
    wr_cfg = val_cfg.get("weight_randomisation", {})
    if wr_cfg.get("enabled", False) and W_pert is not None:
        try:
            from analysis.network_perturbation import run_weight_randomisation
            wr = run_weight_randomisation(
                W=W_pert,
                n_shuffles=wr_cfg.get("n_shuffles", 5),
                seed=seed,
                output_dir=val_dir,
            )
            results["weight_randomisation"] = wr
            logger.info("  Weight randomisation: %s", wr.get("judgment", "?"))
        except Exception as e:
            logger.warning("  Weight randomisation failed: %s", e)
    elif wr_cfg.get("enabled", False) and W_pert is None:
        logger.warning("  Weight randomisation skipped: no square connectivity matrix.")

    # 4h: Subnetwork Scaling (Experiment 7) — OPTIONAL
    ss_cfg = val_cfg.get("subnetwork_scaling", {})
    if ss_cfg.get("enabled", False) and W_pert is not None:
        try:
            from analysis.network_perturbation import run_subnetwork_scaling
            N_full = W_pert.shape[0]
            default_scales = [
                s for s in [120, 160, 200] if s < N_full
            ] or [max(5, N_full // 2)]
            ss = run_subnetwork_scaling(
                W=W_pert,
                scales=ss_cfg.get("scales", default_scales),
                n_subnetworks=ss_cfg.get("n_subnetworks", 10),
                seed=seed,
                output_dir=val_dir,
            )
            results["subnetwork_scaling"] = ss
            logger.info("  Subnetwork scaling: %s", ss.get("judgment", "?"))
        except Exception as e:
            logger.warning("  Subnetwork scaling failed: %s", e)
    elif ss_cfg.get("enabled", False) and W_pert is None:
        logger.warning("  Subnetwork scaling skipped: no square connectivity matrix.")

    # 4i: Structure-Preserving Random Network (Experiment 4) — OPTIONAL
    # Complements the existing random comparison (4b) by preserving the
    # empirical edge weight distribution while rewiring connections.
    sp_cfg = val_cfg.get("structure_preserving_random", {})
    if sp_cfg.get("enabled", False) and W_pert is not None:
        try:
            from analysis.network_perturbation import run_structure_preserving_random
            spr = run_structure_preserving_random(
                W=W_pert,
                n_random=sp_cfg.get("n_random", 5),
                seed=seed,
                output_dir=val_dir,
            )
            results["structure_preserving_random"] = spr
            logger.info("  Structure-preserving random: %s", spr.get("judgment", "?"))
        except Exception as e:
            logger.warning("  Structure-preserving random network failed: %s", e)
    elif sp_cfg.get("enabled", False) and W_pert is None:
        logger.warning("  Structure-preserving random skipped: no square matrix.")

    # 4j: Graph-structure random comparison (TASK 1) — OPTIONAL
    # Compares brain graph vs degree-preserving and fully-random baselines using
    # analytical spectral metrics + GNN-derived LLE/D2 from Phase 3 results.
    rg_cfg = val_cfg.get("graph_structure_comparison", {})
    if rg_cfg.get("enabled", False) and W_pert is not None:
        try:
            from analysis.random_comparison import run_graph_structure_comparison
            rg = run_graph_structure_comparison(
                W=W_pert,
                trajectories=trajs,
                lyapunov_results=results.get("lyapunov"),
                attractor_results=results.get("attractor"),
                n_random=rg_cfg.get("n_random", 5),
                seed=seed,
                output_dir=val_dir,
            )
            results["graph_structure_comparison"] = rg
            brain_pr = rg.get("brain_graph", {}).get("participation_ratio", float("nan"))
            rand_pr = rg.get("fully_random", {}).get("participation_ratio", float("nan"))
            logger.info(
                "  Graph structure comparison: brain PR=%.3f, random PR=%.3f",
                brain_pr, rand_pr,
            )
        except Exception as e:
            logger.warning("  Graph structure comparison failed: %s", e)
    elif rg_cfg.get("enabled", False) and W_pert is None:
        logger.warning("  Graph structure comparison skipped: no square connectivity matrix.")

    # 4k: Input dimension control (TASK 2) — OPTIONAL
    # Tests whether low-dimensional dynamics arise from network structure vs input drive.
    # Operates on existing GNN trajectories — no additional model calls.
    idc_cfg = val_cfg.get("input_dimension_control", {})
    if idc_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.input_dimension_control import run_input_dimension_control
            idc = run_input_dimension_control(
                trajectories=trajs,
                noise_sigma=idc_cfg.get("noise_sigma", 0.5),
                low_dim_k=idc_cfg.get("low_dim_k", 3),
                seed=seed,
                output_dir=val_dir,
            )
            results["input_dimension_control"] = idc
            no_input_lle = idc.get("no_input", {}).get("lle", float("nan"))
            hd_lle = idc.get("high_dim_noise", {}).get("lle", float("nan"))
            logger.info(
                "  Input dim control: no-input LLE=%.4f, high-dim-noise LLE=%.4f",
                no_input_lle, hd_lle,
            )
        except Exception as e:
            logger.warning("  Input dimension control failed: %s", e)

    # 4l: Node ablation (TASK 4) — OPTIONAL
    # Ranks nodes by how much the dynamical manifold depends on each one.
    # Operates on existing GNN trajectories — no additional model calls.
    na_cfg = val_cfg.get("node_ablation", {})
    if na_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.node_ablation import run_node_ablation
            na = run_node_ablation(
                trajectories=trajs,
                n_top_variance=na_cfg.get("n_top_variance", 50),
                n_random=na_cfg.get("n_random_sample", 50),
                seed=seed,
                output_dir=val_dir,
            )
            results["node_ablation"] = na
            top5 = na.get("top_nodes", [])[:5]
            logger.info("  Node ablation: top-5 nodes by |ΔLLE|: %s", top5)
        except Exception as e:
            logger.warning("  Node ablation failed: %s", e)

    # 4m: Predictive dimension (TASK 7) — OPTIONAL
    # Finds the optimal VAR(1) predictor dimension (MSE/AIC/BIC vs PCA dim m).
    pd_val_cfg = val_cfg.get("predictive_dimension", {})
    if pd_val_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.predictive_dimension import run_predictive_dimension
            pdim = run_predictive_dimension(
                trajectories=trajs,
                max_dim=pd_val_cfg.get("max_dim", 10),
                train_fraction=pd_val_cfg.get("train_fraction", 0.70),
                seed=seed,
                output_dir=val_dir,
            )
            results["predictive_dimension"] = pdim
            logger.info(
                "  Predictive dimension: optimal_m (AIC)=%d, (BIC)=%d",
                pdim.get("optimal_m_aic", -1),
                pdim.get("optimal_m_bic", -1),
            )
        except Exception as e:
            logger.warning("  Predictive dimension failed: %s", e)


def run_phase5_advanced(cfg: dict, results: Dict[str, Any],
                        simulator, output_dir: Path,
                        modality: str = "fmri") -> None:
    """Phase 5: Optional advanced analyses (expensive)."""
    adv_cfg = cfg.get("advanced", {})
    trajs = results.get("trajectories")
    _step(5, "Advanced Analysis (optional)")

    adv_dir = output_dir / "advanced"
    adv_dir.mkdir(exist_ok=True)

    # 5a: Virtual stimulation
    stim_cfg = adv_cfg.get("stimulation", {})
    if stim_cfg.get("enabled", True) and trajs is not None:
        try:
            from experiments.virtual_stimulation import run_virtual_stimulation
            targets = [n for n in stim_cfg.get("target_nodes", [0, 100])
                       if n < simulator.n_regions]
            if not targets:
                targets = [0]
            # For joint mode: log which modality each target node falls into
            if modality == "joint" and hasattr(simulator, "n_fmri_regions"):
                for tn in targets:
                    _tmod = "fmri" if tn < simulator.n_fmri_regions else "eeg"
                    _tch = (tn if tn < simulator.n_fmri_regions
                            else tn - simulator.n_fmri_regions)
                    logger.info(
                        "  target_node=%d → %s node %d (joint modality index mapping)",
                        tn, _tmod, _tch,
                    )
            stim = run_virtual_stimulation(
                simulator=simulator,
                target_nodes=targets,
                amplitude=stim_cfg.get("amplitude", 0.5),
                frequency=stim_cfg.get("frequency", 10.0),
                stim_steps=stim_cfg.get("duration", 30),
                patterns=stim_cfg.get("patterns", ["sine"]),
                output_dir=adv_dir,
            )
            results["stimulation"] = stim
            logger.info("  Virtual stimulation: %d patterns × %d nodes.",
                        len(stim), len(targets))
        except Exception as e:
            logger.warning("  Virtual stimulation failed: %s", e)

    # 5b: Energy constraint
    if adv_cfg.get("energy_constraint", {}).get("enabled", False) and trajs is not None:
        try:
            from experiments.energy_constraint import run_energy_budget_analysis
            ec = run_energy_budget_analysis(
                trajectories=trajs,
                state_bounds=simulator.state_bounds,
                output_dir=adv_dir,
            )
            results["energy_budget"] = ec
            logger.info(
                "  Energy budget: E*=%.4f ± %.4f",
                ec.get("E_mean", 0), ec.get("E_std", 0),
            )
        except Exception as e:
            logger.warning("  Energy constraint failed: %s", e)

    # 5c: Phase diagram
    if adv_cfg.get("phase_diagram", {}).get("enabled", False):
        try:
            from spectral_dynamics.e5_phase_diagram import run_phase_diagram
            pd_cfg = adv_cfg.get("phase_diagram", {})
            # run_phase_diagram requires a square (N×N) matrix.
            # Prefer: DMD operator (already N×N) → FC from trajectories.
            # The response matrix R is often non-square (n_nodes × N_regions).
            W_pd = _get_square_connectivity(results, trajs)
            if W_pd is not None:
                # Compute LLE reference from GNN trajectories generated by Phase 1
                lle_ref = None
                _trajs = results.get("trajectories")
                if _trajs is not None:
                    try:
                        from analysis.lyapunov import rosenstein_lyapunov
                        import numpy as _np
                        _lles = [
                            rosenstein_lyapunov(_trajs[i])[0]
                            for i in range(min(10, len(_trajs)))
                        ]
                        _valid = [v for v in _lles if _np.isfinite(v)]
                        if _valid:
                            lle_ref = float(_np.median(_valid))
                    except Exception:
                        pass
                phase = run_phase_diagram(
                    W_pd, output_dir=adv_dir,
                    g_min=pd_cfg.get("g_min", 0.1),
                    g_max=pd_cfg.get("g_max", 3.0),
                    g_step=pd_cfg.get("g_step", 0.2),
                    lle_reference=lle_ref,
                )
                results["phase_diagram"] = phase
                logger.info("  Phase diagram: done.")
            else:
                logger.warning("  Phase diagram skipped: no square connectivity matrix available.")
        except Exception as e:
            logger.warning("  Phase diagram failed: %s", e)

    # 5d: Controllability
    if adv_cfg.get("controllability", {}).get("enabled", False):
        try:
            from analysis.controllability import run_controllability_analysis
            # run_controllability_analysis requires a square (N×N) matrix.
            # Prefer: DMD operator → FC from trajectories.
            W_ctrl = _get_square_connectivity(results, trajs)
            if W_ctrl is not None:
                ctrl = run_controllability_analysis(
                    response_matrix=W_ctrl, output_dir=adv_dir,
                )
                results["controllability"] = ctrl
                logger.info("  Controllability: done.")
            else:
                logger.warning("  Controllability skipped: no square connectivity matrix available.")
        except Exception as e:
            logger.warning("  Controllability analysis failed: %s", e)

    # 5e: Information flow
    if_cfg = adv_cfg.get("information_flow", {})
    if if_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.information_flow import run_information_flow_analysis
            iflow = run_information_flow_analysis(
                trajectories=trajs,
                n_source_regions=if_cfg.get("n_source_regions", 20),
                n_target_regions=if_cfg.get("n_target_regions", 20),
                output_dir=adv_dir,
            )
            results["information_flow"] = iflow
            logger.info("  Information flow: done.")
        except Exception as e:
            logger.warning("  Information flow failed: %s", e)

    # 5f: Critical slowing down (P0-3: enabled by default as H3 evidence)
    if adv_cfg.get("critical_slowing_down", {}).get("enabled", True) and trajs is not None:
        try:
            from analysis.critical_slowing_down import run_critical_slowing_down_analysis
            csd_cfg = adv_cfg.get("critical_slowing_down", {})
            csd = run_critical_slowing_down_analysis(
                trajectories=trajs, output_dir=adv_dir,
            )
            results["critical_slowing_down"] = csd
            logger.info(
                "  Critical slowing down: ar1_tau=%.3f (p=%.3f), var_tau=%.3f (p=%.3f)",
                csd.get("ar1_trend_tau", float("nan")),
                csd.get("ar1_trend_p", float("nan")),
                csd.get("var_trend_tau", float("nan")),
                csd.get("var_trend_p", float("nan")),
            )
        except Exception as e:
            logger.warning("  Critical slowing down failed: %s", e)

    # 5g: Dynamic lesion experiment (TASK 5) — OPTIONAL
    # Requires prior node_ablation (4l) results to select top nodes.
    # Operates on existing GNN trajectories — no additional model calls.
    lesion_cfg = adv_cfg.get("lesion_dynamics", {})
    if lesion_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.node_ablation import run_lesion_dynamics
            na_results = results.get("node_ablation", {})
            top_nodes = na_results.get("top_nodes", list(range(10)))
            lesion = run_lesion_dynamics(
                trajectories=trajs,
                top_nodes=top_nodes,
                n_control=lesion_cfg.get("n_control", 10),
                n_lesion_nodes=lesion_cfg.get("n_lesion_nodes", 10),
                lesion_step=lesion_cfg.get("lesion_step", 500),
                seed=cfg["data_generation"].get("seed", 42),
                output_dir=adv_dir,
            )
            results["lesion_dynamics"] = lesion
            n_rows = len(lesion.get("lesion_results", []))
            logger.info("  Lesion dynamics: %d nodes analysed (step=%d).",
                        n_rows, lesion.get("lesion_step", 500))
        except Exception as e:
            logger.warning("  Lesion dynamics failed: %s", e)

    # 5h: Granger causality (TASK 6) — OPTIONAL
    # Pairwise Granger F-statistic matrix on GNN trajectories.
    # Complements the Transfer-Entropy analysis in 5e.
    gc_cfg = adv_cfg.get("granger_causality", {})
    if gc_cfg.get("enabled", False) and trajs is not None:
        try:
            from analysis.granger_causality import run_granger_analysis
            # Node-importance CSV for overlap analysis
            _na_dir = adv_dir.parent / "validation"
            _ni_csv = _na_dir / "node_importance.csv"
            gc = run_granger_analysis(
                trajectories=trajs,
                max_lag=gc_cfg.get("max_lag", 1),
                n_src=gc_cfg.get("n_src", None),
                n_tgt=gc_cfg.get("n_tgt", None),
                top_k=gc_cfg.get("top_k", 20),
                node_importance_csv=_ni_csv if _ni_csv.exists() else None,
                output_dir=adv_dir,
            )
            results["granger_causality"] = gc
            top_src = gc.get("top_sources", [])[:5]
            logger.info("  Granger causality: top-5 sources: %s", top_src)
        except Exception as e:
            logger.warning("  Granger causality failed: %s", e)


def run_phase6_synthesis(cfg: dict, results: Dict[str, Any],
                         output_dir: Path) -> None:
    """Phase 6: Synthesis — consistency checks & hypothesis evaluation."""
    _step(6, "Synthesis & Report")

    # ── P2-1: Run metadata ────────────────────────────────────────────────────
    import hashlib
    dg = cfg.get("data_generation", {})
    sim_cfg = cfg.get("simulator", {})
    try:
        cfg_hash = hashlib.sha256(
            json.dumps(cfg, sort_keys=True, default=str).encode()
        ).hexdigest()[:12]
    except Exception:
        cfg_hash = "unknown"

    report: Dict[str, Any] = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "metadata": {
            "seed": dg.get("seed", 42),
            "n_init": dg.get("n_init"),
            "steps": dg.get("steps"),
            "modality": sim_cfg.get("modality", "fmri"),
            "cfg_hash": cfg_hash,
            "output_dir": str(output_dir),
        },
        # P0-4: evidence grading
        "connectivity_source": results.get("connectivity_source") or "unknown",
        "evidence_grade": results.get("evidence_grade") or "unknown",
    }

    # ── P0-2: DMD interpretation note ─────────────────────────────────────────
    report["dmd_interpretation_note"] = (
        "DMD operator A is a least-squares linear approximation of the "
        "dynamics on the attractor sample; it is NOT equivalent to the "
        "point-wise Jacobian. Eigenvalues reflect linearised structure "
        "(Koopman finite-dimensional projection). For chaos assessment, "
        "use Rosenstein LLE as the primary indicator."
    )

    # ── Consistency check: Rosenstein LLE vs DMD spectral radius ──────────────
    lya = results.get("lyapunov", {})
    dmd = results.get("dmd_spectrum", {})
    mean_lle = lya.get("mean_lyapunov")
    rho_dmd = dmd.get("spectral_radius")

    if mean_lle is not None and rho_dmd is not None:
        # Rosenstein LLE > 0 implies chaos; DMD ρ ≈ 1.0 implies near-criticality.
        # Consistency rule: only flag contradiction when signs clearly disagree:
        #   ρ < 1 (subcritical) + λ > 0 (chaotic), or ρ > 1 (supercritical) + λ < 0.
        # Near-zero/near-critical cases are treated as consistent with either
        # regime — DMD is linearised and may not capture nonlinear effects, so
        # a marginal mismatch is expected and not flagged.
        lle_sign = "positive" if mean_lle > 0.01 else (
            "negative" if mean_lle < -0.01 else "near_zero"
        )
        rho_sign = "supercritical" if rho_dmd > 1.01 else (
            "subcritical" if rho_dmd < 0.99 else "near_critical"
        )
        consistent = not (
            (lle_sign == "positive" and rho_sign == "subcritical") or
            (lle_sign == "negative" and rho_sign == "supercritical")
        )
        report["consistency"] = {
            "rosenstein_lle": float(mean_lle),
            "dmd_spectral_radius": float(rho_dmd),
            "lle_sign": lle_sign,
            "rho_sign": rho_sign,
            "consistent": consistent,
        }
        # Nonlinearity index: how much the nonlinear LLE deviates from the
        # linearised prediction.  λ_DMD_max = ln(ρ) gives the linearised
        # largest Lyapunov exponent.  A large gap indicates strong nonlinearity.
        lambda_dmd_max = float(np.log(max(rho_dmd, 1e-30)))
        if abs(mean_lle) > 1e-6:
            nonlinearity_index = abs(mean_lle - lambda_dmd_max) / abs(mean_lle)
        else:
            nonlinearity_index = abs(lambda_dmd_max)
        report["consistency"]["lambda_dmd_max"] = lambda_dmd_max
        report["consistency"]["nonlinearity_index"] = round(nonlinearity_index, 4)
        # P0-2: flag high nonlinearity → DMD conclusions are reference only
        if nonlinearity_index > 1.0:
            report["consistency"]["dmd_note"] = (
                "nonlinearity_index > 1.0: DMD linearised approximation "
                "deviates strongly from nonlinear dynamics. "
                "DMD spectral conclusions are for reference only."
            )
        if consistent:
            logger.info(
                "  ✓ Consistency: Rosenstein λ=%.5f (%s) ↔ DMD ρ=%.4f (%s)"
                "  [nonlinearity=%.2f]",
                mean_lle, lle_sign, rho_dmd, rho_sign, nonlinearity_index,
            )
        else:
            logger.warning(
                "  ⚠ Inconsistency: Rosenstein λ=%.5f (%s) vs DMD ρ=%.4f (%s)"
                "  [nonlinearity=%.2f]\n"
                "    DMD captures linearised dynamics only; "
                "nonlinear effects dominate.",
                mean_lle, lle_sign, rho_dmd, rho_sign, nonlinearity_index,
            )

    # ── Surrogate validation cross-check ──────────────────────────────────────
    surr = results.get("surrogate_test", {})
    if surr.get("is_nonlinear") is not None and mean_lle is not None:
        report["nonlinearity"] = {
            "is_nonlinear": surr["is_nonlinear"],
            "rosenstein_lle": float(mean_lle),
            "surrogate_lle_mean": surr.get("surrogate_lle_mean"),
            "z_score": surr.get("z_score"),
        }
        if surr["is_nonlinear"]:
            logger.info("  ✓ Surrogate test confirms nonlinear dynamics.")
        else:
            logger.info("  ○ Dynamics may be explainable by linear process.")

    # ── Hypothesis evaluation ─────────────────────────────────────────────────
    hypotheses = _evaluate_hypotheses(results)
    report["hypotheses"] = hypotheses
    for h_id, h_info in hypotheses.items():
        logger.info(
            "  %s: %s — %s",
            h_id, h_info["verdict"], h_info["summary"],
        )

    # ── P0-4: Confidence flags (auto-downgrade when evidence_grade == C) ──────
    evidence_grade = results.get("evidence_grade") or "unknown"
    confidence_flags: Dict[str, str] = {}
    for h_id, h_info in hypotheses.items():
        verdict = h_info.get("verdict", "INSUFFICIENT_DATA")
        # Downgrade H1 (structural) when only FC is available (no causal interpretation)
        if evidence_grade == "C" and h_id == "H1":
            confidence_flags[h_id] = "LOW (FC only — no causal interpretation)"
        elif verdict == "SUPPORTED":
            confidence_flags[h_id] = "HIGH" if evidence_grade == "A" else (
                "MEDIUM" if evidence_grade == "B" else "LOW"
            )
        elif verdict == "NOT_SUPPORTED":
            confidence_flags[h_id] = "HIGH" if evidence_grade == "A" else "MEDIUM"
        else:
            confidence_flags[h_id] = "INSUFFICIENT_DATA"
    report["confidence_flags"] = confidence_flags
    logger.info("  Confidence flags: %s", confidence_flags)

    # ── Regime summary ────────────────────────────────────────────────────────
    regime = lya.get("chaos_regime", {}).get("regime", "unknown")
    report["regime"] = regime
    report["lle"] = float(mean_lle) if mean_lle is not None else None
    report["dmd_rho"] = float(rho_dmd) if rho_dmd is not None else None

    # ── 9-question verification table (Experiments 1–8) ──────────────────────
    # Synthesises results from the 8 validation experiments into a concise
    # verification table answering the three core questions.
    report["validation_table"] = _build_validation_table(results)
    logger.info("  Validation table:")
    for row in report["validation_table"]:
        status = "✓" if row.get("verified") else ("○" if row.get("verified") is None else "✗")
        logger.info("    %s Q%d: %s — %s",
                    status, row["question_id"], row["question_short"], row["verdict"])

    # Save report
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("  → Report saved: %s", report_path)

    results["report"] = report


def _build_validation_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build the 9-question verification table from all experiment results.

    Returns a list of dicts, one per question, with keys:
        question_id    int
        question_short str
        experiment     str  (which experiment answered this)
        data_source    str  (which results key)
        verdict        str
        evidence       str
        verified       bool or None (None = not tested)
    """

    def _q(qid, short, exp, src, verdict, evidence, verified):
        return {
            "question_id": qid,
            "question_short": short,
            "experiment": exp,
            "data_source": src,
            "verdict": verdict,
            "evidence": evidence,
            "verified": verified,
        }

    rows = []

    # Q1: Single strange attractor? (Exp 1: Lyapunov Spectrum)
    lsp = results.get("lyapunov_spectrum", {})
    sa = lsp.get("strange_attractor")
    lya = results.get("lyapunov", {})
    lle = lya.get("mean_lyapunov")
    if sa is not None:
        verified_q1 = sa in ("likely", "possible")
        verdict_q1 = f"strange_attractor={sa}"
        ev_q1 = lsp.get("assessment", {}).get("evidence", [])
        ev_q1_str = "; ".join(ev_q1[:3]) if ev_q1 else (
            f"K-Y={lsp.get('ky_dimension', '?'):.2f}, n_pos={lsp.get('n_positive', '?')}"
            if isinstance(lsp.get("ky_dimension"), float) else "see lyapunov_spectrum_report.json"
        )
    else:
        verified_q1 = None
        verdict_q1 = "not_tested (Phase 3e DMD disabled or failed)"
        ev_q1_str = "Enable dmd_spectrum in config to compute Lyapunov spectrum."
    rows.append(_q(1, "single strange attractor", "Exp1 (Lyapunov Spectrum)",
                   "lyapunov_spectrum", verdict_q1, ev_q1_str, verified_q1))

    # Q2: Low-dimensional chaos stable? (Exp 3: Node Contribution + Exp 7: Subnetwork)
    nc = results.get("node_contribution", {})
    ss = results.get("subnetwork_scaling", {})
    has_core = nc.get("has_core")
    core_pct = nc.get("top10_contribution_pct", float("nan"))
    ss_stab = ss.get("stability_index")
    if has_core is not None or ss_stab is not None:
        verified_q2 = (has_core is True) or (ss_stab is not None and ss_stab < 0.15)
        ev_parts = []
        if has_core is not None:
            ev_parts.append(f"top10_nodes={core_pct:.1f}%")
        if ss_stab is not None:
            ev_parts.append(f"scale_CV={ss_stab:.1%}")
        verdict_q2 = "low-dim chaos stable" if verified_q2 else "insufficient evidence"
        ev_q2_str = ", ".join(ev_parts) or "no evidence"
    else:
        verified_q2 = None
        verdict_q2 = "not_tested (enable node_contribution and/or subnetwork_scaling)"
        ev_q2_str = ""
    rows.append(_q(2, "low-dimensional chaos stable", "Exp3 (Node Contribution) + Exp7 (Subnetwork)",
                   "node_contribution, subnetwork_scaling", verdict_q2, ev_q2_str, verified_q2))

    # Q3: Dynamical core nodes? (Exp 3: Node Contribution)
    if has_core is not None:
        core_size = nc.get("dynamical_core_size", 0)
        N_reg = results.get("lyapunov_spectrum", {}).get("spectrum", np.array([])).shape[0]
        verdict_q3 = (
            f"core_size={core_size} nodes explain ≥50% (top10={core_pct:.1f}%)"
            if has_core else f"no strong core (top10={core_pct:.1f}% ≤ 50%)"
        )
        ev_q3_str = nc.get("top_nodes", [])[:5]
        ev_q3_str = f"top nodes: {ev_q3_str}"
    else:
        has_core = None
        verdict_q3 = "not_tested (enable node_contribution)"
        ev_q3_str = ""
    rows.append(_q(3, "dynamical core nodes exist", "Exp3 (Node Contribution)",
                   "node_contribution", verdict_q3, ev_q3_str, has_core))

    # Q4: Training structure determines dynamics? (Exp 4: Structure-Preserving Random)
    spr = results.get("structure_preserving_random", {})
    if spr:
        delta_rho = spr.get("delta_rho")
        verified_q4 = bool(delta_rho is not None and delta_rho > 0.05)
        verdict_q4 = spr.get("judgment", "?")
        ev_q4_str = f"delta_rho={delta_rho:.4f}" if delta_rho is not None else "no evidence"
    else:
        verified_q4 = None
        verdict_q4 = "not_tested (enable structure_preserving_random)"
        ev_q4_str = ""
    rows.append(_q(4, "training structure determines dynamics", "Exp4 (Structure-Preserving Random)",
                   "structure_preserving_random", verdict_q4, ev_q4_str, verified_q4))

    # Q5: Random networks lose structure? (Phase 4b: random_comparison)
    rc = results.get("random_comparison", {})
    if rc:
        rand_lle = rc.get("random_lle_mean")
        real_lle_val = lle
        if rand_lle is not None and real_lle_val is not None:
            verified_q5 = bool(real_lle_val > rand_lle + 0.01)
            verdict_q5 = (
                f"real LLE={real_lle_val:.4f} > random={rand_lle:.4f} — trained dynamics differ"
                if verified_q5
                else f"real LLE={real_lle_val:.4f} ≈ random={rand_lle:.4f} — similar"
            )
            ev_q5_str = verdict_q5
        else:
            verified_q5 = None
            verdict_q5 = "random comparison run but LLE comparison unavailable"
            ev_q5_str = str(rc)[:200]
    else:
        verified_q5 = None
        verdict_q5 = "not_tested (enable random_comparison)"
        ev_q5_str = ""
    rows.append(_q(5, "random networks lose structure", "Exp4 + Phase4b (random_comparison)",
                   "random_comparison", verdict_q5, ev_q5_str, verified_q5))

    # Q6: Hub nodes control dynamics? (Exp 5: Hub Perturbation)
    hp = results.get("hub_perturbation", {})
    if hp:
        max_shift = max(
            (v for v in hp.get("spectral_shift", {}).values() if v is not None), default=None
        )
        verified_q6 = bool(max_shift is not None and max_shift > 0.10)
        verdict_q6 = hp.get("judgment", "?")
        ev_q6_str = f"max_rho_shift={max_shift:.1%}" if max_shift is not None else "?"
    else:
        verified_q6 = None
        verdict_q6 = "not_tested (enable hub_perturbation)"
        ev_q6_str = ""
    rows.append(_q(6, "hub nodes control manifold", "Exp5 (Hub Perturbation)",
                   "hub_perturbation", verdict_q6, ev_q6_str, verified_q6))

    # Q7: Weight structure matters? (Exp 6: Weight Randomisation)
    wr = results.get("weight_randomisation", {})
    if wr:
        rho_shift = wr.get("rho_shift")
        verified_q7 = bool(rho_shift is not None and rho_shift > 0.10)
        verdict_q7 = wr.get("judgment", "?")
        ev_q7_str = f"rho_shift={rho_shift:.1%}" if rho_shift is not None else "?"
    else:
        verified_q7 = None
        verdict_q7 = "not_tested (enable weight_randomisation)"
        ev_q7_str = ""
    rows.append(_q(7, "weight structure determines dynamics", "Exp6 (Weight Randomisation)",
                   "weight_randomisation", verdict_q7, ev_q7_str, verified_q7))

    # Q8: Scale invariance / subnetwork stability? (Exp 7: Subnetwork Scaling)
    if ss:
        stab = ss.get("stability_index")
        verified_q8 = bool(stab is not None and stab < 0.15)
        verdict_q8 = ss.get("judgment", "?")
        ev_q8_str = f"rho_CV={stab:.1%}" if stab is not None else "?"
    else:
        verified_q8 = None
        verdict_q8 = "not_tested (enable subnetwork_scaling)"
        ev_q8_str = ""
    rows.append(_q(8, "dynamics scale-invariant", "Exp7 (Subnetwork Scaling)",
                   "subnetwork_scaling", verdict_q8, ev_q8_str, verified_q8))

    # Q9: Modalities share dynamical core? (Exp 8: already addressed by --modality both)
    # Check if we have results for both fMRI and EEG modalities (both mode)
    rows.append(_q(
        9, "modalities share dynamical core",
        "Exp8 (--modality both)",
        "fmri_report + eeg_report",
        "compare fmri/eeg/pipeline_report.json files with --modality both",
        "Use --modality both to run pipeline for each modality and compare.",
        None,  # always not_tested in single-modality run
    ))

    return rows


def _evaluate_hypotheses(results: Dict[str, Any]) -> Dict[str, Dict]:
    """Evaluate hypotheses H1–H5 from combined results."""
    H: Dict[str, Dict] = {}

    # H1: Low-rank spectral structure
    spec = results.get("spectral", {})
    pr = spec.get("participation_ratio")
    n_dom = spec.get("n_dominant")
    N = spec.get("n_regions")
    evidence_grade = results.get("evidence_grade") or "unknown"
    if pr is not None and N is not None and N > 0:
        ratio = pr / N
        supported = ratio < 0.3
        h1_note = ""
        if evidence_grade == "C":
            h1_note = (
                " [Grade C: FC only — spectral structure reflects "
                "correlational, not causal/Jacobian, connectivity.]"
            )
        H["H1"] = {
            "name": "Low-rank spectral structure",
            "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "summary": f"PR/N={ratio:.2f} ({'<' if supported else '≥'}0.30), "
                       f"n_dominant={n_dom}{h1_note}",
            "PR": float(pr),
            "PR_N_ratio": float(ratio),
            "evidence_grade": evidence_grade,
        }
    else:
        H["H1"] = {"name": "Low-rank spectral structure",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No spectral data."}

    # H2: Low-dimensional dynamics
    # Evidence: PCA n@90%, correlation dimension D₂ (distribution), linearised K-Y dimension
    pca = results.get("pca", {})
    n90 = pca.get("n_components_90")
    ad = results.get("attractor_dimension", {})
    d2 = ad.get("D2")  # scalar mean (backward-compat)
    d2_mean = ad.get("D2_mean", d2)
    d2_std = ad.get("D2_std")
    d2_fail_rate = ad.get("D2_fail_rate")
    d2_reliable = ad.get("D2_h2_reliable", True)
    ky_lin = ad.get("KY_linearised")
    if n90 is not None and N is not None and N > 0:
        dim_ratio = n90 / N
        supported = dim_ratio < 0.15
        summary_parts = [f"n@90%/N={dim_ratio:.2f} ({'<' if supported else '≥'}0.15)"]
        if d2_mean is not None and np.isfinite(d2_mean):
            d2_str = f"D₂={d2_mean:.2f}"
            if d2_std is not None:
                d2_str += f"±{d2_std:.2f}"
            if d2_fail_rate is not None:
                d2_str += f" (fail={d2_fail_rate:.0%})"
            if not d2_reliable:
                d2_str += " [unreliable: high fail rate]"
            summary_parts.append(d2_str)
        if ky_lin is not None:
            summary_parts.append(f"K-Y_lin={ky_lin:.2f}")
        H["H2"] = {
            "name": "Low-dimensional dynamics",
            "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "summary": ", ".join(summary_parts),
            "n_components_90": n90,
            "dim_ratio": float(dim_ratio),
            "D2": float(d2_mean) if d2_mean is not None and np.isfinite(d2_mean) else None,
            "D2_std": float(d2_std) if d2_std is not None else None,
            "D2_fail_rate": float(d2_fail_rate) if d2_fail_rate is not None else None,
            "D2_reliable": d2_reliable,
            "KY_linearised": float(ky_lin) if ky_lin is not None else None,
        }
    else:
        H["H2"] = {"name": "Low-dimensional dynamics",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No PCA data."}

    # H3: Near-critical dynamics
    # P0-3: Include CSD (critical slowing down) as a parallel evidence stream.
    lya = results.get("lyapunov", {})
    regime = lya.get("chaos_regime", {}).get("regime")
    lle = lya.get("mean_lyapunov")
    csd = results.get("critical_slowing_down", {})
    if regime is not None:
        near_critical = regime in _NEAR_CRITICAL_REGIMES
        summary_parts = [
            f"regime={regime}",
            f"λ={lle:.5f}" if lle is not None else "",
        ]
        h3_entry: Dict[str, Any] = {
            "name": "Near-critical dynamics",
            "verdict": "SUPPORTED" if near_critical else "NOT_SUPPORTED",
            "regime": regime,
            "lle": float(lle) if lle is not None else None,
        }
        # CSD evidence for H3 (P0-3)
        if csd:
            ar1_tau = csd.get("ar1_trend_tau")
            ar1_p = csd.get("ar1_trend_p")
            var_tau = csd.get("var_trend_tau")
            var_p = csd.get("var_trend_p")
            ews_score = csd.get("ews_score_mean")
            csd_parts = []
            if ar1_tau is not None:
                csd_parts.append(f"AR1_tau={ar1_tau:.2f}(p={ar1_p:.3f})")
            if var_tau is not None:
                csd_parts.append(f"Var_tau={var_tau:.2f}(p={var_p:.3f})")
            if ews_score is not None:
                csd_parts.append(f"EWS={ews_score:.2f}")
            if csd_parts:
                summary_parts.append("CSD:[" + ", ".join(csd_parts) + "]")
            h3_entry["csd"] = {
                "ar1_trend_tau": ar1_tau,
                "ar1_trend_p": ar1_p,
                "var_trend_tau": var_tau,
                "var_trend_p": var_p,
                "ews_score_mean": ews_score,
            }
            # Consistent CSD (rising AR1/variance near criticality) strengthens H3
            csd_consistent = (
                ar1_tau is not None and ar1_tau > 0 and
                var_tau is not None and var_tau > 0
            )
            h3_entry["csd_consistent_with_criticality"] = csd_consistent
            if csd_consistent:
                summary_parts.append("CSD_consistent=True")
        else:
            summary_parts.append("CSD: not available")

        h3_entry["summary"] = ", ".join(p for p in summary_parts if p)
        H["H3"] = h3_entry
    else:
        H["H3"] = {"name": "Near-critical dynamics",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No Lyapunov data."}

    # H4: Brain oscillations
    psd = results.get("power_spectrum", {})
    ba = psd.get("band_analysis", {})
    dom_f = ba.get("dominant_freq_hz")
    if dom_f is not None:
        has_oscillation = dom_f > 0.001
        H["H4"] = {
            "name": "Brain oscillations",
            "verdict": "SUPPORTED" if has_oscillation else "NOT_SUPPORTED",
            "summary": f"f_dom={dom_f:.4f} Hz [{ba.get('dominant_freq_band', '?')}]",
            "dominant_freq_hz": float(dom_f),
        }
    else:
        H["H4"] = {"name": "Brain oscillations",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No PSD data."}

    # H5: Energy constraint maintains criticality
    ec = results.get("energy_budget", {})
    if ec:
        H["H5"] = {
            "name": "Energy constraint maintains criticality",
            "verdict": "TESTABLE",
            "summary": f"E*={ec.get('E_mean', 0):.4f}. "
                       "Run with --energy-budget to test.",
        }
    else:
        H["H5"] = {"name": "Energy constraint maintains criticality",
                    "verdict": "NOT_TESTED",
                    "summary": "Enable energy_constraint to test."}

    return H


# ═══════════════════════════════════════════════════════════════════════════════
#  Per-modality phase runner helper
# ═══════════════════════════════════════════════════════════════════════════════

def _run_phases_for_modality(
    cfg: dict,
    simulator,
    modality: str,
    output_dir: Path,
    device: str,
) -> Dict[str, Any]:
    """
    Run all six analysis phases for a single modality.

    This helper is called by :func:`run_pipeline` for each requested modality.
    It accepts an already-constructed *simulator* so that the expensive model /
    graph loading is done exactly once, even when ``modality='both'`` triggers
    two sequential runs.

    Args:
        cfg:        Merged configuration dictionary.
        simulator:  ``BrainDynamicsSimulator`` (or energy-constrained wrapper)
                    already configured for *modality*.
        modality:   ``"fmri"``, ``"eeg"``, or ``"joint"``.
        output_dir: Per-modality output directory (created if absent).
        device:     Compute device string.

    Returns:
        results dict containing all computed artefacts for this modality.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    results.update(run_phase1_data(cfg, simulator, output_dir, device))
    run_phase2_structure(cfg, results, output_dir)
    run_phase3_dynamics(cfg, results, simulator, output_dir, modality=modality)
    run_phase4_validation(cfg, results, output_dir)
    run_phase5_advanced(cfg, results, simulator, output_dir, modality=modality)
    run_phase6_synthesis(cfg, results, output_dir)

    if cfg["output"].get("save_plots"):
        _save_summary_plots(results, output_dir, simulator)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: dict) -> Dict[str, Any]:
    """
    Run the complete dynamics analysis pipeline.

    Loads a trained TwinBrainDigitalTwin model and drives the full analysis
    workflow.  The ``simulator.modality`` config key controls which brain-data
    stream(s) are analysed:

    * ``"fmri"``  — analyse the fMRI BOLD stream only.
    * ``"eeg"``   — analyse the EEG stream only.
    * ``"both"``  — run all phases independently for each modality available
                    in the graph cache.  Results are stored under
                    ``output_dir/fmri/`` and ``output_dir/eeg/`` and returned
                    as ``{"fmri": {...}, "eeg": {...}}``.
    * ``"joint"`` — single pipeline run using a **combined fMRI+EEG** state
                    vector.  ``predict_future()`` is called once; each
                    modality's predictions are z-score normalised per channel,
                    then concatenated into a joint state of dimension
                    N_fmri + N_eeg.  Requires both node types in the graph
                    cache.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        * Single-modality / joint: flat results dict.
        * ``modality="both"``:  ``{"fmri": {...}, "eeg": {...}}``.

    Raises:
        ValueError: If model_path or graph_path is not specified, or if the
                    requested modality is absent from the graph cache.
    """
    from loader.load_model import load_trained_model, load_graph_for_inference
    from simulator.brain_dynamics_simulator import BrainDynamicsSimulator
    import torch

    t0 = time.time()
    output_dir = Path(cfg["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Resolve device ────────────────────────────────────────────────────────
    device_cfg = cfg["model"].get("device", "auto")
    device = (
        "cuda" if device_cfg == "auto" and torch.cuda.is_available()
        else "cpu" if device_cfg == "auto" else device_cfg
    )
    logger.info("Device: %s", device)

    # ── Load model & graph ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Loading model and graph cache...")
    model_path = cfg["model"]["path"]
    graph_path = cfg["model"]["graph_path"]
    if not model_path or not graph_path:
        raise ValueError("model.path and model.graph_path are required.")

    twin = load_trained_model(
        checkpoint_path=model_path, device=device,
        config_path=cfg["model"].get("config_path"),
    )
    k_cross_modal = cfg["model"].get("k_cross_modal", 5)
    base_graph = load_graph_for_inference(
        graph_path=graph_path, device=device,
        k_cross_modal=k_cross_modal,
    )

    # ── Determine which modalities to run ─────────────────────────────────────
    modality_cfg = cfg["simulator"].get("modality", "fmri")
    available_modalities = list(base_graph.node_types)

    if modality_cfg == "joint":
        # Joint mode requires BOTH fmri and eeg in the graph cache.
        if "fmri" not in available_modalities or "eeg" not in available_modalities:
            raise ValueError(
                f"modality='joint' requires the graph cache to contain both "
                f"'fmri' and 'eeg' nodes.\n"
                f"Graph cache node types: {available_modalities}\n"
                "Hint: joint mode uses a single predict_future() call that "
                "concatenates z-score-normalised fMRI and EEG predictions into "
                "one joint state vector for unified dynamics analysis."
            )
        logger.info(
            "modality='joint': single model call processing fMRI(%d) + EEG(%d) "
            "→ concatenated z-score state vector, single dynamics metric set.",
            base_graph["fmri"].x.shape[0],
            base_graph["eeg"].x.shape[0],
        )
        modalities_to_run = ["joint"]

    elif modality_cfg == "both":
        modalities_to_run = [m for m in ["fmri", "eeg"] if m in available_modalities]
        if not modalities_to_run:
            raise ValueError(
                f"modality='both' specified, but the graph cache contains neither "
                f"'fmri' nor 'eeg' nodes.\n"
                f"Graph cache node types: {available_modalities}"
            )
        logger.info(
            "modality='both': running pipeline sequentially for %s "
            "(outputs in output_dir/<modality>/)",
            modalities_to_run,
        )

    else:
        # Single modality
        if modality_cfg not in available_modalities:
            raise ValueError(
                f"Requested modality='{modality_cfg}' not found in graph cache "
                f"node types.\n"
                f"Graph cache node types: {available_modalities}\n"
                f"Supported modalities: 'fmri', 'eeg', 'both', 'joint'.\n"
                f"Other node types found: "
                f"{[m for m in available_modalities if m not in ('fmri', 'eeg')]}"
            )
        modalities_to_run = [modality_cfg]

    # ── Dispatch ──────────────────────────────────────────────────────────────
    def _make_simulator(mod: str):
        """Create simulator + optional energy-constraint wrapper for modality."""
        sim = BrainDynamicsSimulator(
            model=twin,
            base_graph=base_graph,
            modality=mod,
            fmri_subsample=cfg["simulator"].get("fmri_subsample", 25),
            device=device,
        )
        if mod == "joint":
            logger.info(
                "  Simulator [joint]: N_fmri=%d, N_eeg=%d, N_joint=%d, dt=%.4f s/TR",
                sim.n_fmri_regions, sim.n_eeg_regions, sim.n_regions, sim.dt,
            )
        else:
            logger.info(
                "  Simulator [%s]: n_regions=%d, dt=%.4f s/TR",
                mod, sim.n_regions, sim.dt,
            )
        ec_budget = cfg.get("advanced", {}).get("energy_constraint", {}).get("E_budget")
        if ec_budget is not None:
            from experiments.energy_constraint import EnergyConstrainedSimulator
            sim = EnergyConstrainedSimulator(sim, E_budget=float(ec_budget))
            logger.info("⚡ Energy constraint: E_budget=%.4f", ec_budget)
        return sim

    if len(modalities_to_run) == 1:
        # Single or joint: flat result dict (backward-compatible)
        mod = modalities_to_run[0]
        simulator = _make_simulator(mod)
        results = _run_phases_for_modality(cfg, simulator, mod, output_dir, device)
        elapsed = time.time() - t0
        logger.info("=" * 60)
        logger.info(
            "✓ Pipeline [%s] complete in %.1f s. Results: %s",
            mod.upper(), elapsed, output_dir.resolve(),
        )
        return results

    # Both modalities: run sequentially, store under output_dir/{modality}/
    all_results: Dict[str, Any] = {}
    for mod in modalities_to_run:
        logger.info("=" * 60)
        logger.info("◆ Starting [%s] modality pipeline", mod.upper())
        mod_output_dir = output_dir / mod
        mod_output_dir.mkdir(parents=True, exist_ok=True)
        try:
            simulator = _make_simulator(mod)
            all_results[mod] = _run_phases_for_modality(
                cfg, simulator, mod, mod_output_dir, device,
            )
        except Exception as exc:
            logger.error("  [%s] pipeline failed: %s", mod.upper(), exc)
            all_results[mod] = {"error": str(exc)}

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "✓ Both-modality pipeline complete in %.1f s. Results in: %s/{%s}",
        elapsed, output_dir.resolve(), ",".join(modalities_to_run),
    )
    return all_results


def _pca_burnin(n_steps: int) -> int:
    """Canonical PCA burn-in formula shared by ``run_phase3_dynamics`` and
    ``_save_summary_plots``.

    With context-end-aligned initial states (``from_data=True``, ``step_idx``
    set to the natural continuation of each context window), there is no large
    correction transient.  A small burn-in of ``T // 10`` is sufficient to
    discard the micro-perturbation noise and the first prediction chunk.
    The cap at ``T // 2`` ensures that at least half the trajectory is always
    available for analysis even for very short test trajectories.
    """
    return min(max(10, n_steps // 10), n_steps // 2) if n_steps > 0 else 0


def _save_summary_plots(results: Dict[str, Any], output_dir: Path,
                        simulator) -> None:
    """Save summary visualisations."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    try:
        from visualization.trajectory_plot import (
            plot_pca_trajectories,
            plot_region_heatmap,
            plot_trajectory_norms,
            plot_trajectory_convergence,
            plot_lyapunov_histogram,
            plot_lyapunov_growth,
            plot_basin_sizes,
        )
    except ImportError:
        logger.debug("Visualization modules unavailable, skipping plots.")
        return

    trajs = results.get("trajectories")
    if trajs is not None:
        try:
            plot_trajectory_norms(trajs, save_path=plots_dir / "trajectory_norms.png")
        except Exception:
            pass
        try:
            # Small burnin via _pca_burnin(): context-end-aligned x0 has no large
            # correction transient.  Capped at T//2 for test safety.
            plot_pca_trajectories(
                trajs, save_path=plots_dir / "pca_trajectories.png",
                burnin=_pca_burnin(trajs.shape[1]),
            )
        except Exception:
            pass
        try:
            plot_region_heatmap(
                trajs[0], title="Free Dynamics — Trajectory 0",
                save_path=plots_dir / "region_heatmap.png",
            )
        except Exception:
            pass

    lya = results.get("lyapunov")
    if lya is not None:
        try:
            plot_lyapunov_histogram(
                lya["lyapunov_values"],
                save_path=plots_dir / "lyapunov_histogram.png",
            )
        except Exception:
            pass

    att = results.get("attractor")
    if att is not None:
        try:
            plot_basin_sizes(
                att["basin_distribution"],
                save_path=plots_dir / "basin_sizes.png",
            )
        except Exception:
            pass

    conv = results.get("convergence")
    if conv is not None:
        try:
            plot_trajectory_convergence(
                conv["mean_distances"],
                save_path=plots_dir / "trajectory_convergence.png",
            )
        except Exception:
            pass

    logger.info("  → Plots saved to: %s", plots_dir)
