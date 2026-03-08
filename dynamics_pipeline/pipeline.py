#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dynamics_pipeline — Unified Brain Dynamics Analysis Pipeline
============================================================

Combines ``twinbrain-dynamics`` (model-driven) and ``spectral_dynamics``
(matrix-driven) into a single, config-driven orchestrator.

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
_TD_DIR = _REPO_ROOT / "twinbrain-dynamics"
_SD_DIR = _REPO_ROOT / "spectral_dynamics"

for _p in (_REPO_ROOT, _TD_DIR, _SD_DIR):
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
        n_nodes = min(rm_cfg.get("n_nodes", simulator.n_regions), simulator.n_regions)
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
    W = None
    w_label = "response_matrix"
    if R is not None:
        W = R
    elif trajs is not None:
        # Compute functional connectivity from trajectories
        stacked = trajs.reshape(-1, trajs.shape[-1])
        W = np.corrcoef(stacked.T)
        W = np.nan_to_num(W, nan=0.0)
        w_label = "fc"
        logger.info("  No response matrix; using FC from trajectories.")

    if W is None:
        logger.warning("  No connectivity matrix available, skipping Phase 2.")
        return

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
                spec["n_dominant"], spec["gap_ratio"],
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
                comm["modularity_Q"], comm["n_communities"],
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
                W, trajectories=trajs, label=w_label, output_dir=struct_dir,
            )
            results["modal_energy"] = modal
            logger.info(
                "  Modal energy: top5=%.1f%%, n@90%%=%d",
                modal.get("cumulative_energy_top5", 0) * 100,
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
            burnin = max(0, trajs.shape[1] // 10) if trajs.ndim >= 2 else 0
            pca = run_pca_attractor(
                trajectories=trajs,
                burnin=burnin,
                output_dir=dyn_dir,
            )
            results["pca"] = pca
            logger.info(
                "  PCA: var_top5=%.1f%%, n@90%%=%d",
                pca.get("variance_top5", 0) * 100,
                pca.get("n_components_90", 0),
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
        # D₂ from Grassberger-Procaccia
        try:
            from analysis.embedding_dimension import correlation_dimension
            ref_idx = trajs.shape[0] // 2
            ref_traj = trajs[ref_idx]
            burnin = max(0, ref_traj.shape[0] // 10)
            d2_out = correlation_dimension(ref_traj[burnin:])
            dim_results["D2"] = d2_out.get("D2", float("nan"))
            dim_results["D2_fit_r2"] = d2_out.get("fit_r2", float("nan"))
            logger.info(
                "  Attractor dim: D₂=%.2f (R²=%.3f)",
                dim_results["D2"], dim_results["D2_fit_r2"],
            )
        except Exception as e:
            logger.warning("  D₂ computation failed: %s", e)

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
        R = results.get("response_matrix")
        if R is not None:
            try:
                from spectral_dynamics.e5_phase_diagram import run_phase_diagram
                pd_cfg = adv_cfg.get("phase_diagram", {})
                phase = run_phase_diagram(
                    R, output_dir=adv_dir,
                    g_min=pd_cfg.get("g_min", 0.1),
                    g_max=pd_cfg.get("g_max", 3.0),
                    g_step=pd_cfg.get("g_step", 0.2),
                )
                results["phase_diagram"] = phase
                logger.info("  Phase diagram: done.")
            except Exception as e:
                logger.warning("  Phase diagram failed: %s", e)

    # 5d: Controllability
    if adv_cfg.get("controllability", {}).get("enabled", False):
        R = results.get("response_matrix")
        if R is not None:
            try:
                from analysis.controllability import run_controllability_analysis
                ctrl = run_controllability_analysis(
                    response_matrix=R, output_dir=adv_dir,
                )
                results["controllability"] = ctrl
                logger.info("  Controllability: done.")
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

    # 5f: Critical slowing down
    if adv_cfg.get("critical_slowing_down", {}).get("enabled", False) and trajs is not None:
        try:
            from analysis.critical_slowing_down import run_critical_slowing_down_analysis
            csd = run_critical_slowing_down_analysis(
                trajectories=trajs, output_dir=adv_dir,
            )
            results["critical_slowing_down"] = csd
            logger.info("  Critical slowing down: done.")
        except Exception as e:
            logger.warning("  Critical slowing down failed: %s", e)


def run_phase6_synthesis(cfg: dict, results: Dict[str, Any],
                         output_dir: Path) -> None:
    """Phase 6: Synthesis — consistency checks & hypothesis evaluation."""
    _step(6, "Synthesis & Report")

    report: Dict[str, Any] = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

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

    # ── Regime summary ────────────────────────────────────────────────────────
    regime = lya.get("chaos_regime", {}).get("regime", "unknown")
    report["regime"] = regime
    report["lle"] = float(mean_lle) if mean_lle is not None else None
    report["dmd_rho"] = float(rho_dmd) if rho_dmd is not None else None

    # Save report
    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    logger.info("  → Report saved: %s", report_path)

    results["report"] = report


def _evaluate_hypotheses(results: Dict[str, Any]) -> Dict[str, Dict]:
    """Evaluate hypotheses H1–H5 from combined results."""
    H: Dict[str, Dict] = {}

    # H1: Low-rank spectral structure
    spec = results.get("spectral", {})
    pr = spec.get("participation_ratio")
    n_dom = spec.get("n_dominant")
    N = spec.get("n_regions")
    if pr is not None and N is not None and N > 0:
        ratio = pr / N
        supported = ratio < 0.3
        H["H1"] = {
            "name": "Low-rank spectral structure",
            "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "summary": f"PR/N={ratio:.2f} ({'<' if supported else '≥'}0.30), "
                       f"n_dominant={n_dom}",
            "PR": float(pr),
            "PR_N_ratio": float(ratio),
        }
    else:
        H["H1"] = {"name": "Low-rank spectral structure",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No spectral data."}

    # H2: Low-dimensional dynamics
    # Evidence: PCA n@90%, correlation dimension D₂, linearised K-Y dimension
    pca = results.get("pca", {})
    n90 = pca.get("n_components_90")
    ad = results.get("attractor_dimension", {})
    d2 = ad.get("D2")
    ky_lin = ad.get("KY_linearised")
    if n90 is not None and N is not None and N > 0:
        dim_ratio = n90 / N
        supported = dim_ratio < 0.15
        summary_parts = [f"n@90%/N={dim_ratio:.2f} ({'<' if supported else '≥'}0.15)"]
        if d2 is not None and np.isfinite(d2):
            summary_parts.append(f"D₂={d2:.2f}")
        if ky_lin is not None:
            summary_parts.append(f"K-Y_lin={ky_lin:.2f}")
        H["H2"] = {
            "name": "Low-dimensional dynamics",
            "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "summary": ", ".join(summary_parts),
            "n_components_90": n90,
            "dim_ratio": float(dim_ratio),
            "D2": float(d2) if d2 is not None and np.isfinite(d2) else None,
            "KY_linearised": float(ky_lin) if ky_lin is not None else None,
        }
    else:
        H["H2"] = {"name": "Low-dimensional dynamics",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No PCA data."}

    # H3: Near-critical dynamics
    lya = results.get("lyapunov", {})
    regime = lya.get("chaos_regime", {}).get("regime")
    lle = lya.get("mean_lyapunov")
    if regime is not None:
        near_critical = regime in _NEAR_CRITICAL_REGIMES
        H["H3"] = {
            "name": "Near-critical dynamics",
            "verdict": "SUPPORTED" if near_critical else "NOT_SUPPORTED",
            "summary": f"regime={regime}, λ={lle:.5f}" if lle is not None else f"regime={regime}",
            "regime": regime,
            "lle": float(lle) if lle is not None else None,
        }
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
            plot_pca_trajectories(
                trajs, save_path=plots_dir / "pca_trajectories.png",
                burnin=max(0, trajs.shape[1] // 10),
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
