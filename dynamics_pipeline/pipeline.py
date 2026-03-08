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

# ── Burnin policy for power spectrum ──────────────────────────────────────────
_PS_BURNIN_MIN = 20
_PS_BURNIN_FRACTION = 10


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
            from e1_spectral_analysis import run_spectral_analysis
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
            from c_community_structure import run_community_structure
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
            from d_hierarchical_structure import run_hierarchical_structure
            hier = run_hierarchical_structure(W, label=w_label, output_dir=struct_dir)
            results["hierarchy"] = hier
            logger.info("  Hierarchy index: %.4f", hier.get("hierarchy_index", 0))
        except Exception as e:
            logger.warning("  Hierarchical analysis failed: %s", e)

    # 2d: Modal energy
    if ns_cfg.get("modal_energy", {}).get("enabled", True) and trajs is not None:
        try:
            from e2_e3_modal_projection import run_modal_projection
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
            from a_connectivity_visualization import run_connectivity_visualization
            comm_labels = results.get("community", {}).get("labels")
            run_connectivity_visualization(
                W, community_labels=comm_labels,
                label=w_label, output_dir=struct_dir,
            )
            logger.info("  Connectivity plots saved.")
        except Exception as e:
            logger.warning("  Connectivity visualization failed: %s", e)


def run_phase3_dynamics(cfg: dict, results: Dict[str, Any],
                        simulator, output_dir: Path) -> None:
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

    # 3d: Lyapunov exponent (Rosenstein)
    lya_cfg = dyn_cfg.get("lyapunov", {})
    if lya_cfg.get("enabled", True):
        try:
            from analysis.lyapunov import run_lyapunov_analysis
            modality = cfg.get("simulator", {}).get("modality", "fmri")
            method = "rosenstein"
            lya = run_lyapunov_analysis(
                trajectories=trajs,
                simulator=simulator,
                method=method,
                convergence_result=results.get("convergence"),
                convergence_threshold=lya_cfg.get("convergence_threshold", 0.05),
                n_segments=lya_cfg.get("n_segments", 3),
                rosenstein_max_lag=lya_cfg.get("max_lag", 50),
                rosenstein_min_sep=lya_cfg.get("min_sep", 20),
                rosenstein_delay_embed_dim=lya_cfg.get("delay_embed_dim", 0),
                rosenstein_delay_embed_tau=lya_cfg.get("delay_embed_tau", 1),
                output_dir=dyn_dir,
            )
            results["lyapunov"] = lya
            regime = lya["chaos_regime"]["regime"]
            mean_lya = lya.get("mean_lyapunov", float("nan"))
            logger.info("  Lyapunov (Rosenstein): λ=%.5f, regime=%s", mean_lya, regime)
        except Exception as e:
            logger.warning("  Lyapunov analysis failed: %s", e)

    # 3e: DMD Jacobian spectrum (replaces Wolf-GS which has context dilution)
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
            results["dmd_spectrum"] = dmd
            logger.info(
                "  DMD spectrum: ρ=%.4f, n_slow=%d, n_Hopf=%d, f_dom=%.4f Hz",
                dmd["spectral_radius"], dmd["n_slow_modes"],
                dmd["n_hopf_pairs"], dmd["dominant_oscillation_hz"],
            )
        except Exception as e:
            logger.warning("  DMD Jacobian analysis failed: %s", e)

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
            from f_pca_attractor import run_pca_attractor
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
                from e4_structural_perturbation import run_structural_perturbation
                pert = run_structural_perturbation(R, output_dir=val_dir)
                results["perturbation"] = pert
                logger.info("  Structural perturbation: done (see report).")
            except Exception as e:
                logger.warning("  Structural perturbation failed: %s", e)


def run_phase5_advanced(cfg: dict, results: Dict[str, Any],
                        simulator, output_dir: Path) -> None:
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
                from e5_phase_diagram import run_phase_diagram
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
        # Rosenstein LLE > 0 implies chaos; DMD ρ ≈ 1.0 implies near-criticality
        # They should be consistent: ρ < 1 ↔ λ < 0, ρ ≈ 1 ↔ λ ≈ 0, ρ > 1 ↔ λ > 0
        # Note: DMD is linearized, so it may underestimate nonlinear chaos
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
        if consistent:
            logger.info(
                "  ✓ Consistency: Rosenstein λ=%.5f (%s) ↔ DMD ρ=%.4f (%s)",
                mean_lle, lle_sign, rho_dmd, rho_sign,
            )
        else:
            logger.warning(
                "  ⚠ Inconsistency: Rosenstein λ=%.5f (%s) vs DMD ρ=%.4f (%s)\n"
                "    Possible cause: DMD captures linearised dynamics only;\n"
                "    nonlinear effects may push the system across the chaos boundary.",
                mean_lle, lle_sign, rho_dmd, rho_sign,
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
    pca = results.get("pca", {})
    n90 = pca.get("n_components_90")
    modal = results.get("modal_energy", {})
    if n90 is not None and N is not None and N > 0:
        dim_ratio = n90 / N
        supported = dim_ratio < 0.15
        H["H2"] = {
            "name": "Low-dimensional dynamics",
            "verdict": "SUPPORTED" if supported else "NOT_SUPPORTED",
            "summary": f"n@90%/N={dim_ratio:.2f} ({'<' if supported else '≥'}0.15), "
                       f"n@90%={n90}",
            "n_components_90": n90,
            "dim_ratio": float(dim_ratio),
        }
    else:
        H["H2"] = {"name": "Low-dimensional dynamics",
                    "verdict": "INSUFFICIENT_DATA", "summary": "No PCA data."}

    # H3: Near-critical dynamics
    lya = results.get("lyapunov", {})
    regime = lya.get("chaos_regime", {}).get("regime")
    lle = lya.get("mean_lyapunov")
    if regime is not None:
        near_critical = regime in ("edge_of_chaos", "marginal_stable", "weakly_chaotic")
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
#  Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg: dict) -> Dict[str, Any]:
    """
    Run the complete dynamics analysis pipeline.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Dictionary of all analysis results.
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
    base_graph = load_graph_for_inference(
        graph_path=graph_path, device=device,
    )

    modality = cfg["simulator"].get("modality", "fmri")

    # ── Create simulator ──────────────────────────────────────────────────────
    simulator = BrainDynamicsSimulator(
        model=twin, base_graph=base_graph, modality=modality, device=device,
    )
    logger.info(
        "Simulator: modality=%s, n_regions=%d, dt=%.4f s",
        modality, simulator.n_regions, simulator.dt,
    )

    # ── Energy constraint wrapper (optional) ──────────────────────────────────
    ec_budget = cfg.get("advanced", {}).get("energy_constraint", {}).get("E_budget")
    if ec_budget is not None:
        from experiments.energy_constraint import EnergyConstrainedSimulator
        simulator = EnergyConstrainedSimulator(simulator, E_budget=float(ec_budget))
        logger.info("⚡ Energy constraint: E_budget=%.4f", ec_budget)

    # ── Run phases ────────────────────────────────────────────────────────────
    results: Dict[str, Any] = {}

    results.update(run_phase1_data(cfg, simulator, output_dir, device))
    run_phase2_structure(cfg, results, output_dir)
    run_phase3_dynamics(cfg, results, simulator, output_dir)
    run_phase4_validation(cfg, results, output_dir)
    run_phase5_advanced(cfg, results, simulator, output_dir)
    run_phase6_synthesis(cfg, results, output_dir)

    # ── Plots ─────────────────────────────────────────────────────────────────
    if cfg["output"].get("save_plots"):
        _save_summary_plots(results, output_dir, simulator)

    elapsed = time.time() - t0
    logger.info("=" * 60)
    logger.info(
        "✓ Pipeline complete in %.1f s. Results: %s", elapsed, output_dir.resolve(),
    )
    return results


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
