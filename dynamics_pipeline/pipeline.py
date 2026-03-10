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

    **NOTE for E5 (phase diagram)**: FC should be avoided for spectral-radius
    analysis because its Marchenko-Pastur bulk makes ρ(FC) >> 1 even for stable
    dynamics.  The DMD operator is preferred when the response matrix is
    non-square, because it represents the best linear approximation of the
    GNN Jacobian at the attractor.

    Returns None when no square matrix can be constructed.
    """
    R = results.get("response_matrix")
    if R is not None and R.ndim == 2 and R.shape[0] == R.shape[1]:
        return R

    # Non-square response matrix (n_nodes × N_regions): prefer DMD operator.
    if R is not None and R.ndim == 2 and R.shape[0] != R.shape[1]:
        logger.info(
            "  Response matrix is non-square (%s); trying DMD operator next.",
            "×".join(str(d) for d in R.shape),
        )

    # Try DMD operator from Phase 3e (always N×N)
    dmd = results.get("dmd_spectrum", {})
    dmd_A = dmd.get("dmd_A")
    if dmd_A is not None and dmd_A.ndim == 2 and dmd_A.shape[0] == dmd_A.shape[1]:
        logger.info("  Using DMD operator (N=%d) as square connectivity.", dmd_A.shape[0])
        return dmd_A

    # Fall back to FC from trajectories — NOTE: ρ(FC) is typically >> 1 due to
    # Marchenko-Pastur bulk; E5 spectral-radius conclusions will be unreliable.
    if trajs is not None:
        stacked = trajs.reshape(-1, trajs.shape[-1])
        W_fc = np.corrcoef(stacked.T)
        W_fc = np.nan_to_num(W_fc, nan=0.0)
        logger.warning(
            "  _get_square_connectivity: falling back to FC (N=%d). "
            "ρ(FC) >> 1 — E5 phase-diagram results will be unreliable. "
            "Enable dmd_spectrum in config for a better linearisation.",
            W_fc.shape[0],
        )
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

    # ── Amplitude normalisation (optional, for studying joint-mode dynamics) ──
    # When simulator.modality == 'joint' the concatenated z-score state can
    # have high L2 norms, which inflates the DMD spectral radius beyond 1 even
    # when the system is near-critical.  Setting normalize_amplitude=true
    # projects each time step onto the unit sphere so that only the directional
    # dynamics are analysed.  Both trajectories AND the amplitude-normalised
    # copy are stored; downstream phases use the (potentially normalised) copy.
    sim_cfg = cfg.get("simulator", {})
    if sim_cfg.get("normalize_amplitude", False):
        norms = np.linalg.norm(trajectories, axis=-1, keepdims=True)
        # Guard against zero norms (degenerate states)
        norms = np.where(norms < 1e-30, 1.0, norms)
        trajectories_norm = trajectories / norms
        results["trajectories_raw"] = trajectories          # original
        results["trajectories"] = trajectories_norm         # normalised (used by all phases)
        mean_norm = float(np.mean(np.linalg.norm(trajectories, axis=-1)))
        logger.info(
            "  Amplitude normalisation: each state vector projected onto unit sphere. "
            "Original mean ||x||₂=%.4f. "
            "DMD spectral radius may now differ from the unnormalised run.",
            mean_norm,
        )
    else:
        results["amplitude_normalised"] = False

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

    # 2b: Community detection — always uses FC (Pearson correlation) matrix.
    # Scientific rationale (DeepSeek review / Bullmore & Sporns 2012):
    #   Louvain maximises modularity Q for *undirected* similarity graphs.
    #   The response matrix R is asymmetric and causal; Q(R) ≈ 0 is expected
    #   by construction (incoming and outgoing weights cancel in the Newman–
    #   Girvan formula), not a substantive finding about brain organisation.
    #   Functional connectivity (Pearson correlation of free-dynamics
    #   trajectories) is the appropriate input for brain modularity analysis.
    comm_cfg = ns_cfg.get("community", {})
    if comm_cfg.get("enabled", True):
        try:
            from spectral_dynamics.c_community_structure import run_community_structure
            if trajs is not None:
                _stacked_comm = trajs.reshape(-1, trajs.shape[-1])
                W_comm = np.corrcoef(_stacked_comm.T)
                W_comm = np.nan_to_num(W_comm, nan=0.0)
                comm_label = "fc"
                if connectivity_source == "response_matrix":
                    logger.info(
                        "  Community detection: using FC (Pearson correlation from "
                        "trajectories) instead of response_matrix.  "
                        "Louvain Q is defined for undirected graphs; the causal "
                        "response matrix is asymmetric → Q≈0 by construction."
                    )
            else:
                # No trajectories available — fall back to W with an explicit warning
                W_comm = W
                comm_label = w_label
                logger.warning(
                    "  Community detection: no trajectory data available; "
                    "falling back to %s.  Q may be uninformative for asymmetric R.",
                    w_label,
                )
            n_null = comm_cfg.get("n_null", 100)
            comm = run_community_structure(
                W_comm, k_range=comm_cfg.get("k_range", [3, 4, 5, 6, 7, 8]),
                label=comm_label, output_dir=struct_dir,
                n_null=n_null,
            )
            results["community"] = comm
            sig_info = comm.get("q_significance")
            if sig_info:
                logger.info(
                    "  Community (FC): Q=%.4f, k=%d, method=%s | "
                    "null z=%.2f, p=%.4f (%s)",
                    comm["modularity_q"], comm["n_communities"],
                    comm.get("method", "unknown"),
                    sig_info["z_score"], sig_info["p_value"],
                    "significant" if sig_info["significant"] else "not significant",
                )
            else:
                logger.info(
                    "  Community (FC): Q=%.4f, k=%d, method=%s",
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
            vc = stab.get("classification_counts", {})  # Method C (adaptive)
            n_traj = max(sum(vc.values()), 1) if vc else 1
            primary_c = max(vc, key=lambda k: vc.get(k, 0)) if vc else "N/A"
            logger.info(
                "  Stability (Method C adaptive): %s (%.0f%% of trajectories)",
                primary_c,
                vc.get(primary_c, 0) / n_traj * 100,
            )
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
                att.get("kmeans_k", 0),
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
        n90 = pca.get("n_components_90pct")  # correct key from f_pca_attractor.py
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
                emb.get("fnn", {}).get("min_sufficient_dim", "?"),
                emb.get("corr_dim", {}).get("D2", float("nan")),
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
            csd_agg = csd.get("aggregate", {})
            logger.info(
                "  Critical slowing down: ac1_tau=%.3f, var_tau=%.3f, ews_score=%.3f",
                csd_agg.get("ac1_tau_mean", float("nan")),
                csd_agg.get("var_tau_mean", float("nan")),
                csd_agg.get("ews_score_mean", float("nan")),
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
                         output_dir: Path,
                         modality: Optional[str] = None) -> None:
    """Phase 6: Synthesis — consistency checks & hypothesis evaluation.

    Args:
        cfg:        Merged pipeline configuration.
        results:    Accumulated analysis results from phases 1–5.
        output_dir: Directory where ``pipeline_report.json`` and
                    ``analysis_report.md`` are written.
        modality:   The modality actually being analysed (``"fmri"``,
                    ``"eeg"``, or ``"joint"``).  When provided, this value
                    is used in the report metadata instead of
                    ``cfg["simulator"]["modality"]`` (which is ``"both"``
                    in ``--modality both`` runs, not the per-modality string).
    """
    _step(6, "Synthesis & Report")

    # ── P2-1: Run metadata ────────────────────────────────────────────────────
    import hashlib
    dg = cfg.get("data_generation", {})
    sim_cfg = cfg.get("simulator", {})
    # Prefer the caller-supplied modality (correct when running --modality both,
    # where cfg["simulator"]["modality"] == "both" but each sub-run is "fmri"
    # or "eeg").  Fall back to cfg value for single-modality / joint runs.
    actual_modality = modality or sim_cfg.get("modality", "fmri")
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
            "modality": actual_modality,
            "normalize_amplitude": sim_cfg.get("normalize_amplitude", False),
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

    # ── Multi-source LLE consistency audit (addresses reviewer Issue 1) ───────
    # Collect all LLE values computed by different sub-analyses and methods.
    # Expected variation across methods/trajectory subsets is normal (different
    # n_traj, lle_steps, initialisation noise) but large disagreements should
    # be flagged for the researcher.
    lle_audit: Dict[str, Any] = {}
    if mean_lle is not None:
        lle_audit["rosenstein_main"] = {
            "value": float(mean_lle),
            "source": "Phase 3d Rosenstein (primary, all trajectories)",
        }
    surr_lle = surr.get("rosenstein_lle") if surr else None
    if surr_lle is not None:
        lle_audit["surrogate_test_lle"] = {
            "value": float(surr_lle),
            "source": "Phase 4a surrogate test (n_traj subset)",
        }
    rc = results.get("random_comparison") or {}
    brain_gnn_lle = None
    graph_cmp = results.get("graph_structure_comparison") or {}
    if graph_cmp:
        _bg = graph_cmp.get("brain_graph", {})
        brain_gnn_lle = _bg.get("lle_gnn")
        brain_tanh_lle = _bg.get("lle")
        if brain_gnn_lle is not None:
            lle_audit["graph_comparison_gnn_lle"] = {
                "value": float(brain_gnn_lle),
                "source": "Phase 4b graph comparison (GNN, from Phase 1)",
            }
        if brain_tanh_lle is not None:
            lle_audit["graph_comparison_tanh_lle"] = {
                "value": float(brain_tanh_lle),
                "source": (
                    "Phase 4b graph comparison [tanh(W@x) ANALYTICAL SURROGATE — "
                    "NOT the GNN model; may be 0 when spectral_radius(W)<1]"
                ),
                "is_analytical_surrogate": True,
            }
    idc = results.get("input_dimension_control") or {}
    if idc:
        for cond_name, cond_val in idc.items():
            if isinstance(cond_val, dict) and "lle" in cond_val:
                lle_audit[f"input_control_{cond_name}"] = {
                    "value": float(cond_val["lle"]),
                    "source": f"Phase 5 input-dim control ({cond_name})",
                    "is_analytical_surrogate": False,
                }
    if lle_audit:
        # Exclude analytical surrogate entries (tanh(W@x)) from the consistency
        # range calculation — they measure a different model, not the GNN.
        finite_lles = [
            v["value"] for v in lle_audit.values()
            if isinstance(v, dict)
            and isinstance(v.get("value"), float)
            and np.isfinite(v["value"])
            and not v.get("is_analytical_surrogate", False)
        ]
        if len(finite_lles) >= 2:
            lle_range = max(finite_lles) - min(finite_lles)
            lle_audit["_range_excl_tanh"] = round(lle_range, 6)
            if lle_range > 0.02:
                lle_audit["_range_note"] = (
                    "LLE estimates vary by >{:.4f} across methods/subsets.  "
                    "This is expected: Rosenstein LLE depends on n_traj, "
                    "lle_steps, embedding parameters, and trajectory "
                    "initialisation.  The Phase 3d estimate (all trajectories, "
                    "n_segments adjusted for convergence) is the primary value.  "
                    "Large discrepancies may also reflect transient vs. "
                    "attractor-phase sampling.".format(lle_range)
                )
                logger.warning(
                    "  ⚠ LLE multi-source range=%.4f (methods disagree). "
                    "Primary estimate: λ=%.5f (Phase 3d Rosenstein). "
                    "See pipeline_report.json['lle_audit'] for details.",
                    lle_range,
                    mean_lle if mean_lle is not None else float("nan"),
                )
            else:
                lle_audit["_range_note"] = (
                    f"LLE estimates consistent (range={lle_range:.4f})."
                )
                logger.info(
                    "  ✓ LLE multi-source consistency: range=%.4f across %d sources.",
                    lle_range, len(finite_lles),
                )
    if lle_audit:
        report["lle_audit"] = lle_audit

    # ── SVD vs PCA variance note (addresses reviewer Issue 3) ─────────────────
    # The response matrix SVD (Phase 2 spectral) and trajectory PCA (Phase 3g)
    # may show very different variance concentrations.  This is scientifically
    # EXPECTED and NOT a contradiction:
    #   SVD(R):   measures how causal influence is distributed across response
    #             directions.  High-rank R (energy spread over many singular
    #             vectors) means causal influence propagates through many
    #             independent pathways.
    #   PCA(trajectories): measures how much of the spontaneous state variance
    #             lies in a few directions.  Low rank means the free dynamics
    #             are confined to a low-dimensional manifold (H2).
    # A high-rank causal matrix can produce a low-dimensional spontaneous
    # manifold through nonlinear constraints and attractor geometry.
    pca_res = results.get("pca", {})
    spec_res = results.get("spectral", {})
    if pca_res and spec_res:
        n90 = pca_res.get("n_components_90pct")  # correct key from f_pca_attractor.py
        pr = spec_res.get("participation_ratio")
        N_reg = spec_res.get("n_regions")
        if n90 is not None and pr is not None and N_reg is not None and N_reg > 0:
            N = int(N_reg)
            report["svd_vs_pca_note"] = {
                "spectral_PR": round(float(pr), 2),
                "pca_n90": int(n90),
                "n_regions": N,
                "explanation": (
                    "SVD(response_matrix) PR={pr:.1f}/{N} reflects how causal "
                    "influence is spread across response directions (high rank "
                    "= influence propagates via many pathways).  "
                    "PCA(trajectories) n@90%={n90}/{N} reflects the "
                    "dimensionality of spontaneous state variance "
                    "(low rank = dynamics confined to a manifold, H2).  "
                    "A high-rank causal map can coexist with low-dimensional "
                    "free dynamics via nonlinear attractor geometry — this is "
                    "NOT a contradiction.".format(pr=pr, N=N, n90=n90)
                ),
            }
            logger.info(
                "  SVD(R) PR=%.1f vs PCA n@90%%=%d (of N=%d): expected to "
                "differ — causal diversity ≠ dynamical dimensionality.",
                pr, n90, N,
            )

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

    # Generate clean AI-readable analysis report
    _generate_ai_report(report, results, output_dir)

    results["report"] = report


def _generate_ai_report(
    report: Dict[str, Any],
    results: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate a clean, human/AI-readable Markdown analysis report.

    Synthesises all six pipeline phases into a single ``analysis_report.md``
    file, structured for direct submission to an LLM.  The report avoids
    verbose log lines and numerical dumps; instead it presents key metrics,
    hypothesis verdicts, and evidence in a concise, annotated format.

    Args:
        report:     Phase-6 report dict (output of run_phase6_synthesis).
        results:    Full results dict accumulated across all phases.
        output_dir: Directory where ``analysis_report.md`` will be saved.
    """
    lines: List[str] = []

    def _h(n: int, text: str) -> None:
        lines.append("\n" + "#" * n + " " + text)

    def _row(key: str, val: Any, note: str = "") -> None:
        val_str = str(val) if val is not None else "N/A"
        if note:
            lines.append("- **{}**: {}  *({})* ".format(key, val_str, note))
        else:
            lines.append(f"- **{key}**: {val_str}")

    def _na(val: Any, fmt: str = ".4f") -> str:
        if val is None:
            return "N/A"
        try:
            return format(float(val), fmt)
        except (TypeError, ValueError):
            return str(val)

    # ── Header ────────────────────────────────────────────────────────────────
    meta = report.get("metadata", {})
    ts = report.get("run_timestamp", "unknown")
    lines.append(f"# Brain Dynamics Analysis Report")
    lines.append(f"\n**Generated**: {ts}  ")
    lines.append(f"**Modality**: {meta.get('modality', 'fmri')}  ")
    lines.append(f"**Config hash**: {meta.get('cfg_hash', 'unknown')}  ")
    lines.append(f"**n_init**: {meta.get('n_init', 'N/A')}  "
                 f"**steps**: {meta.get('steps', 'N/A')}  "
                 f"**seed**: {meta.get('seed', 42)}  ")
    norm_amp = meta.get("normalize_amplitude", False)
    lines.append(
        f"**Amplitude normalised**: {'✓ Yes (unit-sphere projection)' if norm_amp else 'No (raw amplitudes)'}  "
    )
    if norm_amp:
        lines.append(
            "> ⚠ Amplitude normalisation is active: each state vector was projected "
            "onto the unit sphere before all Phase 2–6 analyses. "
            "DMD spectral radius reflects directional dynamics only; "
            "compare with a non-normalised run to assess amplitude-driven effects."
        )
    conn_src = report.get("connectivity_source", "unknown")
    ev_grade = report.get("evidence_grade", "unknown")
    lines.append(f"**Connectivity source**: {conn_src}  "
                 f"**Evidence grade**: {ev_grade}  ")
    lines.append(
        "\n> **How to use**: paste this file directly into an LLM prompt. "
        "All key metrics, hypothesis verdicts, and confidence levels are "
        "included. Raw arrays and per-step logs are in the JSON outputs."
    )

    # ── Phase 1: Data Generation ──────────────────────────────────────────────
    _h(2, "Phase 1 — Data Generation")
    trajs = results.get("trajectories")
    if trajs is not None and hasattr(trajs, "shape"):
        n_traj, steps, n_reg = trajs.shape
        lines.append(
            f"Free-dynamics trajectories generated: "
            f"**{n_traj}** trajectories × **{steps}** steps × **{n_reg}** regions"
        )
    else:
        lines.append("Free-dynamics trajectories: N/A (Phase 1 output not stored in memory)")
    rm = results.get("response_matrix")
    if rm is not None and hasattr(rm, "shape"):
        lines.append(f"Response matrix: **{rm.shape[0]}×{rm.shape[1]}** (causal, asymmetric)")
    else:
        lines.append("Response matrix: N/A")

    # ── Phase 2: Network Structure ────────────────────────────────────────────
    _h(2, "Phase 2 — Network Structure")

    # 2a Spectral
    spec = results.get("spectral", {})
    if spec:
        _h(3, "2a Spectral Decomposition")
        _row("Spectral radius ρ(W)", _na(spec.get("spectral_radius")),
             "≈1 → near-critical (H1)")
        _row("Participation ratio PR/N",
             _na(spec.get("participation_ratio_norm",
                          (spec.get("participation_ratio") or 0) /
                          max(spec.get("n_regions", 1), 1)),
                 ".3f"),
             "< 0.3 → low-rank (H1)")
        _row("Spectral gap ratio", _na(spec.get("spectral_gap_ratio"), ".3f"))
        _row("n_dominant modes", spec.get("n_dominant", "N/A"))

    # 2b Community
    comm = results.get("community", {})
    if comm:
        _h(3, "2b Community Structure")
        Q = comm.get("modularity_q")
        k = comm.get("n_communities")
        method = comm.get("method", "unknown")
        interp = comm.get("q_interpretation", "?")
        lines.append(f"- **Method**: {method}  **k**: {k}  **Q**: {_na(Q)}  "
                     f"**Interpretation**: {interp}")
        sig = comm.get("q_significance")
        if sig:
            sig_verdict = "significant (p<0.05)" if sig.get("significant") else "not significant (p≥0.05)"
            null_model_label = sig.get("null_model", "degree-preserving rewiring")
            lines.append(
                f"- **Q significance test** (null model: {null_model_label}, n={sig.get('n_null')}):"
                f"  null mean={_na(sig.get('null_mean'), '.4f')} ± "
                f"{_na(sig.get('null_std'), '.4f')},"
                f"  z={_na(sig.get('z_score'), '.2f')},"
                f"  p={_na(sig.get('p_value'), '.4f')} — **{sig_verdict}**"
            )
            if not sig.get("significant"):
                lines.append(
                    "  > ⚠ Q is not significantly higher than random networks with the same "
                    "degree sequence. The detected community structure may not reflect true "
                    "functional organisation; interpret with caution."
                )
        else:
            lines.append("  *(Q significance test not run — set community.n_null > 0 in config)*")

    # 2c Hierarchy
    hier = results.get("hierarchy", {})
    if hier:
        _h(3, "2c Hierarchical Structure")
        _row("Hierarchy index", _na(hier.get("hierarchy_index"), ".4f"))

    # 2d Modal energy
    me = results.get("modal_energy", {})
    if me:
        _h(3, "2d Modal Energy")
        _row("Top-5 modes cumulative energy",
             f"{_na(me.get('energy_top5_pct'), '.1f')}%",
             "< 5 modes → 90% energy = low-dimensional (H2)")
        _row("Modes for 90% energy", me.get("n_modes_90pct", "N/A"))

    # ── Phase 3: Dynamics Characterisation ───────────────────────────────────
    _h(2, "Phase 3 — Dynamics Characterisation")

    # 3a Stability
    stab = results.get("stability", {})
    if stab:
        _h(3, "3a Stability Classification")
        # Primary regime: dominant class in Method-C (adaptive) counts
        counts_c = stab.get("classification_counts", {})
        if counts_c:
            primary = max(counts_c, key=lambda k: counts_c.get(k, 0))
            frac = counts_c.get(primary, 0) / max(sum(counts_c.values()), 1)
            _row("Primary regime (Method C)", f"{primary} ({frac:.0%} of trajectories)")
        else:
            _row("Primary regime (Method C)", "N/A")
        # Method-C fraction breakdown
        n_traj = sum(counts_c.values()) if counts_c else 0
        if n_traj > 0:
            parts = [
                f"{k}={v/n_traj:.0%}"
                for k, v in counts_c.items()
                if v > 0
            ]
            _row("Method C breakdown", ", ".join(parts))
        dr = stab.get("delta_ratio_stats", {})
        if dr:
            _row("delta_ratio (mean)", _na(dr.get("mean"), ".4f"),
                 "relative oscillation amplitude; > adaptive threshold → limit cycle / weakly chaotic")

    # 3b Attractor
    attr = results.get("attractor", {})
    if attr:
        _h(3, "3b Attractor Analysis")
        _row("n_attractors (KMeans)", attr.get("kmeans_k", "N/A"))
        sil = attr.get("silhouette_score")
        if sil is not None:
            _row("silhouette score", _na(sil, ".4f"))
        _row("DBSCAN n_clusters", attr.get("dbscan_n_clusters", "N/A"))

    # 3c Convergence
    conv = results.get("convergence", {})
    if conv:
        _h(3, "3c Trajectory Convergence")
        _row("distance_ratio", _na(conv.get("distance_ratio"), ".4f"))
        _row("label", conv.get("label", "N/A"))

    # 3d Lyapunov (Rosenstein) — primary chaos indicator
    lya = results.get("lyapunov", {})
    if lya:
        _h(3, "3d Lyapunov Exponent (Rosenstein) — PRIMARY CHAOS INDICATOR")
        mean_lle = lya.get("mean_lyapunov")
        regime_info = lya.get("chaos_regime", {})
        regime = regime_info.get("regime", "unknown")
        lines.append(
            f"- **λ₁ (Rosenstein LLE)**: {_na(mean_lle, '.5f')}  "
            f"**Regime**: {regime}"
        )
        seg_vals = lya.get("segment_lles")
        if seg_vals:
            seg_str = ", ".join(_na(v, ".4f") for v in seg_vals)
            lines.append(f"- Segment estimates: [{seg_str}]")
        lines.append(
            f"  > λ<0 → stable, λ≈0 → edge of chaos (near-critical), "
            f"λ>0 → chaotic. Near-critical regimes: "
            f"edge_of_chaos / marginal_stable / weakly_chaotic."
        )

    # 3e DMD spectrum (linearised — not the chaos indicator)
    dmd = results.get("dmd_spectrum", {})
    if dmd:
        _h(3, "3e Linearised Spectrum (DMD) — complementary, NOT the chaos indicator")
        _row("DMD spectral radius ρ_DMD", _na(dmd.get("spectral_radius"), ".4f"),
             "ρ≈1 → near-critical (linearised view)")
        _row("n_slow modes (|Re(λ)|<0.05)", dmd.get("n_slow_modes", "N/A"))
        _row("n_Hopf pairs", dmd.get("n_hopf_pairs", "N/A"),
             "oscillatory modes")
        _row("K-Y dimension (linearised)", _na(dmd.get("linearised_ky_dim"), ".2f"))
        lines.append(
            f"  > {report.get('dmd_interpretation_note', '')}"
        )

    # 3f PSD
    psd = results.get("power_spectrum", {})
    if psd:
        _h(3, "3f Power Spectral Density")
        ba = psd.get("band_analysis", {})
        _row("Dominant frequency", f"{_na(ba.get('dominant_freq_hz'), '.4f')} Hz",
             ba.get("dominant_freq_band", ""))
        # Band power fractions are stored inside band_analysis["bands_used"] as
        # a list of (name, low, high) tuples, not as top-level pct keys.
        # Report Nyquist and dominant band only.
        _row("Nyquist frequency", f"{_na(ba.get('nyquist_hz'), '.4f')} Hz")

    # 3g PCA
    pca_res = results.get("pca", {})
    if pca_res:
        _h(3, "3g PCA Dimensionality")
        _row("n@90% variance (linear embedding dim)", pca_res.get("n_components_90pct", "N/A"),
             "upper bound on attractor dim (H2)")
        _row("Top-5 modes cumulative variance",
             f"{_na(pca_res.get('variance_top5_pct'), '.1f')}%")

    # 3h Attractor dimension
    d2_res = results.get("attractor_dimension", {})
    if d2_res:
        _h(3, "3h Attractor Dimension (three-way estimate)")
        d2_mean = d2_res.get("D2_mean", d2_res.get("D2"))
        d2_std = d2_res.get("D2_std")
        d2_str = _na(d2_mean, ".2f")
        if d2_std is not None and np.isfinite(float(d2_std)):
            d2_str += f"±{d2_std:.2f}"
        fail = d2_res.get("D2_fail_rate")
        if fail is not None:
            d2_str += f" (fail={fail:.0%})"
        _row("D₂ (correlation dimension, nonlinear)", d2_str)
        _row("K-Y_linear (DMD linearised)", _na(d2_res.get("KY_linearised"), ".2f"))
        _row("PCA n@90% (linear upper bound)", d2_res.get("PCA_n90", "N/A"))

    # ── Phase 4: Statistical Validation ──────────────────────────────────────
    _h(2, "Phase 4 — Statistical Validation")

    # 4a Surrogate test
    surr = results.get("surrogate_test", {})
    if surr:
        _h(3, "4a Surrogate Data Test")
        _row("real LLE", _na(surr.get("real_lle"), ".5f"))
        _row("is_nonlinear", surr.get("is_nonlinear", "N/A"))
        _row("z_score vs phase-randomised", _na(surr.get("z_score"), ".2f"))

    # 4b Random comparison
    rc = results.get("random_comparison", {})
    if rc:
        _h(3, "4b Random Network Comparison")
        for cond, val in rc.items():
            if isinstance(val, dict) and "mean_lyapunov" in val:
                _row(f"  {cond} LLE", _na(val.get("mean_lyapunov"), ".5f"))

    # 4c Embedding dimension
    emb = results.get("embedding_dimension", {})
    if emb:
        _h(3, "4c Embedding Dimension")
        _row("FNN min sufficient dim", emb.get("fnn", {}).get("min_sufficient_dim", "N/A"))
        _row("D₂ (correlation dim)", _na(emb.get("corr_dim", {}).get("D2"), ".2f"))

    # 4e Intrinsic dimension (TwoNN)
    id_res = results.get("intrinsic_dimension", {})
    if id_res:
        _h(3, "4e TwoNN Intrinsic Dimension")
        _row("mean local dim", _na(id_res.get("d_mean"), ".2f"))
        _row("median local dim", _na(id_res.get("d_median"), ".2f"))
        _row("std", _na(id_res.get("d_std"), ".2f"))
        n_valid = id_res.get("n_valid", 0)
        n_total = id_res.get("n_total", 0)
        _row("n_valid / n_total", f"{n_valid} / {n_total}")

    # 4j Graph structure comparison
    gsc = results.get("graph_structure_comparison", {})
    if gsc:
        _h(3, "4j Graph Structure Comparison")
        bg = gsc.get("brain_graph", {})
        dp = gsc.get("degree_preserving", {})
        fr = gsc.get("fully_random", {})
        lines.append(f"| Network | LLE (tanh) | PCA dim |")
        lines.append(f"|---------|-----------|---------|")
        lines.append(f"| Brain graph | {_na(bg.get('lle'), '.4f')} | {bg.get('pca_dim_90pct', 'N/A')} |")
        lines.append(f"| Degree-preserving | {_na(dp.get('lle_mean'), '.4f')} | {dp.get('pca_dim_90pct', 'N/A')} |")
        lines.append(f"| Fully random | {_na(fr.get('lle_mean'), '.4f')} | {fr.get('pca_dim_90pct', 'N/A')} |")

    # ── Phase 5: Advanced (optional) ─────────────────────────────────────────
    adv_keys = [
        ("virtual_stimulation", "5a Virtual Stimulation"),
        ("energy_constraint", "5b Energy Constraint"),
        ("controllability", "5c Controllability"),
        ("information_flow", "5e Information Flow (TE)"),
        ("critical_slowing_down", "5f Critical Slowing Down"),
    ]
    phase5_present = any(results.get(k) for k, _ in adv_keys)
    if phase5_present:
        _h(2, "Phase 5 — Advanced (optional)")
        for key, title in adv_keys:
            val = results.get(key)
            if val and isinstance(val, dict):
                _h(3, title)
                for k2, v2 in val.items():
                    if not isinstance(v2, (dict, list, np.ndarray)) and v2 is not None:
                        lines.append(f"- **{k2}**: {v2}")

    # ── Phase 6: Synthesis ────────────────────────────────────────────────────
    _h(2, "Phase 6 — Synthesis")

    # Consistency check
    cons = report.get("consistency")
    if cons:
        _h(3, "Rosenstein LLE vs DMD Consistency")
        _row("Rosenstein λ₁", _na(cons.get("rosenstein_lle"), ".5f"))
        _row("DMD ρ (spectral radius)", _na(cons.get("dmd_spectral_radius"), ".4f"))
        _row("LLE sign", cons.get("lle_sign", "N/A"))
        _row("DMD regime", cons.get("rho_sign", "N/A"))
        consistent = cons.get("consistent")
        _row("Consistent", "✓ Yes" if consistent else "✗ No (see note)")
        _row("Nonlinearity index Δ", _na(cons.get("nonlinearity_index"), ".4f"),
             "Δ>1 → strong nonlinearity, DMD is reference only")
        if cons.get("dmd_note"):
            lines.append(f"  > ⚠ {cons['dmd_note']}")

    # Hypotheses
    hypotheses = report.get("hypotheses", {})
    if hypotheses:
        _h(3, "Hypothesis Evaluation")
        lines.append("\n| Hypothesis | Question | Verdict | Evidence |")
        lines.append("|-----------|---------|---------|---------|")
        h_questions = {
            "H1": "Low-rank spectral structure?",
            "H2": "Low-dimensional dynamics?",
            "H3": "Near-critical regime?",
            "H4": "Brain-like oscillations?",
            "H5": "Energy constraint maintains criticality?",
        }
        conf_flags = report.get("confidence_flags", {})
        for h_id in sorted(hypotheses.keys()):
            h = hypotheses[h_id]
            verdict = h.get("verdict", "N/A")
            summary = h.get("summary", "")
            conf = conf_flags.get(h_id, "N/A")
            q_txt = h_questions.get(h_id, "")
            icon = "✓" if verdict == "SUPPORTED" else ("✗" if verdict == "NOT_SUPPORTED" else "○")
            lines.append(
                f"| **{h_id}** ({conf}) | {q_txt} | {icon} {verdict} | {summary} |"
            )

    # Confidence flags summary
    if conf_flags:
        _h(3, "Confidence Flags")
        for h_id, flag in conf_flags.items():
            lines.append(f"- **{h_id}**: {flag}")

    # LLE audit
    lle_audit = report.get("lle_audit", {})
    if lle_audit:
        _h(3, "LLE Multi-Source Audit")
        for src, info in lle_audit.items():
            if isinstance(info, dict):
                tag = " *(analytical surrogate)*" if info.get("is_analytical_surrogate") else ""
                lines.append(
                    f"- **{src}**: {_na(info.get('value'), '.5f')}"
                    f"  — {info.get('source', '')}{tag}"
                )
            elif src.startswith("_range"):
                lines.append(f"- {src}: {info}")

    # Validation table
    vt = report.get("validation_table", [])
    if vt:
        _h(3, "9-Question Verification Table")
        lines.append("\n| Q# | Question | Verdict | Evidence |")
        lines.append("|---|---------|---------|---------|")
        for row in vt:
            qid = row.get("question_id", "?")
            short = row.get("question_short", "?")
            verdict = row.get("verdict", "?")
            ev = str(row.get("evidence", ""))[:80]
            verified = row.get("verified")
            icon = "✓" if verified is True else ("○" if verified is None else "✗")
            lines.append(f"| {icon} Q{qid} | {short} | {verdict} | {ev} |")

    # Regime summary
    regime = report.get("regime", "unknown")
    lle_val = report.get("lle")
    rho_val = report.get("dmd_rho")
    _h(3, "Overall Summary")
    lines.append(
        f"| Metric | Value |"
        f"\n|--------|-------|"
        f"\n| Dynamical regime | **{regime}** |"
        f"\n| Rosenstein LLE | **{_na(lle_val, '.5f')}** |"
        f"\n| DMD spectral radius | **{_na(rho_val, '.4f')}** |"
        f"\n| Connectivity source | {conn_src} |"
        f"\n| Evidence grade | {ev_grade} |"
    )

    # Write file
    md_text = "\n".join(lines) + "\n"
    report_path = output_dir / "analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    logger.info("  → AI analysis report saved: %s", report_path)


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
        # |delta_rho| > 0.05 in either direction means the trained structure is distinctive.
        # (delta_rho = original_rho - random_rho; negative = training reduces spectral radius)
        verified_q4 = bool(delta_rho is not None and abs(delta_rho) > 0.05)
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
        # rc has keys like "random_sr1.50" with mean_lyapunov inside, not "random_lle_mean".
        # Compute the mean LLE across all random spectral-radius conditions.
        rand_lles = [
            v.get("mean_lyapunov")
            for k, v in rc.items()
            if k.startswith("random_sr") and isinstance(v, dict)
        ]
        rand_lles = [x for x in rand_lles if x is not None and np.isfinite(x)]
        rand_lle = float(np.mean(rand_lles)) if rand_lles else None
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
    n90 = pca.get("n_components_90pct")  # correct key from f_pca_attractor.py
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
            "n_components_90pct": n90,
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
            csd_agg = csd.get("aggregate", {})
            # CSD returns ac1_tau_mean / var_tau_mean (Kendall τ); no p-values.
            ac1_tau = csd_agg.get("ac1_tau_mean")
            var_tau = csd_agg.get("var_tau_mean")
            ews_score = csd_agg.get("ews_score_mean")
            csd_parts = []
            if ac1_tau is not None:
                csd_parts.append(f"AC1_tau={ac1_tau:.2f}")
            if var_tau is not None:
                csd_parts.append(f"Var_tau={var_tau:.2f}")
            if ews_score is not None:
                csd_parts.append(f"EWS={ews_score:.2f}")
            if csd_parts:
                summary_parts.append("CSD:[" + ", ".join(csd_parts) + "]")
            h3_entry["csd"] = {
                "ac1_tau_mean": ac1_tau,
                "var_tau_mean": var_tau,
                "ews_score_mean": ews_score,
            }
            # Rising AR1 and variance → consistent with approach to critical transition
            csd_consistent = (
                ac1_tau is not None and ac1_tau > 0 and
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
    run_phase6_synthesis(cfg, results, output_dir, modality=modality)

    if cfg["output"].get("save_plots"):
        _save_summary_plots(results, output_dir, simulator)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Both-mode summary helper
# ═══════════════════════════════════════════════════════════════════════════════

def _write_both_mode_summary(
    output_dir: Path,
    modalities: List[str],
    all_results: Dict[str, Any],
    elapsed: float,
) -> None:
    """Write a root-level summary for ``--modality both`` runs.

    Creates two files in *output_dir*:

    * ``both_mode_summary.json`` — machine-readable index with per-modality
      paths and key metrics.
    * ``analysis_report.md`` — human/AI-readable overview that links to each
      modality's ``analysis_report.md`` and shows a side-by-side comparison
      of the most important dynamics metrics.

    This prevents confusion when users run ``--modality both`` into the same
    directory as a previous ``joint`` or single-modality run: they now find
    a clear index rather than stale per-modality reports.
    """
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # ── JSON index ────────────────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "mode": "both",
        "run_timestamp": ts,
        "elapsed_seconds": round(elapsed, 1),
        "modalities_run": modalities,
        "results_directories": {
            mod: str(output_dir / mod) for mod in modalities
        },
        "per_modality_metrics": {},
    }

    _KEY_METRICS = [
        ("lyapunov", "mean_lyapunov"),
        ("lyapunov", "chaos_regime"),
        ("convergence", "distance_ratio"),
        ("convergence", "convergence_label"),
        ("dmd_spectrum", "spectral_radius"),
        ("stability", "fraction_converged"),
        ("stability", "fraction_limit_cycle"),
        ("stability", "fraction_unstable"),
    ]
    for mod in modalities:
        r = all_results.get(mod, {})
        if "error" in r:
            summary["per_modality_metrics"][mod] = {"error": r["error"]}
            continue
        m: Dict[str, Any] = {}
        for section, key in _KEY_METRICS:
            v = r.get(section)
            if isinstance(v, dict):
                raw = v.get(key)
                if isinstance(raw, dict):
                    raw = raw.get("regime", raw)  # chaos_regime → string
                m[f"{section}.{key}"] = raw
        summary["per_modality_metrics"][mod] = m

    try:
        with open(output_dir / "both_mode_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
    except Exception as exc:
        logger.warning("both-mode summary JSON write failed: %s", exc)

    # ── Markdown overview ─────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append("# Brain Dynamics Analysis — Both-Modality Mode\n")
    lines.append(f"**Generated**: {ts}  ")
    lines.append(f"**Total runtime**: {elapsed:.1f} s  ")
    lines.append(f"**Modalities analysed**: {', '.join(modalities)}  \n")
    lines.append(
        "> **Note**: Each modality was run independently through the full "
        "6-phase pipeline. Per-modality detailed reports are in the "
        "sub-directories listed below. This file provides a comparative "
        "overview only.\n"
    )

    lines.append("## Per-Modality Report Locations\n")
    for mod in modalities:
        rel = f"{mod}/analysis_report.md"
        lines.append(f"- **{mod.upper()}**: `{output_dir / mod}/analysis_report.md`")
    lines.append("")

    lines.append("## Side-by-Side Key Metrics\n")
    lines.append("| Metric | " + " | ".join(m.upper() for m in modalities) + " |")
    lines.append("|--------|" + "|".join("-----" for _ in modalities) + "|")

    _DISPLAY = [
        ("LLE (Rosenstein λ₁)",   "lyapunov.mean_lyapunov"),
        ("Chaos regime",           "lyapunov.chaos_regime"),
        ("Convergence ratio",      "convergence.distance_ratio"),
        ("Convergence label",      "convergence.convergence_label"),
        ("DMD spectral radius ρ",  "dmd_spectrum.spectral_radius"),
        ("Frac. converged",        "stability.fraction_converged"),
        ("Frac. limit-cycle",      "stability.fraction_limit_cycle"),
        ("Frac. unstable",         "stability.fraction_unstable"),
    ]
    for label, key in _DISPLAY:
        vals = []
        for mod in modalities:
            v = summary["per_modality_metrics"].get(mod, {}).get(key)
            if v is None:
                vals.append("N/A")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append(f"| {label} | " + " | ".join(vals) + " |")

    lines.append("\n## Interpretation Guide\n")
    lines.append(
        "When comparing modalities, expect **fMRI** and **EEG** to show "
        "different dynamics: fMRI captures slow hemodynamic signals (TR ≈ 2 s) "
        "and often exhibits convergence toward a limit-cycle attractor. EEG "
        "captures fast electrical activity and may show higher Lyapunov exponents "
        "and lower convergence. **This is normal and expected — not a bug.** "
        "The `--modality both` flag is designed precisely to reveal these "
        "differences.\n"
    )
    lines.append(
        "> For joint analysis of fMRI + EEG in a single state vector, use "
        "`--modality joint`.  Note that joint mode results can be scale-sensitive "
        "(see `pipeline_report.json → metadata → modality` in each sub-directory)."
    )

    md_text = "\n".join(lines)
    try:
        with open(output_dir / "analysis_report.md", "w", encoding="utf-8") as f:
            f.write(md_text)
        logger.info("  both-mode summary: %s", output_dir / "analysis_report.md")
    except Exception as exc:
        logger.warning("both-mode summary Markdown write failed: %s", exc)


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

    # ── Write a root-level summary so users always find the right files ────────
    _write_both_mode_summary(output_dir, modalities_to_run, all_results, elapsed)

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
