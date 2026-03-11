"""
narrative.py — Three-layer scientific narrative synthesis
==========================================================

Synthesises all Phase 1 and Phase 2 evidence into a structured three-layer
scientific narrative:

  Layer 1 — PHENOMENON
      What do we observe?
      The high-dimensional GNN collapses onto a low-dimensional, near-critical
      attractor.  Supported by: D₂, PCA n@90%, Rosenstein LLE, DMD ρ.

  Layer 2 — CAUSAL PROOF
      Is this emergent from the trained model?
      The attractor structure is NOT explained by input statistics, data
      distribution, or random network structure.
      Supported by: surrogate test, weight randomisation, input-dim control,
      node ablation, random network comparison.

  Layer 3 — MECHANISM HYPOTHESES
      What might generate this phenomenon?
      Based on available evidence (mode-node coupling, spectral structure,
      DMD modes), we propose specific, testable mechanism hypotheses.
      These are hypotheses, not claims — proving them requires additional data.

Output: a Markdown report (``analysis_narrative.md``) and a summary JSON
(``phase2_report.json``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main narrative builder
# ─────────────────────────────────────────────────────────────────────────────

def build_narrative(
    phase1_data: Dict[str, Any],
    phase2_results: Dict[str, Any],
    modality: str = "fmri",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Build the three-layer narrative from all available evidence.

    Parameters
    ----------
    phase1_data:
        Dict from ``loader.load_phase1_results``.
    phase2_results:
        Dict with keys ``mode_node_coupling``, ``attractor_fingerprint``,
        ``causal_chain`` (from ``analyses.py``).
    modality:
        Brain modality string used in the text ("fmri", "eeg", "joint").
    output_dir:
        If provided, writes ``analysis_narrative.md`` and ``phase2_report.json``.

    Returns
    -------
    dict with keys:
        layer1   dict   — Layer 1 phenomenon summary
        layer2   dict   — Layer 2 causal evidence summary
        layer3   dict   — Layer 3 mechanism hypotheses
        summary  str    — One-paragraph plain-English abstract
        confidence str  — "high" | "moderate" | "low"
        markdown str    — Full Markdown report text
    """
    results = phase1_data.get("results") or {}
    pr      = phase1_data.get("pipeline_report") or {}

    layer1 = _layer1_phenomenon(results, phase1_data, modality)
    layer2 = _layer2_causal(phase2_results.get("causal_chain", {}), results)
    layer3 = _layer3_mechanism(
        phase2_results.get("mode_node_coupling", {}),
        phase2_results.get("attractor_fingerprint", {}),
        results,
        phase1_data,
        modality,
    )

    # Overall confidence
    confidence = _overall_confidence(layer1, layer2, layer3)

    summary = _make_summary(layer1, layer2, layer3, modality, confidence)
    markdown = _render_markdown(layer1, layer2, layer3, summary, confidence,
                                modality, phase2_results)

    narrative: Dict[str, Any] = {
        "layer1":     layer1,
        "layer2":     layer2,
        "layer3":     layer3,
        "summary":    summary,
        "confidence": confidence,
        "markdown":   markdown,
    }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save Markdown report
        (output_dir / "analysis_narrative.md").write_text(
            markdown, encoding="utf-8"
        )
        # Save JSON (without large markdown field to keep it readable)
        json_out = {k: v for k, v in narrative.items() if k != "markdown"}
        (output_dir / "phase2_report.json").write_text(
            json.dumps(json_out, indent=2, default=_json_serial),
            encoding="utf-8",
        )
        logger.info("Phase 2 narrative written to %s", output_dir)

    return narrative


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1: Phenomenon
# ─────────────────────────────────────────────────────────────────────────────

def _layer1_phenomenon(
    results: Dict[str, Any],
    phase1_data: Dict[str, Any],
    modality: str,
) -> Dict[str, Any]:
    """Extract and summarise the phenomenon description metrics."""
    layer: Dict[str, Any] = {"name": "Phenomenon Description"}
    evidence: List[str] = []
    metrics: Dict[str, Any] = {}

    # ── State-space dimensionality ────────────────────────────────────────────
    pca = results.get("pca") or {}
    n90 = pca.get("n_components_90pct")
    N   = (results.get("spectral") or {}).get("n_regions")
    if N is None:
        trajs = phase1_data.get("trajectories")
        N     = int(trajs.shape[2]) if trajs is not None else None

    ad  = results.get("attractor_dimension") or {}
    d2      = ad.get("D2_mean", ad.get("D2"))
    d2_std  = ad.get("D2_std")
    ky_lin  = ad.get("KY_linearised")

    metrics["N_input"] = N
    metrics["D2"]      = round(float(d2), 3)  if d2  is not None else None
    metrics["D2_std"]  = round(float(d2_std), 3) if d2_std is not None else None
    metrics["pca_n90"] = n90
    metrics["KY_lin"]  = round(float(ky_lin), 3) if ky_lin is not None else None

    if N and d2:
        cr = float(d2) / float(N)
        metrics["collapse_ratio"] = round(cr, 5)
        evidence.append(
            f"**Dimensionality collapse**: The {N}-node {modality.upper()} system "
            f"collapses onto a D₂≈{d2:.2f}"
            + (f"±{d2_std:.2f}" if d2_std else "")
            + f"-dimensional attractor "
            f"(only {cr * 100:.1f}% of input space)."
        )
    elif n90:
        evidence.append(
            f"**Dimensionality collapse**: PCA n@90%={n90} dimensions explain "
            f"90% of trajectory variance in the {N or '?'}-dimensional system."
        )

    # ── Near-criticality (LLE) ────────────────────────────────────────────────
    lya    = results.get("lyapunov") or {}
    lle    = lya.get("mean_lyapunov")
    regime = (lya.get("chaos_regime") or {}).get("regime")
    metrics["rosenstein_lle"] = round(float(lle), 5) if lle is not None else None
    metrics["regime"]         = regime

    dmd    = results.get("dmd_spectrum") or {}
    rho    = dmd.get("spectral_radius")
    n_slow = dmd.get("n_slow_modes")
    n_hopf = dmd.get("n_hopf_pairs")
    metrics["dmd_rho"]    = round(float(rho), 4)  if rho is not None else None
    metrics["n_slow_dmd"] = n_slow
    metrics["n_hopf"]     = n_hopf

    if lle is not None:
        evidence.append(
            f"**Near-criticality**: Rosenstein LLE λ={lle:.5f} "
            f"(regime: {regime or '?'}). "
            + (f"DMD spectral radius ρ={rho:.4f}. " if rho else "")
        )

    # ── Spectral structure ────────────────────────────────────────────────────
    spec = results.get("spectral") or phase1_data.get("spectral_summary") or {}
    pr_val     = spec.get("participation_ratio")
    n_dom      = spec.get("n_dominant")
    gap_ratio  = spec.get("gap_ratio")
    metrics["PR"] = round(float(pr_val), 4) if pr_val is not None else None
    metrics["n_dominant"] = n_dom

    if pr_val and N:
        evidence.append(
            f"**Low-rank connectivity**: PR/N={float(pr_val)/float(N):.3f} "
            f"(participation ratio {float(pr_val):.1f} out of {N}); "
            + (f"{n_dom} dominant eigenvalue(s)." if n_dom else "")
        )

    # ── Convergence ───────────────────────────────────────────────────────────
    conv       = results.get("convergence") or {}
    dist_ratio = conv.get("distance_ratio")
    metrics["distance_ratio"] = round(float(dist_ratio), 4) if dist_ratio is not None else None

    if dist_ratio is not None:
        evidence.append(
            f"**Attractor coherence**: distance_ratio={dist_ratio:.4f} — "
            + ("trajectories converge to a single coherent attractor."
               if dist_ratio < 0.1 else
               "moderate convergence across trajectories."
               if dist_ratio < 0.5 else
               "weak convergence, possible multiple basins.")
        )

    # ── Oscillatory structure (PSD) ───────────────────────────────────────────
    psd = results.get("power_spectrum") or phase1_data.get("power_spectrum_report") or {}
    f_dom = psd.get("dominant_frequency_hz")
    if f_dom:
        metrics["dominant_frequency_hz"] = round(float(f_dom), 4)
        evidence.append(
            f"**Oscillations**: dominant frequency {f_dom:.4f} Hz in PSD — "
            "rhythmic dynamics on the attractor manifold."
        )

    if n_hopf:
        evidence.append(
            f"**DMD Hopf modes**: {n_hopf} Hopf oscillation pair(s) detected "
            "in the linearised transfer operator."
        )
    if n_slow:
        evidence.append(
            f"**Slow modes**: {n_slow} slow DMD mode(s) with near-neutral stability "
            "(long-relaxation-time directions on the manifold)."
        )

    layer["metrics"]  = metrics
    layer["evidence"] = evidence
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2: Causal Proof
# ─────────────────────────────────────────────────────────────────────────────

def _layer2_causal(
    causal: Dict[str, Any],
    results: Dict[str, Any],
) -> Dict[str, Any]:
    """Summarise the causal chain evidence."""
    layer: Dict[str, Any] = {"name": "Causal Proof of Emergence"}

    if not causal:
        layer["verdict"] = "insufficient_data"
        layer["summary"] = (
            "Causal chain analysis could not be run (missing Phase 1 data). "
            "Enable node_ablation, input_dimension_control, and "
            "weight randomisation in the dynamics_pipeline config."
        )
        layer["evidence_table"]  = []
        layer["emergence_score"] = None
        return layer

    layer["emergence_score"] = causal.get("emergence_score")
    layer["emergence_label"] = causal.get("emergence_label")
    layer["evidence_table"]  = causal.get("evidence_table", [])
    layer["n_supported"]     = causal.get("n_supported")
    layer["n_total"]         = causal.get("n_total")
    layer["summary"]         = causal.get("interpretation", "")

    # Short verdict
    sc = causal.get("emergence_score", 0.0) or 0.0
    if sc >= 0.75:
        layer["verdict"] = "strong_support"
    elif sc >= 0.50:
        layer["verdict"] = "moderate_support"
    elif sc >= 0.25:
        layer["verdict"] = "weak_support"
    else:
        layer["verdict"] = "insufficient_data"

    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Layer 3: Mechanism Hypotheses
# ─────────────────────────────────────────────────────────────────────────────

def _layer3_mechanism(
    mnc: Dict[str, Any],
    fp:  Dict[str, Any],
    results: Dict[str, Any],
    phase1_data: Dict[str, Any],
    modality: str,
) -> Dict[str, Any]:
    """
    Generate mechanism hypotheses from available structural evidence.

    These are HYPOTHESES, not claims.  Each hypothesis is marked with the
    evidence that motivates it and the additional data that would be needed
    to confirm or refute it.
    """
    layer: Dict[str, Any] = {"name": "Mechanism Hypotheses"}
    hypotheses: List[Dict[str, Any]] = []

    # ── H-M1: Spectral gap → attractor contraction ───────────────────────────
    spec    = results.get("spectral") or phase1_data.get("spectral_summary") or {}
    gap     = spec.get("gap_ratio")
    rho_W   = spec.get("spectral_radius")
    dmd     = results.get("dmd_spectrum") or {}
    n_slow  = dmd.get("n_slow_modes")

    if gap is not None and gap > 1.2:
        hypotheses.append({
            "id": "H-M1",
            "title": "Spectral gap drives attractor contraction",
            "evidence": [
                f"Connectivity spectral gap_ratio={gap:.2f} > 1.2 (dominant eigenvalue "
                "substantially separated from the rest).",
                f"{n_slow or '?'} slow DMD modes — directions where A ≈ I (near-neutral).",
            ],
            "hypothesis": (
                "The large gap between the leading and subleading eigenvalues of the "
                "connectivity matrix W creates a dominant contraction direction: "
                "almost all perturbations decay exponentially along the subspace "
                "spanned by subleading eigenvectors, while the leading eigenvector "
                "direction is nearly neutral.  This spectral gap is the geometric "
                "origin of the low-dimensional manifold — the system is 'funnelled' "
                "onto the slow subspace of W."
            ),
            "prediction": (
                "Perturbing the spectral gap (e.g., by scaling subleading eigenvalues "
                "upward) should increase D₂ proportionally.  This can be tested by "
                "modifying W before running Phase 1."
            ),
            "required_evidence": "Controlled spectral-gap manipulation experiment.",
            "confidence": "moderate",
        })

    # ── H-M2: Hub nodes as manifold scaffolding ───────────────────────────────
    na   = results.get("node_ablation") or {}
    mnc_nodes = mnc.get("global_top_nodes") or []
    mnc_dom   = mnc.get("dominant_node")

    if mnc_nodes or na.get("top_nodes"):
        top_nodes_str = (
            ", ".join(str(n) for n in mnc_nodes[:5])
            if mnc_nodes else "unknown (node ablation not run)"
        )
        hypotheses.append({
            "id": "H-M2",
            "title": "High-degree hub nodes scaffold the low-dimensional manifold",
            "evidence": [
                f"Nodes with highest DMD mode loading: [{top_nodes_str}].",
                "Node ablation (Phase 1) quantifies causal hub importance." if na else
                "Node ablation not yet run — DMD loading is circumstantial.",
            ],
            "hypothesis": (
                "A small set of hub nodes — those with the highest summed loading "
                "across all slow and Hopf DMD modes — act as scaffolding for the "
                "attractor manifold.  Their high in/out-degree in the connectivity "
                "matrix ensures that perturbations along their activity dimension "
                "propagate globally and dominate the slow dynamics.  Removing these "
                "nodes disrupts the manifold (Δλ, ΔD₂ larger than for random nodes)."
            ),
            "prediction": (
                "Ablating the top-5 DMD-loading nodes should produce a larger "
                "increase in D₂ and shift in LLE than ablating 5 randomly-chosen "
                "nodes.  This can be tested with run_node_ablation() in Phase 1."
            ),
            "required_evidence": "Node ablation experiment targeting DMD-loading hubs.",
            "confidence": "moderate" if mnc_nodes else "low",
        })

    # ── H-M3: Training pressure toward near-criticality ──────────────────────
    lle  = (results.get("lyapunov") or {}).get("mean_lyapunov")
    dmd_rho = dmd.get("spectral_radius")

    if lle is not None and abs(float(lle)) < 0.05:
        hypotheses.append({
            "id": "H-M3",
            "title": "Gradient-descent training pressure maintains near-criticality",
            "evidence": [
                f"Rosenstein LLE={float(lle):.5f} ≈ 0 (edge-of-chaos).",
                f"DMD spectral radius ρ={float(dmd_rho):.4f} ≈ 1." if dmd_rho else
                "DMD ρ unavailable.",
            ],
            "hypothesis": (
                "Long-horizon sequence prediction (the model's training objective) "
                "selectively rewards weight configurations where the spectral radius "
                "of the linearised dynamics is close to 1 (ρ ≈ 1, LLE ≈ 0).  "
                "This is because: (1) ρ < 1 → information erasure → poor long-range "
                "prediction; (2) ρ > 1 → exponential divergence → unpredictable "
                "futures.  The optimal predictor lives at the critical boundary.  "
                "Gradient descent on the prediction loss therefore implicitly "
                "self-organises the weight spectrum toward ρ ≈ 1."
            ),
            "prediction": (
                "A randomly-initialised (untrained) version of the same architecture "
                "should NOT show ρ ≈ 1 in free dynamics.  After training, ρ should "
                "consistently converge to ≈ 1 regardless of initialisation."
            ),
            "required_evidence": (
                "Compare Phase 1 dynamics of trained vs randomly-initialised weights. "
                "Access to training loss curve to check correlation between ρ and loss."
            ),
            "confidence": "speculative",
        })

    # ── H-M4: Hopf-oscillation origin in multi-scale connectivity ─────────────
    n_hopf = dmd.get("n_hopf_pairs", 0)
    comm   = results.get("community") or {}
    n_comm = comm.get("n_communities") or comm.get("k")

    if n_hopf and n_hopf > 0:
        hypotheses.append({
            "id": "H-M4",
            "title": "Hopf oscillations arise from multi-scale community interactions",
            "evidence": [
                f"{n_hopf} Hopf mode pair(s) in DMD linearised spectrum.",
                f"Community structure: {n_comm} communities detected." if n_comm else
                "Community structure not available.",
            ],
            "hypothesis": (
                "The Hopf oscillation pairs in the linearised DMD spectrum correspond "
                "to inter-community oscillatory modes.  When two communities of "
                "strongly-coupled nodes interact with a time-delay or phase-shifted "
                "coupling (as arises naturally in brain-scale connectivity), the "
                "linearised dynamics exhibits complex-conjugate eigenvalue pairs "
                "— the fingerprint of a Hopf bifurcation.  The oscillation frequency "
                f"({dmd.get('dominant_oscillation_hz', '?'):.4f} Hz) encodes the "
                f"inter-community coupling time scale."
                if dmd.get("dominant_oscillation_hz") else
                "The oscillation frequency encodes the inter-community coupling "
                "time scale."
            ),
            "prediction": (
                "Shuffling inter-community edges (while preserving intra-community "
                "edges) should abolish Hopf pairs.  Shuffling intra-community edges "
                "should reduce n_Hopf by less."
            ),
            "required_evidence": (
                "Targeted edge-shuffling experiment separating intra- vs "
                "inter-community connections."
            ),
            "confidence": "moderate" if n_comm else "low",
        })

    # ── H-M5: Attractor fingerprint stability as robustness evidence ──────────
    fp_score = fp.get("stability_score")
    fp_label = fp.get("stability_label")

    if fp_score is not None:
        hypotheses.append({
            "id": "H-M5",
            "title": "The attractor geometry is globally stable (not initial-condition-specific)",
            "evidence": [
                f"Attractor fingerprint stability score={fp_score:.3f} ({fp_label}): "
                "PCA subspace principal angles between different trajectory windows.",
            ],
            "hypothesis": (
                "The low-dimensional manifold is a **global** attractor of the trained "
                "model, not a local artefact of specific initial conditions.  "
                "The high subspace-angle similarity across temporal windows and "
                "independently-sampled trajectory groups indicates that all initial "
                "conditions fall into the basin of the same low-dimensional attractor."
            ),
            "prediction": (
                "Trajectories started from maximally diverse random initial conditions "
                "should all converge to the same PCA subspace within O(context_length) "
                "steps.  This is already confirmed by the fingerprint stability score."
            ),
            "required_evidence": "Already available — fingerprint score provides direct evidence.",
            "confidence": "high" if fp_label == "high" else "moderate",
        })

    layer["hypotheses"] = hypotheses
    layer["n_hypotheses"] = len(hypotheses)
    layer["note"] = (
        "These are mechanism HYPOTHESES motivated by structural evidence, "
        "not causal claims.  Confirming them requires the additional experiments "
        "listed under 'required_evidence' for each hypothesis."
    )
    return layer


# ─────────────────────────────────────────────────────────────────────────────
# Confidence / summary helpers
# ─────────────────────────────────────────────────────────────────────────────

def _overall_confidence(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    layer3: Dict[str, Any],
) -> str:
    n_ev1 = len(layer1.get("evidence", []))
    l2_sc = layer2.get("emergence_score")
    fp_ok = any(
        h.get("confidence") in ("high", "moderate")
        for h in (layer3.get("hypotheses") or [])
    )

    if n_ev1 >= 4 and l2_sc is not None and l2_sc >= 0.5 and fp_ok:
        return "high"
    elif n_ev1 >= 2 and (l2_sc is None or l2_sc >= 0.25):
        return "moderate"
    else:
        return "low"


def _make_summary(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    layer3: Dict[str, Any],
    modality: str,
    confidence: str,
) -> str:
    m1   = layer1.get("metrics") or {}
    N    = m1.get("N_input")
    d2   = m1.get("D2")
    lle  = m1.get("rosenstein_lle")
    regime = m1.get("regime")
    n90  = m1.get("pca_n90")
    sc   = layer2.get("emergence_score")
    n_hyp = (layer3.get("n_hypotheses") or 0)

    parts: List[str] = []
    if N and d2:
        parts.append(
            f"The trained TwinBrain {modality.upper()} model ({N} regions) "
            f"spontaneously collapses onto a low-dimensional attractor "
            f"(D₂≈{d2:.2f}; PCA n@90%={n90 or '?'})"
        )
    if lle is not None:
        parts.append(
            f"operating near the edge of chaos (λ={lle:.5f}, regime={regime})"
        )
    if sc is not None:
        em_str = "strong" if sc >= 0.75 else "moderate" if sc >= 0.5 else "weak"
        parts.append(
            f"This is an emergent property of the trained weights "
            f"({em_str} causal evidence, score={sc:.2f}), not attributable to "
            "input statistics or random structure"
        )
    if n_hyp:
        parts.append(
            f"We propose {n_hyp} mechanism hypothesis(es) linking spectral, "
            "community, and hub-node structure to the observed attractor"
        )
    parts.append(
        "Full mechanistic confirmation requires additional targeted experiments "
        "(see Layer 3 hypotheses)"
    )

    return ". ".join(parts) + "."


# ─────────────────────────────────────────────────────────────────────────────
# Markdown renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_markdown(
    layer1: Dict[str, Any],
    layer2: Dict[str, Any],
    layer3: Dict[str, Any],
    summary: str,
    confidence: str,
    modality: str,
    phase2_results: Dict[str, Any],
) -> str:
    lines: List[str] = []

    lines += [
        "# Phase 2 — Scientific Narrative Report",
        "",
        f"> **Overall confidence**: `{confidence}`",
        "",
        "---",
        "",
        "## Abstract",
        "",
        summary,
        "",
        "---",
        "",
    ]

    # ── Layer 1 ────────────────────────────────────────────────────────────────
    lines += [
        "## Layer 1: Phenomenon Description",
        "",
        "*What do we observe?*",
        "",
    ]
    for ev in layer1.get("evidence", []):
        lines.append(f"- {ev}")
    lines.append("")

    m1 = layer1.get("metrics") or {}
    if m1:
        lines += ["**Key metrics:**", "", "| Metric | Value |", "|--------|-------|"]
        for k, v in m1.items():
            if v is not None:
                lines.append(f"| {k} | {v} |")
        lines.append("")

    lines += ["---", ""]

    # ── Layer 2 ────────────────────────────────────────────────────────────────
    lines += [
        "## Layer 2: Causal Proof of Emergence",
        "",
        "*Is this emergent from the trained model, not from input or random structure?*",
        "",
    ]
    sc = layer2.get("emergence_score")
    lb = layer2.get("emergence_label")
    if sc is not None:
        lines.append(
            f"**Emergence score**: {sc:.3f} ({lb}) — "
            f"{layer2.get('n_supported', '?')}/{layer2.get('n_total', '?')} "
            "evidence items support emergence."
        )
        lines.append("")
    lines.append(layer2.get("summary", "") + "")
    lines.append("")

    et = layer2.get("evidence_table") or []
    if et:
        lines += [
            "| Evidence | Verdict | Detail |",
            "|----------|---------|--------|",
        ]
        for e in et:
            v_icon = {"supports": "✓", "neutral": "~", "refutes": "✗", "missing": "?"}.get(
                e.get("verdict", "?"), "?"
            )
            detail = str(e.get("detail", "")).replace("|", "\\|")
            lines.append(
                f"| **{e.get('name', '?')}** | {v_icon} {e.get('verdict', '?')} | {detail} |"
            )
        lines.append("")

    lines += ["---", ""]

    # ── Layer 3 ────────────────────────────────────────────────────────────────
    lines += [
        "## Layer 3: Mechanism Hypotheses",
        "",
        f"> {layer3.get('note', '')}",
        "",
    ]

    for h in layer3.get("hypotheses") or []:
        hid   = h.get("id", "?")
        title = h.get("title", "?")
        conf  = h.get("confidence", "?")
        lines += [
            f"### {hid}: {title}",
            "",
            f"**Confidence**: `{conf}`",
            "",
            "**Motivating evidence:**",
        ]
        for ev in h.get("evidence", []):
            lines.append(f"- {ev}")
        lines += [
            "",
            "**Hypothesis:**",
            "",
            h.get("hypothesis", ""),
            "",
            "**Prediction (testable):**",
            "",
            h.get("prediction", ""),
            "",
            f"**Required to confirm:** {h.get('required_evidence', '?')}",
            "",
            "---",
            "",
        ]

    # ── Attractor fingerprint ──────────────────────────────────────────────────
    fp = phase2_results.get("attractor_fingerprint") or {}
    if fp and not fp.get("error"):
        lines += [
            "## Attractor Fingerprint Stability",
            "",
            f"**Stability score**: {fp.get('stability_score', '?')} "
            f"({fp.get('stability_label', '?')})",
            "",
            fp.get("interpretation", ""),
            "",
            f"Mean cos(θ₁) = {fp.get('mean_cos_theta1', '?')} "
            f"± {fp.get('std_cos_theta1', '?')} across "
            f"{len(fp.get('pair_labels', []))} split pair(s).",
            "",
            "---",
            "",
        ]

    # ── Mode-node coupling summary ─────────────────────────────────────────────
    mnc = phase2_results.get("mode_node_coupling") or {}
    if mnc and not mnc.get("error"):
        top5 = (mnc.get("global_top_nodes") or [])[:5]
        lines += [
            "## DMD Mode–Node Coupling",
            "",
            f"**Slow/Hopf modes identified**: {mnc.get('n_slow_modes', '?')}",
            "",
            f"**Top-5 nodes by global DMD loading**: {top5}",
            "",
            "These nodes have the highest combined loading across all slow and Hopf "
            "DMD modes — they are the structural scaffolding of the low-dimensional "
            "manifold.  (See `mode_node_heatmap.png`.)",
            "",
            "---",
            "",
        ]

    lines += [
        "## Conclusion",
        "",
        f"**Overall confidence**: `{confidence}`",
        "",
        summary,
        "",
        "*Generated by `phase2_analysis` — Phase 2 Scientific Narrative Synthesis.*",
    ]

    return "\n".join(lines)


def _json_serial(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
