"""
run.py — Phase 2 entry point
==============================

Reads Phase 1 (dynamics_pipeline) output and runs the three-layer
scientific narrative synthesis:

  1. DMD Mode–Node Coupling
  2. Attractor Fingerprint Stability
  3. Causal Chain Evaluation
  4. Three-layer narrative synthesis → analysis_narrative.md + phase2_report.json

Usage::

    # From repository root
    python -m phase2_analysis.run \\
        --phase1-dir outputs/dynamics_pipeline \\
        --output     outputs/phase2

    # Quick mode (faster, fewer components)
    python -m phase2_analysis.run \\
        --phase1-dir outputs/dynamics_pipeline --quick

    # Skip specific analyses
    python -m phase2_analysis.run \\
        --phase1-dir outputs/dynamics_pipeline \\
        --skip mode_node_coupling attractor_fingerprint

    # Direct invocation
    python phase2_analysis/run.py --phase1-dir outputs/dynamics_pipeline

Outputs::

    <output_dir>/
    ├── analysis_narrative.md     # Human-readable 3-layer report (Markdown)
    ├── phase2_report.json        # Machine-readable summary
    ├── mode_node_coupling.json   # DMD mode × brain-node coupling
    ├── mode_node_heatmap.png     # Heatmap visualisation
    ├── attractor_fingerprint.json
    └── attractor_subspace_angles.png
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2_analysis")

# Add brain_dynamics to path for the sub-analyses
def _ensure_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [
        repo_root / "brain_dynamics",
        repo_root,
    ]:
        s = str(candidate)
        if s not in sys.path:
            sys.path.insert(0, s)


_ensure_path()


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(
    phase1_dir: Path,
    output_dir: Optional[Path] = None,
    modality: str = "fmri",
    skip: Optional[Set[str]] = None,
    n_components: int = 5,
    n_splits: int = 4,
    n_top_nodes: int = 15,
    burnin: int = 0,
    quick: bool = False,
) -> Dict[str, Any]:
    """
    Run the full Phase 2 analysis and return a results dict.

    Parameters
    ----------
    phase1_dir:
        Path to the Phase 1 output directory (from dynamics_pipeline).
    output_dir:
        Where to write Phase 2 outputs.  Defaults to
        ``<phase1_dir>/phase2/``.
    modality:
        Brain modality label for the narrative ("fmri", "eeg", "joint").
    skip:
        Set of analysis names to skip: {"mode_node_coupling",
        "attractor_fingerprint", "causal_chain"}.
    n_components:
        PCA components for the fingerprint analysis.
    n_splits:
        Number of temporal splits for fingerprint stability.
    n_top_nodes:
        Number of top-ranked nodes in mode-node coupling output.
    burnin:
        Trajectory burnin steps (passed to all analyses).
    quick:
        If True, reduces n_splits and n_top_nodes for faster runs.

    Returns
    -------
    dict with keys:
        mode_node_coupling     dict
        attractor_fingerprint  dict
        causal_chain           dict
        narrative              dict  (layer1, layer2, layer3, summary, markdown)
    """
    from phase2_analysis.loader import load_phase1_results
    from phase2_analysis.analyses import mode_node_coupling, attractor_fingerprint, causal_chain
    from phase2_analysis.narrative import build_narrative

    skip = skip or set()
    if quick:
        n_splits    = min(n_splits, 2)
        n_top_nodes = min(n_top_nodes, 10)

    if output_dir is None:
        output_dir = phase1_dir / "phase2"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Phase 2 Scientific Narrative Synthesis")
    logger.info("  Phase 1 dir : %s", phase1_dir)
    logger.info("  Output dir  : %s", output_dir)
    logger.info("=" * 60)

    # ── Load Phase 1 artefacts ────────────────────────────────────────────────
    logger.info("[Load] Reading Phase 1 artefacts …")
    data = load_phase1_results(phase1_dir)

    trajs   = data.get("trajectories")
    dmd_op  = data.get("dmd_operator")
    dmd_eig = data.get("dmd_eigenvalues")

    if trajs is None:
        logger.warning(
            "  trajectories.npy not found in %s — "
            "attractor_fingerprint and mode_node_coupling will be limited.",
            phase1_dir,
        )
    if dmd_op is None:
        logger.info(
            "  dynamics/jacobian_dmd.npy not found — "
            "DMD operator will be re-estimated from trajectories."
        )
    if dmd_eig is None and dmd_op is not None:
        logger.info(
            "  dynamics/jacobian_eigenvalues.npy not found — "
            "eigenvalues will be recomputed from the DMD operator."
        )

    phase2: Dict[str, Any] = {}

    # ── 1. DMD Mode–Node Coupling ─────────────────────────────────────────────
    if "mode_node_coupling" not in skip and trajs is not None:
        logger.info("[1/3] DMD Mode–Node Coupling …")
        try:
            mnc = mode_node_coupling(
                trajectories=trajs,
                dmd_operator=dmd_op,
                dmd_eigenvalues=dmd_eig,
                jacobian_report=data.get("jacobian_report"),
                n_top_nodes=n_top_nodes,
                burnin=burnin,
                output_dir=output_dir,
            )
            phase2["mode_node_coupling"] = mnc
            n_modes = mnc.get("n_slow_modes", 0)
            dom     = mnc.get("dominant_node")
            logger.info(
                "  Done: %d slow/Hopf modes identified; dominant node = %s",
                n_modes, dom,
            )
        except Exception as exc:
            logger.exception("  mode_node_coupling failed: %s", exc)
            phase2["mode_node_coupling"] = {"error": str(exc)}
    else:
        if trajs is None:
            phase2["mode_node_coupling"] = {"error": "trajectories_not_found"}
        else:
            phase2["mode_node_coupling"] = {"skipped": True}

    # ── 2. Attractor Fingerprint ───────────────────────────────────────────────
    if "attractor_fingerprint" not in skip and trajs is not None:
        logger.info("[2/3] Attractor Fingerprint Stability …")
        try:
            fp = attractor_fingerprint(
                trajectories=trajs,
                n_components=n_components,
                burnin=burnin,
                n_splits=n_splits,
                output_dir=output_dir,
            )
            phase2["attractor_fingerprint"] = fp
            logger.info(
                "  Done: stability_score=%.4f (%s)",
                fp.get("stability_score", 0.0),
                fp.get("stability_label", "?"),
            )
        except Exception as exc:
            logger.exception("  attractor_fingerprint failed: %s", exc)
            phase2["attractor_fingerprint"] = {"error": str(exc)}
    else:
        if trajs is None:
            phase2["attractor_fingerprint"] = {"error": "trajectories_not_found"}
        else:
            phase2["attractor_fingerprint"] = {"skipped": True}

    # ── 3. Causal Chain ────────────────────────────────────────────────────────
    if "causal_chain" not in skip:
        logger.info("[3/3] Causal Chain Evaluation …")
        try:
            cc = causal_chain(phase1_data=data, output_dir=output_dir)
            phase2["causal_chain"] = cc
            logger.info(
                "  Done: emergence_score=%.3f (%s) — %d/%d items support emergence",
                cc.get("emergence_score", 0.0),
                cc.get("emergence_label", "?"),
                cc.get("n_supported", 0),
                cc.get("n_total", 0),
            )
        except Exception as exc:
            logger.exception("  causal_chain failed: %s", exc)
            phase2["causal_chain"] = {"error": str(exc)}
    else:
        phase2["causal_chain"] = {"skipped": True}

    # ── 4. Narrative synthesis ─────────────────────────────────────────────────
    logger.info("[4/4] Building 3-layer narrative …")
    try:
        narrative = build_narrative(
            phase1_data=data,
            phase2_results=phase2,
            modality=modality,
            output_dir=output_dir,
        )
        phase2["narrative"] = narrative
        logger.info(
            "  Narrative written (confidence=%s); Markdown → %s",
            narrative.get("confidence"),
            output_dir / "analysis_narrative.md",
        )
    except Exception as exc:
        logger.exception("  Narrative synthesis failed: %s", exc)
        phase2["narrative"] = {"error": str(exc)}

    # ── Done ───────────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2 complete.  Outputs in %s", output_dir)
    logger.info("=" * 60)

    return phase2


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Phase 2: Scientific Narrative Synthesis for brain dynamics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--phase1-dir", required=True, type=Path,
        help="Path to Phase 1 (dynamics_pipeline) output directory.",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Phase 2 output directory (default: <phase1-dir>/phase2/).",
    )
    p.add_argument(
        "--modality", default="fmri",
        choices=["fmri", "eeg", "joint", "both"],
        help="Brain modality label (used in narrative text). Default: fmri.",
    )
    p.add_argument(
        "--skip", nargs="*", default=[],
        choices=["mode_node_coupling", "attractor_fingerprint", "causal_chain"],
        help="Analyses to skip.",
    )
    p.add_argument(
        "--n-components", type=int, default=5,
        help="PCA components for fingerprint stability (default: 5).",
    )
    p.add_argument(
        "--n-splits", type=int, default=4,
        help="Temporal splits for fingerprint stability (default: 4).",
    )
    p.add_argument(
        "--n-top-nodes", type=int, default=15,
        help="Top-ranked nodes in mode-node coupling output (default: 15).",
    )
    p.add_argument(
        "--burnin", type=int, default=0,
        help="Trajectory burnin steps (default: 0).",
    )
    p.add_argument(
        "--quick", action="store_true",
        help="Reduce n_splits and n_top_nodes for a faster run.",
    )
    p.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable DEBUG logging.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> None:
    p = _build_parser()
    args = p.parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    phase1_dir = args.phase1_dir
    if not phase1_dir.exists():
        p.error(f"Phase 1 directory not found: {phase1_dir}")

    run_phase2(
        phase1_dir=phase1_dir,
        output_dir=args.output,
        modality=args.modality,
        skip=set(args.skip or []),
        n_components=args.n_components,
        n_splits=args.n_splits,
        n_top_nodes=args.n_top_nodes,
        burnin=args.burnin,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
