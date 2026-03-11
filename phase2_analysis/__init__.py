"""
phase2_analysis — Phase 2 Scientific Narrative Synthesis
=========================================================

Reads Phase 1 (dynamics_pipeline) output and synthesises a three-layer
scientific narrative:

  Layer 1 — Phenomenon description:
      What do we observe?  (low-dimensional attractor, near-criticality)

  Layer 2 — Causal proof:
      Is this emergent from the trained model, not from input or noise?
      (surrogate test, weight randomisation, input-dim control, node ablation)

  Layer 3 — Mechanism hypotheses:
      Which internal structures plausibly generate the phenomenon?
      (DMD mode–node coupling, attractor subspace stability, spectral structure)

Entry point::

    python -m phase2_analysis.run \\
        --phase1-dir outputs/dynamics_pipeline \\
        --output    outputs/phase2

Or import directly::

    from phase2_analysis.run import run_phase2
    results = run_phase2(phase1_dir=Path("outputs/dynamics_pipeline"),
                         output_dir=Path("outputs/phase2"))
"""
