# phase2_analysis — Phase 2 Scientific Narrative Synthesis

> **Second-phase analysis that synthesises Phase 1 (dynamics_pipeline) results into a three-layer scientific story about emergent brain dynamics.**

---

## Purpose

Phase 1 (`dynamics_pipeline`) generates raw metrics — LLE, D₂, DMD, surrogate tests, etc.
Phase 2 synthesises those metrics into a structured scientific argument:

| Layer | Question | Evidence sources |
|-------|----------|-----------------|
| **1 — Phenomenon** | *What do we observe?* | D₂, PCA n@90%, Rosenstein LLE, DMD ρ, convergence, PSD |
| **2 — Causal proof** | *Is this emergent from the trained model?* | Surrogate test, weight randomisation, input-dim control, node ablation, random networks |
| **3 — Mechanism** | *What might generate it?* (hypotheses only) | DMD mode–node coupling, attractor fingerprint, spectral gap |

---

## Quick Start

```bash
# After running Phase 1:
python -m phase2_analysis.run \
    --phase1-dir outputs/dynamics_pipeline \
    --output     outputs/phase2

# Quick run (fewer splits, faster)
python -m phase2_analysis.run \
    --phase1-dir outputs/dynamics_pipeline --quick
```

**Outputs:**

```
outputs/phase2/
├── analysis_narrative.md       ← Human-readable 3-layer report
├── phase2_report.json          ← Machine-readable summary
├── mode_node_coupling.json     ← DMD mode × brain-node loading
├── mode_node_heatmap.png       ← Heatmap visualisation
├── attractor_fingerprint.json  ← Subspace stability scores
├── causal_chain.json           ← Causal evidence table
└── attractor_subspace_angles.png
```

---

## New Analyses

### 1. DMD Mode–Node Coupling

**Question:** *Which brain nodes drive the low-dimensional manifold?*

For each slow and Hopf mode of the DMD linearised transfer operator A,
computes the absolute eigenvector loading on every brain region. Nodes
with high summed loading across all modes are the structural scaffolding
of the attractor manifold.

- Input: `trajectories.npy` + `jacobian_dmd.npy` (Phase 1 outputs)
- Output: `mode_node_coupling.json`, `mode_node_heatmap.png`
- Key result: `global_top_nodes` — ranked list of hub nodes driving the manifold

### 2. Attractor Fingerprint Stability

**Question:** *Is the attractor geometry stable across different initial conditions?*

Splits trajectories into multiple temporal windows and trajectory groups,
fits a PCA subspace to each split, then computes principal subspace angles
between all pairs. A high stability score (cos θ₁ ≈ 1) means the system
reliably collapses onto the same low-dimensional manifold regardless of
starting point.

- Input: `trajectories.npy` (Phase 1 output)
- Output: `attractor_fingerprint.json`, `attractor_subspace_angles.png`
- Key result: `stability_score` ∈ [0, 1], `stability_label` (high/moderate/low)

### 3. Causal Chain Evaluation

**Question:** *Is the attractor emergent from trained weights?*

Reads five classes of causal evidence from Phase 1 JSON outputs and
synthesises them into a single `emergence_score` ∈ [0, 1]:

| ID | Evidence | Phase 1 source |
|----|----------|----------------|
| E1 | Surrogate test (real LLE > surrogate LLE) | `validation/surrogate_test.json` |
| E2 | Weight randomisation (LLE changes under weight shuffle) | `validation/analysis_comparison.json` |
| E3 | Input dimension control (D₂ stable under zero-input) | `pipeline_report.json` → `input_dimension_control` |
| E4 | Node ablation (hub removal disrupts attractor) | `pipeline_report.json` → `node_ablation` |
| E5 | Random network comparison (LLE lower than structurally-equivalent random networks) | `pipeline_report.json` → `graph_structure_comparison` |

**Emergence labels:**
- `strong` (score ≥ 0.75, n ≥ 3): all key tests passed
- `moderate` (score ≥ 0.50): most tests passed
- `weak` (score ≥ 0.25): mixed evidence
- `insufficient` (score < 0.25 or missing data): enable optional Phase 1 analyses

---

## How to Get Strong Causal Evidence

To get `emergence_score ≥ 0.75`, enable these optional analyses in
`dynamics_pipeline/config.yaml` **before** running Phase 1:

```yaml
validation:
  node_ablation:
    enabled: true          # E4
  input_dimension_control:
    enabled: true          # E3
  structural_perturbation:
    enabled: true          # E2
  graph_structure_comparison:
    enabled: true          # E5
```

---

## Options

```
python -m phase2_analysis.run --help

  --phase1-dir PATH     Phase 1 output directory (required)
  --output PATH         Phase 2 output directory (default: <phase1-dir>/phase2/)
  --modality STR        fmri | eeg | joint | both  (default: fmri)
  --skip [NAMES...]     Skip specific analyses
  --n-components INT    PCA components for fingerprint (default: 5)
  --n-splits INT        Temporal splits for fingerprint (default: 4)
  --n-top-nodes INT     Top nodes in coupling output (default: 15)
  --burnin INT          Trajectory burnin steps (default: 0)
  --quick               Fast run (fewer splits, fewer nodes)
  --verbose             DEBUG logging
```

---

## Scientific Notes

### Layer 3 hypotheses are HYPOTHESES, not claims

Phase 2 explicitly labels Layer 3 outputs as **mechanism hypotheses**,
not causal claims. Each hypothesis includes:
- **Motivating evidence** (what we currently see)
- **The hypothesis** (what mechanism could explain it)
- **A testable prediction** (what experiment would confirm/refute it)
- **Required evidence** (what additional data is needed)

This is by design: with only model-output trajectories, we can describe
the phenomenon and prove it is model-emergent, but we cannot fully
explain *why* the model generates it (that requires training-time data,
weight gradients, or architectural experiments).

### Relationship to Phase 1

Phase 2 does **not** re-run Phase 1 analyses. It reads Phase 1 artefacts and
runs three new analyses:
- Mode–node coupling is new (not in Phase 1)
- Attractor fingerprint stability is new (Phase 1 only tests convergence, not subspace identity)
- Causal chain synthesis reads Phase 1 JSON outputs and scores them

---

## Two-Phase Analysis Summary

```
Phase 1 (dynamics_pipeline)          Phase 2 (phase2_analysis)
─────────────────────────────────    ─────────────────────────────────────
Generate trajectories              →  Load & synthesise
Characterise dynamics (LLE, DMD)   →  Mode-node coupling (new)
Validate (surrogate, ablation)     →  Attractor fingerprint (new)
Synthesise H1–H5 verdicts          →  Causal chain + 3-layer narrative (new)
pipeline_report.json               →  phase2_report.json + analysis_narrative.md
```
