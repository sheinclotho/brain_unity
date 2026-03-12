# dynamics_pipeline — Unified Brain Dynamics Analysis (Phase 1)

> **Configuration-driven, six-phase pipeline for TwinBrain model dynamics.**
> Combines `brain_dynamics` analysis tools into a single configurable run.
>
> Phase 1 generates trajectories and characterises dynamics.
> **Phase 2** (`phase2_analysis/`) synthesises results into a scientific narrative.

---

## Quick Start

```bash
# Full analysis (fMRI, default)
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt

# Quick exploratory run
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick

# Specific phases only
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 3

# EEG / joint / both modalities
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality eeg
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality joint
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality both

# Then run Phase 2 for the scientific narrative:
python -m phase2_analysis.run --phase1-dir outputs/dynamics_pipeline
```

---

## Scientific Goals

Five testable hypotheses about the trained TwinBrain model:

| Hypothesis | Question | Key analyses |
|-----------|----------|-------------|
| **H1** | Low-rank connectivity structure? | Spectral radius ρ, PR, n_dominant, gap_ratio |
| **H2** | Low-dimensional dynamics? | D₂ (Grassberger-Procaccia), K-Y_lin (DMD), PCA n@90% |
| **H3** | Near-critical operating point? | Rosenstein λ₁, DMD ρ, CSD (critical slowing down) |
| **H4** | Brain-like oscillations? | PSD dominant frequency, DMD Hopf pairs |
| **H5** | Metabolic constraint → criticality? | Energy-budget L1 projection experiment |

---

## Pipeline Architecture

```
Phase 1: Data Generation
    1a  Free dynamics trajectories  (n_init × T × N)
    1b  Response matrix            (N × N causal effects)
    ↓
Phase 2: Network Structure
    2a  Spectral decomposition      ρ, PR, gap_ratio → H1
    2b  Community detection         Q, k communities → H3
    2c  Hierarchical structure      hierarchy_index  → H3
    2d  Modal energy                n@90% → H2
    2e  Connectivity visualisation
    ↓
Phase 3: Dynamics Characterisation
    3a  Stability classification    fixed_point / limit_cycle / chaos
    3b  Attractor analysis          KMeans basins
    3c  Trajectory convergence      distance_ratio
    3d  Lyapunov exponent (Rosenstein λ₁)  → H3  ← ONLY nonlinear chaos criterion
    3e  Linearised spectrum (DMD)   ρ_DMD, n_slow, n_Hopf → complementary structure
    3f  Power spectrum              f_dom → H4
    3g  PCA dimension               n@90% → H2
    3h  Attractor dimension         D₂, K-Y_lin, PCA n@90% → H2
    ↓
Phase 4: Statistical Validation
    4a  Surrogate test              phase-rand / shuffle / AR(1)
    4b  Random network comparison   3 spectral radii × 5 seeds
    4c  Embedding dimension         FNN, D₂, Takens
    4d  Structural perturbation     weight shuffle / degree-preserving rewire
    4e  TwoNN intrinsic dimension   local manifold dimension
    4j  Graph-structure comparison  brain vs degree-preserved vs fully-random
    4k  Input dimension control     no-input / high-dim noise / 3-D drive
    4l  Node ablation               ΔLLE, ΔD₂, Procrustes distance per node
    4m  Predictive dimension        VAR(1) AIC/BIC elbow vs D₂
    ↓
Phase 5: Advanced (optional, mostly disabled by default)
    5a  Virtual stimulation
    5b  Energy-budget experiment     → H5
    5c  Phase diagram (g-scan)
    5d  Controllability
    5e  Information flow (TE)
    5f  Critical slowing down
    5g  Dynamic lesion              (requires 4l results)
    5h  Granger causality
    ↓
Phase 6: Synthesis
    Rosenstein LLE vs DMD consistency + nonlinearity index
    Surrogate cross-validation
    H1–H5 verdicts (SUPPORTED / NOT_SUPPORTED / INSUFFICIENT_DATA)
    pipeline_report.json + analysis_report.md
```

> **4j–4m and 5g–5h operate on Phase 1 trajectories — no extra model calls.**

---

## Key Methodological Choices

### Lyapunov: Rosenstein only (Wolf/FTLE removed)

Rosenstein LLE (λ₁) is the **only reliable nonlinear chaos criterion** in this pipeline.

- **Wolf-GS removed**: TwinBrain's ST-GCN encoder shares 199/200 of the context window
  across all k perturbation directions → all λᵢ converge to the same value (context dilution).
  Code deleted from `b_lyapunov_spectrum.py` and `run_dynamics_analysis.py Step 15`.
- **FTLE removed from default**: `rollout(steps=1)` shares 199/200 of context history →
  effective ε ≈ ε/200, triggering numerical floor for stable systems. Removed from default
  method; used only as explicit override.
- **Rosenstein**: operates on trajectory data directly, zero extra model calls, no context dilution.

### DMD: linearised structure, not chaos criterion

DMD fits the optimal linear transfer operator `A: x(t+1) ≈ A·x(t)` from trajectory pairs.
Its eigenvalues give the **linearised** Lyapunov spectrum — a structural complement, not
a replacement for Rosenstein:

| Metric | Source | Scientific role |
|--------|--------|----------------|
| Rosenstein λ₁ | trajectory data | Nonlinear chaos criterion (authoritative) |
| DMD ρ | trajectory data | Linearised stability; Hopf/slow-mode structure |
| K-Y_lin | DMD eigenvalues | Linear estimate of attractor dimension |
| Nonlinearity index Δ | both | `\|λ_Rosen - λ_DMD_max\| / \|λ_Rosen\|`; Δ > 1 → DMD results marked as reference only |

### Attractor dimension: three-way estimate

D₂ (nonlinear) **>** K-Y_lin (linearised) **>** PCA n@90% (linear upper bound).
Phase 3h reports all three. D₂ uses multi-trajectory, multi-window distributions with
explicit quality metrics (r², fail rate).

---

## Modalities

| Mode | Description | Output |
|------|-------------|--------|
| `fmri` | fMRI BOLD (default) | flat `output_dir/` |
| `eeg` | EEG | flat `output_dir/` |
| `both` | Run pipeline twice (fMRI then EEG) | `output_dir/fmri/` + `output_dir/eeg/` |
| `joint` | Concatenated z-score state vector [z_fmri ∥ z_eeg] | flat `output_dir/` |

**`joint` mode**: Lyapunov method automatically forced to `rosenstein`
(Wolf/FTLE require bounded state space; z-score is unbounded).

---

## Configuration

All parameters in `config.yaml`. CLI flags override config:

```yaml
simulator:
  modality: "fmri"            # fmri | eeg | joint | both
  normalize_amplitude: false  # joint mode amplitude diagnostics only

data_generation:
  n_init: 100        # trajectories (quick: 20)
  steps:  500        # steps per trajectory (quick: 100)
  seed:   42

dynamics:
  lyapunov:
    method: "rosenstein"   # always rosenstein; joint forces this automatically
    n_segments: 3          # three-segment sampling for robustness
  dmd_spectrum:
    enabled: true
  attractor_dimension:
    enabled: true
    d2:
      k_traj: 10         # trajectories for D₂ distribution
      m_windows: 5        # windows per trajectory
      min_r2: 0.90        # minimum fit quality

validation:
  surrogate_test:
    enabled: true
    n_surrogates: 19   # → p < 0.05 rank test (quick: 9)
  node_ablation:          { enabled: false }   # optional (slow)
  input_dimension_control: { enabled: false }   # optional
```

### Quick mode

`--quick` reduces: n_init 100→20, steps 500→100, n_surrogates 19→9, corr_dim disabled.

---

## Output Directory

```
outputs/dynamics_pipeline/
├── trajectories.npy              # (n_init, T, N)
├── response_matrix.npy           # (N, N)
├── structure/
│   ├── spectral_summary_*.json
│   └── community_structure_*.json
├── dynamics/
│   ├── lyapunov_report.json
│   ├── jacobian_report.json      # DMD results
│   ├── jacobian_dmd.npy          # DMD operator A (N×N) ← used by Phase 2
│   ├── jacobian_eigenvalues.npy  # complex eigenvalues
│   └── *.png
├── validation/
│   ├── surrogate_test.json
│   └── analysis_comparison.json
├── advanced/
└── pipeline_report.json          # full H1–H5 verdicts
```

---

## Known Issues & Design Decisions

### Attractor ambiguity threshold

When `initial_std < 0.087` (0.3 × random baseline), distance_ratio may
decrease because trajectories converge onto a **chaotic attractor** (not to a
fixed point). The convergence test cannot distinguish these cases. Resolution:
check Rosenstein λ₁ sign — `λ > 0` with `distance_ratio < 0.1` → chaotic
attractor attraction, not stability.

### FC vs response matrix connectivity

Community detection, spectral analysis, and E5 phase diagrams work best with
the **response matrix** (A-grade causal evidence). If unavailable, fall back
to Pearson FC (C-grade). The `connectivity_source` field in reports records
which was used; Phase 6 automatically downgrades H1 confidence when FC is used.

### Wolf-GS context dilution (historical)

Wolf-GS and `b_lyapunov_spectrum.py` have been permanently deleted (not disabled).
The Wolf-Benettin fallback in `wc_dynamics.py` is for Wilson-Cowan systems only.
DMD is the structural complement; Rosenstein is the LLE authority.

---

## What Phase 2 Adds

After Phase 1 completes, run `phase2_analysis` to get:

- **DMD mode–node coupling**: which brain regions drive each slow/Hopf mode
- **Attractor fingerprint stability**: PCA subspace angle consistency across windows
- **Causal chain evaluation**: structured evidence table (E1–E5) + emergence score
- **Three-layer narrative report**: phenomenon → causal proof → mechanism hypotheses

```bash
python -m phase2_analysis.run --phase1-dir outputs/dynamics_pipeline
# → outputs/dynamics_pipeline/phase2/analysis_narrative.md
```

See `phase2_analysis/README.md` for details.
