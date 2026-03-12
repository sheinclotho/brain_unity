#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI entry point for the unified dynamics analysis pipeline.

Usage::

    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt
    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \\
        --phases 1 2 3 --n-init 50 --steps 200
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

# Allow running this file directly (e.g. ``python dynamics_pipeline/run.py``)
# in addition to the standard module invocation
# (``python -m dynamics_pipeline.run``).  The relative import below requires
# ``__package__`` to be set, which Python only sets automatically for the
# module-invocation style.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "dynamics_pipeline"

from .pipeline import run_pipeline

logger = logging.getLogger("dynamics_pipeline")

# Stems/prefixes of training checkpoint files that must NOT be treated as graph caches.
_CHECKPOINT_STEMS: frozenset = frozenset({"best_model", "swa_model"})
_CHECKPOINT_PREFIX: str = "checkpoint_epoch_"


def _find_graph_pts(folder: Path) -> List[Path]:
    """Return a sorted list of graph-cache ``.pt`` files inside *folder*.

    Training checkpoints (``best_model.pt``, ``swa_model.pt``,
    ``checkpoint_epoch_*.pt``) are excluded automatically.
    """
    result: List[Path] = []
    for f in sorted(folder.glob("*.pt")):
        stem = f.stem
        if stem not in _CHECKPOINT_STEMS and not stem.startswith(_CHECKPOINT_PREFIX):
            result.append(f)
    return result

# ── Default configuration ─────────────────────────────────────────────────────

_DEFAULTS = {
    "model": {
        "path": None,
        "graph_path": None,
        "config_path": None,
        "device": "auto",
        # Number of cross-modal (EEG→fMRI) edges added per EEG electrode when
        # rebuilding graph structure for inference (see API.md §2.5).
        "k_cross_modal": 5,
    },
    "simulator": {
        # Which modality to analyse: 'fmri', 'eeg', 'both', or 'joint'.
        # - 'fmri'  : analyse the fMRI BOLD stream only.
        # - 'eeg'   : analyse the EEG stream only.
        # - 'both'  : run the full pipeline independently for each modality,
        #             outputting separate results under output_dir/fmri/ and
        #             output_dir/eeg/.  Requires both node types in the graph.
        # - 'joint' : single pipeline run using a COMBINED fMRI+EEG state
        #             vector (z-score normalised, then concatenated).
        #             Requires both 'fmri' and 'eeg' nodes in the graph cache.
        "modality": "fmri",
        # Internal fMRI sub-sampling factor (kept for compatibility; model
        # infers dt from graph).
        "fmri_subsample": 25,
    },
    "data_generation": {
        "n_init": 100,
        "steps": 500,
        "seed": 42,
        # Number of sliding context windows used for free-dynamics rollouts.
        # None = auto-compute from trajectory length.  Set explicitly to limit
        # the number of distinct historical contexts sampled.
        "n_temporal_windows": None,
    },
    "network_structure": {
        "enabled": True,
        "spectral": {"enabled": True},
        "community": {"enabled": True, "k_range": [3, 4, 5, 6, 7, 8]},
        "hierarchy": {"enabled": False},
        "modal_energy": {"enabled": True},
        "visualization": {"enabled": True},
    },
    "dynamics": {
        "stability": {"enabled": True, "delay_dt": 50},
        "attractor": {"enabled": True, "tail_steps": 50, "k_candidates": [2, 3, 4, 5, 6]},
        "convergence": {"enabled": True, "n_pairs": 50},
        "lyapunov": {
            "enabled": True,
            # Method: 'rosenstein' (default, zero extra model calls, no
            # context-dilution bias), 'wolf', 'ftle', or 'both'.
            # joint modality always forces 'rosenstein'.
            "method": "rosenstein",
            "max_lag": 50, "min_sep": 20, "n_segments": 3,
            "convergence_threshold": 0.05,
            "delay_embed_dim": 0, "delay_embed_tau": 1,
            # Parallel workers for Wolf/FTLE (1 = sequential).
            # Keep at 1 for GPU inference to avoid VRAM contention.
            "n_workers": 1,
        },
        "dmd_spectrum": {"enabled": True, "tail_steps": 20, "n_states": 3},
        "power_spectrum": {"enabled": True},
        "pca": {"enabled": True},
        "attractor_dimension": {"enabled": True},
    },
    "validation": {
        "surrogate_test": {
            "enabled": True, "n_surrogates": 19,
            "types": ["phase_randomize", "shuffle", "ar"], "n_traj_sample": 5,
        },
        "random_comparison": {
            "enabled": True, "spectral_radii": [0.9, 1.5, 2.0], "n_seeds": 5,
        },
        "embedding_dimension": {
            "enabled": True, "fnn_max_dim": 8, "corr_dim": True,
            "check_leakage": True, "train_fraction": 0.7,
        },
        "graph_structure_comparison": {
            "enabled": True, "n_random": 5,
        },
        "input_dimension_control": {
            "enabled": True, "noise_sigma": 0.5, "low_dim_k": 3,
        },
        "node_ablation": {
            "enabled": True, "n_top_variance": 50, "n_random_sample": 50,
        },
        "perturbation": {"enabled": False},
    },
    "advanced": {
        "stimulation": {
            "enabled": True, "target_nodes": [0, 100], "amplitude": 0.5,
            "frequency": 10.0, "duration": 30, "patterns": ["sine"],
        },
        "response_matrix": {"enabled": True, "n_nodes": 10, "stim_amplitude": 0.5},
        "energy_constraint": {"enabled": True},
        "phase_diagram": {"enabled": False},
        "controllability": {"enabled": False},
        "information_flow": {"enabled": False, "n_source_regions": 20, "n_target_regions": 20},
        "critical_slowing_down": {"enabled": False},
    },
    "output": {
        "directory": "outputs/dynamics_pipeline",
        "save_trajectories": True,
        "save_plots": True,
    },
}

_QUICK_OVERRIDES = {
    "data_generation": {"n_init": 20, "steps": 100},
    "advanced": {"response_matrix": {"n_nodes": 10}},
    "validation": {
        "surrogate_test": {"n_surrogates": 9, "n_traj_sample": 3},
        "embedding_dimension": {"fnn_max_dim": 4, "corr_dim": False},
    },
    # n_segments=1: trades robustness (3-segment early/mid/late sampling) for
    # speed. Acceptable in quick mode since with only 100 steps, segments would
    # overlap heavily anyway. Full runs should use n_segments=3.
    "dynamics": {"lyapunov": {"n_segments": 1}},
}


def _merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (ImportError, FileNotFoundError):
        return {}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified Brain Dynamics Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full run:    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt
  Quick run:   python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
  Phases 1-3:  python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 2 3
  EEG mode:    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality eeg
  Both modes:  python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality both
  Joint mode:  python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality joint
  Folder mode: python -m dynamics_pipeline.run --model best_model.pt --graph outputs/graph_cache/
        """,
    )
    p.add_argument("--model", required=True, help="Trained model checkpoint (.pt)")
    p.add_argument(
        "--graph",
        required=True,
        help=(
            "Graph cache file (.pt) **or** a folder containing multiple graph-cache "
            ".pt files.  When a folder is given, all .pt files in it (sorted "
            "alphabetically, training checkpoints excluded) are used as context "
            "sources for free-dynamics trajectories.  The first file becomes the "
            "primary graph (used for the response matrix and all other analyses); "
            "the remaining files provide additional context windows so that "
            "n_init trajectories cycle through diverse historical recordings even "
            "when each individual file has T ≤ context_length."
        ),
    )
    p.add_argument("--config", type=Path, default=None, help="YAML config override")
    p.add_argument("--training-config", default=None, help="Training config.yaml")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument(
        "--modality",
        choices=["fmri", "eeg", "both", "joint"],
        default=None,
        help=(
            "Which brain-data stream(s) to analyse: "
            "'fmri' (default), 'eeg', "
            "'both' (run pipeline twice, once per modality, into output/<modality>/), "
            "'joint' (combined fMRI+EEG z-score state vector)."
        ),
    )
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    p.add_argument("--n-init", type=int, default=None, help="Number of trajectories")
    p.add_argument("--steps", type=int, default=None, help="Steps per trajectory")
    p.add_argument("--quick", action="store_true", help="Quick mode (reduced params)")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p.add_argument("--phases", type=int, nargs="+", default=None,
                   help="Run only specific phases (1-6)")
    p.add_argument("--energy-budget", type=float, default=None,
                   help="Energy constraint budget (L1 projection)")
    p.add_argument(
        "--n-workers",
        type=int,
        default=None,
        dest="n_workers",
        help=(
            "Parallel workers for Lyapunov Wolf/FTLE analysis (overrides config, "
            "default 1). Set >1 for CPU multi-core acceleration. "
            "Keep at 1 for GPU inference to avoid VRAM contention. "
            "Rosenstein method (default) is not affected by this flag."
        ),
    )
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Build config
    # Priority (highest to lowest):
    #   1. CLI flags (--steps, --n-init, etc.)
    #   2. --config <path>  (explicit user override)
    #   3. config.yaml co-located with this file  (auto-detected)
    #   4. _DEFAULTS  (hardcoded fallback)
    cfg = dict(_DEFAULTS)
    _default_cfg_path = Path(__file__).parent / "config.yaml"
    if _default_cfg_path.exists():
        cfg = _merge(cfg, _load_yaml(_default_cfg_path))
        logger.debug("Auto-loaded default config: %s", _default_cfg_path)
    if args.config:
        cfg = _merge(cfg, _load_yaml(args.config))
    if args.quick:
        cfg = _merge(cfg, _QUICK_OVERRIDES)
        logger.info("⚡ Quick mode: n_init=20, steps=100, reduced validation.")

    # CLI overrides
    cfg["model"]["path"] = args.model
    # --graph may point to a single .pt file OR a folder of .pt files.
    # If a folder: collect all .pt files (sorted, checkpoints excluded) and
    # store as a list so pipeline.run_pipeline() can use them for context diversity.
    _graph_arg = Path(args.graph)
    if _graph_arg.is_dir():
        _graph_pts = _find_graph_pts(_graph_arg)
        if not _graph_pts:
            logger.error(
                "--graph folder '%s' contains no valid .pt graph-cache files "
                "(training checkpoints are excluded automatically).",
                _graph_arg,
            )
            sys.exit(1)
        logger.info(
            "--graph folder: %d .pt file(s) found in '%s'. "
            "Primary graph: '%s'. Extra graphs: %d.",
            len(_graph_pts), _graph_arg, _graph_pts[0].name, len(_graph_pts) - 1,
        )
        cfg["model"]["graph_path"] = [str(p) for p in _graph_pts]
    else:
        cfg["model"]["graph_path"] = args.graph
    if args.training_config:
        cfg["model"]["config_path"] = args.training_config
    if args.output:
        cfg["output"]["directory"] = args.output
    if args.modality:
        cfg["simulator"]["modality"] = args.modality
    if args.device:
        cfg["model"]["device"] = args.device
    if args.n_init is not None:
        cfg["data_generation"]["n_init"] = args.n_init
    if args.steps is not None:
        cfg["data_generation"]["steps"] = args.steps
    if args.no_plots:
        cfg["output"]["save_plots"] = False
    if args.energy_budget is not None:
        cfg["advanced"]["energy_constraint"]["E_budget"] = args.energy_budget
        # Passing --energy-budget implies the user wants the budget analysis to
        # run (Phase 5b: run_energy_budget_analysis from existing trajectories).
        # Without setting enabled=True the Phase 5 check would silently skip it.
        cfg["advanced"]["energy_constraint"]["enabled"] = True
    if args.n_workers is not None:
        cfg["dynamics"]["lyapunov"]["n_workers"] = args.n_workers

    # Phase selection: disable phases not requested
    if args.phases:
        phases = set(args.phases)
        if 2 not in phases:
            cfg["network_structure"]["enabled"] = False
        if 3 not in phases:
            for k in cfg["dynamics"]:
                if isinstance(cfg["dynamics"][k], dict):
                    cfg["dynamics"][k]["enabled"] = False
        if 4 not in phases:
            for k in cfg["validation"]:
                if isinstance(cfg["validation"][k], dict):
                    cfg["validation"][k]["enabled"] = False
        if 5 not in phases:
            for k in cfg["advanced"]:
                if isinstance(cfg["advanced"][k], dict):
                    cfg["advanced"][k]["enabled"] = False

    run_pipeline(cfg)


if __name__ == "__main__":
    main()
