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

from .pipeline import run_pipeline

logger = logging.getLogger("dynamics_pipeline")

# ── Default configuration ─────────────────────────────────────────────────────

_DEFAULTS = {
    "model": {
        "path": None,
        "graph_path": None,
        "config_path": None,
        "device": "auto",
    },
    "simulator": {"modality": "fmri"},
    "data_generation": {"n_init": 100, "steps": 500, "seed": 42},
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
            "enabled": True, "max_lag": 50, "min_sep": 20, "n_segments": 3,
            "convergence_threshold": 0.05, "delay_embed_dim": 0, "delay_embed_tau": 1,
        },
        "dmd_spectrum": {"enabled": True, "tail_steps": 20, "n_states": 3},
        "power_spectrum": {"enabled": True},
        "pca": {"enabled": True},
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
        "perturbation": {"enabled": False},
    },
    "advanced": {
        "stimulation": {
            "enabled": True, "target_nodes": [0, 100], "amplitude": 0.5,
            "frequency": 10.0, "duration": 30, "patterns": ["sine"],
        },
        "response_matrix": {"enabled": True, "n_nodes": 10, "stim_amplitude": 0.5},
        "energy_constraint": {"enabled": False},
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
  Full run:   python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt
  Quick run:  python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
  Phases 1-3: python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 2 3
        """,
    )
    p.add_argument("--model", required=True, help="Trained model checkpoint (.pt)")
    p.add_argument("--graph", required=True, help="Graph cache file (.pt)")
    p.add_argument("--config", type=Path, default=None, help="YAML config override")
    p.add_argument("--training-config", default=None, help="Training config.yaml")
    p.add_argument("--output", default=None, help="Output directory")
    p.add_argument("--modality", choices=["fmri", "eeg", "both", "joint"], default=None)
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default=None)
    p.add_argument("--n-init", type=int, default=None, help="Number of trajectories")
    p.add_argument("--steps", type=int, default=None, help="Steps per trajectory")
    p.add_argument("--quick", action="store_true", help="Quick mode (reduced params)")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    p.add_argument("--phases", type=int, nargs="+", default=None,
                   help="Run only specific phases (1-6)")
    p.add_argument("--energy-budget", type=float, default=None,
                   help="Energy constraint budget (L1 projection)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    args = _parse_args()

    # Build config
    cfg = dict(_DEFAULTS)
    if args.config:
        cfg = _merge(cfg, _load_yaml(args.config))
    if args.quick:
        cfg = _merge(cfg, _QUICK_OVERRIDES)
        logger.info("⚡ Quick mode: n_init=20, steps=100, reduced validation.")

    # CLI overrides
    cfg["model"]["path"] = args.model
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
