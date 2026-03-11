"""
loader.py — Phase 1 result loader
===================================

Loads trajectories, response matrix, JSON reports and optional .npy artefacts
produced by dynamics_pipeline from a Phase 1 output directory.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file, return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def _load_npy(path: Path) -> Optional[np.ndarray]:
    """Load a .npy file, return None if missing or malformed."""
    if not path.exists():
        return None
    try:
        return np.load(str(path), allow_pickle=False)
    except Exception as exc:
        logger.warning("Could not load %s: %s", path, exc)
        return None


def load_phase1_results(phase1_dir: Path) -> Dict[str, Any]:
    """
    Load all available Phase 1 artefacts from *phase1_dir*.

    Returns
    -------
    dict with keys (all optional — callers must use ``.get()``):

        trajectories         np.ndarray  (n_traj, T, N)
        response_matrix      np.ndarray  (N, N)
        dmd_operator         np.ndarray  (N, N)   — jacobian_dmd.npy
        dmd_eigenvalues      np.ndarray  (N,)     — jacobian_eigenvalues.npy
        pipeline_report      dict        pipeline_report.json
        jacobian_report      dict        dynamics/jacobian_report.json
        lyapunov_report      dict        dynamics/lyapunov_report.json
        power_spectrum_report dict       dynamics/power_spectrum_report.json
        stability_metrics    dict        dynamics/stability_metrics.json
        surrogate_test       dict        validation/surrogate_test.json
        analysis_comparison  dict        validation/analysis_comparison.json
        embedding_dimension  dict        validation/embedding_dimension.json
        spectral_summary     dict        first structure/spectral_summary_*.json
        community_structure  dict        first structure/community_structure_*.json
    """
    d = phase1_dir

    out: Dict[str, Any] = {"phase1_dir": str(d)}

    # ── Core arrays ──────────────────────────────────────────────────────────
    out["trajectories"]    = _load_npy(d / "trajectories.npy")
    out["response_matrix"] = _load_npy(d / "response_matrix.npy")
    out["dmd_operator"]    = _load_npy(d / "dynamics" / "jacobian_dmd.npy")
    out["dmd_eigenvalues"] = _load_npy(d / "dynamics" / "jacobian_eigenvalues.npy")

    # ── JSON reports ─────────────────────────────────────────────────────────
    out["pipeline_report"]       = _load_json(d / "pipeline_report.json")
    out["jacobian_report"]       = _load_json(d / "dynamics" / "jacobian_report.json")
    out["lyapunov_report"]       = _load_json(d / "dynamics" / "lyapunov_report.json")
    out["power_spectrum_report"] = _load_json(d / "dynamics" / "power_spectrum_report.json")
    out["stability_metrics"]     = _load_json(d / "dynamics" / "stability_metrics.json")
    out["surrogate_test"]        = _load_json(d / "validation" / "surrogate_test.json")
    out["analysis_comparison"]   = _load_json(d / "validation" / "analysis_comparison.json")
    out["embedding_dimension"]   = _load_json(d / "validation" / "embedding_dimension.json")

    # ── Spectral / community (take first match) ───────────────────────────────
    struct_dir = d / "structure"
    for p in sorted(struct_dir.glob("spectral_summary_*.json")):
        out["spectral_summary"] = _load_json(p)
        break
    for p in sorted(struct_dir.glob("community_structure_*.json")):
        out["community_structure"] = _load_json(p)
        break

    # ── Extract nested sub-dicts from pipeline_report for convenience ────────
    pr = out.get("pipeline_report") or {}
    out["results"] = pr.get("results", {})

    n_loaded = sum(1 for k, v in out.items() if k != "phase1_dir" and v is not None)
    logger.info("Phase 1 loader: %d artefacts found in %s", n_loaded, d)
    return out
