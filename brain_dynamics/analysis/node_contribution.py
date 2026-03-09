"""
Node Contribution Analysis — Experiment 3
==========================================

Identifies which nodes (brain regions) are the **primary drivers** of the
low-dimensional dynamical manifold.

Three complementary metrics are computed:

1. **PCA loading**
   PCA is fitted on all trajectory data (n_traj × T, N).
   ``contribution_i = Σₖ |PC_k[i]| / Σᵢ Σₖ |PC_k[i]|``  (top-k PCs).
   Reveals which nodes dominate the principal axes of variance.

2. **DMD mode loading**
   From the slow modes of the DMD operator A (modes with |Re(λ_ct)| < threshold,
   corresponding to long relaxation times).
   ``contribution_i = Σₖ |mode_k[i]| / Σᵢ Σₖ |mode_k[i]|``
   Reveals which nodes drive the slow (near-critical) dynamics.

3. **Response matrix participation**
   Row-sum of |R|: how much node i influences all other nodes.
   Column-sum of |R|: how much node i is influenced by others.
   Combined: ``influence_i = (row_sum_i + col_sum_i) / 2``

Outputs
-------
node_contribution_rank.csv      — per-node contribution scores (all three metrics)
top_nodes_report.json           — top-k nodes + "dynamical core" assessment
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Contribution threshold: top nodes with cumulative contribution > this are the "core"
_CORE_CUMULATIVE_THRESH = 0.50   # top nodes explaining 50% of contribution


def _pca_node_contribution(
    trajectories: np.ndarray,
    n_components: int = 10,
) -> np.ndarray:
    """
    Compute per-node PCA contribution over ``n_components`` PCs.

    Returns normalised contribution vector of shape (N,).
    """
    try:
        from sklearn.decomposition import PCA as _PCA
        flat = trajectories.reshape(-1, trajectories.shape[-1])
        k = min(n_components, flat.shape[0] - 1, flat.shape[1])
        pca = _PCA(n_components=k)
        pca.fit(flat)
        # Weight each PC by its explained variance ratio
        weights = pca.explained_variance_ratio_[:k]  # shape (k,)
        # loadings shape: (k, N); absolute and weighted sum
        loadings = np.abs(pca.components_)  # (k, N)
        contribution = (loadings * weights[:, None]).sum(axis=0)  # (N,)
        total = contribution.sum()
        if total > 1e-12:
            contribution = contribution / total
        return contribution
    except ImportError:
        # Fallback: manual SVD
        flat = trajectories.reshape(-1, trajectories.shape[-1]).astype(np.float64)
        flat -= flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        k = min(n_components, Vt.shape[0])
        contribution = np.abs(Vt[:k]).mean(axis=0)
        total = contribution.sum()
        return contribution / total if total > 1e-12 else contribution


def _dmd_node_contribution(
    dmd_spectrum: Dict[str, Any],
    n_slow: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Per-node DMD slow-mode contribution.

    Uses the eigenvectors of the slow DMD modes (|Re(λ)| < threshold).
    Returns normalised contribution vector of shape (N,) or None if not available.
    """
    eigvecs = dmd_spectrum.get("eigenvectors")
    eigvals = dmd_spectrum.get("eigenvalues")
    if eigvecs is None or eigvals is None:
        return None
    eigvecs = np.asarray(eigvecs)
    eigvals = np.asarray(eigvals)
    if eigvecs.ndim != 2 or len(eigvals) != eigvecs.shape[1]:
        return None

    # Slow modes: those with smallest |ln|μ|| — weakest decay/growth
    log_abs = np.log(np.maximum(np.abs(eigvals), 1e-30))
    abs_decay = np.abs(log_abs)
    n_use = n_slow if n_slow is not None else max(1, dmd_spectrum.get("n_slow_modes", 5))
    slow_idx = np.argsort(abs_decay)[:n_use]

    slow_vecs = np.abs(eigvecs[:, slow_idx])  # (N, n_use)
    contribution = slow_vecs.sum(axis=1)       # (N,)
    total = contribution.sum()
    if total > 1e-12:
        contribution = contribution / total
    return contribution


def _response_matrix_participation(
    response_matrix: np.ndarray,
) -> np.ndarray:
    """
    Per-node participation from response matrix.

    Combines row-sum (outgoing influence) and column-sum (incoming influence).
    Returns normalised contribution vector of shape (N,).
    """
    R = np.abs(response_matrix)
    if R.ndim != 2:
        return np.ones(R.shape[0]) / R.shape[0]
    # Handle non-square matrix: use row-sum only
    if R.shape[0] != R.shape[1]:
        contribution = R.sum(axis=1)
    else:
        contribution = 0.5 * (R.sum(axis=1) + R.sum(axis=0))
    total = contribution.sum()
    if total > 1e-12:
        contribution = contribution / total
    return contribution


def run_node_contribution(
    trajectories: np.ndarray,
    dmd_spectrum: Optional[Dict[str, Any]] = None,
    response_matrix: Optional[np.ndarray] = None,
    top_k: int = 20,
    n_pca_components: int = 10,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Node contribution analysis (Experiment 3).

    Identifies which nodes are the primary drivers of the low-dimensional
    dynamical manifold via three independent metrics.

    Parameters
    ----------
    trajectories:
        Free-dynamics trajectories, shape (n_traj, T, N).
    dmd_spectrum:
        Result from run_jacobian_analysis (Phase 3e).  Optional.
    response_matrix:
        N×N (or n_nodes×N) causal response matrix from Phase 1.  Optional.
    top_k:
        How many top-contributing nodes to report in detail.
    n_pca_components:
        Number of PCA components used for PCA contribution.
    output_dir:
        Directory to save CSV/JSON outputs.

    Returns
    -------
    Dict with keys:
        pca_contribution        np.ndarray (N,)
        dmd_contribution        np.ndarray or None
        rm_contribution         np.ndarray or None
        combined_contribution   np.ndarray (N,) — mean of available metrics
        top_nodes               list of int — top_k node indices (combined)
        top_cumulative_pct      float — cumulative contribution of top_k nodes
        dynamical_core_nodes    list of int — nodes explaining 50% (combined)
        dynamical_core_size     int
        has_core                bool — top 10 nodes > 50% contribution
    """
    N = trajectories.shape[-1]
    result: Dict[str, Any] = {}

    # ── 1. PCA contribution ───────────────────────────────────────────────────
    pca_contrib = _pca_node_contribution(trajectories, n_components=n_pca_components)
    result["pca_contribution"] = pca_contrib

    # ── 2. DMD slow-mode contribution ────────────────────────────────────────
    dmd_contrib: Optional[np.ndarray] = None
    if dmd_spectrum is not None:
        dmd_contrib = _dmd_node_contribution(dmd_spectrum)
        result["dmd_contribution"] = dmd_contrib

    # ── 3. Response matrix participation ─────────────────────────────────────
    rm_contrib: Optional[np.ndarray] = None
    if response_matrix is not None:
        rm_contrib = _response_matrix_participation(response_matrix)
        result["rm_contribution"] = rm_contrib

    # ── Combined contribution (mean of available metrics) ────────────────────
    available = [pca_contrib]
    if dmd_contrib is not None:
        available.append(dmd_contrib)
    if rm_contrib is not None:
        available.append(rm_contrib)
    combined = np.stack(available, axis=0).mean(axis=0)  # (N,)
    total = combined.sum()
    if total > 1e-12:
        combined = combined / total
    result["combined_contribution"] = combined

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranked_idx = np.argsort(combined)[::-1]
    top_nodes = ranked_idx[:top_k].tolist()
    top_cumulative = float(combined[ranked_idx[:top_k]].sum())
    result["top_nodes"] = top_nodes
    result["top_cumulative_pct"] = round(top_cumulative * 100, 2)

    # Dynamical core: smallest set of nodes explaining >= 50%
    cumsum = np.cumsum(combined[ranked_idx])
    core_size = int(np.searchsorted(cumsum, _CORE_CUMULATIVE_THRESH) + 1)
    core_nodes = ranked_idx[:core_size].tolist()
    result["dynamical_core_nodes"] = core_nodes
    result["dynamical_core_size"] = core_size
    # Experiment 3 judgment: top-10 nodes > 50%?
    top10_pct = float(combined[ranked_idx[:10]].sum())
    result["has_core"] = bool(top10_pct > _CORE_CUMULATIVE_THRESH)
    result["top10_contribution_pct"] = round(top10_pct * 100, 2)

    logger.info(
        "  Node contribution: core_size=%d/%d, top10=%.1f%%, has_core=%s",
        core_size, N, top10_pct * 100, result["has_core"],
    )

    # ── Save outputs ──────────────────────────────────────────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # CSV: per-node contributions
        csv_path = output_dir / "node_contribution_rank.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["rank", "node_id", "combined_contribution_pct",
                      "pca_contribution_pct"]
            if dmd_contrib is not None:
                header.append("dmd_contribution_pct")
            if rm_contrib is not None:
                header.append("rm_contribution_pct")
            writer.writerow(header)
            for rank, node_id in enumerate(ranked_idx[:N], start=1):
                row = [
                    rank,
                    int(node_id),
                    round(float(combined[node_id]) * 100, 4),
                    round(float(pca_contrib[node_id]) * 100, 4),
                ]
                if dmd_contrib is not None:
                    row.append(round(float(dmd_contrib[node_id]) * 100, 4))
                if rm_contrib is not None:
                    row.append(round(float(rm_contrib[node_id]) * 100, 4))
                writer.writerow(row)

        # JSON report
        report = {
            "n_regions": N,
            "n_metrics_used": len(available),
            "metrics_used": (
                ["pca"]
                + (["dmd"] if dmd_contrib is not None else [])
                + (["response_matrix"] if rm_contrib is not None else [])
            ),
            "top_k": top_k,
            "top_nodes": top_nodes,
            "top_k_cumulative_pct": result["top_cumulative_pct"],
            "top10_contribution_pct": result["top10_contribution_pct"],
            "dynamical_core_size": core_size,
            "dynamical_core_nodes": core_nodes,
            "has_dynamical_core": result["has_core"],
            "judgment": (
                "low-dimensional dynamical core exists "
                f"(top-10 nodes explain {top10_pct * 100:.1f}% > 50%)"
                if result["has_core"]
                else f"no strong core (top-10 nodes explain {top10_pct * 100:.1f}% ≤ 50%)"
            ),
            "top10_details": [
                {
                    "rank": i + 1,
                    "node_id": int(ranked_idx[i]),
                    "combined_pct": round(float(combined[ranked_idx[i]]) * 100, 3),
                    "pca_pct": round(float(pca_contrib[ranked_idx[i]]) * 100, 3),
                }
                for i in range(min(10, N))
            ],
        }
        with open(output_dir / "top_nodes_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(
            "  Node contribution outputs saved: %s", output_dir,
        )

    return result
