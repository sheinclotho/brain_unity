"""
granger_causality.py
====================
TASK 6: Information Flow Analysis (Granger Causality)

Estimates directed information flow between brain regions using pairwise
Granger causality (bivariate, OLS F-test).

For each ordered pair (j → i), two VAR models are fitted::

    Restricted:   x_i(t) = a_0 + Σ_l a_l · x_i(t-l)  + ε_r
    Unrestricted: x_i(t) = a_0 + Σ_l a_l · x_i(t-l)
                               + Σ_l b_l · x_j(t-l) + ε_u

The Granger F-statistic (log ratio of residual variances) is used as the
edge weight in the causal network::

    G[j, i] = log(var(ε_r) / var(ε_u))   (clamped to ≥ 0)

Out-flow, in-flow, and net-flow are then derived for each node, and the
top source / sink nodes are identified.

Overlap with node importance (from node_ablation.py) is analysed when the
``node_importance_csv`` parameter is provided.

Outputs
-------
  outputs/granger_flow.csv       — per-node flow metrics
  outputs/causal_network.png     — heatmap + source/sink bar chart
"""
from __future__ import annotations

import csv
import json
import logging
import struct
import time
import zlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_STD_GUARD = 1e-12


# ---------------------------------------------------------------------------
# Core pairwise Granger
# ---------------------------------------------------------------------------

def granger_pair(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 1,
) -> float:
    """Bivariate Granger F-statistic for source → target.

    Fits both restricted (target AR only) and unrestricted (target + source)
    OLS models and returns ``log(RSS_r / RSS_u)`` clamped to ≥ 0.

    Parameters
    ----------
    source, target:
        1-D time series of length T.
    max_lag:
        VAR order p.

    Returns
    -------
    float
        Granger measure ≥ 0 (0 = no Granger causality from source to target).
    """
    T = len(target)
    if T < 2 * max_lag + 10:
        return 0.0

    p = max_lag
    n = T - p  # number of usable observations

    # Build lagged design matrices
    Y = target[p:]  # (n,)
    X_restricted = np.ones((n, p + 1), dtype=np.float64)
    X_full = np.ones((n, 2 * p + 1), dtype=np.float64)
    for lag in range(1, p + 1):
        X_restricted[:, lag] = target[p - lag: T - lag]
        X_full[:, lag] = target[p - lag: T - lag]
        X_full[:, p + lag] = source[p - lag: T - lag]

    def _ols_rss(A, b):
        try:
            coef, res, _, _ = np.linalg.lstsq(A, b, rcond=None)
            resid = b - A @ coef
            return float(np.dot(resid, resid))
        except np.linalg.LinAlgError:
            return float("nan")

    rss_r = _ols_rss(X_restricted, Y)
    rss_u = _ols_rss(X_full, Y)

    if np.isnan(rss_r) or np.isnan(rss_u) or rss_u < _STD_GUARD:
        return 0.0

    g = np.log(max(rss_r, _STD_GUARD) / rss_u)
    return float(max(0.0, g))


# ---------------------------------------------------------------------------
# Full Granger matrix
# ---------------------------------------------------------------------------

def compute_granger_matrix(
    trajectories: np.ndarray,
    max_lag: int = 1,
    n_src: Optional[int] = None,
    n_tgt: Optional[int] = None,
    aggregate: str = "mean",
) -> np.ndarray:
    """Compute pairwise Granger causality matrix.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)``.
    max_lag:
        VAR lag order.
    n_src, n_tgt:
        If provided, only compute Granger for the first ``n_src`` source nodes
        and ``n_tgt`` target nodes (for speed).  The matrix is still ``(N, N)``
        with zeros outside the computed block.
    aggregate:
        ``'mean'`` or ``'median'`` across trajectories.

    Returns
    -------
    np.ndarray
        ``(N, N)`` matrix where ``G[j, i] = G-causality from j → i``.
    """
    trajs = np.asarray(trajectories, dtype=np.float64)
    if trajs.ndim == 2:
        trajs = trajs[np.newaxis]

    n_traj, T, N = trajs.shape
    n_src = n_src or N
    n_tgt = n_tgt or N
    n_src = min(n_src, N)
    n_tgt = min(n_tgt, N)

    logger.info("Granger matrix: N=%d, computing %d×%d block, lag=%d, n_traj=%d",
                N, n_src, n_tgt, max_lag, n_traj)

    G = np.zeros((N, N), dtype=np.float64)

    t0 = time.time()
    for j in range(n_src):
        for i in range(n_tgt):
            if j == i:
                continue
            vals = []
            for k in range(n_traj):
                g = granger_pair(trajs[k, :, j], trajs[k, :, i], max_lag)
                vals.append(g)
            if aggregate == "median":
                G[j, i] = float(np.median(vals))
            else:
                G[j, i] = float(np.mean(vals))

        if j % 20 == 0:
            elapsed = time.time() - t0
            rows_done = (j + 1) * n_tgt
            rows_total = n_src * n_tgt
            eta = elapsed / rows_done * (rows_total - rows_done) if rows_done > 0 else 0.0
            logger.info("  Granger: %d/%d rows done (ETA %.0fs)", j + 1, n_src, eta)

    return G


# ---------------------------------------------------------------------------
# Flow metrics
# ---------------------------------------------------------------------------

def compute_flow_metrics(G: np.ndarray) -> Dict:
    """Compute out-flow, in-flow, net-flow per node from Granger matrix.

    Parameters
    ----------
    G:
        ``(N, N)`` Granger matrix.  ``G[j, i]`` = j → i causality.

    Returns
    -------
    dict with keys ``out_flow``, ``in_flow``, ``net_flow`` (each ndarray N).
    """
    out_flow = G.sum(axis=1)   # total outgoing causality
    in_flow = G.sum(axis=0)    # total incoming causality
    net_flow = out_flow - in_flow
    return {
        "out_flow": out_flow,
        "in_flow": in_flow,
        "net_flow": net_flow,
    }


def _top_nodes(vals: np.ndarray, k: int) -> List[int]:
    return list(np.argsort(vals)[-k:][::-1])


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_granger_analysis(
    trajectories: np.ndarray,
    max_lag: int = 1,
    n_src: Optional[int] = None,
    n_tgt: Optional[int] = None,
    top_k: int = 20,
    node_importance_csv: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """Full Granger causality analysis pipeline.

    Parameters
    ----------
    trajectories:
        Shape ``(n_traj, T, N)`` or ``(T, N)``.
    max_lag:
        VAR lag order.
    n_src, n_tgt:
        Block limits for computation speed.
    top_k:
        Number of top sources/sinks to report.
    node_importance_csv:
        Path to ``node_importance.csv`` from node ablation (for overlap).
    output_dir:
        Directory to save outputs.

    Returns
    -------
    dict with 'granger_matrix', 'out_flow', 'in_flow', 'net_flow',
    'top_sources', 'top_sinks', 'top_drivers', and optional 'overlap'.
    """
    G = compute_granger_matrix(
        trajectories, max_lag=max_lag, n_src=n_src, n_tgt=n_tgt
    )
    flow = compute_flow_metrics(G)
    out_flow = flow["out_flow"]
    in_flow = flow["in_flow"]
    net_flow = flow["net_flow"]

    N = G.shape[0]
    top_sources = _top_nodes(out_flow, min(top_k, N))
    top_sinks = _top_nodes(in_flow, min(top_k, N))
    top_drivers = _top_nodes(net_flow, min(top_k, N))

    logger.info("  Top-5 sources: %s", top_sources[:5])
    logger.info("  Top-5 sinks:   %s", top_sinks[:5])
    logger.info("  Top-5 drivers: %s", top_drivers[:5])

    results: Dict = {
        "granger_matrix": G,
        "out_flow": out_flow,
        "in_flow": in_flow,
        "net_flow": net_flow,
        "top_sources": top_sources,
        "top_sinks": top_sinks,
        "top_drivers": top_drivers,
    }

    # Overlap analysis with node ablation importance
    if node_importance_csv is not None:
        overlap = _overlap_with_importance(top_sources, node_importance_csv, label="sources")
        results["overlap_sources"] = overlap

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save raw Granger matrix
        np.save(output_dir / "granger_matrix.npy", G)

        # Save per-node CSV
        _save_flow_csv(out_flow, in_flow, net_flow, output_dir / "granger_flow.csv")

        # Save network plot
        try:
            import sys
            _REPO = Path(__file__).resolve().parents[3]
            for _p in (_REPO, _REPO / "scripts"):
                if _p.exists() and str(_p) not in sys.path:
                    sys.path.insert(0, str(_p))
            from scripts.utils.visualization import plot_causal_network
            plot_causal_network(
                G, top_k=top_k, title="Granger causal network",
                save_path=str(output_dir / "causal_network.png"),
            )
        except Exception as exc:
            logger.warning("  Plot failed: %s", exc)
            _fallback_png(str(output_dir / "causal_network.png"))

        # Save JSON report
        report = {
            "top_sources": top_sources[:top_k],
            "top_sinks": top_sinks[:top_k],
            "top_drivers": top_drivers[:top_k],
            "mean_out_flow": float(out_flow.mean()),
            "mean_in_flow": float(in_flow.mean()),
            "max_granger": float(G.max()),
            "n_nonzero_edges": int((G > 0).sum()),
            "overlap_sources": results.get("overlap_sources"),
        }
        try:
            with open(output_dir / "granger_report.json", "w") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_flow_csv(out_flow, in_flow, net_flow, path: Path) -> None:
    try:
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["node", "out_flow", "in_flow", "net_flow"])
            w.writeheader()
            for i, (o, n_, net) in enumerate(zip(out_flow, in_flow, net_flow)):
                w.writerow({"node": i, "out_flow": float(o),
                            "in_flow": float(n_), "net_flow": float(net)})
        logger.info("  Saved %s", path)
    except Exception as exc:
        logger.warning("  CSV save failed: %s", exc)


def _overlap_with_importance(
    top_granger: List[int],
    csv_path: Path,
    label: str = "sources",
) -> Dict:
    """Compute overlap between Granger top nodes and ablation top nodes."""
    try:
        ablation_nodes = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ablation_nodes.append(int(row["node"]))
        ablation_top = set(ablation_nodes[:len(top_granger)])
        granger_set = set(top_granger)
        overlap = ablation_top & granger_set
        return {
            "n_granger": len(top_granger),
            "n_ablation": len(ablation_top),
            "overlap_count": len(overlap),
            "overlap_fraction": len(overlap) / max(len(top_granger), 1),
            "overlap_nodes": list(overlap),
            "label": label,
        }
    except Exception as exc:
        logger.debug("  Overlap analysis failed: %s", exc)
        return {}


def _fallback_png(path: str) -> None:
    """Write a minimal valid 2x2 grey PNG using stdlib only."""
    def _c(tag, data):
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xffffffff)
    # Two deflate-compressed scanlines: filter-byte + 3-byte RGB pixels
    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_c(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)))
        fh.write(_c(b"IDAT", zlib.compress(b"\x00\xff\x80\x80\x00\x40\xc0\x80")))
        fh.write(_c(b"IEND", b""))
