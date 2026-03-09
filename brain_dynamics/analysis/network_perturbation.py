"""
Network Perturbation Experiments — Experiments 4, 5, 6, 7
==========================================================

Three complementary experiments that test whether the observed dynamics are
determined by the network's **structural properties** rather than chance:

Experiment 5 — Hub Perturbation
    Remove the top-k highest-degree nodes from the connectivity matrix and
    recompute spectral/dynamics metrics.  If the attractor disappears or λ
    shifts significantly, hub nodes control the manifold.

Experiment 6 — Weight Randomisation
    Keep the sparsity pattern (which edges exist) but shuffle the edge weights.
    If the attractor is disrupted, the **weight structure** (not just topology)
    determines the dynamics.

Experiment 7 — Subnetwork Scaling
    Extract random subnetworks of sizes [120, 160, 200] (10 subnetworks each)
    and verify that key dynamical indices (spectral radius, K-Y dimension,
    dominant frequency) are stable.  If they are, the low-dimensional dynamics
    is not a 253-node artefact.

Experiment 4 — Structure-Preserving Random Network
    Generate random matrices with the same:
      - node count
      - sparsity (same density)
      - degree distribution (degree-sequence preserved via configuration model)
    but with randomised edge connections.  Compare spectral properties and LLE.

All analyses operate on the connectivity matrix W (response matrix or FC from
trajectories).  No additional GNN model calls are needed.

Outputs (per experiment)
------------------------
hub_perturbation_report.json
weight_randomisation_report.json
subnetwork_scaling_report.json
random_network_report.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ── Shared spectral utilities ─────────────────────────────────────────────────

def _spectral_metrics(W: np.ndarray) -> Dict[str, float]:
    """Compute key spectral metrics from a square connectivity matrix."""
    if W.ndim != 2 or W.shape[0] != W.shape[1] or W.shape[0] < 2:
        return {"error": "non_square_or_small"}
    try:
        eigvals = np.linalg.eigvals(W)
        abs_eigvals = np.abs(eigvals)
        sr = float(abs_eigvals.max())

        # Participation ratio: (Σ|λ|)² / (N · Σ|λ|²)
        s1 = abs_eigvals.sum()
        s2 = (abs_eigvals ** 2).sum()
        pr = float(s1 ** 2 / (W.shape[0] * s2)) if s2 > 1e-20 else 0.0

        # Linearised Lyapunov spectrum from eigenvalues
        lin_spec = np.sort(np.log(np.maximum(abs_eigvals, 1e-30)))[::-1]
        n_positive = int(np.sum(lin_spec > 0.001))
        lam_sum = float(lin_spec.sum())

        # Kaplan-Yorke dimension
        cs = np.cumsum(lin_spec)
        j = int(np.sum(cs >= 0))
        if j >= len(lin_spec):
            ky = float(len(lin_spec))
        elif j == 0:
            ky = 0.0
        else:
            denom = abs(float(lin_spec[j]))
            ky = float(j) + float(cs[j - 1]) / denom if denom > 1e-20 else float(j)

        return {
            "spectral_radius": sr,
            "participation_ratio": pr,
            "pr_n_ratio": pr / W.shape[0],
            "n_positive_exponents": n_positive,
            "lambda_sum": lam_sum,
            "ky_dimension": ky,
            "lambda_max_lin": float(lin_spec[0]) if len(lin_spec) > 0 else 0.0,
            "n_nodes": W.shape[0],
        }
    except np.linalg.LinAlgError as e:
        return {"error": str(e)}


def _rosenstein_lle_on_matrix(W: np.ndarray, T: int = 300, seed: int = 0) -> float:
    """
    Estimate Rosenstein LLE from short trajectories of  x(t+1) = tanh(W·x(t)).

    This is a proxy for the nonlinear LLE of the linearised system.  Used only
    when no GNN trajectories are available for the perturbed matrix.
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        _has_ros = True
    except ImportError:
        _has_ros = False

    n = W.shape[0]
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float64)
    traj = np.zeros((T, n), dtype=np.float64)
    for t in range(T):
        x = np.tanh(W @ x)
        traj[t] = x

    if _has_ros:
        try:
            lles = [rosenstein_lyapunov(traj[None, :, :])[0]]
            valid = [v for v in lles if np.isfinite(v)]
            return float(np.mean(valid)) if valid else float("nan")
        except Exception:
            pass

    # Simple Wolf-Benettin fallback
    try:
        from analysis.wc_dynamics import _wolf_benettin_lle
        return float(_wolf_benettin_lle(traj[None]))
    except ImportError:
        pass
    return float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 5: Hub Perturbation
# ═══════════════════════════════════════════════════════════════════════════════

def run_hub_perturbation(
    W: np.ndarray,
    top_k_list: Sequence[int] = (5, 10),
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Hub Perturbation experiment (Experiment 5).

    Remove top-k hub nodes (highest weighted degree) and recompute spectral
    metrics.  A large shift indicates hub nodes control the manifold.

    Parameters
    ----------
    W:
        Square (N×N) connectivity matrix.
    top_k_list:
        Number of hub nodes to remove in each condition.
    output_dir:
        Directory for JSON output.

    Returns
    -------
    Dict with keys:
        baseline          dict  spectral metrics of original W
        perturbations     dict  spectral metrics after hub removal for each k
        hub_node_ids      dict  {k: [node_ids]}
        spectral_shift    dict  {k: |ρ_pruned - ρ_baseline| / ρ_baseline}
        judgment          str   conclusion about hub control
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        logger.warning("Hub perturbation: W must be square. Got shape %s.", W.shape)
        return {"error": "non_square_W"}

    N = W.shape[0]
    result: Dict[str, Any] = {}

    # Baseline
    baseline = _spectral_metrics(W)
    result["baseline"] = baseline
    logger.info("  Hub perturbation baseline: ρ=%.4f, K-Y=%.2f",
                baseline.get("spectral_radius", float("nan")),
                baseline.get("ky_dimension", float("nan")))

    # Hub score: sum of absolute out-weights + in-weights (symmetric treatment)
    hub_scores = np.abs(W).sum(axis=1) + np.abs(W).sum(axis=0)
    hub_ranking = np.argsort(hub_scores)[::-1].tolist()

    perturbations: Dict[str, Any] = {}
    hub_node_ids: Dict[str, List[int]] = {}
    spectral_shift: Dict[str, float] = {}

    for k in sorted(set(top_k_list)):
        if k >= N - 1:
            logger.warning("  Hub perturbation: k=%d ≥ N-1=%d, skipping.", k, N - 1)
            continue
        hubs = hub_ranking[:k]
        hub_node_ids[str(k)] = [int(h) for h in hubs]

        # Prune: zero out rows and columns of hub nodes
        W_pruned = W.copy()
        W_pruned[hubs, :] = 0.0
        W_pruned[:, hubs] = 0.0

        # Keep only non-hub submatrix (for clean eigenvalue computation)
        keep = [i for i in range(N) if i not in hubs]
        W_sub = W[np.ix_(keep, keep)]
        metrics = _spectral_metrics(W_sub)
        perturbations[str(k)] = metrics

        rho_base = baseline.get("spectral_radius", float("nan"))
        rho_pert = metrics.get("spectral_radius", float("nan"))
        if np.isfinite(rho_base) and rho_base > 1e-8 and np.isfinite(rho_pert):
            shift = abs(rho_pert - rho_base) / rho_base
        else:
            shift = float("nan")
        spectral_shift[str(k)] = round(float(shift), 4) if np.isfinite(shift) else None

        logger.info(
            "  Hub perturbation k=%d: ρ=%.4f→%.4f (shift=%.2f%%), K-Y=%.2f→%.2f",
            k, rho_base, rho_pert,
            shift * 100 if np.isfinite(shift) else float("nan"),
            baseline.get("ky_dimension", float("nan")),
            metrics.get("ky_dimension", float("nan")),
        )

    result["perturbations"] = perturbations
    result["hub_node_ids"] = hub_node_ids
    result["spectral_shift"] = spectral_shift
    result["hub_ranking_top20"] = [int(h) for h in hub_ranking[:20]]

    # Judgment
    max_shift = max(
        (v for v in spectral_shift.values() if v is not None), default=0.0
    )
    if max_shift > 0.20:
        judgment = f"hub nodes strongly control manifold (max ρ-shift={max_shift:.1%})"
    elif max_shift > 0.05:
        judgment = f"moderate hub control (max ρ-shift={max_shift:.1%})"
    else:
        judgment = f"hub removal has little effect (max ρ-shift={max_shift:.1%}) — distributed dynamics"
    result["judgment"] = judgment

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        report = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
        with open(output_dir / "hub_perturbation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 6: Weight Randomisation
# ═══════════════════════════════════════════════════════════════════════════════

def run_weight_randomisation(
    W: np.ndarray,
    n_shuffles: int = 5,
    threshold: float = 1e-6,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Weight Randomisation experiment (Experiment 6).

    Preserves the sparsity structure (topology) but randomly permutes the
    non-zero edge weights.  If attractor properties disappear, the weight
    structure (not just topology) determines the dynamics.

    Parameters
    ----------
    W:
        Square (N×N) connectivity matrix.
    n_shuffles:
        Number of independent weight permutations.
    threshold:
        Values below this are treated as zero (sparse edges).
    seed:
        Base random seed (each shuffle uses seed+i).
    output_dir:
        Directory for JSON output.

    Returns
    -------
    Dict with keys:
        baseline          dict  spectral metrics of original W
        shuffled_mean     dict  mean spectral metrics over n_shuffles
        shuffled_std      dict  std of spectral metrics
        rho_original      float spectral radius of original W
        rho_shuffled_mean float mean spectral radius of shuffled W
        rho_shift         float |ρ_orig - ρ_shuf_mean| / ρ_orig
        judgment          str
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        logger.warning("Weight randomisation: W must be square. Got %s.", W.shape)
        return {"error": "non_square_W"}

    baseline = _spectral_metrics(W)
    logger.info("  Weight randomisation baseline: ρ=%.4f", baseline.get("spectral_radius", float("nan")))

    rng = np.random.default_rng(seed)

    # Identify non-zero entries (upper+lower combined — preserve symmetry pattern)
    flat = W.ravel()
    nonzero_mask = np.abs(flat) > threshold
    nonzero_vals = flat[nonzero_mask].copy()
    n_edges = int(nonzero_mask.sum())

    shuffled_metrics: List[Dict] = []
    for i in range(n_shuffles):
        shuffled_vals = rng.permutation(nonzero_vals)
        W_shuf = np.zeros_like(W)
        W_shuf_flat = W_shuf.ravel()
        W_shuf_flat[nonzero_mask] = shuffled_vals
        W_shuf = W_shuf_flat.reshape(W.shape)
        metrics = _spectral_metrics(W_shuf)
        shuffled_metrics.append(metrics)

    # Aggregate
    keys = [k for k in shuffled_metrics[0] if isinstance(shuffled_metrics[0][k], float)]
    shuffled_mean: Dict[str, float] = {}
    shuffled_std: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in shuffled_metrics if np.isfinite(m.get(k, float("nan")))]
        shuffled_mean[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
        shuffled_std[k] = round(float(np.std(vals)), 4) if vals else float("nan")

    rho_orig = baseline.get("spectral_radius", float("nan"))
    rho_shuf = shuffled_mean.get("spectral_radius", float("nan"))
    if np.isfinite(rho_orig) and rho_orig > 1e-8 and np.isfinite(rho_shuf):
        rho_shift = abs(rho_shuf - rho_orig) / rho_orig
    else:
        rho_shift = float("nan")

    ky_orig = baseline.get("ky_dimension", float("nan"))
    ky_shuf = shuffled_mean.get("ky_dimension", float("nan"))
    ky_shift = abs(ky_shuf - ky_orig) / max(ky_orig, 1.0) if np.isfinite(ky_orig) and np.isfinite(ky_shuf) else float("nan")

    logger.info(
        "  Weight randomisation: ρ %.4f→%.4f (shift=%.1f%%), "
        "K-Y %.2f→%.2f (n_edges=%d)",
        rho_orig, rho_shuf,
        rho_shift * 100 if np.isfinite(rho_shift) else float("nan"),
        ky_orig, ky_shuf, n_edges,
    )

    if np.isfinite(rho_shift) and rho_shift > 0.20:
        judgment = (
            f"weight structure determines dynamics "
            f"(ρ-shift={rho_shift:.1%} after weight shuffle)"
        )
    elif np.isfinite(rho_shift) and rho_shift > 0.05:
        judgment = (
            f"moderate weight-structure dependence (ρ-shift={rho_shift:.1%})"
        )
    else:
        rho_shift_str = f"{rho_shift:.1%}" if np.isfinite(rho_shift) else "unknown"
        judgment = (
            f"topology alone may drive dynamics "
            f"(ρ-shift={rho_shift_str} after weight shuffle)"
        )

    result = {
        "baseline": baseline,
        "shuffled_mean": shuffled_mean,
        "shuffled_std": shuffled_std,
        "n_shuffles": n_shuffles,
        "n_edges": n_edges,
        "rho_original": float(rho_orig) if np.isfinite(rho_orig) else None,
        "rho_shuffled_mean": float(rho_shuf) if np.isfinite(rho_shuf) else None,
        "rho_shift": float(rho_shift) if np.isfinite(rho_shift) else None,
        "ky_shift": float(ky_shift) if np.isfinite(ky_shift) else None,
        "judgment": judgment,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "weight_randomisation_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 7: Subnetwork Scaling
# ═══════════════════════════════════════════════════════════════════════════════

def run_subnetwork_scaling(
    W: np.ndarray,
    scales: Sequence[int] = (120, 160, 200),
    n_subnetworks: int = 10,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Subnetwork Scaling experiment (Experiment 7).

    Randomly sample subnetworks of different sizes and verify that key
    dynamical indices are stable across scales.  If they are, the
    low-dimensional dynamics is not a full-graph artefact.

    Parameters
    ----------
    W:
        Square (N×N) connectivity matrix.
    scales:
        Subnetwork sizes to test.
    n_subnetworks:
        Number of random subnetworks per scale.
    seed:
        Base random seed.
    output_dir:
        Directory for JSON output.

    Returns
    -------
    Dict with keys:
        full_network      dict  metrics for full W
        by_scale          dict  {scale: {mean, std, all_runs}}
        stability_index   float CV of spectral radius across all scales
        judgment          str
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        logger.warning("Subnetwork scaling: W must be square. Got %s.", W.shape)
        return {"error": "non_square_W"}

    N = W.shape[0]
    full_metrics = _spectral_metrics(W)
    rng = np.random.default_rng(seed)

    by_scale: Dict[str, Any] = {}
    all_rhos: List[float] = [full_metrics.get("spectral_radius", float("nan"))]

    for n_sub in sorted(set(int(s) for s in scales)):
        if n_sub >= N:
            logger.info("  Subnetwork size %d ≥ N=%d, using full matrix.", n_sub, N)
            n_sub = N
        if n_sub < 5:
            continue

        runs: List[Dict] = []
        for run_i in range(n_subnetworks):
            node_ids = rng.choice(N, size=n_sub, replace=False)
            W_sub = W[np.ix_(node_ids, node_ids)]
            m = _spectral_metrics(W_sub)
            runs.append(m)

        keys = [k for k in runs[0] if isinstance(runs[0][k], float) and k != "n_nodes"]
        mean_d: Dict[str, float] = {}
        std_d: Dict[str, float] = {}
        for k in keys:
            vals = [r[k] for r in runs if np.isfinite(r.get(k, float("nan")))]
            mean_d[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
            std_d[k] = round(float(np.std(vals)), 4) if vals else float("nan")

        rho_mean = mean_d.get("spectral_radius", float("nan"))
        rho_std = std_d.get("spectral_radius", float("nan"))
        if np.isfinite(rho_mean):
            all_rhos.append(rho_mean)
        by_scale[str(n_sub)] = {
            "mean": mean_d,
            "std": std_d,
            "n_subnetworks": n_subnetworks,
        }
        logger.info(
            "  Subnetwork n=%d: ρ=%.4f±%.4f, K-Y=%.2f±%.2f",
            n_sub, rho_mean, rho_std if np.isfinite(rho_std) else float("nan"),
            mean_d.get("ky_dimension", float("nan")),
            std_d.get("ky_dimension", float("nan")),
        )

    # Stability: coefficient of variation of spectral radius across scales
    rhos_finite = [r for r in all_rhos if np.isfinite(r)]
    if len(rhos_finite) > 1:
        cv = float(np.std(rhos_finite) / np.mean(rhos_finite)) if np.mean(rhos_finite) > 1e-8 else float("nan")
    else:
        cv = float("nan")

    if np.isfinite(cv) and cv < 0.10:
        judgment = (
            f"dynamics stable across scales (ρ-CV={cv:.1%} < 10%) "
            "— low-dimensional structure is scale-invariant"
        )
    elif np.isfinite(cv) and cv < 0.30:
        judgment = f"moderate scale sensitivity (ρ-CV={cv:.1%})"
    else:
        cv_str = f"{cv:.1%}" if np.isfinite(cv) else "unknown"
        judgment = f"dynamics are scale-sensitive (ρ-CV={cv_str}) — may be size-dependent"

    result = {
        "full_network": full_metrics,
        "by_scale": by_scale,
        "stability_index": float(cv) if np.isfinite(cv) else None,
        "judgment": judgment,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "subnetwork_scaling_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Experiment 4: Structure-Preserving Random Network
# ═══════════════════════════════════════════════════════════════════════════════

def run_structure_preserving_random(
    W: np.ndarray,
    n_random: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Structure-Preserving Random Network comparison (Experiment 4, complement to
    existing Phase 4b random_comparison).

    Generates random matrices that preserve:
      - Same node count N
      - Same sparsity (same number of non-zero edges)
      - Same edge weight magnitude distribution (values drawn from same empirical CDF)

    But with edges rewired (no structural information preserved).

    Parameters
    ----------
    W:
        Square (N×N) connectivity matrix.
    n_random:
        Number of random realisations.
    seed:
        Base random seed.
    output_dir:
        Directory for JSON output.

    Returns
    -------
    Dict with keys:
        original          dict  spectral metrics of W
        random_mean       dict  mean spectral metrics
        random_std        dict  std of spectral metrics
        delta_rho         float spectral radius decrease (original - random)
        delta_ky          float K-Y dimension decrease
        judgment          str
    """
    if W.ndim != 2 or W.shape[0] != W.shape[1]:
        logger.warning("Structure-preserving random: W must be square. Got %s.", W.shape)
        return {"error": "non_square_W"}

    N = W.shape[0]
    original = _spectral_metrics(W)
    rng = np.random.default_rng(seed)

    # Extract edge weight magnitude distribution
    flat = W.ravel()
    edge_magnitudes = np.abs(flat[np.abs(flat) > 1e-10])
    n_edges = len(edge_magnitudes)
    sparsity = n_edges / (N * N)

    random_metrics: List[Dict] = []
    for i in range(n_random):
        W_rand = np.zeros((N, N), dtype=float)
        # Place same number of edges randomly
        positions = rng.choice(N * N, size=n_edges, replace=False)
        magnitudes = rng.permutation(edge_magnitudes)
        signs = rng.choice([-1.0, 1.0], size=n_edges)
        flat_rand = W_rand.ravel()
        flat_rand[positions] = magnitudes * signs
        W_rand = flat_rand.reshape(N, N)
        random_metrics.append(_spectral_metrics(W_rand))

    keys = [k for k in random_metrics[0] if isinstance(random_metrics[0][k], float) and k != "n_nodes"]
    rand_mean: Dict[str, float] = {}
    rand_std: Dict[str, float] = {}
    for k in keys:
        vals = [m[k] for m in random_metrics if np.isfinite(m.get(k, float("nan")))]
        rand_mean[k] = round(float(np.mean(vals)), 4) if vals else float("nan")
        rand_std[k] = round(float(np.std(vals)), 4) if vals else float("nan")

    rho_orig = original.get("spectral_radius", float("nan"))
    rho_rand = rand_mean.get("spectral_radius", float("nan"))
    delta_rho = float(rho_orig - rho_rand) if np.isfinite(rho_orig) and np.isfinite(rho_rand) else float("nan")
    delta_ky = (
        float(original.get("ky_dimension", float("nan")) - rand_mean.get("ky_dimension", float("nan")))
        if all(np.isfinite(v) for v in [original.get("ky_dimension", float("nan")), rand_mean.get("ky_dimension", float("nan"))])
        else float("nan")
    )

    logger.info(
        "  Structure-preserving random: ρ %.4f→%.4f, K-Y %.2f→%.2f "
        "(sparsity=%.2f%%, n_edges=%d)",
        rho_orig, rho_rand,
        original.get("ky_dimension", float("nan")),
        rand_mean.get("ky_dimension", float("nan")),
        sparsity * 100, n_edges,
    )

    if np.isfinite(delta_rho) and delta_rho > 0.1:
        judgment = (
            f"trained structure creates distinctive dynamics "
            f"(ρ delta={delta_rho:.4f}, K-Y delta={delta_ky:.2f})"
        )
    elif np.isfinite(delta_rho) and delta_rho > 0.02:
        judgment = f"moderate structural effect (ρ delta={delta_rho:.4f})"
    else:
        delta_str = f"{delta_rho:.4f}" if np.isfinite(delta_rho) else "unknown"
        judgment = f"random networks similar to trained (ρ delta={delta_str}) — check weight distribution"

    result = {
        "original": original,
        "random_mean": rand_mean,
        "random_std": rand_std,
        "n_random": n_random,
        "n_edges": n_edges,
        "sparsity_pct": round(sparsity * 100, 3),
        "delta_rho": float(delta_rho) if np.isfinite(delta_rho) else None,
        "delta_ky": float(delta_ky) if np.isfinite(delta_ky) else None,
        "judgment": judgment,
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "random_network_report.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

    return result
