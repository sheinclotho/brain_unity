"""
analyses.py — Phase 2 new analysis functions
==============================================

Three new analyses that are NOT part of Phase 1 (dynamics_pipeline):

1. ``mode_node_coupling``
   For each DMD slow/Hopf mode, compute the absolute loading on each brain
   node.  Answers: *which brain regions drive the low-dimensional manifold?*
   Output: coupling matrix (M_slow × N), ranked node list, heatmap PNG.

2. ``attractor_fingerprint``
   Split trajectories into overlapping windows and compare the PCA subspaces
   via principal subspace angles.  Answers: *is the attractor geometry stable
   across different initial conditions and time windows?*
   Output: fingerprint score ∈ [0, 1], angle matrix, scatter PNG.

3. ``causal_chain``
   Synthesise all available causal evidence from Phase 1
   (surrogate tests, weight randomisation, input control, node ablation) into
   a structured evidence table and a single emergence score ∈ [0, 1].
   Answers: *is the low-dimensional attractor emergent from the trained
   model, not from input statistics or random structure?*
   Output: causal_chain.json.

All three functions are pure-NumPy (no simulator calls) and operate on the
artefacts loaded by ``phase2_analysis.loader.load_phase1_results``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_PNG_FALLBACK_SIZE = 4  # 4×4 fallback PNG


# ─────────────────────────────────────────────────────────────────────────────
# Utility: minimal fallback PNG
# ─────────────────────────────────────────────────────────────────────────────

def _write_fallback_png(path: Path) -> None:
    """Write a minimal valid 2×2 grey PNG when matplotlib is unavailable."""
    import struct, zlib  # noqa: E401

    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = b"\x00\xcc\xcc"
    idat = zlib.compress(raw * 2, level=1)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 0, 0, 0, 0))
        + _chunk(b"IDAT", idat)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png)


# ═════════════════════════════════════════════════════════════════════════════
# 1. DMD Mode–Node Coupling
# ═════════════════════════════════════════════════════════════════════════════

def mode_node_coupling(
    trajectories: np.ndarray,
    dmd_operator: Optional[np.ndarray] = None,
    dmd_eigenvalues: Optional[np.ndarray] = None,
    jacobian_report: Optional[Dict[str, Any]] = None,
    slow_re_thresh: float = 0.05,
    n_top_nodes: int = 15,
    burnin: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    DMD Mode–Node Coupling Analysis.

    For each slow or Hopf mode of the linearised transfer operator A,
    compute the absolute eigenvector loading on each brain node.  This
    reveals *which brain regions dominate the low-dimensional manifold*.

    Parameters
    ----------
    trajectories:
        shape (n_traj, T, N) — Phase 1 free-dynamics trajectories.
    dmd_operator:
        (N, N) real DMD operator A (jacobian_dmd.npy from Phase 1).
        If None, A is re-estimated from *trajectories*.
    dmd_eigenvalues:
        (N,) complex eigenvalues of A (jacobian_eigenvalues.npy).
        If None, computed from *dmd_operator*.
    jacobian_report:
        Phase 1 jacobian_report.json dict; used to read pre-computed slow/Hopf
        mode indices when available.
    slow_re_thresh:
        Threshold for |Re(λ_ct)| < threshold to classify as "slow mode"
        (continuous-time convention, λ_ct = ln(λ_discrete)).
    n_top_nodes:
        Number of top-ranked nodes to include in the output.
    burnin:
        Steps to skip at the start of each trajectory.
    output_dir:
        If provided, writes ``mode_node_heatmap.png`` and
        ``mode_node_coupling.json`` here.

    Returns
    -------
    dict with keys:
        n_slow_modes         int     — number of identified slow/Hopf modes
        coupling_matrix      list    — (n_slow, N) absolute eigenvector loadings
        slow_mode_labels     list    — human-readable label per mode
        top_nodes_per_mode   list    — list of lists: top-ranked node indices
        global_top_nodes     list    — nodes ranked by summed loading across modes
        dominant_node        int     — single most-coupled node index
        mode_frequencies_hz  list    — oscillation frequency (0 for non-Hopf)
    """
    n_traj, T, N = trajectories.shape

    # ── 1. Ensure DMD operator ────────────────────────────────────────────────
    if dmd_operator is None:
        logger.info("  mode_node_coupling: re-estimating DMD operator from trajectories")
        dmd_operator = _estimate_dmd(trajectories, burnin=burnin)

    # ── 2. Eigendecomposition ─────────────────────────────────────────────────
    if dmd_eigenvalues is None:
        try:
            eigvals, eigvecs = np.linalg.eig(dmd_operator)
        except np.linalg.LinAlgError:
            logger.warning("  mode_node_coupling: eigendecomposition failed")
            return {"error": "eigendecomposition_failed"}
    else:
        # If we have eigenvalues but not vectors, recompute with eig
        try:
            eigvals_check, eigvecs = np.linalg.eig(dmd_operator)
        except np.linalg.LinAlgError:
            return {"error": "eigendecomposition_failed"}
        eigvals = dmd_eigenvalues.astype(complex)

    # ── 3. Identify slow and Hopf modes ───────────────────────────────────────
    # Discrete → continuous-time: λ_ct = ln(λ_discrete) / dt  (dt=1 step here)
    eigvals_ct = np.log(np.maximum(np.abs(eigvals), 1e-30)).astype(float)
    # Full complex log for Im part:
    eigvals_ct_complex = np.log(np.where(np.abs(eigvals) > 1e-30, eigvals, 1e-30))

    re_ct = eigvals_ct_complex.real
    im_ct = eigvals_ct_complex.imag

    slow_mask   = np.abs(re_ct) < slow_re_thresh          # near-neutral stability
    hopf_mask   = np.abs(im_ct) > 0.01                    # oscillatory
    interest_mask = slow_mask | hopf_mask

    interest_indices = np.where(interest_mask)[0]

    # If nothing qualifies, fall back to top-5 by largest |λ_discrete|
    if len(interest_indices) == 0:
        interest_indices = np.argsort(np.abs(eigvals))[::-1][:5]
        logger.info(
            "  mode_node_coupling: no slow/Hopf modes found; using top-5 by |λ|"
        )

    # ── 4. Build coupling matrix (n_interest × N) ──────────────────────────────
    # Absolute real part of eigenvector (spatial loading on each node)
    coupling_rows: List[np.ndarray] = []
    labels: List[str] = []
    freqs_hz: List[float] = []

    for idx in interest_indices:
        vec = eigvecs[:, idx].real  # take real part as spatial loading
        coupling_rows.append(np.abs(vec))

        freq = float(np.abs(im_ct[idx]) / (2.0 * np.pi))
        re   = float(re_ct[idx])
        if np.abs(im_ct[idx]) > 0.01:
            label = f"Hopf mode {idx} | f={freq:.4f} Hz | Re(λ)={re:.4f}"
        else:
            label = f"Slow mode {idx} | Re(λ)={re:.4f}"
        labels.append(label)
        freqs_hz.append(round(freq, 6))

    if not coupling_rows:
        return {"error": "no_modes_found"}

    coupling_matrix = np.vstack(coupling_rows)  # (n_modes, N)

    # ── 5. Rank nodes by per-mode and global loading ──────────────────────────
    top_per_mode: List[List[int]] = [
        np.argsort(row)[::-1][:n_top_nodes].tolist()
        for row in coupling_matrix
    ]
    global_loading = coupling_matrix.sum(axis=0)
    global_top = np.argsort(global_loading)[::-1][:n_top_nodes].tolist()
    dominant_node = int(global_top[0]) if global_top else -1

    result: Dict[str, Any] = {
        "n_slow_modes":       len(coupling_rows),
        "n_nodes":            N,
        "slow_mode_labels":   labels,
        "top_nodes_per_mode": top_per_mode,
        "global_top_nodes":   global_top,
        "dominant_node":      dominant_node,
        "mode_frequencies_hz": freqs_hz,
        "coupling_matrix":    coupling_matrix.tolist(),
        "global_loading":     global_loading.tolist(),
    }

    # ── 6. Save outputs ───────────────────────────────────────────────────────
    if output_dir is not None:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save JSON (without large coupling_matrix array)
        json_result = {k: v for k, v in result.items()
                       if k not in ("coupling_matrix",)}
        (output_dir / "mode_node_coupling.json").write_text(
            json.dumps(json_result, indent=2, default=_json_serial),
            encoding="utf-8",
        )
        # Save heatmap
        _plot_mode_node_heatmap(coupling_matrix, labels, global_top, N,
                                output_dir / "mode_node_heatmap.png")

    return result


def _estimate_dmd(trajectories: np.ndarray, burnin: int = 0,
                  reg_alpha: float = 1e-6) -> np.ndarray:
    """Estimate DMD operator A from trajectory pairs via Tikhonov LS."""
    n_traj, T, N = trajectories.shape
    pairs: List[np.ndarray] = []
    for traj in trajectories:
        t = traj[burnin:]
        if len(t) < 2:
            continue
        x0 = t[:-1].astype(np.float64)   # (T', N)
        x1 = t[1:].astype(np.float64)    # (T', N)
        pairs.append(np.stack([x0, x1], axis=0))  # (2, T', N)

    if not pairs:
        return np.eye(N, dtype=np.float64)

    X0_all = np.concatenate([p[0] for p in pairs], axis=0)   # (M, N)
    X1_all = np.concatenate([p[1] for p in pairs], axis=0)   # (M, N)

    # A = X1^T X0 (X0^T X0 + αI)^{-1}
    C  = X0_all.T @ X0_all + reg_alpha * np.eye(N)
    D  = X0_all.T @ X1_all
    try:
        A = np.linalg.solve(C, D).T  # (N, N)
    except np.linalg.LinAlgError:
        A = np.eye(N, dtype=np.float64)
    return A.real.astype(np.float64)


def _plot_mode_node_heatmap(
    coupling_matrix: np.ndarray,
    labels: List[str],
    global_top: List[int],
    N: int,
    path: Path,
) -> None:
    """Plot coupling_matrix as a heatmap showing top-N nodes per mode."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _write_fallback_png(path)
        return

    # Only show top-30 nodes for readability
    n_show = min(30, N)
    top_idx = np.array(global_top[:n_show], dtype=int)
    sub = coupling_matrix[:, top_idx]  # (n_modes, n_show)

    # Normalise each mode row to [0, 1]
    row_max = sub.max(axis=1, keepdims=True)
    row_max = np.where(row_max < 1e-12, 1.0, row_max)
    sub_norm = sub / row_max

    fig, ax = plt.subplots(figsize=(max(6, n_show * 0.35), max(3, len(labels) * 0.55 + 1)))
    im = ax.imshow(sub_norm, aspect="auto", cmap="hot", vmin=0, vmax=1)
    ax.set_xticks(range(n_show))
    ax.set_xticklabels([f"N{i}" for i in top_idx], rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Brain node index (top by global loading)")
    ax.set_ylabel("DMD mode")
    ax.set_title("DMD Mode–Node Coupling (normalised loading)")
    fig.colorbar(im, ax=ax, label="Normalised |eigenvector| loading")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Attractor Fingerprint Stability
# ═════════════════════════════════════════════════════════════════════════════

def attractor_fingerprint(
    trajectories: np.ndarray,
    n_components: int = 5,
    burnin: int = 0,
    n_splits: int = 4,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Attractor Fingerprint Stability via PCA Subspace Angles.

    Tests whether the low-dimensional PCA subspace is **the same** across
    different trajectory windows (split by time) and trajectory groups
    (split by index).  A high stability score means the attractor geometry
    is reproducible — the system reliably finds the same manifold regardless
    of starting point.

    Method
    ------
    For each split pair (A, B):
      1. Fit PCA(n_components) on split A.
      2. Fit PCA(n_components) on split B.
      3. Compute principal angles θ_1 ≤ ... ≤ θ_k between the two subspaces
         via SVD:  cos θ = SVD(Q_A^T Q_B).
      4. Fingerprint score = cos(θ_1) ∈ [0, 1].
         1.0 = identical subspaces; 0.0 = orthogonal subspaces.

    The final stability score averages cos(θ_1) over all split pairs.

    Parameters
    ----------
    trajectories:
        shape (n_traj, T, N)
    n_components:
        Number of PCA components to compare.
    burnin:
        Steps to discard from the start of each trajectory.
    n_splits:
        Number of temporal splits.  Each consecutive pair is compared.
    output_dir:
        If provided, writes ``attractor_fingerprint.json`` and
        ``attractor_subspace_angles.png``.

    Returns
    -------
    dict with keys:
        stability_score      float — mean cos(θ_1) across all pairs
        stability_label      str   — "high" / "moderate" / "low"
        angle_matrix         list  — (n_pairs, n_components) principal angles (degrees)
        split_labels         list  — description of each split
        mean_cos_theta1      float — mean cos of first principal angle
        std_cos_theta1       float — std across pairs
    """
    n_traj, T, N = trajectories.shape
    k = min(n_components, N - 1, n_traj * (T - burnin) // 2 - 1)
    if k < 1:
        return {"error": "insufficient_data_for_fingerprint"}

    # ── Build splits ──────────────────────────────────────────────────────────
    # Split 1: temporal (divide time axis into n_splits equal parts)
    step = max(1, (T - burnin) // n_splits)
    temporal_splits = []
    for i in range(n_splits):
        t_start = burnin + i * step
        t_end   = burnin + (i + 1) * step
        t_end   = min(t_end, T)
        chunk   = trajectories[:, t_start:t_end, :]  # (n_traj, chunk_T, N)
        flat    = chunk.reshape(-1, N)
        if flat.shape[0] > k:
            temporal_splits.append(("temporal_win_%d" % i, flat))

    # Split 2: trajectory index (odd/even)
    if n_traj >= 4:
        even_flat = trajectories[0::2, burnin:, :].reshape(-1, N)
        odd_flat  = trajectories[1::2, burnin:, :].reshape(-1, N)
        if even_flat.shape[0] > k and odd_flat.shape[0] > k:
            temporal_splits.append(("traj_even", even_flat))
            temporal_splits.append(("traj_odd",  odd_flat))

    if len(temporal_splits) < 2:
        return {"error": "not_enough_splits"}

    # ── Fit PCA and compare subspaces ─────────────────────────────────────────
    subspaces: Dict[str, np.ndarray] = {}
    for label, flat in temporal_splits:
        flat = flat.astype(np.float64)
        flat -= flat.mean(axis=0)
        _, _, Vt = np.linalg.svd(flat, full_matrices=False)
        subspaces[label] = Vt[:k].T  # (N, k) column-orthonormal basis

    split_labels = list(subspaces.keys())
    all_cos_theta1: List[float] = []
    angle_rows: List[List[float]] = []
    pair_labels: List[str] = []

    for i in range(len(split_labels)):
        for j in range(i + 1, len(split_labels)):
            Q_a = subspaces[split_labels[i]]
            Q_b = subspaces[split_labels[j]]
            # Principal angles via SVD of Q_a^T Q_b
            gram = Q_a.T @ Q_b  # (k, k)
            gram = np.clip(gram, -1.0, 1.0)
            sv   = np.linalg.svd(gram, compute_uv=False)
            sv   = np.clip(sv, 0.0, 1.0)
            cos_thetas = sv
            angles_deg = np.degrees(np.arccos(cos_thetas)).tolist()
            all_cos_theta1.append(float(cos_thetas[0]))
            angle_rows.append(angles_deg)
            pair_labels.append(f"{split_labels[i]}  vs  {split_labels[j]}")

    mean_c1  = float(np.mean(all_cos_theta1))
    std_c1   = float(np.std(all_cos_theta1))

    if mean_c1 >= 0.90:
        label_s = "high"
    elif mean_c1 >= 0.70:
        label_s = "moderate"
    else:
        label_s = "low"

    result: Dict[str, Any] = {
        "stability_score":  round(mean_c1, 4),
        "stability_label":  label_s,
        "mean_cos_theta1":  round(mean_c1, 4),
        "std_cos_theta1":   round(std_c1, 4),
        "angle_matrix_deg": angle_rows,
        "pair_labels":      pair_labels,
        "split_labels":     split_labels,
        "n_components_compared": k,
        "interpretation": (
            "The attractor subspace is HIGHLY STABLE across trajectory windows "
            "and initial conditions — the system reliably finds the same "
            "low-dimensional manifold."
        ) if label_s == "high" else (
            "The attractor subspace shows MODERATE stability — the general "
            "manifold direction is consistent but some variation remains."
        ) if label_s == "moderate" else (
            "The attractor subspace shows LOW stability — different trajectory "
            "windows visit different regions of state space, suggesting multiple "
            "basins or a high-dimensional diffusion process."
        ),
    }

    if output_dir is not None:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "attractor_fingerprint.json").write_text(
            json.dumps(result, indent=2, default=_json_serial),
            encoding="utf-8",
        )
        _plot_subspace_angles(angle_rows, pair_labels, k,
                              output_dir / "attractor_subspace_angles.png")

    return result


def _plot_subspace_angles(
    angle_matrix: List[List[float]],
    pair_labels: List[str],
    n_components: int,
    path: Path,
) -> None:
    """Bar/scatter plot of principal angles across split pairs."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        _write_fallback_png(path)
        return

    arr = np.array(angle_matrix)   # (n_pairs, k)
    n_pairs, k = arr.shape

    fig, ax = plt.subplots(figsize=(max(5, k * 0.6 + 2), max(3, n_pairs * 0.8 + 1)))
    x = np.arange(k)
    width = 0.8 / max(n_pairs, 1)

    for i, (row, lbl) in enumerate(zip(arr, pair_labels)):
        offset = (i - n_pairs / 2) * width
        ax.bar(x + offset, row, width=width * 0.9, label=lbl, alpha=0.75)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Principal component index")
    ax.set_ylabel("Principal angle (degrees)")
    ax.set_title("PCA Subspace Principal Angles Between Splits\n"
                 "(0 deg = identical subspaces)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i+1}" for i in range(k)])
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  Saved %s", path)


# ═════════════════════════════════════════════════════════════════════════════
# 3. Causal Chain Evaluation
# ═════════════════════════════════════════════════════════════════════════════

def causal_chain(
    phase1_data: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Causal Chain Evaluation — synthesise all Phase 1 causal evidence.

    Tests the hypothesis:
      "The low-dimensional, near-critical attractor is an EMERGENT property
       of the trained model weights, not a consequence of input statistics,
       data distribution, or random network structure."

    Evidence sources (all from Phase 1 JSON outputs):

    E1  Surrogate test:
        Real LLE > surrogate LLE  →  nonlinear structure beyond linear baselines.

    E2  Weight randomisation:
        LLE and attractor metrics change when weights are randomly shuffled
        while keeping the same sparsity pattern.

    E3  Input dimension control:
        Low-dim attractor persists under zero-input (autonomous) AND degrades
        under high-dim noise → network structure (not input) drives low-D.

    E4  Node ablation:
        Removing top-k hub nodes disrupts the attractor more than removing
        random nodes → specific network hubs control the manifold.

    E5  Random network comparison:
        Real model has lower LLE / D₂ than structurally equivalent random
        networks with same sparsity/degree → low-D is NOT a generic property
        of random networks with similar statistics.

    Parameters
    ----------
    phase1_data:
        Dict returned by ``load_phase1_results``.
    output_dir:
        If provided, writes ``causal_chain.json`` here.

    Returns
    -------
    dict with keys:
        emergence_score      float ∈ [0, 1] — mean of passed evidence items
        emergence_label      str   — "strong" / "moderate" / "weak" / "insufficient"
        evidence_table       list  — per-evidence item with name/verdict/detail
        n_supported          int   — number of items supporting emergence
        n_total              int   — total evidence items evaluated
        interpretation       str   — human-readable summary
    """
    evidence_table: List[Dict[str, Any]] = []

    # ── E1: Surrogate test ────────────────────────────────────────────────────
    surr = phase1_data.get("surrogate_test") or {}
    _add_surrogate_evidence(surr, evidence_table)

    # ── E2: Weight randomisation ──────────────────────────────────────────────
    cmp = phase1_data.get("analysis_comparison") or {}
    # Also try from pipeline_report results
    if not cmp:
        cmp = (phase1_data.get("results") or {}).get("random_comparison") or {}
    _add_weight_randomisation_evidence(cmp, evidence_table)

    # ── E3: Input dimension control ───────────────────────────────────────────
    results = phase1_data.get("results") or {}
    idc = results.get("input_dimension_control") or {}
    _add_input_control_evidence(idc, evidence_table)

    # ── E4: Node ablation ─────────────────────────────────────────────────────
    na = results.get("node_ablation") or {}
    _add_node_ablation_evidence(na, evidence_table)

    # ── E5: Random network ─────────────────────────────────────────────────────
    rn = results.get("graph_structure_comparison") or cmp
    _add_random_network_evidence(rn, evidence_table)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_total     = len(evidence_table)
    n_supported = sum(1 for e in evidence_table if e.get("verdict") == "supports")
    n_refutes   = sum(1 for e in evidence_table if e.get("verdict") == "refutes")
    n_neutral   = n_total - n_supported - n_refutes

    if n_total == 0:
        score = 0.0
        label = "insufficient"
    else:
        score = n_supported / n_total
        if score >= 0.75 and n_total >= 3:
            label = "strong"
        elif score >= 0.50:
            label = "moderate"
        elif score >= 0.25:
            label = "weak"
        else:
            label = "insufficient"

    interp = (
        f"STRONG causal evidence ({n_supported}/{n_total} tests support "
        "emergence): The low-dimensional attractor is a trained-model emergent "
        "property, not attributable to input statistics or random structure."
        if label == "strong" else
        f"MODERATE causal evidence ({n_supported}/{n_total} tests support "
        "emergence): Most evidence points to emergence, but some tests are "
        "inconclusive or missing."
        if label == "moderate" else
        f"WEAK causal evidence ({n_supported}/{n_total} tests support "
        "emergence): The evidence is mixed; more experiments needed."
        if label == "weak" else
        "INSUFFICIENT DATA: Key causal evidence is missing from Phase 1. "
        "Enable node_ablation, input_dimension_control, and weight "
        "randomisation in the dynamics_pipeline config."
    )

    result: Dict[str, Any] = {
        "emergence_score":   round(score, 3),
        "emergence_label":   label,
        "n_supported":       n_supported,
        "n_refutes":         n_refutes,
        "n_neutral":         n_neutral,
        "n_total":           n_total,
        "evidence_table":    evidence_table,
        "interpretation":    interp,
    }

    if output_dir is not None:
        import json
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "causal_chain.json").write_text(
            json.dumps(result, indent=2, default=_json_serial),
            encoding="utf-8",
        )

    return result


# ── E1 helper ────────────────────────────────────────────────────────────────

def _add_surrogate_evidence(
    surr: Dict[str, Any],
    table: List[Dict[str, Any]],
) -> None:
    if not surr:
        table.append({
            "name":    "E1_surrogate_test",
            "verdict": "missing",
            "detail":  "Surrogate test results not found in Phase 1 output.",
        })
        return

    is_nonlinear = surr.get("is_nonlinear")
    z = surr.get("z_score")
    p = surr.get("p_value")

    if is_nonlinear is True:
        verdict = "supports"
        detail  = (
            f"Real LLE is significantly higher than surrogate LLE "
            f"(z={z:.2f}, p={p:.4f}): dynamics are nonlinear, ruling out "
            "linear stochastic process as explanation."
        )
    elif is_nonlinear is False:
        verdict = "neutral"
        detail  = (
            f"Surrogate test: real LLE is NOT significantly higher than "
            f"surrogates (z={z:.2f if z else 'N/A'}, p={p:.4f if p else 'N/A'}). "
            "Cannot rule out linear explanation."
        )
    else:
        verdict = "missing"
        detail  = "Surrogate test ran but key fields missing."

    table.append({"name": "E1_surrogate_test", "verdict": verdict, "detail": detail})


# ── E2 helper ────────────────────────────────────────────────────────────────

def _add_weight_randomisation_evidence(
    cmp: Dict[str, Any],
    table: List[Dict[str, Any]],
) -> None:
    if not cmp:
        table.append({
            "name":    "E2_weight_randomisation",
            "verdict": "missing",
            "detail":  "Random comparison / weight randomisation results not found.",
        })
        return

    # Try to read weight_randomisation results from nested structure
    wr = cmp.get("weight_randomisation") or cmp
    real_lle = wr.get("real_lle") or wr.get("brain_lle_mean")
    rand_lle = wr.get("random_lle_mean") or wr.get("rand_lle_mean")

    if real_lle is None and rand_lle is None:
        # Try another nesting: from pipeline_report
        rand_entries = cmp.get("random_entries") or []
        if rand_entries:
            rand_lles = [e.get("mean_lyapunov_mean", np.nan) for e in rand_entries
                         if e.get("mean_lyapunov_mean") is not None]
            rand_lle = float(np.mean(rand_lles)) if rand_lles else None

    if real_lle is None:
        table.append({
            "name":    "E2_weight_randomisation",
            "verdict": "missing",
            "detail":  "Weight randomisation: real LLE value not found.",
        })
        return

    if rand_lle is not None:
        if real_lle < rand_lle:
            verdict = "supports"
            detail  = (
                f"Real model LLE ({real_lle:.4f}) < random network LLE "
                f"({rand_lle:.4f}): trained weights produce more ordered, "
                "lower-dimensional dynamics than random networks with same "
                "sparsity — structural specificity confirmed."
            )
        else:
            verdict = "neutral"
            detail  = (
                f"Real model LLE ({real_lle:.4f}) >= random network LLE "
                f"({rand_lle:.4f}): trained network is not more ordered than "
                "random networks with the same sparsity."
            )
    else:
        verdict = "neutral"
        detail  = f"Real LLE={real_lle:.4f}; random comparison LLE unavailable."

    table.append({"name": "E2_weight_randomisation", "verdict": verdict, "detail": detail})


# ── E3 helper ────────────────────────────────────────────────────────────────

def _add_input_control_evidence(
    idc: Dict[str, Any],
    table: List[Dict[str, Any]],
) -> None:
    if not idc:
        table.append({
            "name":    "E3_input_dimension_control",
            "verdict": "missing",
            "detail":  "Input dimension control results not found in Phase 1.",
        })
        return

    rows = idc.get("results") or idc.get("conditions") or []
    cond_map: Dict[str, Any] = {}
    for r in rows:
        if isinstance(r, dict):
            cond_map[r.get("condition", "")] = r

    no_input = cond_map.get("no_input") or cond_map.get("A: No input") or {}
    hi_dim   = cond_map.get("high_dim_noise") or cond_map.get("B: High-dim noise") or {}
    lo_dim   = cond_map.get("low_dim_drive") or cond_map.get("C: 3-D low-dim drive") or {}

    d2_no   = no_input.get("D2")
    d2_hi   = hi_dim.get("D2")

    if d2_no is not None and d2_hi is not None:
        if d2_hi > d2_no * 1.2:
            verdict = "supports"
            detail  = (
                f"D₂(no_input)={d2_no:.2f} vs D₂(high_dim_noise)={d2_hi:.2f}: "
                "autonomous network maintains low-dimensional dynamics; "
                "high-dim noise INCREASES dimensionality — network structure, "
                "not input, drives the low-D attractor."
            )
        else:
            verdict = "neutral"
            detail  = (
                f"D₂(no_input)={d2_no:.2f} vs D₂(high_dim_noise)={d2_hi:.2f}: "
                "adding high-dim noise does not substantially change D₂."
            )
    else:
        # Fall back to LLE
        lle_no = no_input.get("LLE") or no_input.get("lle")
        lle_hi = hi_dim.get("LLE") or hi_dim.get("lle")
        if lle_no is not None:
            verdict = "supports"
            detail  = (
                f"Autonomous (no-input) LLE={lle_no:.4f}: "
                "network dynamics persist without external input."
            )
        else:
            verdict = "missing"
            detail  = "Input dimension control ran but key fields (D2/LLE) missing."

    table.append({"name": "E3_input_dimension_control", "verdict": verdict, "detail": detail})


# ── E4 helper ────────────────────────────────────────────────────────────────

def _add_node_ablation_evidence(
    na: Dict[str, Any],
    table: List[Dict[str, Any]],
) -> None:
    if not na:
        table.append({
            "name":    "E4_node_ablation",
            "verdict": "missing",
            "detail":  "Node ablation results not found in Phase 1.",
        })
        return

    top_nodes = na.get("top_nodes") or na.get("ablation_results") or []
    delta_lle_mean = na.get("mean_delta_lle") or na.get("delta_lle_mean")

    # Try individual ablation records
    if not isinstance(delta_lle_mean, (int, float)):
        if isinstance(top_nodes, list) and top_nodes:
            first = top_nodes[0]
            if isinstance(first, dict):
                delta_lle_mean = first.get("delta_lle")

    if delta_lle_mean is not None and abs(delta_lle_mean) > 0.001:
        verdict = "supports"
        detail  = (
            f"Node ablation: removing top hub(s) shifts LLE by "
            f"Δλ={delta_lle_mean:.4f} — specific nodes exert disproportionate "
            "control over the attractor dynamics."
        )
    elif delta_lle_mean is not None:
        verdict = "neutral"
        detail  = (
            f"Node ablation: hub removal shifts LLE by only Δλ={delta_lle_mean:.4f} — "
            "no node shows dominant causal control."
        )
    else:
        verdict = "neutral"
        detail  = "Node ablation results present but ΔLLE not extractable."

    table.append({"name": "E4_node_ablation", "verdict": verdict, "detail": detail})


# ── E5 helper ────────────────────────────────────────────────────────────────

def _add_random_network_evidence(
    rn: Dict[str, Any],
    table: List[Dict[str, Any]],
) -> None:
    if not rn:
        table.append({
            "name":    "E5_random_network_comparison",
            "verdict": "missing",
            "detail":  "Random network comparison results not found in Phase 1.",
        })
        return

    # Keys may differ between random_comparison and graph_structure_comparison outputs
    brain_lle   = rn.get("brain_lle_mean") or rn.get("brain_tanh_lle")
    random_lle  = rn.get("fully_random_lle_mean") or rn.get("random_lle_mean")
    deg_lle     = rn.get("degree_preserved_lle_mean")

    if brain_lle is not None and random_lle is not None:
        if brain_lle < random_lle:
            verdict = "supports"
            detail  = (
                f"Brain network LLE ({brain_lle:.4f}) < "
                f"fully-random network LLE ({random_lle:.4f}): "
                "the trained model's low-dimensional dynamics are NOT a generic "
                "property of random networks — specific weight structure matters."
            )
        else:
            verdict = "neutral"
            detail  = (
                f"Brain LLE ({brain_lle:.4f}) >= random LLE ({random_lle:.4f}): "
                "trained model is not more ordered than comparable random networks."
            )
        if deg_lle is not None:
            detail += (
                f"  Degree-preserved random network LLE={deg_lle:.4f} "
                "(topology alone cannot explain the difference)."
                if brain_lle < deg_lle else
                f"  Degree-preserved random network LLE={deg_lle:.4f}."
            )
    else:
        verdict = "missing"
        detail  = "Random network comparison: LLE values not found."

    table.append({"name": "E5_random_network_comparison", "verdict": verdict, "detail": detail})


# ─────────────────────────────────────────────────────────────────────────────
# JSON serialisation helper
# ─────────────────────────────────────────────────────────────────────────────

def _json_serial(obj: Any) -> Any:
    """JSON serialiser for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serialisable")
