"""
visualization.py
----------------
Reusable plotting helpers for brain-dynamics experiments.

Functions
---------
plot_phase_diagram(rho_values, lle_values, d2_values, ...)
    rho vs LLE / D2 / PCA-dim phase diagram.

plot_dimension_comparison(labels, lle_values, d2_values, pca_dims, ...)
    Side-by-side bar plots comparing dynamics metrics across conditions.

plot_manifold_projection(trajectories_list, labels, ...)
    Joint-PCA 3-D and 2-D scatter of multiple trajectory sets.

plot_causal_network(te_matrix, ...)
    Heatmap + source/sink bar chart for a directed information-flow matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
for _p in (_REPO, _REPO / "brain_dynamics"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _configure():
    """Configure matplotlib and return (plt, ok)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        try:
            from spectral_dynamics.plot_utils import configure_matplotlib
            configure_matplotlib()
        except Exception:
            pass
        return plt, True
    except ImportError:
        return None, False


def _fallback_png(path: str) -> None:
    """Write a minimal valid 2x2 grey PNG using stdlib only."""
    import struct
    import zlib

    def _chunk(tag: bytes, data: bytes) -> bytes:
        raw = tag + data
        return struct.pack(">I", len(data)) + raw + struct.pack(">I", zlib.crc32(raw) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0)))
        fh.write(_chunk(b"IDAT", zlib.compress(b"\x00\xff\x80\x80\x00\x40\xc0\x80")))
        fh.write(_chunk(b"IEND", b""))


# ---------------------------------------------------------------------------
# plot_phase_diagram
# ---------------------------------------------------------------------------

def plot_phase_diagram(
    rho_values: Sequence[float],
    lle_values: Sequence[float],
    d2_values: Sequence[float],
    pca_dims: Optional[Sequence[float]] = None,
    original_rho: Optional[float] = None,
    save_path: Optional[str] = None,
):
    """Phase diagram: spectral radius (rho) vs LLE, D2, and optionally PCA dim.

    Parameters
    ----------
    rho_values:
        Array of spectral radius values (x-axis).
    lle_values:
        LLE at each rho.
    d2_values:
        Correlation dimension D2 at each rho.
    pca_dims:
        PCA intrinsic dimension at each rho (optional third panel).
    original_rho:
        Mark this rho with a vertical green dashed line.
    save_path:
        File path to save the figure (PNG).  None = don't save.
    """
    plt, ok = _configure()
    if not ok:
        if save_path:
            _fallback_png(str(save_path))
        return None

    n_panels = 3 if pca_dims is not None else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))

    rho = np.asarray(rho_values)
    lle = np.asarray(lle_values)
    d2 = np.asarray(d2_values)

    def _vline(ax):
        if original_rho is not None:
            ax.axvline(original_rho, color="green", linestyle=":", linewidth=1.5,
                       label=f"original rho={original_rho:.2f}")

    # Panel 0: LLE
    axes[0].plot(rho, lle, "b-o", markersize=5)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1, label="LLE=0 (critical)")
    _vline(axes[0])
    axes[0].set_xlabel("Spectral radius (rho)")
    axes[0].set_ylabel("LLE")
    axes[0].set_title("LLE vs spectral radius")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Panel 1: D2
    axes[1].plot(rho, d2, "r-s", markersize=5)
    _vline(axes[1])
    axes[1].set_xlabel("Spectral radius (rho)")
    axes[1].set_ylabel("Correlation dimension D2")
    axes[1].set_title("D2 vs spectral radius")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Panel 2: PCA dim (optional)
    if pca_dims is not None and n_panels > 2:
        pca = np.asarray(pca_dims)
        axes[2].plot(rho, pca, "g-^", markersize=5)
        _vline(axes[2])
        axes[2].set_xlabel("Spectral radius (rho)")
        axes[2].set_ylabel("PCA dim (90% var)")
        axes[2].set_title("PCA intrinsic dim vs spectral radius")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_dimension_comparison
# ---------------------------------------------------------------------------

def plot_dimension_comparison(
    labels: Sequence[str],
    lle_values: Sequence[float],
    d2_values: Sequence[float],
    pca_dims: Sequence[float],
    save_path: Optional[str] = None,
):
    """Bar plots comparing LLE, D2, and PCA dimension across conditions.

    The first bar (index 0) is highlighted in ``steelblue`` (real brain
    network); the rest are ``salmon`` (control/random baselines).

    Parameters
    ----------
    labels:
        One label per condition.
    lle_values, d2_values, pca_dims:
        Metric values for each condition.
    save_path:
        PNG output path.
    """
    plt, ok = _configure()
    if not ok:
        if save_path:
            _fallback_png(str(save_path))
        return None

    n = len(labels)
    x = np.arange(n)
    colors = ["steelblue"] + ["salmon"] * (n - 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, vals, ylabel, title in [
        (axes[0], lle_values, "LLE", "Largest Lyapunov Exponent"),
        (axes[1], d2_values, "D2", "Correlation Dimension D2"),
        (axes[2], pca_dims, "PCA dim (90% var)", "PCA Intrinsic Dimension"),
    ]:
        ax.bar(x, vals, color=colors, width=0.5)
        if ax is axes[0]:
            ax.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_manifold_projection
# ---------------------------------------------------------------------------

def plot_manifold_projection(
    trajectories_list: List[np.ndarray],
    labels: List[str],
    n_components: int = 3,
    save_path: Optional[str] = None,
):
    """Joint-PCA projection of multiple trajectory sets (3-D + 2-D scatter).

    Parameters
    ----------
    trajectories_list:
        Each element: ``(n_traj, T, N)`` or ``(n_frames, N)``.
    labels:
        One label per trajectory set.
    n_components:
        PCA dimension (>= 3 for 3-D plot).
    save_path:
        PNG output path.
    """
    plt, ok = _configure()
    if not ok:
        if save_path:
            _fallback_png(str(save_path))
        return None

    def _flat(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        return arr

    flats = [_flat(t) for t in trajectories_list]
    joint = np.vstack(flats)
    mu = joint.mean(axis=0)
    joint_c = joint - mu
    try:
        _, _, Vt = np.linalg.svd(joint_c, full_matrices=False)
        comps = Vt[:n_components]
    except np.linalg.LinAlgError:
        comps = np.eye(min(n_components, joint_c.shape[1]))

    colors = ["steelblue", "salmon", "green", "purple", "orange"]

    fig = plt.figure(figsize=(13, 5))
    try:
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        ax3 = fig.add_subplot(121, projection="3d")
    except Exception:
        ax3 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for i, (flat, label) in enumerate(zip(flats, labels)):
        P = (flat - mu) @ comps.T
        c = colors[i % len(colors)]
        idx = np.linspace(0, len(P) - 1, min(2000, len(P)), dtype=int)
        try:
            ax3.scatter(P[idx, 0], P[idx, 1], P[idx, 2], s=1, alpha=0.3, c=c, label=label)
        except Exception:
            ax3.scatter(P[idx, 0], P[idx, 1], s=1, alpha=0.3, c=c, label=label)
        ax2.scatter(P[idx, 0], P[idx, 1], s=1, alpha=0.3, c=c, label=label)

    try:
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
        ax3.set_zlabel("PC3")
    except Exception:
        ax3.set_xlabel("PC1")
        ax3.set_ylabel("PC2")
    ax3.set_title("3-D PCA manifold")
    ax3.legend(fontsize=8, markerscale=5)

    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")
    ax2.set_title("2-D PCA projection")
    ax2.legend(fontsize=8, markerscale=5)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# plot_causal_network
# ---------------------------------------------------------------------------

def plot_causal_network(
    flow_matrix: np.ndarray,
    top_k: int = 20,
    node_labels: Optional[List[str]] = None,
    title: str = "Causal information flow",
    save_path: Optional[str] = None,
):
    """Heatmap + source/sink bar chart for a directed flow matrix.

    Parameters
    ----------
    flow_matrix:
        ``(N, N)`` array where ``M[i, j]`` is the directed flow from node *i*
        to node *j* (e.g. Granger F-statistic or Transfer Entropy).
    top_k:
        Number of top sources and sinks to show in the bar chart.
    node_labels:
        Optional per-node label strings.
    title:
        Figure suptitle.
    save_path:
        PNG output path.
    """
    plt, ok = _configure()
    if not ok:
        if save_path:
            _fallback_png(str(save_path))
        return None

    M = np.asarray(flow_matrix, dtype=float)
    N = M.shape[0]

    out_flow = M.sum(axis=1)
    in_flow = M.sum(axis=0)

    top_src = np.argsort(out_flow)[-top_k:][::-1]
    top_snk = np.argsort(in_flow)[-top_k:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(title, fontsize=12)

    # Heatmap
    im = axes[0].imshow(M, cmap="hot", aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=axes[0], shrink=0.8)
    axes[0].set_title(f"Flow matrix ({N}x{N})")
    axes[0].set_xlabel("Target node")
    axes[0].set_ylabel("Source node")

    # Bar chart: top sources (out-flow) and sinks (in-flow)
    k = min(top_k, N)
    ypos = np.arange(k)

    def _lbl(idx):
        if node_labels and idx < len(node_labels):
            return node_labels[idx]
        return str(idx)

    src_labels = [_lbl(n) for n in top_src[:k]]
    axes[1].barh(ypos, out_flow[top_src[:k]], color="steelblue", alpha=0.8, label="Out-flow")
    # Overlay sinks with negative bars for comparison
    snk_overlap = np.zeros(k)
    for j, sn in enumerate(top_snk[:k]):
        if sn in top_src[:k]:
            pos = list(top_src[:k]).index(sn)
            snk_overlap[pos] = in_flow[sn]
    axes[1].barh(ypos, -snk_overlap, color="salmon", alpha=0.8, label="In-flow (sinks)")
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels(src_labels, fontsize=8)
    axes[1].set_xlabel("Information flow")
    axes[1].set_title(f"Top-{k} sources / sinks")
    axes[1].axvline(0, color="k", linewidth=0.8)
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
