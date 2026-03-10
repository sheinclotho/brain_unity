"""
Line Attractor Analysis (Enhanced Version)
==========================================
Tests the hypothesis that brain dynamics reduce to a 2-D slow manifold
with line-attractor structure.

New/Improved plots added:
- slow_manifold_1d_dpc1.png     : Effective 1D dynamics along PC1 (key for neutrality)
- pc2_contraction.png           : Clean view of PC2 contraction with fit
- combined_highlight_manifold.png : Trajectories + field + presumed line attractor

Usage (standalone):
    python -m analysis.line_attractor_enhanced \\
        --trajectories outputs/dynamics_pipeline/dynamics/trajectories.npy \\
        --output outputs/dynamics_pipeline/attractor

Reference:
  Seung (1996), Mante et al. (2013), etc.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Constants ────────────────────────────────────────────────────────────────
_DIV_GUARD: float = 1e-9
_NEUTRALITY_THRESH: float = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Core analysis functions (unchanged from your original)
# ─────────────────────────────────────────────────────────────────────────────

def _pca_project(trajectories: np.ndarray, burnin: int = 0) -> tuple:
    try:
        from sklearn.decomposition import PCA
    except ImportError as e:
        raise ImportError("scikit-learn is required") from e
    n_traj, T, N = trajectories.shape
    T_eff = T - burnin
    X_eff = trajectories[:, burnin:, :].reshape(-1, N)
    pca = PCA(n_components=2, random_state=42)
    Z_flat = pca.fit_transform(X_eff)
    Z = Z_flat.reshape(n_traj, T_eff, 2)
    return Z, pca, pca.explained_variance_ratio_


def _estimate_velocity(Z: np.ndarray) -> tuple:
    dZ = Z[:, 1:, :] - Z[:, :-1, :]
    Z_mid = Z[:, :-1, :]
    return Z_mid.reshape(-1, 2), dZ.reshape(-1, 2)


def _fit_linear_field(pos: np.ndarray, vel: np.ndarray) -> tuple:
    try:
        from sklearn.linear_model import LinearRegression
    except ImportError as e:
        raise ImportError("scikit-learn is required") from e
    model1 = LinearRegression().fit(pos, vel[:, 0])
    model2 = LinearRegression().fit(pos, vel[:, 1])
    r2_1 = float(model1.score(pos, vel[:, 0]))
    r2_2 = float(model2.score(pos, vel[:, 1]))
    return model1, model2, r2_1, r2_2


def _assess_line_attractor(model1, model2) -> Dict:
    c1 = model1.coef_
    ic1 = float(model1.intercept_)
    c2 = model2.coef_
    ic2 = float(model2.intercept_)
    lambda_pc2 = float(c2[1])
    pc1_neutral = bool(
        abs(c1[0]) < _NEUTRALITY_THRESH and
        abs(c1[1]) < _NEUTRALITY_THRESH and
        abs(ic1) < _NEUTRALITY_THRESH
    )
    pc2_contracts = bool(
        lambda_pc2 < -0.01 and
        abs(c2[0]) < 0.2 * abs(lambda_pc2 + _DIV_GUARD)
    )
    is_line = pc1_neutral and pc2_contracts
    interpretation = "line_attractor" if is_line else \
                     "partial_contraction" if pc2_contracts else \
                     "neutral_plane" if pc1_neutral else "other"
    return {
        "dPC1_coefs": {"PC1": float(c1[0]), "PC2": float(c1[1]), "intercept": ic1},
        "dPC2_coefs": {"PC1": float(c2[0]), "PC2": float(c2[1]), "intercept": ic2},
        "lambda_pc2": lambda_pc2,
        "pc1_neutral": pc1_neutral,
        "pc2_contracts": pc2_contracts,
        "is_line_attractor": is_line,
        "interpretation": interpretation,
    }

# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers ─────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pc_dynamics(pc_mid: np.ndarray, dpc: np.ndarray, pc_idx: int,
                      output_path: Path, model=None) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pc_mid, dpc, s=1, alpha=0.3, color="steelblue", rasterized=True)
    if model is not None:
        x_lin = np.linspace(pc_mid.min(), pc_mid.max(), 200)
        X_lin = np.column_stack([x_lin if pc_idx==1 else np.zeros_like(x_lin),
                                 x_lin if pc_idx==2 else np.zeros_like(x_lin)])
        y_lin = model.predict(X_lin)
        ax.plot(x_lin if pc_idx==1 else np.zeros_like(x_lin),
                y_lin, color="red", lw=1.5, label="fit (other PC=0)")
        ax.legend(fontsize=8)
    ax.axhline(0, color="k", lw=0.8, ls="--", alpha=0.6)
    ax.set_xlabel(f"PC{pc_idx}")
    ax.set_ylabel(f"dPC{pc_idx}/dt")
    ax.set_title(f"PC{pc_idx} Dynamics")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_slow_manifold_1d_dynamics(
    pos: np.ndarray, vel: np.ndarray, model1,
    output_path: Path,
    manifold_threshold: float = 3.0,
    bin_width: Optional[float] = None,
    min_points_per_bin: int = 20
) -> None:
    import matplotlib.pyplot as plt
    from scipy.stats import binned_statistic

    close = np.abs(pos[:, 1]) < manifold_threshold
    if close.sum() < 50:
        logger.warning("Too few points near PC2=0")
        return

    pc1_near = pos[close, 0]
    dpc1_near = vel[close, 0]

    if bin_width is None:
        bin_width = (pc1_near.max() - pc1_near.min()) / 25

    bins = np.arange(pc1_near.min(), pc1_near.max() + bin_width, bin_width)
    mean_d, _, _ = binned_statistic(pc1_near, dpc1_near, 'mean', bins=bins)
    count, _, _ = binned_statistic(pc1_near, dpc1_near, 'count', bins=bins)
    std_d, _, _ = binned_statistic(pc1_near, dpc1_near, 'std', bins=bins)

    valid = count >= min_points_per_bin
    centers = (bins[:-1] + bins[1:]) / 2

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(centers[valid], mean_d[valid],
                yerr=std_d[valid]/np.sqrt(count[valid]),
                fmt='o-', color='darkred', ecolor='gray', capsize=3,
                ms=5, lw=1.2, label='mean ± SEM')
    ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.7)

    x_dense = np.linspace(pc1_near.min(), pc1_near.max(), 200)
    X_pred = np.column_stack([x_dense, np.zeros_like(x_dense)])
    dpc1_pred = model1.predict(X_pred)
    ax.plot(x_dense, dpc1_pred, '--', color='teal', lw=2,
            label='linear model (PC2=0)')

    ax.set_xlabel("PC1")
    ax.set_ylabel("dPC1/dt  (along slow manifold)")
    ax.set_title("1D Effective Dynamics on Line Attractor")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_pc2_contraction_with_fit(
    pos: np.ndarray, vel: np.ndarray, model2, output_path: Path
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(pos[:, 1], vel[:, 1], s=1.5, alpha=0.4, color="C0", rasterized=True)

    pc2_rng = np.linspace(pos[:, 1].min(), pos[:, 1].max(), 200)
    X_pred = np.column_stack([np.zeros_like(pc2_rng), pc2_rng])
    v_pred = model2.predict(X_pred)
    ax.plot(pc2_rng, v_pred, color="C3", lw=1.8, label="linear fit")

    ax.axhline(0, color="k", ls="--", lw=0.8, alpha=0.6)
    ax.axvline(0, color="k", ls="--", lw=0.8, alpha=0.6)

    ax.set_xlabel("PC2")
    ax.set_ylabel("dPC2/dt")
    ax.set_title("Contraction toward the line (PC2 → 0)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)

    slope = model2.coef_[1]
    ax.text(0.02, 0.98,
            f"slope = {slope:.4f}\nR² = {model2.score(pos, vel[:,1]):.3f}",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(facecolor='white', alpha=0.85))

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def _plot_combined_highlight_manifold(
    Z: np.ndarray, evr: np.ndarray, model1, model2,
    output_path: Path, n_show: int = 12, n_grid: int = 45
) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 7.5))

    # faint background trajectories
    for i in range(Z.shape[0]):
        ax.plot(Z[i, :, 0], Z[i, :, 1], lw=0.4, alpha=0.12, color="gray")

    # highlight few trajectories
    cmap = plt.get_cmap("inferno")
    for i in range(min(n_show, Z.shape[0])):
        t_norm = np.linspace(0, 1, Z.shape[1])
        colors = cmap(t_norm)
        ax.scatter(Z[i, :, 0], Z[i, :, 1], c=colors, s=2, alpha=0.9, rasterized=True)
        ax.plot(Z[i, 0, 0], Z[i, 0, 1], 'o', ms=6, mec='w', mfc='lime', mew=1.5)
        ax.plot(Z[i, -1, 0], Z[i, -1, 1], '^', ms=7, mec='w', mfc='red', mew=1.5)

    # presumed line attractor
    x_rng = np.array([Z[:, :, 0].min(), Z[:, :, 0].max()])
    ax.plot(x_rng, [0, 0], '--', color='white', lw=2.8, zorder=10,
            label="presumed line attractor (PC2=0)")

    # velocity field
    pc1_min, pc1_max = Z[:, :, 0].min(), Z[:, :, 0].max()
    pc2_min, pc2_max = Z[:, :, 1].min(), Z[:, :, 1].max()
    x = np.linspace(pc1_min, pc1_max, n_grid)
    y = np.linspace(pc2_min, pc2_max, n_grid)
    Xg, Yg = np.meshgrid(x, y)
    grid = np.column_stack([Xg.ravel(), Yg.ravel()])
    U = model1.predict(grid).reshape(Xg.shape)
    V = model2.predict(grid).reshape(Yg.shape)
    speed = np.sqrt(U**2 + V**2)
    strm = ax.streamplot(Xg, Yg, U, V, color=speed, cmap='magma_r',
                         density=1.4, linewidth=0.9, arrowsize=0.9)
    fig.colorbar(strm.lines, ax=ax, label='speed', shrink=0.7)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_title("Trajectories + Velocity Field\n(white dashed = line attractor)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main draw function ───────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

def draw(
    trajectories_path: Union[str, Path, None] = None,
    trajectories: Optional[np.ndarray] = None,
    output_dir: Union[str, Path, None] = None,
    burnin: int = 0,
    n_grid: int = 40,
    n_show: int = 12,
) -> Dict:
    if trajectories is None:
        if trajectories_path is None:
            raise ValueError("Provide trajectories_path or trajectories")
        traj_path = Path(trajectories_path)
        if not traj_path.exists():
            raise FileNotFoundError(str(traj_path))
        trajectories = np.load(traj_path)
    else:
        traj_path = None

    if trajectories.ndim != 3:
        raise ValueError(f"Expected (n_traj, T, N), got {trajectories.shape}")

    n_traj, T, N = trajectories.shape
    logger.info(f"Loaded: {n_traj} traj, T={T}, N={N}, burnin={burnin}")

    if output_dir is None:
        output_dir = traj_path.parent if traj_path else Path(".")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # PCA + velocity + fit
    Z, pca, evr = _pca_project(trajectories, burnin)
    pos, vel = _estimate_velocity(Z)
    model1, model2, r2_1, r2_2 = _fit_linear_field(pos, vel)
    assessment = _assess_line_attractor(model1, model2)

    logger.info(f"PC1: {evr[0]*100:.1f}%, PC2: {evr[1]*100:.1f}%")
    logger.info(f"dPC1 R²={r2_1:.3f}, dPC2 R²={r2_2:.3f}")
    logger.info(f"Assessment: {assessment['interpretation']} "
                f"(λ_PC2={assessment['lambda_pc2']:.4f})")

    # ── Plots ────────────────────────────────────────────────────────────────
    _plot_pc_dynamics(pos[:, 0], vel[:, 0], 1, out/"pc1_dynamics.png", model1)
    _plot_pc_dynamics(pos[:, 1], vel[:, 1], 2, out/"pc2_dynamics.png", model2)

    # 新增关键诊断图
    _plot_slow_manifold_1d_dynamics(
        pos, vel, model1,
        out / "slow_manifold_dPC1.png"
    )

    _plot_pc2_contraction_with_fit(
        pos, vel, model2,
        out / "pc2_contraction.png"
    )

    _plot_combined_highlight_manifold(
        Z, evr, model1, model2,
        out / "combined_highlight_manifold.png",
        n_show=n_show, n_grid=n_grid
    )

    # Report
    report = {
        "pca_explained_variance_ratio": evr.tolist(),
        "linear_fit": {
            "dPC1": {"R2": r2_1, **assessment["dPC1_coefs"]},
            "dPC2": {"R2": r2_2, **assessment["dPC2_coefs"]},
        },
        "assessment": assessment,
        "n_trajectories": int(n_traj),
        "output_dir": str(out),
    }
    with open(out / "line_attractor_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


# CLI
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectories", default="trajectories.npy")
    parser.add_argument("--output", default=None)
    parser.add_argument("--burnin", type=int, default=0)
    parser.add_argument("--n-grid", type=int, default=40)
    parser.add_argument("--n-show", type=int, default=12)
    args = parser.parse_args()

    draw(
        trajectories_path=args.trajectories,
        output_dir=args.output,
        burnin=args.burnin,
        n_grid=args.n_grid,
        n_show=args.n_show,
    )