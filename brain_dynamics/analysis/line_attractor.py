"""
Line Attractor Analysis
=======================

Tests the hypothesis that the brain dynamics reduce to a 2-D slow manifold
with **line-attractor** structure:

    High-dimensional state x(t)   (N regions)
        ↓  PCA
    z(t) = [PC1, PC2]
        ↓  velocity estimation
    dz/dt ≈ F(z)

If the system is a line attractor the effective dynamics satisfy:

    dPC1/dt ≈ 0           (PC1 axis is neutral — the "line")
    dPC2/dt ≈ −a·PC2      (PC2 contracts linearly toward zero)

Usage (standalone)::

    python -m analysis.line_attractor \\
        --trajectories outputs/dynamics_pipeline/dynamics/trajectories.npy \\
        --output outputs/dynamics_pipeline/attractor

Usage (library)::

    from analysis.line_attractor import draw
    results = draw(
        trajectories_path="outputs/dynamics_pipeline/dynamics/trajectories.npy",
        output_dir="outputs/dynamics_pipeline/attractor",
    )

The function saves four PNG files plus a JSON report to ``output_dir``:

  pc1_dynamics.png         — scatter: PC1 vs dPC1/dt
  pc2_dynamics.png         — scatter: PC2 vs dPC2/dt (expect linear slope)
  vector_field.png         — quiver plot of estimated 2-D velocity field
  pc_phase_portrait.png    — PC1-PC2 phase portrait with trajectories
  line_attractor_report.json

Reference
---------
  Seung (1996) "How the brain keeps the eyes still." PNAS 93:13339–13344.
  Mante et al. (2013) "Context-dependent computation by recurrent dynamics." Nature.
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

# Allow running as a module from any directory
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# ── Constants ─────────────────────────────────────────────────────────────────
# Guard against division by zero in assessment ratios
_DIV_GUARD: float = 1e-9
# Coefficient magnitude below which a term is considered "neutral" (no effect)
_NEUTRALITY_THRESH: float = 0.05


# ─────────────────────────────────────────────────────────────────────────────
# Core analysis
# ─────────────────────────────────────────────────────────────────────────────

def _pca_project(trajectories: np.ndarray, burnin: int = 0) -> tuple:
    """PCA-project trajectories to 2-D.

    Args:
        trajectories: (n_traj, T, N) float array.
        burnin:       Steps to discard at the start of each trajectory.

    Returns:
        Z:    (n_traj, T_eff, 2) — PC coordinates for all trajectories.
        pca:  fitted sklearn PCA object.
        evr:  explained variance ratio (length-2 array).
    """
    try:
        from sklearn.decomposition import PCA as _PCA
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn is required for line attractor analysis") from e

    n_traj, T, N = trajectories.shape
    T_eff = T - burnin

    X_eff = trajectories[:, burnin:, :]            # (n_traj, T_eff, N)
    X_flat = X_eff.reshape(-1, N)                  # (n_traj*T_eff, N)

    pca = _PCA(n_components=2, random_state=42)
    Z_flat = pca.fit_transform(X_flat)             # (n_traj*T_eff, 2)
    Z = Z_flat.reshape(n_traj, T_eff, 2)           # (n_traj, T_eff, 2)

    return Z, pca, pca.explained_variance_ratio_


def _estimate_velocity(Z: np.ndarray) -> tuple:
    """Finite-difference velocity estimates and mid-point positions.

    Args:
        Z: (n_traj, T_eff, 2) — PC coordinates.

    Returns:
        pos:  (n_points, 2)  — mid-point PC positions
        vel:  (n_points, 2)  — finite-difference velocity [dPC1, dPC2]
    """
    dZ = Z[:, 1:, :] - Z[:, :-1, :]     # (n_traj, T_eff-1, 2) — velocity
    Z_mid = Z[:, :-1, :]                  # (n_traj, T_eff-1, 2) — position at left edge

    pos = Z_mid.reshape(-1, 2)
    vel = dZ.reshape(-1, 2)
    return pos, vel


def _fit_linear_field(pos: np.ndarray, vel: np.ndarray) -> tuple:
    """Fit linear regression  dPCi = a_i*PC1 + b_i*PC2 + c_i  for i=1,2.

    Args:
        pos: (n_points, 2) — [PC1_mid, PC2_mid].
        vel: (n_points, 2) — [dPC1, dPC2].

    Returns:
        model1: fitted LinearRegression for dPC1
        model2: fitted LinearRegression for dPC2
        r2_1:   R² for dPC1 model
        r2_2:   R² for dPC2 model
    """
    try:
        from sklearn.linear_model import LinearRegression as _LR
    except ImportError as e:  # pragma: no cover
        raise ImportError("scikit-learn is required for line attractor analysis") from e

    model1 = _LR().fit(pos, vel[:, 0])
    model2 = _LR().fit(pos, vel[:, 1])

    r2_1 = float(model1.score(pos, vel[:, 0]))
    r2_2 = float(model2.score(pos, vel[:, 1]))

    return model1, model2, r2_1, r2_2


def _assess_line_attractor(model1, model2) -> Dict:
    """Assess whether the fitted dynamics are consistent with a line attractor.

    A line attractor has:
        dPC1/dt ≈ 0          (all coefficients ≈ 0)
        dPC2/dt ≈ −a·PC2     (coefficient of PC2 is negative, PC1 coeff ≈ 0)

    Returns a dict with assessment flags and numeric evidence.
    """
    c1 = model1.coef_          # [a1, b1]  for dPC1
    ic1 = float(model1.intercept_)
    c2 = model2.coef_          # [a2, b2]  for dPC2
    ic2 = float(model2.intercept_)

    # PC2 contraction rate (should be < 0 for line attractor)
    lambda_pc2 = float(c2[1])   # coefficient of PC2 in dPC2 equation

    # Neutrality of PC1: all coefficients close to zero
    pc1_neutral = bool(
        abs(c1[0]) < _NEUTRALITY_THRESH
        and abs(c1[1]) < _NEUTRALITY_THRESH
        and abs(ic1) < _NEUTRALITY_THRESH
    )
    # PC2 contraction: lambda_pc2 < 0 and other terms small
    pc2_contracts = bool(
        lambda_pc2 < -0.01
        and abs(c2[0]) < 0.2 * abs(lambda_pc2 + _DIV_GUARD)
    )

    is_line_attractor = pc1_neutral and pc2_contracts

    # Qualitative label
    if is_line_attractor:
        interpretation = "line_attractor"
    elif pc2_contracts and not pc1_neutral:
        interpretation = "partial_contraction (PC1 not neutral)"
    elif not pc2_contracts and pc1_neutral:
        interpretation = "neutral_plane (PC2 not contracting)"
    else:
        interpretation = "other"

    return {
        "dPC1_coefs": {"PC1": float(c1[0]), "PC2": float(c1[1]), "intercept": ic1},
        "dPC2_coefs": {"PC1": float(c2[0]), "PC2": float(c2[1]), "intercept": ic2},
        "lambda_pc2": lambda_pc2,
        "pc1_neutral": pc1_neutral,
        "pc2_contracts": pc2_contracts,
        "is_line_attractor": is_line_attractor,
        "interpretation": interpretation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _plot_pc_dynamics(pc_mid: np.ndarray, dpc: np.ndarray, pc_idx: int,
                      output_path: Path, model=None) -> None:
    """Scatter plot of PCi vs dPCi/dt with optional linear fit overlay."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping %s", output_path.name)
        return

    try:
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        pass

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(pc_mid, dpc, s=1, alpha=0.3, color="steelblue", rasterized=True)

    if model is not None:
        x_lin = np.linspace(pc_mid.min(), pc_mid.max(), 200)
        # Predict with the other PC fixed at 0 to show the marginal relationship
        X_lin = np.column_stack([x_lin, np.zeros_like(x_lin)])
        y_lin = model.predict(X_lin)
        ax.plot(x_lin, y_lin, color="red", linewidth=1.5, label="Linear fit (PC2=0)")
        ax.legend(fontsize=8)

    ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_xlabel(f"PC{pc_idx}")
    ax.set_ylabel(f"dPC{pc_idx}/dt")
    ax.set_title(f"PC{pc_idx} Dynamics")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_vector_field(pos: np.ndarray, model1, model2, output_path: Path,
                       n_grid: int = 30) -> None:
    """Quiver plot of the estimated 2-D velocity field."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping %s", output_path.name)
        return

    try:
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        pass

    pc1_vals = pos[:, 0]
    pc2_vals = pos[:, 1]

    x = np.linspace(pc1_vals.min(), pc1_vals.max(), n_grid)
    y = np.linspace(pc2_vals.min(), pc2_vals.max(), n_grid)
    Xg, Yg = np.meshgrid(x, y)

    grid_pts = np.column_stack([Xg.ravel(), Yg.ravel()])
    U = model1.predict(grid_pts).reshape(Xg.shape)
    V = model2.predict(grid_pts).reshape(Yg.shape)

    # Normalise arrow lengths for readability
    speed = np.sqrt(U ** 2 + V ** 2)
    max_speed = speed.max()
    if max_speed > 1e-12:
        U_norm = U / max_speed
        V_norm = V / max_speed
    else:
        U_norm, V_norm = U, V

    fig, ax = plt.subplots(figsize=(6, 6))
    q = ax.quiver(Xg, Yg, U_norm, V_norm, speed, cmap="viridis",
                  angles="xy", scale_units="xy", scale=n_grid / 4,
                  width=0.003, alpha=0.85)
    plt.colorbar(q, ax=ax, label="Speed (a.u.)")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Estimated 2-D Velocity Field\n(arrows normalised; colour = speed)")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


def _plot_phase_portrait(Z: np.ndarray, evr: np.ndarray,
                         output_path: Path, n_show: int = 10) -> None:
    """PC1-PC2 phase portrait with trajectory colour-coded by time."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping %s", output_path.name)
        return

    try:
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        pass

    n_traj = Z.shape[0]
    cmap = plt.get_cmap("plasma")

    fig, ax = plt.subplots(figsize=(6, 5))
    for i in range(min(n_show, n_traj)):
        pc1 = Z[i, :, 0]
        pc2 = Z[i, :, 1]
        T = len(pc1)
        colors = cmap(np.linspace(0, 1, T))
        ax.scatter(pc1, pc2, c=colors, s=0.5, alpha=0.6, rasterized=True)
        ax.plot(pc1[0], pc2[0], "o", color="blue", markersize=4, zorder=5)
        ax.plot(pc1[-1], pc2[-1], "^", color="red", markersize=4, zorder=5)

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var.)")
    ax.set_title(
        f"PC1-PC2 Phase Portrait ({min(n_show, n_traj)} traj)\n"
        "blue circle=start, red triangle=end, colour=time"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def draw(
    trajectories_path: Union[str, Path, None] = None,
    trajectories: Optional[np.ndarray] = None,
    output_dir: Union[str, Path, None] = None,
    burnin: int = 0,
    n_grid: int = 30,
    n_show: int = 10,
) -> Dict:
    """Analyse and visualise the 2-D effective dynamics for line-attractor structure.

    Reuses trajectory data already saved by the pipeline
    (``outputs/dynamics_pipeline/dynamics/trajectories.npy``).

    Either ``trajectories_path`` or ``trajectories`` must be provided.

    Args:
        trajectories_path: Path to ``trajectories.npy`` (shape ``(n_traj, T, N)``).
                           Ignored if *trajectories* is given directly.
        trajectories:      Pre-loaded array of shape ``(n_traj, T, N)``.
        output_dir:        Directory for output PNGs and JSON.  Defaults to
                           the same directory as *trajectories_path* or the
                           current working directory.
        burnin:            Steps to discard at the start of each trajectory
                           (removes initial transient).
        n_grid:            Grid resolution for vector-field quiver plot.
        n_show:            Maximum number of trajectories shown in the phase
                           portrait.

    Returns:
        Dict with keys:
          "pca_explained_variance_ratio" — (2,) array
          "linear_fit_r2"               — {"dPC1": float, "dPC2": float}
          "assessment"                  — line-attractor assessment dict
          "output_dir"                  — str path

    Raises:
        ValueError: if neither *trajectories_path* nor *trajectories* is given.
        FileNotFoundError: if *trajectories_path* does not exist.
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    if trajectories is None:
        if trajectories_path is None:
            raise ValueError("Provide either trajectories_path or trajectories array.")
        traj_path = Path(trajectories_path)
        if not traj_path.exists():
            raise FileNotFoundError(f"Trajectories file not found: {traj_path}")
        logger.info("Loading trajectories from %s", traj_path)
        trajectories = np.load(traj_path)
    else:
        traj_path = Path(trajectories_path) if trajectories_path is not None else None

    if trajectories.ndim != 3:
        raise ValueError(
            f"Expected trajectories shape (n_traj, T, N); got {trajectories.shape}"
        )

    n_traj, T, N = trajectories.shape
    logger.info("Trajectories: n_traj=%d, T=%d, N=%d, burnin=%d", n_traj, T, N, burnin)

    # ── Output directory ──────────────────────────────────────────────────────
    if output_dir is None:
        if traj_path is not None:
            output_dir = traj_path.parent
        else:
            output_dir = Path(".")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── Step 1: PCA projection ────────────────────────────────────────────────
    logger.info("Step 1/4 — PCA projection to 2 components")
    Z, pca, evr = _pca_project(trajectories, burnin=burnin)
    logger.info(
        "  PC1 explains %.1f%% variance, PC2 explains %.1f%%",
        evr[0] * 100, evr[1] * 100,
    )

    # ── Step 2: Velocity estimation ───────────────────────────────────────────
    logger.info("Step 2/4 — Finite-difference velocity estimation")
    pos, vel = _estimate_velocity(Z)
    logger.info("  %d velocity samples", len(pos))

    # ── Step 3: Linear regression ─────────────────────────────────────────────
    logger.info("Step 3/4 — Linear regression: dPCi = f(PC1, PC2)")
    model1, model2, r2_1, r2_2 = _fit_linear_field(pos, vel)

    c1 = model1.coef_
    c2 = model2.coef_
    logger.info(
        "  dPC1 = %.4f*PC1 + %.4f*PC2 + %.4f  (R²=%.3f)",
        c1[0], c1[1], float(model1.intercept_), r2_1,
    )
    logger.info(
        "  dPC2 = %.4f*PC1 + %.4f*PC2 + %.4f  (R²=%.3f)",
        c2[0], c2[1], float(model2.intercept_), r2_2,
    )

    assessment = _assess_line_attractor(model1, model2)
    logger.info(
        "  Assessment: %s  (PC1 neutral=%s, PC2 contracts=%s, λ_PC2=%.4f)",
        assessment["interpretation"],
        assessment["pc1_neutral"],
        assessment["pc2_contracts"],
        assessment["lambda_pc2"],
    )
    if assessment["is_line_attractor"]:
        logger.info(
            "  ✓ System is consistent with a LINE ATTRACTOR structure."
        )
    else:
        logger.info(
            "  ✗ System does not clearly exhibit line-attractor structure."
        )

    # ── Step 4: Plots ─────────────────────────────────────────────────────────
    logger.info("Step 4/4 — Generating plots")

    _plot_pc_dynamics(pos[:, 0], vel[:, 0], pc_idx=1,
                      output_path=out / "pc1_dynamics.png", model=model1)
    _plot_pc_dynamics(pos[:, 1], vel[:, 1], pc_idx=2,
                      output_path=out / "pc2_dynamics.png", model=model2)
    _plot_vector_field(pos, model1, model2,
                       output_path=out / "vector_field.png", n_grid=n_grid)
    _plot_phase_portrait(Z, evr,
                         output_path=out / "pc_phase_portrait.png", n_show=n_show)

    # ── Save JSON report ──────────────────────────────────────────────────────
    report = {
        "pca_explained_variance_ratio": evr.tolist(),
        "linear_fit": {
            "dPC1": {
                "coef_PC1": float(c1[0]),
                "coef_PC2": float(c1[1]),
                "intercept": float(model1.intercept_),
                "R2": r2_1,
            },
            "dPC2": {
                "coef_PC1": float(c2[0]),
                "coef_PC2": float(c2[1]),
                "intercept": float(model2.intercept_),
                "R2": r2_2,
            },
        },
        "assessment": assessment,
        "n_trajectories": int(n_traj),
        "T": int(T),
        "N_regions": int(N),
        "burnin": int(burnin),
        "output_dir": str(out),
    }

    report_path = out / "line_attractor_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved report: %s", report_path)

    return {
        "pca_explained_variance_ratio": evr,
        "linear_fit_r2": {"dPC1": r2_1, "dPC2": r2_2},
        "assessment": assessment,
        "output_dir": str(out),
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Line-attractor analysis of brain dynamics trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trajectories",
        default="outputs/dynamics_pipeline/dynamics/trajectories.npy",
        help="Path to trajectories.npy (shape n_traj × T × N)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output directory (default: same as --trajectories)",
    )
    p.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="Steps to discard at the start of each trajectory",
    )
    p.add_argument(
        "--n-grid",
        type=int,
        default=30,
        help="Grid resolution for vector-field quiver plot",
    )
    p.add_argument(
        "--n-show",
        type=int,
        default=10,
        help="Max trajectories to overlay in phase portrait",
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _parse_args()
    draw(
        trajectories_path=args.trajectories,
        output_dir=args.output,
        burnin=args.burnin,
        n_grid=args.n_grid,
        n_show=args.n_show,
    )
