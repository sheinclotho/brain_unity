"""
E5: Coupling Strength Phase Diagram
=====================================

扫描耦合强度 g，建立"稳定→临界→不稳定"相图，验证假设 H3：

  H3  谱结构使系统自然接近临界（g ≈ 1 时接近临界点）

实验逻辑
--------
对 g ∈ [g_min, g_max] 计算：

1. **谱半径 ρ(g·W)**：线性稳定性判据（analytical, no simulation）
   - g·ρ(W) < 1：线性稳定区
   - g·ρ(W) = 1：线性临界点
   - g·ρ(W) > 1：线性不稳定区

2. **LLE**（可选）：若传入 GNN 预计算轨迹（trajectories 参数），
   则在 g=1 处用 Rosenstein 法估计一个参考 LLE 值。
   其余 g 值的 LLE 不做估计（返回 NaN），因为不同 g 对应不同模型，
   应由 twinbrain-dynamics 管线在不同缩放参数下分别运行。

**如果真实连接矩阵的谱半径 ρ(W) ≈ 1，则 g=1 时系统自然处于临界附近。**
这正是验证 H3 的关键观察点。

输出文件
--------
  phase_diagram_{label}.json    — 所有 g 值的谱半径数值 + 参考 LLE
  phase_diagram_{label}.png     — 谱半径 vs g 相图（含参考 LLE 标注）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Utility functions (operate on pre-computed trajectories from GNN)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_oscillation_amplitude(trajs: np.ndarray) -> float:
    """
    计算振荡幅度：所有轨迹所有脑区的时间标准差的均值。

    Args:
        trajs: shape (n_traj, T, N), pre-computed from GNN model.

    Returns:
        scalar，振荡幅度。
    """
    per_traj_per_region_std = trajs.std(axis=1)  # (n_traj, N)
    return float(per_traj_per_region_std.mean())


def _lle_from_trajs(trajs: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """
    Compute median LLE from pre-generated GNN trajectories.

    Delegates to canonical ``analysis.lyapunov.rosenstein_lyapunov``
    from twinbrain-dynamics when available on sys.path.

    Args:
        trajs:   shape (n_traj, T, N), GNN model trajectories.
        max_lag: Rosenstein tracking lag.
        min_sep: Minimum temporal separation.

    Returns:
        Median LLE across trajectories; NaN if unavailable.
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        lles = [
            rosenstein_lyapunov(trajs[i], max_lag=max_lag, min_temporal_sep=min_sep)[0]
            for i in range(len(trajs))
        ]
    except ImportError:
        logger.warning("_lle_from_trajs: analysis.lyapunov not available.")
        return float("nan")

    valid = [v for v in lles if np.isfinite(v)]
    return float(np.median(valid)) if valid else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Phase diagram sweep (analytical)
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_diagram(
    W: np.ndarray,
    g_min: float = 0.1,
    g_max: float = 3.0,
    g_step: float = 0.1,
    lle_reference: Optional[float] = None,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    扫描耦合强度 g，计算谱半径（解析），验证假设 H3。

    本函数只做纯谱分析（无模型调用、无动力学仿真）。
    LLE 参考值由流水线（run_all 或 dynamics_pipeline）在前序步骤中
    用 GNN 轨迹估计后以 lle_reference 传入；若未提供则不标注。

    Args:
        W:             连接矩阵 (N, N)，来自 TwinBrain GNN 输出。
        g_min/max:     耦合强度扫描范围。
        g_step:        步长。
        lle_reference: 由流水线在 g=1 处估计的 GNN LLE 参考值（可选）。
                       不应由用户手动提供——应由 run_all 内部计算后传入。
        output_dir:    结果保存目录。
        label:         矩阵标签。

    Returns:
        dict 包含:
          g_values, spectral_radii, actual_rho_W,
          g_linear_critical, lle_at_g1, h3_supported
    """
    g_values = np.arange(g_min, g_max + g_step / 2, g_step)
    g_values = np.round(g_values, 3)

    eigvals_W = np.linalg.eigvals(W.astype(np.float64))
    rho_W = float(np.abs(eigvals_W).max())
    g_linear_critical = 1.0 / rho_W if rho_W > 1e-8 else float("inf")

    spectral_radii: List[float] = []
    for g in g_values:
        gW = (g * W).astype(np.float64)
        rho_gW = float(np.abs(np.linalg.eigvals(gW)).max())
        spectral_radii.append(rho_gW)

    logger.info(
        "E5 相图: ρ(W)=%.3f, g_linear_critical=%.2f  (谱半径解析扫描)",
        rho_W, g_linear_critical,
    )
    if lle_reference is not None:
        logger.info("  g=1 LLE 参考值 (来自流水线): %.4f", lle_reference)

    result = {
        "g_values": g_values.tolist(),
        "spectral_radii": spectral_radii,
        "actual_rho_W": rho_W,
        "g_linear_critical": round(g_linear_critical, 3),
        "lle_at_g1": round(lle_reference, 5) if lle_reference is not None else None,
        "h3_supported": abs(rho_W - 1.0) < 0.2,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"phase_diagram_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("保存相图数据: %s", json_path)
        _try_plot_phase_diagram(result, out / f"phase_diagram_{label}.png", label)

    return result


def _try_plot_phase_diagram(result: Dict, output_path: Path, label: str) -> None:
    """Plot spectral radius vs coupling strength g phase diagram."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    g = np.array(result["g_values"])
    rhos = np.array(result["spectral_radii"])

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(g, rhos, "k-o", ms=3, lw=1.5, label="ρ(g·W)")
    ax.axhline(1.0, ls="--", color="red", lw=1, label="rho=1 (linear critical)")
    ax.axvline(result["g_linear_critical"], ls=":", color="gray", lw=1,
               label=f"g_critical={result['g_linear_critical']:.2f}")

    # Mark reference LLE if available
    lle_ref = result.get("lle_at_g1")
    if lle_ref is not None:
        g1_idx = int(np.argmin(np.abs(g - 1.0)))
        ax.annotate(
            f"GNN LLE={lle_ref:.3f}",
            xy=(g[g1_idx], rhos[g1_idx]),
            xytext=(g[g1_idx] + 0.2, rhos[g1_idx] + 0.1),
            fontsize=8, color="blue",
            arrowprops=dict(arrowstyle="->", color="blue", lw=0.8),
        )

    ax.set_xlabel("Coupling strength g")
    ax.set_ylabel("Spectral radius rho(g·W)")
    ax.set_title(f"Coupling Strength Phase Diagram (spectral radius scan)  [{label}]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)

