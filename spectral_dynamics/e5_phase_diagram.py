"""
E5: Coupling Strength Phase Diagram
=====================================

扫描耦合强度 g，建立"稳定→振荡→混沌"相图，验证假设 H3：

  H3  谱结构使系统自然接近临界（g ≈ 1 时振荡，即边缘混沌）

实验逻辑
--------
定义缩放后的 WC 动力学：

  x(t+1) = clip(tanh(g · W · x(t)), 0, 1)

其中 W 为真实连接矩阵（响应矩阵 R 或功能连接 FC），g 为耦合强度系数。

对 g ∈ [g_min, g_max]（默认 [0.1, 3.0]，步长 0.1）计算三个指标：

1. **Lyapunov 指数 LLE**：Rosenstein 法，区分稳定/振荡/混沌
2. **振荡幅度**：mean[std over time of x(t)]，反映动力学活跃度
3. **谱半径 ρ(g·W)**：线性近似的稳定性判据

**关键物理意义**：
- g·ρ(W) < 1：线性稳定区（系统衰减到不动点）
- g·ρ(W) ≈ 1：线性临界点（但 tanh 非线性使实际边界约为 g·ρ ≈ 1.5，见 AGENTS.md）
- LLE ≈ 0：边缘混沌（Edge of Chaos），计算最优性能区域
- LLE > 0：混沌区，长期预测不可靠

**如果真实连接矩阵的谱半径 ρ(W) ≈ 1，则 g=1 时系统自然处于临界附近。**
这正是验证 H3 的关键观察点。

批判性注意事项
--------------
1. **WC 模型是真实大脑的极简化**：tanh 激活函数在神经科学上有根据，
   但 GNN 的实际动力学可能截然不同。相图仅对 WC 框架有效。
2. **LLE 对短轨迹的估计误差较大**：建议 steps >= 200，n_traj >= 20。
   本实验每个 g 值运行 n_traj 条轨迹，取中位数减少噪声。
3. **振荡幅度与 LLE 并非单调关系**：在混沌区，振荡幅度可能反而降低
   （系统在多个吸引子间快速跳转，平均幅度小）。需结合两者解读。
4. **相图的 g 轴应解读为"相对于实际系统的耦合倍数"**，而非绝对值。

输出文件
--------
  phase_diagram_{label}.json    — 所有 g 值的指标数值
  phase_diagram_{label}.png     — 三子图相图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Shared WC utilities — canonical implementations in twinbrain-dynamics
from analysis.wc_dynamics import wc_simulate as _wc_simulate, rosenstein_lle_on_wc as _rosenstein_lle_on_wc

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics per g
# ─────────────────────────────────────────────────────────────────────────────


def _compute_oscillation_amplitude(trajs: np.ndarray) -> float:
    """
    计算振荡幅度：所有轨迹所有脑区的时间标准差的均值。

    高 std → 动力学活跃（振荡或混沌）
    低 std → 收敛到不动点

    Args:
        trajs: shape (n_traj, T, N)

    Returns:
        scalar，振荡幅度。
    """
    per_traj_per_region_std = trajs.std(axis=1)  # (n_traj, N)
    return float(per_traj_per_region_std.mean())


def _lle_from_trajs(trajs: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """
    Compute median LLE over pre-generated WC trajectories.

    Delegates to ``analysis.wc_dynamics.rosenstein_lle_on_wc`` (canonical
    implementation); the duplicate ``_rosenstein_from_twinbrain`` and
    ``_simple_rosenstein`` functions have been removed.
    """
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        lles = [
            rosenstein_lyapunov(trajs[i], max_lag=max_lag, min_temporal_sep=min_sep)[0]
            for i in range(len(trajs))
        ]
    except ImportError:
        from analysis.wc_dynamics import wolf_benettin_lle
        lles = [wolf_benettin_lle(trajs[i]) for i in range(len(trajs))]

    valid = [v for v in lles if np.isfinite(v)]
    return float(np.median(valid)) if valid else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Phase diagram sweep
# ─────────────────────────────────────────────────────────────────────────────

def run_phase_diagram(
    W: np.ndarray,
    g_min: float = 0.1,
    g_max: float = 3.0,
    g_step: float = 0.1,
    n_traj: int = 20,
    steps: int = 300,
    warmup: int = 50,
    max_lag: int = 30,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    seed: int = 42,
) -> Dict:
    """
    扫描耦合强度 g，计算 LLE、振荡幅度、谱半径。

    Args:
        W:          连接矩阵 (N, N)。
        g_min/max:  耦合强度扫描范围。
        g_step:     步长。
        n_traj:     每个 g 值运行的轨迹数。
        steps:      每条轨迹步数（不含 warmup）。
        warmup:     热身步数（不记录）。
        max_lag:    Rosenstein 最大追踪滞后。
        output_dir: 结果保存目录。
        label:      矩阵标签。
        seed:       随机种子。

    Returns:
        dict 包含:
          g_values, spectral_radii, lles, oscillation_amplitudes
          critical_g_lle:   LLE 最接近 0 的 g 值
          actual_rho:       W 的谱半径（g=1 时的线性临界 g_critical = 1/rho）
          g_linear_critical: 线性临界点（1/rho(W)）
    """
    g_values = np.arange(g_min, g_max + g_step / 2, g_step)
    g_values = np.round(g_values, 3)

    # Spectral radius of original W
    eigvals_W = np.linalg.eigvals(W.astype(np.float64))
    rho_W = float(np.abs(eigvals_W).max())
    g_linear_critical = 1.0 / rho_W if rho_W > 1e-8 else float("inf")

    lles: List[float] = []
    osc_amps: List[float] = []
    spectral_radii: List[float] = []

    logger.info("E5 相图扫描: g ∈ [%.1f, %.1f], step=%.2f, n_traj=%d",
                g_min, g_max, g_step, n_traj)

    for g in g_values:
        gW = (g * W).astype(np.float64)
        rho_gW = float(np.abs(np.linalg.eigvals(gW)).max())
        spectral_radii.append(rho_gW)

        trajs = _wc_simulate(W, n_traj=n_traj, steps=steps, g=g,
                             warmup=warmup, seed=seed)
        lle = _lle_from_trajs(trajs, max_lag=max_lag)
        osc = _compute_oscillation_amplitude(trajs)

        lles.append(lle)
        osc_amps.append(osc)
        logger.debug("  g=%.2f: rho=%.3f, LLE=%.4f, amp=%.4f", g, rho_gW, lle, osc)

    # Find edge-of-chaos g (LLE closest to 0)
    lle_arr = np.array(lles)
    valid_mask = np.isfinite(lle_arr)
    if valid_mask.any():
        critical_idx = int(np.argmin(np.abs(lle_arr[valid_mask])))
        critical_g = float(g_values[valid_mask][critical_idx])
        critical_lle = float(lle_arr[valid_mask][critical_idx])
    else:
        critical_g = float("nan")
        critical_lle = float("nan")

    result = {
        "g_values": g_values.tolist(),
        "spectral_radii": spectral_radii,
        "lles": lles,
        "oscillation_amplitudes": osc_amps,
        "actual_rho_W": rho_W,
        "g_linear_critical": round(g_linear_critical, 3),
        "critical_g_lle0": round(critical_g, 3),
        "critical_lle": round(critical_lle, 5) if np.isfinite(critical_lle) else None,
        "h3_supported": abs(critical_g - 1.0) < 0.3 or abs(g_linear_critical - 1.0) < 0.2,
    }

    logger.info(
        "E5 结果: ρ(W)=%.3f, g_linear_critical=%.2f, critical_g_lle≈0=%.2f",
        rho_W, g_linear_critical, critical_g,
    )

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
    """绘制三子图相图：LLE / 振荡幅度 / 谱半径 vs g。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    g = np.array(result["g_values"])
    lles = np.array(result["lles"], dtype=float)
    osc = np.array(result["oscillation_amplitudes"])
    rhos = np.array(result["spectral_radii"])

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # LLE
    axes[0].plot(g, lles, "b-o", ms=3, lw=1.5)
    axes[0].axhline(0, ls="--", color="red", lw=1, label="LLE=0 (边缘混沌)")
    axes[0].axvline(result["g_linear_critical"], ls=":", color="gray", lw=1,
                    label=f"线性临界 g={result['g_linear_critical']:.2f}")
    axes[0].axvline(result["critical_g_lle0"], ls="--", color="orange", lw=1,
                    label=f"LLE≈0 g={result['critical_g_lle0']:.2f}")
    axes[0].set_ylabel("Lyapunov LLE")
    axes[0].legend(fontsize=7)
    axes[0].set_title(f"耦合强度相图  [{label}]")

    # Oscillation amplitude
    axes[1].plot(g, osc, "g-o", ms=3, lw=1.5)
    axes[1].axvline(result["g_linear_critical"], ls=":", color="gray", lw=1)
    axes[1].set_ylabel("振荡幅度 (std)")

    # Spectral radius
    axes[2].plot(g, rhos, "k-o", ms=3, lw=1.5)
    axes[2].axhline(1.0, ls="--", color="red", lw=1, label="ρ=1 (线性临界)")
    axes[2].set_ylabel("谱半径 ρ(g·W)")
    axes[2].set_xlabel("耦合强度 g")
    axes[2].legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)
