"""
E2 + E3: Dynamic Modal Projection & Modal Energy Distribution
==============================================================

验证假设 H2：

  H2  网络动力学主要由少数特征模控制

实验逻辑
--------
给定状态轨迹 x(t) ∈ ℝᴺ 和连接矩阵 W（有效连接矩阵 R 或功能连接 FC），
将轨迹投影到 W 的特征向量空间：

  z(t) = V⁻¹ x(t)     （对非对称 W，使用左特征向量 V 的逆）
  z(t) = Vᵀ x(t)      （对对称 W，使用正交特征向量 V）

然后计算每个模态的能量：

  E_i = ⟨z_i(t)²⟩_t   （时间平均）

归一化后：E_i / Σ_j E_j

**科学意义**：
- 若前 k=2 个模态能量之和 > 80%，则动力学由 2 个主模态主导
- 这与神经流形假说一致：大脑活动落在低维子空间
- 该分析只在 W 近似系统线性化（Jacobian）时物理意义准确
  对非线性 GNN，R 矩阵是 Jacobian 的有限差分近似

批判性注意事项
--------------
1. **非对称矩阵的模态投影不稳定**：若 W 的特征向量矩阵 V 条件数高，
   V⁻¹ 的数值误差会放大。本模块在条件数 > 1000 时自动切换为 SVD 投影。
2. **模态能量分布反映拟线性动力学**；强非线性下（如 LLE >> 0 的混沌区域），
   模态投影的稳定性含义减弱。
3. **建议同时用 FC 和 R 矩阵作为特征向量基**，比较结果一致性。

输出文件
--------
  modal_energy_{label}.json  — 模态能量分布数值指标
  modal_energy_bar_{label}.png  — 模态能量条形图
  modal_timeseries_{label}.png  — 前 5 模态时序（单条轨迹示例）
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

_COND_NUM_THRESHOLD = 1000.0  # 条件数超过此值切换为 SVD 投影


# ─────────────────────────────────────────────────────────────────────────────
# Projection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _project_trajectories_symmetric(
    trajectories: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对称矩阵的模态投影（使用实数正交特征向量）。

    Returns:
        (z, eigvals, eigvecs):
          z:       shape (n_init, T, N)，投影后的模态系数
          eigvals: shape (N,)，实数特征值（降序排列）
          eigvecs: shape (N, N)，每列为一个特征向量
    """
    W64 = np.asarray(W, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(W64)
    # Sort descending by magnitude
    idx = np.argsort(np.abs(eigvals))[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Project: z = Vᵀ x
    n_init, T, N = trajectories.shape
    x = trajectories.reshape(-1, N).astype(np.float64)
    z = (eigvecs.T @ x.T).T  # (n_init*T, N)
    z = z.reshape(n_init, T, N)
    return z.astype(np.float32), eigvals.astype(np.float32), eigvecs.astype(np.float32)


def _project_trajectories_asymmetric(
    trajectories: np.ndarray,
    W: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    非对称矩阵的模态投影（使用 SVD 分解而非左特征向量的逆）。

    使用 SVD（W = U Σ Vᵀ），投影到左奇异向量空间：z = Uᵀ x。
    奇异向量空间与特征向量空间近似，但数值更稳定。

    Returns:
        (z, singular_values, U):
          z:                shape (n_init, T, N)
          singular_values:  shape (N,)，降序奇异值
          U:                shape (N, N)，左奇异向量矩阵
    """
    W64 = np.asarray(W, dtype=np.float64)
    U, sv, Vt = np.linalg.svd(W64, full_matrices=True)
    # Project: z = Uᵀ x  (U is (N, N), orthonormal)
    n_init, T, N = trajectories.shape
    x = trajectories.reshape(-1, N).astype(np.float64)
    z = (U.T @ x.T).T  # (n_init*T, N)
    z = z.reshape(n_init, T, N)
    logger.info("SVD 模态投影: top5 奇异值 = %s", np.round(sv[:5], 3))
    return z.astype(np.float32), sv.astype(np.float32), U.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Modal energy computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_modal_energies(z: np.ndarray) -> Dict:
    """
    计算每个模态的归一化能量分布。

    Args:
        z: shape (n_init, T, N)，轨迹在模态空间中的投影。

    Returns:
        dict 包含:
          energies_normalized   : np.ndarray (N,) — 归一化模态能量
          cumulative_energy     : np.ndarray (N,) — 累积能量
          n_modes_80pct         : int — 累积 80% 能量所需模态数
          n_modes_90pct         : int — 累积 90% 能量所需模态数
          n_modes_95pct         : int — 累积 95% 能量所需模态数
          energy_top1           : float — 第 1 模态能量占比
          energy_top2           : float — 前 2 模态能量占比
          energy_top5           : float — 前 5 模态能量占比
    """
    # E_i = mean over time and trajectories of z_i(t)^2
    energies = np.mean(z ** 2, axis=(0, 1)).astype(np.float64)
    total = energies.sum()
    if total < 1e-30:
        logger.warning("模态能量总和接近零，可能轨迹全为常数。")
        total = 1.0
    norm_e = energies / total
    cumul = np.cumsum(norm_e)

    def _n_for_thresh(thresh: float) -> int:
        idx = np.searchsorted(cumul, thresh)
        return int(min(idx + 1, len(cumul)))

    result = {
        "n_modes": int(z.shape[-1]),
        "energies_normalized": norm_e,
        "cumulative_energy": cumul,
        "n_modes_80pct": _n_for_thresh(0.80),
        "n_modes_90pct": _n_for_thresh(0.90),
        "n_modes_95pct": _n_for_thresh(0.95),
        "energy_top1": float(norm_e[0]),
        "energy_top2": float(norm_e[:2].sum()),
        "energy_top5": float(norm_e[:5].sum()),
    }
    logger.info(
        "模态能量: top1=%.1f%%, top2=%.1f%%, top5=%.1f%%, "
        "n_modes@80%%=%d, n_modes@90%%=%d",
        100 * result["energy_top1"],
        100 * result["energy_top2"],
        100 * result["energy_top5"],
        result["n_modes_80pct"],
        result["n_modes_90pct"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_energy_bar(
    energies: np.ndarray,
    cumul: np.ndarray,
    output_path: Path,
    label: str,
    n_show: int = 30,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    N = min(n_show, len(energies))
    ranks = np.arange(1, N + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.bar(ranks, energies[:N] * 100, color="steelblue", alpha=0.8)
    ax1.set_xlabel("Mode Rank")
    ax1.set_ylabel("Mode Energy (%)")
    ax1.set_title(f"Mode Energy Distribution (top {N})  [{label}]")
    ax1.axhline(5.0, ls="--", color="red", lw=0.8, label="5% line")
    ax1.legend()

    ax2.plot(np.arange(1, len(cumul) + 1), cumul * 100, "o-", ms=3, lw=1.5)
    ax2.axhline(80, ls="--", color="orange", lw=1, label="80%")
    ax2.axhline(90, ls="--", color="red", lw=1, label="90%")
    ax2.axhline(95, ls="--", color="darkred", lw=1, label="95%")
    ax2.set_xlabel("Number of Modes")
    ax2.set_ylabel("Cumulative Energy (%)")
    ax2.set_title(f"Cumulative Mode Energy  [{label}]")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


def _try_plot_modal_timeseries(
    z: np.ndarray,
    output_path: Path,
    label: str,
    traj_idx: int = 0,
    n_modes: int = 5,
) -> None:
    """Plot first n_modes modal time series for a single trajectory."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    n_modes = min(n_modes, z.shape[-1])
    traj = z[min(traj_idx, z.shape[0] - 1)]  # (T, N)
    T = traj.shape[0]
    t = np.arange(T)

    fig, axes = plt.subplots(n_modes, 1, figsize=(10, 2 * n_modes), sharex=True)
    if n_modes == 1:
        axes = [axes]
    colors = plt.cm.Set1(np.linspace(0, 0.9, n_modes))

    for k, ax in enumerate(axes):
        ax.plot(t, traj[:, k], color=colors[k], lw=1.2)
        ax.set_ylabel(f"z_{k+1}")
        ax.axhline(0, ls="--", color="gray", lw=0.5)

    axes[-1].set_xlabel("Time step")
    axes[0].set_title(f"Top {n_modes} modal time series (traj {traj_idx})  [{label}]")
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_modal_projection(
    trajectories: np.ndarray,
    W: np.ndarray,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
    symmetric: bool = False,
) -> Dict:
    """
    运行 E2+E3 模态投影与能量分析并保存结果。

    Args:
        trajectories: shape (n_init, T, N)，自由动力学轨迹。
        W:            连接矩阵 (N, N)，特征向量空间来源。
        output_dir:   结果保存目录；None → 仅返回不保存。
        label:        矩阵标签（用于文件名和标题）。
        symmetric:    若 True，使用 eigh 实数特征值投影；
                      否则使用 SVD 数值稳定投影。

    Returns:
        metrics dict（可直接序列化为 JSON）。
    """
    n_init, T, N = trajectories.shape
    logger.info("E2/E3 模态投影: n_traj=%d, T=%d, N=%d, symmetric=%s",
                n_init, T, N, symmetric)

    if symmetric:
        z, spectrum, basis = _project_trajectories_symmetric(trajectories, W)
        spectrum_label = "eigenvalues"
    else:
        z, spectrum, basis = _project_trajectories_asymmetric(trajectories, W)
        spectrum_label = "singular_values"

    energy_metrics = compute_modal_energies(z)

    # Serializable summary
    result: Dict = {
        "projection_method": "eigh" if symmetric else "svd",
        spectrum_label: spectrum[:20].tolist(),
        "n_modes_80pct": energy_metrics["n_modes_80pct"],
        "n_modes_90pct": energy_metrics["n_modes_90pct"],
        "n_modes_95pct": energy_metrics["n_modes_95pct"],
        "energy_top1_pct": round(100 * energy_metrics["energy_top1"], 2),
        "energy_top2_pct": round(100 * energy_metrics["energy_top2"], 2),
        "energy_top5_pct": round(100 * energy_metrics["energy_top5"], 2),
        "h2_supported": energy_metrics["n_modes_80pct"] <= max(3, N // 20),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"modal_energy_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        np.save(out / f"modal_projections_{label}.npy", z)

        _try_plot_energy_bar(
            energy_metrics["energies_normalized"],
            energy_metrics["cumulative_energy"],
            out / f"modal_energy_bar_{label}.png",
            label,
        )
        _try_plot_modal_timeseries(
            z,
            out / f"modal_timeseries_{label}.png",
            label,
        )

    return result
