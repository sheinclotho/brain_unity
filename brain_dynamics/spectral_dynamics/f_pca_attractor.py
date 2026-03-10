"""
Module F: PCA Dimensionality & Attractor Projection
======================================================

用主成分分析（PCA）从两个角度验证低维动力学：

F1 解释方差曲线
---------------
  PCA 把 T × N 轨迹矩阵投影到 N 个主成分方向，按解释方差排序。
  若前 k 个 PC 累积方差 > 90%，动力学维度 ≤ k（线性估计）。
  
  - 输出：pca_variance_curve_{label}.png
  - 记录：variance_top2, variance_top5, n_components_90pct

F2 吸引子投影（Attractor Visualization）
-----------------------------------------
  PC1-PC2 散点图（2D）和 PC1-PC2-PC3 轨迹图（3D）是直观展示吸引子
  几何结构的最简方法。

  - 颜色编码时间（蓝→红）可以看出轨迹演化方向
  - 多条轨迹叠加可以检测收敛性（吸引子的迹象）

  - 输出：attractor_projection_2d_{label}.png
            attractor_projection_3d_{label}.png

**PCA vs Takens 嵌入 vs FNN 的比较**：

| 方法 | 前提假设 | 优点 | 缺点 |
|-----|---------|-----|------|
| PCA (本模块) | 线性降维 | 直观、快速 | 低估非线性维度 |
| Takens 嵌入 | 单通道观测 | 理论保证 | 需要正确的 m 和 τ |
| FNN/D₂ | 无 | 非线性 | 计算慢、对噪声敏感 |

→ PCA 是快速初步估计的首选；非线性动力学应与 FNN 结果对比。

**科学批判**：
1. PCA 是线性方法。若吸引子是非线性流形（如螺旋、Lorenz 吸引子），
   PCA 会**高估**所需维度（因为需要多个 PC 才能覆盖弯曲流形）。
2. 轨迹的 PCA 与状态空间的 PCA 不同：
   - 时间轴 PCA（T 样本 × N 特征）→ 本模块的标准做法
   - 空间轴 PCA（N 样本 × T 特征）→ "functional connectivity" 的概念
3. Burn-in 步骤（跳过前 burnin 步）很重要：
   若包含瞬态（trajectory 从初始状态收敛到吸引子的过程），
   PCA 的主方向会被瞬态污染，低估吸引子的真实维度。

输出文件
--------
  pca_variance_curve_{label}.png      — 解释方差曲线
  attractor_projection_2d_{label}.png — 2D 吸引子投影
  attractor_projection_3d_{label}.png — 3D 吸引子投影
  pca_results_{label}.json            — 数值摘要
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sklearn.decomposition import PCA
from spectral_dynamics.plot_utils import write_fallback_png as _write_fallback_png

logger = logging.getLogger(__name__)

# Poincaré section constants
_POINCARE_CROSSING_EPS: float = 1e-12   # threshold for near-zero slope in crossing interpolation
_POINCARE_PERIODIC_THRESH: float = 1e-6  # range < this → trivially periodic (fixed point)
_POINCARE_TIGHT_RATIO: float = 0.15      # std/range < this → clustered (periodic)
_POINCARE_QUASI_RATIO: float = 0.50      # std/range < this → smooth manifold (quasi-periodic)


# ─────────────────────────────────────────────────────────────────────────────
# Delay-embedding helpers (Takens' theorem)
# ─────────────────────────────────────────────────────────────────────────────

def _estimate_tau(obs: np.ndarray, max_tau: int = 100) -> int:
    """Estimate optimal delay τ as the first zero-crossing of the
    autocorrelation function (Takens / Fraser-Swinney criterion).

    Args:
        obs:     1-D observation time series (float64).
        max_tau: Maximum τ to search.

    Returns:
        τ (int, ≥ 1).  Falls back to 1 if no zero-crossing is found.
    """
    obs = obs - obs.mean()
    norm = float(np.dot(obs, obs))
    if norm < 1e-30:
        return 1
    for tau in range(1, min(max_tau, len(obs) // 4)):
        acf = float(np.dot(obs[:-tau], obs[tau:])) / norm
        if acf <= 0.0:
            return tau
    return max(1, min(max_tau, len(obs) // 8))


def _build_delay_portrait(
    trajectories: np.ndarray,
    burnin: int,
    m: int = 3,
    tau: Optional[int] = None,
) -> np.ndarray:
    """Build Takens delay-embedding coordinates from multi-channel trajectories.

    For each trajectory the *mean across all channels* is used as the scalar
    observation y(t).  The delay portrait is then::

        z(t) = [y(t), y(t+τ), y(t+2τ), ..., y(t+(m-1)τ)]

    which, by Takens' theorem, preserves the topological structure of the
    attractor for m ≥ 2D+1 (D = intrinsic dimension).  This avoids the
    flattening artefacts of PCA linear projection.

    Args:
        trajectories: shape (n_traj, T, N).
        burnin:       Steps to skip at the start of each trajectory.
        m:            Embedding dimension (default 3 for 2-D / 3-D portraits).
        tau:          Time delay (steps).  If None, estimated automatically
                      from the mean-channel autocorrelation of the first
                      trajectory.

    Returns:
        X_delay: shape (n_traj * T_embed, m) — all trajectories stacked.
        tau_used: int — the τ actually used.
        T_embed: int — number of delay-vector time points per trajectory.
    """
    n_traj, T, N = trajectories.shape
    burnin = min(burnin, T - 1)

    # Scalar observable: mean over all channels (after burnin)
    obs_all = trajectories[:, burnin:, :].mean(axis=-1)   # (n_traj, T_eff)
    T_eff = obs_all.shape[1]

    # Estimate τ from first trajectory if not supplied
    if tau is None:
        tau = _estimate_tau(obs_all[0].astype(np.float64))

    T_embed = T_eff - (m - 1) * tau
    if T_embed < 4:
        # Fallback stage 1: halve τ
        tau = max(1, tau // 2)
        T_embed = T_eff - (m - 1) * tau
    if T_embed < 4:
        # Fallback stage 2: minimum-possible embedding (m=2, τ=1 for m-param,
        # overriding any caller-supplied m > 2 with a warning).
        if m != 3:
            logger.warning(
                "_build_delay_portrait: reducing m from %d to 3 (τ=1) "
                "because T_eff=%d is too short for the requested embedding.",
                m, T_eff,
            )
        m = 3
        tau = 1
        T_embed = T_eff - (m - 1) * tau
    if T_embed < 4:
        # Trajectories are too short even for the minimal embedding;
        # return an empty array so callers can skip plotting gracefully.
        logger.warning(
            "_build_delay_portrait: T_eff=%d too short for delay embedding "
            "(need T_eff >= 4 + 2*tau); returning empty array.",
            T_eff,
        )
        return np.empty((0, m), dtype=np.float64), tau, 0

    # Build delay portrait.
    # Each slice obs[j*tau : j*tau + T_embed] has exactly T_embed elements
    # by construction: for j in [0, m-1], end = j*tau + T_embed
    # ≤ (m-1)*tau + T_embed = T_eff (by definition of T_embed). ✓
    rows = []
    for i in range(n_traj):
        obs = obs_all[i].astype(np.float64)
        # Centre each trajectory's observable around its mean so that
        # zero-crossings in compute_poincare_section correspond to crossings
        # through the trajectory's own mean level (not the global zero).
        # This is standard practice in Poincaré-section analysis and does not
        # change the topological structure of the attractor.
        obs = obs - obs.mean()
        mat = np.stack([obs[j * tau: j * tau + T_embed] for j in range(m)], axis=1)
        rows.append(mat)

    X_delay = np.concatenate(rows, axis=0)   # (n_traj * T_embed, m)
    return X_delay, tau, T_embed


# ─────────────────────────────────────────────────────────────────────────────
# F1: PCA explained variance
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca(
    trajectories: np.ndarray,
    burnin: int = 0,
    n_components: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    对自由动力学轨迹运行 PCA，计算解释方差分布。

    所有轨迹拼接后做 PCA（T_total = n_traj × steps，去掉 burnin 步）。

    Args:
        trajectories:  shape (n_traj, T, N)。
        burnin:        每条轨迹跳过的前 burnin 步（去除瞬态）。
        n_components:  保留的 PC 数；None → min(T_total, N)。
        seed:          随机种子（对 randomized SVD 的 PCA 影响）。

    Returns:
        dict 包含:
          explained_variance_ratio:  list (n_components,)
          cumulative_variance:       list (n_components,)
          n_components_50pct:        int
          n_components_80pct:        int
          n_components_90pct:        int
          n_components_95pct:        int
          variance_top1:             float (%)
          variance_top2:             float (%)
          variance_top5:             float (%)
          pca_object:                sklearn PCA (not JSON-serializable)
          X_pca:                     np.ndarray (T_total, n_components)
    """
    n_traj, T, N = trajectories.shape
    burnin = min(burnin, T - 1)

    # Stack: (n_traj * (T - burnin), N)
    X = trajectories[:, burnin:, :].reshape(-1, N).astype(np.float64)

    if n_components is None:
        n_components = min(X.shape[0], N)
    n_components = min(n_components, min(X.shape))

    pca = PCA(n_components=n_components, random_state=seed, svd_solver="full")
    X_pca = pca.fit_transform(X)

    evr = pca.explained_variance_ratio_
    cumul = np.cumsum(evr)

    def _n_for_thresh(thresh: float) -> int:
        idx = np.searchsorted(cumul, thresh)
        return int(min(idx + 1, len(cumul)))

    result = {
        "n_samples": int(X.shape[0]),
        "n_features": N,
        "n_components": n_components,
        "explained_variance_ratio": evr.tolist(),
        "cumulative_variance": cumul.tolist(),
        "n_components_50pct": _n_for_thresh(0.50),
        "n_components_80pct": _n_for_thresh(0.80),
        "n_components_90pct": _n_for_thresh(0.90),
        "n_components_95pct": _n_for_thresh(0.95),
        "variance_top1": round(float(evr[0]) * 100, 2),
        "variance_top2": round(float(evr[:2].sum()) * 100, 2) if len(evr) >= 2 else None,
        "variance_top5": round(float(evr[:5].sum()) * 100, 2) if len(evr) >= 5 else None,
        # Non-serializable extras (stripped before JSON save)
        "pca_object": pca,
        "X_pca": X_pca,
    }
    logger.info(
        "PCA: top1=%.1f%%, top2=%.1f%%, top5=%.1f%%, "
        "n@80%%=%d, n@90%%=%d",
        result["variance_top1"],
        result["variance_top2"] or 0,
        result["variance_top5"] or 0,
        result["n_components_80pct"],
        result["n_components_90pct"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# F2: Attractor projection
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_variance_curve(
    evr: List[float],
    cumul: List[float],
    output_path: Path,
    label: str,
    n_components_90pct: int,
) -> None:
    """F1: 解释方差曲线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    n = len(evr)
    ranks = np.arange(1, n + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Bar: individual explained variance
    ax1.bar(ranks[:min(30, n)], np.array(evr[:min(30, n)]) * 100,
            color="steelblue", alpha=0.8, edgecolor="k", lw=0.3)
    ax1.set_xlabel("PC Rank")
    ax1.set_ylabel("Explained Variance (%)")
    ax1.set_title(f"PCA Explained Variance (Top 30 PCs)  [{label}]")

    # Cumulative
    ax2.plot(ranks, np.array(cumul) * 100, "o-", ms=3, lw=1.5)
    ax2.axhline(80, ls="--", color="orange", lw=1, label="80%")
    ax2.axhline(90, ls="--", color="red", lw=1, label="90%")
    ax2.axhline(95, ls="--", color="darkred", lw=1, label="95%")
    ax2.axvline(n_components_90pct, ls=":", color="green", lw=1,
                label=f"90%@PC{n_components_90pct}")
    ax2.set_xlabel("Number of PCs")
    ax2.set_ylabel("Cumulative Explained Variance (%)")
    ax2.set_title(f"Cumulative Explained Variance  [{label}]")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


def _try_plot_attractor_2d(
    X_pca: np.ndarray,
    n_traj: int,
    steps_per_traj: int,
    output_path: Path,
    label: str,
    max_traj_show: int = 8,
) -> None:
    """F2 2D: PC1 vs PC2 attractor projection.

    Two-panel layout:
      Left  — density heatmap (hexbin) of all trajectory points: reveals where
              the trajectory *spends its time* (attractor invariant measure).
      Right — sparse trajectory overlay (subsampled): reveals *how* the
              trajectory moves through the attractor.

    Replaces the old per-segment T-1 loop which produced visually noisy,
    uninterpretable plots when T was large.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    n_show = min(max_traj_show, n_traj)
    T = steps_per_traj
    # Subsampling stride: target ~100 visible points per trajectory.
    stride = max(1, T // 100)

    # Collect all points for the density panel (exactly n_show trajectories)
    all_pts = X_pca[:n_show * T, :2]

    fig, (ax_dens, ax_traj) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left panel: density heatmap ──────────────────────────────────────────
    hb = ax_dens.hexbin(
        all_pts[:, 0], all_pts[:, 1],
        gridsize=40, cmap="YlOrRd", mincnt=1,
    )
    plt.colorbar(hb, ax=ax_dens, label="Visit count")
    ax_dens.set_xlabel("PC1")
    ax_dens.set_ylabel("PC2")
    ax_dens.set_title(
        f"Attractor Density (PC1-PC2, {n_show} traj)\n"
        f"Bright = trajectory spends more time here  [{label}]"
    )
    ax_dens.set_aspect("equal", adjustable="datalim")

    # ── Right panel: sparse trajectory overlay ───────────────────────────────
    traj_colors = plt.cm.tab10(np.linspace(0, 1, n_show))
    cmap_time = plt.cm.viridis

    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_pca.shape[0]:
            break
        traj_pca = X_pca[start:end, :2]

        # Sub-sample: indices at regular stride
        idx = np.arange(0, T, stride)
        sub = traj_pca[idx]
        sub_t = idx / float(T - 1 if T > 1 else 1)  # normalised time in [0,1]

        # Thin connecting line (single colour per trajectory)
        ax_traj.plot(sub[:, 0], sub[:, 1], "-",
                     color=traj_colors[i], lw=0.8, alpha=0.45, zorder=2)
        # Scattered dots coloured by time within the trajectory
        sc = ax_traj.scatter(sub[:, 0], sub[:, 1],
                             c=sub_t, cmap=cmap_time,
                             s=6, alpha=0.8, zorder=3, linewidths=0)
        # Start / end markers
        ax_traj.scatter(traj_pca[0, 0], traj_pca[0, 1],
                        color=traj_colors[i], s=50, marker="o",
                        edgecolors="k", linewidths=0.5, zorder=5)
        ax_traj.scatter(traj_pca[-1, 0], traj_pca[-1, 1],
                        color=traj_colors[i], s=50, marker="X",
                        edgecolors="k", linewidths=0.5, zorder=5)

    # Single shared time colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_time, norm=plt.Normalize(0, T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax_traj, label="Time step")
    ax_traj.set_xlabel("PC1")
    ax_traj.set_ylabel("PC2")
    ax_traj.set_title(
        f"Attractor Trajectories (PC1-PC2, {n_show} traj)\n"
        f"Circle=start, X=end, colour=time  [{label}]"
    )
    ax_traj.set_aspect("equal", adjustable="datalim")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


def _try_plot_attractor_3d(
    X_pca: np.ndarray,
    n_traj: int,
    steps_per_traj: int,
    output_path: Path,
    label: str,
    max_traj_show: int = 5,
) -> None:
    """F2 3D: PC1-PC2-PC3 attractor projection.

    Each trajectory is plotted as a subsampled scatter (dots coloured by time)
    connected by a thin per-trajectory line.  This is dramatically cleaner
    than the old approach of drawing T-1 individual coloured line segments,
    which produced visually noisy / jagged plots for long trajectories.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    if X_pca.shape[1] < 3:
        logger.warning("PC 数量不足 3，跳过 3D 投影。")
        return

    n_show = min(max_traj_show, n_traj)
    T = steps_per_traj
    # Subsampling: target ~80 visible points per trajectory to keep the
    # 3-D plot legible without losing temporal structure.
    stride = max(1, T // 80)

    traj_colors = plt.cm.tab10(np.linspace(0, 1, max(n_show, 1)))
    cmap_time = plt.cm.viridis

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_pca.shape[0]:
            break
        traj_pca = X_pca[start:end, :3]

        # Sub-sample indices
        idx = np.arange(0, T, stride)
        sub = traj_pca[idx]
        sub_t = idx / float(T - 1 if T > 1 else 1)  # [0,1] normalised time

        # Thin background line (single per-trajectory colour, low alpha)
        ax.plot(sub[:, 0], sub[:, 1], sub[:, 2],
                "-", color=traj_colors[i], lw=0.7, alpha=0.30, zorder=2)
        # Scatter dots coloured by time
        ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2],
                   c=sub_t, cmap=cmap_time, s=8, alpha=0.75, zorder=3,
                   linewidths=0)
        # Start / end markers
        ax.scatter(*traj_pca[0], color=traj_colors[i], s=40, marker="o",
                   depthshade=False, edgecolors="k", linewidths=0.5, zorder=5)
        ax.scatter(*traj_pca[-1], color=traj_colors[i], s=40, marker="X",
                   depthshade=False, edgecolors="k", linewidths=0.5, zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(
        f"Attractor 3D Projection (PC1-PC3, {n_show} traj)  [{label}]\n"
        f"Circle=start, X=end, colour=time (dark→early, bright→late)"
    )
    # Shared time colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap_time, norm=plt.Normalize(0, T))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.1, label="Time step")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_attractor(
    trajectories: np.ndarray,
    output_dir: Optional[Path] = None,
    label: str = "trajectories",
    burnin: int = 0,
    n_components: Optional[int] = None,
    seed: int = 42,
) -> Dict:
    """
    运行模块 F：PCA 维度估计 + 吸引子可视化。

    Args:
        trajectories:  shape (n_traj, T, N)，自由动力学轨迹。
        output_dir:    输出目录。
        label:         文件名标签。
        burnin:        每条轨迹跳过的前 burnin 步（去除瞬态）。
        n_components:  最多保留的 PC 数；None → min(T_total, N)。
        seed:          随机种子。

    Returns:
        metrics dict（可序列化为 JSON）。
    """
    n_traj, T, N = trajectories.shape
    burnin = min(burnin, T - 1)
    T_eff = T - burnin
    logger.info("F PCA+吸引子: n_traj=%d, T=%d (burnin=%d), N=%d", n_traj, T, burnin, N)

    pca_result = compute_pca(trajectories, burnin=burnin,
                             n_components=n_components, seed=seed)

    # Serializable subset
    result: Dict = {
        "n_traj": n_traj,
        "n_steps": T,
        "burnin": burnin,
        "n_features": N,
        "n_components": pca_result["n_components"],
        "variance_top1_pct": pca_result["variance_top1"],
        "variance_top2_pct": pca_result["variance_top2"],
        "variance_top5_pct": pca_result["variance_top5"],
        "n_components_50pct": pca_result["n_components_50pct"],
        "n_components_80pct": pca_result["n_components_80pct"],
        "n_components_90pct": pca_result["n_components_90pct"],
        "n_components_95pct": pca_result["n_components_95pct"],
        "h2_pca_supported": pca_result["n_components_90pct"] <= max(3, N // 10),
        "pca_efficiency_ratio": round(
            pca_result["n_components_90pct"] / N, 3
        ),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        json_path = out / f"pca_results_{label}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save first 20 PCs
        np.save(
            out / f"pca_projections_{label}.npy",
            pca_result["X_pca"][:, :min(20, pca_result["n_components"])]
        )

        _try_plot_variance_curve(
            pca_result["explained_variance_ratio"],
            pca_result["cumulative_variance"],
            out / f"pca_variance_curve_{label}.png",
            label,
            pca_result["n_components_90pct"],
        )

        _try_plot_attractor_2d(
            pca_result["X_pca"],
            n_traj=n_traj,
            steps_per_traj=T_eff,
            output_path=out / f"attractor_projection_2d_{label}.png",
            label=label,
        )

        _try_plot_attractor_3d(
            pca_result["X_pca"],
            n_traj=n_traj,
            steps_per_traj=T_eff,
            output_path=out / f"attractor_projection_3d_{label}.png",
            label=label,
        )

        # ── New Test 2a: Phase Portrait (pair-wise PC projections) ──────────
        X_pca = pca_result["X_pca"]
        evr = pca_result["explained_variance_ratio"]
        if X_pca.shape[1] >= 2:
            _try_plot_phase_portrait_pair(
                X_pca, n_traj, T_eff, evr,
                pc_a=0, pc_b=1,
                output_path=out / "phase_portrait_pc1_pc2.png",
            )
        if X_pca.shape[1] >= 3:
            _try_plot_phase_portrait_pair(
                X_pca, n_traj, T_eff, evr,
                pc_a=0, pc_b=2,
                output_path=out / "phase_portrait_pc1_pc3.png",
            )
            _try_plot_phase_portrait_pair(
                X_pca, n_traj, T_eff, evr,
                pc_a=1, pc_b=2,
                output_path=out / "phase_portrait_pc2_pc3.png",
            )

        # ── New Test 2b: Delay-embedding (Takens) phase portrait ─────────────
        # Delay embedding preserves nonlinear attractor topology while PCA
        # is a linear projection that can flatten curved manifolds.
        # We generate BOTH so users can compare.
        X_delay, tau_used, T_embed = _build_delay_portrait(
            trajectories, burnin=burnin, m=3,
        )
        result["delay_embed_tau"] = int(tau_used)
        result["delay_embed_T"] = int(T_embed)
        if T_embed > 0:
            _try_plot_delay_portrait(
                X_delay, n_traj=n_traj, steps_per_traj=T_embed, tau=tau_used,
                output_path=out / "phase_portrait_delay_embed.png",
            )

            # ── New Test 3: Poincaré Section (delay-embedding coordinates) ────
            # Use delay-embedding coordinates instead of PC coordinates so the
            # section reflects the true attractor geometry (not a linear shadow).
            poincare = compute_poincare_section(X_delay, n_traj, T_embed)
            result["poincare_n_crossings"] = poincare["n_crossings"]
            result["poincare_interpretation"] = poincare["interpretation"]
            np.save(out / "poincare_points.npy", poincare["points"])
            _try_plot_poincare_section(
                poincare,
                output_path=out / "poincare_section.png",
                coord_labels=("y(t+tau)", "y(t+2*tau)"),
                section_label="y(t)=0, y(t+tau)>0",
            )
        else:
            result["poincare_n_crossings"] = 0
            result["poincare_interpretation"] = "trajectory_too_short"
            # Save empty array so downstream code that expects the file always
            # finds it regardless of whether delay embedding was possible.
            np.save(out / "poincare_points.npy", np.empty((0, 2), dtype=np.float32))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# New Test 2: Phase Portrait (pair-wise PC projections)
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_phase_portrait_pair(
    X_pca: np.ndarray,
    n_traj: int,
    steps_per_traj: int,
    evr: List[float],
    pc_a: int,
    pc_b: int,
    output_path: Path,
    n_traj_show: int = 5,
) -> None:
    """
    New Test 2: Phase portrait for any pair of PCs.

    Overlays up to `n_traj_show` trajectories in (PC_{pc_a+1}, PC_{pc_b+1})
    space, coloured by time.  Useful for direct attractor geometry inspection.

    Args:
        X_pca:          shape (n_traj * T_eff, n_pcs) — all trajectories stacked.
        n_traj:         number of trajectories.
        steps_per_traj: T_eff = T - burnin.
        evr:            explained variance ratios list.
        pc_a, pc_b:     0-indexed PC pair to plot.
        output_path:    where to save the PNG.
        n_traj_show:    max trajectories to overlay.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _write_fallback_png(output_path)
        return

    if X_pca.shape[1] <= max(pc_a, pc_b):
        return

    n_show = min(n_traj_show, n_traj)
    T = steps_per_traj
    cmap = plt.cm.viridis
    colors_t = cmap(np.linspace(0.05, 0.95, T))

    fig, ax = plt.subplots(figsize=(6, 6))
    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_pca.shape[0]:
            break
        pc_x = X_pca[start:end, pc_a]
        pc_y = X_pca[start:end, pc_b]
        for t in range(T - 1):
            ax.plot(pc_x[t:t+2], pc_y[t:t+2], color=colors_t[t], lw=0.8, alpha=0.7)
        ax.scatter(pc_x[0], pc_y[0], color="blue", s=25, zorder=5, marker="o")
        ax.scatter(pc_x[-1], pc_y[-1], color="red", s=25, zorder=5, marker="x")

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time Step")

    var_a = float(evr[pc_a]) * 100 if pc_a < len(evr) else 0
    var_b = float(evr[pc_b]) * 100 if pc_b < len(evr) else 0
    ax.set_xlabel(f"PC{pc_a+1}  ({var_a:.1f}% var)")
    ax.set_ylabel(f"PC{pc_b+1}  ({var_b:.1f}% var)")
    ax.set_title(
        f"Phase Portrait: PC{pc_a+1} vs PC{pc_b+1}\n"
        f"({n_show} trajectories, blue=start, red×=end, colour=time)"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存 phase portrait: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# New Test 3: Poincaré Section
# ─────────────────────────────────────────────────────────────────────────────

def compute_poincare_section(
    X_pca: np.ndarray,
    n_traj: int,
    steps_per_traj: int,
) -> Dict:
    """
    New Test 3: Compute a Poincaré section through the plane PC1 = 0, PC2 > 0.

    For each trajectory, detects upward-crossing zero-crossings of PC1
    (PC1(t) < 0 and PC1(t+1) >= 0) where PC2 > 0 at the crossing point.
    Records (PC2_cross, PC3_cross) at each crossing via linear interpolation.

    Interpretation:
      - Single cluster → periodic orbit
      - Smooth curve   → quasi-periodic (torus)
      - Scattered / fractal → chaotic

    Args:
        X_pca:          shape (n_traj * T_eff, n_pcs).
        n_traj:         number of trajectories.
        steps_per_traj: T_eff.

    Returns:
        dict with keys:
          points:          np.ndarray (n_crossings, 2) — (PC2_cross, PC3_cross)
          n_crossings:     int
          interpretation:  str — "periodic" | "quasi-periodic" | "chaotic"
          has_pc3:         bool
    """
    has_pc3 = X_pca.shape[1] >= 3
    T = steps_per_traj
    crossings_pc2: List[float] = []
    crossings_pc3: List[float] = []

    for i in range(n_traj):
        start = i * T
        end = min(start + T, X_pca.shape[0])
        if end - start < 4:
            continue
        pc1 = X_pca[start:end, 0]
        pc2 = X_pca[start:end, 1] if X_pca.shape[1] >= 2 else np.zeros(end - start)
        pc3 = X_pca[start:end, 2] if has_pc3 else np.zeros(end - start)

        # Detect upward zero-crossings: PC1(t) < 0 → PC1(t+1) >= 0
        t_arr = np.arange(len(pc1) - 1)
        cross_mask = (pc1[t_arr] < 0) & (pc1[t_arr + 1] >= 0)
        cross_times = t_arr[cross_mask]

        for t in cross_times:
            # Linear interpolation to find exact crossing fraction
            dpc1 = pc1[t + 1] - pc1[t]
            if abs(dpc1) < _POINCARE_CROSSING_EPS:
                alpha = 0.0
            else:
                alpha = -pc1[t] / dpc1   # in [0, 1]
            pc2_c = float(pc2[t] + alpha * (pc2[t + 1] - pc2[t]))
            pc3_c = float(pc3[t] + alpha * (pc3[t + 1] - pc3[t]))
            # Only keep crossings with PC2 > 0 (half-plane condition)
            if pc2_c > 0:
                crossings_pc2.append(pc2_c)
                crossings_pc3.append(pc3_c)

    n_cross = len(crossings_pc2)
    if n_cross == 0:
        points = np.empty((0, 2), dtype=np.float32)
        interp = "no_crossings"
    else:
        points = np.column_stack([crossings_pc2, crossings_pc3]).astype(np.float32)
        # Classify based on spread of crossing points.
        # Thresholds (relative to range):
        #   _POINCARE_TIGHT_RATIO  → clustered   → periodic
        #   _POINCARE_QUASI_RATIO  → smooth curve → quasi-periodic
        #   otherwise              → scattered    → chaotic
        if n_cross < 5:
            interp = "insufficient_crossings"
        else:
            pc2_arr = np.array(crossings_pc2)
            pc3_arr = np.array(crossings_pc3)
            spread_pc2 = float(np.std(pc2_arr))
            spread_pc3 = float(np.std(pc3_arr) if has_pc3 else 0.0)
            spread = max(spread_pc2, spread_pc3)
            range_pc2 = float(np.ptp(pc2_arr)) if len(pc2_arr) > 1 else 0.0
            if range_pc2 < _POINCARE_PERIODIC_THRESH:
                interp = "periodic"
            elif spread / (range_pc2 + 1e-9) < _POINCARE_TIGHT_RATIO:
                interp = "periodic"
            elif spread / (range_pc2 + 1e-9) < _POINCARE_QUASI_RATIO:
                interp = "quasi-periodic"
            else:
                interp = "chaotic"

    return {
        "points": points,
        "n_crossings": n_cross,
        "interpretation": interp,
        "has_pc3": has_pc3,
    }


def _try_plot_poincare_section(
    poincare: Dict,
    output_path: Path,
    coord_labels: tuple = ("y(t+tau)", "y(t+2*tau)"),
    section_label: str = "y(t)=0, y(t+tau)>0",
) -> None:
    """Poincaré section scatter plot.

    Accepts delay-embedding coordinate labels so the axes correctly describe
    what the coordinates represent (delay-embedding by default, or PC labels
    for legacy callers).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _write_fallback_png(output_path)
        return

    points = poincare["points"]
    n_cross = poincare["n_crossings"]
    interp = poincare["interpretation"]

    fig, ax = plt.subplots(figsize=(6, 6))

    if n_cross == 0:
        ax.text(0.5, 0.5, f"No crossings detected\n(section {section_label})",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        c0 = points[:, 0]
        c1 = points[:, 1]
        # Colour by crossing order (trajectory time)
        cmap = plt.cm.plasma
        sc = ax.scatter(c0, c1,
                        c=np.arange(n_cross), cmap=cmap,
                        s=20, alpha=0.8, edgecolors="k", lw=0.3, zorder=3)
        plt.colorbar(sc, ax=ax, label="Crossing index (time order)")

    ax.set_xlabel(coord_labels[0])
    ax.set_ylabel(coord_labels[1])
    ax.set_title(
        f"Poincaré Section  ({section_label})\n"
        f"n_crossings={n_cross},  interpretation: {interp}\n"
        "cluster→periodic | curve→quasi-periodic | scatter→chaotic"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved Poincare section: %s", output_path)


# ─────────────────────────────────────────────────────────────────────────────
# Delay-embedding (Takens) phase portrait
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_delay_portrait(
    X_delay: np.ndarray,
    n_traj: int,
    steps_per_traj: int,
    tau: int,
    output_path: Path,
    n_traj_show: int = 5,
) -> None:
    """Plot delay-embedding phase portrait: y(t) vs y(t+tau) [vs y(t+2*tau)].

    Uses Takens delay embedding (mean channel observable) so the phase portrait
    preserves the nonlinear attractor topology that PCA linear projection may
    flatten or distort.

    If the embedding dimension is ≥ 3, a 3-D trajectory plot is also produced
    alongside the 2-D panel.

    Args:
        X_delay:        shape (n_traj * T_embed, m) from _build_delay_portrait.
        n_traj:         number of trajectories.
        steps_per_traj: T_embed (delay-adjusted trajectory length).
        tau:            delay lag used (for axis labels).
        output_path:    PNG save path.
        n_traj_show:    max trajectories to overlay.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        _write_fallback_png(output_path)
        return

    m = X_delay.shape[1]
    if m < 2:
        _write_fallback_png(output_path)
        return

    has_3d = m >= 3
    n_show = min(n_traj_show, n_traj)
    T = steps_per_traj
    cmap = plt.cm.viridis
    colors_t = cmap(np.linspace(0.05, 0.95, T))

    if has_3d:
        fig = plt.figure(figsize=(12, 6))
        ax2 = fig.add_subplot(1, 2, 1)
        ax3 = fig.add_subplot(1, 2, 2, projection="3d")
    else:
        fig, ax2 = plt.subplots(figsize=(6, 6))

    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_delay.shape[0]:
            break
        y0 = X_delay[start:end, 0]
        y1 = X_delay[start:end, 1]

        # 2-D portrait
        for t in range(T - 1):
            ax2.plot(y0[t:t+2], y1[t:t+2], color=colors_t[t], lw=0.8, alpha=0.7)
        ax2.scatter(y0[0], y1[0], color="blue", s=25, zorder=5, marker="o")
        ax2.scatter(y0[-1], y1[-1], color="red", s=25, zorder=5, marker="x")

        # 3-D portrait
        if has_3d:
            y2 = X_delay[start:end, 2]
            ax3.plot(y0, y1, y2, lw=0.6, alpha=0.6)
            ax3.scatter(y0[0], y1[0], y2[0], color="blue", s=20)
            ax3.scatter(y0[-1], y1[-1], y2[-1], color="red", s=20, marker="^")

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax2, label="Time Step")

    ax2.set_xlabel(f"y(t)")
    ax2.set_ylabel(f"y(t+{tau})")
    ax2.set_title(
        f"Phase Portrait: Delay Embedding (tau={tau})\n"
        f"({n_show} traj, blue=start, redX=end, colour=time)"
    )
    ax2.grid(True, alpha=0.2)

    if has_3d:
        ax3.set_xlabel(f"y(t)")
        ax3.set_ylabel(f"y(t+{tau})")
        ax3.set_zlabel(f"y(t+{2*tau})")
        ax3.set_title(f"3-D Delay Portrait (tau={tau})")

    fig.suptitle("Takens Delay Embedding Phase Portrait", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved delay-embedding phase portrait: %s", output_path)
