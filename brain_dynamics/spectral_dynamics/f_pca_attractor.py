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
    """F2 2D: PC1 vs PC2 吸引子投影，颜色编码时间。"""
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

    fig, ax = plt.subplots(figsize=(7, 6))
    cmap = plt.cm.coolwarm

    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_pca.shape[0]:
            break
        traj_pca = X_pca[start:end, :2]
        colors = cmap(np.linspace(0.1, 0.9, T))
        # Draw trajectory with color gradient
        for t in range(T - 1):
            ax.plot(traj_pca[t:t+2, 0], traj_pca[t:t+2, 1],
                    color=colors[t], lw=0.8, alpha=0.7)
        # Mark start and end
        ax.scatter(traj_pca[0, 0], traj_pca[0, 1], color="blue", s=30, zorder=5,
                   marker="o", alpha=0.8)
        ax.scatter(traj_pca[-1, 0], traj_pca[-1, 1], color="red", s=30, zorder=5,
                   marker="x", alpha=0.8)

    # Colorbar for time
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=0, vmax=T))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="Time Step")

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Attractor Projection (PC1 vs PC2, {n_show} traj)\n"
                 f"Blue=start, Red x=end, color=time  [{label}]")
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
    """F2 3D: PC1-PC2-PC3 吸引子投影。"""
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
    cmap = plt.cm.coolwarm

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")

    for i in range(n_show):
        start = i * T
        end = start + T
        if end > X_pca.shape[0]:
            break
        traj_pca = X_pca[start:end, :3]
        colors = cmap(np.linspace(0.1, 0.9, T))
        for t in range(T - 1):
            ax.plot(
                traj_pca[t:t+2, 0], traj_pca[t:t+2, 1], traj_pca[t:t+2, 2],
                color=colors[t], lw=0.8, alpha=0.6,
            )
        ax.scatter(traj_pca[0, 0], traj_pca[0, 1], traj_pca[0, 2],
                   color="blue", s=25, zorder=5)
        ax.scatter(traj_pca[-1, 0], traj_pca[-1, 1], traj_pca[-1, 2],
                   color="red", s=25, marker="x", zorder=5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.set_title(f"Attractor 3D Projection (PC1-PC3, {n_show} traj)  [{label}]")
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

        # ── New Test 2: Phase Portrait (pair-wise PC projections) ─────────────
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

        # ── New Test 3: Poincaré Section ──────────────────────────────────────
        poincare = compute_poincare_section(X_pca, n_traj, T_eff)
        result["poincare_n_crossings"] = poincare["n_crossings"]
        result["poincare_interpretation"] = poincare["interpretation"]
        np.save(out / "poincare_points.npy", poincare["points"])
        _try_plot_poincare_section(
            poincare,
            output_path=out / "poincare_section.png",
        )

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


def _try_plot_poincare_section(poincare: Dict, output_path: Path) -> None:
    """
    New Test 3 plot: PC2_cross vs PC3_cross at Poincaré section crossings.
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
        ax.text(0.5, 0.5, "No crossings detected\n(section PC1=0, PC2>0)",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
    else:
        pc2_c = points[:, 0]
        pc3_c = points[:, 1]
        # Colour by crossing order (trajectory time)
        cmap = plt.cm.plasma
        sc = ax.scatter(pc2_c, pc3_c,
                        c=np.arange(n_cross), cmap=cmap,
                        s=20, alpha=0.8, edgecolors="k", lw=0.3, zorder=3)
        plt.colorbar(sc, ax=ax, label="Crossing index (time order)")

    ax.set_xlabel("PC2 at crossing")
    ax.set_ylabel("PC3 at crossing")
    ax.set_title(
        f"Poincaré Section  (PC1=0, PC2>0)\n"
        f"n_crossings={n_cross},  interpretation: {interp}\n"
        "cluster→periodic | curve→quasi-periodic | scatter→chaotic"
    )
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存 Poincare section: %s", output_path)
