"""
Trajectory Plot
===============

可视化自由动力学轨迹：
  - 多条轨迹的状态范数随时间变化
  - 轨迹汇聚 / 发散示意图（降维到 PCA 前两维）
  - 逐脑区活动热图
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _MPL_AVAILABLE = False
    logger.warning("matplotlib 未安装，轨迹可视化功能不可用。")


def plot_trajectory_norms(
    trajectories: np.ndarray,
    times: Optional[np.ndarray] = None,
    max_show: int = 20,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制多条轨迹的 L2 范数随时间变化。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        times:        shape (steps,)，时间轴（秒）；None → 用步数。
        max_show:     最多显示的轨迹数。
        save_path:    保存路径；None → 显示到屏幕。
    """
    if not _MPL_AVAILABLE:
        return

    n_traj, steps, _ = trajectories.shape
    norms = np.linalg.norm(trajectories, axis=2)  # (n_init, steps)

    if times is None:
        times = np.arange(steps, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(10, 5))

    indices = np.linspace(0, n_traj - 1, min(max_show, n_traj), dtype=int)
    cmap = plt.get_cmap("viridis", len(indices))

    for k, idx in enumerate(indices):
        ax.plot(times, norms[idx], alpha=0.6, color=cmap(k), linewidth=0.8)

    ax.set_xlabel("Time (s)" if times[1] < 1.0 else "Step")
    ax.set_ylabel("State L₂ norm")
    ax.set_title(f"Free Dynamics — Trajectory Norms ({len(indices)} of {n_traj})")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_pca_trajectories(
    trajectories: np.ndarray,
    max_show: int = 50,
    save_path: Optional[Path] = None,
) -> None:
    """
    用 PCA 将高维轨迹投影到 2D，并绘制轨迹路径。

    Args:
        trajectories: shape (n_init, steps, n_regions)。
        max_show:     最多显示轨迹数。
        save_path:    保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    try:
        from sklearn.decomposition import PCA
    except ImportError:  # pragma: no cover
        logger.warning("scikit-learn 未安装，跳过 PCA 轨迹图。")
        return

    n_traj, steps, n_regions = trajectories.shape
    all_states = trajectories.reshape(-1, n_regions)  # (n_init*steps, n_regions)

    pca = PCA(n_components=2)
    pca.fit(all_states)
    var_ratio = pca.explained_variance_ratio_

    indices = np.linspace(0, n_traj - 1, min(max_show, n_traj), dtype=int)
    cmap = plt.get_cmap("plasma", len(indices))

    fig, ax = plt.subplots(figsize=(8, 7))

    for k, idx in enumerate(indices):
        traj_2d = pca.transform(trajectories[idx])  # (steps, 2)
        ax.plot(traj_2d[:, 0], traj_2d[:, 1], alpha=0.5, color=cmap(k), linewidth=0.7)
        ax.scatter(traj_2d[0, 0], traj_2d[0, 1], color=cmap(k), marker="o", s=10, zorder=5)
        ax.scatter(traj_2d[-1, 0], traj_2d[-1, 1], color=cmap(k), marker="*", s=20, zorder=5)

    ax.set_xlabel(f"PC1 ({var_ratio[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({var_ratio[1]:.1%} var)")
    ax.set_title("Free Dynamics in PCA Space (○ start, ★ end)")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


def plot_region_heatmap(
    trajectory: np.ndarray,
    times: Optional[np.ndarray] = None,
    title: str = "Region Activity Heatmap",
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制单条轨迹的逐脑区活动热图（时间 × 脑区）。

    Args:
        trajectory: shape (steps, n_regions)。
        times:      shape (steps,)，时间轴。
        title:      图标题。
        save_path:  保存路径。
    """
    if not _MPL_AVAILABLE:
        return

    steps, n_regions = trajectory.shape
    if times is None:
        times = np.arange(steps, dtype=np.float32)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        trajectory.T,
        aspect="auto",
        origin="lower",
        cmap="hot",
        extent=[times[0], times[-1], 0, n_regions],
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(im, ax=ax, label="Activity (normalised)")
    ax.set_xlabel("Time (s)" if times[1] < 1.0 else "Step")
    ax.set_ylabel("Region index")
    ax.set_title(title)

    _save_or_show(fig, save_path)


# ── Internal helpers ──────────────────────────────────────────────────────────

def _save_or_show(fig: "plt.Figure", save_path: Optional[Path]) -> None:
    """Save figure to file or display it."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("  → 图表已保存: %s", save_path)
    else:
        plt.show()
    plt.close(fig)


def plot_trajectory_convergence(
    mean_distances: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制轨迹对均值距离随时间的变化（收敛分析图）。

    Args:
        mean_distances: shape (steps,)，均值 L2 距离。
        save_path:      保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    steps = len(mean_distances)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(steps), mean_distances, color="steelblue", linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Pairwise Distance")
    ax.set_title("Trajectory Convergence — Mean Pairwise Distance over Time\n"
                 "(↓ converging, → no structure, ↑ diverging)")
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_lyapunov_growth(
    log_growth_curve: np.ndarray,
    renorm_steps: int = 20,
    mean_lle: Optional[float] = None,
    chaos_regime: Optional[str] = None,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制 Wolf 方法的累积对数增长曲线（log-growth curve）。

    **左图 — 逐周期对数增长（Per-Period Log Growth）**
    每个重归一化周期的 log(r/ε) 值，用柱状图显示。
    - 正值（红色）→ 扰动放大 → 该周期混沌
    - 负值（蓝色）→ 扰动收缩 → 该周期收敛
    - 覆盖滑动平均线（绿色）→ 显示 LLE 估计随周期的收敛过程
    - 橙色阴影 → ±1σ 带，显示周期到周期的波动范围

    **注意：后段柱子高度一致是正常现象。**
    Wolf 方法在扰动向量对齐到优势 Lyapunov 向量（通常 3–6 个周期）后，
    每个周期的增长率会收敛到一个稳定值（= LLE × renorm_steps），
    产生"完全相同"的柱子。这不是代码错误，而是算法收敛的数学必然结果。

    **右图 — 去趋势累积对数增长（Detrended Cumulative Log Growth）**
    原始累积和为 S(t) = Σ log_growth[k]，对于已收敛的 LLE 这是一条完美直线。
    此图显示去线性趋势后的残差：S_detrended(t) = S(t) − fit(t)。
    残差揭示：
    - 暂态区（前几个周期）：误差最大，因 Lyapunov 向量还未对齐
    - 收敛后：残差接近零（稳定直线）
    - 非稳态或多吸引子系统：残差持续波动（混沌行为标志）

    Args:
        log_growth_curve: shape (n_periods,)，每个重归一化周期的 log(r/ε) 值。
        renorm_steps:     每个周期的步数（用于计算累积步轴）。
        mean_lle:         平均 LLE（显示在标题中）。
        chaos_regime:     混沌分类标签。
        save_path:        保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    n_periods = len(log_growth_curve)
    if n_periods == 0:
        return

    cumulative = np.cumsum(log_growth_curve)
    steps_axis = np.arange(1, n_periods + 1) * renorm_steps

    # Linear fit to cumulative sum (slope = LLE per step)
    coeffs = np.polyfit(steps_axis, cumulative, deg=1)
    fit_line = np.polyval(coeffs, steps_axis)
    fit_slope = coeffs[0]

    # Detrended cumulative: residuals after removing linear trend.
    # For a fully-converged stable system this will be flat near zero.
    # For a chaotic / non-stationary system this will fluctuate.
    detrended = cumulative - fit_line

    # Running mean (causal, window = min(5, n_periods//4+1))
    win = max(1, min(5, n_periods // 4 + 1))
    running_mean = np.convolve(log_growth_curve, np.ones(win) / win, mode="valid")
    # Pad left to match length
    pad_left = n_periods - len(running_mean)
    running_mean = np.concatenate([np.full(pad_left, np.nan), running_mean])

    # Detect convergence: require 3 consecutive periods within 10% of final value
    # to avoid false positives from transient dips into the tolerance window.
    final_rm = running_mean[~np.isnan(running_mean)][-1] if n_periods > win else None
    converge_period = None
    if final_rm is not None and abs(final_rm) > 1e-9:
        tol = abs(final_rm) * 0.10   # 10% tolerance
        min_consecutive = 3
        consecutive = 0
        for k in range(pad_left, n_periods):
            if abs(running_mean[k] - final_rm) < tol:
                consecutive += 1
                if consecutive >= min_consecutive and converge_period is None:
                    converge_period = k - min_consecutive + 1
            else:
                consecutive = 0

    # ±1σ statistics (from the converged portion if detectable)
    conv_start = converge_period if converge_period is not None else n_periods // 3
    conv_values = log_growth_curve[conv_start:]
    conv_mean = float(np.mean(conv_values)) if len(conv_values) > 0 else float(np.mean(log_growth_curve))
    conv_std = float(np.std(conv_values)) if len(conv_values) > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    # ── Left: per-period log-growth with running mean and ±1σ ─────────────────
    ax = axes[0]
    bar_colors = ["tomato" if v > 0 else "steelblue" for v in log_growth_curve]
    ax.bar(np.arange(n_periods), log_growth_curve,
           color=bar_colors, alpha=0.60, edgecolor="white", label="Per-period log(r/ε)")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(conv_mean, color="red", linewidth=1.2, linestyle="-",
               label=f"converged mean = {conv_mean:.4f}")

    # ±1σ band around converged mean
    if conv_std > 0:
        ax.fill_between(
            np.arange(n_periods),
            conv_mean - conv_std, conv_mean + conv_std,
            alpha=0.12, color="orange", label=f"±1σ = {conv_std:.4f}",
        )

    # Running mean line (shows convergence)
    ax.plot(np.arange(n_periods), running_mean, color="limegreen", linewidth=1.5,
            linestyle="-", zorder=5, label=f"Running mean (w={win})")

    # Shade the transient region
    if converge_period is not None and converge_period > 0:
        ax.axvspan(-0.5, converge_period - 0.5, alpha=0.06, color="gray",
                   label=f"Transient (0–{converge_period - 1})")
        ax.axvline(converge_period - 0.5, color="gray", linewidth=0.8, linestyle=":")

    ax.set_xlabel("Renormalization Period")
    ax.set_ylabel("log(r/ε) per period")
    ax.set_title(
        "Per-Period Log Growth\n"
        "✓ Stable systems: bars converge to identical height (LLE stabilised)\n"
        "  Chaotic systems: bars fluctuate around mean LLE — bimodal ok",
        fontsize=9,
    )
    ax.legend(fontsize=7, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    # ── Right: detrended cumulative log-growth ────────────────────────────────
    ax = axes[1]
    ax.plot(steps_axis, detrended, color="navy", linewidth=1.5,
            label="Detrended cumulative (S(t) − linear fit)")
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)

    # Show the magnitude of transient deviation
    if converge_period is not None and converge_period > 0:
        ax.axvspan(steps_axis[0], steps_axis[converge_period - 1],
                   alpha=0.08, color="gray", label=f"Transient (0–{converge_period - 1})")

    # R² of the linear fit (1.0 = perfect straight line after convergence)
    ss_res = float(np.sum((cumulative - fit_line) ** 2))
    ss_tot = float(np.sum((cumulative - cumulative.mean()) ** 2))
    r2 = 1.0 - ss_res / (ss_tot + 1e-30)

    lbl_parts = []
    if mean_lle is not None:
        lbl_parts.append(f"LLE = {mean_lle:.5f}")
    if chaos_regime:
        lbl_parts.append(chaos_regime.upper())
    lbl_parts.append(f"R²(linear) = {r2:.4f}")

    ax.set_xlabel("Step")
    ax.set_ylabel("S(t) − linear trend")
    ax.set_title(
        "Detrended Cumulative Log-Growth\n" +
        ("  |  ".join(lbl_parts) if lbl_parts else ""),
        fontsize=9,
    )
    ax.text(
        0.02, 0.05,
        "R²≈1 → LLE fully converged (straight line)\n"
        "R²<1 → transient or non-stationary dynamics",
        transform=ax.transAxes, fontsize=7, va="bottom", color="dimgray",
    )
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path)


def plot_lyapunov_histogram(
    lyapunov_values: np.ndarray,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制 Lyapunov 指数分布直方图。

    Args:
        lyapunov_values: shape (n_init,)，每条轨迹的 Lyapunov 指数。
        save_path:       保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    mean_lam = float(np.mean(lyapunov_values))
    median_lam = float(np.median(lyapunov_values))

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lyapunov_values, bins=min(30, max(10, len(lyapunov_values) // 3)),
            color="salmon", edgecolor="white", alpha=0.85)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1.0, label="λ=0")
    ax.axvline(mean_lam, color="red", linestyle="-", linewidth=1.2,
               label=f"mean={mean_lam:.4f}")
    ax.axvline(median_lam, color="orange", linestyle=":", linewidth=1.2,
               label=f"median={median_lam:.4f}")
    ax.set_xlabel("Lyapunov Exponent λ (per step)")
    ax.set_ylabel("Count")
    ax.set_title("Lyapunov Exponent Distribution\n"
                 "(λ<0: convergent, λ≈0: marginal, λ>0: chaotic)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    _save_or_show(fig, save_path)


def plot_basin_sizes(
    basin_distribution: dict,
    save_path: Optional[Path] = None,
) -> None:
    """
    绘制吸引子 basin 大小分布条形图。

    Args:
        basin_distribution: {attractor_id: fraction} 字典（int 键或 str 键均可）。
        save_path:          保存路径；None → 显示。
    """
    if not _MPL_AVAILABLE:
        return

    labels_abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    ids = sorted(basin_distribution.keys())
    fractions = [basin_distribution[k] for k in ids]
    bar_labels = [
        (f"Attractor {labels_abc[k]}" if isinstance(k, int) and k < len(labels_abc)
         else str(k))
        for k in ids
    ]

    fig, ax = plt.subplots(figsize=(max(4, len(ids) * 1.5), 4))
    cmap = plt.get_cmap("tab10", len(ids))
    bars = ax.bar(bar_labels, fractions, color=[cmap(i) for i in range(len(ids))],
                  edgecolor="white")
    for bar, frac in zip(bars, fractions):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.01,
            f"{frac:.1%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.set_ylim(0, min(1.15, max(fractions) + 0.15))
    ax.set_ylabel("Basin Size (fraction of trajectories)")
    ax.set_title("Attractor Basin Size Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    _save_or_show(fig, save_path)
