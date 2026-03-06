"""
Lyapunov Exponent Analysis
============================

估计系统最大 Lyapunov 指数（Largest Lyapunov Exponent, LLE），
判断系统是否接近混沌状态。

提供两种估计方法，均可从已生成的 trajectories.npy 使用：

**方法 1 — Wolf-Benettin 重归一化（默认，推荐）**

  经典的 Wolf 1985 方法，以周期性重归一化避免扰动向量饱和/发散：

    1. 选择随机单位扰动方向 e
    2. 演化 renorm_steps 步，测量扰动增长 r = ||δ(t+τ)|| / ε
    3. 累积 S += log(r/ε)，将 δ 重归一化回 ε
    4. 重复 n_periods 次
    5. LLE = S / (n_periods × renorm_steps)

  优点：
  - 不受指数发散/收敛饱和影响
  - 结果代表时间平均的真实 Lyapunov 指数
  - 对 WC 的有界 [0,1] 状态空间特别稳定

**方法 2 — 简单 FTLE 线性拟合（快速对照）**

  运行双轨迹，对 log(||δ(t)||) 做线性回归。
  计算快速但可能受瞬态/饱和影响。

**混沌状态分类**（基于 mean LLE）：

  ┌────────────────┬──────────────────────────────────────────────────────┐
  │ LLE            │ 解释                                                  │
  ├────────────────┼──────────────────────────────────────────────────────┤
  │ λ < −0.01      │ 稳定收敛（stable） — 系统强烈排斥扰动                │
  │ −0.01 ≤ λ < 0  │ 边缘稳定（marginal stable） — 弱收敛                 │
  │ 0 ≤ λ < 0.01   │ 混沌边缘（edge of chaos） — 中性稳定                │
  │ 0.01 ≤ λ < 0.1 │ 弱混沌（weakly chaotic） — 轨迹缓慢发散              │
  │ λ ≥ 0.1        │ 强混沌（strongly chaotic） — 轨迹快速发散            │
  └────────────────┴──────────────────────────────────────────────────────┘

输出：
  lyapunov_values.npy          — shape (n_traj,)，每条轨迹的 LLE
  log_growth_curve.npy         — shape (n_renorm_periods,)，平均对数增长曲线
  lyapunov_report.json         — 汇总统计 + 混沌评估
  plots/lyapunov_histogram.png — LLE 分布直方图（含阈值标注）
  plots/lyapunov_growth.png    — 对数增长曲线（用于直观展示指数增长/收敛）
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default perturbation magnitude
_DEFAULT_EPSILON: float = 1e-5

# Number of steps between each Wolf renormalization step
_DEFAULT_RENORM_STEPS: int = 20

# Minimum number of periods required for a reliable LLE estimate
_MIN_PERIODS: int = 5

# Chaos classification thresholds (per-step LLE)
_THRESH_STRONGLY_STABLE: float = -0.01   # λ < this → stable
_THRESH_MARGINAL: float = 0.0            # λ < this → marginal stable
_THRESH_EDGE_CHAOS: float = 0.01         # λ < this → edge of chaos
_THRESH_WEAK_CHAOS: float = 0.1          # λ < this → weakly chaotic
# λ ≥ _THRESH_WEAK_CHAOS → strongly chaotic


# ══════════════════════════════════════════════════════════════════════════════
# Wolf-Benettin renormalization method (primary)
# ══════════════════════════════════════════════════════════════════════════════

def wolf_largest_lyapunov(
    simulator,
    x0: np.ndarray,
    total_steps: int,
    renorm_steps: int = _DEFAULT_RENORM_STEPS,
    epsilon: float = _DEFAULT_EPSILON,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, np.ndarray]:
    """
    Wolf-Benettin 方法估计单条轨迹的最大 Lyapunov 指数。

    通过周期性重归一化扰动向量，避免指数发散导致的数值饱和，
    给出时间平均的真实 LLE 估计。

    Args:
        simulator:      ``BrainDynamicsSimulator`` 实例（需支持 rollout）。
        x0:             初始状态，shape (n_regions,)，值域 [0, 1]。
        total_steps:    总模拟步数。
        renorm_steps:   每次重归一化之间的步数（默认 20）。
        epsilon:        扰动幅度（默认 1e-5）。
        rng:            已初始化的随机数生成器；None → 使用固定种子 0。

    Returns:
        (lle, log_growth):
          lle:         时间平均最大 Lyapunov 指数（每步单位）。
          log_growth:  每个周期的 log(r/ε)，shape (n_periods,)。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    n = len(x0)
    n_periods = max(1, total_steps // renorm_steps)

    # Initial random perturbation direction (unit vector)
    perturb = rng.standard_normal(n).astype(np.float64)
    perturb /= (np.linalg.norm(perturb) + 1e-15)

    x_cur = x0.astype(np.float32).copy()
    log_growth: List[float] = []
    log_sum = 0.0

    # `wolf_rollout_pair` correctly handles both WC mode (fixed equilibrium)
    # and twin mode (advancing context window).  Falling back to the old
    # `rollout()` pair is kept only for custom simulator objects that have
    # not yet implemented this method.
    use_wolf_pair = hasattr(simulator, "wolf_rollout_pair")
    wolf_context = None  # opaque per-mode state carried between periods

    for _ in range(n_periods):
        # Perturbed initial point for this period
        x_pert = np.clip(
            x_cur + (epsilon * perturb).astype(np.float32), 0.0, 1.0
        )

        if use_wolf_pair:
            # Preferred path: per-mode correct Wolf integration.
            x_after, x_after_pert, wolf_context = simulator.wolf_rollout_pair(
                x_base=x_cur,
                x_pert=x_pert,
                steps=renorm_steps,
                wolf_context=wolf_context,
            )
        else:
            # Legacy fallback: resets equilibrium / context every period.
            # Kept for backward compatibility with custom simulator objects.
            traj_base, _ = simulator.rollout(
                x0=x_cur, steps=renorm_steps, stimulus=None
            )
            traj_pert, _ = simulator.rollout(
                x0=x_pert, steps=renorm_steps, stimulus=None
            )
            x_after = traj_base[-1]
            x_after_pert = traj_pert[-1]

        # Separation vector and its magnitude
        delta = (
            np.asarray(x_after_pert, dtype=np.float64)
            - np.asarray(x_after, dtype=np.float64)
        )
        r = np.linalg.norm(delta)

        # Guard against numerical underflow
        r = max(r, np.finfo(np.float64).tiny)

        # Accumulate log growth: log(r/ε) = log(r) - log(ε)
        growth = np.log(r) - np.log(epsilon)
        log_sum += growth
        log_growth.append(growth)

        # Renormalize: reset separation to ε in the direction of current delta
        perturb = delta / (r + 1e-15)
        x_cur = np.asarray(x_after, dtype=np.float32)

    lle = log_sum / (n_periods * renorm_steps)
    return float(lle), np.array(log_growth, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# Simple FTLE (fast alternative / cross-check)
# ══════════════════════════════════════════════════════════════════════════════

def ftle_lyapunov(
    trajectory: np.ndarray,
    simulator,
    epsilon: float = _DEFAULT_EPSILON,
    skip_fraction: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    简单有限时间 Lyapunov 指数（FTLE），通过对 log(δ(t)) 线性回归估计 LLE。

    较 Wolf 方法更快（只需两条轨迹），但精度较低：
    - 无重归一化，δ(t) 可能饱和（[0,1] 有界状态空间）
    - 适合与 Wolf 方法对比验证，或轨迹数量多时的快速筛查

    Args:
        trajectory:     shape (T, n_regions)，已计算好的轨迹。
        simulator:      ``BrainDynamicsSimulator`` 实例。
        epsilon:        初始扰动幅度（默认 1e-5）。
        skip_fraction:  线性拟合时跳过前段的比例（避免瞬态）。
        rng:            随机数生成器；None → 使用 seed=0。

    Returns:
        lambda_est: FTLE 估计值（每步）。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    T, n_regions = trajectory.shape
    x0 = trajectory[0].copy()

    # Random unit perturbation vector (fixed seed → reproducible per call)
    perturb = rng.standard_normal(n_regions).astype(np.float32)
    perturb /= (np.linalg.norm(perturb) + 1e-15)
    x0_perturbed = np.clip(x0 + epsilon * perturb, 0.0, 1.0)

    traj_orig, _ = simulator.rollout(x0=x0, steps=T, stimulus=None)
    traj_pert, _ = simulator.rollout(x0=x0_perturbed, steps=T, stimulus=None)

    dist = np.linalg.norm((traj_pert - traj_orig).astype(np.float64), axis=1)
    # Use 1e-15 as the floor (larger than float64 tiny ≈ 2.2e-308 but still
    # prevents log(0); tiny would cause log(dist) ≈ −708 and bias the linear fit)
    dist = np.maximum(dist, 1e-15)
    log_dist = np.log(dist)

    skip = max(1, int(T * skip_fraction))
    t_vals = np.arange(len(log_dist) - skip, dtype=np.float64)
    y_vals = log_dist[skip:]

    if len(t_vals) < 2:
        return 0.0

    coeffs = np.polyfit(t_vals, y_vals, deg=1)
    return float(coeffs[0])


# ══════════════════════════════════════════════════════════════════════════════
# Chaos regime classification
# ══════════════════════════════════════════════════════════════════════════════

def classify_chaos_regime(mean_lle: float) -> Dict:
    """
    基于平均 LLE 对系统混沌状态进行分类。

    分类标准（基于每步 LLE）：

      λ < -0.01       → stable（稳定收敛）
      -0.01 ≤ λ < 0   → marginal_stable（边缘稳定，弱收敛）
      0 ≤ λ < 0.01    → edge_of_chaos（混沌边缘，中性稳定）
      0.01 ≤ λ < 0.1  → weakly_chaotic（弱混沌）
      λ ≥ 0.1         → strongly_chaotic（强混沌）

    Args:
        mean_lle: 所有轨迹的平均最大 Lyapunov 指数。

    Returns:
        {
          "regime":           str,    # 分类标签
          "is_chaotic":       bool,   # λ ≥ 0 → True
          "near_chaos_edge":  bool,   # |λ| < 0.01 → True
          "interpretation_zh": str,   # 中文解释
        }
    """
    if mean_lle < _THRESH_STRONGLY_STABLE:
        regime = "stable"
        is_chaotic = False
        near_edge = False
        interpretation = (
            f"系统稳定收敛（λ={mean_lle:.4f}）。扰动随时间指数衰减，"
            "轨迹对初始条件不敏感，不存在混沌行为。"
        )
    elif mean_lle < _THRESH_MARGINAL:
        regime = "marginal_stable"
        is_chaotic = False
        near_edge = abs(mean_lle) < 0.01
        interpretation = (
            f"系统弱收敛（λ={mean_lle:.4f}）。接近中性稳定，"
            "系统对扰动不强烈排斥，可能存在慢速衰减的亚稳态。"
        )
    elif mean_lle < _THRESH_EDGE_CHAOS:
        regime = "edge_of_chaos"
        is_chaotic = False
        near_edge = True
        interpretation = (
            f"系统处于混沌边缘（λ={mean_lle:.4f}≈0）。"
            "扰动既不增长也不收缩，系统在有序与混沌之间徘徊——"
            "这是神经系统计算最优性能的工作点（Langton, 1990）。"
        )
    elif mean_lle < _THRESH_WEAK_CHAOS:
        regime = "weakly_chaotic"
        is_chaotic = True
        near_edge = True
        interpretation = (
            f"系统弱混沌（λ={mean_lle:.4f}>0）。轨迹缓慢发散，"
            "对初始条件敏感，但混沌强度较低，可能仍有可辨识的吸引子结构。"
        )
    else:
        regime = "strongly_chaotic"
        is_chaotic = True
        near_edge = False
        interpretation = (
            f"系统强混沌（λ={mean_lle:.4f}>>0）。轨迹快速发散，"
            "对初始条件高度敏感，不存在稳定吸引子，长期预测不可靠。"
        )

    return {
        "regime": regime,
        "is_chaotic": is_chaotic,
        "near_chaos_edge": near_edge,
        "interpretation_zh": interpretation,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main analysis entry point
# ══════════════════════════════════════════════════════════════════════════════

def run_lyapunov_analysis(
    trajectories: np.ndarray,
    simulator,
    epsilon: float = _DEFAULT_EPSILON,
    renorm_steps: int = _DEFAULT_RENORM_STEPS,
    skip_fraction: float = 0.1,
    method: str = "wolf",
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹估计最大 Lyapunov 指数，并给出混沌评估。

    Args:
        trajectories:   shape (n_init, steps, n_regions)。
        simulator:      ``BrainDynamicsSimulator`` 实例（WC 模式最优）。
        epsilon:        扰动幅度（默认 1e-5）。
        renorm_steps:   Wolf 方法的重归一化周期步数（默认 20）。
        skip_fraction:  FTLE 方法跳过前段比例（仅 method="ftle" 时使用）。
        method:         "wolf"（默认，推荐）或 "ftle"（快速对照）。
        output_dir:     保存 lyapunov_values.npy、log_growth_curve.npy、
                        lyapunov_report.json；None → 不保存。

    Returns:
        results: {
          "lyapunov_values":      np.ndarray (n_init,)  每条轨迹的 LLE
          "mean_lyapunov":        float
          "median_lyapunov":      float
          "std_lyapunov":         float
          "fraction_positive":    float      λ > 0（混沌）的比例
          "fraction_negative":    float      λ < 0（收敛）的比例
          "log_growth_curve":     np.ndarray 平均对数增长曲线（Wolf 方法）
          "chaos_regime":         Dict       混沌状态分类结果
          "method":               str        使用的估计方法
        }
    """
    n_traj, total_steps, n_regions = trajectories.shape
    logger.info(
        "Lyapunov 指数分析: method=%s, %d 条轨迹, 每条 %d 步, ε=%.2e",
        method,
        n_traj,
        total_steps,
        epsilon,
    )

    rng = np.random.default_rng(42)
    values = np.zeros(n_traj, dtype=np.float64)
    all_log_growth: List[np.ndarray] = []
    log_interval = max(1, n_traj // 5)

    for i in range(n_traj):
        traj_i = trajectories[i]   # (T, N)
        x0 = traj_i[0].copy()

        if method == "wolf":
            lle, lg = wolf_largest_lyapunov(
                simulator=simulator,
                x0=x0,
                total_steps=total_steps,
                renorm_steps=renorm_steps,
                epsilon=epsilon,
                rng=rng,
            )
            values[i] = lle
            all_log_growth.append(lg)
        else:
            values[i] = ftle_lyapunov(
                trajectory=traj_i,
                simulator=simulator,
                epsilon=epsilon,
                skip_fraction=skip_fraction,
                rng=rng,
            )

        if (i + 1) % log_interval == 0:
            logger.info("  %d/%d 轨迹完成  当前均值 λ=%.4f", i + 1, n_traj, float(np.mean(values[:i+1])))

    values = values.astype(np.float32)
    mean_lam = float(np.mean(values))
    median_lam = float(np.median(values))
    std_lam = float(np.std(values))
    frac_pos = float((values > 0).mean())
    frac_neg = float((values < 0).mean())

    # Mean log-growth curve (Wolf method only)
    if all_log_growth:
        min_len = min(len(lg) for lg in all_log_growth)
        log_growth_curve = np.stack([lg[:min_len] for lg in all_log_growth]).mean(axis=0)
    else:
        log_growth_curve = np.array([], dtype=np.float64)

    # Chaos regime classification
    chaos_info = classify_chaos_regime(mean_lam)

    # Logging summary
    logger.info(
        "  均值 λ=%.5f  中位数=%.5f  std=%.5f  "
        "混沌比例=%.1f%%  收敛比例=%.1f%%",
        mean_lam, median_lam, std_lam,
        frac_pos * 100, frac_neg * 100,
    )
    logger.info("  混沌评估: [%s] %s", chaos_info["regime"].upper(), chaos_info["interpretation_zh"])

    results: Dict = {
        "lyapunov_values": values,
        "mean_lyapunov": mean_lam,
        "median_lyapunov": median_lam,
        "std_lyapunov": std_lam,
        "fraction_positive": frac_pos,
        "fraction_negative": frac_neg,
        "log_growth_curve": log_growth_curve,
        "chaos_regime": chaos_info,
        "method": method,
        "renorm_steps": int(renorm_steps),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "lyapunov_values.npy", values)
        logger.info("  → 已保存: %s/lyapunov_values.npy", output_dir)

        if len(log_growth_curve) > 0:
            np.save(output_dir / "log_growth_curve.npy", log_growth_curve)
            logger.info("  → 已保存: %s/log_growth_curve.npy", output_dir)

        # Save human-readable JSON report
        report = {
            "mean_lyapunov": mean_lam,
            "median_lyapunov": median_lam,
            "std_lyapunov": std_lam,
            "fraction_positive": frac_pos,
            "fraction_negative": frac_neg,
            "n_trajectories": int(n_traj),
            "total_steps": int(total_steps),
            "epsilon": epsilon,
            "renorm_steps": int(renorm_steps),
            "method": method,
            "chaos_regime": chaos_info["regime"],
            "is_chaotic": chaos_info["is_chaotic"],
            "near_chaos_edge": chaos_info["near_chaos_edge"],
            "interpretation": chaos_info["interpretation_zh"],
        }
        with open(output_dir / "lyapunov_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s/lyapunov_report.json", output_dir)

    return results
