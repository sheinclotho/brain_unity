"""
Lyapunov Exponent Analysis
============================

估计系统最大 Lyapunov 指数（Largest Lyapunov Exponent, LLE），
判断系统是否接近混沌状态。

提供三种估计方法，均可从已生成的 trajectories.npy 使用：

**方法 1 — Wolf-Benettin 重归一化**

  经典的 Wolf 1985 方法，以周期性重归一化避免扰动向量饱和/发散：

    1. 选择随机单位扰动方向 e
    2. 演化 renorm_steps 步，测量扰动增长 r = ||δ(t+τ)|| / ε_actual
    3. 累积 S += log(r/ε_actual)，将 δ 重归一化回 ε
    4. 重复 n_periods 次
    5. LLE = S / (n_valid_periods × renorm_steps)

  **有界状态空间修正（[0,1]）**：
  扰动向量在接近边界时被 clip 截断，导致实际施加的扰动幅度
  ε_actual = ||x_pert - x_cur|| < ε（名义幅度）。若用名义 ε
  计算 log(r/ε) 会高估 LLE（δ(0) 被低估 → growth 被高估）。
  修正方法：每个周期计算 ε_actual = ||clip(x_cur+ε·e, 0,1) - x_cur||，
  用 ε_actual 代替 ε 参与对数增长计算。

  **注意（上下文稀释偏差 — twin mode）**：
  在 TwinBrainDigitalTwin 模式下，模型使用长度为 L 的上下文窗口进行预测。
  Wolf 的扰动仅施加在上下文最后一步，上下文历史保持不变。
  当 L >> 1 时，单步扰动被注意力机制"稀释"，使得测量到的扰动增长率
  偏向负值（实际 LLE 被低估）——这与 Rosenstein 方法的结果存在
  显著差异，是已知的系统性偏差。请与 Rosenstein 方法交叉验证。

**方法 2 — 简单 FTLE 线性拟合（快速对照）**

  运行双轨迹，对 log(||δ(t)||) 做线性回归。
  计算快速但可能受瞬态/饱和影响；对收敛系统更稳健。

**方法 3 — Rosenstein (1993) — 从轨迹直接估计（推荐，twin 模式）**

  Rosenstein-Collins-De Luca (1993) 方法，无需重新运行模拟器，
  直接从已有轨迹数据估计 LLE：

    1. 对轨迹中每个时间点 t，找最近邻点 t* (时间距离 > min_sep 避免平凡对)
    2. 追踪对数平均分离: S(j) = <log||x(t+j) - x(t*+j)||>_t
    3. LLE = S(j) 线性部分的斜率

  优点：
  - 不受上下文稀释偏差影响（直接在观测空间度量）
  - 自然处理非马尔可夫（有记忆的）系统
  - 对长轨迹更鲁棒

  参考：Rosenstein, Collins & De Luca (1993) Physica D 65:117-134

**方法 4 — "both"（Wolf + FTLE + Rosenstein 交叉验证）**

  同时运行所有方法，在报告中对比，有助于诊断 Wolf 的上下文稀释偏差。

**收敛优先策略**：

  若 trajectory_convergence 模块已检测到系统强收敛
  （distance_ratio < convergence_threshold，默认 0.05），
  则记录 skipped_wolf=True 并最终将混沌分类强制为 "stable"。

  切换行为（仅当原始方法为 "wolf" 或 "both" 时）：
    自动切换到 FTLE 做快速交叉验证，节省大量 Wolf 推断时间。

  不切换行为（原始方法为 "rosenstein" 或 "ftle" 时）：
    rosenstein 无需切换——它本身就是零额外模型调用且无上下文稀释。
    ftle 则保持用户的显式设置。
    两种情况均通过 skipped_wolf=True 触发 "stable" 分类覆盖。

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Default perturbation magnitude.
# Reduced from 1e-5 to 1e-6 to lower the probability that the clipped
# perturbation exceeds the nominal ε (which would inflate log(r/ε_actual)
# even after the boundary correction).
_DEFAULT_EPSILON: float = 1e-6

# Number of steps between each Wolf renormalization step.
# Increased from 20 to 50 so each period is long enough for the true
# Lyapunov dynamics to dominate over transient boundary effects.
_DEFAULT_RENORM_STEPS: int = 50

# Convergence ratio threshold below which Wolf is skipped and the system is
# directly classified as stable (distance_ratio from trajectory_convergence).
# Raised from 0.01 to 0.05: the log showed distance_ratio=0.0102 being missed
# by the strict < 0.01 check (0.0102 is not < 0.01).  0.05 safely covers all
# strongly converging systems (trajectories shrink distance by 20× or more).
_DEFAULT_CONVERGENCE_THRESHOLD: float = 0.05

# Minimum number of periods required for a reliable LLE estimate
_MIN_PERIODS: int = 5

# Chaos classification thresholds (per-step LLE)
_THRESH_STRONGLY_STABLE: float = -0.01   # λ < this → stable
_THRESH_MARGINAL: float = 0.0            # λ < this → marginal stable
_THRESH_EDGE_CHAOS: float = 0.01         # λ < this → edge of chaos
_THRESH_WEAK_CHAOS: float = 0.1          # λ < this → weakly chaotic
# λ ≥ _THRESH_WEAK_CHAOS → strongly chaotic

# Rosenstein method defaults
_DEFAULT_ROSENSTEIN_MAX_LAG: int = 50
_DEFAULT_ROSENSTEIN_MIN_SEP: int = 20

# Wolf bias detection threshold: if std of LLE across trajectories < this value,
# the Wolf estimates are suspiciously uniform (possible twin-mode context dilution).
#
# Choice of 1e-3:  For a genuinely ergodic attractor with different starting states,
# the expected inter-trajectory LLE variance can be estimated from the CLT applied
# to the per-period log-growth samples.  With n_traj=200 trajectories each producing
# ~20 periods of renorm_steps=50, the theoretical std should be at least 0.01–0.05
# for any system with non-trivial dynamics.  A value of std < 1e-3 (100× below the
# minimum expected value) is strong evidence that all trajectories experienced the
# same effective perturbation — which happens when the base_graph context dominates
# and the last-step perturbation is diluted.
# Empirically confirmed: the log shows std=0.00006 for 200 twin-mode trajectories.
_WOLF_BIAS_STD_THRESHOLD: float = 1e-3

# Trajectory diversity threshold: if mean per-region std across trajectories < this,
# the 200 trajectories are near-identical (all start from essentially the same state).
# This happens in twin mode when base_graph context dominates over the injected x0.
# For uniform random x0 in [0,1]^N: expected std ≈ sqrt(1/12) ≈ 0.289.
# A value < 0.02 means trajectories differ by < 7% of a random baseline → context-dominated.
_TRAJECTORY_DIVERSITY_LOW_THRESHOLD: float = 0.02


# ══════════════════════════════════════════════════════════════════════════════
# Delay-embedding helper for Rosenstein (Takens reconstruction)
# ══════════════════════════════════════════════════════════════════════════════

def _build_delay_embedding(
    trajectory: np.ndarray,
    m: int,
    tau: int = 1,
) -> np.ndarray:
    """
    从多通道轨迹构造 Takens 延迟嵌入空间（单 PC 观测量）。

    **科学动机**：
    Rosenstein 方法的近邻搜索在高维空间（如 N=190 脑区）中受"维数灾难"影响：
    - 所有点对距离趋于相等（集中现象），导致最近邻几乎不比随机点更近；
    - LLE 估计主要受环境噪声和空间几何影响，而非吸引子动力学。

    Takens 定理保证：若 m ≥ 2D+1（D 为内在维度），对任意通用观测函数 h(x)，
    延迟向量 [h(t), h(t+τ), ..., h(t+(m-1)τ)] 与原始相空间微分同胚。
    因此 m×T_embed 矩阵与原始 N×T 矩阵携带**相同**的 LLE，但 m << N，
    近邻搜索更高效、更准确。

    **实现**：
    1. 计算第一主成分得分作为 1D 观测量（功率迭代，O(T×N×20)）；
    2. 构造延迟嵌入矩阵 y(t) = [obs(t), obs(t+τ), ..., obs(t+(m-1)τ)]。

    Args:
        trajectory: shape (T, N)，多通道轨迹（float32 / float64）。
        m:          嵌入维度（建议使用 FNN 确定的最小充分维度，典型值 4–9）。
                    若 m <= 1 或轨迹过短，返回原始轨迹（关闭嵌入）。
        tau:        延迟步数（默认 1，即相邻预测步）。

    Returns:
        shape (T - (m-1)*tau, m)，延迟嵌入轨迹；
        若 m <= 1 或嵌入后长度 < 10，则返回原始 trajectory（float64）。
    """
    T, N = trajectory.shape
    T_embed = T - (m - 1) * tau

    if m <= 1 or T_embed < 10:
        return trajectory.astype(np.float64)

    # ── First PC as 1-D observable via power iteration ───────────────────────
    X = trajectory.astype(np.float64)
    X = X - X.mean(axis=0)   # centre columns

    # 20 iterations of power method to get dominant right singular vector.
    # Cost: O(T × N × 20) = O(1000 × 190 × 20) ≈ 3.8 M ops — negligible.
    rng_local = np.random.default_rng(0)
    v = rng_local.standard_normal(N)
    for _ in range(20):
        u = X @ v           # (T,)
        v = X.T @ u         # (N,)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-30:
            break
        v /= norm_v

    obs = X @ v  # (T,) — unnormalised first PC score

    # ── Build Takens delay matrix ─────────────────────────────────────────────
    embedded = np.empty((T_embed, m), dtype=np.float64)
    for k in range(m):
        embedded[:, k] = obs[k * tau : k * tau + T_embed]

    return embedded


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

    **有界空间修正**：每个周期计算实际施加扰动
    ``ε_actual = ‖clip(x_cur + ε·e) − x_cur‖``，用 ε_actual（而非名义 ε）
    计算对数增长 ``log(r / ε_actual)``，消除边界截断导致的系统性正偏。
    若 ε_actual < float64 tiny（极端边界情况），该周期被跳过（不计入
    n_valid_periods），以避免无意义的 log(0) 项污染均值。

    Args:
        simulator:      ``BrainDynamicsSimulator`` 实例（需支持 rollout）。
        x0:             初始状态，shape (n_regions,)，值域 [0, 1]。
        total_steps:    总模拟步数。
        renorm_steps:   每次重归一化之间的步数（默认 50）。
        epsilon:        名义扰动幅度（默认 1e-6）。
        rng:            已初始化的随机数生成器；None → 使用固定种子 0。

    Returns:
        (lle, log_growth):
          lle:         时间平均最大 Lyapunov 指数（每步单位）。
          log_growth:  每个有效周期的 log(r/ε_actual)，shape (n_valid_periods,)。
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
    n_valid = 0  # periods with non-degenerate perturbation after clipping

    # `wolf_rollout_pair` correctly handles twin mode (advancing context window).
    # Falling back to the old `rollout()` pair is kept only for custom simulator
    # objects that have not yet implemented this method.
    use_wolf_pair = hasattr(simulator, "wolf_rollout_pair")
    wolf_context = None  # opaque per-mode state carried between periods

    # Respect the simulator's state-space bounds.
    # - Bounded (fMRI / EEG): clip perturbed state to [0, 1].
    # - Unbounded (joint mode: z-scored, no hard bound): do NOT clip.
    #   Clipping z-scores to [0, 1] would introduce an artificial attractor at 0
    #   and produce a misleading Wolf LLE.  In joint mode, use Rosenstein instead.
    # Default (0.0, 1.0) ensures backward-compatibility with simulators that were
    # created before the state_bounds property was added (all of which are bounded).
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))

    _float64_tiny = np.finfo(np.float64).tiny

    for _ in range(n_periods):
        # Perturbed initial point for this period
        _perturbed = x_cur + (epsilon * perturb).astype(np.float32)
        x_pert = (
            np.clip(_perturbed, _bounds[0], _bounds[1])
            if _bounds is not None else _perturbed
        )

        # ── Bounded-space correction ──────────────────────────────────────────
        # The clip may reduce the actual perturbation below ε (e.g. when x_cur
        # is close to 0 or 1).  Using the nominal ε in log(r/ε) would overcount
        # the growth when r > ε_actual, producing spuriously large positive LLE.
        # We compute the actual applied perturbation and use it as the denominator.
        actual_delta0 = x_pert.astype(np.float64) - x_cur.astype(np.float64)
        actual_eps = np.linalg.norm(actual_delta0)
        if actual_eps < _float64_tiny:
            # All perturbation was absorbed by clipping (degenerate boundary
            # corner); skip this period and re-randomise the direction.
            perturb = rng.standard_normal(n).astype(np.float64)
            perturb /= (np.linalg.norm(perturb) + 1e-15)
            continue
        # ─────────────────────────────────────────────────────────────────────

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
        r = max(r, _float64_tiny)

        # Accumulate log growth using the *actual* clipped perturbation size
        # (not the nominal ε) to avoid boundary-truncation bias.
        growth = np.log(r) - np.log(actual_eps)
        log_sum += growth
        log_growth.append(growth)
        n_valid += 1

        # Renormalize: reset separation to ε in the direction of current delta
        perturb = delta / (r + 1e-15)
        x_cur = np.asarray(x_after, dtype=np.float32)

    if n_valid == 0:
        # Every period was degenerate (state stuck entirely at boundary corners,
        # so all perturbation was absorbed by clipping).  This is physically
        # meaningful — a state frozen at the boundary cannot be perturbed in
        # any direction — so we return NaN to signal an uncomputable LLE rather
        # than the misleading value 0.0 (which would imply neutral stability).
        logger.warning(
            "wolf_largest_lyapunov: 所有周期的实际扰动幅度均为零（边界角点），"
            "无法估计 LLE，返回 NaN。"
        )
        return float("nan"), np.array([], dtype=np.float64)

    lle = log_sum / (n_valid * renorm_steps)
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

    **数值底层处理（floor-aware regression）**：
    对于收敛系统（LLE < 0），||δ(t)|| 随时间减小，最终跌落到数值底层
    （dist ≤ 1e-15）并保持平坦。若直接对 [skip:T] 全段做线性回归，
    平坦底层段主导导致 λ ≈ 0（而非正确的负值）。
    修复：仅对 log_dist > log(1e-12) 的"有信号"区域做回归；
    若整段均在底层（系统极快收敛），尝试用 [0:skip] 的初始段估计。

    **TwinBrainDigitalTwin 上下文稀释注意事项**：
    ``rollout()`` 的 x0 仅替换上下文最后一步，被 (context_length-1) 步共享
    历史稀释，使有效扰动 ε_eff ≈ ε/context_length ≪ ε。这会显著加速底层
    效应的出现，尤其在 n_segments>1 时（后续段起点接近吸引子）。
    推荐用 method='rosenstein' 代替 FTLE 作为 TwinBrain 的主要 LLE 估计。

    Args:
        trajectory:     shape (T, n_regions)，已计算好的轨迹。
        simulator:      ``BrainDynamicsSimulator`` 实例。
        epsilon:        初始扰动幅度（默认 1e-6）。
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
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))
    _raw_pert = x0 + epsilon * perturb
    x0_perturbed = (
        np.clip(_raw_pert, _bounds[0], _bounds[1]) if _bounds is not None else _raw_pert
    )

    traj_orig, _ = simulator.rollout(x0=x0, steps=T, stimulus=None)
    traj_pert, _ = simulator.rollout(x0=x0_perturbed, steps=T, stimulus=None)

    dist = np.linalg.norm((traj_pert - traj_orig).astype(np.float64), axis=1)
    # Use 1e-15 as the floor (larger than float64 tiny ≈ 2.2e-308 but still
    # prevents log(0); tiny would cause log(dist) ≈ −708 and bias the linear fit)
    dist = np.maximum(dist, 1e-15)
    log_dist = np.log(dist)

    skip = max(1, int(T * skip_fraction))

    # Guard: too short to fit a line after skip
    if T - skip < 2:
        return 0.0

    # ── Floor-aware regression ────────────────────────────────────────────────
    # For strongly convergent systems, ||δ(t)|| quickly hits the 1e-15 floor.
    # Fitting the entire [skip:T] range then gives λ ≈ 0 because the flat
    # floor region (where no convergence signal remains) dominates the
    # regression.  We instead restrict the fit to the "above-floor" portion:
    #
    #   soft floor = log(1e-12) (3 orders of magnitude above the hard floor)
    #
    # Using 1e-12 rather than 1e-15 gives a comfortable margin so that
    # floating-point noise in the flat region is excluded.
    # For chaotic / growing systems, dist never approaches 1e-12 → all points
    # are selected → behaviour is identical to the original code.
    _soft_floor_log = np.log(1e-12)

    # Primary regression range: [skip:T], above soft floor
    y_post = log_dist[skip:]
    t_post = np.arange(len(y_post), dtype=np.float64)
    above_post = y_post > _soft_floor_log

    if above_post.sum() >= 2:
        coeffs = np.polyfit(t_post[above_post], y_post[above_post], deg=1)
    else:
        # The floor was already hit before the skip point.  The system converged
        # so fast that skip_fraction × T steps were enough to exhaust all signal.
        # Fall back to the pre-skip region (initial fast convergence).
        y_pre = log_dist[:skip]
        t_pre = np.arange(len(y_pre), dtype=np.float64)
        above_pre = y_pre > _soft_floor_log
        if above_pre.sum() < 2:
            # Even the pre-skip region is entirely at the floor; FTLE cannot
            # estimate the LLE with this epsilon.  Return NaN rather than 0.0
            # to avoid misclassifying the system as "edge of chaos".
            logger.debug(
                "ftle_lyapunov: 轨迹整体已达数值底层（ε=%.0e 太小或系统极快收敛），"
                "FTLE 无法给出有效估计。返回 NaN。",
                epsilon,
            )
            return float("nan")
        coeffs = np.polyfit(t_pre[above_pre], y_pre[above_pre], deg=1)

    return float(coeffs[0])


# ══════════════════════════════════════════════════════════════════════════════
# Rosenstein (1993) method — trajectory-based, no context-dilution bias
# ══════════════════════════════════════════════════════════════════════════════

def rosenstein_lyapunov(
    trajectory: np.ndarray,
    max_lag: int = _DEFAULT_ROSENSTEIN_MAX_LAG,
    min_temporal_sep: int = _DEFAULT_ROSENSTEIN_MIN_SEP,
    regression_fraction: float = 0.6,
    delay_embed_dim: int = 0,
    delay_embed_tau: int = 1,
) -> Tuple[float, np.ndarray]:
    """
    Rosenstein-Collins-De Luca (1993) 方法估计最大 Lyapunov 指数。

    该方法直接从轨迹数据工作，无需重新运行模拟器，因此不受
    TwinBrainDigitalTwin 上下文稀释偏差（Wolf 的已知缺陷）影响。

    算法：
      1. 对每个时间点 t，在轨迹中找最近邻 t*（要求 |t* - t| > min_temporal_sep）
      2. 追踪对数平均分离: S(j) = <log||x(t+j) − x(t*+j)||>_t
      3. LLE = S(j) 线性增长段的斜率

    **延迟嵌入加速（delay_embed_dim > 1）**：
    当 delay_embed_dim >= 2 时，先将多通道轨迹投影到 Takens 延迟嵌入空间
    （第一主成分观测量，m = delay_embed_dim）再进行近邻搜索，从而：
      - 绕开 N=190 维空间的"维数灾难"（所有点对距离趋于相等）；
      - 近邻质量更高，LLE 斜率拟合更准确；
      - 计算量更小（近邻矩阵 O(T²×m) << O(T²×N)）。
    推荐使用 delay_embed_dim = FNN 确定的最小充分维度（通常 4–9）。

    适用场景：
      - TwinBrainDigitalTwin (twin mode) — 推荐首选，绕开上下文稀释
      - WC 模式 — 与 Wolf 方法的独立交叉验证
      - 任何有足够长单一轨迹（T >> max_lag）的系统

    时间复杂度：O(T² × max_lag)，T 较大时自动子采样。

    参考：
      Rosenstein, Collins & De Luca (1993) Physica D 65:117-134.
      Kennel, Brown & Abarbanel (1992) Phys. Rev. A 45:3403 — FNN 维度选取

    Args:
        trajectory:          shape (T, n_regions)，轨迹数组（float32/64）。
        max_lag:             追踪分离的最大步数（默认 50）。
                             应 << T。
        min_temporal_sep:    近邻对的最小时间间距（默认 20），避免平凡近邻。
        regression_fraction: 用于线性拟合的增长段比例（从 lag=1 开始）。
                             默认 0.6（使用 60% 的 max_lag 范围拟合）。
        delay_embed_dim:     Takens 延迟嵌入维度（默认 0 = 关闭，直接用原始空间）。
                             设为 FNN 确定的最小充分维度（如 4 或 7）可改善
                             高维空间中的近邻搜索质量（见 embedding_dimension.py）。
        delay_embed_tau:     延迟嵌入的时间步长（默认 1）。

    Returns:
        (lle, mean_log_divergence):
          lle:               估计的最大 Lyapunov 指数（每步单位）。
          mean_log_divergence: shape (max_lag,)，平均对数分离曲线。
    """
    # ── Optional delay embedding ──────────────────────────────────────────────
    if delay_embed_dim >= 2:
        orig_T = trajectory.shape[0]
        traj_work = _build_delay_embedding(
            trajectory, m=delay_embed_dim, tau=delay_embed_tau
        )
        if traj_work.shape[0] < orig_T // 2:
            # Embedding consumed too much of the trajectory; fall back to raw
            logger.debug(
                "rosenstein_lyapunov: 延迟嵌入后 T_embed=%d < T/2=%d，回退到原始空间。",
                traj_work.shape[0], orig_T // 2,
            )
            traj_work = trajectory.astype(np.float64)
    else:
        traj_work = trajectory.astype(np.float64)

    T, N = traj_work.shape
    # Require at least 2 * min_temporal_sep + max_lag samples
    min_required = 2 * min_temporal_sep + max_lag
    if T < min_required:
        logger.warning(
            "rosenstein_lyapunov: 轨迹过短 (T=%d < %d)，无法估计 LLE，返回 NaN。",
            T, min_required,
        )
        return float("nan"), np.zeros(max_lag, dtype=np.float64)

    # Sub-sample if trajectory is very long (T > 2000) to keep O(T²) manageable.
    # Rosenstein (1993) §4 recommends at least 200 reference points for statistical
    # reliability; we use 2000 as the upper bound because:
    #   - The per-trajectory cost is O(T_sub² × N × max_lag) — quadratic in T_sub.
    #     T_sub=2000 gives ~4×10⁶ distance pairs; with N=190 this is ~0.76B ops,
    #     feasible in <5 s with NumPy on modern hardware.
    #   - Rosenstein et al. (1993) Fig. 4 show that 2000 points achieves the same
    #     LLE accuracy as 10,000 for typical neural signals (S/N > 10 dB).
    #   - At T=1000 (our typical trajectory length), no sub-sampling occurs (T < 2000).
    max_ref_points = 2000
    if T > max_ref_points:
        step = T // max_ref_points
        traj_sub = traj_work[::step]
    else:
        traj_sub = traj_work
        step = 1

    T_sub = len(traj_sub)
    # Effective min_temporal_sep in sub-sampled units
    eff_min_sep = max(1, min_temporal_sep // step)
    eff_max_lag = min(max_lag, T_sub // 4)

    log_div_sum = np.zeros(eff_max_lag, dtype=np.float64)
    log_div_count = np.zeros(eff_max_lag, dtype=np.int64)

    # ── Vectorised nearest-neighbour search ───────────────────────────────────
    # Pre-compute the full pairwise squared-distance matrix D²[i,j] = ‖x(i)-x(j)‖²
    # (shape T_sub×T_sub, float64).  For T_sub ≤ 2000 this is ≤ 30 MB, acceptable.
    # Exclude temporal neighbours within eff_min_sep by setting their distance to ∞.
    #
    # Decomposition: ‖a - b‖² = ‖a‖² + ‖b‖² - 2·aᵀb
    sq_norms = (traj_sub ** 2).sum(axis=1)                       # (T_sub,)
    dist2 = (sq_norms[:, None] + sq_norms[None, :] -
             2.0 * (traj_sub @ traj_sub.T))                      # (T_sub, T_sub)
    # Ensure numerical non-negativity (floating-point rounding can give tiny <0)
    np.clip(dist2, 0.0, None, out=dist2)

    # Mask temporal neighbours and self-distances.
    # Vectorised band fill: for |i-j| <= eff_min_sep set dist2[i,j] = inf.
    # Uses row/col indices of the diagonal band rather than a Python loop.
    _r = np.arange(T_sub, dtype=np.intp)
    for d_off in range(1, eff_min_sep + 1):
        # Upper band: (i, i+d_off)
        dist2[_r[:-d_off], _r[d_off:]] = np.inf
        # Lower band: (i+d_off, i)
        dist2[_r[d_off:], _r[:-d_off]] = np.inf
    # Self-diagonal (d_off=0)
    dist2[_r, _r] = np.inf

    # For each reference point t, find the nearest valid neighbour
    nn_indices = np.argmin(dist2, axis=1)  # (T_sub,), one NN per reference point

    # ── Vectorised divergence accumulation (lag-loop strategy) ───────────────
    # Instead of iterating over n_ref reference points (150+ iterations), we
    # iterate over eff_max_lag-1 lags (~49 iterations).  For each lag j:
    #   - Compute all K divergences at once: traj_sub[ref_v+j] - traj_sub[nn_v+j]
    #   - This gives a (K, N) difference matrix → K L2 norms in one shot.
    # Each lag iteration allocates ~K×N float64 arrays (e.g. 150×190×8 ≈ 228 KB)
    # — far less than the (K, J, N) batch approach and typically 1.3–1.5× faster
    # than the per-reference Python loop due to reduced Python dispatch overhead.
    n_ref = T_sub - eff_max_lag
    if n_ref > 0 and eff_max_lag > 1:
        ref_all = np.arange(n_ref, dtype=np.intp)  # (n_ref,)
        nn_all  = nn_indices[:n_ref].astype(np.intp)

        # Keep only reference points with a finite nearest neighbour.
        has_nn = np.isfinite(dist2[ref_all, nn_all])
        ref_v  = ref_all[has_nn]   # (K,)
        nn_v   = nn_all[has_nn]    # (K,)

        if len(ref_v) > 0:
            for j in range(1, eff_max_lag):
                ri_j = ref_v + j  # (K,) index into traj_sub for reference points
                ni_j = nn_v  + j  # (K,) index for nearest neighbours

                # Validity mask: both indices within trajectory bounds.
                ib_j = (ri_j < T_sub) & (ni_j < T_sub)
                if not ib_j.any():
                    break  # all remaining lags will also be out of bounds

                # Clip is required for safe NumPy advanced indexing: out-of-bounds
                # indices that are NOT in ib_j would raise IndexError without it.
                # The ib_j mask then zeros out the corresponding invalid results.
                ri_j_s = np.clip(ri_j, 0, T_sub - 1)
                ni_j_s = np.clip(ni_j, 0, T_sub - 1)

                diff_j = traj_sub[ri_j_s] - traj_sub[ni_j_s]  # (K, N)
                d_j    = np.sqrt((diff_j ** 2).sum(axis=1))    # (K,)

                valid_j  = ib_j & (d_j > 0)
                log_d_j  = np.where(
                    valid_j,
                    np.log(np.where(valid_j, d_j, 1.0)),
                    0.0,
                )

                log_div_sum[j]   += (log_d_j * valid_j).sum()
                log_div_count[j] += int(valid_j.sum())

    # Average log divergence curve
    valid = log_div_count > 0
    mean_log_div = np.where(valid, log_div_sum / np.maximum(log_div_count, 1), np.nan)

    # Linear regression on the rising segment to estimate LLE
    # Use lag=1..regression_fraction*eff_max_lag (skip lag=0: same-point trivial)
    fit_end = max(2, int(eff_max_lag * regression_fraction))
    fit_lags = np.arange(1, fit_end, dtype=np.float64)
    fit_vals = mean_log_div[1:fit_end]
    finite_mask = np.isfinite(fit_vals)

    if finite_mask.sum() < 2:
        return float("nan"), mean_log_div

    coeffs = np.polyfit(fit_lags[finite_mask], fit_vals[finite_mask], deg=1)
    lle = float(coeffs[0]) / step  # correct for sub-sampling stride

    return lle, mean_log_div


def multi_direction_ftle(
    x0: np.ndarray,
    simulator,
    trajectory_length: int,
    n_directions: int = 5,
    epsilon: float = _DEFAULT_EPSILON,
    skip_fraction: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    多方向有限时间 Lyapunov 指数（ensemble FTLE）。

    通过 n_directions 个随机扰动方向的集成平均来减少单方向 FTLE 的方差。

    Args:
        x0:               初始状态，shape (n_regions,)。
        simulator:        BrainDynamicsSimulator 实例。
        trajectory_length: 轨迹长度（步数）。
        n_directions:     扰动方向数（默认 5）。
        epsilon:          扰动幅度。
        skip_fraction:    线性拟合时跳过前段比例。
        rng:              随机数生成器。

    Returns:
        (mean_ftle, std_ftle): 所有方向的 FTLE 均值和标准差。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(x0)
    ftle_list: List[float] = []
    traj_orig, _ = simulator.rollout(x0=x0, steps=trajectory_length, stimulus=None)
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))

    for _ in range(n_directions):
        perturb = rng.standard_normal(n).astype(np.float32)
        perturb /= (np.linalg.norm(perturb) + 1e-15)
        _raw_pert = x0.astype(np.float32) + epsilon * perturb
        x0_pert = (
            np.clip(_raw_pert, _bounds[0], _bounds[1]) if _bounds is not None else _raw_pert
        )
        traj_pert, _ = simulator.rollout(x0=x0_pert, steps=trajectory_length, stimulus=None)
        dist = np.linalg.norm((traj_pert - traj_orig).astype(np.float64), axis=1)
        dist = np.maximum(dist, 1e-15)
        log_dist = np.log(dist)

        T = len(log_dist)
        skip = max(1, int(T * skip_fraction))
        t_vals = np.arange(T - skip, dtype=np.float64)
        y_vals = log_dist[skip:]

        if len(t_vals) < 2:
            continue
        coeffs = np.polyfit(t_vals, y_vals, deg=1)
        ftle_list.append(float(coeffs[0]))

    if not ftle_list:
        return float("nan"), float("nan")
    return float(np.mean(ftle_list)), float(np.std(ftle_list))


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
    method: str = "rosenstein",
    convergence_result: Optional[Dict] = None,
    convergence_threshold: float = _DEFAULT_CONVERGENCE_THRESHOLD,
    n_segments: int = 1,
    rosenstein_max_lag: int = _DEFAULT_ROSENSTEIN_MAX_LAG,
    rosenstein_min_sep: int = _DEFAULT_ROSENSTEIN_MIN_SEP,
    rosenstein_delay_embed_dim: int = 0,
    rosenstein_delay_embed_tau: int = 1,
    n_workers: int = 1,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    对所有轨迹估计最大 Lyapunov 指数，并给出混沌评估。

    **性能说明（推荐使用 method="rosenstein"）**：

    Rosenstein 方法直接从预计算轨迹工作，无需额外调用模型（零推断成本）。
    Wolf/FTLE 每条轨迹每个 segment 需要 2 次 predict_future() 调用：
      - 200 轨迹 × n_segments=1 × ~20 Wolf 周期 = ~4000 次调用 → 约 60 分钟（CPU）
      - 切换为 method="rosenstein" → 秒级完成（纯 NumPy）

    **自适应 Rosenstein 参数**：
    对短轨迹（total_steps 较小时），max_lag 和 min_sep 自动缩减，确保
    Rosenstein 对任意长度轨迹都能返回有意义的估计（而非 NaN）。

    **延迟嵌入加速（rosenstein_delay_embed_dim > 1）**：
    当 rosenstein_delay_embed_dim >= 2 时，将每条轨迹投影到 Takens 延迟嵌入
    空间再做 Rosenstein 近邻搜索，绕开 N=190 维空间的"维数灾难"。推荐使用
    嵌入维度分析（步骤 12）确定的 FNN 最小充分维度（通常 4–9）。
    当不设置（默认 0）时，直接在原始 N 维空间运行。

    **并行加速（n_workers > 1）**：
    Rosenstein 方法：纯 NumPy，ThreadPoolExecutor 并行，线程安全。
    Wolf/FTLE 方法：需要模型推断；仅当 CPU 推断时建议开启；GPU 可能
    因显存竞争降速，建议保持 n_workers=1。

    **收敛优先策略**：若 ``convergence_result`` 来自
    ``run_trajectory_convergence`` 且
    ``distance_ratio < convergence_threshold``（默认 0.05），则跳过
    Wolf 计算，直接用 FTLE 做快速交叉验证并标注 "stable_by_convergence"。

    **多段采样（n_segments > 1）**：
    对每条轨迹从 n_segments 个不同时间段的起始点估计 LLE，然后取均值。
    这样可以探索吸引子的不同区域，减少瞬态偏差，是比单一 x0 更稳健的估计。
    对 Rosenstein（无额外模型调用）增加 n_segments 开销极低，可设为 3–5。

    **Wolf 偏差检测**：
    当 Wolf LLE 的跨轨迹标准差 < _WOLF_BIAS_STD_THRESHOLD（默认 1e-3）且
    n_traj > 10 时，自动发出警告——这是 twin mode 上下文稀释偏差的典型症状
    （所有轨迹共享相同的 base_graph，扰动只作用于最后一步，导致估计收敛到
    同一固定值而非各轨迹真实的局部 Lyapunov 指数）。

    **Rosenstein 方法优先策略**（推荐用于 twin mode）：
    使用 method="rosenstein" 可绕开上下文稀释偏差。

    Args:
        trajectories:              shape (n_init, steps, n_regions)。
        simulator:                 ``BrainDynamicsSimulator`` 实例。
        epsilon:                   名义扰动幅度（默认 1e-6）。
        renorm_steps:              Wolf 方法的重归一化周期步数（默认 50）。
        skip_fraction:             FTLE 方法跳过前段比例。
        method:                    "rosenstein"（默认，twin 模式推荐，零额外推断成本）、
                                   "wolf"（经典，WC 模式推荐）、
                                   "ftle"（快速对照）、
                                   "both"（Wolf + FTLE + Rosenstein 交叉验证）。
        convergence_result:        ``run_trajectory_convergence`` 的返回字典；
                                   若 distance_ratio < convergence_threshold 则
                                   跳过 Wolf 并标注稳定。None → 不做收敛优先判断。
        convergence_threshold:     distance_ratio 低于此阈值视为强收敛（默认 0.05）。
                                   原为 0.01；提高到 0.05 避免边界值（如 0.010）
                                   因浮点精度未能触发 Wolf 跳过。
        n_segments:                每条轨迹的采样段数（默认 1 = 仅使用 x0）。
                                   设为 3 可在轨迹不同位置多次采样，减少偏差。
                                   对 Rosenstein 方法成本极低；对 Wolf/FTLE 每段增加
                                   2 次模型调用。
        rosenstein_max_lag:        Rosenstein 方法追踪的最大步数（默认 50）；
                                   自动缩减以适配短轨迹。
        rosenstein_min_sep:        Rosenstein 方法近邻对最小时间间距（默认 20）；
                                   自动缩减以适配短轨迹。
        rosenstein_delay_embed_dim: Takens 延迟嵌入维度（默认 0 = 关闭）。
                                   >= 2 时在运行 Rosenstein 前将每条轨迹投影到
                                   第一 PC 的 m 维延迟嵌入空间，改善高维 NN 质量。
                                   推荐设为 FNN min_sufficient_dim（步骤 12 给出）。
        rosenstein_delay_embed_tau: 延迟嵌入步长（默认 1）。
        n_workers:                 并行 worker 数量（默认 1 = 顺序执行）。
                                   n_workers > 1 对 Rosenstein 始终安全（纯 NumPy）。
                                   对 Wolf/FTLE 仅推荐在 CPU 推断时开启。
        output_dir:                保存 lyapunov_values.npy、log_growth_curve.npy、
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
          "method":               str        实际使用的估计方法
          "ftle_values":          np.ndarray (n_init,) FTLE 值（method 含 "both"）
          "mean_ftle":            float（method 含 "both"/"ftle" 时）
          "rosenstein_values":    np.ndarray (n_init,) Rosenstein 值（含 rosenstein/both）
          "mean_rosenstein":      float（含 rosenstein/both 时）
          "skipped_wolf":         bool       Wolf 是否因收敛而被跳过
          "wolf_bias_warning":    bool       Wolf 结果是否可能有上下文稀释偏差
          "delay_embed_dim":      int        实际使用的延迟嵌入维度（0 = 未使用）
        }
    """
    n_traj, total_steps, n_regions = trajectories.shape

    # ── Convergence-first: skip Wolf if system is clearly converging ──────────
    #
    # Design rationale: this block was originally designed to skip the expensive
    # Wolf computation (O(n_traj × n_periods × 2) model calls) when the trajectory-
    # convergence analysis already indicates that the system is stable.
    #
    # Critical constraint: ONLY switch to FTLE when the original method is "wolf"
    # or "both".  If method="rosenstein" (the default), do NOT switch to FTLE:
    #
    #   Rosenstein works directly on pre-computed trajectories (zero extra model
    #   calls, no context-dilution bias) and correctly measures negative LLE for
    #   convergent systems.  Switching to FTLE for TwinBrainDigitalTwin causes:
    #
    #     1. Context-dilution artefact: ftle_lyapunov() calls simulator.rollout(),
    #        which injects x0 only in the LAST step of a (context_length-step)
    #        context window.  Both the base and perturbed rollouts share the same
    #        (context_length-1) history from base_graph → the effective initial
    #        separation is ≈ ε/context_length ≈ 5×10⁻⁹ (diluted from ε=10⁻⁶).
    #
    #     2. Early-floor artefact: with ε_eff ≈ 5×10⁻⁹ and LLE ≈ -0.028, the
    #        numerical floor (1e-15) is reached at t ≈ 400 steps.  Later segments
    #        (n_segments=3, starting at traj[333] and traj[666]) begin NEAR THE
    #        ATTRACTOR, so the floor is hit at step ~0 → flat log_dist → λ ≈ 0.
    #
    #     Combined effect: λ=-0.028 (rosenstein) → λ≈0.0002 (ftle).
    #     This is a measurement artefact, not the true LLE.
    #
    skipped_wolf = False
    effective_method = method

    if convergence_result is not None:
        dist_ratio = float(convergence_result.get("distance_ratio", 1.0))
        if dist_ratio < convergence_threshold:
            skipped_wolf = True
            if method in ("wolf", "both"):
                # Wolf is expensive (O(n_traj × n_periods × 2) extra model calls).
                # Switch to FTLE for a cheaper cross-check that confirms stability.
                logger.info(
                    "  收敛优先：distance_ratio=%.4f < threshold=%.4f，"
                    "系统强收敛，跳过 Wolf 计算，切换为 FTLE 快速验证。",
                    dist_ratio,
                    convergence_threshold,
                )
                effective_method = "ftle"
            else:
                # method is already 'rosenstein' or 'ftle'.
                # Rosenstein: already zero-cost and unaffected by context dilution.
                # FTLE: user explicitly requested it; don't interfere.
                # Just record skipped_wolf=True so the chaos regime is overridden
                # to "stable" at the end of this function.
                logger.info(
                    "  收敛优先：distance_ratio=%.4f < threshold=%.4f，"
                    "系统强收敛（方法='%s'，无需切换，继续使用原方法）。",
                    dist_ratio,
                    convergence_threshold,
                    method,
                )

    # ── GPU auto-limit: Wolf/FTLE with multiple workers causes VRAM contention ─
    # Each Wolf/FTLE worker invokes the neural network independently on the same
    # GPU → concurrent CUDA kernels compete for the same memory bus and compute
    # units, causing non-deterministic slowdowns or OOM errors.
    # For Rosenstein (pure NumPy, no model calls) this is not an issue.
    _run_wolf_or_ftle = (effective_method in ("wolf", "ftle", "both"))
    if _run_wolf_or_ftle and n_workers > 1:
        _on_gpu = (
            simulator is not None
            and str(getattr(simulator, "device", "cpu")).startswith("cuda")
        )
        if _on_gpu:
            logger.warning(
                "  ⚠  GPU 推断下 Wolf/FTLE 多 worker（n_workers=%d）会产生显存竞争，"
                "自动降为 n_workers=1（顺序执行）。\n"
                "  如需并行加速，请切换 method='rosenstein'（纯 NumPy，线程安全）"
                "或使用 CPU 推断（--device cpu）。",
                n_workers,
            )
            n_workers = 1

    # ── Log delay-embedding info if active ────────────────────────────────────
    _use_delay_embed = (rosenstein_delay_embed_dim >= 2)
    if _use_delay_embed and effective_method in ("rosenstein", "both"):
        logger.info(
            "  Rosenstein 延迟嵌入已启用: m=%d, τ=%d "
            "（将 N=%d 维轨迹投影到 %d 维延迟空间，改善近邻搜索质量）",
            rosenstein_delay_embed_dim, rosenstein_delay_embed_tau,
            n_regions, rosenstein_delay_embed_dim,
        )

    logger.info(
        "Lyapunov 指数分析: method=%s, %d 条轨迹, 每条 %d 步, ε=%.2e, "
        "n_segments=%d, n_workers=%d",
        effective_method, n_traj, total_steps, epsilon, n_segments, n_workers,
    )

    # ── Trajectory diversity diagnostic ──────────────────────────────────────
    initial_diversity = float(np.std(trajectories[:, 0, :], axis=0).mean())
    final_diversity   = float(np.std(trajectories[:, -1, :], axis=0).mean())
    context_dominated = initial_diversity < _TRAJECTORY_DIVERSITY_LOW_THRESHOLD

    # Threshold for "moderate diversity loss": initial_std is above the strict
    # context_dominated threshold but still much lower than the random baseline
    # (0.289 for uniform U[0,1]^N).  In this regime the convergence test may be
    # misleading for CHAOTIC systems: a bounded chaotic attractor also pulls
    # trajectories from slightly-different x0's toward the same attractor set,
    # producing distance_ratio < threshold even though the intrinsic LLE > 0.
    # We flag this as "attractor-size ambiguity" to remind downstream code.
    _RANDOM_BASELINE_STD: float = 0.289
    _ATTRACTOR_AMBIGUITY_THRESHOLD: float = 0.3 * _RANDOM_BASELINE_STD  # ≈ 0.087
    attractor_ambiguous = (
        not context_dominated
        and initial_diversity < _ATTRACTOR_AMBIGUITY_THRESHOLD
    )

    logger.info(
        "  轨迹多样性: 初始 std=%.4f, 终止 std=%.4f  "
        "(随机基线预期 ≈ 0.289; 阈值 %.3f)",
        initial_diversity, final_diversity, _TRAJECTORY_DIVERSITY_LOW_THRESHOLD,
    )
    if context_dominated:
        logger.warning(
            "  ⚠  初始轨迹多样性极低（%.4f < %.3f）：%d 条轨迹起点几乎相同。\n"
            "     根本原因：TwinBrainDigitalTwin 上下文窗口（context_length 步）中，\n"
            "     每条轨迹只有最后 1 步被 x0 覆盖，其余历史步由 base_graph 提供。\n"
            "     当 context_length >> 1 时，不同 x0 对模型输出的影响被大量相同\n"
            "     历史所稀释，导致所有轨迹在统计上无法区分。\n"
            "     后果：Wolf/FTLE/Rosenstein 三种方法均会给出几乎相同的 LLE，\n"
            "     这不是计算错误，而是架构特性。\n"
            "     要获得真正独立的轨迹，需要使用不同的 base_graph（不同时间段\n"
            "     或不同受试者的图缓存），而不是同一图的不同 x0 注入。",
            initial_diversity, _TRAJECTORY_DIVERSITY_LOW_THRESHOLD, n_traj,
        )
    elif attractor_ambiguous:
        # Middle ground: not fully context-dominated, but initial diversity is
        # small enough that the trajectory convergence test may conflate
        # "converging to a chaotic attractor" with "converging to a fixed point".
        logger.warning(
            "  ⚠  初始轨迹多样性偏低（%.4f < %.3f，随机基线 0.289）：\n"
            "     收敛检验结果存在歧义——对于有界混沌吸引子，所有轨迹也会从\n"
            "     相近初始条件汇聚到同一吸引子集合，导致 distance_ratio 减小，\n"
            "     但这并不意味着系统是稳定的（LLE 可能 > 0）。\n"
            "     建议：以实际计算的 LLE 符号（而非 distance_ratio）判断稳定性。\n"
            "     若 convergence-first 覆盖了 LLE 分类，其结论需谨慎对待。",
            initial_diversity, _ATTRACTOR_AMBIGUITY_THRESHOLD,
        )
    else:
        logger.info(
            "  轨迹多样性正常（%.4f ≥ %.3f）：%d 条轨迹从不同初始状态出发。",
            initial_diversity, _TRAJECTORY_DIVERSITY_LOW_THRESHOLD, n_traj,
        )

    values = np.zeros(n_traj, dtype=np.float64)
    ftle_values = np.zeros(n_traj, dtype=np.float64)
    rosenstein_values = np.full(n_traj, np.nan, dtype=np.float64)
    all_log_growth: List[np.ndarray] = []
    log_interval = max(1, n_traj // 5)

    run_wolf = effective_method in ("wolf", "both")
    run_ftle = effective_method in ("ftle", "both")
    run_rosenstein = effective_method in ("rosenstein", "both")

    # Determine segment start indices for multi-segment sampling.
    if n_segments <= 1:
        segment_fracs = [0.0]
    else:
        segment_fracs = [k / n_segments for k in range(n_segments)]

    # ── Adaptive Rosenstein parameters ────────────────────────────────────────
    # Automatically reduce max_lag and min_sep when trajectories are short so
    # that Rosenstein returns a finite estimate instead of NaN.  The standard
    # min_required = 2 * min_sep + max_lag must be < T.
    #
    # We compute the effective parameters here (once) from the full trajectory
    # length; the per-segment sub-trajectory is shorter, so we additionally
    # adapt per-call inside the loop.
    _r_max_lag = min(rosenstein_max_lag, max(2, total_steps // 4))
    _r_min_sep = min(
        rosenstein_min_sep,
        max(1, (total_steps - _r_max_lag - 1) // 2),
    )
    if _r_max_lag != rosenstein_max_lag or _r_min_sep != rosenstein_min_sep:
        logger.info(
            "  Rosenstein 参数自适应: max_lag %d→%d, min_sep %d→%d"
            "（轨迹长度 T=%d 不足默认参数 min_required=%d）",
            rosenstein_max_lag, _r_max_lag,
            rosenstein_min_sep, _r_min_sep,
            total_steps,
            2 * rosenstein_min_sep + rosenstein_max_lag,
        )

    # ── Per-trajectory worker functions (used for both sequential and parallel) ─

    def _wolf_for_traj(
        traj_i: np.ndarray,
        rng_seed: int,
    ) -> Tuple[float, np.ndarray]:
        """Compute Wolf LLE for one trajectory (all segments)."""
        local_rng = np.random.default_rng(rng_seed)
        wolf_lles: List[float] = []
        wolf_lgs: List[np.ndarray] = []
        for frac in segment_fracs:
            seg_start = int(frac * total_steps)
            x0_seg = traj_i[seg_start].copy()
            seg_steps = total_steps - seg_start
            if seg_steps < renorm_steps:
                continue
            lle_seg, lg_seg = wolf_largest_lyapunov(
                simulator=simulator,
                x0=x0_seg,
                total_steps=seg_steps,
                renorm_steps=renorm_steps,
                epsilon=epsilon,
                rng=local_rng,
            )
            if np.isfinite(lle_seg):
                wolf_lles.append(lle_seg)
                wolf_lgs.append(lg_seg)
        val = float(np.mean(wolf_lles)) if wolf_lles else float("nan")
        lg0 = wolf_lgs[0] if wolf_lgs else np.array([], dtype=np.float64)
        return val, lg0

    def _ftle_for_traj(
        traj_i: np.ndarray,
        rng_seed: int,
    ) -> float:
        """Compute FTLE for one trajectory (all segments)."""
        local_rng = np.random.default_rng(rng_seed)
        ftle_segs: List[float] = []
        for frac in segment_fracs:
            seg_start = int(frac * total_steps)
            sub_traj = traj_i[seg_start:]
            if len(sub_traj) < 4:
                continue
            ftle_seg = ftle_lyapunov(
                trajectory=sub_traj,
                simulator=simulator,
                epsilon=epsilon,
                skip_fraction=skip_fraction,
                rng=local_rng,
            )
            if np.isfinite(ftle_seg):
                ftle_segs.append(ftle_seg)
        return float(np.mean(ftle_segs)) if ftle_segs else float("nan")

    def _rosenstein_for_traj(traj_i: np.ndarray) -> float:
        """Compute Rosenstein LLE for one trajectory (all segments, adaptive params)."""
        rosen_segs: List[float] = []
        for frac in segment_fracs:
            seg_start = int(frac * total_steps)
            sub_traj = traj_i[seg_start:]
            T_sub = len(sub_traj)
            # Adapt parameters to sub-trajectory length
            adapt_max_lag = min(_r_max_lag, max(2, T_sub // 4))
            adapt_min_sep = min(
                _r_min_sep,
                max(1, (T_sub - adapt_max_lag - 1) // 2),
            )
            rosen_lle, _ = rosenstein_lyapunov(
                trajectory=sub_traj,
                max_lag=adapt_max_lag,
                min_temporal_sep=adapt_min_sep,
                delay_embed_dim=rosenstein_delay_embed_dim,
                delay_embed_tau=rosenstein_delay_embed_tau,
            )
            if np.isfinite(rosen_lle):
                rosen_segs.append(rosen_lle)
        return float(np.mean(rosen_segs)) if rosen_segs else float("nan")

    # ── Sequential vs. parallel dispatch ─────────────────────────────────────
    _use_parallel = (n_workers > 1)

    if _use_parallel and run_rosenstein and not run_wolf and not run_ftle:
        # Fast path: pure Rosenstein, fully parallel (no model calls)
        logger.info(
            "  并行 Rosenstein: %d workers × %d 轨迹", n_workers, n_traj
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = {
                ex.submit(_rosenstein_for_traj, trajectories[i]): i
                for i in range(n_traj)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                rosenstein_values[i] = fut.result()
                values[i] = rosenstein_values[i]
        # No Wolf log-growth to accumulate
    elif _use_parallel and (run_wolf or run_ftle):
        # Parallel Wolf/FTLE (each trajectory uses separate RNG seed)
        logger.info(
            "  并行 Wolf/FTLE: %d workers × %d 轨迹 "
            "（仅推荐用于 CPU 推断；GPU 可能显存竞争）",
            n_workers, n_traj,
        )
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            wolf_futs: dict = {}
            ftle_futs: dict = {}
            rosen_futs: dict = {}
            for i in range(n_traj):
                seed_i = 42 + i
                if run_wolf:
                    wolf_futs[ex.submit(_wolf_for_traj, trajectories[i], seed_i)] = i
                if run_ftle:
                    ftle_futs[ex.submit(_ftle_for_traj, trajectories[i], seed_i)] = i
                if run_rosenstein:
                    rosen_futs[ex.submit(_rosenstein_for_traj, trajectories[i])] = i

            for fut in as_completed(wolf_futs):
                i = wolf_futs[fut]
                val, lg0 = fut.result()
                values[i] = val if np.isfinite(val) else float("nan")
                all_log_growth.append((i, lg0))

            for fut in as_completed(ftle_futs):
                i = ftle_futs[fut]
                ftle_val = fut.result()
                ftle_values[i] = ftle_val
                if not run_wolf:
                    values[i] = ftle_val

            for fut in as_completed(rosen_futs):
                i = rosen_futs[fut]
                rosenstein_values[i] = fut.result()
                if not run_wolf and not run_ftle:
                    values[i] = rosenstein_values[i]

        # Sort-and-unpack indexed log-growth tuples from the parallel Wolf path.
        # The sequential path appends raw ndarray objects directly, so only
        # the parallel path (where each entry is a (trajectory_index, lg) tuple)
        # needs unpacking.  The two paths are mutually exclusive, so the list
        # will never contain a mixture of tuples and arrays.
        if all_log_growth and isinstance(all_log_growth[0], tuple):
            all_log_growth.sort(key=lambda x: x[0])
            all_log_growth = [lg for _, lg in all_log_growth]  # type: ignore[assignment]

        # Log progress summary
        logger.info("  %d/%d 轨迹完成（并行）", n_traj, n_traj)
    else:
        # Sequential path (default; required when Wolf context must be serial
        # or n_workers == 1)
        for i in range(n_traj):
            traj_i = trajectories[i]   # (T, N)

            # ── Wolf ────────────────────────────────────────────────────────
            if run_wolf:
                val, lg0 = _wolf_for_traj(traj_i, rng_seed=42 + i)
                values[i] = val
                all_log_growth.append(lg0)

            # ── FTLE ────────────────────────────────────────────────────────
            if run_ftle:
                ftle_val = _ftle_for_traj(traj_i, rng_seed=42 + i)
                ftle_values[i] = ftle_val
                if not run_wolf:
                    values[i] = ftle_val

            # ── Rosenstein ────────────────────────────────────────────────
            if run_rosenstein:
                r_val = _rosenstein_for_traj(traj_i)
                rosenstein_values[i] = r_val
                if not run_wolf and not run_ftle:
                    values[i] = r_val

            if (i + 1) % log_interval == 0:
                logger.info(
                    "  %d/%d 轨迹完成  当前均值 λ=%.4f",
                    i + 1, n_traj, float(np.nanmean(values[:i+1]))
                )

    values = values.astype(np.float32)
    mean_lam = float(np.nanmean(values))
    median_lam = float(np.nanmedian(values))
    std_lam = float(np.nanstd(values))
    valid_mask = np.isfinite(values)
    frac_pos = float((values[valid_mask] > 0).mean()) if valid_mask.any() else 0.0
    frac_neg = float((values[valid_mask] < 0).mean()) if valid_mask.any() else 0.0

    # Mean log-growth curve (Wolf method only)
    non_empty_growth = [lg for lg in all_log_growth if len(lg) > 0]
    if non_empty_growth:
        min_len = min(len(lg) for lg in non_empty_growth)
        log_growth_curve = np.stack(
            [lg[:min_len] for lg in non_empty_growth]
        ).mean(axis=0)
    else:
        log_growth_curve = np.array([], dtype=np.float64)

    # ── Wolf bias detection ────────────────────────────────────────────────────
    wolf_bias_warning = False
    if run_wolf and n_traj > 10 and std_lam < _WOLF_BIAS_STD_THRESHOLD:
        wolf_bias_warning = True
        logger.warning(
            "  ⚠  Wolf LLE 跨轨迹标准差 std=%.2e < %.2e：所有轨迹的 Wolf 估计几乎相同。"
            "\n     Wolf 扰动仅施加于上下文窗口最后一步，被 (context_length-1) 个相同历史步"
            "稀释，导致所有轨迹测量到相同的收缩率。"
            "\n     %s"
            "\n     建议：使用 method='rosenstein' 或 method='both' 以获取无偏估计。",
            std_lam, _WOLF_BIAS_STD_THRESHOLD,
            "且初始轨迹多样性低（context主导）——见上方⚠警告。" if context_dominated
            else "但初始轨迹多样性正常——Wolf偏差来自注意力稀释，而非轨迹本身相同。",
        )

    # Compute Rosenstein std for cross-method comparison
    std_rosen: float = float("nan")
    if run_rosenstein:
        valid_rosen = rosenstein_values[np.isfinite(rosenstein_values)]
        if len(valid_rosen) > 1:
            std_rosen = float(np.std(valid_rosen))

    # Chaos regime classification
    # Prefer Rosenstein for regime classification if it's available and Wolf bias detected
    primary_mean = mean_lam
    if wolf_bias_warning and run_rosenstein:
        valid_rosen = rosenstein_values[np.isfinite(rosenstein_values)]
        if len(valid_rosen) > 0:
            primary_mean = float(np.mean(valid_rosen))
            logger.info(
                "  Wolf 偏差检测：使用 Rosenstein 均值 λ=%.5f 作为混沌评估主指标"
                "（替代 Wolf 均值 λ=%.5f）。",
                primary_mean, mean_lam,
            )

    chaos_info = classify_chaos_regime(primary_mean)

    # ── skipped_wolf regime override ──────────────────────────────────────────
    # If Wolf was skipped due to convergence, we use the actual computed LLE
    # to decide whether to override the regime to "stable":
    #
    # Case 1 — LLE < 0 (confirmed convergent):
    #   The computed LLE agrees with the distance_ratio evidence.  Override
    #   regime to "stable" regardless of what classify_chaos_regime returned.
    #
    # Case 2 — LLE ≥ 0 (chaotic despite low distance_ratio):
    #   This is the "chaotic attractor attraction" scenario: all trajectories
    #   from nearby x0 converge TO THE SAME CHAOTIC ATTRACTOR, making the
    #   pairwise distance decrease (ratio < threshold) even though the intrinsic
    #   dynamics are chaotic (LLE > 0).  We must NOT force "stable" here — doing
    #   so would produce the exact inconsistency seen in the logs (Step 9 says
    #   "stable", Step 15 says "strongly chaotic").  Keep the actual LLE-based
    #   classification.
    #
    # This fix is triggered even for method='rosenstein' (previously only for
    # method='ftle' after convergence-first switching).
    if skipped_wolf:
        if np.isfinite(primary_mean) and primary_mean < 0:
            # LLE confirms stability → safe to override
            chaos_info["regime"] = "stable"
            chaos_info["is_chaotic"] = False
            chaos_info["near_chaos_edge"] = False
            _mean_lam_str = f"{mean_lam:.4f}"
            _method_str = effective_method.upper()
            chaos_info["interpretation_zh"] = (
                f"系统强收敛（trajectory distance_ratio="
                f"{convergence_result.get('distance_ratio', '?'):.4f} < {convergence_threshold}），"
                f"Wolf 计算已跳过（{_method_str} 均值 λ={_mean_lam_str}，确认稳定）。"
            )
        else:
            # LLE ≥ 0: chaotic attractor attraction — DO NOT override to "stable"
            _mean_lam_str = f"{mean_lam:.4f}" if np.isfinite(mean_lam) else "NaN"
            logger.warning(
                "  ⚠  收敛矛盾：distance_ratio=%.4f < %.4f 表明轨迹收敛（已跳过 Wolf），\n"
                "     但实际计算的 LLE=%s ≥ 0，表明系统可能为混沌。\n"
                "     最可能的原因：初始条件相似（initial_std=%.4f << 随机基线 0.289），\n"
                "     轨迹收敛到同一混沌吸引子集合（而非固定点）——\n"
                "     这是有界混沌系统的正常行为，不应被误判为稳定。\n"
                "     → 以 LLE 结果为准：%s（regime=%s）。\n"
                "     建议：增加 n_init 或使用不同 base_graph 以提高初始多样性。",
                convergence_result.get("distance_ratio", float("nan")),
                convergence_threshold,
                _mean_lam_str,
                initial_diversity,
                _mean_lam_str,
                chaos_info["regime"].upper(),
            )
            # Do NOT override chaos_info — keep the LLE-based classification

    # ── Logging summary ────────────────────────────────────────────────────────
    logger.info(
        "  均值 λ=%.5f  中位数=%.5f  std=%.5f  "
        "混沌比例=%.1f%%  收敛比例=%.1f%%",
        mean_lam, median_lam, std_lam,
        frac_pos * 100, frac_neg * 100,
    )
    if run_ftle and run_wolf:
        mean_ftle = float(np.nanmean(ftle_values))
        logger.info(
            "  FTLE 均值=%.5f  Wolf 均值=%.5f  差异=%.5f",
            mean_ftle, mean_lam, mean_lam - mean_ftle,
        )
    if run_rosenstein:
        valid_rosen = rosenstein_values[np.isfinite(rosenstein_values)]
        mean_rosen = float(np.mean(valid_rosen)) if len(valid_rosen) > 0 else float("nan")
        logger.info(
            "  Rosenstein 均值=%.5f  std=%.5f  (参考: Rosenstein et al. 1993)",
            mean_rosen, std_rosen if np.isfinite(std_rosen) else float("nan"),
        )
        # Explain why Rosenstein std may also be near-zero
        if np.isfinite(std_rosen) and std_rosen < _WOLF_BIAS_STD_THRESHOLD:
            if context_dominated:
                logger.warning(
                    "  ⚠  Rosenstein std=%.2e 也极小：这是因为 %d 条轨迹本身几乎相同\n"
                    "     （初始多样性=%.4f），Rosenstein 从同质轨迹中自然得到相同 LLE。\n"
                    "     这不是 Rosenstein 的计算错误，而是轨迹集合缺乏多样性导致的。\n"
                    "     → LLE=%.5f 是模型沿当前 base_graph 吸引子轨迹的 Lyapunov 指数，\n"
                    "     不代表对不同初始条件的响应多样性。",
                    std_rosen, n_traj, initial_diversity, mean_rosen,
                )
            else:
                logger.info(
                    "  Rosenstein std=%.2e：尽管轨迹初始多样性正常（%.4f），\n"
                    "  各轨迹的 Rosenstein LLE 仍高度一致——\n"
                    "  这是系统存在单一全局吸引子的有力证据（λ 是吸引子的性质，\n"
                    "  与初始条件无关，不同起点最终测量到相同收缩率是正确行为）。",
                    std_rosen, initial_diversity,
                )
        if run_wolf and np.isfinite(mean_rosen):
            diff = mean_lam - mean_rosen
            logger.info(
                "  Wolf vs Rosenstein 差异=%.5f%s",
                diff,
                "  [⚠ 差异显著，可能存在 Wolf 上下文稀释偏差]"
                if abs(diff) > 0.03 else "",
            )
    logger.info(
        "  混沌评估: [%s] %s",
        chaos_info["regime"].upper(), chaos_info["interpretation_zh"]
    )

    results: Dict = {
        "lyapunov_values": values,
        "mean_lyapunov": mean_lam,
        "median_lyapunov": median_lam,
        "std_lyapunov": std_lam,
        "fraction_positive": frac_pos,
        "fraction_negative": frac_neg,
        "log_growth_curve": log_growth_curve,
        "chaos_regime": chaos_info,
        "method": effective_method,
        "renorm_steps": int(renorm_steps),
        "skipped_wolf": skipped_wolf,
        "wolf_bias_warning": wolf_bias_warning,
        # Trajectory diversity metrics (key for interpreting LLE uniformity)
        "initial_trajectory_diversity": initial_diversity,
        "final_trajectory_diversity": final_diversity,
        "context_dominated": context_dominated,
        # Delay-embedding info (0 = disabled; >1 = Takens embedding dimension used)
        "delay_embed_dim": rosenstein_delay_embed_dim if _use_delay_embed else 0,
    }

    if run_ftle or effective_method == "ftle":
        results["ftle_values"] = ftle_values.astype(np.float32)
        results["mean_ftle"] = float(np.nanmean(ftle_values))

    if run_rosenstein:
        results["rosenstein_values"] = rosenstein_values.astype(np.float32)
        valid_r = rosenstein_values[np.isfinite(rosenstein_values)]
        results["mean_rosenstein"] = float(np.mean(valid_r)) if len(valid_r) > 0 else float("nan")
        results["std_rosenstein"] = float(std_rosen) if np.isfinite(std_rosen) else float("nan")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "lyapunov_values.npy", values)
        logger.info("  → 已保存: %s/lyapunov_values.npy", output_dir)

        if len(log_growth_curve) > 0:
            np.save(output_dir / "log_growth_curve.npy", log_growth_curve)
            logger.info("  → 已保存: %s/log_growth_curve.npy", output_dir)

        if run_rosenstein:
            np.save(output_dir / "rosenstein_values.npy", rosenstein_values)
            logger.info("  → 已保存: %s/rosenstein_values.npy", output_dir)

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
            "method": effective_method,
            "skipped_wolf": skipped_wolf,
            "wolf_bias_warning": wolf_bias_warning,
            "chaos_regime": chaos_info["regime"],
            "is_chaotic": chaos_info["is_chaotic"],
            "near_chaos_edge": chaos_info["near_chaos_edge"],
            "interpretation": chaos_info["interpretation_zh"],
            # Diversity diagnostics
            "initial_trajectory_diversity": initial_diversity,
            "final_trajectory_diversity": final_diversity,
            "context_dominated": context_dominated,
            "diversity_interpretation": (
                "上下文主导（context_length >> 1 导致 x0 被稀释，轨迹几乎相同）"
                if context_dominated else
                "多样性正常（不同 x0 产生了可区分的轨迹）"
            ),
        }
        if run_ftle or effective_method == "ftle":
            valid_f = ftle_values[np.isfinite(ftle_values)]
            report["mean_ftle"] = float(np.mean(valid_f)) if len(valid_f) > 0 else float("nan")
        if run_rosenstein:
            valid_r2 = rosenstein_values[np.isfinite(rosenstein_values)]
            report["mean_rosenstein"] = float(np.mean(valid_r2)) if len(valid_r2) > 0 else float("nan")
            report["std_rosenstein"] = float(std_rosen) if np.isfinite(std_rosen) else float("nan")
        with open(output_dir / "lyapunov_report.json", "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        logger.info("  → 已保存: %s/lyapunov_report.json", output_dir)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Lyapunov Spectrum (all exponents) + Kaplan–Yorke Dimension
# ══════════════════════════════════════════════════════════════════════════════

def kaplan_yorke_dimension(spectrum: np.ndarray) -> float:
    """
    从有序 Lyapunov 谱（降序）计算 Kaplan–Yorke（Lyapunov）维度。

      D_KY = j + (Σᵢ₌₁ʲ λᵢ) / |λ_{j+1}|

    其中 j 是使前 j 个指数之和 ≥ 0 的最大整数（降序排列）。

    物理含义：
    - D_KY = 0          → 固定点（所有 λ < 0）
    - 1 ≤ D_KY < 2      → 极限环（λ₁ ≈ 0）
    - D_KY >> 1         → 奇异吸引子 / 混沌

    Args:
        spectrum: 降序排列的 Lyapunov 指数数组（nats/step 或 1/s 均可，
                  只要单位一致）。

    Returns:
        D_KY: Kaplan–Yorke 维度（float）。若谱为空返回 0.0。
    """
    spectrum = np.asarray(spectrum, dtype=np.float64)
    spectrum = np.sort(spectrum)[::-1]   # ensure descending
    N = len(spectrum)
    if N == 0 or spectrum[0] < 0.0:
        return 0.0

    cumsum = np.cumsum(spectrum)
    # j = number of terms for which cumsum is still non-negative
    j = int(np.sum(cumsum >= 0.0))
    if j >= N:
        return float(N)          # conservative / Hamiltonian system
    if j == 0:
        return 0.0

    lam_next = float(spectrum[j])
    if abs(lam_next) < 1e-20:
        return float(j)

    return float(j) + float(cumsum[j - 1]) / abs(lam_next)


def lyapunov_spectrum_wolf(
    simulator,
    trajectory: np.ndarray,
    n_exponents: int = 10,
    renorm_steps: int = 50,
    epsilon: float = 1e-6,
    seed: int = 42,
) -> np.ndarray:
    """
    用 Gram-Schmidt 正交化的 Wolf-Benettin 方法计算前 k 个 Lyapunov 指数。

    原理
    ----
    维护 k 个正交扰动向量 Q[:, i]，每个重归一化周期：
      1. 对每个扰动向量 eᵢ 运行一次 Wolf 演化（使用当前轨迹状态作为
         基准），估计 J·eᵢ ≈ (x(t+τ|x₀+ε·eᵢ) − x(t+τ|x₀)) / ε_actual。
      2. 对更新后的扰动向量矩阵做 QR 分解（Gram-Schmidt 等价）。
      3. 累积 log|Rᵢᵢ|（对角元的对数 = 第 i 个方向的增长率）。
    最终：λᵢ = Σ log|Rᵢᵢ| / (n_valid_periods × renorm_steps)

    **上下文推进（Wolf context advancement）**：
    每个周期使用 ``simulator.wolf_rollout_pair`` 进行 base+pert 双轨演化。
    基准轨迹的历史上下文在每个周期结束后前进（通过 ``wolf_context``
    传递），使得后续周期能真正延续之前的历史，而非每次重置到 base_graph。

    若不推进上下文（直接调用 ``rollout()``），k 条扰动轨迹都共享同一
    base_graph 历史，所有方向的增长率由上下文稀释主导而非真实动力学，
    导致 λ₁ ≈ λ₂ ≈ ... ≈ λₖ > 0 ——这是一个已知的严重偏差。

    计算成本（每次 ``wolf_rollout_pair`` 调用 = 2 次 predict_future）：
      k × n_periods 次 wolf_rollout_pair 调用，每次 2 次推断。
      n_periods = (len(trajectory) − 1) // renorm_steps
      例：k=10, T=1000, renorm_steps=50 → 10 × 19 × 2 = 380 次推断。
      （与旧实现相比：旧实现是 (k+1) × 19 = 209 次，但存在上下文稀释偏差）

    Args:
        simulator:    ``BrainDynamicsSimulator`` 实例（TwinBrainDigitalTwin 模式）。
        trajectory:   预计算基准轨迹，shape (T, N)，用于提供每周期起点 x(t)。
        n_exponents:  要计算的指数数量 k（默认 10；最大 n_regions）。
        renorm_steps: 每个 Wolf 周期的步数（默认 50）。
        epsilon:      扰动幅度（默认 1e-6）。
        seed:         随机种子，用于初始正交扰动矩阵。

    Returns:
        spectrum: shape (k,)，降序排列的 Lyapunov 指数（nats/step）。
                  若可用周期数不足，返回全 NaN。
    """
    T, N = trajectory.shape
    k = min(n_exponents, N)
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))

    # Check if the simulator supports context-advancing Wolf rollout.
    # wolf_rollout_pair() advances the HeteroData context window across periods,
    # eliminating the context-dilution bias that plagues plain rollout().
    _use_wolf_pair = callable(getattr(simulator, "wolf_rollout_pair", None))

    rng = np.random.default_rng(seed)

    # Initialise k orthonormal perturbation vectors via QR decomposition
    Q_raw = rng.standard_normal((N, k)).astype(np.float64)
    Q, _ = np.linalg.qr(Q_raw)     # Q: (N, k), orthonormal columns

    log_sum = np.zeros(k, dtype=np.float64)
    n_periods = 0

    # wolf_context is advanced across periods (None → auto-init on first call)
    wolf_context = None

    t = 0
    while t + renorm_steps < T:
        x_base = trajectory[t].astype(np.float64)

        # ── Evolved perturbation vectors ──────────────────────────────────────
        evolved = np.zeros((N, k), dtype=np.float64)
        x_base_end: Optional[np.ndarray] = None
        next_wolf_context = None  # will be set from first wolf_rollout_pair call

        for i in range(k):
            x_pert_raw = (x_base + epsilon * Q[:, i]).astype(np.float32)
            if _bounds is not None:
                x_pert_raw = np.clip(x_pert_raw, _bounds[0], _bounds[1])
            eps_actual = float(np.linalg.norm(
                x_pert_raw.astype(np.float64) - x_base
            )) + 1e-30

            if _use_wolf_pair:
                # Correct path: use the SAME pre-advance wolf_context for all k
                # directions.  wolf_rollout_pair clones the context internally,
                # so repeated calls with the same wolf_context are safe.
                # next_wolf_context is the context advanced by the BASE rollout.
                # The `x_base_end is None` guard ensures we capture the base
                # endpoint and advanced context only from direction i=0.  For
                # directions i>1 the base endpoint is already known (x_base_end
                # is not None) so we skip the capture.  Only one context
                # advancement per period is needed; remaining ctx_out values
                # are discarded.
                end_base_i, end_pert_i, ctx_out = simulator.wolf_rollout_pair(
                    x_base=x_base.astype(np.float32),
                    x_pert=x_pert_raw,
                    steps=renorm_steps,
                    wolf_context=wolf_context,
                )
                if x_base_end is None:
                    # First direction: save base endpoint and advanced context
                    x_base_end = end_base_i.astype(np.float64)
                    next_wolf_context = ctx_out
                evolved[:, i] = (
                    end_pert_i.astype(np.float64) - x_base_end
                ) / eps_actual
            else:
                # Legacy fallback: plain rollout() (context resets every call).
                # This suffers from context-dilution but at least keeps the code
                # running on simulators that don't have wolf_rollout_pair.
                if x_base_end is None:
                    traj_base, _ = simulator.rollout(
                        x0=x_base.astype(np.float32), steps=renorm_steps
                    )
                    x_base_end = traj_base[-1].astype(np.float64)
                traj_pert, _ = simulator.rollout(
                    x0=x_pert_raw, steps=renorm_steps
                )
                x_pert_end = traj_pert[-1].astype(np.float64)
                evolved[:, i] = (x_pert_end - x_base_end) / eps_actual

        # Advance context for the next period
        if _use_wolf_pair and next_wolf_context is not None:
            wolf_context = next_wolf_context

        # Gram-Schmidt via thin QR: Q_new columns = normalised orthogonal basis
        Q_new, R = np.linalg.qr(evolved)

        # log|Rᵢᵢ| = log growth of i-th direction
        r_diag = np.abs(np.diag(R))
        r_diag = np.maximum(r_diag, 1e-30)
        log_sum += np.log(r_diag)

        Q = Q_new
        n_periods += 1
        t += renorm_steps

    if n_periods < 1:
        logger.warning("Lyapunov 谱：可用周期数不足（T=%d, renorm_steps=%d），返回 NaN。",
                       T, renorm_steps)
        return np.full(k, float("nan"))

    spectrum = log_sum / (n_periods * renorm_steps)   # nats / step, descending
    # Sort descending (QR preserves rough ordering but not guaranteed)
    spectrum = np.sort(spectrum)[::-1]

    logger.info(
        "Lyapunov 谱（Wolf-GS）: k=%d, periods=%d, "
        "λ₁=%.5f, λₖ=%.5f, D_KY=%.2f",
        k, n_periods, float(spectrum[0]), float(spectrum[-1]),
        kaplan_yorke_dimension(spectrum),
    )
    return spectrum


def run_lyapunov_spectrum_analysis(
    trajectories: np.ndarray,
    simulator,
    n_exponents: int = 10,
    renorm_steps: int = 50,
    epsilon: float = 1e-6,
    seed: int = 42,
    n_traj_sample: int = 5,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    在多条轨迹上估计 Lyapunov 谱并计算 Kaplan–Yorke 维度。

    从 ``trajectories`` 中随机抽取 ``n_traj_sample`` 条轨迹，对每条调用
    ``lyapunov_spectrum_wolf``，然后对谱取均值，计算 D_KY。

    计算成本提示：
      每条轨迹 k × (T // renorm_steps) 次额外模型调用。
      建议仅在 ``n_traj_sample=3–5`` 时使用，避免过长运行时间。

    Args:
        trajectories:    shape (n_traj, T, N)。
        simulator:       ``BrainDynamicsSimulator`` 实例。
        n_exponents:     要估计的 Lyapunov 指数数量（默认 10）。
        renorm_steps:    Wolf 周期步数（默认 50）。
        epsilon:         扰动幅度（默认 1e-6）。
        seed:            随机种子。
        n_traj_sample:   从 trajectories 中随机采样的轨迹数（默认 5）。
        output_dir:      结果保存目录；None → 不保存。

    Returns:
        dict 包含:
          mean_spectrum       : np.ndarray (k,)，多轨迹平均谱（降序）
          all_spectra         : np.ndarray (n_traj_sample, k)，各轨迹谱
          kaplan_yorke_dim    : float，D_KY（基于均值谱）
          lambda1             : float，最大 Lyapunov 指数
          n_positive          : int，λ > 0.01 的数量
          sum_positive        : float，正指数之和（KS 熵估计）
          classification      : str，动力学分类（与 classify_chaos_regime 一致）
    """
    n_traj, T, N = trajectories.shape
    k = min(n_exponents, N)
    rng = np.random.default_rng(seed)

    # Sample trajectories to compute spectrum for
    sample_idx = rng.choice(n_traj, size=min(n_traj_sample, n_traj), replace=False)
    all_spectra: List[np.ndarray] = []

    for i, idx in enumerate(sample_idx):
        logger.info("Lyapunov 谱分析: 轨迹 %d/%d (traj_idx=%d)",
                    i + 1, len(sample_idx), idx)
        try:
            spec = lyapunov_spectrum_wolf(
                simulator=simulator,
                trajectory=trajectories[idx],
                n_exponents=k,
                renorm_steps=renorm_steps,
                epsilon=epsilon,
                seed=seed + i,
            )
            if np.any(np.isfinite(spec)):
                all_spectra.append(spec)
        except Exception as exc:
            logger.warning("  轨迹 %d 谱估计失败: %s", idx, exc)

    if not all_spectra:
        raise RuntimeError("所有轨迹的 Lyapunov 谱估计均失败。")

    spectra_arr = np.vstack(all_spectra)             # (n_valid, k)
    # Average spectrum (NaN-safe)
    mean_spec = np.nanmean(spectra_arr, axis=0)      # (k,)
    mean_spec = np.sort(mean_spec)[::-1]             # ensure descending

    dky = kaplan_yorke_dimension(mean_spec)
    l1 = float(mean_spec[0])
    ks_entropy = float(mean_spec[mean_spec > 0].sum())
    n_pos = int((mean_spec > 0.01).sum())

    chaos_info = classify_chaos_regime(l1)
    classification = chaos_info["regime"]

    result: Dict = {
        "mean_spectrum": mean_spec,
        "all_spectra": spectra_arr,
        "kaplan_yorke_dim": round(dky, 4),
        "lambda1": round(l1, 6),
        "n_positive": n_pos,
        "sum_positive": round(ks_entropy, 6),
        "classification": classification,
        "n_traj_used": len(all_spectra),
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "lyapunov_spectrum.npy", mean_spec)
        np.save(out / "lyapunov_spectra_all.npy", spectra_arr)

        json_result = {k2: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k2, v in result.items()}
        with open(out / "lyapunov_spectrum_report.json", "w", encoding="utf-8") as fh:
            json.dump(json_result, fh, indent=2, ensure_ascii=False)

        logger.info("  → 保存 Lyapunov 谱: %s/lyapunov_spectrum_report.json", out)
        _try_plot_spectrum(mean_spec, spectra_arr, dky, out / "lyapunov_spectrum.png")

    return result


def _try_plot_spectrum(
    mean_spec: np.ndarray,
    all_spectra: np.ndarray,
    dky: float,
    output_path: Path,
) -> None:
    """绘制 Lyapunov 谱排名图（含逐轨迹谱 + 均值谱 + 累积和 + D_KY 标注）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    k = len(mean_spec)
    ranks = np.arange(1, k + 1)
    cumsum = np.cumsum(mean_spec)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: spectrum bars with per-trajectory shading
    ax = axes[0]
    # Per-trajectory lines (light, behind)
    for spec in all_spectra:
        spec_s = np.sort(spec)[::-1]
        ax.plot(ranks, spec_s, color="gray", alpha=0.25, linewidth=0.8)
    # Mean spectrum bars
    pos_mask = mean_spec > 0
    ax.bar(ranks[pos_mask], mean_spec[pos_mask], color="tomato", alpha=0.8,
           label="λ > 0 (expanding)")
    ax.bar(ranks[~pos_mask], mean_spec[~pos_mask], color="steelblue", alpha=0.8,
           label="λ ≤ 0 (contracting)")
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Rank  i")
    ax.set_ylabel("λᵢ  (nats / step)")
    ax.set_title(f"Lyapunov Spectrum  [k={k}]\n"
                 f"λ₁={float(mean_spec[0]):.4f}  D_KY={dky:.2f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # Right: cumulative sum with D_KY annotation
    ax2 = axes[1]
    ax2.plot(ranks, cumsum, "k-o", ms=3, lw=1.5)
    ax2.axhline(0, color="red", lw=0.8, ls="--", label="Σλ = 0")
    ax2.fill_between(ranks, cumsum, 0, where=(cumsum >= 0),
                     alpha=0.15, color="tomato")
    ax2.fill_between(ranks, cumsum, 0, where=(cumsum < 0),
                     alpha=0.15, color="steelblue")
    # Mark D_KY on x-axis
    if 0 < dky <= k:
        ax2.axvline(dky, color="purple", lw=1.2, ls=":",
                    label=f"D_KY = {dky:.2f}")
    ax2.set_xlabel("Rank  i")
    ax2.set_ylabel("Σᵢ₌₁ⁱ λ")
    ax2.set_title("Cumulative Lyapunov Sum\n"
                  "D_KY = where cumsum crosses zero")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存谱图: %s", output_path)
