"""
Energy Constraint Experiment
==============================

验证假设：**血氧代谢限制（有限能量通量）是大脑维持临界态的机制之一**。

**生理背景**
大脑皮层消耗全身约 20% 的能量（来自血氧合 ATP）。当血流不足时，
神经元自发放电受到抑制，大规模网络振荡减弱。本模块将此约束形式化为
α（能量通量系数）或动态 E(t)（代谢能量变量），并检验它是否能控制
系统的动力学状态（固定点 / 振荡 / 混沌）。

三个实验
--------

**实验 A：α 参数扫描（简化能量约束）**

  x(t+1) = α · F(x(t))     α ∈ [0.1, 2.5]

  其中 F 为 ``simulator.rollout``（TwinBrainDigitalTwin）的一步预测。
  α 解释为**能量通量系数**：
  - α < 1：能量受限（血氧不足），活动被抑制
  - α = 1：正常能量平衡（无约束基线）
  - α > 1：能量过剩（过度激活）

  # 计算成本（Experiment A，使用真实 GNN 模型）
  # ─────────────────────────────────────────────────────
  # n_alpha=20, n_init=5, steps=200:
  #   模型调用次数：20 × 5 × 200 = 20,000 次 rollout(steps=1)
  #   CPU 时间：~2-5 分钟（取决于模型大小）
  #   GPU 时间：~20-60 秒（RTX 3090 级别）
  #   内存：每次 rollout 约 20 MB VRAM → 峰值 ~100 MB VRAM（序列调用）
  #   磁盘：alpha_scan.json (~5 KB) + alpha_scan.png (~200 KB)
  # 快速模式（n_init=3, steps=50, n_alpha=10）：~2,000 calls，~30s GPU
  # ─────────────────────────────────────────────────────

**实验 B：三条件并排比较（PCA 可视化）**

  在 PCA 相空间并排展示三种能量状态：
    - 能量匮乏 (α ≈ 0.3)：轨迹收缩，动力学消亡
    - 正常/无约束 (α = 1.0)：中等振荡，近临界
    - 能量过剩 (α ≈ 1.8)：振荡扩大或混沌

  # 计算成本（Experiment B）
  # ─────────────────────────────────────────────────────
  # 3 × n_traj × steps = 3 × 10 × 150 = 4,500 次 rollout(steps=1)
  # GPU 时间：~10-30 秒
  # ─────────────────────────────────────────────────────

**实验 C：动态能量变量 E(t)**

  E(t+1) = clip(E(t) + α − β · mean(|x(t)|), 0, 10)
  x(t+1) = clip(F(x(t)) · g(E(t)), bounds)

  能量门控函数（修正版）：
    g(E) = 2 · sigmoid(4 · (E − E_ref))          ← g(E_ref)=1.0

  关键性质：
    g(E_ref) = 1.0  → 参考能量时无约束（与 α=1 等价）
    g(0)    ≈ 0.04  → 能量耗尽时活动几乎为零
    g → 2  as E→∞   → 能量过剩时活动约翻倍

  注意：早期版本 g(E) = sigmoid(4·(E-E_ref)) 导致 g(E_ref)=0.5（参考状态
  活动已被抑制 50%，不合生理逻辑）。现已修正。

  # 计算成本（Experiment C）
  # ─────────────────────────────────────────────────────
  # n_init=5, steps=300：
  #   模型调用次数：5 × 300 = 1,500 次 rollout(steps=1)
  #   GPU 时间：~5-15 秒
  # ─────────────────────────────────────────────────────

输出文件
--------
  energy_constraint_alpha_scan.json  — 实验 A 各 α 值指标
  energy_constraint_alpha_scan.png   — 含动力学区域标注的分岔图
  energy_comparison.png              — 实验 B 三条件并排 PCA 图
  energy_dynamic_E.json              — 实验 C E(t) 轨迹统计
  energy_dynamic_E.png               — E(t) + g(E) + 活动时序图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rosenstein_simple(traj: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """
    轻量版 Rosenstein LLE（直接用于能量约束扫描，避免循环依赖）。
    优先调用 twinbrain-dynamics 的标准实现。
    """
    import sys
    from pathlib import Path as _Path
    _td = _Path(__file__).parent
    if str(_td) not in sys.path:
        sys.path.insert(0, str(_td))
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        lle, _ = rosenstein_lyapunov(traj, max_lag=max_lag, min_temporal_sep=min_sep)
        return float(lle)
    except Exception:
        pass

    # Fallback: first-PC Rosenstein
    T, N = traj.shape
    ts_c = traj - traj.mean(axis=0)
    _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
    proj = ts_c @ Vt[0]
    max_lag = min(max_lag, T // 4)
    min_sep = min(min_sep, (T - max_lag - 1) // 2)
    if min_sep < 2 or max_lag < 5:
        return float("nan")
    D2 = (proj[:, None] - proj[None, :]) ** 2
    for k in range(min(min_sep, T)):
        for sign in (-1, 1):
            idx = np.arange(T) + sign * k
            valid = (idx >= 0) & (idx < T)
            D2[np.arange(T)[valid], idx[valid]] = np.inf
    nn = np.argmin(D2, axis=1)
    div = np.zeros(max_lag)
    cnt = np.zeros(max_lag, dtype=int)
    for t in range(T - max_lag):
        d0 = np.sqrt(D2[t, nn[t]]) + 1e-20
        for lag in range(1, max_lag + 1):
            if t + lag >= T or nn[t] + lag >= T:
                break
            div[lag - 1] += np.log(abs(proj[t + lag] - proj[nn[t] + lag]) / d0 + 1e-20)
            cnt[lag - 1] += 1
    m = cnt > 3
    if not m.any():
        return float("nan")
    div[m] /= cnt[m]
    lags = np.where(m)[0][:min(10, m.sum())]
    slope, _ = np.polyfit(lags, div[lags], 1)
    return float(slope)


def _run_alpha_trajectory(
    simulator,
    alpha: float,
    n_init: int,
    steps: int,
    seed: int,
    _bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    """
    生成 α 约束下的 n_init 条轨迹。

    每步：x(t+1) = clip(α · F(x(t)), bounds)

    Args:
        simulator: BrainDynamicsSimulator 实例。
        alpha:     能量系数。
        n_init:    轨迹数。
        steps:     每条轨迹步数。
        seed:      随机种子。
        _bounds:   状态空间边界（None → 无 clip）。

    Returns:
        trajs: shape (n_init, steps, N)。
    """
    rng = np.random.default_rng(seed)
    N = simulator.n_regions

    # Simulate: get initial chunk from simulator, then apply α scaling at each step
    # Implementation: run step-by-step via single-step rollout + scale.
    # For efficiency, run the full trajectory once then scale each frame.
    # NOTE: α·F(x(t)) is NOT the same as running the model with rescaled inputs—
    #       it rescales the *output* (activity), modelling an energy gate on neural output.

    trajs = np.empty((n_init, steps, N), dtype=np.float32)
    for i in range(n_init):
        x0 = simulator.sample_random_state(rng=rng)
        x = x0.astype(np.float32)
        for t in range(steps):
            trajs[i, t] = x
            # One-step prediction
            traj_1, _ = simulator.rollout(x0=x, steps=1)
            x_next = (alpha * traj_1[0]).astype(np.float32)
            if _bounds is not None:
                x_next = np.clip(x_next, _bounds[0], _bounds[1])
            x = x_next
    return trajs


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: α scan
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_constraint_scan(
    simulator,
    alpha_min: float = 0.1,
    alpha_max: float = 2.5,
    alpha_step: float = 0.1,
    n_init: int = 5,
    steps: int = 200,
    warmup: int = 50,
    rosenstein_max_lag: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    扫描能量系数 α，绘制"能量约束分岔图"（使用真实 GNN 模型）。

    x(t+1) = clip(α · F(x(t)), bounds)    F = TwinBrainDigitalTwin 一步预测

    α 的三个动力学区间：
      α·ρ(W_eff) < ~0.9 → 固定点（系统衰减，活动消亡）
      α·ρ(W_eff) ≈ 1.0  → 近临界振荡（系统最优计算区）
      α·ρ(W_eff) > ~1.5 → 混沌（预测能力下降）

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────
    # 参数：n_alpha = ceil((α_max-α_min)/step) ≈ 25，n_init=5，steps=200
    #   模型调用次数：n_alpha × n_init × steps = 25 × 5 × 200 = 25,000
    #   CPU 时间（小模型）：~3-8 分钟
    #   GPU 时间（RTX 3090）：~30-90 秒
    #   内存：每 α 值峰值 ~n_init × N × 4B × batch ≈ 5 × 190 × 4 ≈ 4 KB（序列调用）
    # 快速模式（n_init=3, steps=50, n_alpha=10）：1,500 calls, ~10s GPU
    # ─────────────────────────────────────────────────────────────────────

    Args:
        simulator:          BrainDynamicsSimulator 实例。
        alpha_min/max/step: α 的扫描范围和步长（默认 0.1–2.5，步长 0.1）。
        n_init:             每个 α 值的轨迹数（默认 5，快速；20 更稳健）。
        steps:              每条轨迹热身后的记录步数（默认 200）。
        warmup:             热身步数（默认 50，不计入指标）。
        rosenstein_max_lag: Rosenstein LLE 的最大追踪滞后（默认 30）。
        seed:               随机种子。
        output_dir:         结果保存目录；None → 不保存。

    Returns:
        dict 包含 alpha_values, lles, osc_amplitudes, mean_activities,
                  critical_alpha, bifurcation_found。
    """
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))

    alpha_vals = np.round(
        np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step), 3
    )

    lles: List[float] = []
    osc_amps: List[float] = []
    mean_acts: List[float] = []

    logger.info(
        "能量约束 α 扫描: α ∈ [%.1f, %.1f], step=%.2f, n_init=%d, steps=%d",
        alpha_min, alpha_max, alpha_step, n_init, steps,
    )

    for alpha in alpha_vals:
        trajs = _run_alpha_trajectory(
            simulator, alpha, n_init, steps + warmup, seed, _bounds
        )
        # Discard warmup
        trajs_rec = trajs[:, warmup:, :]    # (n_init, steps, N)
        T_rec = trajs_rec.shape[1]

        # LLE (Rosenstein on each trajectory's first-PC)
        traj_lles = []
        for i in range(n_init):
            lle = _rosenstein_simple(
                trajs_rec[i], max_lag=rosenstein_max_lag,
                min_sep=max(5, T_rec // 10),
            )
            if np.isfinite(lle):
                traj_lles.append(lle)
        lle_med = float(np.median(traj_lles)) if traj_lles else float("nan")

        osc = float(trajs_rec.std(axis=1).mean())
        act = float(trajs_rec.mean())

        lles.append(lle_med)
        osc_amps.append(osc)
        mean_acts.append(act)

        logger.debug("  α=%.2f: LLE=%.4f, osc=%.4f, mean_act=%.4f",
                     alpha, lle_med, osc, act)

    lle_arr = np.array(lles, dtype=float)
    valid = np.isfinite(lle_arr)
    if valid.any():
        crit_idx = int(np.argmin(np.abs(lle_arr[valid])))
        critical_alpha = float(alpha_vals[valid][crit_idx])
        # Bifurcation found if there is a clear sign change in LLE
        bifurcation_found = bool(
            (lle_arr[valid] < -0.01).any() and (lle_arr[valid] > 0.01).any()
        )
    else:
        critical_alpha = float("nan")
        bifurcation_found = False

    result = {
        "alpha_values": alpha_vals.tolist(),
        "lles": lles,
        "osc_amplitudes": osc_amps,
        "mean_activities": mean_acts,
        "critical_alpha": round(critical_alpha, 3),
        "bifurcation_found": bifurcation_found,
        "n_init": n_init,
        "steps_per_traj": steps,
    }

    logger.info(
        "能量约束扫描结果: critical_α=%.2f, bifurcation=%s",
        critical_alpha, bifurcation_found,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "energy_constraint_alpha_scan.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存 α 扫描结果: %s", json_path)
        _try_plot_alpha_scan(result, out / "energy_constraint_alpha_scan.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: dynamic energy variable E(t)
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamic_energy_experiment(
    simulator,
    alpha: float = 0.5,
    beta: float = 0.5,
    E_ref: float = 1.0,
    E_init: float = 1.0,
    steps: int = 300,
    n_init: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    运行动态能量变量实验（实验 C）。

    方程（修正版）：
      E(t+1) = clip(E(t) + α − β · mean(|x(t)|), 0, 10)
      x(t+1) = clip(F(x(t)) · g(E(t)), bounds)
      g(E)   = 2 · sigmoid(4 · (E − E_ref))    ← 修正：g(E_ref)=1.0

    修复说明：
      早期版本使用 g(E) = sigmoid(4·(E-E_ref))，导致 g(E_ref)=0.5，
      即正常能量下活动被压缩 50%，不符合"E_ref 为无约束工作点"的设计意图。
      现已修正为 g(E_ref)=1.0，与 α=1 基线完全等价。

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────
    # n_init=5, steps=300:
    #   模型调用次数：5 × 300 = 1,500 次 rollout(steps=1)
    #   GPU 时间（RTX 3090）：~5-15 秒
    #   内存：序列调用，峰值 < 50 MB VRAM
    # ─────────────────────────────────────────────────────────────────────

    Args:
        simulator: BrainDynamicsSimulator 实例。
        alpha:     能量供应速率（单位/步；对应血流速率，默认 0.5）。
        beta:      活动消耗系数（默认 0.5）。
        E_ref:     能量参考值（g=1.0 时的 E 值，对应正常血氧饱和度，默认 1.0）。
        E_init:    初始能量（默认 1.0 = E_ref → 从正常状态出发）。
        steps:     模拟步数（默认 300；建议 ≥200 以观察稳态）。
        n_init:    初始状态数量（默认 5）。
        seed:      随机种子。
        output_dir: 结果保存目录；None → 不保存。

    Returns:
        dict 包含 E_mean, E_std, activity_mean, homeostasis_achieved,
                  alpha, beta, E_ref。
    """
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))
    rng = np.random.default_rng(seed)
    N = simulator.n_regions

    E_trajs = np.empty((n_init, steps), dtype=np.float32)
    act_trajs = np.empty((n_init, steps), dtype=np.float32)
    g_trajs = np.empty((n_init, steps), dtype=np.float32)

    for i in range(n_init):
        x0 = simulator.sample_random_state(rng=rng)
        x = x0.astype(np.float32)
        E = float(E_init)

        for t in range(steps):
            E_trajs[i, t] = E
            act_trajs[i, t] = float(np.mean(np.abs(x)))

            # Energy gate: g(E_ref) = 1.0 (corrected formula)
            g_E = 2.0 / (1.0 + np.exp(-4.0 * (E - E_ref)))
            g_trajs[i, t] = g_E

            # One-step prediction F(x)
            traj_1, _ = simulator.rollout(x0=x, steps=1)
            F_x = traj_1[0].astype(np.float64)

            # Update state with energy gate
            x_next = (g_E * F_x).astype(np.float32)
            if _bounds is not None:
                x_next = np.clip(x_next, _bounds[0], _bounds[1])

            # Update energy (use pre-update activity for consumption)
            consumption = beta * float(np.mean(np.abs(x)))
            E = float(np.clip(E + alpha - consumption, 0.0, 10.0))
            x = x_next

    # Analyse steady state (last 100 steps or last half)
    ss_start = max(0, steps - 100)
    E_ss = E_trajs[:, ss_start:]
    act_ss = act_trajs[:, ss_start:]

    E_mean = float(E_ss.mean())
    E_std = float(E_ss.std())
    act_mean = float(act_ss.mean())

    homeostasis = abs(E_mean - E_ref) < 0.3 * E_ref

    logger.info(
        "动态能量实验: α=%.2f, β=%.2f, E_ref=%.2f → "
        "E_稳态=%.4f±%.4f, 活动均值=%.4f, 稳态=%s",
        alpha, beta, E_ref, E_mean, E_std, act_mean, homeostasis,
    )

    result = {
        "E_mean": round(E_mean, 6),
        "E_std": round(E_std, 6),
        "activity_mean": round(act_mean, 6),
        "homeostasis_achieved": homeostasis,
        "alpha": alpha,
        "beta": beta,
        "E_ref": E_ref,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "energy_dynamic_E.npy", E_trajs)
        np.save(out / "energy_dynamic_activity.npy", act_trajs)

        json_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in result.items()}
        json_path = out / "energy_dynamic_E.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_result, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存动态能量结果: %s", json_path)

        _try_plot_dynamic_energy(E_trajs, act_trajs, g_trajs,
                                 alpha, beta, E_ref,
                                 out / "energy_dynamic_E.png")

    return result
        output_dir: 结果保存目录；None → 不保存。

    Returns:
        dict 包含:
          E_trajectories  : np.ndarray (n_init, steps)，能量时序
          x_mean_act      : np.ndarray (n_init, steps)，平均活动时序
          E_mean          : float，稳态均值 E
          E_std           : float，稳态标准差（振荡指示）
          activity_mean   : float，稳态均值活动
          homeostasis_achieved: bool，E 是否稳定在 E_ref 附近
    """
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))
    rng = np.random.default_rng(seed)

    E_trajs = np.empty((n_init, steps), dtype=np.float32)
    act_trajs = np.empty((n_init, steps), dtype=np.float32)
    g_trajs = np.empty((n_init, steps), dtype=np.float32)

    for i in range(n_init):
        x0 = simulator.sample_random_state(rng=rng)
        x = x0.astype(np.float32)
        E = float(E_init)

        for t in range(steps):
            E_trajs[i, t] = E
            act_trajs[i, t] = float(np.mean(np.abs(x)))

            # Energy gate (corrected): g(E_ref) = 1.0
            g_E = 2.0 / (1.0 + np.exp(-4.0 * (E - E_ref)))
            g_trajs[i, t] = g_E

            # One-step prediction F(x)
            traj_1, _ = simulator.rollout(x0=x, steps=1)
            F_x = traj_1[0].astype(np.float64)

            # Update state with energy gate
            x_next = (g_E * F_x).astype(np.float32)
            if _bounds is not None:
                x_next = np.clip(x_next, _bounds[0], _bounds[1])

            # Update energy (consumption from pre-update activity)
            consumption = beta * float(np.mean(np.abs(x)))
            E = float(np.clip(E + alpha - consumption, 0.0, 10.0))
            x = x_next

    # Analyse steady state (last 100 steps or last half)
    ss_start = max(0, steps - 100)
    E_ss = E_trajs[:, ss_start:]
    act_ss = act_trajs[:, ss_start:]

    E_mean = float(E_ss.mean())
    E_std = float(E_ss.std())
    act_mean = float(act_ss.mean())

    homeostasis = abs(E_mean - E_ref) < 0.3 * E_ref

    logger.info(
        "动态能量实验: α=%.2f, β=%.2f, E_ref=%.2f → "
        "E_稳态=%.4f±%.4f, 活动均值=%.4f, 稳态=%s",
        alpha, beta, E_ref, E_mean, E_std, act_mean, homeostasis,
    )

    result = {
        "E_mean": round(E_mean, 6),
        "E_std": round(E_std, 6),
        "activity_mean": round(act_mean, 6),
        "homeostasis_achieved": homeostasis,
        "alpha": alpha,
        "beta": beta,
        "E_ref": E_ref,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        np.save(out / "energy_dynamic_E.npy", E_trajs)
        np.save(out / "energy_dynamic_activity.npy", act_trajs)

        json_result = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in result.items()}
        json_path = out / "energy_dynamic_E.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_result, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存动态能量结果: %s", json_path)

        _try_plot_dynamic_energy(E_trajs, act_trajs, g_trajs,
                                 alpha, beta, E_ref,
                                 out / "energy_dynamic_E.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: three-condition comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_comparison(
    simulator,
    alpha_low: float = 0.3,
    alpha_high: float = 1.8,
    n_traj: int = 10,
    steps: int = 150,
    warmup: int = 20,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    三条件并排比较（使用真实 GNN 模型）：
      - 能量匮乏  (α = alpha_low  ≈ 0.3)：轨迹收缩→固定点
      - 正常无约束 (α = 1.0)：中等振荡，近临界
      - 能量过剩  (α = alpha_high ≈ 1.8)：振荡扩大或混沌

    直接在 PCA 相空间中并排可视化三种状态，最直观地支持
    "能量通量是动力学状态控制参数"的假设。

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────
    # 3 × n_traj × (warmup+steps) = 3 × 10 × 170 = 5,100 次 rollout(steps=1)
    # GPU 时间（RTX 3090）：~15-30 秒
    # 内存：< 100 MB VRAM（序列调用）
    # ─────────────────────────────────────────────────────────────────────

    Args:
        simulator:  BrainDynamicsSimulator 实例。
        alpha_low:  能量匮乏系数（默认 0.3）。
        alpha_high: 能量过剩系数（默认 1.8）。
        n_traj:     每条件的轨迹数（默认 10）。
        steps:      每条轨迹步数（热身后，默认 150）。
        warmup:     热身步数（默认 20）。
        seed:       随机种子。
        output_dir: 保存目录（None → 不保存）。

    Returns:
        dict 包含 conditions (各条件的 lle/osc/mean_activity)。
    """
    _bounds = getattr(simulator, "state_bounds", (0.0, 1.0))
    rng = np.random.default_rng(seed)

    conditions = {"low": alpha_low, "normal": 1.0, "high": alpha_high}
    x0s = [simulator.sample_random_state(rng=rng) for _ in range(n_traj)]

    all_trajs: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict] = {}

    for cond_name, alpha in conditions.items():
        trajs_list = []
        for x0 in x0s:
            traj = _run_alpha_trajectory(
                simulator, alpha, 1, steps + warmup,
                int(rng.integers(0, 2**31)), _bounds,
            )
            trajs_list.append(traj[0, warmup:])  # (steps, N)
        trajs = np.stack(trajs_list, axis=0)  # (n_traj, steps, N)
        all_trajs[cond_name] = trajs

        traj_lles = [
            _rosenstein_simple(trajs[i], max_lag=20, min_sep=max(5, steps // 10))
            for i in range(n_traj)
        ]
        valid = [v for v in traj_lles if np.isfinite(v)]
        lle = float(np.median(valid)) if valid else float("nan")
        stats[cond_name] = {
            "alpha": alpha,
            "lle": round(lle, 5),
            "osc_amplitude": round(float(trajs.std(axis=1).mean()), 5),
            "mean_activity": round(float(trajs.mean()), 5),
        }

    result = {
        "conditions": stats,
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
    }
    logger.info(
        "三条件比较: low_LLE=%.4f, normal_LLE=%.4f, high_LLE=%.4f",
        stats["low"]["lle"], stats["normal"]["lle"], stats["high"]["lle"],
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / "energy_comparison.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot_comparison(all_trajs, stats, alpha_low, alpha_high,
                             out / "energy_comparison.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_alpha_scan(result: Dict, output_path: Path) -> None:
    """
    绘制能量约束分岔图（LLE + 振荡幅度 + 均值活动），
    含三个动力学区域的彩色背景标注。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    alphas = np.array(result["alpha_values"])
    lles = np.array(result["lles"], dtype=float)
    osc = np.array(result["osc_amplitudes"])
    acts = np.array(result["mean_activities"])
    crit = result["critical_alpha"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    a_lo, a_hi = float(alphas.min()), float(alphas.max())

    # Regime backgrounds
    if np.isfinite(lles).any():
        lle_f = np.where(np.isfinite(lles), lles, 0.0)
        step = float(np.median(np.diff(alphas))) if len(alphas) > 1 else 0.1
        for mask, color, name in [
            (lle_f < -0.05,  "#6baed6", "固定点区"),
            ((lle_f >= -0.05) & (lle_f < 0.05), "#74c476", "近临界区"),
            (lle_f >= 0.05,  "#fb6a4a", "混沌区"),
        ]:
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            lo = float(alphas[idx[0]]) - step / 2
            hi = float(alphas[idx[-1]]) + step / 2
            for ax in axes:
                ax.axvspan(lo, hi, alpha=0.12, color=color,
                           label=name if ax is axes[0] else "")

    # LLE
    ax = axes[0]
    ax.plot(alphas, lles, "b-o", ms=4, lw=1.8)
    ax.axhline(0, ls="--", color="red", lw=1.2, label="LLE=0 (边缘混沌)")
    if np.isfinite(crit):
        ax.axvline(crit, ls=":", color="orange", lw=1.8,
                   label=f"临界 α*={crit:.2f}")
    ax.set_ylabel("Lyapunov LLE", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title(
        "能量约束分岔图  [GNN 模型]\n"
        "x(t+1) = α·F(x(t))  |  F = TwinBrainDigitalTwin",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)

    # Oscillation amplitude
    axes[1].plot(alphas, osc, "-o", color="forestgreen", ms=4, lw=1.8)
    if np.isfinite(crit):
        axes[1].axvline(crit, ls=":", color="orange", lw=1.8)
    axes[1].set_ylabel("振荡幅度 (std)", fontsize=9)
    axes[1].grid(True, alpha=0.3)

    # Mean activity
    axes[2].plot(alphas, acts, "-o", color="dimgray", ms=4, lw=1.8)
    axes[2].axhline(0.5, ls="--", color="gray", lw=0.9, label="mean=0.5")
    if np.isfinite(crit):
        axes[2].axvline(crit, ls=":", color="orange", lw=1.8)
    axes[2].set_ylabel("均值活动水平", fontsize=9)
    axes[2].set_xlabel("能量系数 α", fontsize=9)
    axes[2].legend(fontsize=7)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存分岔图: %s", output_path)


def _try_plot_comparison(
    all_trajs: Dict[str, np.ndarray],
    stats: Dict[str, Dict],
    alpha_low: float,
    alpha_high: float,
    output_path: Path,
) -> None:
    """三条件并排 PCA 比较图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
    except ImportError:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            # Minimal PCA fallback
            class PCA:
                def __init__(self, n_components=2):
                    self._V = None
                def fit_transform(self, X):
                    Xc = X - X.mean(0)
                    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
                    self._V = Vt[:2]
                    return Xc @ Vt[:2].T
                def transform(self, X):
                    return (X - X.mean(0)) @ self._V.T
        except ImportError:
            return

    normal_trajs = all_trajs["normal"]
    X_fit = normal_trajs.reshape(-1, normal_trajs.shape[2])
    pca = PCA(n_components=2)
    pca.fit_transform(X_fit)

    cond_order  = ["low", "normal", "high"]
    cond_colors = {"low": "#2166ac", "normal": "#1a9850", "high": "#d73027"}
    cond_short  = {
        "low":    f"能量匮乏\nα={alpha_low}",
        "normal": "正常无约束\nα=1.0",
        "high":   f"能量过剩\nα={alpha_high}",
    }
    t_axis = np.linspace(0, 1, normal_trajs.shape[1])
    cmap_time = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(14, 8))
    axes_row1 = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
    ax_overlay = fig.add_subplot(2, 3, 4)
    ax_osc     = fig.add_subplot(2, 3, 5)
    ax_act     = fig.add_subplot(2, 3, 6)

    for ax, cond in zip(axes_row1, cond_order):
        trajs = all_trajs[cond]
        n = trajs.shape[0]
        T_rec = trajs.shape[1]
        for i in range(n):
            X_proj = pca.transform(trajs[i])
            for k in range(T_rec - 1):
                ax.plot(X_proj[k:k+2, 0], X_proj[k:k+2, 1],
                        color=cmap_time(t_axis[k]), lw=0.6, alpha=0.65)
        ax.set_title(
            f"{cond_short[cond]}\nLLE={stats[cond]['lle']:.4f}", fontsize=8
        )
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    for cond in cond_order:
        trajs = all_trajs[cond]
        mid_i = trajs.shape[0] // 2
        X_proj = pca.transform(trajs[mid_i])
        ax_overlay.plot(X_proj[:, 0], X_proj[:, 1],
                        color=cond_colors[cond], lw=1.3, alpha=0.9,
                        label=cond_short[cond].replace("\n", " "))
        ax_overlay.plot(*X_proj[0, :2], "o", color=cond_colors[cond], ms=5)
        ax_overlay.plot(*X_proj[-1, :2], "*", color=cond_colors[cond], ms=7)
    ax_overlay.set_title("三条件叠加 PCA", fontsize=8)
    ax_overlay.set_xlabel("PC1", fontsize=7)
    ax_overlay.set_ylabel("PC2", fontsize=7)
    ax_overlay.legend(fontsize=6)
    ax_overlay.tick_params(labelsize=6)
    ax_overlay.grid(True, alpha=0.2)

    x_pos = np.arange(3)
    conds = cond_order
    colors = [cond_colors[c] for c in conds]
    short = [cond_short[c] for c in conds]
    ax_osc.bar(x_pos, [stats[c]["osc_amplitude"] for c in conds], color=colors)
    ax_osc.set_xticks(x_pos)
    ax_osc.set_xticklabels(short, fontsize=7)
    ax_osc.set_ylabel("振荡幅度 (std)", fontsize=8)
    ax_osc.set_title("振荡幅度对比", fontsize=8)
    ax_osc.grid(True, alpha=0.3, axis="y")
    ax_act.bar(x_pos, [stats[c]["mean_activity"] for c in conds], color=colors)
    ax_act.set_xticks(x_pos)
    ax_act.set_xticklabels(short, fontsize=7)
    ax_act.set_ylabel("均值活动", fontsize=8)
    ax_act.set_title("均值活动对比", fontsize=8)
    ax_act.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        "能量约束三条件比较  [GNN 模型]\n"
        "蓝=初期 → 红=末期  |  上行：PCA 相空间  下行：定量指标",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存三条件比较图: %s", output_path)


def _try_plot_dynamic_energy(
    E_trajs: np.ndarray,
    act_trajs: np.ndarray,
    g_trajs: np.ndarray,
    alpha: float,
    beta: float,
    E_ref: float,
    output_path: Path,
) -> None:
    """绘制动态能量 E(t)、能量门控 g(E) 和活动均值时序图（3 行）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = E_trajs.shape[1]
    t_axis = np.arange(steps)
    n = E_trajs.shape[0]
    cmap = plt.get_cmap("coolwarm", max(n, 2))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # E(t)
    for i in range(n):
        axes[0].plot(t_axis, E_trajs[i], color=cmap(i / max(n - 1, 1)),
                     alpha=0.7, lw=1.0)
    axes[0].axhline(E_ref, ls="--", color="black", lw=1.2,
                    label=f"E_ref={E_ref:.1f} (g=1.0 工作点)")
    axes[0].set_ylabel("能量 E(t)", fontsize=9)
    axes[0].set_title(
        f"动态能量变量实验  [GNN 模型]  α={alpha}, β={beta}\n"
        f"g(E)=2·sigmoid(4·(E−E_ref)),  g(E_ref)=1.0（无约束基线）",
        fontsize=9,
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # g(E(t))
    for i in range(n):
        axes[1].plot(t_axis, g_trajs[i], color=cmap(i / max(n - 1, 1)),
                     alpha=0.7, lw=1.0)
    axes[1].axhline(1.0, ls="--", color="black", lw=1.0,
                    label="g=1.0（无约束）")
    axes[1].axhline(0.5, ls=":", color="gray", lw=0.8, label="g=0.5（50% 抑制）")
    axes[1].set_ylabel("能量门控 g(E)", fontsize=9)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    # Activity
    for i in range(n):
        axes[2].plot(t_axis, act_trajs[i], color=cmap(i / max(n - 1, 1)),
                     alpha=0.7, lw=1.0)
    axes[2].set_ylabel("均值 |x(t)|", fontsize=9)
    axes[2].set_xlabel("步骤", fontsize=9)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存动态能量图: %s", output_path)
