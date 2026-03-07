"""
Energy Constraint Experiment
==============================

验证假设：**血氧代谢限制（有限能量通量）是大脑维持临界态的机制之一**。

两个实验
--------

**实验 A：α 参数扫描（简化能量约束）**

  x(t+1) = α · F(x(t))     α ∈ [0.5, 1.5]

  其中 F 为 ``simulator.rollout``（TwinBrainDigitalTwin 或 WC）的一步预测。
  α 解释为**能量通量系数**：
  - α < 1：能量受限（血氧不足），活动被抑制
  - α = 1：正常能量平衡
  - α > 1：能量过剩（癫痫？过度激活）

  扫描指标：
  - Rosenstein LLE：区分固定点 / 振荡 / 混沌
  - 振荡幅度 std(x)：反映动力学活跃度
  - 均值活动水平 mean(x)：是否漂移

  预期结果（若假设成立）：
  - α 小 → 固定点（系统衰减）
  - α 中等（≈1）→ 振荡（limit cycle，近临界）
  - α 大 → 混沌

**实验 B：动态能量变量 E(t)**

  E(t+1) = E(t) + α − β · mean(|x(t)|)          能量平衡方程
  x(t+1) = F(x(t)) · g(E(t))                     能量调制活动

  其中 g(E) = sigmoid(4 · (E − E_ref))            软饱和控制函数

  生理解释：
  - E(t)：可用代谢能量（BOLD 信号代理）
  - α：基础能量供应速率（血流）
  - β·mean(|x|)：神经活动的代谢消耗
  - g(E)：当能量低时抑制活动，高时放大活动

  预期结果：
  - 系统在 E ≈ E_ref 附近自稳定（homeostatic feedback）
  - 若 α/β 比值适中 → 系统自组织到边缘混沌

输出文件
--------
  energy_constraint_alpha_scan.json  — 实验 A 各 α 值指标
  energy_constraint_alpha_scan.png   — 双子图相图
  energy_dynamic_E.json              — 实验 B E(t) 轨迹统计
  energy_dynamic_E.png               — E(t) 时序 + 活动轨迹图

**注意**：实验 A 通过对 ``simulator.rollout`` 输出直接乘以 α 来实现；
这意味着每个 α 值均需完整运行轨迹，计算量为 n_alpha × n_init 次 rollout。
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
    alpha_min: float = 0.5,
    alpha_max: float = 1.5,
    alpha_step: float = 0.1,
    n_init: int = 10,
    steps: int = 200,
    warmup: int = 50,
    rosenstein_max_lag: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    扫描能量系数 α，绘制"能量约束分岔图"。

    Args:
        simulator:          BrainDynamicsSimulator 实例。
        alpha_min/max/step: α 的扫描范围和步长。
        n_init:             每个 α 值的轨迹数（默认 10）。
        steps:              每条轨迹总步数（含 warmup）。
        warmup:             热身步数（不计入指标）。
        rosenstein_max_lag: Rosenstein LLE 的最大追踪滞后。
        seed:               随机种子。
        output_dir:         结果保存目录；None → 不保存。

    Returns:
        dict 包含:
          alpha_values      : List[float]
          lles              : List[float]，各 α 的中位数 LLE
          osc_amplitudes    : List[float]，振荡幅度（std）
          mean_activities   : List[float]，均值活动水平
          critical_alpha    : float，LLE 最接近 0 的 α 值
          bifurcation_found : bool
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
            simulator, alpha, n_init, steps, seed, _bounds
        )
        # Discard warmup
        trajs_rec = trajs[:, warmup:, :]    # (n_init, steps-warmup, N)
        T_rec = trajs_rec.shape[1]

        # LLE (Rosenstein on each trajectory's first-PC)
        traj_lles = []
        for i in range(n_init):
            lle = _rosenstein_simple(
                trajs_rec[i], max_lag=rosenstein_max_lag, min_sep=max(5, T_rec // 10)
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
    运行动态能量变量实验（实验 B）。

    方程：
      E(t+1) = clip(E(t) + α − β · mean(|x(t)|), 0, 10)
      x(t+1) = clip(F(x(t)) · g(E(t)), bounds)
      g(E)   = sigmoid(4 · (E − E_ref))    ∈ (0, 1)

    Args:
        simulator: BrainDynamicsSimulator 实例。
        alpha:     能量供应速率（默认 0.5）。
        beta:      活动消耗系数（默认 0.5）。
        E_ref:     能量参考值（g=0.5 时的 E 值，默认 1.0）。
        E_init:    初始能量（默认 1.0 = E_ref）。
        steps:     模拟步数（默认 300）。
        n_init:    初始状态数量（默认 5）。
        seed:      随机种子。
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
    N = simulator.n_regions

    E_trajs = np.empty((n_init, steps), dtype=np.float32)
    act_trajs = np.empty((n_init, steps), dtype=np.float32)

    for i in range(n_init):
        x0 = simulator.sample_random_state(rng=rng)
        x = x0.astype(np.float32)
        E = float(E_init)

        for t in range(steps):
            E_trajs[i, t] = E
            act_trajs[i, t] = float(np.mean(np.abs(x)))

            # One-step prediction F(x)
            traj_1, _ = simulator.rollout(x0=x, steps=1)
            F_x = traj_1[0].astype(np.float64)

            # Energy gate
            g_E = 1.0 / (1.0 + np.exp(-4.0 * (E - E_ref)))

            # Update state
            x_next = (g_E * F_x).astype(np.float32)
            if _bounds is not None:
                x_next = np.clip(x_next, _bounds[0], _bounds[1])

            # Update energy
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
        "E_trajectories": E_trajs,
        "x_mean_act": act_trajs,
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

        _try_plot_dynamic_energy(E_trajs, act_trajs, alpha, beta, E_ref,
                                 out / "energy_dynamic_E.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_alpha_scan(result: Dict, output_path: Path) -> None:
    """绘制 α 扫描分岔图（LLE + 振荡幅度 + 均值活动）。"""
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

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    # LLE
    ax = axes[0]
    ax.plot(alphas, lles, "b-o", ms=4, lw=1.5)
    ax.axhline(0, ls="--", color="red", lw=1, label="LLE=0 (edge of chaos)")
    if np.isfinite(crit):
        ax.axvline(crit, ls=":", color="orange", lw=1.5,
                   label=f"critical α={crit:.2f}")
    ax.set_ylabel("Lyapunov LLE")
    ax.legend(fontsize=8)
    ax.set_title("Energy Constraint Bifurcation Diagram\n"
                 "x(t+1) = α · F(x(t))")
    ax.grid(True, alpha=0.3)

    # Oscillation amplitude
    axes[1].plot(alphas, osc, "g-o", ms=4, lw=1.5)
    if np.isfinite(crit):
        axes[1].axvline(crit, ls=":", color="orange", lw=1.5)
    axes[1].set_ylabel("Oscillation Amp (std)")
    axes[1].grid(True, alpha=0.3)

    # Mean activity
    axes[2].plot(alphas, acts, "k-o", ms=4, lw=1.5)
    axes[2].axhline(0.5, ls="--", color="gray", lw=0.8, label="activity=0.5")
    axes[2].set_ylabel("Mean Activity")
    axes[2].set_xlabel("Energy coefficient α")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存 α 扫描图: %s", output_path)


def _try_plot_dynamic_energy(
    E_trajs: np.ndarray,
    act_trajs: np.ndarray,
    alpha: float,
    beta: float,
    E_ref: float,
    output_path: Path,
) -> None:
    """绘制动态能量 E(t) 和活动均值时序图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = E_trajs.shape[1]
    t_axis = np.arange(steps)
    n = E_trajs.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # E(t)
    ax = axes[0]
    cmap = plt.get_cmap("coolwarm", n)
    for i in range(n):
        ax.plot(t_axis, E_trajs[i], color=cmap(i / max(n - 1, 1)),
                alpha=0.6, lw=0.9)
    ax.axhline(E_ref, ls="--", color="black", lw=1,
               label=f"E_ref={E_ref:.1f}  (g=0.5)")
    ax.set_ylabel("Energy E(t)")
    ax.set_title(f"Dynamic Energy Variable  [α={alpha}, β={beta}]\n"
                 f"E_mean={float(E_trajs[:, -100:].mean()):.3f}, "
                 f"E_std={float(E_trajs[:, -100:].std()):.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Mean activity
    ax2 = axes[1]
    for i in range(n):
        ax2.plot(t_axis, act_trajs[i], color=cmap(i / max(n - 1, 1)),
                 alpha=0.6, lw=0.9)
    ax2.set_ylabel("Mean |x(t)|")
    ax2.set_xlabel("Step")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存动态能量图: %s", output_path)
