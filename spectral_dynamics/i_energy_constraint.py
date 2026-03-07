"""
I: Energy Constraint Experiment (spectral_dynamics 版)
=======================================================

验证假设 H5：

  H5  有限代谢能量通量（血氧限制）是维持大脑近临界态的机制之一。

与 twinbrain-dynamics 的关系
-----------------------------
``twinbrain-dynamics/experiments/energy_constraint.py`` 使用真实的
TwinBrainDigitalTwin 模型（F = GNN 预测）。本模块使用 Wilson–Cowan 动力学
F(x) = tanh(W·x)（W 来自响应矩阵或功能连接矩阵），无需模型推断，
适合在 spectral_dynamics 管线中快速验证假设。

两个实验
--------

**实验 A：α 扫描（简化能量约束）**

  x(t+1) = clip(α · tanh(W · x(t)), 0, 1)

  注意：与 E5 相图（``g · W``，在激活前缩放）不同，
  这里 α 在激活**后**缩放，物理含义是"输出能量门控"而非"突触增益"。

  当 W = FC（功能连接矩阵，对称）：
  - α < 1/ρ(W)：系统收缩到固定点
  - α ≈ 1/ρ(W)：振荡（近临界）
  - α > 1/ρ(W)：混沌

**实验 B：动态能量变量 E(t)**

  E(t+1) = E(t) + α − β · mean(|x(t)|)
  x(t+1) = clip(g(E(t)) · tanh(W · x(t)), 0, 1)
  g(E) = sigmoid(4 · (E − E_ref))

输出文件
--------
  energy_alpha_scan_{label}.json / .png   — 实验 A 相图
  energy_dynamic_E_{label}.json / .png    — 实验 B 时序图
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Attempt to reuse twinbrain-dynamics Rosenstein implementation
_TD_DIR = Path(__file__).parent.parent / "twinbrain-dynamics"
if _TD_DIR.exists() and str(_TD_DIR) not in sys.path:
    sys.path.insert(0, str(_TD_DIR))


def _rosenstein(traj: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """Rosenstein LLE — delegates to twinbrain-dynamics if available."""
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        v, _ = rosenstein_lyapunov(traj, max_lag=max_lag, min_temporal_sep=min_sep)
        return float(v)
    except Exception:
        pass
    # Inline fallback (PC1 projection)
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


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: α scan
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_alpha_scan(
    W: np.ndarray,
    alpha_min: float = 0.5,
    alpha_max: float = 1.5,
    alpha_step: float = 0.1,
    n_traj: int = 20,
    steps: int = 300,
    warmup: int = 50,
    max_lag: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    扫描输出能量系数 α，建立 WC 模型的"能量约束分岔图"。

    x(t+1) = clip(α · tanh(W · x(t)), 0, 1)    α ∈ [α_min, α_max]

    这与 E5（耦合强度相图）的关键区别：
    - E5: x(t+1) = tanh(g·W·x)   → g 在激活前缩放（突触增益）
    - 本实验: x(t+1) = α·tanh(W·x) → α 在激活后缩放（能量门控）

    Args:
        W:            连接矩阵 (N, N)。
        alpha_min/max: 扫描范围。
        alpha_step:   步长。
        n_traj:       每个 α 值的轨迹数。
        steps:        每条轨迹步数。
        warmup:       热身步数（不计入指标）。
        max_lag:      Rosenstein LLE 最大追踪步。
        seed:         随机种子。
        output_dir:   保存目录。
        label:        文件名标签。

    Returns:
        dict 包含 alpha_values, lles, osc_amplitudes, mean_activities,
                  critical_alpha, bifurcation_found, rho_W。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    eigvals_W = np.linalg.eigvals(W64)
    rho_W = float(np.abs(eigvals_W).max())

    alpha_vals = np.round(
        np.arange(alpha_min, alpha_max + alpha_step / 2, alpha_step), 3
    )

    lles: List[float] = []
    osc_amps: List[float] = []
    mean_acts: List[float] = []

    logger.info(
        "I: 能量 α 扫描 [%s]: α ∈ [%.1f, %.1f], step=%.2f, "
        "ρ(W)=%.3f, n_traj=%d",
        label, alpha_min, alpha_max, alpha_step, rho_W, n_traj,
    )

    for alpha in alpha_vals:
        trajs = np.empty((n_traj, steps, N), dtype=np.float32)
        for i in range(n_traj):
            x = rng.random(N).astype(np.float64)
            for _ in range(warmup):
                x = np.clip(alpha * np.tanh(W64 @ x), 0.0, 1.0)
            for t in range(steps):
                trajs[i, t] = x.astype(np.float32)
                x = np.clip(alpha * np.tanh(W64 @ x), 0.0, 1.0)

        traj_lles = [
            _rosenstein(trajs[i], max_lag=max_lag, min_sep=max(5, steps // 10))
            for i in range(n_traj)
        ]
        valid_lles = [v for v in traj_lles if np.isfinite(v)]
        lles.append(float(np.median(valid_lles)) if valid_lles else float("nan"))
        osc_amps.append(float(trajs.std(axis=1).mean()))
        mean_acts.append(float(trajs.mean()))

        logger.debug("  α=%.2f: LLE=%.4f", alpha, lles[-1])

    lle_arr = np.array(lles, dtype=float)
    valid = np.isfinite(lle_arr)
    if valid.any():
        crit_idx = int(np.argmin(np.abs(lle_arr[valid])))
        critical_alpha = float(alpha_vals[valid][crit_idx])
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
        "rho_W": round(rho_W, 4),
        "label": label,
    }

    logger.info(
        "I: α 扫描完成 [%s]: critical_α=%.2f, bifurcation=%s",
        label, critical_alpha, bifurcation_found,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"energy_alpha_scan_{label}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("I: 保存 α 扫描结果: %s", json_path)
        _try_plot_alpha_scan(result, out / f"energy_alpha_scan_{label}.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: dynamic energy E(t)
# ─────────────────────────────────────────────────────────────────────────────

def run_dynamic_energy(
    W: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5,
    E_ref: float = 1.0,
    E_init: float = 1.0,
    steps: int = 400,
    n_traj: int = 5,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    运行动态能量变量实验（WC 版）。

    E(t+1) = clip(E(t) + α − β·mean(|x(t)|), 0, 10)
    x(t+1) = clip(g(E(t)) · tanh(W · x(t)), 0, 1)
    g(E) = sigmoid(4·(E − E_ref))

    Args:
        W:       连接矩阵 (N, N)。
        alpha:   能量供应速率。
        beta:    代谢消耗系数。
        E_ref:   能量参考值（g=0.5 时的 E 值）。
        E_init:  初始能量值。
        steps:   模拟步数。
        n_traj:  轨迹数。
        seed:    随机种子。
        output_dir: 保存目录。
        label:   文件名标签。

    Returns:
        dict 含 E_mean, E_std, activity_mean, homeostasis_achieved。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    E_trajs = np.empty((n_traj, steps), dtype=np.float32)
    act_trajs = np.empty((n_traj, steps), dtype=np.float32)

    for i in range(n_traj):
        x = rng.random(N).astype(np.float64)
        E = float(E_init)
        for t in range(steps):
            E_trajs[i, t] = E
            act_trajs[i, t] = float(np.mean(np.abs(x)))
            g_E = 1.0 / (1.0 + np.exp(-4.0 * (E - E_ref)))
            x = np.clip(g_E * np.tanh(W64 @ x), 0.0, 1.0)
            E = float(np.clip(E + alpha - beta * float(np.mean(np.abs(x))), 0.0, 10.0))

    ss_start = max(0, steps - 100)
    E_mean = float(E_trajs[:, ss_start:].mean())
    E_std = float(E_trajs[:, ss_start:].std())
    act_mean = float(act_trajs[:, ss_start:].mean())
    homeostasis = abs(E_mean - E_ref) < 0.3 * E_ref

    logger.info(
        "I: 动态能量 [%s] α=%.2f β=%.2f: E_ss=%.4f±%.4f, 稳态=%s",
        label, alpha, beta, E_mean, E_std, homeostasis,
    )

    result = {
        "E_mean": round(E_mean, 6),
        "E_std": round(E_std, 6),
        "activity_mean": round(act_mean, 6),
        "homeostasis_achieved": homeostasis,
        "alpha": alpha,
        "beta": beta,
        "E_ref": E_ref,
        "label": label,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"energy_dynamic_E_{label}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot_dynamic_energy(
            E_trajs, act_trajs, alpha, beta, E_ref,
            out / f"energy_dynamic_E_{label}.png", label,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_alpha_scan(result: Dict, output_path: Path) -> None:
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
    rho = result.get("rho_W", float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    for ax, y, ylabel, color in zip(
        axes,
        [lles, osc, acts],
        ["Lyapunov LLE", "Oscillation Amp (std)", "Mean Activity"],
        ["steelblue", "forestgreen", "dimgray"],
    ):
        ax.plot(alphas, y, "-o", color=color, ms=4, lw=1.5)
        if np.isfinite(crit):
            ax.axvline(crit, ls=":", color="orange", lw=1.5,
                       label=f"critical α={crit:.2f}")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    axes[0].axhline(0, ls="--", color="red", lw=1, label="LLE=0")
    if np.isfinite(rho):
        axes[0].axvline(1.0 / rho if rho > 0 else float("inf"),
                        ls="--", color="purple", lw=0.8, alpha=0.6,
                        label=f"1/ρ(W)={1/rho:.2f}" if rho > 0 else "")
    axes[0].set_title(
        f"Energy Constraint Bifurcation  [{result['label']}]\n"
        f"x(t+1) = α·tanh(W·x)  |  ρ(W)={rho:.3f}, "
        f"bifurcation={'YES' if result['bifurcation_found'] else 'NO'}"
    )
    axes[0].legend(fontsize=8)
    axes[2].set_xlabel("Energy coefficient α")

    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存 α 扫描图: %s", output_path)


def _try_plot_dynamic_energy(
    E_trajs, act_trajs, alpha, beta, E_ref, output_path, label
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = E_trajs.shape[1]
    t = np.arange(steps)
    n = E_trajs.shape[0]
    cmap = plt.get_cmap("coolwarm", n)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for i in range(n):
        axes[0].plot(t, E_trajs[i], color=cmap(i / max(n - 1, 1)), alpha=0.6, lw=0.9)
        axes[1].plot(t, act_trajs[i], color=cmap(i / max(n - 1, 1)), alpha=0.6, lw=0.9)

    axes[0].axhline(E_ref, ls="--", color="black", lw=1, label=f"E_ref={E_ref}")
    axes[0].set_ylabel("Energy E(t)")
    axes[0].set_title(
        f"Dynamic Energy Variable  [{label}]  α={alpha}, β={beta}\n"
        f"WC: x(t+1) = g(E)·tanh(W·x),  g(E)=sigmoid(4·(E−E_ref))"
    )
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("Mean |x(t)|")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存动态能量图: %s", output_path)
