"""
I: Energy Constraint Experiment (spectral_dynamics 版)
=======================================================

验证假设 H5：

  H5  有限代谢能量供应上限（扩散约束）是维持大脑近临界态的机制之一。

**修正后的生理背景**
--------------------
大脑皮层的能量供应来源是毛细血管中氧合血红蛋白（HbO₂）向神经元的**被动扩散**。
在神经元发放的微秒到毫秒时间尺度上，血流速度可视为近似**恒定输入**，
即血液带来的氧气供应率 α_supply 基本不变。

真正起约束作用的是**扩散驱动的弹性上限**：
  - 神经活动高 → 局部 [O₂] 浓度下降 → 扩散梯度增大 → 更多 O₂ 从毛细血管扩散入
  - 神经活动低 → [O₂] 浓度回升 → 扩散梯度减小 → 扩散减弱

但这种补偿是**有上限的**（E_max）：血液中 HbO₂ 的最大释放量是固定的（生理天花板）。
即使神经活动极高，O₂ 供应也不会无限增加。

**正确的离散时间 O₂ 动力学方程（扩散模型）：**

  E(t+1) = E(t) + k_diff · (E_max − E(t)) − β · activity(t)
          = (1 − k_diff) · E(t) + k_diff · E_max − β · activity(t)

  其中：
    E(t)     : 神经元局部可用 O₂ 浓度（无量纲化）
    E_max    : 生理天花板（毛细血管最大可释放量）—— **这是关键控制参数**
    k_diff   : 扩散/摄取速率常数（典型值 0.1–0.3）
    β        : 每单位神经活动的 O₂ 消耗率
    α_supply : 血液中 HbO₂ 的补充速率（近似常数，已折入 k_diff·E_max）

  无神经活动时：E → E_max（完全氧合静息态）
  高神经活动时：E 下降，但被动扩散部分补偿，不会低于 E_min

  **E_max 才是临界态的关键参数：**
  - E_max 过小（低血氧饱和度）→ E 始终偏低 → 活动受限 → 固定点
  - E_max 适中（生理正常）   → E 在 E_ref 附近波动 → 近临界振荡
  - E_max 过大（超氧化）     → E 始终充足 → 活动不受限 → 可能混沌

  这与"血流不足时能量减少"的错误模型不同：血流几乎恒定，
  关键约束来自扩散上限 E_max，而非血流速率。

与 twinbrain-dynamics 的关系
-----------------------------
``twinbrain-dynamics/experiments/energy_constraint.py`` 使用真实的
TwinBrainDigitalTwin 模型（F = GNN 预测）。本模块使用 Wilson–Cowan 动力学
F(x) = tanh(W·x)，无需模型推断，适合**快速验证假设**。

GNN 模型在训练时隐式学习了数据中的能量约束（BOLD 信号本身反映 HbO₂ 消耗），
但本实验对其施加**显式能量约束**，检验该约束是否确实对应临界态维持机制。

E5 vs 实验 I — 区别与互补
--------------------------
  E5 (phase diagram): x(t+1) = tanh(g·W·x)  → g 缩放突触权重（可塑性/兴奋性）
  I-A (α scan):       x(t+1) = α·tanh(W·x)  → α 缩放输出（能量通量代理）
  I-D (diffusion):    E(t+1) = (1-k)·E(t) + k·E_max − β·act  → 物理扩散方程

  三者的临界点机制不同，但数值上的临界 α ≈ 临界 g ≈ 1/ρ(W)，
  提供了三条独立证据链，互相验证 H5。

四个实验
--------

**实验 A：α 扫描（简化能量约束，快速代理）**

  x(t+1) = clip(α · tanh(W · x(t)), 0, 1)

  α ≈ E_max 的代理：小 α = 低能量天花板，大 α = 高能量天花板（或无上限）
  扫描指标：LLE / 振荡幅度 / 均值活动

**实验 B：三条件并排比较（PCA 相空间可视化）**

  并排展示三种能量状态（α_low / 1.0 / α_high）的 PCA 轨迹，
  直接可视化"能量上限"控制动力学状态转变

**实验 C：动态能量变量（旧 sigmoid 门控版，保留作对照）**

  E(t+1) = clip(E(t) + α_supply − β·mean(|x|), 0, 10)
  x(t+1) = clip(g(E(t))·tanh(W·x), 0, 1)
  g(E) = 2·sigmoid(4·(E−E_ref))   ← g(E_ref)=1.0

**实验 D：O₂ 扩散模型（物理正确版）** ← 新增

  E(t+1) = (1−k_diff)·E(t) + k_diff·E_max − β·activity(t)
  x(t+1) = clip(g(E(t))·tanh(W·x), 0, 1)
  g(E)    = E / E_max                                        ← 线性能量门控

  扫描 E_max：直接检验"O₂ 扩散天花板是临界态控制参数"的假设

输出文件
--------
  energy_alpha_scan_{label}.json / .png   — 实验 A 分岔图（含动力学区标注）
  energy_comparison_{label}.png           — 实验 B PCA 三条件并排图
  energy_dynamic_E_{label}.json / .png    — 实验 C 动态能量时序图
  energy_cap_scan_{label}.json / .png     — 实验 D E_max 扫描分岔图
  energy_diffusion_{label}.json / .png    — 实验 D 单次扩散模型轨迹图
"""

**实验 C：动态能量变量 E(t)**

  E(t+1) = clip(E(t) + α_supply − β · mean(|x(t)|), 0, 10)
  x(t+1) = clip(g(E(t)) · tanh(W · x(t)), 0, 1)

  能量门控函数（修正版）：
    g(E) = 2 · sigmoid(4 · (E − E_ref))        ← 在 E=E_ref 时 g=1.0（无约束基线）
         = 2 / (1 + exp(−4·(E−E_ref)))

  关键性质：
    g(E_ref)    = 1.0   → E 在参考值时系统行为与 α=1 完全相同（无约束基线）
    g(0)        ≈ 0.04  → 能量耗尽时活动几乎为零
    g(2·E_ref)  ≈ 1.96  → 能量过剩时活动约翻倍（近似不约束上限 2.0）

  注意：早期版本使用 g(E)=sigmoid(4·(E-E_ref))，其 g(E_ref)=0.5，
  意味着正常状态下活动已被抑制 50%，这在生理上不合理。
  现已修正为 g(E_ref)=1.0（见 2026-02-25 修复日志）。

  稳态验证：若 E 收敛至 E_ref 附近，说明系统存在自稳定的代谢反馈，
  即"血氧稳态"将系统锁定在临界能量附近。

输出文件
--------
  energy_alpha_scan_{label}.json / .png   — 实验 A 分岔图（带动力学状态区）
  energy_comparison_{label}.png           — 实验 B PCA 三条件并排图
  energy_dynamic_E_{label}.json / .png    — 实验 C 动态能量时序图
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


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rosenstein(traj: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """Rosenstein LLE — delegates to twinbrain-dynamics if available."""
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        v, _ = rosenstein_lyapunov(traj, max_lag=max_lag, min_temporal_sep=min_sep)
        return float(v)
    except Exception:
        pass
    # Inline fallback (PC1 projection only)
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
            div[lag - 1] += np.log(
                abs(proj[t + lag] - proj[nn[t] + lag]) / d0 + 1e-20
            )
            cnt[lag - 1] += 1
    m = cnt > 3
    if not m.any():
        return float("nan")
    div[m] /= cnt[m]
    lags = np.where(m)[0][:min(10, m.sum())]
    slope, _ = np.polyfit(lags, div[lags], 1)
    return float(slope)


def _g_energy(E: float, E_ref: float) -> float:
    """
    能量门控函数 g(E) = 2 · sigmoid(4 · (E − E_ref))。

    性质：
      g(E_ref) = 1.0  → 在参考能量时完全无约束（与 α=1 等价）
      g(0)     ≈ 0.04 → 能量耗尽时活动几乎为零
      g → 2   as E → ∞（能量过剩上限）

    Bug note（已修复）：原来使用 sigmoid(4·(E-E_ref)) 导致 g(E_ref)=0.5，
    即参考状态下活动已被抑制 50%，不符合"E_ref 为正常工作点"的生理假设。
    """
    return 2.0 / (1.0 + np.exp(-4.0 * (E - E_ref)))


def _simulate_wc(
    W64: np.ndarray,
    x0: np.ndarray,
    steps: int,
    alpha: float = 1.0,
    E_ref: float = 1.0,
    E_init: Optional[float] = None,
    alpha_supply: Optional[float] = None,
    beta_consume: Optional[float] = None,
) -> tuple:
    """
    单条 WC 轨迹 — 支持固定 α 和动态 E(t)（sigmoid 门控版，实验 C）。

    返回 (traj, E_traj)，E_traj 在固定 α 模式下为全 alpha 常量数组。
    """
    N = W64.shape[0]
    traj = np.empty((steps, N), dtype=np.float64)
    E_traj = np.empty(steps, dtype=np.float64)
    x = x0.copy().astype(np.float64)
    E = float(E_init) if E_init is not None else None

    for t in range(steps):
        if E is not None:
            # Dynamic energy mode (Experiment C: sigmoid gate)
            g = _g_energy(E, E_ref)
            traj[t] = x
            E_traj[t] = E
            x_next = np.clip(g * np.tanh(W64 @ x), 0.0, 1.0)
            consumption = float(beta_consume) * float(np.mean(np.abs(x)))
            E = float(np.clip(E + float(alpha_supply) - consumption, 0.0, 10.0))
        else:
            # Fixed alpha mode (Experiment A)
            traj[t] = x
            E_traj[t] = alpha
            x_next = np.clip(alpha * np.tanh(W64 @ x), 0.0, 1.0)
        x = x_next

    return traj.astype(np.float32), E_traj.astype(np.float32)


def _simulate_wc_diffusion(
    W64: np.ndarray,
    x0: np.ndarray,
    steps: int,
    E_max: float,
    k_diff: float,
    beta: float,
    E_init: Optional[float] = None,
) -> tuple:
    """
    单条 WC 轨迹 — 物理正确的 O₂ 扩散模型（实验 D）。

    方程：
      E(t+1) = (1 − k_diff) · E(t) + k_diff · E_max − β · mean(|x(t)|)
      x(t+1) = clip(g(E(t)) · tanh(W · x(t)), 0, 1)
      g(E)   = E / E_max            ← 线性能量门控（不是 sigmoid）

    生理解释：
      k_diff · E_max   → 毛细血管被动扩散入量（与浓度梯度成正比）
      (1 − k_diff)·E(t) → 上一步残余 O₂
      β · activity     → 神经活动消耗
      g(E) = E/E_max   → 可用 O₂ 分数对活动的线性门控
                          (0 = 完全窒息，1 = 完全充氧，对应无约束基线)

    关键属性：
      - 无神经活动时：E → E_max（完全充氧静息态）
      - 高神经活动时：E < E_max（被动扩散无法完全补偿消耗）
      - E_max 是唯一的外部约束参数，血流率 ≈ k_diff·E_max（近似恒定）

    返回 (traj, E_traj, g_traj)，形状均为 (steps, N) 或 (steps,)。
    """
    N = W64.shape[0]
    traj = np.empty((steps, N), dtype=np.float64)
    E_traj = np.empty(steps, dtype=np.float64)
    g_traj = np.empty(steps, dtype=np.float64)
    x = x0.copy().astype(np.float64)
    E = float(E_init if E_init is not None else E_max)   # default: fully oxygenated

    for t in range(steps):
        traj[t] = x
        E_traj[t] = E
        g = float(np.clip(E / max(E_max, 1e-9), 0.0, 1.0))
        g_traj[t] = g
        x_next = np.clip(g * np.tanh(W64 @ x), 0.0, 1.0)
        activity = float(np.mean(np.abs(x)))
        # Diffusion update: passive inflow proportional to deficit, minus consumption
        E_next = (1.0 - k_diff) * E + k_diff * E_max - beta * activity
        E = float(np.clip(E_next, 0.0, E_max))
        x = x_next

    return (traj.astype(np.float32),
            E_traj.astype(np.float32),
            g_traj.astype(np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# Experiment A: α scan — bifurcation diagram
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_alpha_scan(
    W: np.ndarray,
    alpha_min: float = 0.1,
    alpha_max: float = 2.5,
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
    扫描输出能量系数 α，建立"能量约束分岔图"。

    x(t+1) = clip(α · tanh(W · x(t)), 0, 1)    α ∈ [α_min, α_max]

    三个动力学区间（理论预期）：
      α·ρ(W) < ~0.9 → 固定点（系统衰减）
      α·ρ(W) ≈ 1.0  → 近临界振荡
      α·ρ(W) > ~1.5 → 混沌（对 tanh 非线性）

    与 E5（突触增益相图）的区别：
      E5: x(t+1) = tanh(g·W·x)  → g 放大突触权重（兴奋性/可塑性）
      I:  x(t+1) = α·tanh(W·x)  → α 放大神经输出（能量门控）
    二者临界 α/g 值数值相近（均为 1/ρ(W)），但物理来源完全不同。

    # 计算成本（典型值）
    # ─────────────────────────────────────────────────────
    # N=190 脑区，n_alpha=25, n_traj=20, steps=300:
    #   时间：~2-8 秒（纯 NumPy，无模型调用）
    #   内存：n_traj × steps × N × 4B = 20×300×190×4B ≈ 4.4 MB/α 值
    #         全部 α 值同时在内存中：≈ 4.4 MB（逐 α 处理，不累积）
    # 快速模式（n_traj=5, steps=100）：~0.3 秒
    # ─────────────────────────────────────────────────────

    Args:
        W:            连接矩阵 (N, N)。
        alpha_min/max: 扫描范围（默认 0.1–2.5，宽于 E5 的 g 范围，确保全三个动力学区均可见）。
        alpha_step:   步长（默认 0.1，产生 25 个 α 值）。
        n_traj:       每个 α 值的轨迹数（默认 20，建议 ≥10 以稳健估计中位数 LLE）。
        steps:        热身后的记录步数（默认 300）。
        warmup:       热身步数（默认 50，丢弃初始瞬态）。
        max_lag:      Rosenstein LLE 最大追踪步（默认 30）。
        seed:         随机种子。
        output_dir:   保存目录（None → 不保存）。
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
        "ρ(W)=%.3f (理论临界 α*≈%.2f), n_traj=%d",
        label, alpha_min, alpha_max, alpha_step,
        rho_W, 1.0 / max(rho_W, 1e-6), n_traj,
    )

    for alpha in alpha_vals:
        trajs_list = []
        for _ in range(n_traj):
            x0 = rng.random(N)
            traj, _ = _simulate_wc(W64, x0, steps + warmup, alpha=alpha)
            trajs_list.append(traj[warmup:])  # discard warmup

        trajs = np.stack(trajs_list, axis=0)  # (n_traj, steps, N)

        traj_lles = [
            _rosenstein(trajs[i], max_lag=max_lag, min_sep=max(5, steps // 10))
            for i in range(n_traj)
        ]
        valid_lles = [v for v in traj_lles if np.isfinite(v)]
        lles.append(float(np.median(valid_lles)) if valid_lles else float("nan"))
        osc_amps.append(float(trajs.std(axis=1).mean()))
        mean_acts.append(float(trajs.mean()))

        logger.debug("  α=%.2f: LLE=%.4f, osc=%.4f, mean=%.4f",
                     alpha, lles[-1], osc_amps[-1], mean_acts[-1])

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
# Experiment B: three-condition PCA comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_comparison_wc(
    W: np.ndarray,
    alpha_low: float = 0.3,
    alpha_high: float = 1.8,
    n_traj: int = 15,
    steps: int = 200,
    warmup: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    三条件并排比较：能量匮乏 / 正常无约束 / 能量过剩。

    在同一 PCA 空间中展示三种能量状态的轨迹，直观证明 α 是动力学状态
    的控制参数，支持"能量约束决定临界态"假设。

    三种条件：
      1. 能量匮乏  (α = alpha_low  ≈ 0.3)：轨迹收缩→固定点，无振荡
      2. 正常无约束 (α = 1.0)：中等振荡，吸引子可见
      3. 能量过剩  (α = alpha_high ≈ 1.8)：振荡扩大或混沌

    # 计算成本
    # ─────────────────────────────────────────────────────
    # N=190，n_traj=15，steps=200，3 个条件：
    #   时间：~1-3 秒（纯 NumPy）
    #   内存：3 × 15 × 200 × 190 × 4B ≈ 7 MB
    # ─────────────────────────────────────────────────────

    Args:
        W:          连接矩阵 (N, N)。
        alpha_low:  能量匮乏系数（建议 < 1/ρ(W)）。
        alpha_high: 能量过剩系数（建议 > 1.5/ρ(W)）。
        n_traj:     每条件的轨迹数（默认 15）。
        steps:      每条轨迹步数（默认 200）。
        warmup:     热身步数（默认 30）。
        seed:       随机种子。
        output_dir: 保存目录。
        label:      文件名标签。

    Returns:
        dict 包含各条件的 LLE, osc_amplitude, mean_activity,
              pca_variance_top2, rho_W。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    rho_W = float(np.abs(np.linalg.eigvals(W64)).max())
    conditions = {
        "low":    alpha_low,
        "normal": 1.0,
        "high":   alpha_high,
    }
    condition_labels = {
        "low":    f"能量匮乏 α={alpha_low} (LLE<0, 固定点)",
        "normal": "正常无约束 α=1.0 (基线)",
        "high":   f"能量过剩 α={alpha_high} (LLE>0, 混沌?)",
    }

    all_trajs: Dict[str, np.ndarray] = {}
    stats: Dict[str, Dict] = {}

    # Use same initial states for fair comparison
    x0s = [rng.random(N) for _ in range(n_traj)]

    for cond_name, alpha in conditions.items():
        trajs_list = []
        for x0 in x0s:
            traj, _ = _simulate_wc(W64, x0.copy(), steps + warmup, alpha=alpha)
            trajs_list.append(traj[warmup:])
        trajs = np.stack(trajs_list, axis=0)   # (n_traj, steps, N)
        all_trajs[cond_name] = trajs

        traj_lles = [
            _rosenstein(trajs[i], max_lag=20, min_sep=max(5, steps // 10))
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
        logger.debug(
            "  %s: α=%.2f, LLE=%.4f, osc=%.4f",
            cond_name, alpha, lle, stats[cond_name]["osc_amplitude"],
        )

    result = {
        "conditions": stats,
        "rho_W": round(rho_W, 4),
        "alpha_low": alpha_low,
        "alpha_high": alpha_high,
        "label": label,
    }
    logger.info(
        "I: 三条件比较 [%s]: low_LLE=%.4f, normal_LLE=%.4f, high_LLE=%.4f",
        label,
        stats["low"]["lle"],
        stats["normal"]["lle"],
        stats["high"]["lle"],
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"energy_comparison_{label}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot_comparison(
            all_trajs, stats, condition_labels, rho_W, alpha_low, alpha_high,
            out / f"energy_comparison_{label}.png", label,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Experiment C: dynamic energy E(t)
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
    运行动态能量变量实验 C（WC 版）。

    E(t+1) = clip(E(t) + α_supply − β·mean(|x(t)|), 0, 10)
    x(t+1) = clip(g(E(t)) · tanh(W · x(t)), 0, 1)
    g(E)   = 2·sigmoid(4·(E−E_ref))          ← g(E_ref)=1.0（无约束基线）

    关键性质：
      - 若 α_supply / β ≈ 均值活动水平，则 E 收敛至 E_ref 附近
      - g(E_ref) = 1.0 意味着稳态与 α=1 等价（系统自组织到"正常工作点"）
      - 若 α/β 过大：E 持续升高 → g > 1 → 活动增强 → 消耗增大 → 最终平衡
      - 若 α/β 过小：E 持续降低 → g < 1 → 活动减弱 → 消耗减小 → 最终平衡

    homeostasis_achieved 判据：|E_稳态 − E_ref| < 0.3·E_ref

    # 计算成本
    # ─────────────────────────────────────────────────────
    # N=190，n_traj=5，steps=400：
    #   时间：~0.2-0.5 秒（纯 NumPy）
    #   内存：n_traj × steps × (N+1) × 4B ≈ 1.5 MB
    # ─────────────────────────────────────────────────────

    Args:
        W:         连接矩阵 (N, N)。
        alpha:     能量供应速率（单位/步；对应血流速率）。
        beta:      代谢消耗系数（单位/均值活动）。
        E_ref:     能量参考值（g=1.0 时的 E 值，对应正常血氧饱和度）。
        E_init:    初始能量（默认等于 E_ref → 从正常状态出发）。
        steps:     模拟步数（建议 ≥ 200 以观察稳态）。
        n_traj:    轨迹数（不同初始神经状态）。
        seed:      随机种子。
        output_dir: 保存目录。
        label:     文件名标签。

    Returns:
        dict 含 E_mean, E_std, activity_mean, homeostasis_achieved,
              alpha, beta, E_ref。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    E_trajs = np.empty((n_traj, steps), dtype=np.float32)
    act_trajs = np.empty((n_traj, steps), dtype=np.float32)

    for i in range(n_traj):
        x0 = rng.random(N)
        traj, E_traj = _simulate_wc(
            W64, x0, steps,
            E_init=E_init,
            alpha_supply=alpha,
            beta_consume=beta,
            E_ref=E_ref,
        )
        E_trajs[i] = E_traj
        # Record mean |x(t)| from the raw trajectory
        act_trajs[i] = np.abs(traj).mean(axis=1).astype(np.float32)

    ss_start = max(0, steps - 100)
    E_mean = float(E_trajs[:, ss_start:].mean())
    E_std = float(E_trajs[:, ss_start:].std())
    act_mean = float(act_trajs[:, ss_start:].mean())
    homeostasis = abs(E_mean - E_ref) < 0.3 * E_ref

    logger.info(
        "I: 动态能量 [%s] α=%.2f β=%.2f E_ref=%.2f: "
        "E_稳态=%.4f±%.4f, 活动均值=%.4f, homeostasis=%s",
        label, alpha, beta, E_ref, E_mean, E_std, act_mean, homeostasis,
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
# Experiment D: O₂ diffusion model + E_max scan
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_cap_scan(
    W: np.ndarray,
    E_max_min: float = 0.3,
    E_max_max: float = 2.5,
    E_max_step: float = 0.2,
    k_diff: float = 0.15,
    beta: float = 0.10,
    n_traj: int = 15,
    steps: int = 300,
    warmup: int = 50,
    max_lag: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    扫描氧气扩散上限 E_max，验证"有限能量天花板是临界态控制参数"假设。

    物理方程：
      E(t+1) = (1−k_diff)·E(t) + k_diff·E_max − β·mean(|x(t)|)
      x(t+1) = clip((E/E_max) · tanh(W·x(t)), 0, 1)

    关键设计：
      - 血流速率 ≈ 恒定（折入 k_diff·E_max）
      - E_max 是唯一的扫描参数（生理天花板）
      - 无神经活动时：E → E_max（完全充氧）
      - 高神经活动时：E < E_max（被动扩散补偿不足）

    参数选择指南（N=190，ρ(W)≈1）：
      - beta = 0.10 : 使得 mean activity ≈ 0.3–0.5 时，消耗 ≈ 0.03–0.05/步
      - k_diff = 0.15: 扩散时间常数 ≈ 1/k_diff ≈ 7 步（近似 fMRI TR 尺度）
      - E_max ≈ beta*act/k_diff + noise: 临界点附近，供需接近平衡

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────
    # n_E_max ≈ 12, n_traj=15, steps=300: 全部纯 NumPy，无模型调用
    # N=190: 时间 ≈ 3–10 秒；N=30（测试）≈ 0.2 秒
    # 内存：n_traj × steps × N × 4B ≈ 15×300×190×4 ≈ 3.4 MB/E_max 值
    # ─────────────────────────────────────────────────────────────────────

    Args:
        W:          连接矩阵 (N, N)。
        E_max_min/max: E_max 扫描范围。
        E_max_step: 步长。
        k_diff:     扩散/摄取速率常数（默认 0.15）。
        beta:       每单位活动的 O₂ 消耗率（默认 0.10）。
        n_traj:     每个 E_max 值的轨迹数（默认 15）。
        steps:      热身后的记录步数（默认 300）。
        warmup:     热身步数（默认 50）。
        max_lag:    Rosenstein LLE 最大追踪步（默认 30）。
        seed:       随机种子。
        output_dir: 保存目录。
        label:      文件名标签。

    Returns:
        dict 包含 E_max_values, lles, osc_amplitudes, mean_activities,
                  mean_E_fractions, critical_E_max, bifurcation_found, rho_W。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    rho_W = float(np.abs(np.linalg.eigvals(W64)).max())
    E_max_vals = np.round(
        np.arange(E_max_min, E_max_max + E_max_step / 2, E_max_step), 3
    )

    lles: List[float] = []
    osc_amps: List[float] = []
    mean_acts: List[float] = []
    mean_E_fracs: List[float] = []  # mean(E/E_max) — energy utilisation fraction

    logger.info(
        "I: E_max 扫描 [%s]: E_max ∈ [%.1f, %.1f], k_diff=%.2f, β=%.2f, "
        "ρ(W)=%.3f, n_traj=%d",
        label, E_max_min, E_max_max, k_diff, beta, rho_W, n_traj,
    )

    for E_max in E_max_vals:
        trajs_list = []
        E_frac_list = []
        for _ in range(n_traj):
            x0 = rng.random(N)
            traj, E_t, _ = _simulate_wc_diffusion(
                W64, x0, steps + warmup, E_max=E_max,
                k_diff=k_diff, beta=beta, E_init=E_max,
            )
            trajs_list.append(traj[warmup:])
            E_frac_list.append(float(E_t[warmup:].mean()) / max(E_max, 1e-9))

        trajs = np.stack(trajs_list, axis=0)

        traj_lles = [
            _rosenstein(trajs[i], max_lag=max_lag, min_sep=max(5, steps // 10))
            for i in range(n_traj)
        ]
        valid = [v for v in traj_lles if np.isfinite(v)]
        lles.append(float(np.median(valid)) if valid else float("nan"))
        osc_amps.append(float(trajs.std(axis=1).mean()))
        mean_acts.append(float(trajs.mean()))
        mean_E_fracs.append(float(np.mean(E_frac_list)))

        logger.debug("  E_max=%.2f: LLE=%.4f, E_frac=%.3f",
                     E_max, lles[-1], mean_E_fracs[-1])

    lle_arr = np.array(lles, dtype=float)
    valid_mask = np.isfinite(lle_arr)
    if valid_mask.any():
        crit_idx = int(np.argmin(np.abs(lle_arr[valid_mask])))
        critical_E_max = float(E_max_vals[valid_mask][crit_idx])
        bifurcation_found = bool(
            (lle_arr[valid_mask] < -0.01).any() and
            (lle_arr[valid_mask] > 0.01).any()
        )
    else:
        critical_E_max = float("nan")
        bifurcation_found = False

    result = {
        "E_max_values": E_max_vals.tolist(),
        "lles": lles,
        "osc_amplitudes": osc_amps,
        "mean_activities": mean_acts,
        "mean_E_fractions": mean_E_fracs,
        "critical_E_max": round(critical_E_max, 3),
        "bifurcation_found": bifurcation_found,
        "rho_W": round(rho_W, 4),
        "k_diff": k_diff,
        "beta": beta,
        "label": label,
    }
    logger.info(
        "I: E_max 扫描完成 [%s]: critical_E_max=%.2f, bifurcation=%s",
        label, critical_E_max, bifurcation_found,
    )

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"energy_cap_scan_{label}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot_cap_scan(result, out / f"energy_cap_scan_{label}.png")

    return result


def run_energy_diffusion_model(
    W: np.ndarray,
    E_max: float = 1.0,
    k_diff: float = 0.15,
    beta: float = 0.10,
    steps: int = 400,
    n_traj: int = 5,
    E_max_list: Optional[List[float]] = None,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    O₂ 扩散模型单次运行（或多个 E_max 对比）。

    方程：
      E(t+1) = (1−k_diff)·E(t) + k_diff·E_max − β·mean(|x(t)|)
      x(t+1) = clip((E/E_max)·tanh(W·x(t)), 0, 1)

    若提供 E_max_list，则并排运行多个 E_max 值（有上限 vs 无上限对比）。
    "无约束"对应 E_max = 极大值（如 5.0）：系统始终 g(E)≈1.0，等价于 α=1。

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────
    # 单次：n_traj=5, steps=400, N=190 → ~0.1–0.5 秒（纯 NumPy）
    # 三组对比：~0.3–1.5 秒
    # 内存：n_traj × steps × N × 4B ≈ 1.5 MB/E_max 值
    # ─────────────────────────────────────────────────────────────────────

    Args:
        W:           连接矩阵 (N, N)。
        E_max:       默认扩散上限（若不提供 E_max_list）。
        k_diff:      扩散速率常数（默认 0.15）。
        beta:        消耗率（默认 0.10）。
        steps:       模拟步数（默认 400）。
        n_traj:      轨迹数（默认 5）。
        E_max_list:  若提供，则多个 E_max 并排对比（如 [0.4, 1.0, 3.0]）。
        seed:        随机种子。
        output_dir:  保存目录。
        label:       文件名标签。

    Returns:
        dict 含 E_mean, E_std, activity_mean, homeostasis（若单次运行）
             或 conditions（若 E_max_list 对比模式）。
    """
    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)

    # Single or multi-E_max mode
    if E_max_list is None:
        E_max_list_use = [E_max]
        multi_mode = False
    else:
        E_max_list_use = list(E_max_list)
        multi_mode = True

    # Use same initial states for fair comparison
    x0s = [rng.random(N) for _ in range(n_traj)]

    conditions: Dict[str, Dict] = {}
    all_E_trajs: Dict[str, np.ndarray] = {}
    all_act_trajs: Dict[str, np.ndarray] = {}
    all_g_trajs: Dict[str, np.ndarray] = {}

    for em in E_max_list_use:
        E_trajs_list = []
        act_trajs_list = []
        g_trajs_list = []
        for x0 in x0s:
            traj, E_t, g_t = _simulate_wc_diffusion(
                W64, x0.copy(), steps,
                E_max=em, k_diff=k_diff, beta=beta, E_init=em,
            )
            E_trajs_list.append(E_t)
            act_trajs_list.append(np.abs(traj).mean(axis=1).astype(np.float32))
            g_trajs_list.append(g_t)

        E_trajs_arr = np.stack(E_trajs_list)
        act_trajs_arr = np.stack(act_trajs_list)
        g_trajs_arr = np.stack(g_trajs_list)
        all_E_trajs[str(em)] = E_trajs_arr
        all_act_trajs[str(em)] = act_trajs_arr
        all_g_trajs[str(em)] = g_trajs_arr

        ss_start = max(0, steps - 100)
        E_ss = E_trajs_arr[:, ss_start:]
        act_ss = act_trajs_arr[:, ss_start:]
        E_mean = float(E_ss.mean())
        E_std = float(E_ss.std())
        act_mean = float(act_ss.mean())
        homeostasis = abs(E_mean - em) < 0.3 * max(em, 1e-6)
        conditions[str(em)] = {
            "E_max": em,
            "E_mean": round(E_mean, 5),
            "E_std": round(E_std, 5),
            "E_fraction": round(E_mean / max(em, 1e-9), 4),
            "activity_mean": round(act_mean, 5),
            "homeostasis_achieved": homeostasis,
        }
        logger.info(
            "I: 扩散模型 [%s] E_max=%.2f: E_ss=%.4f±%.4f, act=%.4f, homeostasis=%s",
            label, em, E_mean, E_std, act_mean, homeostasis,
        )

    result: Dict = {
        "conditions": conditions,
        "k_diff": k_diff,
        "beta": beta,
        "label": label,
    }
    if not multi_mode:
        result.update(conditions[str(E_max)])

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"energy_diffusion_{label}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot_diffusion_model(
            all_E_trajs, all_act_trajs, all_g_trajs,
            E_max_list_use, k_diff, beta,
            out / f"energy_diffusion_{label}.png", label,
        )

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot_alpha_scan(result: Dict, output_path: Path) -> None:
    """
    绘制能量约束分岔图，含三个动力学区的彩色背景标注。

    三个区：
      固定点区  (LLE < −0.05)：蓝色背景
      近临界区  (−0.05 ≤ LLE < 0.05)：绿色背景
      混沌区    (LLE ≥ 0.05)：红色背景
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
    rho = result.get("rho_W", float("nan"))

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    ax_lle, ax_osc, ax_act = axes

    # ── Colored regime backgrounds (applied to all subplots) ──────────────
    a_lo, a_hi = float(alphas.min()), float(alphas.max())
    for ax in axes:
        ax.axvspan(a_lo, a_hi, alpha=0.04, color="white")
    # Estimate regime boundaries from LLE
    if np.isfinite(lles).any():
        lle_f = np.where(np.isfinite(lles), lles, 0.0)
        # Fixed-point zone: LLE < -0.05
        fp_mask = lle_f < -0.05
        # Chaotic zone: LLE > 0.05
        ch_mask = lle_f > 0.05
        # Near-critical zone: in between
        nc_mask = ~fp_mask & ~ch_mask

        for mask, color, name in [
            (fp_mask, "#6baed6", "固定点区"),
            (nc_mask, "#74c476", "近临界区"),
            (ch_mask, "#fb6a4a", "混沌区"),
        ]:
            if mask.any():
                # Find contiguous spans
                idx = np.where(mask)[0]
                a_span_lo = float(alphas[idx[0]]) - alpha_step_safe(alphas) / 2
                a_span_hi = float(alphas[idx[-1]]) + alpha_step_safe(alphas) / 2
                for ax in axes:
                    ax.axvspan(a_span_lo, a_span_hi, alpha=0.12, color=color,
                               label=name if ax is ax_lle else "")

    # ── LLE ───────────────────────────────────────────────────────────────────
    ax_lle.plot(alphas, lles, "b-o", ms=4, lw=1.8)
    ax_lle.axhline(0, ls="--", color="red", lw=1.2, label="LLE=0 (边缘混沌)")
    if np.isfinite(crit):
        ax_lle.axvline(crit, ls=":", color="orange", lw=1.8,
                       label=f"临界 α*={crit:.2f}")
    if np.isfinite(rho) and rho > 1e-12:
        ax_lle.axvline(1.0 / rho, ls="--", color="purple", lw=0.9, alpha=0.7,
                       label=f"1/ρ(W)={1/rho:.2f}（线性临界）")
    ax_lle.set_ylabel("Lyapunov 指数 (LLE)", fontsize=9)
    ax_lle.legend(fontsize=7, ncol=2)
    ax_lle.set_title(
        f"能量约束分岔图  [{result['label']}]\n"
        f"x(t+1) = α·tanh(W·x)  |  ρ(W)={rho:.3f},  "
        f"分岔={'已检测到' if result['bifurcation_found'] else '未检测到'}",
        fontsize=9,
    )
    ax_lle.grid(True, alpha=0.3)

    # ── Oscillation amplitude ─────────────────────────────────────────────────
    ax_osc.plot(alphas, osc, "-o", color="forestgreen", ms=4, lw=1.8)
    if np.isfinite(crit):
        ax_osc.axvline(crit, ls=":", color="orange", lw=1.8)
    ax_osc.set_ylabel("振荡幅度 (std)", fontsize=9)
    ax_osc.grid(True, alpha=0.3)

    # ── Mean activity ─────────────────────────────────────────────────────────
    ax_act.plot(alphas, acts, "-o", color="dimgray", ms=4, lw=1.8)
    ax_act.axhline(0.5, ls="--", color="gray", lw=0.9, label="mean=0.5")
    if np.isfinite(crit):
        ax_act.axvline(crit, ls=":", color="orange", lw=1.8)
    ax_act.set_ylabel("均值活动水平", fontsize=9)
    ax_act.set_xlabel("能量系数 α", fontsize=9)
    ax_act.legend(fontsize=7)
    ax_act.grid(True, alpha=0.3)

    # Add regime labels at top of LLE panel
    _add_regime_text(ax_lle, alphas, lles, a_lo, a_hi)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存分岔图: %s", output_path)


def alpha_step_safe(alphas: np.ndarray) -> float:
    """Safe estimate of alpha step size."""
    if len(alphas) < 2:
        return 0.1
    return float(np.median(np.diff(alphas)))


def _add_regime_text(ax, alphas, lles, a_lo, a_hi):
    """Add regime label text at top of LLE axis."""
    try:
        import matplotlib.pyplot as plt
        lle_f = np.where(np.isfinite(lles), lles, 0.0)
        ymax = ax.get_ylim()[1]
        step = alpha_step_safe(alphas)
        for mask, name, color in [
            (lle_f < -0.05, "固定点", "#2171b5"),
            ((lle_f >= -0.05) & (lle_f < 0.05), "近临界", "#238b45"),
            (lle_f >= 0.05, "混沌", "#cb181d"),
        ]:
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            mid_alpha = float(alphas[idx[len(idx) // 2]])
            ax.text(mid_alpha, ymax * 0.92, name, ha="center", va="top",
                    fontsize=7, color=color, fontweight="bold")
    except Exception:
        pass


def _try_plot_comparison(
    all_trajs: Dict[str, np.ndarray],
    stats: Dict[str, Dict],
    condition_labels: Dict[str, str],
    rho_W: float,
    alpha_low: float,
    alpha_high: float,
    output_path: Path,
    label: str,
) -> None:
    """
    三条件并排 PCA 比较图（2 行 × 3 列）。

    行 1：各条件的 PCA 相空间轨迹（蓝→红时间渐变）
    行 2 左：三条件叠加在同一 PCA 空间（颜色按条件）
    行 2 中：振荡幅度条形图
    行 2 右：均值活动条形图
    """
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
            from numpy.linalg import svd as _svd
            # Minimal PCA fallback
            class PCA:
                def __init__(self, n_components=2):
                    self.n_components = n_components
                    self.components_ = None
                def fit_transform(self, X):
                    Xc = X - X.mean(0)
                    _, _, Vt = _svd(Xc, full_matrices=False)
                    self.components_ = Vt[: self.n_components]
                    return Xc @ Vt[: self.n_components].T
                def transform(self, X):
                    Xc = X - X.mean(0)
                    return Xc @ self.components_.T
        except ImportError:
            return

    # Fit PCA on combined "normal" trajectories to get shared basis
    normal_trajs = all_trajs["normal"]          # (n_traj, steps, N)
    T_rec = normal_trajs.shape[1]
    X_fit = normal_trajs.reshape(-1, normal_trajs.shape[2])
    pca = PCA(n_components=2)
    pca.fit_transform(X_fit)

    cond_order = ["low", "normal", "high"]
    cond_colors = {"low": "#2166ac", "normal": "#1a9850", "high": "#d73027"}
    cond_short  = {"low": f"α={alpha_low}\n(匮乏)", "normal": "α=1.0\n(基线)",
                   "high": f"α={alpha_high}\n(过剩)"}
    t_axis = np.linspace(0, 1, T_rec)
    cmap_time = plt.get_cmap("plasma")

    fig = plt.figure(figsize=(14, 8))
    # Row 1: 3 PCA phase spaces
    axes_row1 = [fig.add_subplot(2, 3, i + 1) for i in range(3)]
    # Row 2: overlay + bar charts
    ax_overlay = fig.add_subplot(2, 3, 4)
    ax_osc     = fig.add_subplot(2, 3, 5)
    ax_act     = fig.add_subplot(2, 3, 6)

    for ax, cond in zip(axes_row1, cond_order):
        trajs = all_trajs[cond]
        info = stats[cond]
        n = trajs.shape[0]
        for i in range(n):
            X_proj = pca.transform(trajs[i])
            for k in range(T_rec - 1):
                ax.plot(X_proj[k:k+2, 0], X_proj[k:k+2, 1],
                        color=cmap_time(t_axis[k]), lw=0.6, alpha=0.7)
        ax.set_title(
            f"{cond_short[cond]}\nLLE={info['lle']:.4f}",
            fontsize=8,
        )
        ax.set_xlabel("PC1", fontsize=7)
        ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Overlay: all three conditions in one PCA space (median trajectory only)
    for cond in cond_order:
        trajs = all_trajs[cond]
        mid_i = trajs.shape[0] // 2
        X_proj = pca.transform(trajs[mid_i])
        ax_overlay.plot(X_proj[:, 0], X_proj[:, 1],
                        color=cond_colors[cond], lw=1.2, alpha=0.85,
                        label=cond_short[cond].replace("\n", " "))
        ax_overlay.plot(*X_proj[0, :2], "o", color=cond_colors[cond], ms=5)
        ax_overlay.plot(*X_proj[-1, :2], "*", color=cond_colors[cond], ms=7)
    ax_overlay.set_title("三条件叠加 PCA（中位轨迹）", fontsize=8)
    ax_overlay.set_xlabel("PC1", fontsize=7)
    ax_overlay.set_ylabel("PC2", fontsize=7)
    ax_overlay.legend(fontsize=6)
    ax_overlay.tick_params(labelsize=6)
    ax_overlay.grid(True, alpha=0.2)

    # Bar charts
    conds = cond_order
    osc_vals = [stats[c]["osc_amplitude"] for c in conds]
    act_vals = [stats[c]["mean_activity"] for c in conds]
    colors = [cond_colors[c] for c in conds]
    x_pos = np.arange(3)
    short_names = [cond_short[c].replace("\n", "\n") for c in conds]

    ax_osc.bar(x_pos, osc_vals, color=colors, alpha=0.8)
    ax_osc.set_xticks(x_pos)
    ax_osc.set_xticklabels(short_names, fontsize=7)
    ax_osc.set_ylabel("振荡幅度 (std)", fontsize=8)
    ax_osc.set_title("振荡幅度对比", fontsize=8)
    ax_osc.grid(True, alpha=0.3, axis="y")

    ax_act.bar(x_pos, act_vals, color=colors, alpha=0.8)
    ax_act.set_xticks(x_pos)
    ax_act.set_xticklabels(short_names, fontsize=7)
    ax_act.set_ylabel("均值活动", fontsize=8)
    ax_act.set_title("均值活动对比", fontsize=8)
    ax_act.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"能量约束三条件比较  [{label}]  ρ(W)={rho_W:.3f}\n"
        "上行：PCA 相空间（蓝=初期→红=末期）  下行：叠加图 + 定量指标",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存三条件比较图: %s", output_path)


def _try_plot_dynamic_energy(
    E_trajs, act_trajs, alpha, beta, E_ref, output_path, label
):
    """
    绘制动态能量 E(t) 和活动均值时序图。

    图中包含：
    - E(t) 轨迹（多条初始条件）
    - E_ref 参考线（g=1.0 工作点）
    - 均值活动 mean|x(t)|
    - g(E) 缩放因子时序（右轴）
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    steps = E_trajs.shape[1]
    t = np.arange(steps)
    n = E_trajs.shape[0]
    cmap = plt.get_cmap("coolwarm", max(n, 2))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # E(t) with E_ref reference
    ax0 = axes[0]
    for i in range(n):
        ax0.plot(t, E_trajs[i], color=cmap(i / max(n - 1, 1)), alpha=0.7, lw=1.0)
    ax0.axhline(E_ref, ls="--", color="black", lw=1.2, label=f"E_ref={E_ref} (g=1.0 工作点)")
    ax0.set_ylabel("能量 E(t)", fontsize=9)
    ax0.set_title(
        f"动态能量变量实验  [{label}]  α_supply={alpha}, β={beta}\n"
        f"g(E) = 2·sigmoid(4·(E−E_ref)),  g(E_ref) = 1.0（正常工作点）",
        fontsize=9,
    )
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    # g(E(t)) — energy gate value
    ax1 = axes[1]
    for i in range(n):
        g_traj = 2.0 / (1.0 + np.exp(-4.0 * (E_trajs[i] - E_ref)))
        ax1.plot(t, g_traj, color=cmap(i / max(n - 1, 1)), alpha=0.7, lw=1.0)
    ax1.axhline(1.0, ls="--", color="black", lw=1.0, label="g=1.0（无约束基线）")
    ax1.axhline(0.5, ls=":", color="gray",  lw=0.8, label="g=0.5（50% 抑制）")
    ax1.set_ylabel("能量门控 g(E)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Mean activity
    ax2 = axes[2]
    for i in range(n):
        ax2.plot(t, act_trajs[i], color=cmap(i / max(n - 1, 1)), alpha=0.7, lw=1.0)
    ax2.set_ylabel("均值 |x(t)|", fontsize=9)
    ax2.set_xlabel("步骤", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存动态能量图: %s", output_path)


def _try_plot_cap_scan(result: Dict, output_path: Path) -> None:
    """
    绘制 E_max 扫描分岔图（4 行）：
    LLE / 振荡幅度 / 均值活动 / 能量利用率（mean E/E_max）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    E_maxs = np.array(result["E_max_values"])
    lles = np.array(result["lles"], dtype=float)
    osc = np.array(result["osc_amplitudes"])
    acts = np.array(result["mean_activities"])
    fracs = np.array(result.get("mean_E_fractions", np.ones_like(acts)))
    crit = result["critical_E_max"]
    rho = result.get("rho_W", float("nan"))

    fig, axes = plt.subplots(4, 1, figsize=(10, 13), sharex=True)

    # Regime backgrounds
    if np.isfinite(lles).any():
        lle_f = np.where(np.isfinite(lles), lles, 0.0)
        step = float(np.median(np.diff(E_maxs))) if len(E_maxs) > 1 else 0.2
        for mask, color, name in [
            (lle_f < -0.05, "#6baed6", "固定点区"),
            ((lle_f >= -0.05) & (lle_f < 0.05), "#74c476", "近临界区"),
            (lle_f >= 0.05, "#fb6a4a", "混沌区"),
        ]:
            idx = np.where(mask)[0]
            if not len(idx):
                continue
            lo = float(E_maxs[idx[0]]) - step / 2
            hi = float(E_maxs[idx[-1]]) + step / 2
            for ax in axes:
                ax.axvspan(lo, hi, alpha=0.12, color=color,
                           label=name if ax is axes[0] else "")

    for ax, y, ylabel, color in zip(
        axes,
        [lles, osc, acts, fracs],
        ["Lyapunov LLE", "振荡幅度 (std)", "均值活动", "能量利用率 E/E_max"],
        ["steelblue", "forestgreen", "dimgray", "darkorange"],
    ):
        ax.plot(E_maxs, y, "-o", color=color, ms=4, lw=1.8)
        if np.isfinite(crit):
            ax.axvline(crit, ls=":", color="orange", lw=1.8,
                       label=f"临界 E_max*={crit:.2f}")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    axes[0].axhline(0, ls="--", color="red", lw=1.2, label="LLE=0 (边缘混沌)")
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title(
        f"O₂ 扩散上限扫描  [{result['label']}]  ρ(W)={rho:.3f}\n"
        f"E(t+1) = (1-k_diff)E + k_diff·E_max − β·act  |  "
        f"k_diff={result['k_diff']}, β={result['beta']}\n"
        f"分岔={'已检测到' if result['bifurcation_found'] else '未检测到'}  "
        f"临界 E_max*={crit:.2f}",
        fontsize=9,
    )
    axes[-1].set_xlabel("O₂ 扩散上限 E_max（生理天花板）", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存 E_max 扫描图: %s", output_path)


def _try_plot_diffusion_model(
    all_E_trajs: Dict[str, np.ndarray],
    all_act_trajs: Dict[str, np.ndarray],
    all_g_trajs: Dict[str, np.ndarray],
    E_max_list: List[float],
    k_diff: float,
    beta: float,
    output_path: Path,
    label: str,
) -> None:
    """
    绘制扩散模型多 E_max 条件下的 E(t)、g(E)、mean|x(t)| 时序对比图。

    每行对应一个 E_max 值（或对比多个），清晰展示：
    - 小 E_max（低氧天花板）→ g 始终 < 1 → 活动受抑 → 固定点
    - 适中 E_max（生理正常）→ g 在 0.5–1.0 振荡 → 近临界
    - 大 E_max（高氧/超氧） → g ≈ 1 始终 → 活动不受限 → 潜在混沌
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n_cond = len(E_max_list)
    if n_cond == 0:
        return

    cmap_cond = plt.get_cmap("plasma", max(n_cond, 2))
    steps = next(iter(all_E_trajs.values())).shape[1]
    t = np.arange(steps)

    fig, axes = plt.subplots(3, n_cond, figsize=(5 * n_cond, 9), sharex=True)
    if n_cond == 1:
        axes = axes[:, np.newaxis]   # ensure 2D indexing

    condition_labels = {
        True:  "高 E_max (无约束)",
        False: "中 E_max",
        None:  "低 E_max",
    }

    for j, em in enumerate(E_max_list):
        key = str(em)
        E_arr = all_E_trajs.get(key, np.zeros((1, steps)))
        act_arr = all_act_trajs.get(key, np.zeros((1, steps)))
        g_arr = all_g_trajs.get(key, np.zeros((1, steps)))
        n = E_arr.shape[0]
        c = cmap_cond(j / max(n_cond - 1, 1))

        # E(t)
        for i in range(n):
            axes[0, j].plot(t, E_arr[i], color=c, alpha=0.6, lw=0.9)
        axes[0, j].axhline(em, ls="--", color="black", lw=1.0,
                           label=f"E_max={em:.2f}")
        axes[0, j].set_ylabel("E(t)", fontsize=8)
        axes[0, j].set_title(f"E_max = {em:.2f}", fontsize=9)
        axes[0, j].legend(fontsize=7)
        axes[0, j].grid(True, alpha=0.3)

        # g(E) = E/E_max
        for i in range(n):
            axes[1, j].plot(t, g_arr[i], color=c, alpha=0.6, lw=0.9)
        axes[1, j].axhline(1.0, ls="--", color="black", lw=0.9,
                           label="g=1.0 (无约束)")
        axes[1, j].set_ylim(-0.05, 1.15)
        axes[1, j].set_ylabel("g(E) = E/E_max", fontsize=8)
        axes[1, j].legend(fontsize=7)
        axes[1, j].grid(True, alpha=0.3)

        # Activity
        for i in range(n):
            axes[2, j].plot(t, act_arr[i], color=c, alpha=0.6, lw=0.9)
        axes[2, j].set_ylabel("均值 |x(t)|", fontsize=8)
        axes[2, j].set_xlabel("步骤", fontsize=8)
        axes[2, j].grid(True, alpha=0.3)

    fig.suptitle(
        f"O₂ 扩散模型 [{label}]  k_diff={k_diff}, β={beta}\n"
        "行1: O₂浓度E(t)  行2: 能量门控g(E)=E/E_max  行3: 神经活动\n"
        "有限E_max（扩散天花板）↔ 无限E_max（无约束）的动力学差异",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存扩散模型图: %s", output_path)
