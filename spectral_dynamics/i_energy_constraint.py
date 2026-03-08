"""
I: Energy Constraint Experiment (spectral_dynamics 版)
=======================================================

验证假设 H5：有限代谢能量是维持大脑近临界态的机制之一。

与 twinbrain-dynamics 的关系
-----------------------------
twinbrain-dynamics 版对真实 GNN 施加约束（见 energy_constraint.py）。
本模块用 Wilson–Cowan 动力学替代 GNN，秒级运行，用于快速验证假设方向。

科学原理（与 GNN 版完全一致）
------------------------------
每步：
  y       = tanh(W · x(t))            ← WC 预测（无约束期望状态）
  x(t+1) = proj_E(y)                  ← L1 球投影到能量可行集

能量可行集：E = {z ∈ [0,1]^N : mean(z) ≤ E_budget}
投影解（软阈值）：x(t+1)_i = max(y_i - λ*, 0)，λ* 使 mean(x(t+1)) = E_budget

为什么是 L1 投影而不是均匀缩放：
  - 均匀缩放 (g·y): 所有神经元等比例减小，拓扑不变，科学意义为零
  - L1 投影 (proj_E(y)): 弱激活置零，强激活保留 → winner-takes-all
    → 稀疏神经表征 → 改变吸引子拓扑 → 与 Olshausen & Field (1996) 稀疏编码一致

约束边界即临界流形：
  E_budget << E*: 约束总激活 → 系统被压制 → 固定点 (LLE << 0)
  E_budget ≈ E*: 约束约 50% 激活 → 系统在边界振荡 → 近临界 (LLE ≈ 0)
  E_budget >> E*: 约束从不激活 → 无约束 WC 动力学 (LLE 由 ρ(W) 决定)
  理论临界点：E_budget* ≈ mean(tanh(W·x)) at x = x_ref

输出文件
--------
  energy_constraint_{label}.json  — 各条件 LLE / 振荡幅度 / 均值活动 / 投影激活率
  energy_constraint_{label}.png   — PCA 相空间 + LLE + 投影激活率 对比图
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_TD_DIR = Path(__file__).parent.parent / "twinbrain-dynamics"
if _TD_DIR.exists() and str(_TD_DIR) not in sys.path:
    sys.path.insert(0, str(_TD_DIR))


# ─────────────────────────────────────────────────────────────────────────────
# L1-ball projection (same logic as GNN version)
# ─────────────────────────────────────────────────────────────────────────────

def _project_energy_wc(y: np.ndarray, E_budget: float) -> Tuple["np.ndarray", bool]:
    """
    将 WC 输出 y ∈ [0,1]^N 投影到 {z : mean(z) ≤ E_budget}。

    返回 (x_projected: np.ndarray, constraint_was_active: bool)。

    # 计算成本：50 次二分法迭代，每次 O(N)；N=190 时 < 0.1 ms
    """
    current = float(y.mean())
    if current <= E_budget:
        return y.astype(np.float32), False  # feasible, no constraint

    lo, hi = 0.0, float(y.max())
    for _ in range(50):
        lam = (lo + hi) * 0.5
        proj = np.maximum(y - lam, 0.0)
        if float(proj.mean()) <= E_budget:
            hi = lam
        else:
            lo = lam

    result = np.clip(np.maximum(y - (lo + hi) * 0.5, 0.0), 0.0, 1.0)
    return result.astype(np.float32), True


# ─────────────────────────────────────────────────────────────────────────────
# LLE helper
# ─────────────────────────────────────────────────────────────────────────────

def _rosenstein(traj: np.ndarray, max_lag: int = 30, min_sep: int = 10) -> float:
    """Rosenstein LLE，优先调用 twinbrain-dynamics 实现。"""
    try:
        from analysis.lyapunov import rosenstein_lyapunov
        v, _ = rosenstein_lyapunov(traj, max_lag=max_lag, min_temporal_sep=min_sep)
        return float(v)
    except Exception:
        pass
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
    div, cnt = np.zeros(max_lag), np.zeros(max_lag, dtype=int)
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
    return float(np.polyfit(lags, div[lags], 1)[0])


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_constraint_wc(
    W: np.ndarray,
    E_budget_values: Optional[List[float]] = None,
    n_traj: int = 15,
    steps: int = 200,
    warmup: int = 30,
    max_lag: int = 30,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    label: str = "matrix",
) -> Dict:
    """
    对 WC 模型施加能量约束，对比有/无约束的动力学差异。

    每步：y = tanh(W·x)，x(t+1) = proj_E(y)
    对比条件由 E_budget_values 指定，None 或 inf 表示无约束。

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────────
    # N=190, 4 条件, n_traj=15, steps=200: ~1–3 秒（纯 NumPy，无 GPU）
    # ─────────────────────────────────────────────────────────────────────────

    Args:
        W:               连接矩阵 (N, N)。
        E_budget_values: 各条件的能量预算（None 或 inf = 无约束基线）。
                         默认 [None, 0.5, 0.35, 0.20]。
                         建议先运行无约束版本观察 E* = mean(|x|)，
                         再选 E_budget 为 E* 的倍数。
        n_traj:  每条件轨迹数（默认 15）。
        steps:   热身后记录步数（默认 200）。
        warmup:  热身步数（默认 30）。
        max_lag: LLE 最大追踪步。
        seed:    随机种子。
        output_dir: 保存目录。
        label:   文件名标签。

    Returns:
        dict 含 conditions（各条件 lle/osc/mean_activity/projection_rate），
             E_star（无约束典型能量），rho_W，hypothesis_supported。
    """
    if E_budget_values is None:
        E_budget_values = [None, 0.5, 0.35, 0.20]

    rng = np.random.default_rng(seed)
    N = W.shape[0]
    W64 = W.astype(np.float64)
    rho_W = float(np.abs(np.linalg.eigvals(W64)).max())

    conditions: Dict[str, Dict] = {}
    all_trajs: Dict[str, np.ndarray] = {}
    E_star: float = float("nan")

    for E_budget in E_budget_values:
        unconstrained = E_budget is None or not np.isfinite(float(E_budget))
        cname = "无约束（基线）" if unconstrained else f"E_budget={E_budget:.2f}"

        trajs = np.empty((n_traj, steps, N), dtype=np.float32)
        proj_active_counts = []

        for i in range(n_traj):
            x = rng.random(N)
            # warmup
            for _ in range(warmup):
                y = np.tanh(W64 @ x)
                if not unconstrained:
                    x, _ = _project_energy_wc(y, E_budget)
                else:
                    x = np.clip(y, 0.0, 1.0)
            # record
            proj_active = 0
            for t in range(steps):
                trajs[i, t] = x
                y = np.tanh(W64 @ x)
                if not unconstrained:
                    x, active = _project_energy_wc(y, E_budget)
                    if active:
                        proj_active += 1
                else:
                    x = np.clip(y, 0.0, 1.0)
            proj_active_counts.append(proj_active / steps)

        all_trajs[cname] = trajs

        # Estimate E* from unconstrained condition
        if unconstrained:
            E_star = float(trajs.mean())

        lles = [
            _rosenstein(trajs[i], max_lag=max_lag, min_sep=max(5, steps // 10))
            for i in range(n_traj)
        ]
        valid = [v for v in lles if np.isfinite(v)]
        lle = float(np.median(valid)) if valid else float("nan")
        proj_rate = float(np.mean(proj_active_counts))

        conditions[cname] = {
            "E_budget": None if unconstrained else E_budget,
            "lle": round(lle, 5),
            "osc_amplitude": round(float(trajs.std(axis=1).mean()), 5),
            "mean_activity": round(float(trajs.mean()), 5),
            "projection_rate": round(proj_rate, 4),  # fraction of steps where constraint was active
        }
        logger.debug("  %s: LLE=%.4f, proj_rate=%.2f", cname, lle, proj_rate)

    logger.info(
        "I: 能量约束实验 [%s]: ρ(W)=%.3f, E*=%.4f",
        label, rho_W, E_star,
    )
    for cname, info in conditions.items():
        logger.info(
            "  %s: LLE=%.4f, osc=%.4f, proj_rate=%.2f",
            cname, info["lle"], info["osc_amplitude"], info["projection_rate"],
        )

    # Hypothesis: does some constrained condition have LLE closer to 0 than unconstrained?
    baseline_lle = next(
        (info["lle"] for info in conditions.values() if info["E_budget"] is None),
        float("nan"),
    )
    constrained_lles = [
        info["lle"] for info in conditions.values() if info["E_budget"] is not None
    ]
    valid_c = [v for v in constrained_lles if np.isfinite(v)]
    hypothesis_supported = (
        np.isfinite(baseline_lle) and bool(valid_c) and
        min(abs(v) for v in valid_c) < abs(baseline_lle) - 0.01
    )

    result = {
        "conditions": conditions,
        "E_star": round(E_star, 5) if np.isfinite(E_star) else None,
        "rho_W": round(rho_W, 4),
        "label": label,
        "hypothesis_supported": hypothesis_supported,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / f"energy_constraint_{label}.json", "w",
                  encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        _try_plot(all_trajs, conditions, rho_W, E_star, label,
                  out / f"energy_constraint_{label}.png")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot(
    all_trajs: Dict[str, np.ndarray],
    conditions: Dict[str, Dict],
    rho_W: float,
    E_star: float,
    label: str,
    output_path: Path,
) -> None:
    """
    3 行 × n_conditions 列：
      行 1 — PCA 相空间（蓝→红时间渐变）
      行 2 — 能量（mean活动）时序（观察约束激活情况）
      行 3 — LLE / 投影激活率 汇总条形图
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    cond_names = list(conditions.keys())
    n = len(cond_names)
    if n == 0:
        return

    first = all_trajs[cond_names[0]]
    X_fit = first.reshape(-1, first.shape[2])
    X_fit_c = X_fit - X_fit.mean(0)
    try:
        from sklearn.decomposition import PCA as _PCA
        pca = _PCA(n_components=2)
        pca.fit(X_fit_c)
        def _proj(X): return pca.transform(X - X.mean(0))
    except ImportError:
        _, _, Vt = np.linalg.svd(X_fit_c, full_matrices=False)
        _V = Vt[:2]
        def _proj(X): return (X - X.mean(0)) @ _V.T

    T_rec = first.shape[1]
    t_norm = np.linspace(0, 1, T_rec)
    cmap_time = plt.get_cmap("plasma")
    cmap_cond = plt.get_cmap("tab10", n)

    fig = plt.figure(figsize=(4 * n, 11))

    # Row 1: PCA phase space
    for j, cname in enumerate(cond_names):
        ax = fig.add_subplot(3, n, j + 1)
        trajs = all_trajs[cname]
        n_t = trajs.shape[0]
        for i in range(n_t):
            pr = _proj(trajs[i])
            for k in range(T_rec - 1):
                ax.plot(pr[k:k+2, 0], pr[k:k+2, 1],
                        color=cmap_time(t_norm[k]), lw=0.7, alpha=0.55)
        info = conditions[cname]
        ax.set_title(
            f"{cname}\nLLE={info['lle']:.4f}  proj={info['projection_rate']:.0%}",
            fontsize=8,
        )
        ax.set_xlabel("PC1", fontsize=7)
        if j == 0:
            ax.set_ylabel("PC2", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Row 2: mean activity over time (shows whether constraint is binding)
    for j, cname in enumerate(cond_names):
        ax = fig.add_subplot(3, n, n + j + 1)
        trajs = all_trajs[cname]
        act = trajs.mean(axis=2)   # (n_traj, steps) — mean over neurons
        ax.fill_between(
            np.arange(T_rec),
            act.min(axis=0), act.max(axis=0),
            color=cmap_cond(j), alpha=0.25,
        )
        ax.plot(np.arange(T_rec), act.mean(axis=0),
                color=cmap_cond(j), lw=1.5)
        eb = conditions[cname].get("E_budget")
        if eb is not None:
            ax.axhline(eb, ls="--", color="red", lw=1.0,
                       label=f"E_budget={eb:.2f}")
            ax.legend(fontsize=6)
        if np.isfinite(E_star):
            ax.axhline(E_star, ls=":", color="gray", lw=0.8,
                       label=f"E*={E_star:.2f}")
        ax.set_ylabel("mean 活动" if j == 0 else "", fontsize=7)
        ax.set_xlabel("步骤", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2)

    # Row 3: summary bars
    ax_lle  = fig.add_subplot(3, 3, 7)
    ax_proj = fig.add_subplot(3, 3, 8)
    ax_osc  = fig.add_subplot(3, 3, 9)

    x_pos = np.arange(n)
    colors = [cmap_cond(j) for j in range(n)]
    short = [c[:10] + ("…" if len(c) > 10 else "") for c in cond_names]

    ax_lle.bar(x_pos, [conditions[c]["lle"] for c in cond_names],
               color=colors, alpha=0.85)
    ax_lle.axhline(0, ls="--", color="red", lw=1, label="临界 LLE=0")
    ax_lle.set_xticks(x_pos); ax_lle.set_xticklabels(short, fontsize=6)
    ax_lle.set_title("Lyapunov 指数", fontsize=8)
    ax_lle.legend(fontsize=6); ax_lle.grid(True, alpha=0.3, axis="y")

    ax_proj.bar(x_pos, [conditions[c]["projection_rate"] for c in cond_names],
                color=colors, alpha=0.85)
    ax_proj.set_xticks(x_pos); ax_proj.set_xticklabels(short, fontsize=6)
    ax_proj.set_ylim(0, 1.05)
    ax_proj.set_title("约束激活率\n(1=总是约束, 0=从不)", fontsize=7)
    ax_proj.grid(True, alpha=0.3, axis="y")

    ax_osc.bar(x_pos, [conditions[c]["osc_amplitude"] for c in cond_names],
               color=colors, alpha=0.85)
    ax_osc.set_xticks(x_pos); ax_osc.set_xticklabels(short, fontsize=6)
    ax_osc.set_title("振荡幅度", fontsize=8)
    ax_osc.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"能量约束实验（WC 版）[{label}]  ρ(W)={rho_W:.3f}  E*={E_star:.3f}\n"
        "行1: PCA 相空间  行2: 均值活动时序（红虚线=E_budget）  行3: 汇总指标\n"
        "科学预测：E_budget≈E* 时 LLE≈0（临界），约束激活率≈50%",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info("I: 保存图: %s", output_path)
