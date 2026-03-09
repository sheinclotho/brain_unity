"""
I: Energy Constraint Experiment (spectral_dynamics 版)
=======================================================

验证假设 H5：有限代谢能量是维持大脑近临界态的机制之一。

本模块是 twinbrain-dynamics 能量约束实验的 spectral_dynamics 入口。
它不运行任何动力学模拟——所有轨迹由 twinbrain-dynamics 管线生成，
本模块只做结果分析和可视化。

与 twinbrain-dynamics 的关系
-----------------------------
twinbrain-dynamics 版（experiments/energy_constraint.py）：
  - 用真实 GNN 模型运行带能量约束的动力学
  - 生成轨迹 (n_traj, T, N)
  - 调用 run_energy_budget_analysis() 分析结果

本模块：
  - 接受已由 GNN 生成的轨迹作为输入
  - 直接调用 run_energy_budget_analysis() 分析
  - 在 spectral_dynamics 工作流中提供一致的接口

科学原理
--------
能量可行集：E = {z : mean(|z|) ≤ E_budget}

建议工作流（GNN 版）：
  1. 运行基线（无约束）：python run_dynamics_analysis.py ...
  2. 分析基线能量：run_energy_budget() 报告 E* 和推荐预算
  3. 以 --energy-budget X 重新运行，用 GNN 生成约束轨迹
  4. 对比 LLE / PCA / oscillation amplitude

输出文件
--------
  energy_budget_{label}.json  — E* / 推荐预算 / 分区域能量
  energy_budget_{label}.png   — 能量分布直方图
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["run_energy_budget"]


def run_energy_budget(
    trajectories: np.ndarray,
    state_bounds=None,
    output_dir: Optional[Path] = None,
    label: str = "gnn",
) -> Dict:
    """
    分析 GNN 轨迹的能量分布，为 --energy-budget 参数提供参考值。

    委托给 twinbrain-dynamics 的 ``run_energy_budget_analysis``。
    若 twinbrain-dynamics 不在 sys.path，则执行内置的简化版本。

    Args:
        trajectories:  GNN 自由动力学轨迹，shape (n_traj, T, N)。
                       由 twinbrain-dynamics 管线（步骤 3）生成。
        state_bounds:  (lo, hi) 或 None（z-scored 无界数据传 None）。
        output_dir:    结果保存目录；None → 不保存。
        label:         文件名标签。

    Returns:
        dict 包含：
          E_mean, E_std, E_median, E_per_region,
          recommended_budgets (tight/moderate/natural/relaxed)
    """
    try:
        from experiments.energy_constraint import run_energy_budget_analysis
        result = run_energy_budget_analysis(
            trajectories=trajectories,
            state_bounds=state_bounds,
            output_dir=Path(output_dir) / label if output_dir else None,
        )
    except ImportError:
        logger.warning(
            "run_energy_budget: experiments.energy_constraint not importable; "
            "using built-in fallback."
        )
        result = _energy_budget_builtin(trajectories, state_bounds)

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"energy_budget_{label}.json"
        # E_per_region is ndarray — convert to list for JSON
        serialisable = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in result.items()
        }
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(serialisable, fh, indent=2, ensure_ascii=False)
        logger.info("保存能量预算分析: %s", json_path)
        _try_plot(result, out / f"energy_budget_{label}.png", label)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Built-in fallback (used when twinbrain-dynamics is not on sys.path)
# ─────────────────────────────────────────────────────────────────────────────

_TIGHT    = 0.4
_MODERATE = 0.7
_NATURAL  = 1.0
_RELAXED  = 1.3


def _energy_budget_builtin(
    trajectories: np.ndarray,
    state_bounds=None,
) -> Dict:
    """
    Minimal energy budget analysis without twinbrain-dynamics dependency.

    Matches the output schema of ``run_energy_budget_analysis``.
    """
    n_traj, T, N = trajectories.shape
    burnin = max(10, T // 10)
    traj_valid = trajectories[:, burnin:, :]

    E_traj = np.abs(traj_valid).mean(axis=(1, 2))
    E_mean   = float(E_traj.mean())
    E_std    = float(E_traj.std())
    E_median = float(np.median(E_traj))
    E_per_region = np.abs(traj_valid).mean(axis=(0, 1))

    recommended = {
        "tight_constraint":    round(_TIGHT    * E_mean, 6),
        "moderate_constraint": round(_MODERATE * E_mean, 6),
        "natural":             round(_NATURAL  * E_mean, 6),
        "relaxed":             round(_RELAXED  * E_mean, 6),
    }

    logger.info("能量分析(builtin): E*=%.4f ± %.4f", E_mean, E_std)
    return {
        "E_mean":             E_mean,
        "E_std":              E_std,
        "E_median":           E_median,
        "E_per_region":       E_per_region,
        "recommended_budgets": recommended,
        "n_traj":             n_traj,
        "n_steps_used":       T - burnin,
        "burnin":             burnin,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def _try_plot(result: Dict, output_path: Path, label: str) -> None:
    """Energy distribution histogram + recommended budget annotations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from spectral_dynamics.plot_utils import configure_matplotlib
        configure_matplotlib()
    except ImportError:
        return

    E_per_region = result.get("E_per_region")
    if E_per_region is None:
        return
    if isinstance(E_per_region, list):
        E_per_region = np.array(E_per_region)

    rec = result.get("recommended_budgets", {})
    E_mean = result.get("E_mean", float("nan"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Histogram of per-region mean energy
    ax = axes[0]
    ax.hist(E_per_region, bins=30, color="steelblue", alpha=0.75, edgecolor="k", lw=0.4)
    ax.axvline(E_mean, ls="--", color="red", lw=1.5, label=f"E*={E_mean:.4f}")
    ax.set_xlabel("Region Mean Energy E")
    ax.set_ylabel("Region Count")
    ax.set_title("Region Energy Distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Recommended budgets bar chart
    ax2 = axes[1]
    if rec:
        keys = list(rec.keys())
        vals = [rec[k] for k in keys]
        colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
        ax2.bar(keys, vals, color=colors[:len(keys)], alpha=0.8, edgecolor="k", lw=0.5)
        ax2.axhline(E_mean, ls="--", color="gray", lw=1, label=f"E*={E_mean:.4f}")
        ax2.set_ylabel("Recommended E_budget")
        ax2.set_title("Recommended Energy Budgets")
        ax2.legend(fontsize=8)
        ax2.set_xticklabels(keys, rotation=20, ha="right", fontsize=8)
        for i, v in enumerate(vals):
            ax2.text(i, v + 0.002, f"{v:.4f}", ha="center", va="bottom", fontsize=7)
        ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Energy Constraint Budget Analysis  [{label}]", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("保存: %s", output_path)
