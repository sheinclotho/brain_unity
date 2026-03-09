"""
Energy Constraint Experiment
==============================

对训练好的 GNN 施加显式能量约束，验证"有限代谢能量是维持大脑近临界态的机制之一"假设。

用法
----
只需加一个命令行标志，对模型施加能量约束后重新运行完整分析管线：

  # 基线（无约束）：
  python run_dynamics_analysis.py --quick --model M.pt --graph G.pt \\
      --output outputs/baseline

  # 能量约束 0.7（30% 活动被压制）：
  python run_dynamics_analysis.py --quick --model M.pt --graph G.pt \\
      --energy-budget 0.7 --output outputs/energy_0.7

  # 强约束 0.4：
  python run_dynamics_analysis.py --quick --model M.pt --graph G.pt \\
      --energy-budget 0.4 --output outputs/energy_0.4

对比三个目录中的 lyapunov_histogram.png、pca_trajectories.png 等图，
即可观察能量约束对动力学状态的影响。

科学原理
--------
*为什么不能用 g·F(x)？*

  x(t+1) = g · F(x(t))  只是均匀缩放输出，拓扑完全不变：
    - 每个神经元的激活比例不变
    - 吸引子形状不变，只是等比例收缩
    - 无论 g 取多少，只要 g>0，轨迹的几何结构是等价的
  这不是能量约束，只是换了一把尺子。

*正确实现：L1 球投影（软阈值）*

  每步预测后对输出施加能量约束投影：
    y = F(x(t))                               ← GNN 预测（无约束期望状态）
    x(t+1) = proj_E(y)                        ← 投影到能量可行集

  能量可行集：  E = {z : mean(|z|) ≤ E_budget}
  投影解（软阈值）：
    x(t+1)_i = sign(y_i) · max(|y_i| - λ*, 0)
    其中 λ* 使 mean(|x(t+1)|) = E_budget

  这与均匀缩放本质不同：
    - 弱激活（|y_i| < λ*）被置零（winner-takes-all）
    - 强激活（|y_i| > λ*）小幅减少
    - 稀疏度随 E_budget 降低而增大
    - 吸引子拓扑被改变，而非只是等比缩放

*自回归反馈：约束后的状态进入 GNN 的下一步上下文*

  本实现逐步运行（num_steps=1），每步将约束后的 x(t+1) 注入上下文窗口，
  作为 GNN 下一步预测的历史输入。这确保：
    - GNN 的"记忆"反映的是约束下的真实历史，而非假想的无约束历史
    - 能量约束真正影响动力学演化，而非事后裁剪峰值

*假设的可检验预测*

  若"有限能量维持临界态"假设成立：
    E_budget << E*  → LLE << 0（系统被压制，固定点）
    E_budget ≈ E*  → LLE ≈ 0 （近临界，振荡丰富）
    E_budget >> E*  → LLE > 0 （无约束，可能过激）
  其中 E* = 模型在无约束状态下的典型能量 = mean(|x_unconstrained|)。

  若假设不成立：LLE 对 E_budget 不敏感，或 E_budget=E* 时 LLE 不接近 0。

*计算成本*

  逐步推断（num_steps=1 × steps 次 GNN 调用）：
    quick 模式（steps=200, n_init=20）：200 × 20 = 4,000 次 GNN 调用
    GPU（RTX 3090）：~30–90 秒（比批量调用慢约 4×，因为无批处理）
    CPU（无 GPU） ：~10–30 分钟
  若速度是瓶颈，可用 --steps 50 --n-init 5 先做快速验证。

文献背景
--------
本实现对应以下经典工作的计算实现：
  - Lennie (2003) "The cost of cortical computation"：L1 能量约束
  - Olshausen & Field (1996) 稀疏编码：L1 投影产生稀疏神经表征
  - Shew et al. (2011) "Information capacity and transmission are maximized
    in balanced cortical networks with neuronal avalanches"：
    代谢约束下，临界态最大化信息传输效率
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Budget multipliers relative to the natural energy E*
_TIGHT_FACTOR:    float = 0.4   # strong suppression, ~80% of states zeroed
_MODERATE_FACTOR: float = 0.7   # moderate suppression, ~50% of time active
_NATURAL_FACTOR:  float = 1.0   # no effective constraint (boundary at E*)
_RELAXED_FACTOR:  float = 1.3   # slight relaxation, almost unconstrained


# ─────────────────────────────────────────────────────────────────────────────
# Core: L1-ball projection (the science)
# ─────────────────────────────────────────────────────────────────────────────

def _project_energy(
    y: np.ndarray,
    E_budget: float,
    state_bounds: Optional[Tuple[float, float]],
) -> np.ndarray:
    """
    将状态 y 投影到能量可行集 {z : mean(|z|) ≤ E_budget}。

    对于非负有界状态（fMRI, state_bounds=(0,1)）：
      投影解为 max(y_i - λ*, 0)，λ* 使 mean(max(y-λ*, 0)) = E_budget
      → 弱激活置零（稀疏），强激活保留（略减）

    对于可负无界状态（joint/EEG, state_bounds=None）：
      投影解为软阈值 sign(y_i)·max(|y_i|-λ*, 0)
      → 同上，但对正负对称

    若 mean(|y|) ≤ E_budget（已满足约束）：返回 y 不变。

    # 计算成本（单次调用）
    # ─────────────────────────────────────────────────────────────────────────
    # 二分法 50 次迭代，每次 O(N)；N=190 时 < 0.1 ms
    # ─────────────────────────────────────────────────────────────────────────
    """
    current = float(np.mean(np.abs(y)))
    if current <= E_budget:
        return y.astype(np.float32)   # already feasible, no constraint active

    # Binary search for threshold λ* such that mean(|proj(y, λ)|) = E_budget
    lo, hi = 0.0, float(np.abs(y).max())
    for _ in range(50):
        lam = (lo + hi) * 0.5
        if state_bounds is not None:
            proj = np.maximum(y - lam, state_bounds[0])
        else:
            proj = np.sign(y) * np.maximum(np.abs(y) - lam, 0.0)
        if float(np.mean(np.abs(proj))) <= E_budget:
            hi = lam
        else:
            lo = lam

    lam_star = (lo + hi) * 0.5
    if state_bounds is not None:
        result = np.clip(np.maximum(y - lam_star, state_bounds[0]),
                         state_bounds[0], state_bounds[1])
    else:
        result = np.sign(y) * np.maximum(np.abs(y) - lam_star, 0.0)

    return result.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Calibration: estimate typical unconstrained energy
# ─────────────────────────────────────────────────────────────────────────────

def estimate_typical_energy(
    simulator,
    n_init: int = 5,
    steps: int = 30,
    seed: int = 0,
) -> float:
    """
    快速估计无约束模型的典型能量 E* = mean(|x|)。

    用于为 --energy-budget 提供标定：
      E_budget = 0.8 × E* 表示约束为典型能量的 80%（中等约束）。

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────────
    # n_init=5, steps=30：150 次 GNN 调用，~5–15 秒 GPU
    # ─────────────────────────────────────────────────────────────────────────
    """
    rng = np.random.default_rng(seed)
    energies = []
    for i in range(n_init):
        x0 = simulator.sample_random_state(rng=rng)
        traj, _ = simulator.rollout(x0=x0, steps=steps,
                                    context_window_idx=i % simulator.n_temporal_windows)
        energies.append(float(np.mean(np.abs(traj))))
    E_star = float(np.mean(energies))
    logger.info(
        "估计典型能量 E* = %.4f  (n_init=%d, steps=%d)",
        E_star, n_init, steps,
    )
    return E_star


# ─────────────────────────────────────────────────────────────────────────────
# EnergyConstrainedSimulator
# ─────────────────────────────────────────────────────────────────────────────

class EnergyConstrainedSimulator:
    """
    对 BrainDynamicsSimulator 施加显式能量约束的包装器。

    机制
    ----
    每步：
      1. 调用 GNN 预测 y = F(history)
      2. L1 投影：x_constrained = proj_E(y)，使 mean(|x|) ≤ E_budget
      3. 将 x_constrained（而非原始预测 y）注入上下文，作为下一步的历史

    步骤 3 是关键：GNN 的记忆反映的是约束后的真实历史，
    从而能量约束真正影响动力学的自我演化，而非事后裁剪。

    参数
    ----
    simulator : BrainDynamicsSimulator
        被包装的模拟器。
    E_budget : float
        能量预算上限 = 允许的最大 mean(|x|)。
        建议用 estimate_typical_energy() 先估计 E*，再选 E_budget 为 E* 的倍数。
        E_budget = E*     → 约束刚好在边界（约 50% 时间激活）
        E_budget = 0.7·E* → 中等约束（约 80% 时间激活）
        E_budget = 0.4·E* → 强约束（几乎总是激活）
        E_budget = 1e9    → 无约束（等价于不加包装器）

    # 计算成本
    # ─────────────────────────────────────────────────────────────────────────
    # 每步 = 1 次 predict_future(num_steps=1) + 1 次 _project_energy
    # 比批量 rollout(steps=N) 慢约 4×（因为无批处理）
    # steps=200, n_init=20：4,000 次 GNN 调用，GPU ~30–90 秒
    # ─────────────────────────────────────────────────────────────────────────
    """

    def __init__(self, simulator, E_budget: float) -> None:
        self._sim = simulator
        self.E_budget = float(E_budget)

    # ── Pass-through properties ───────────────────────────────────────────────
    def __getattr__(self, name: str):
        return getattr(self._sim, name)

    @property
    def n_regions(self) -> int:            return self._sim.n_regions
    @property
    def state_bounds(self):                return self._sim.state_bounds
    @property
    def dt(self) -> float:                 return self._sim.dt
    @property
    def modality(self) -> str:             return self._sim.modality
    @property
    def n_temporal_windows(self) -> int:   return self._sim.n_temporal_windows

    def sample_random_state(self, rng=None, from_data: bool = False,
                            step_idx=None) -> np.ndarray:
        return self._sim.sample_random_state(
            rng=rng, from_data=from_data, step_idx=step_idx
        )

    # ── Core: constrained step-by-step rollout ────────────────────────────────
    def rollout(
        self,
        x0=None,
        steps: int = 50,
        stimulus=None,
        context_window_idx: int = 0,
        **_kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        逐步运行受能量约束的自回归轨迹。

        每步使用 predict_future(num_steps=1) 获取单步预测，
        对预测结果施加 L1 投影，再将约束后状态注入上下文窗口。

        这与原始 rollout 的唯一区别：
          - 上下文随约束后状态更新（而非原始预测）
          - 每步的输入历史反映了能量约束下的真实演化路径
        """
        import torch
        from simulator.brain_dynamics_simulator import _advance_context

        modality = self._sim.modality
        _bounds  = self._sim.state_bounds
        n = self._sim.n_regions

        if modality == "joint":
            logger.warning(
                "EnergyConstrainedSimulator: joint 模态下，能量投影应用于联合状态向量，"
                "但由于 joint 上下文存储原始（非 z-scored）值，"
                "约束后的状态无法直接注入上下文——上下文将以原始预测（无约束）前进。\n"
                "若需要完整约束反馈，建议使用 modality='fmri' 或 modality='eeg'。"
            )

        context = self._sim._get_context_for_window(context_window_idx)
        self._sim._inject_x0_into_context(context, x0)

        trajectory = np.empty((steps, n), dtype=np.float32)
        times      = np.arange(steps, dtype=np.float32) * self._sim.dt

        for t in range(steps):
            # ── 1. GNN one-step prediction ──────────────────────────────────
            pred_dict = self._sim.model.predict_future(context, num_steps=1)

            if modality == "joint":
                # joint: build z-normalised concatenated vector then project
                fmri_pred = pred_dict.get("fmri")
                eeg_pred  = pred_dict.get("eeg")
                if fmri_pred is None or eeg_pred is None:
                    raise RuntimeError("joint 模态缺少 fmri 或 eeg 预测。")
                y = self._sim._z_normalise_joint(fmri_pred, eeg_pred)[0]  # (n,)
                x_next = _project_energy(y, self.E_budget, _bounds)
                trajectory[t] = x_next
                # Limitation: joint context stores raw (un-z-scored) values;
                # feeding back constrained z-scored state requires un-z-scoring
                # which involves the full modality split. For now, advance context
                # with the unconstrained predictions.  This means the GNN's memory
                # is unconstrained for joint mode — the output is still projected.
                context = _advance_context(context, pred_dict)
            else:
                pred = pred_dict[modality]           # [N, 1, 1]
                y = pred[:, 0, 0].detach().cpu().numpy()   # (N,)

                # ── 2. L1 energy projection ─────────────────────────────────
                x_next = _project_energy(y, self.E_budget, _bounds)
                trajectory[t] = x_next

                # ── 3. Feed constrained state back into context ─────────────
                # Replace the GNN's prediction tensor with the constrained state
                # so the next step's history reflects what actually happened.
                constrained_tensor = torch.from_numpy(
                    x_next[:, np.newaxis, np.newaxis]   # [N, 1, 1]
                ).to(pred.device).float()
                context_pred = {modality: constrained_tensor}
                context = _advance_context(context, context_pred)

        return trajectory, times

    def __repr__(self) -> str:
        return (
            f"EnergyConstrainedSimulator("
            f"E_budget={self.E_budget:.4f}, "
            f"n_regions={self.n_regions}, "
            f"modality={self.modality!r})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 16 helper: energy budget analysis (no scan, no extra model calls)
# ─────────────────────────────────────────────────────────────────────────────

def run_energy_budget_analysis(
    trajectories: np.ndarray,
    state_bounds=None,
    output_dir: Optional[Path] = None,
) -> Dict:
    """
    从已有自由动力学轨迹统计能量分布，为 ``--energy-budget`` 提供参考。

    *不需要额外的模型调用*——直接从步骤 3 的轨迹计算。

    能量定义
    --------
    E(t) = mean(|x(t)|)，即当前状态的平均绝对活动值。

    对于 [0,1] 归一化数据：典型 E* ≈ 0.2–0.5。
    对于 z-scored 数据（state_bounds=None）：典型 E* ≈ 0.4–0.8（|z-score| 均值）。

    建议的 E_budget 参考值（相对于 E*）
    ------------------------------------
    - tight_constraint (0.4 × E*)：强约束，~80% 激活被稀疏化，系统趋向固定点
    - moderate_constraint (0.7 × E*)：中等约束，约 50% 时间约束激活
    - natural (1.0 × E*)：无实际约束（E_budget 等于典型能量，刚好在边界）
    - relaxed (1.3 × E*)：放宽约束，几乎不影响动力学

    设计原则（为什么不需要"扫描"）
    --------------------------------
    能量约束实验的正确流程是：
      1. 运行本函数，获得 E* 和建议值
      2. 选择不同的 E_budget（tight / moderate / natural）
      3. 每次以 ``--energy-budget X`` 重新运行完整管线
      4. 比较不同 E_budget 下的 Lyapunov λ 和 PCA 轨迹

    这等价于真正的"扫描"，但各点结果保存在独立目录，便于对比分析。
    在单次管线运行中做扫描（调用 n_alpha × n_init 次 rollout）的方法
    本质上是重复执行多个完整管线，效率低且难以可视化。

    Args:
        trajectories: 自由动力学轨迹，shape (n_traj, T, N)。
        state_bounds: (lo, hi) 或 None；None 表示 z-scored 无界数据。
        output_dir:   结果保存目录；None → 不保存。

    Returns:
        dict 包含：
          E_mean               — 所有轨迹的平均能量 E*
          E_std                — 轨迹间能量标准差
          E_median             — 能量中位数
          E_per_region         — 各脑区平均能量，shape (N,)
          recommended_budgets  — 建议值字典
          n_traj, n_steps_used, burnin
    """
    n_traj, T, N = trajectories.shape
    burnin = max(10, T // 10)
    traj_valid = trajectories[:, burnin:, :]   # (n_traj, T-burnin, N)

    # Per-trajectory mean energy
    E_traj = np.abs(traj_valid).mean(axis=(1, 2))   # (n_traj,)
    E_mean   = float(E_traj.mean())
    E_std    = float(E_traj.std())
    E_median = float(np.median(E_traj))

    # Per-region mean energy
    E_per_region = np.abs(traj_valid).mean(axis=(0, 1))   # (N,)

    recommended = {
        "tight_constraint":    round(_TIGHT_FACTOR    * E_mean, 6),
        "moderate_constraint": round(_MODERATE_FACTOR * E_mean, 6),
        "natural":             round(_NATURAL_FACTOR  * E_mean, 6),
        "relaxed":             round(_RELAXED_FACTOR  * E_mean, 6),
    }

    logger.info(
        "能量分析: E* = %.4f ± %.4f（中位数 %.4f）",
        E_mean, E_std, E_median,
    )
    logger.info(
        "  建议 E_budget: 紧=%.4f, 中等=%.4f, 自然=%.4f",
        recommended["tight_constraint"],
        recommended["moderate_constraint"],
        recommended["natural"],
    )

    result: Dict = {
        "E_mean":              E_mean,
        "E_std":               E_std,
        "E_median":            E_median,
        "E_per_region":        E_per_region.tolist(),
        "recommended_budgets": recommended,
        "n_traj":              n_traj,
        "n_steps_used":        int(T - burnin),
        "burnin":              burnin,
        "state_bounds":        list(state_bounds) if state_bounds is not None else None,
    }

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        report_path = out / "energy_budget_report.json"
        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, ensure_ascii=False)
        logger.info("  → 保存能量预算报告: %s", report_path)
        _try_plot_energy_budget(result, out)

    return result


def _try_plot_energy_budget(result: Dict, output_dir: Path) -> None:
    """绘制每脑区能量分布柱状图和建议 E_budget 参考线。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    E_per_region = np.array(result["E_per_region"])
    recommended  = result["recommended_budgets"]
    E_mean       = result["E_mean"]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(len(E_per_region)), E_per_region, color="steelblue", alpha=0.7, width=1.0)

    # Derive colors from a fixed palette; fall back to a default colour for
    # any key not listed, so the plot stays correct even if key names change.
    _palette = ["red", "orange", "green", "purple", "brown", "grey"]
    for idx, (label, val) in enumerate(recommended.items()):
        color = _palette[idx % len(_palette)]
        ax.axhline(val, color=color, linestyle="--", lw=1.2,
                   label=f"E_budget={val:.4f} ({label})")

    ax.set_xlabel("Brain Region Index")
    ax.set_ylabel("Mean |activity|")
    ax.set_title(
        f"Energy Distribution per Region\n"
        f"E* = {E_mean:.4f} ± {result['E_std']:.4f}  |  "
        f"N={len(E_per_region)} regions, {result['n_steps_used']} steps"
    )
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    save_path = output_dir / "energy_budget.png"
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("  → 保存能量预算图: %s", save_path)
