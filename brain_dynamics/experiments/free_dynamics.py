"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 确定可用的随机起始位置数量
2. 为每条轨迹随机采样一个等长上下文窗口（context_start ∈ [0, T - ctx_len]）
3. x0 = 上下文末步对应的数据值（上下文 + x0 在时序上连续）
4. 运行 steps 步自回归预测，记录轨迹

输出文件：outputs/trajectories.npy

**随机等长上下文窗口策略（替换旧的变长回退策略）**：

  所有轨迹的上下文长度完全相同（= context_length 步），区别仅在于起始位置。
  对每条轨迹 i，随机采样 context_start ∈ [0, max(0, T - ctx_len)]：

    轨迹 i → context_start_i = rand([0, T - ctx_len])
            → 上下文 = data[:, context_start_i : context_start_i + ctx_len]
            → x0     = data[:, context_start_i + ctx_len - 1]（时序连续）

  这与旧的"prediction_steps 步幅回退"策略有本质区别：
  旧策略产生 window0=T步, window1=T-s步, … 等 **不同长度** 的上下文，
  导致编码器的初始输入质量系统性不均等，干扰后续动力学分析。
  新策略确保所有轨迹经历相同质量的初始化。

  **当 T ≤ context_length 时**（如本次 T=200, ctx_len=200）：
  只有一个有效起始位置（context_start=0），上下文 = data[0:T]。
  100 条轨迹的多样性完全来自 x0 注入噪声（0.3σ 扰动）。
  在近临界系统（LLE ≈ 0）中，0.3σ 扰动足以驱散轨迹覆盖吸引子。
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Allow running as a standalone script or as an imported module
import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator.brain_dynamics_simulator import BrainDynamicsSimulator

logger = logging.getLogger(__name__)


def _estimate_memory_mb(n_init: int, steps: int, n_regions: int) -> float:
    """Estimate trajectory array size in MiB."""
    return n_init * steps * n_regions * 4 / (1024 * 1024)


def run_free_dynamics(
    simulator: BrainDynamicsSimulator,
    n_init: int = 200,
    steps: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    n_temporal_windows: Optional[int] = None,
) -> np.ndarray:
    """
    运行自由动力学实验（TwinBrainDigitalTwin 模式）。

    对每条轨迹：
      1. 从 Uniform[0,1]^n_regions 采样随机初始状态 x0
      2. 选择一个时序窗口作为初始上下文（见下文）
      3. 将 x0 注入该上下文的最后一个时间步
      4. 运行 steps 步自回归预测

    **时序窗口（n_temporal_windows）**：

    TwinBrainDigitalTwin 使用长度为 ``context_length`` 的历史窗口进行预测。
    若所有 n_init 条轨迹均使用同一个上下文窗口（仅最后一步被 x0 覆盖），
    则 context_length-1 步历史完全相同，模型对不同 x0 的响应被大量相同历史
    所稀释，导致轨迹在统计上几乎无法区分（Wolf std ≈ 1.85e-05，见 AGENTS.md）。

    本模块使用 **滑窗策略**（stride = context_length // 4，75% 重叠）从主模态
    时序中提取多个历史窗口。只要主模态时序 ``T > context_length + stride``
    （即 ``T > 1.25 × context_length``），即可得到 ≥2 个窗口：

      轨迹 i 使用窗口 ``i % n_windows``，即 ``x[:, T-L-k*s:T-k*s, :]``。

    与以往要求 ``T ≥ n_windows × context_length``（严格非重叠分块）不同，
    滑窗方案能从更短的时序数据中提取更多有效历史上下文，大幅降低多样性对数据
    长度的要求。

    **模态感知**：窗口数仅由 **主模态**（fmri/eeg，非所有模态的最小值）决定，
    避免次要模态短时序错误瓶颈主模态窗口计数。

    ``n_temporal_windows=None``（默认）：自动使用 ``simulator.n_temporal_windows``。
    ``n_temporal_windows=1``：禁用多窗口，仅使用最近一个历史窗口。

    Args:
        simulator:           BrainDynamicsSimulator 实例（TwinBrainDigitalTwin 模式）。
        n_init:              随机初始状态数量（默认 200）。
        steps:               每条轨迹的模拟步数（默认 1000）。
        seed:                随机种子，确保可重复性。
        output_dir:          若指定，将结果保存为 trajectories.npy；None → 不保存。
        device:              保留参数（兼容性），实际设备由模型决定。
        n_temporal_windows:  使用的时序窗口数（None = 自动，1 = 禁用多窗口）。

    Returns:
        trajectories: shape (n_init, steps, n_regions)，所有轨迹。
    """
    rng = np.random.default_rng(seed)
    n_regions = simulator.n_regions

    # ── Determine data dimensions ─────────────────────────────────────────────
    ctx_len = simulator._get_context_length()
    _nt = simulator.modality if simulator.modality != "joint" else "fmri"
    try:
        T_primary = (
            int(simulator.base_graph[_nt].x.shape[1])
            if _nt in simulator.base_graph.node_types
            and hasattr(simulator.base_graph[_nt], "x")
            else 0
        )
    except Exception:
        T_primary = 0

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, 预计输出大小=%.1f MiB",
        n_init, steps, n_regions, est_mb,
    )

    # ── Random equal-length context window strategy ───────────────────────────
    # All contexts have exactly min(ctx_len, T_primary) steps; the only
    # variation between trajectories is WHERE in the recording we start.
    # This avoids the systematic encoder-quality bias that the old
    # "prediction_steps fallback" introduced by using shorter and shorter
    # contexts for higher window indices.
    #
    # n_valid_starts = max(1, T_primary - ctx_len + 1):
    #   T >= ctx_len  →  uniform draw from [0, T - ctx_len]
    #                    (multiple distinct full-length windows possible)
    #   T < ctx_len   →  always start=0; diversity comes from x0 noise alone
    #
    # x0 alignment: x0_step = context_start + effective_ctx - 1
    # (last step of the selected context window).  Injecting the data value at
    # that step means context + x0 form a CONTIGUOUS BOLD segment → no
    # correction transient in the encoder output.
    if T_primary > 0:
        eff_ctx = min(ctx_len, T_primary)
        n_valid_starts = max(1, T_primary - eff_ctx + 1)
    else:
        eff_ctx = ctx_len
        n_valid_starts = 1

    if n_valid_starts > 1:
        logger.info(
            "  初始化策略: 随机等长上下文窗口"
            "（%d 条轨迹各自随机截取 %d 步历史，"
            "起始位置 ∈ [0, %d]，全部等长，无偏差）。\n"
            "  x0 = 上下文窗口末步对应的数据值（时序连续）。",
            n_init, eff_ctx, n_valid_starts - 1,
        )
    else:
        logger.info(
            "  初始化策略: 单一上下文窗口（主模态 '%s' T=%d ≤ context_length=%d）。\n"
            "  仅有一个等长上下文 [0:%d]，%d 条轨迹的多样性来自 x0 注入噪声。\n"
            "  提示: 若需要更多轨迹多样性，请提供更长的输入时序（T > %d）。",
            _nt, T_primary, ctx_len, eff_ctx, n_init, ctx_len,
        )

    # Warn if caller passed n_temporal_windows (parameter now ignored in favour
    # of the random-start strategy, but we log so the caller knows).
    if n_temporal_windows is not None and int(n_temporal_windows) != 1:
        logger.warning(
            "  n_temporal_windows=%d 参数已忽略：当前使用随机等长上下文策略，"
            "轨迹多样性由 context_start 随机采样而非固定窗口数控制。",
            n_temporal_windows,
        )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    log_interval = max(1, n_init // 10)

    for i in range(n_init):
        # Random context start position — equal length for all trajectories
        context_start = int(rng.integers(0, n_valid_starts))
        # x0 aligned to the last step of the selected context window.
        # context covers [context_start : context_start + eff_ctx]
        # so the last step index is context_start + eff_ctx - 1.
        x0_step = min(context_start + eff_ctx - 1, max(0, T_primary - 1)) if T_primary > 0 else None
        x0 = simulator.sample_random_state(rng, from_data=True, step_idx=x0_step)
        traj, _ = simulator.rollout(
            x0=x0, steps=steps, stimulus=None,
            context_start=context_start,
        )
        trajectories[i] = traj
        if (i + 1) % log_interval == 0:
            logger.info("  %d/%d 初始状态完成", i + 1, n_init)
        # Release cached GPU memory to prevent fragmentation across rollouts.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Log initial vs final state diversity to quantify convergence
    initial_div = float(np.std(trajectories[:, 0, :], axis=0).mean())
    final_div = float(np.std(trajectories[:, -1, :], axis=0).mean())
    logger.info(
        "✓ 自由动力学实验完成，轨迹形状: %s\n"
        "  轨迹多样性: 初始 std=%.4f → 终止 std=%.4f  "
        "(%s)",
        trajectories.shape,
        initial_div,
        final_div,
        "收敛（不同起点趋向相似行为）" if final_div < initial_div * 0.7 else
        "发散（轨迹相互分离）" if final_div > initial_div * 1.3 else
        "保持（多样性基本不变）",
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "trajectories.npy"
        np.save(out_path, trajectories)
        logger.info("  → 已保存: %s", out_path)

    return trajectories


def sample_random_state(
    n_regions: int,
    rng: Optional[np.random.Generator] = None,
    seed: int = 0,
) -> np.ndarray:
    """Sample a uniformly random brain state in [0, 1]^n_regions."""
    if rng is None:
        rng = np.random.default_rng(seed)
    return rng.random(n_regions).astype(np.float32)
