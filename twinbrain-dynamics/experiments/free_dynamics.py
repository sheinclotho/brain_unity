"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 确定可用的时序窗口数（base_graph 时间维度 // context_length）
2. 采样 n_init 个随机初始状态
3. 轨迹 i 使用窗口 i % n_windows 作为历史上下文（不同历史片段）
4. 将 x0 注入选定上下文的最后一个时间步
5. 运行 steps 步自回归预测，记录轨迹

输出文件：outputs/trajectories.npy

**时序窗口机制（解决 Wolf 上下文稀释偏差）**：

  以前所有轨迹共享同一个历史上下文（base_graph 最后 context_length 步），
  只有最后 1 步被不同 x0 覆盖。注意力机制将这 1 步扰动稀释在 L-1 个相同历史中，
  导致所有轨迹几乎相同（Wolf std ≈ 1.85e-05，见 AGENTS.md §Wolf上下文稀释偏差）。

  当 base_graph 时间维度 T ≥ 2 × context_length 时，本模块自动将轨迹分配到
  n_windows = T // context_length 个非重叠历史片段：

    窗口 0：x[:, T-L:T, :]          （最近历史，原有行为）
    窗口 1：x[:, T-2L:T-L, :]       （较早历史）
    窗口 k：x[:, T-(k+1)L:T-kL, :]

  每条轨迹都从真正不同的历史出发，从根本上解决上下文稀释问题。
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

    当 ``base_graph`` 时间维度 ``T > context_length`` 时，可将轨迹分配到
    ``n_windows = T // context_length`` 个非重叠历史窗口：

      轨迹 i 使用窗口 ``i % n_windows``，即 ``x[:, T-(k+1)L:T-kL, :]``。

    这样不同轨迹从真正不同的历史片段出发，显著提升初始多样性，使
    Rosenstein LLE 和 Wolf LLE 在不同轨迹间有实质性差异。

    ``n_temporal_windows=None``（默认）：自动使用 ``simulator.n_temporal_windows``
    （图缓存时间维度所支持的最大非重叠窗口数）。
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

    # ── Determine effective number of temporal windows ────────────────────────
    max_available = simulator.n_temporal_windows
    if n_temporal_windows is None:
        eff_windows = max_available
    else:
        eff_windows = min(int(n_temporal_windows), max_available)
        if int(n_temporal_windows) > max_available:
            logger.warning(
                "  ⚠  n_temporal_windows=%d 超过图缓存支持的最大窗口数 %d"
                "（需要 T_total ≥ n_windows × context_length），实际使用 %d 个窗口。",
                n_temporal_windows, max_available, eff_windows,
            )

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, 预计输出大小=%.1f MiB",
        n_init, steps, n_regions, est_mb,
    )

    if eff_windows > 1:
        logger.info(
            "  时序窗口: 使用 %d 个不同历史上下文窗口（轨迹 i → 窗口 i %% %d）。\n"
            "  每个窗口对应 base_graph 时间序列的一段独立历史，从根本上解决\n"
            "  '所有轨迹共享相同 context_length-1 历史步' 的上下文稀释问题。",
            eff_windows, eff_windows,
        )
    else:
        logger.info(
            "  时序窗口: 仅使用 1 个上下文窗口（图缓存时间序列长度 ≤ 2 × context_length）。\n"
            "  要启用多窗口轨迹多样性，需要更长的图缓存时间序列"
            "（T_total ≥ 2 × context_length）。"
            "增大 free_dynamics.n_temporal_windows 配置值不能弥补数据不足。",
        )
        logger.info(
            "  初始条件: %d 个不同的随机初始状态（均匀采样 Uniform[0,1]^%d），"
            "每条轨迹注入为初始上下文的最后一个时间步。",
            n_init, n_regions,
        )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    log_interval = max(1, n_init // 10)
    for i in range(n_init):
        window_idx = i % eff_windows
        x0 = rng.random(n_regions).astype(np.float32)
        traj, _ = simulator.rollout(
            x0=x0, steps=steps, stimulus=None,
            context_window_idx=window_idx,
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
