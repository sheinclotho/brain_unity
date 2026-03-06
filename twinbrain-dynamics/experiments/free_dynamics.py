"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 采样 n_init 个随机初始状态（注入为初始上下文的最后一步）
2. 从每个初始状态进行长时间自由演化（steps 步）
3. 记录并返回所有状态轨迹

输出文件：outputs/trajectories.npy

注意（twin 模式初始条件多样性）：
  每条轨迹的 x0 是不同的随机向量，注入为 base_graph 上下文的最后一个时间步。
  上下文窗口其余 context_length-1 步由所有轨迹共享。
  因此轨迹的"初始多样性"取决于模型对最后一步扰动的响应强度。
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
) -> np.ndarray:
    """
    运行自由动力学实验（TwinBrainDigitalTwin 模式）。

    对每条轨迹：
      1. 从 Uniform[0,1]^n_regions 采样随机初始状态 x0
      2. 将 x0 注入 base_graph 上下文的最后一个时间步
      3. 运行 steps 步自回归预测
    这样 n_init 条轨迹从不同的初始脑状态出发，但共享同一上下文历史。

    Args:
        simulator:   BrainDynamicsSimulator 实例（必须为 TwinBrainDigitalTwin 模式）。
        n_init:      随机初始状态数量（默认 200）。
        steps:       每条轨迹的模拟步数（默认 1000）。
        seed:        随机种子，确保可重复性。
        output_dir:  若指定，将结果保存为 trajectories.npy；None → 不保存。
        device:      保留参数（兼容性），实际设备由模型决定。

    Returns:
        trajectories: shape (n_init, steps, n_regions)，所有轨迹。
    """
    rng = np.random.default_rng(seed)
    n_regions = simulator.n_regions

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, 预计输出大小=%.1f MiB",
        n_init, steps, n_regions, est_mb,
    )
    logger.info(
        "  初始条件: %d 个不同的随机初始状态（均匀采样 Uniform[0,1]^%d），"
        "每条轨迹注入为初始上下文的最后一个时间步。",
        n_init, n_regions,
    )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    log_interval = max(1, n_init // 10)
    for i in range(n_init):
        x0 = rng.random(n_regions).astype(np.float32)
        traj, _ = simulator.rollout(x0=x0, steps=steps, stimulus=None)
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
