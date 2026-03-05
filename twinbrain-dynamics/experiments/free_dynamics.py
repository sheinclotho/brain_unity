"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 采样 n_init 个随机初始状态
2. 从每个初始状态进行长时间自由演化（steps 步）
3. 记录并返回所有状态轨迹

输出文件：outputs/trajectories.npy
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# Allow running as a standalone script or as an imported module
import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator.brain_dynamics_simulator import BrainDynamicsSimulator

logger = logging.getLogger(__name__)


def run_free_dynamics(
    simulator: BrainDynamicsSimulator,
    n_init: int = 200,
    steps: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    运行自由动力学实验。

    Args:
        simulator:   BrainDynamicsSimulator 实例。
        n_init:      随机初始状态数量（默认 200）。
        steps:       每条轨迹的模拟步数（默认 1000）。
        seed:        随机种子，确保可重复性。
        output_dir:  若指定，将结果保存为 trajectories.npy；
                     None → 不保存。

    Returns:
        trajectories: shape (n_init, steps, n_regions)，所有轨迹。
    """
    rng = np.random.default_rng(seed)
    n_regions = simulator.n_regions

    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d",
        n_init,
        steps,
        n_regions,
    )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)

    for i in range(n_init):
        x0 = rng.random(n_regions).astype(np.float32)
        traj, _ = simulator.rollout(x0=x0, steps=steps, stimulus=None)
        trajectories[i] = traj

        if (i + 1) % max(1, n_init // 10) == 0:
            logger.info("  %d/%d 初始状态完成", i + 1, n_init)

    logger.info("✓ 自由动力学实验完成，轨迹数组形状: %s", trajectories.shape)

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
