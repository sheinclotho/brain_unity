"""
Response Matrix Computation
============================

量化 **刺激传播结构** (Stimulation Propagation Structure)。

定义：
  R[i, j] = response of node j when node i is stimulated

计算步骤：
  for each node i:
      apply continuous stimulus to node i
      measure mean activity change of all nodes j during stimulation window
      R[i, j] = mean(stim_activity[j]) - mean(baseline_activity[j])

输出文件：outputs/response_matrix.npy
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np

import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator.brain_dynamics_simulator import BrainDynamicsSimulator, SinStimulus
from experiments.virtual_stimulation import run_stimulation

logger = logging.getLogger(__name__)


def compute_response_matrix(
    simulator: BrainDynamicsSimulator,
    n_nodes: Optional[int] = None,
    stim_amplitude: float = 0.5,
    stim_duration: int = 50,
    stim_frequency: float = 10.0,
    stim_pattern: str = "sine",
    measure_window: int = 20,
    pre_steps: int = 50,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    计算完整的刺激响应矩阵 R[i, j]。

    Args:
        simulator:       BrainDynamicsSimulator 实例。
        n_nodes:         要刺激的节点数（None → 使用 simulator.n_regions）。
        stim_amplitude:  刺激幅度。
        stim_duration:   每个节点的刺激步数。
        stim_frequency:  刺激频率（Hz）。
        stim_pattern:    刺激模式（sine / square / step / ramp）。
        measure_window:  刺激期间用于平均响应的时间窗口（步）。
        pre_steps:       基线步数（用于计算基线活动均值）。
        seed:            随机种子。
        output_dir:      保存 response_matrix.npy；None → 不保存。

    Returns:
        R: shape (n_nodes, simulator.n_regions)，归一化响应矩阵。
    """
    if n_nodes is None:
        n_nodes = simulator.n_regions

    R = np.zeros((n_nodes, simulator.n_regions), dtype=np.float32)
    rng = np.random.default_rng(seed)

    # Sample ONE shared initial state for all rows.
    # Using the same x0 (hence the same equilibrium) for every row ensures that
    # R[i,j] reflects only the stimulus-propagation structure (W column i), not
    # the interaction between different random equilibria and stimulation.
    # Previously each row used a fresh x0, which caused row norms to vary
    # dramatically whenever the stimulated node's equilibrium sat near the [0,1]
    # boundary (tanh saturation), making the matrix non-comparable across rows.
    x0 = rng.random(simulator.n_regions).astype(np.float32)

    logger.info(
        "响应矩阵计算: n_nodes=%d, stim_duration=%d, pattern=%s",
        n_nodes,
        stim_duration,
        stim_pattern,
    )

    for i in range(n_nodes):
        result = run_stimulation(
            simulator=simulator,
            node=i,
            amplitude=stim_amplitude,
            frequency=stim_frequency,
            stim_steps=stim_duration,
            pre_steps=pre_steps,
            post_steps=0,
            pattern=stim_pattern,
            x0=x0,
        )

        # Baseline: mean across pre-stimulation phase
        baseline = result.pre_trajectory.mean(axis=0)

        # Response: mean across measurement window of stimulation phase
        win_traj = result.stim_trajectory[:measure_window]
        response = win_traj.mean(axis=0) - baseline

        R[i] = response

        if (i + 1) % max(1, n_nodes // 10) == 0:
            logger.info("  %d/%d 节点完成", i + 1, n_nodes)

    col_mean = np.abs(R).mean(axis=0)
    stim_specificity = R.std(axis=1)

    logger.info(
        "✓ 响应矩阵完成。  全局均值=%.4f  最大绝对值=%.4f\n"
        "     列均值 mean=%.4f std=%.4f  (hub节点 ±2σ: %d个)\n"
        "     刺激特异性 mean=%.4f std=%.4f  (低特异性节点: %d个)",
        float(R.mean()),
        float(np.abs(R).max()),
        float(col_mean.mean()),
        float(col_mean.std()),
        int((col_mean > col_mean.mean() + 2 * col_mean.std()).sum()),
        float(stim_specificity.mean()),
        float(stim_specificity.std()),
        int((stim_specificity < stim_specificity.mean() * 0.5).sum()),
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "response_matrix.npy"
        np.save(out_path, R)
        logger.info("  → 已保存: %s", out_path)

    return R
