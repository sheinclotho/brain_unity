"""
Virtual Stimulation Experiment
===============================

模拟对指定脑区施加连续刺激，并观察系统的响应与恢复过程。

刺激定义（见计划书 §7, §10）：
  u(t) = A · sin(2πft)   (或方波 / 阶跃）

刺激 **必须为连续输入**，禁止单时间点脉冲。

模拟流程：
  baseline dynamics  （pre_steps 步，无刺激）
  ↓
  apply stimulus     （stim_steps 步，施加连续刺激）
  ↓
  remove stimulus    （post_steps 步，观察恢复）

输出：每段轨迹的 numpy 数组，供稳定性分析和可视化使用。
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    RampStimulus,
    SinStimulus,
    SquareWaveStimulus,
    Stimulus,
    StepStimulus,
)

logger = logging.getLogger(__name__)


# ── Data containers ───────────────────────────────────────────────────────────

@dataclass
class StimulationResult:
    """Container for a single virtual stimulation experiment result."""

    target_node: int
    stimulus: Stimulus
    pre_trajectory: np.ndarray      # shape (pre_steps, n_regions)
    stim_trajectory: np.ndarray     # shape (stim_steps, n_regions)
    post_trajectory: np.ndarray     # shape (post_steps, n_regions)

    @property
    def full_trajectory(self) -> np.ndarray:
        """Concatenated trajectory across all three phases."""
        return np.concatenate(
            [self.pre_trajectory, self.stim_trajectory, self.post_trajectory],
            axis=0,
        )

    @property
    def peak_response(self) -> np.ndarray:
        """Peak activity in each region during the stimulation phase."""
        return self.stim_trajectory.max(axis=0)

    @property
    def mean_response(self) -> np.ndarray:
        """Mean activity in each region during the stimulation phase."""
        return self.stim_trajectory.mean(axis=0)

    @property
    def recovery_time(self) -> Optional[int]:
        """
        Estimate steps to recovery: first post-stim step where all regions
        are within 0.01 of their pre-stim mean.
        """
        baseline_mean = self.pre_trajectory.mean(axis=0)
        for t, state in enumerate(self.post_trajectory):
            if np.max(np.abs(state - baseline_mean)) < 0.01:
                return t
        return None  # Not recovered within post_steps


def _build_stimulus(
    pattern: str,
    node: int,
    amplitude: float,
    frequency: float,
    stim_steps: int,
    onset: int,
    dt: float,
) -> Stimulus:
    """Factory function — create the appropriate Stimulus subclass."""
    pattern = pattern.lower()
    if pattern == "sine":
        return SinStimulus(
            node=node,
            freq=frequency,
            amp=amplitude,
            duration=stim_steps,
            onset=onset,
        )
    elif pattern == "square":
        return SquareWaveStimulus(
            node=node,
            freq=frequency,
            amp=amplitude,
            duration=stim_steps,
            onset=onset,
            dt=dt,
        )
    elif pattern == "step":
        return StepStimulus(
            node=node,
            amp=amplitude,
            duration=stim_steps,
            onset=onset,
        )
    elif pattern == "ramp":
        return RampStimulus(
            node=node,
            amplitude=amplitude,
            duration=stim_steps,
            onset=onset,
        )
    else:
        raise ValueError(
            f"未知刺激模式: '{pattern}'。支持: sine, square, step, ramp"
        )


# ── Public API ─────────────────────────────────────────────────────────────────

def run_stimulation(
    simulator: BrainDynamicsSimulator,
    node: int,
    amplitude: float = 0.5,
    frequency: float = 10.0,
    stim_steps: int = 200,
    pre_steps: int = 100,
    post_steps: int = 200,
    pattern: str = "sine",
    x0: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> StimulationResult:
    """
    对单个脑区施加连续刺激，并观察三阶段动力学。

    Args:
        simulator:   BrainDynamicsSimulator 实例。
        node:        目标脑区索引（0-indexed）。
        amplitude:   刺激幅度（归一化，0–1）。
        frequency:   刺激频率（Hz，用于 sine / square 模式）。
        stim_steps:  刺激持续步数。
        pre_steps:   刺激前基线步数。
        post_steps:  刺激后恢复步数。
        pattern:     刺激模式（sine / square / step / ramp）。
        x0:          初始脑状态；None → 随机采样。
        rng:         随机数生成器（用于随机初始状态）。

    Returns:
        StimulationResult 数据容器。
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if x0 is None:
        x0 = rng.random(simulator.n_regions).astype(np.float32)

    # Build stimulus (onset = pre_steps, active during stim phase)
    stim = _build_stimulus(
        pattern=pattern,
        node=node,
        amplitude=amplitude,
        frequency=frequency,
        stim_steps=stim_steps,
        onset=pre_steps,
        dt=simulator.dt,
    )

    total_steps = pre_steps + stim_steps + post_steps

    # Run full rollout with stimulus active during [pre_steps, pre_steps+stim_steps)
    full_traj, _ = simulator.rollout(x0=x0, steps=total_steps, stimulus=stim)

    pre_traj = full_traj[:pre_steps]
    stim_traj = full_traj[pre_steps: pre_steps + stim_steps]
    post_traj = full_traj[pre_steps + stim_steps:]

    return StimulationResult(
        target_node=node,
        stimulus=stim,
        pre_trajectory=pre_traj,
        stim_trajectory=stim_traj,
        post_trajectory=post_traj,
    )


def run_virtual_stimulation(
    simulator: BrainDynamicsSimulator,
    target_nodes: List[int],
    amplitude: float = 0.5,
    frequency: float = 10.0,
    stim_steps: int = 200,
    pre_steps: int = 100,
    post_steps: int = 200,
    patterns: List[str] = ("sine",),
    seed: int = 0,
    output_dir: Optional[Path] = None,
) -> Dict[str, List[StimulationResult]]:
    """
    对多个脑区 × 多种刺激模式运行虚拟刺激实验。

    Args:
        simulator:    BrainDynamicsSimulator 实例。
        target_nodes: 目标脑区索引列表。
        amplitude:    刺激幅度。
        frequency:    刺激频率（Hz）。
        stim_steps:   刺激步数。
        pre_steps:    基线步数。
        post_steps:   恢复步数。
        patterns:     刺激模式列表。
        seed:         随机种子。
        output_dir:   保存轨迹的目录；None → 不保存。

    Returns:
        results_by_pattern: Dict[pattern, List[StimulationResult]]
    """
    rng = np.random.default_rng(seed)
    results_by_pattern: Dict[str, List[StimulationResult]] = {}

    total = len(patterns) * len(target_nodes)
    done = 0

    for pattern in patterns:
        results_by_pattern[pattern] = []
        for node in target_nodes:
            x0 = rng.random(simulator.n_regions).astype(np.float32)
            result = run_stimulation(
                simulator=simulator,
                node=node,
                amplitude=amplitude,
                frequency=frequency,
                stim_steps=stim_steps,
                pre_steps=pre_steps,
                post_steps=post_steps,
                pattern=pattern,
                x0=x0,
                rng=rng,
            )
            results_by_pattern[pattern].append(result)
            done += 1
            logger.debug(
                "  [%d/%d] pattern=%s node=%d peak=%.4f",
                done,
                total,
                pattern,
                node,
                result.peak_response[node],
            )

    logger.info(
        "✓ 虚拟刺激实验完成: %d 个模式 × %d 个节点",
        len(patterns),
        len(target_nodes),
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for pattern, res_list in results_by_pattern.items():
            for res in res_list:
                fname = (
                    output_dir
                    / f"stim_traj_{pattern}_node{res.target_node}.npy"
                )
                np.save(fname, res.full_trajectory)
        logger.info("  → 刺激轨迹已保存: %s", output_dir)

    return results_by_pattern
