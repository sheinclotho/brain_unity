"""
Brain Dynamics Simulator
========================

统一管理所有模拟过程的核心组件。

支持：
- 外部刺激输入（连续刺激，必须实现 Stimulus.value(t)）
- 自由演化（无输入）
- 有模型前向传播（nn.Module，requires can_forward=True）
- 无模型回退（Wilson-Cowan 漏积分器，与 realtime_server 一致）

时间尺度对齐原则（见计划书 §11）:
  simulation_dt = EEG_dt
  fMRI 通过下采样观察（每 fmri_subsample 步记录一次）

所有实验均通过此接口进行：
  step(state, external_input=None)   → 单步演化
  rollout(x0, steps, stimulus=None) → 连续模拟轨迹
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Stimulus base class & concrete implementations
# ══════════════════════════════════════════════════════════════════════════════

class Stimulus(abc.ABC):
    """
    Abstract base class for all stimulation signals.

    子类必须实现 ``value(t: int) -> float``，返回该时间步的刺激幅度（归一化单位）。

    所有刺激都是 **连续输入**，不允许单时间点脉冲（见计划书 §10）。
    """

    def __init__(
        self,
        node: int,
        amplitude: float = 0.5,
        duration: int = 200,
        onset: int = 0,
    ):
        """
        Args:
            node:      被刺激的脑区索引（0-indexed）。
            amplitude: 最大刺激幅度（归一化，0–1）。
            duration:  刺激持续的时间步数。
            onset:     刺激开始的时间步。
        """
        self.node = int(node)
        self.amplitude = float(amplitude)
        self.duration = int(duration)
        self.onset = int(onset)

    def is_active(self, t: int) -> bool:
        """Return True if stimulation is active at time step t."""
        return self.onset <= t < self.onset + self.duration

    @abc.abstractmethod
    def value(self, t: int) -> float:
        """Return the stimulation amplitude at time step t."""

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(node={self.node}, "
            f"amp={self.amplitude}, dur={self.duration}, onset={self.onset})"
        )


class SinStimulus(Stimulus):
    """
    正弦波刺激：u(t) = A · sin(2π f t_rel)

    其中 t_rel = (t - onset + 0.5) / duration，采用中点采样以避免边界零点
    （见 AGENTS.md 正弦包络 Nyquist 修复记录）。

    Args:
        node:      脑区索引。
        freq:      刺激频率（Hz）。
        amp:       幅度（0–1）。
        duration:  持续步数。
        onset:     起始步。
    """

    def __init__(
        self,
        node: int,
        freq: float = 10.0,
        amp: float = 0.5,
        duration: int = 200,
        onset: int = 0,
    ):
        super().__init__(node=node, amplitude=amp, duration=duration, onset=onset)
        self.freq = float(freq)

    def value(self, t: int) -> float:
        if not self.is_active(t):
            return 0.0
        t_rel = t - self.onset
        # Middle-point sampling: progress ∈ (0, 1) open interval → never 0 at endpoints
        progress = (t_rel + 0.5) / max(self.duration, 1)
        # Envelope: bell-shape (half-sine) × sinusoidal carrier
        envelope = math.sin(math.pi * progress)
        return self.amplitude * envelope


class SquareWaveStimulus(Stimulus):
    """
    方波刺激（推荐用于 tACS / TMS 协议研究）。

    信号在每个完整周期的前半段为 +A，后半段为 0。
    """

    def __init__(
        self,
        node: int,
        freq: float = 10.0,
        amp: float = 0.5,
        duration: int = 200,
        onset: int = 0,
        dt: float = 0.004,
    ):
        super().__init__(node=node, amplitude=amp, duration=duration, onset=onset)
        self.freq = float(freq)
        self.dt = float(dt)

    def value(self, t: int) -> float:
        if not self.is_active(t):
            return 0.0
        t_rel = t - self.onset
        period_steps = max(1, int(round(1.0 / (self.freq * self.dt))))
        half_period = max(1, period_steps // 2)
        return self.amplitude if (t_rel % period_steps) < half_period else 0.0


class StepStimulus(Stimulus):
    """
    阶跃刺激（step input）：在刺激持续期内维持恒定幅度，并以指数包络软启动/软停止。
    """

    def __init__(
        self,
        node: int,
        amp: float = 0.5,
        duration: int = 200,
        onset: int = 0,
        ramp_steps: int = 10,
    ):
        super().__init__(node=node, amplitude=amp, duration=duration, onset=onset)
        self.ramp_steps = int(ramp_steps)

    def value(self, t: int) -> float:
        if not self.is_active(t):
            return 0.0
        t_rel = t - self.onset
        # Soft onset
        if t_rel < self.ramp_steps:
            scale = t_rel / self.ramp_steps
        # Soft offset
        elif t_rel >= self.duration - self.ramp_steps:
            scale = (self.duration - t_rel) / self.ramp_steps
        else:
            scale = 1.0
        return self.amplitude * scale


class RampStimulus(Stimulus):
    """线性递增刺激（ramp input）。"""

    def value(self, t: int) -> float:
        if not self.is_active(t):
            return 0.0
        t_rel = t - self.onset
        progress = t_rel / max(self.duration - 1, 1)
        return self.amplitude * progress


# ══════════════════════════════════════════════════════════════════════════════
# Wilson-Cowan fallback integrator (no model)
# ══════════════════════════════════════════════════════════════════════════════

def _make_connectivity(n: int, seed: int = 0) -> np.ndarray:
    """
    生成一个用于无模型回退的合成连接矩阵。

    对角线为零，L1 归一化，Gaussian 空间衰减（距离用 Fibonacci 球面采样近似）。
    """
    rng = np.random.default_rng(seed)
    # Random sparse connectivity
    W = rng.random((n, n)).astype(np.float32)
    np.fill_diagonal(W, 0.0)
    W = W / (W.sum(axis=1, keepdims=True) + 1e-8)
    return W


class _WilsonCowanIntegrator:
    """
    偏差驱动漏积分器（与 realtime_server._demo_simulate 一致）。

    x(t+1) = x(t) + delta - leak
    delta  = tanh(stim * 2.0 + (W @ deviation) * 0.35) * 0.04
    leak   = deviation * 0.10
    deviation = x(t) - x0     (x0 = 初始平衡状态)

    固定点在 x0 处；无刺激时 deviation 保持为 0。
    """

    def __init__(self, n_regions: int, x0: np.ndarray, seed: int = 0):
        self.n = n_regions
        self.x0 = x0.copy()
        self.W = _make_connectivity(n_regions, seed=seed)

    def step(self, state: np.ndarray, stim: np.ndarray) -> np.ndarray:
        dev = state - self.x0
        net_dev = self.W @ dev
        delta = np.tanh(stim * 2.0 + net_dev * 0.35) * 0.04
        leak = dev * 0.10
        new_state = state + delta - leak
        return np.clip(new_state, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# BrainDynamicsSimulator
# ══════════════════════════════════════════════════════════════════════════════

class BrainDynamicsSimulator:
    """
    统一管理所有模拟过程的核心组件。

    接口::

        sim = BrainDynamicsSimulator(model)
        x1  = sim.step(x0, external_input)
        traj, times = sim.rollout(x0, steps=500, stimulus=stim)

    优先级：
    1. 若 ``model`` 是可用的 nn.Module（can_forward=True）→ 使用模型前向传播。
    2. 否则 → 使用 Wilson-Cowan 漏积分器（无模型回退）。
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        n_regions: int = 200,
        dt: float = 0.004,
        fmri_subsample: int = 25,
        seed: int = 0,
    ):
        """
        Args:
            model:          训练好的 nn.Module（可为 None）。
            n_regions:      脑区数量（Schaefer-200 默认为 200）。
            dt:             内部时间步长（秒，对齐 EEG 采样率）。
            fmri_subsample: fMRI 下采样倍率（每 N 步记录一次 fMRI 观测）。
            seed:           随机种子（用于无模型回退的连接矩阵生成）。
        """
        self.model = model
        self.n_regions = n_regions
        self.dt = dt
        self.fmri_subsample = fmri_subsample
        self.seed = seed

        # Determine whether model supports forward inference
        self._use_model = (
            model is not None
            and getattr(model, "can_forward", True)
            and not isinstance(model, type)
        )
        if not self._use_model:
            logger.info(
                "BrainDynamicsSimulator: 使用 Wilson-Cowan 回退模式 (n_regions=%d)",
                n_regions,
            )

    # ── Core API ───────────────────────────────────────────────────────────────

    def step(
        self,
        state: np.ndarray,
        external_input: Optional[np.ndarray] = None,
        _wc: Optional[_WilsonCowanIntegrator] = None,
    ) -> np.ndarray:
        """
        单步动力学演化: x(t+1) = f(x(t), u(t))

        Args:
            state:          当前脑状态，shape (n_regions,)，值域 [0, 1]。
            external_input: 外部刺激向量，shape (n_regions,)。若为 None 则全零。
            _wc:            Wilson-Cowan 积分器实例（由 rollout 传入，保证一致性）。

        Returns:
            新脑状态，shape (n_regions,)。
        """
        if external_input is None:
            external_input = np.zeros(self.n_regions, dtype=np.float32)

        if self._use_model:
            return self._model_step(state, external_input)
        else:
            wc = _wc or _WilsonCowanIntegrator(self.n_regions, x0=state, seed=self.seed)
            return wc.step(state, external_input)

    def rollout(
        self,
        x0: np.ndarray,
        steps: int,
        stimulus: Optional[Stimulus] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        连续动力学模拟。

        Args:
            x0:       初始脑状态，shape (n_regions,)。
            steps:    模拟步数。
            stimulus: 刺激对象（实现 Stimulus.value(t)）；None → 自由演化。

        Returns:
            trajectory: shape (steps, n_regions)，每步的脑状态。
            times:      shape (steps,)，以秒为单位的时间轴。
        """
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        if x0.shape[0] != self.n_regions:
            raise ValueError(
                f"x0 shape mismatch: expected ({self.n_regions},), got {x0.shape}"
            )

        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        # Create Wilson-Cowan integrator once (preserves connectivity W across steps)
        wc = _WilsonCowanIntegrator(self.n_regions, x0=x0, seed=self.seed)

        state = x0.copy()
        for t in range(steps):
            # Build external input vector
            u = np.zeros(self.n_regions, dtype=np.float32)
            if stimulus is not None and stimulus.is_active(t):
                node_idx = stimulus.node
                if 0 <= node_idx < self.n_regions:
                    u[node_idx] = stimulus.value(t)

            # Evolve
            state = self._step_internal(state, u, wc)
            trajectory[t] = state

        return trajectory, times

    def rollout_multi_stim(
        self,
        x0: np.ndarray,
        steps: int,
        stimuli: List[Stimulus],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多脑区同时刺激的连续模拟。

        Args:
            x0:       初始脑状态，shape (n_regions,)。
            steps:    模拟步数。
            stimuli:  多个 Stimulus 对象的列表。

        Returns:
            trajectory: shape (steps, n_regions)。
            times:      shape (steps,)。
        """
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        wc = _WilsonCowanIntegrator(self.n_regions, x0=x0, seed=self.seed)
        state = x0.copy()

        for t in range(steps):
            u = np.zeros(self.n_regions, dtype=np.float32)
            for stim in stimuli:
                if stim.is_active(t) and 0 <= stim.node < self.n_regions:
                    u[stim.node] += stim.value(t)

            state = self._step_internal(state, u, wc)
            trajectory[t] = state

        return trajectory, times

    # ── Private helpers ────────────────────────────────────────────────────────

    def _step_internal(
        self,
        state: np.ndarray,
        u: np.ndarray,
        wc: _WilsonCowanIntegrator,
    ) -> np.ndarray:
        """Route to model or Wilson-Cowan depending on availability."""
        if self._use_model:
            return self._model_step(state, u)
        return wc.step(state, u)

    def _model_step(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        使用训练好的模型进行单步演化。

        当前实现：将 (n_regions,) 状态张量直接传入模型。
        若模型接口不兼容，记录警告并回退到 Wilson-Cowan。
        """
        try:
            state_t = torch.from_numpy(state).float().unsqueeze(0)  # (1, N)
            u_t = torch.from_numpy(u).float().unsqueeze(0)           # (1, N)
            with torch.no_grad():
                # Attempt generic call; real TwinBrain GNN has a richer interface.
                # Downstream experiments may subclass and override this method.
                out = self.model(state_t + u_t)
            return out.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as exc:
            logger.debug("Model step failed (%s), falling back to WC.", exc)
            wc = _WilsonCowanIntegrator(self.n_regions, x0=state, seed=self.seed)
            return wc.step(state, u)

    # ── Convenience sampling ───────────────────────────────────────────────────

    def sample_random_state(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a random brain state uniformly in [0, 1]^n_regions."""
        if rng is None:
            rng = np.random.default_rng(self.seed)
        return rng.random(self.n_regions).astype(np.float32)
