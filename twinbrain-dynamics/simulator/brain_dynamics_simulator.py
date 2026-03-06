"""
Brain Dynamics Simulator
========================

统一管理所有模拟过程的核心组件。

支持两种运行模式：

**模型模式**（推荐）
  ``BrainDynamicsSimulator(twin, base_graph)``
  使用 ``TwinBrainDigitalTwin`` 进行基于真实学习动力学的推断：
    - 自由演化：``twin.predict_future()`` 自回归滑窗预测
    - 受刺激演化：``twin.simulate_intervention()`` 潜空间扰动

**Wilson-Cowan 模式**（独立基线，非回退）
  ``BrainDynamicsSimulator(model=None, n_regions=N)``
  偏差驱动漏积分器，仅供基线对比与单元测试使用。
  当 ``model`` 被提供但无法前向推断时，**不会**自动切换到此模式，
  而是直接抛出 ``RuntimeError``。

时间尺度对齐原则（见计划书 §11）:
  - 模型模式：dt = 1 / fmri_sampling_rate（通常 2 s per TR）
  - WC 模式：dt = 0.004 s（EEG 采样率）

所有实验均通过此接口进行：
  rollout(x0, steps, stimulus)          → 连续模拟轨迹
  rollout_multi_stim(x0, steps, stimuli) → 多脑区同时刺激
"""

from __future__ import annotations

import abc
import logging
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# Scaling factor converting normalised stimulus amplitude [0, 1] to latent-space
# standard-deviation units used by TwinBrainDigitalTwin.simulate_intervention().
#
# Neuroscientific rationale (API.md §12.7):
#   delta = 1.0  ≈ natural fluctuation amplitude (mild stimulation)
#   delta = 2.0  ≈ strong excitatory TMS / supra-threshold stimulation
# A normalised amplitude of 1.0 therefore maps to a delta of 2.0 σ, which
# represents a physiologically strong but not extreme perturbation.
_STIM_AMP_TO_LATENT_SIGMA: float = 2.0

# Default GPU memory budget per chunk for batched rollout (in MB).
# Tune down for GPUs with less VRAM (e.g. 256 for 4 GB cards).
_DEFAULT_CHUNK_MEMORY_MiB: int = 512


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the actual device string (``"cuda"`` or ``"cpu"``)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def _compute_chunk_steps(
    n_batch: int,
    n_regions: int,
    max_memory_mib: int = _DEFAULT_CHUNK_MEMORY_MiB,
) -> int:
    """
    Compute how many trajectory-steps fit within *max_memory_mib* of GPU memory.

    The dominant allocation during a batched rollout chunk is the trajectory
    storage tensor of shape ``(chunk_steps, n_batch, n_regions)`` in float32.
    This helper keeps that tensor under the requested memory budget.

    Args:
        n_batch:        Number of parallel trajectories.
        n_regions:      Number of brain regions.
        max_memory_mib: Memory budget per chunk in MiB (default 512 MiB).

    Returns:
        chunk_steps: Maximum step count per chunk (≥ 1).
    """
    bytes_per_step = n_batch * n_regions * 4  # float32 = 4 bytes
    chunk_steps = max(1, int(max_memory_mib * 1024 * 1024 // bytes_per_step))
    return chunk_steps


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
# Wilson-Cowan standalone integrator (baseline / unit tests only)
# ══════════════════════════════════════════════════════════════════════════════

def _make_connectivity(n: int, seed: int = 0) -> np.ndarray:
    """
    生成合成连接矩阵（供 WC 模式使用）。

    对角线为零，L1 归一化，随机稀疏连接。
    """
    rng = np.random.default_rng(seed)
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

    **用途**：仅作为独立基线模型与单元测试使用，不是真实模型的回退选项。
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
# Batched GPU-accelerated Wilson-Cowan integrator
# ══════════════════════════════════════════════════════════════════════════════

class _WilsonCowanIntegratorBatched:
    """
    GPU-accelerated batched Wilson-Cowan integrator (PyTorch backend).

    Processes **all trajectories in parallel** in a single matrix operation,
    making GPU-accelerated free-dynamics experiments feasible even for large
    ``n_init`` × ``steps`` configurations.

    Dynamics are identical to ``_WilsonCowanIntegrator``::

        dev   = state - x0          (deviation from per-trajectory equilibrium)
        delta = tanh(stim·2 + W@dev·0.35) · 0.04
        leak  = dev · 0.10
        x(t+1) = clip(x(t) + delta − leak, 0, 1)

    Shapes (n_batch = number of parallel trajectories, N = n_regions):
        state  : ``(n_batch, N)``  — current brain state for every trajectory
        stim   : ``(N,)``          — shared stimulus broadcast over the batch
        output : ``(n_batch, N)``  — updated state for every trajectory
    """

    def __init__(
        self,
        n_regions: int,
        x0_batch: np.ndarray,
        seed: int = 0,
        device: str = "cpu",
    ):
        """
        Args:
            n_regions:  Number of brain regions (N).
            x0_batch:   Equilibrium states for each trajectory, shape ``(n_batch, N)``.
            seed:       Random seed for the shared connectivity matrix.
            device:     PyTorch device string (``"cpu"``, ``"cuda"``, ``"cuda:0"``…).
        """
        self._device = torch.device(_resolve_device(device))
        W_np = _make_connectivity(n_regions, seed=seed)
        self.W = torch.from_numpy(W_np).float().to(self._device)           # (N, N)
        self.x0 = (
            torch.from_numpy(x0_batch.astype(np.float32)).to(self._device)  # (n_batch, N)
        )

    def step(
        self,
        state: torch.Tensor,
        stim: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single parallel step for all trajectories.

        Args:
            state: ``(n_batch, N)`` current state tensor on the same device.
            stim:  ``(N,)`` stimulus vector (broadcast over the batch).

        Returns:
            Updated state ``(n_batch, N)``.
        """
        dev_state = state - self.x0                          # (n_batch, N)
        net_dev   = torch.matmul(dev_state, self.W.T)        # (n_batch, N)
        delta     = torch.tanh(stim * 2.0 + net_dev * 0.35) * 0.04
        leak      = dev_state * 0.10
        return torch.clamp(state + delta - leak, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# BrainDynamicsSimulator
# ══════════════════════════════════════════════════════════════════════════════

class BrainDynamicsSimulator:
    """
    统一管理所有模拟过程的核心组件。

    **模型模式**（``twin`` 为 ``TwinBrainDigitalTwin`` 实例）::

        twin = load_trained_model("outputs/exp/best_model.pt")
        base_graph = load_graph_for_inference("outputs/graph_cache/sub-01_notask_xx.pt")
        sim = BrainDynamicsSimulator(model=twin, base_graph=base_graph, modality="fmri")

        # Autoregressive free dynamics (model-driven)
        traj, times = sim.rollout(steps=100)

        # Stimulated dynamics
        stim = SinStimulus(node=42, freq=10.0, amp=0.5, duration=30, onset=10)
        traj, times = sim.rollout(steps=80, stimulus=stim)

    **WC 模式**（``model=None``，仅用于基线对比）::

        sim = BrainDynamicsSimulator(model=None, n_regions=200)
        x0 = sim.sample_random_state()
        traj, times = sim.rollout(x0, steps=1000)

    重要：当 ``model`` 被提供但无法进行前向推断（``can_forward=False``），
    将立即抛出 ``RuntimeError``，**不会**自动回退到 WC 模式。
    """

    def __init__(
        self,
        model=None,
        n_regions: int = 200,
        dt: float = 0.004,
        fmri_subsample: int = 25,
        seed: int = 0,
        base_graph: Optional[HeteroData] = None,
        modality: str = "fmri",
        device: str = "auto",
    ):
        """
        Args:
            model:         ``TwinBrainDigitalTwin`` 实例（模型模式）或 ``None``（WC 模式）。
            n_regions:     脑区数量，WC 模式下使用（模型模式从 ``base_graph`` 推断）。
            dt:            内部时间步长（秒），WC 模式下使用（模型模式从采样率推断）。
            fmri_subsample: fMRI 下采样倍率（仅供参考，当前未实现子采样记录）。
            seed:           WC 模式下的随机种子。
            base_graph:    ``HeteroData`` 上下文图（模型模式下必须提供）。
            modality:      分析模态（``"fmri"`` 或 ``"eeg"``，模型模式下使用）。
            device:        计算设备（``"cpu"``、``"cuda"``、``"auto"``）。
                           ``"auto"`` 自动选择 CUDA（若可用）。
                           WC 模式下用于 batched GPU rollout。

        Raises:
            ValueError:  模型模式下未提供 ``base_graph``，
                         或 ``base_graph`` 中不含指定 ``modality``。
            RuntimeError: ``model`` 提供但 ``can_forward=False``（禁止无声回退）。
        """
        self.seed = seed
        self.fmri_subsample = fmri_subsample
        self.modality = modality
        self.base_graph = base_graph
        self.device = _resolve_device(device)

        if model is not None:
            # Hard check: reject models that explicitly declare they cannot forward
            if not getattr(model, "can_forward", True):
                raise RuntimeError(
                    "提供的模型 can_forward=False，无法进行前向推断。\n"
                    "请使用 loader.load_model.load_trained_model() 加载正确的 TwinBrain V5 检查点，\n"
                    "或使用 model=None 以启用独立的 Wilson-Cowan 模式。"
                )

            self._use_model = True
            self._is_twin = hasattr(model, "predict_future")

            if self._is_twin:
                if base_graph is None:
                    raise ValueError(
                        "使用 TwinBrainDigitalTwin 时必须提供 base_graph。\n"
                        "请使用 load_graph_for_inference() 加载图缓存文件。"
                    )
                if modality not in base_graph.node_types:
                    raise ValueError(
                        f"base_graph 中不含模态 '{modality}'。\n"
                        f"可用模态: {base_graph.node_types}"
                    )
                # Infer n_regions and dt from the graph
                self.n_regions = int(base_graph[modality].x.shape[0])
                sr = getattr(base_graph[modality], "sampling_rate", None)
                self.dt = 1.0 / float(sr) if sr else dt
            else:
                # Plain nn.Module (rare, non-TwinBrain model)
                self.n_regions = n_regions
                self.dt = dt

        else:
            # WC mode: standalone, no real model
            self._use_model = False
            self._is_twin = False
            self.n_regions = n_regions
            self.dt = dt
            logger.info(
                "BrainDynamicsSimulator: Wilson-Cowan 独立模式 (n_regions=%d, device=%s)",
                n_regions,
                self.device,
            )

        self.model = model

    # ── Core API ───────────────────────────────────────────────────────────────

    def step(
        self,
        state: np.ndarray,
        external_input: Optional[np.ndarray] = None,
        _wc: Optional[_WilsonCowanIntegrator] = None,
    ) -> np.ndarray:
        """
        单步动力学演化: x(t+1) = f(x(t), u(t))

        **注意**：在模型模式（TwinBrainDigitalTwin）下，单步推理不被支持，
        因为 TwinBrain 以整个上下文窗口为输入（而非单帧状态）。
        请改用 ``rollout()``。

        Args:
            state:          当前脑状态，shape (n_regions,)，值域 [0, 1]。
            external_input: 外部刺激向量，shape (n_regions,)。若为 None 则全零。
            _wc:            Wilson-Cowan 积分器实例（由 rollout 传入）。

        Returns:
            新脑状态，shape (n_regions,)。

        Raises:
            NotImplementedError: TwinBrainDigitalTwin 模式下调用此方法时。
        """
        if self._is_twin:
            raise NotImplementedError(
                "TwinBrainDigitalTwin 不支持单步 step()。\n"
                "请使用 rollout() 或 rollout_multi_stim() 进行轨迹预测。"
            )

        if external_input is None:
            external_input = np.zeros(self.n_regions, dtype=np.float32)

        if self._use_model:
            return self._model_step(state, external_input)
        else:
            wc = _wc or _WilsonCowanIntegrator(self.n_regions, x0=state, seed=self.seed)
            return wc.step(state, external_input)

    def rollout(
        self,
        x0: Optional[np.ndarray] = None,
        steps: int = 50,
        stimulus: Optional[Stimulus] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        连续动力学模拟。

        在**模型模式**下，使用 ``TwinBrainDigitalTwin`` 进行自回归滑窗预测：
        每次调用 ``predict_future()`` 或 ``simulate_intervention()`` 获取
        ``prediction_steps`` 步的预测，然后将上下文窗口滑动 ``prediction_steps`` 步，
        重复直到累计 ``steps`` 步。``x0`` 在模型模式下被忽略（上下文由 ``base_graph``
        提供）。

        在 **WC 模式**下，逐步积分：``x0`` 为必须参数。

        Args:
            x0:       WC 模式的初始脑状态，shape (n_regions,)；模型模式下忽略。
            steps:    模拟步数（模型模式：fMRI TR 数；WC 模式：积分步数）。
            stimulus: 刺激对象（实现 Stimulus.value(t)）；None → 自由演化。

        Returns:
            trajectory: shape (steps, n_regions)，每步的脑状态。
            times:      shape (steps,)，以秒为单位的时间轴。
        """
        if self._is_twin:
            return self._rollout_with_twin(steps=steps, stimulus=stimulus)

        # WC / plain nn.Module mode
        if x0 is None:
            x0 = self.sample_random_state()
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        if x0.shape[0] != self.n_regions:
            raise ValueError(
                f"x0 shape mismatch: expected ({self.n_regions},), got {x0.shape}"
            )
        return self._rollout_wc(x0, steps=steps, stimulus=stimulus)

    def rollout_multi_stim(
        self,
        x0: Optional[np.ndarray] = None,
        steps: int = 50,
        stimuli: Optional[List[Stimulus]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多脑区同时刺激的连续模拟。

        在**模型模式**下，使用 ``TwinBrainDigitalTwin.simulate_intervention()``，
        同时将所有活跃刺激的脑区和幅度打包为 ``interventions`` 字典。

        在 **WC 模式**下，逐步积分并叠加各刺激。

        Args:
            x0:       WC 模式的初始脑状态；模型模式下忽略。
            steps:    模拟步数。
            stimuli:  多个 ``Stimulus`` 对象的列表（None → 自由演化，等同于 rollout）。

        Returns:
            trajectory: shape (steps, n_regions)。
            times:      shape (steps,)。
        """
        if stimuli is None:
            stimuli = []

        if self._is_twin:
            return self._rollout_multi_stim_with_twin(steps=steps, stimuli=stimuli)

        if x0 is None:
            x0 = self.sample_random_state()
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        return self._rollout_multi_stim_wc(x0, steps=steps, stimuli=stimuli)

    def rollout_batched(
        self,
        X0: np.ndarray,
        steps: int = 50,
        stimulus: Optional[Stimulus] = None,
        device: Optional[str] = None,
        chunk_steps: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        GPU-accelerated batched rollout: run *all* trajectories in parallel.

        **WC 模式专用**：此方法仅在 Wilson-Cowan 模式（``model=None``）下工作。
        TwinBrainDigitalTwin 自回归模式不支持批量并行（需要 context-graph 链接），
        请使用 ``rollout()`` 进行序列化预测。

        每条轨迹使用自己的均衡点 x0，所有轨迹共享同一连接矩阵 W（相同 seed）。
        计算通过 PyTorch 矩阵运算向量化，可在 GPU 上高效执行。

        内存管理：轨迹张量按 ``chunk_steps`` 分块写入 CPU，避免 GPU OOM。
        自动块大小基于 :func:`_compute_chunk_steps`（默认每块 ≤512 MiB）。

        Args:
            X0:          初始脑状态，shape ``(n_batch, n_regions)``。
            steps:       模拟步数（WC 积分步数）。
            stimulus:    Stimulus 对象（所有轨迹共享同一刺激节点/幅度）；
                         ``None`` → 自由演化。
            device:      计算设备（``"cpu"``、``"cuda"``、``"auto"``）；
                         ``None`` → 使用 ``self.device``。
            chunk_steps: GPU 内存分块大小（步数）；
                         ``None`` → 由 :func:`_compute_chunk_steps` 自动推断。

        Returns:
            trajectories: shape ``(n_batch, steps, n_regions)``，每条轨迹的状态序列。
            times:        shape ``(steps,)``，以秒为单位的时间轴。

        Raises:
            NotImplementedError: 在模型模式（TwinBrainDigitalTwin）下调用时。
            ValueError:          ``X0`` 形状不符合期望。
        """
        if self._is_twin or self._use_model:
            raise NotImplementedError(
                "rollout_batched() 仅支持 Wilson-Cowan 独立模式（model=None）。\n"
                "TwinBrainDigitalTwin 模式请使用 rollout() 逐条预测。"
            )

        X0 = np.asarray(X0, dtype=np.float32)
        if X0.ndim != 2 or X0.shape[1] != self.n_regions:
            raise ValueError(
                f"X0 形状错误: 期望 (n_batch, {self.n_regions})，"
                f"实际 {X0.shape}"
            )

        _device = _resolve_device(device if device is not None else self.device)
        return self._rollout_wc_batched(
            X0=X0,
            steps=steps,
            stimulus=stimulus,
            device=_device,
            chunk_steps=chunk_steps,
        )

    # ── Private: WC rollout ────────────────────────────────────────────────────

    def _rollout_wc(
        self,
        x0: np.ndarray,
        steps: int,
        stimulus: Optional[Stimulus],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wilson-Cowan step-by-step rollout."""
        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        wc = _WilsonCowanIntegrator(self.n_regions, x0=x0, seed=self.seed)
        state = x0.copy()

        for t in range(steps):
            u = np.zeros(self.n_regions, dtype=np.float32)
            if stimulus is not None and stimulus.is_active(t):
                node_idx = stimulus.node
                if 0 <= node_idx < self.n_regions:
                    u[node_idx] = stimulus.value(t)
            state = self._step_internal(state, u, wc)
            trajectory[t] = state

        return trajectory, times

    def _rollout_multi_stim_wc(
        self,
        x0: np.ndarray,
        steps: int,
        stimuli: List[Stimulus],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Wilson-Cowan step-by-step rollout with multiple stimuli."""
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

    def _rollout_wc_batched(
        self,
        X0: np.ndarray,
        steps: int,
        stimulus: Optional[Stimulus],
        device: str,
        chunk_steps: Optional[int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Vectorised batched WC rollout on the given device.

        All ``n_batch`` trajectories are processed in parallel using PyTorch
        matrix operations.  GPU memory is managed by breaking the time axis
        into ``chunk_steps``-step blocks: each block is computed on the
        device and immediately transferred to CPU before the next block.

        Args:
            X0:          ``(n_batch, n_regions)`` initial states.
            steps:       Total number of integration steps.
            stimulus:    Shared stimulus for all trajectories (or None).
            device:      Resolved PyTorch device string.
            chunk_steps: Steps per GPU chunk; auto-computed if None.

        Returns:
            trajectories: ``(n_batch, steps, n_regions)`` float32 numpy array.
            times:        ``(steps,)`` time axis in seconds.
        """
        n_batch = X0.shape[0]
        _dev = torch.device(device)

        # Shared batched integrator (one W, per-trajectory x0)
        wc_batch = _WilsonCowanIntegratorBatched(
            self.n_regions, x0_batch=X0, seed=self.seed, device=device
        )
        state = torch.from_numpy(X0).float().to(_dev)  # (n_batch, N)
        times = np.arange(steps, dtype=np.float32) * self.dt

        # Auto-size chunks to stay within GPU memory budget (default: 512 MiB).
        # Tune _DEFAULT_CHUNK_MEMORY_MiB or pass chunk_steps explicitly for
        # GPUs with less VRAM.
        if chunk_steps is None:
            chunk_steps = _compute_chunk_steps(n_batch, self.n_regions)

        # Pre-compute stimulus values for all steps: (steps, N)
        stim_all_np = np.zeros((steps, self.n_regions), dtype=np.float32)
        if stimulus is not None:
            for t in range(steps):
                if stimulus.is_active(t) and 0 <= stimulus.node < self.n_regions:
                    stim_all_np[t, stimulus.node] = stimulus.value(t)
        stim_all = torch.from_numpy(stim_all_np).float().to(_dev)  # (steps, N)

        all_chunks: List[np.ndarray] = []
        t_offset = 0
        while t_offset < steps:
            actual_chunk = min(chunk_steps, steps - t_offset)

            # Allocate trajectory buffer for this chunk on the device
            chunk_traj = torch.empty(
                (actual_chunk, n_batch, self.n_regions),
                device=_dev,
                dtype=torch.float32,
            )

            for k in range(actual_chunk):
                u = stim_all[t_offset + k]  # (N,) — broadcast over batch
                state = wc_batch.step(state, u)
                chunk_traj[k] = state

            # Permute to (n_batch, actual_chunk, N) and move to CPU
            all_chunks.append(chunk_traj.permute(1, 0, 2).cpu().numpy())
            t_offset += actual_chunk

        trajectories = np.concatenate(all_chunks, axis=1)  # (n_batch, steps, N)
        return trajectories, times

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _trim_context(self, context: HeteroData) -> HeteroData:
        """
        Trim ``context[modality].x`` to ``predictor.context_length`` timesteps.

        The encoder processes all T timesteps, producing ``[N, T, H]`` activation
        tensors.  The predictor only reads the last ``context_length`` of those T
        steps, so sending more than ``context_length`` timesteps wastes VRAM.
        Trimming to ``context_length`` reduces peak encoder activation memory by
        ``T_base / context_length`` (≈1.9× for T_base=384, context_length=200),
        enabling n_init=200, steps=1000 experiments to fit on an 8 GB GPU.

        The original ``self.base_graph`` is never mutated — this operates on a
        clone returned by ``_clone_hetero_graph``.

        Returns:
            The same ``context`` object with ``context[modality].x`` replaced by
            its last ``context_length`` (or fewer) timesteps.
        """
        _predictor = getattr(getattr(self.model, "model", None), "predictor", None)
        _ctx_len: int = getattr(_predictor, "context_length", 200)
        if _predictor is None or not hasattr(_predictor, "context_length"):
            logger.debug(
                "_trim_context: predictor.context_length not found; "
                "defaulting to %d. Model hierarchy: model=%s, model.model=%s.",
                _ctx_len,
                type(self.model).__name__,
                type(getattr(self.model, "model", None)).__name__,
            )
        ctx_x = context[self.modality].x  # [N, T_base, C]
        if ctx_x.shape[1] > _ctx_len:
            context[self.modality].x = ctx_x[:, -_ctx_len:, :]
        return context

    # ── Private: TwinBrain twin-mode rollout ────────────────────────────────────

    def _rollout_with_twin(
        self,
        steps: int,
        stimulus: Optional[Stimulus],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressive rollout using TwinBrainDigitalTwin.

        For each chunk of ``chunk_size`` steps:
          - If any stimulus is active in that chunk, call ``simulate_intervention()``
            with the peak stimulus amplitude × 2σ scaling, and use the
            ``perturbed`` prediction as the new context.
          - Otherwise, call ``predict_future()`` and use the prediction as context.

        The context window is advanced by ``chunk_size`` steps at each iteration
        (sliding window: drop the oldest chunk, append the new prediction).

        Memory optimisation (8 GB GPU support):
            ``_trim_context`` is called once at rollout start to trim the history
            from T_base (e.g. 384) down to ``predictor.context_length`` (≤200).
            The ``_advance_context`` sliding window then maintains this shorter
            length throughout, so the encoder always receives
            ``[N, context_length, C]`` rather than ``[N, T_base, C]``.
            This reduces peak encoder activation memory by T_base / context_length
            (≈1.9× for T_base=384, context_length=200).
        """
        chunk_size: int = getattr(getattr(self.model, "model", None), "prediction_steps", 50)

        # Clone base_graph and trim context to predictor.context_length steps
        # to reduce encoder activation memory (see _trim_context for details).
        context = self._trim_context(_clone_hetero_graph(self.base_graph))

        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        t_offset = 0
        while t_offset < steps:
            remaining = steps - t_offset
            req_steps = min(chunk_size, remaining)

            # Check whether stimulus is active in this chunk
            if stimulus is not None and any(
                stimulus.is_active(t_offset + i) for i in range(req_steps)
            ):
                # Compute the peak amplitude in this chunk for the perturbation
                peak_amp = max(
                    stimulus.value(t_offset + i)
                    for i in range(req_steps)
                    if stimulus.is_active(t_offset + i)
                )
                # Convert normalised amplitude [0,1] → latent-std delta
                # (API.md §12.7: delta=1.0 ≈ natural fluctuation, 2.0 = strong TMS)
                delta = float(peak_amp) * _STIM_AMP_TO_LATENT_SIGMA
                result = self.model.simulate_intervention(
                    baseline_data=context,
                    interventions={self.modality: ([stimulus.node], delta)},
                    num_prediction_steps=req_steps,
                )
                pred = result["perturbed"][self.modality]  # [N, req_steps, 1]
            else:
                pred_dict = self.model.predict_future(context, num_steps=req_steps)
                if self.modality not in pred_dict:
                    # predict_future now raises RuntimeError directly on decoder
                    # failures, so reaching here means prediction was simply
                    # unavailable (use_prediction=False or all node type sequences
                    # were shorter than _PRED_MIN_SEQ_LEN).
                    raise RuntimeError(
                        f"模型未返回模态 '{self.modality}' 的预测。\n"
                        f"可用模态: {list(pred_dict.keys())}\n"
                        "可能原因:\n"
                        "  1. 模型以 use_prediction=False 训练（不支持自由动力学预测）。\n"
                        "  2. 上下文序列长度 < _PRED_MIN_SEQ_LEN (4)，无法进行预测。\n"
                        "  3. 加载检查点时缺少 config.yaml，导致 prediction_steps 不匹配。\n"
                        f"  n_regions={self.n_regions}, modality={self.modality}"
                    )
                pred = pred_dict[self.modality]  # [N, req_steps, 1]

            # Detach and move to CPU; shape: [req_steps, N]
            chunk_np = pred.squeeze(-1).detach().cpu().numpy().T
            trajectory[t_offset: t_offset + req_steps] = chunk_np[:req_steps]
            t_offset += req_steps

            # Roll the context window forward: drop oldest chunk, append prediction
            if t_offset < steps:
                context = _advance_context(context, pred, self.modality)

        return trajectory, times

    def _rollout_multi_stim_with_twin(
        self,
        steps: int,
        stimuli: List[Stimulus],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressive rollout with multiple simultaneous stimuli using TwinBrainDigitalTwin.

        All active stimuli in a chunk are packed into the ``interventions`` dict.
        Context trimming (same as ``_rollout_with_twin``) is applied to reduce
        encoder activation memory on small GPUs.
        """
        chunk_size: int = getattr(getattr(self.model, "model", None), "prediction_steps", 50)

        context = self._trim_context(_clone_hetero_graph(self.base_graph))
        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        t_offset = 0
        while t_offset < steps:
            remaining = steps - t_offset
            req_steps = min(chunk_size, remaining)

            # Build interventions dict: {modality: ([node_indices], delta)}
            # Aggregate peak amplitudes for each node across the chunk
            node_deltas: Dict[int, float] = {}
            for stim in stimuli:
                if 0 <= stim.node < self.n_regions:
                    for i in range(req_steps):
                        if stim.is_active(t_offset + i):
                            v = stim.value(t_offset + i) * _STIM_AMP_TO_LATENT_SIGMA  # → latent-std
                            node_deltas[stim.node] = max(
                                node_deltas.get(stim.node, 0.0), v
                            )

            if node_deltas:
                # Use the mean delta across all stimulated nodes
                node_list = list(node_deltas.keys())
                mean_delta = float(np.mean(list(node_deltas.values())))
                result = self.model.simulate_intervention(
                    baseline_data=context,
                    interventions={self.modality: (node_list, mean_delta)},
                    num_prediction_steps=req_steps,
                )
                pred = result["perturbed"][self.modality]  # [N, req_steps, 1]
            else:
                pred_dict = self.model.predict_future(context, num_steps=req_steps)
                if self.modality not in pred_dict:
                    raise RuntimeError(
                        f"模型未返回模态 '{self.modality}' 的预测。\n"
                        f"可用模态: {list(pred_dict.keys())}\n"
                        "可能原因: 模型以 use_prediction=False 训练，或上下文序列过短。"
                    )
                pred = pred_dict[self.modality]

            chunk_np = pred.squeeze(-1).detach().cpu().numpy().T
            trajectory[t_offset: t_offset + req_steps] = chunk_np[:req_steps]
            t_offset += req_steps

            if t_offset < steps:
                context = _advance_context(context, pred, self.modality)

        return trajectory, times

    # ── Private: plain nn.Module step (non-twin model mode) ────────────────────

    def _step_internal(
        self,
        state: np.ndarray,
        u: np.ndarray,
        wc: _WilsonCowanIntegrator,
    ) -> np.ndarray:
        """Route to model or WC depending on mode."""
        if self._use_model:
            # Plain nn.Module path (not TwinBrainDigitalTwin)
            return self._model_step(state, u)
        return wc.step(state, u)

    def _model_step(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        使用普通 nn.Module 进行单步演化（非 TwinBrainDigitalTwin 路径）。

        注意：此接口假设模型接受 (1, N) 张量并返回相同形状的张量。
        若模型接口不兼容（如需要 HeteroData），将抛出异常而非回退到 WC。
        """
        state_t = torch.from_numpy(state).float().unsqueeze(0)  # (1, N)
        u_t = torch.from_numpy(u).float().unsqueeze(0)           # (1, N)
        with torch.no_grad():
            out = self.model(state_t + u_t)
        return out.squeeze(0).cpu().numpy().astype(np.float32)

    # ── Convenience sampling ───────────────────────────────────────────────────

    def sample_random_state(
        self, rng: Optional[np.random.Generator] = None
    ) -> np.ndarray:
        """
        采样随机初始脑状态。

        - **WC 模式**：在 [0, 1]^n_regions 上均匀采样。
        - **模型模式**：从 ``base_graph`` 数据的均值附近添加随机噪声，
          以获取在生理范围内的初始状态（仅供参考，模型模式下 rollout 会忽略 x0）。
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        if self._is_twin and self.base_graph is not None:
            # Use mean + small noise from the real data
            x_data = self.base_graph[self.modality].x  # [N, T, 1]
            mean_state = x_data.squeeze(-1).mean(dim=1).cpu().numpy()  # [N]
            noise = rng.normal(0, 0.05, self.n_regions).astype(np.float32)
            return np.clip(mean_state + noise, 0.0, 1.0)

        return rng.random(self.n_regions).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Graph context helpers
# ══════════════════════════════════════════════════════════════════════════════

def _clone_hetero_graph(graph: HeteroData) -> HeteroData:
    """
    Clone a HeteroData graph, detaching tensor data to avoid gradient accumulation.

    Unlike ``graph.clone()``, this also detaches node feature tensors so that
    rolling the context window does not build up a computation graph over multiple
    prediction rounds.

    **Why deep clone?**
    Each autoregressive step produces a new context by concatenating old and new
    tensors.  Without detaching, PyTorch retains the full computation history for
    every tensor in the context window (potentially hundreds of TR steps × N_nodes
    × H_hidden), which causes quadratic memory growth over long rollouts.
    ``detach().clone()`` breaks the gradient tape at each rollout step so that
    memory stays O(T_context) regardless of rollout length.
    """
    cloned = HeteroData()

    # Copy node attributes
    for node_type in graph.node_types:
        src = graph[node_type]
        for attr_name in src.keys():
            val = getattr(src, attr_name, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                setattr(cloned[node_type], attr_name, val.detach().clone())
            else:
                setattr(cloned[node_type], attr_name, val)

    # Copy edge attributes
    for edge_type in graph.edge_types:
        src = graph[edge_type]
        for attr_name in src.keys():
            val = getattr(src, attr_name, None)
            if val is None:
                continue
            if isinstance(val, torch.Tensor):
                setattr(cloned[edge_type], attr_name, val.detach().clone())
            else:
                setattr(cloned[edge_type], attr_name, val)

    # Copy graph-level attributes (subject_idx, run_idx, etc.)
    for attr_name in graph.keys():
        if attr_name not in graph.node_types and attr_name not in [
            f"{s}__{r}__{d}" for s, r, d in graph.edge_types
        ]:
            try:
                val = graph[attr_name]
                if isinstance(val, torch.Tensor):
                    cloned[attr_name] = val.detach().clone()
                else:
                    cloned[attr_name] = val
            except (KeyError, AttributeError):
                pass

    return cloned


def _advance_context(
    context: HeteroData,
    pred: torch.Tensor,
    modality: str,
) -> HeteroData:
    """
    Advance the context window by appending new predictions and dropping the
    oldest time steps.

    The context window length ``T_ctx`` is kept constant:
      new_x[:, :, :] = concat(old_x[:, chunk:, :], pred[:, :, :], dim=1)

    Args:
        context:  Current context HeteroData with x shape [N, T_ctx, 1].
        pred:     New predictions from the model, shape [N, chunk, 1].
        modality: Node type to update (``'fmri'`` or ``'eeg'``).

    Returns:
        New context HeteroData with the same T_ctx but shifted by ``chunk`` steps.
    """
    new_ctx = _clone_hetero_graph(context)
    old_x = context[modality].x  # [N, T_ctx, 1]
    chunk = pred.shape[1]
    pred_clean = pred.detach().to(old_x.device)

    # Sliding window: drop oldest chunk steps, append prediction
    if chunk >= old_x.shape[1]:
        # Prediction is as long as or longer than the context → replace entirely
        new_ctx[modality].x = pred_clean[:, -old_x.shape[1]:, :]
    else:
        new_ctx[modality].x = torch.cat(
            [old_x[:, chunk:, :], pred_clean], dim=1
        )

    return new_ctx
