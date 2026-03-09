"""
Brain Dynamics Simulator
========================

统一管理所有模拟过程的核心组件。

**唯一支持的模式：TwinBrainDigitalTwin**
  ``BrainDynamicsSimulator(twin, base_graph)``
  使用 ``TwinBrainDigitalTwin`` 进行基于真实学习动力学的推断：
    - 自由演化：``twin.predict_future()`` 自回归滑窗预测
    - 受刺激演化：``twin.simulate_intervention()`` 潜空间扰动

时间尺度：dt = 1 / fmri_sampling_rate（通常 2 s per TR，从 base_graph 自动推断）

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

# Default fMRI TR (seconds) when sampling_rate is not stored in the graph cache.
# Standard TR for most whole-brain fMRI acquisitions.
_DEFAULT_FMRI_TR: float = 2.0

# Guard added to per-channel std to prevent division-by-zero during z-normalisation.
_STD_GUARD: float = 1e-8

# Threshold used to detect whether the primary modality's data is z-scored or
# [0,1]-normalised.  V5 graph caches store z-scored BOLD and EEG (values
# ≈ −3 to +3).  Any genuine minimum below this threshold confirms z-scoring,
# so that sample_random_state() and state_bounds can behave correctly.
_ZSCORE_DETECTION_THRESHOLD: float = -0.1

# Per-channel noise scale for sample_random_state() in single-modality mode.
# Expressed as a fraction of the data's own standard deviation.  0.3σ gives a
# moderate diversity of initial conditions while staying well within the
# model's natural operating range (avoids extreme ±2-3σ starting states that
# would require many warm-up steps to recover from).
_INITIAL_STATE_NOISE_SCALE: float = 0.3


def _resolve_device(device: str) -> str:
    """Resolve ``"auto"`` to the actual device string (``"cuda"`` or ``"cpu"``)."""
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


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
# BrainDynamicsSimulator
# ══════════════════════════════════════════════════════════════════════════════

class BrainDynamicsSimulator:
    """
    统一管理所有模拟过程的核心组件。

    **唯一支持的模式：TwinBrainDigitalTwin**::

        twin = load_trained_model("outputs/exp/best_model.pt")
        base_graph = load_graph_for_inference("outputs/graph_cache/sub-01_notask_xx.pt")
        sim = BrainDynamicsSimulator(model=twin, base_graph=base_graph, modality="fmri")

        # Autoregressive free dynamics (model-driven)
        traj, times = sim.rollout(steps=100)

        # Stimulated dynamics
        stim = SinStimulus(node=42, freq=10.0, amp=0.5, duration=30, onset=10)
        traj, times = sim.rollout(steps=80, stimulus=stim)

    ``model`` 是必须参数，且必须是实现了 ``predict_future`` 的
    TwinBrainDigitalTwin 实例。若提供 None 将立即抛出 ValueError。
    """

    # Modalities directly supported by the simulator.
    # - "fmri" / "eeg" : single-modality mode (one prediction stream)
    # - "joint"         : both fMRI and EEG predicted jointly in a single
    #                     predict_future() call; their z-normalised outputs are
    #                     concatenated into a joint state vector for dynamics
    #                     analysis.
    _VALID_MODALITIES = ("fmri", "eeg", "joint")

    def __init__(
        self,
        model,
        base_graph: HeteroData,
        modality: str = "fmri",
        fmri_subsample: int = 25,
        seed: int = 0,
        device: str = "auto",
    ):
        """
        Args:
            model:          TwinBrainDigitalTwin 实例（必须）。
            base_graph:     HeteroData 图缓存（必须）。
            modality:       分析模态（``"fmri"``、``"eeg"`` 或 ``"joint"``）。

                            ``"joint"`` 模式：单次 ``predict_future()`` 调用同时
                            获取 fMRI 和 EEG 预测，各自按通道 z-score 归一化后
                            拼接为联合状态向量 ``[z_fmri | z_eeg]``，用于联合动
                            力学分析（单一 Lyapunov 指数）。需要 base_graph 同时
                            含有 ``'fmri'`` 和 ``'eeg'`` 节点。

            fmri_subsample: fMRI 下采样倍率（保留兼容性，当前未实现子采样）。
            seed:           随机种子（用于 sample_random_state）。
            device:         计算设备（``"auto"`` → CUDA if available）。

        Raises:
            ValueError: model 为 None，或 model 不是 TwinBrainDigitalTwin，
                        或 base_graph 缺少指定 modality，或 modality 不在
                        ``_VALID_MODALITIES`` 中。

        **时间分辨率说明**:
            TwinBrainDigitalTwin 的所有预测步长均以 fMRI TR 为单位（通常 2 s）。
            EEG 节点被预测在同一步长分辨率下，而非 EEG 原生采样率（通常 250 Hz）。
            因此无论选择哪种模态，``dt`` 均为 fMRI TR。若图缓存中存有
            ``sampling_rate`` 属性则从中推断；否则默认 2.0 s/TR。
        """
        if model is None:
            raise ValueError(
                "BrainDynamicsSimulator 需要一个训练好的 TwinBrainDigitalTwin 模型。\n"
                "请使用 loader.load_model.load_trained_model() 加载检查点。\n"
                "Wilson-Cowan 模式已移除——此模块专用于测试训练好的模型。"
            )
        if not hasattr(model, "predict_future"):
            raise ValueError(
                f"model 必须是具有 predict_future() 方法的 TwinBrainDigitalTwin 实例。\n"
                f"实际类型: {type(model).__name__}"
            )
        if base_graph is None:  # type: ignore[comparison-overlap]  # defensive for runtime callers
            raise ValueError(
                "使用 TwinBrainDigitalTwin 时必须提供 base_graph。\n"
                "请使用 load_graph_for_inference() 加载图缓存文件。"
            )

        # ── Validate modality ─────────────────────────────────────────────────
        if modality not in self._VALID_MODALITIES:
            raise ValueError(
                f"modality 必须是 {self._VALID_MODALITIES} 之一，得到 '{modality}'。"
            )
        if modality == "joint":
            if "fmri" not in base_graph.node_types or "eeg" not in base_graph.node_types:
                raise ValueError(
                    "modality='joint' 需要 base_graph 同时包含 'fmri' 和 'eeg' 节点。\n"
                    f"当前可用节点类型: {list(base_graph.node_types)}"
                )
        elif modality not in base_graph.node_types:
            raise ValueError(
                f"base_graph 中不含模态 '{modality}'。\n"
                f"可用模态: {list(base_graph.node_types)}"
            )

        self.seed = seed
        self.fmri_subsample = fmri_subsample
        self.modality = modality
        self.base_graph = base_graph
        self.model = model
        self.device = _resolve_device(device)

        # Always twin mode
        self._is_twin = True

        # ── Infer effective prediction dt ─────────────────────────────────────
        # TwinBrainDigitalTwin predicts ALL modalities at fMRI TR resolution
        # regardless of each modality's native sampling rate.  Using the native
        # EEG sampling rate (e.g. 250 Hz → dt=0.004 s) for the time axis would
        # produce a wrong time axis (50 steps × 0.004 s = 0.2 s vs. the correct
        # 50 × 2.0 s = 100 s for a typical TR).
        #
        # The fMRI TR is used as the single authoritative step size for all
        # modalities.  Fall back to 2.0 s (standard 0.5 Hz TR) when the graph
        # cache does not carry a sampling_rate attribute.
        _fmri_sr = (
            getattr(base_graph["fmri"], "sampling_rate", None)
            if "fmri" in base_graph.node_types else None
        )
        _pred_dt: float = 1.0 / float(_fmri_sr) if _fmri_sr else _DEFAULT_FMRI_TR

        # ── Per-modality initialisation ───────────────────────────────────────
        if modality == "joint":
            # Joint mode: fMRI + EEG predicted together, outputs z-normalised
            # and concatenated into a single state vector.
            self.n_fmri_regions: int = int(base_graph["fmri"].x.shape[0])
            self.n_eeg_regions:  int = int(base_graph["eeg"].x.shape[0])
            self.n_regions = self.n_fmri_regions + self.n_eeg_regions
            self.dt = _pred_dt
            self.native_sampling_rate: Optional[float] = None

            # Pre-compute per-channel normalisation statistics from base_graph
            # for z-normalising joint trajectories (and reversing the
            # normalisation when injecting x0 back into raw context).
            _fmri_x = base_graph["fmri"].x.squeeze(-1)  # [N_fmri, T]
            self._fmri_mean = _fmri_x.mean(dim=1).cpu().numpy().astype(np.float32)
            self._fmri_std  = _fmri_x.std(dim=1).cpu().numpy().astype(np.float32) + _STD_GUARD
            _eeg_x = base_graph["eeg"].x.squeeze(-1)    # [N_eeg, T]
            self._eeg_mean  = _eeg_x.mean(dim=1).cpu().numpy().astype(np.float32)
            self._eeg_std   = _eeg_x.std(dim=1).cpu().numpy().astype(np.float32) + _STD_GUARD

        elif modality == "eeg":
            self.n_regions = int(base_graph["eeg"].x.shape[0])
            _eeg_sr = getattr(base_graph["eeg"], "sampling_rate", None)
            self.native_sampling_rate = float(_eeg_sr) if _eeg_sr else None
            # Use fMRI TR as effective dt (model predicts EEG at TR resolution)
            self.dt = _pred_dt
            if _eeg_sr and abs(_pred_dt - 1.0 / float(_eeg_sr)) > 1e-3:
                logger.warning(
                    "EEG 时间分辨率说明：TwinBrainDigitalTwin 以 fMRI TR 为时间单位"
                    "预测所有模态，EEG 预测步长 = %.4f s（fMRI TR），"
                    "而非 EEG 原生采样率 %.1f Hz（%.4f s/样本）。"
                    "时间轴 `times` 和所有分析参数均以 TR 为单位。",
                    _pred_dt, float(_eeg_sr), 1.0 / float(_eeg_sr),
                )

        else:  # fmri
            self.n_regions = int(base_graph["fmri"].x.shape[0])
            _fmri_sr_local = getattr(base_graph["fmri"], "sampling_rate", None)
            self.native_sampling_rate = float(_fmri_sr_local) if _fmri_sr_local else None
            self.dt = 1.0 / float(_fmri_sr_local) if _fmri_sr_local else _pred_dt

        # ── Determine state-space bounds ──────────────────────────────────────
        # Wolf/FTLE Lyapunov analysis clips perturbed states to these bounds
        # after each renormalisation period.  V5 graph caches store z-scored
        # data for BOTH fMRI and EEG (values ≈ −3 to +3).  Clipping z-scored
        # states to [0, 1] would introduce a spurious attractor at 0 and bias
        # all Lyapunov estimates.
        #
        # We detect z-scored data by checking the minimum value in the primary
        # modality's tensor:
        #   min < −0.1  →  z-scored  →  no hard bounds (None)
        #   min ≥ −0.1  →  [0, 1]-normalised (legacy)  →  bounds = (0.0, 1.0)
        #
        # Joint mode always uses None (concatenated z-scores are unbounded).
        if modality == "joint":
            self._state_bounds: Optional[Tuple[float, float]] = None
        else:
            _nt_min = float(base_graph[modality].x.min())
            self._state_bounds = None if _nt_min < _ZSCORE_DETECTION_THRESHOLD else (0.0, 1.0)
            if self._state_bounds is None:
                logger.debug(
                    "state_bounds: '%s' data min=%.4f < %.1f → z-scored, "
                    "disabling [0,1] clipping for Wolf/FTLE.",
                    modality, _nt_min, _ZSCORE_DETECTION_THRESHOLD,
                )

        logger.info(
            "BrainDynamicsSimulator: TwinBrainDigitalTwin 模式 "
            "(modality=%s, n_regions=%d, dt=%.4f s [fMRI TR], device=%s)",
            modality, self.n_regions, self.dt, self.device,
        )

    # ── Core API ───────────────────────────────────────────────────────────────

    @property
    def state_bounds(self) -> Optional[Tuple[float, float]]:
        """
        State-space bounds used by Wolf/FTLE Lyapunov analysis for clipping
        perturbed states after each renormalisation period.

        The value is determined at construction time by inspecting the primary
        modality's data:

        - **z-scored data** (V5 format, ``data.min() < _ZSCORE_DETECTION_THRESHOLD``):
          returns ``None``.
          Both fMRI BOLD and EEG in the V5 graph cache are z-score normalised
          (values ≈ −3 to +3).  Clipping to ``[0, 1]`` would introduce a
          spurious attractor at 0 and must be avoided.  Lyapunov analysis
          should use the Rosenstein method (no clipping required).

        - **[0, 1]-normalised data** (legacy format,
          ``data.min() ≥ _ZSCORE_DETECTION_THRESHOLD``):
          returns ``(0.0, 1.0)``.  Preserves physical realism of the perturbed
          state during Wolf-Benettin renormalisation.

        - **Joint mode**: always ``None`` (concatenated z-scores are unbounded).
        """
        return self._state_bounds

    def rollout(
        self,
        x0: Optional[np.ndarray] = None,
        steps: int = 50,
        stimulus: Optional[Stimulus] = None,
        context_window_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        连续动力学模拟（TwinBrainDigitalTwin 自回归预测）。

        ``x0`` 注入机制：
            若提供 ``x0``，它将被写入初始上下文的最后一个时间步，覆盖图缓存中
            对应模态的原始数值。这样不同的 ``x0`` 会产生不同的初始脑状态，使
            ``run_free_dynamics`` 中的多条轨迹真正从不同位置出发（计划书 §五）。

        ``context_window_idx`` 时序窗口机制：
            选择从 ``base_graph`` 时间序列的哪个历史窗口作为初始上下文。
            使用滑动步长 ``stride = max(1, context_length // 4)``（75% 重叠），
            从有限时序数据中最大化可用窗口数：

              context_window_idx=0 → 最近 context_length 步（默认行为）
              context_window_idx=1 → 起点向前移动 stride 步
              context_window_idx=k → x[:, T-L-k*s : T-k*s, :] （L=context_length, s=stride）

            可用窗口数通过 ``simulator.n_temporal_windows`` 属性查询（取决于
            主模态时序长度和 stride 大小）。

        Args:
            x0:                 初始脑状态，shape (n_regions,)。
                                注入为初始上下文最后一步（可选）。
            steps:              模拟步数（fMRI TR 数）。
            stimulus:           刺激对象（实现 Stimulus.value(t)）；None → 自由演化。
            context_window_idx: 选择哪个时序窗口作为初始上下文（默认 0 = 最近窗口）。

        Returns:
            trajectory: shape (steps, n_regions)，每步的脑状态。
            times:      shape (steps,)，以秒为单位的时间轴。
        """
        return (
            self._rollout_multi_stim_joint(
                steps=steps,
                stimuli=[stimulus] if stimulus is not None else [],
                x0=x0,
                context_window_idx=context_window_idx,
            ) if self.modality == "joint" else self._rollout_with_twin(
                steps=steps, stimulus=stimulus, x0=x0,
                context_window_idx=context_window_idx,
            )
        )

    def rollout_multi_stim(
        self,
        x0: Optional[np.ndarray] = None,
        steps: int = 50,
        stimuli: Optional[List[Stimulus]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        多脑区同时刺激的连续模拟（TwinBrainDigitalTwin）。

        使用 ``TwinBrainDigitalTwin.simulate_intervention()``，
        同时将所有活跃刺激的脑区和幅度打包为 ``interventions`` 字典。

        Args:
            x0:       初始脑状态；注入为初始上下文最后一步（见 rollout）。
            steps:    模拟步数。
            stimuli:  多个 ``Stimulus`` 对象的列表（None → 自由演化，等同于 rollout）。

        Returns:
            trajectory: shape (steps, n_regions)。
            times:      shape (steps,)。
        """
        if self.modality == "joint":
            return self._rollout_multi_stim_joint(steps=steps, stimuli=stimuli, x0=x0)
        if stimuli is None:
            stimuli = []
        return self._rollout_multi_stim_with_twin(steps=steps, stimuli=stimuli, x0=x0)

    # ── Wolf-Benettin rollout pair ────────────────────────────────────────────

    def wolf_rollout_pair(
        self,
        x_base: np.ndarray,
        x_pert: np.ndarray,
        steps: int,
        wolf_context=None,
    ):
        """
        Run one Wolf-Benettin period: evolve the (base, perturbed) pair for
        *steps* steps, returning the final states and the updated context.

        ``wolf_context`` is the current ``HeteroData`` context window.  The
        base rollout advances the context via ``_advance_context`` so that
        subsequent Wolf periods are genuine continuations of the trajectory
        rather than resets to ``base_graph``.

        Args:
            x_base:       Base starting state, shape ``(n_regions,)``.
            x_pert:       Perturbed starting state, shape ``(n_regions,)``.
            steps:        Number of integration steps per Wolf period
                          (= ``renorm_steps`` in the caller).
            wolf_context: ``HeteroData`` context from the previous period;
                          ``None`` on the very first period (auto-initialised
                          from base_graph).

        Returns:
            (x_after_base, x_after_pert, next_wolf_context):
              x_after_base:      Final base state, shape ``(n_regions,)``.
              x_after_pert:      Final perturbed state, shape ``(n_regions,)``.
              next_wolf_context: Updated HeteroData for the next Wolf period.
        """
        return self._wolf_pair_twin(x_base, x_pert, steps, wolf_context)

    def _z_normalise_joint(
        self, fmri_pred: "torch.Tensor", eeg_pred: "torch.Tensor"
    ) -> np.ndarray:
        """
        Z-normalise and concatenate fMRI + EEG prediction tensors.

        Each modality is normalised per-channel using statistics pre-computed
        from ``base_graph`` in ``__init__``.  The result is a joint state chunk
        ``[z_fmri | z_eeg]`` with shape ``(T, N_fmri + N_eeg)`` where every
        dimension has comparable z-score units — required for distance-based
        dynamics metrics (Lyapunov exponents, FNN embedding dimension).

        Args:
            fmri_pred: ``[N_fmri, T, 1]`` tensor from ``predict_future``/``perturbed``.
            eeg_pred:  ``[N_eeg,  T, 1]`` tensor.

        Returns:
            ``float32`` array of shape ``(T, N_fmri + N_eeg)``.
        """
        fmri_np = fmri_pred.squeeze(-1).detach().cpu().numpy().T  # [T, N_fmri]
        eeg_np  = eeg_pred.squeeze(-1).detach().cpu().numpy().T   # [T, N_eeg]
        fmri_z  = (fmri_np - self._fmri_mean) / self._fmri_std
        eeg_z   = (eeg_np  - self._eeg_mean)  / self._eeg_std
        return np.concatenate([fmri_z, eeg_z], axis=1)

    def _wolf_predict(
        self,
        context: HeteroData,
        steps: int,
    ):
        """
        Run the twin model for *steps* prediction steps from *context*.

        Unlike ``_rollout_with_twin`` (which always resets to ``base_graph``),
        this helper uses exactly the provided context and advances it
        step-by-step.  Used internally by ``_wolf_pair_twin``.

        Returns:
            trajectory:       ``(steps, n_regions)`` float32 numpy array.
            advanced_context: ``HeteroData`` with all modalities advanced by
                              *steps* predictions.
        """
        chunk_size: int = getattr(
            getattr(self.model, "model", None), "prediction_steps", 50
        )
        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        ctx = _clone_hetero_graph(context)
        t_offset = 0

        while t_offset < steps:
            remaining = steps - t_offset
            req_steps = min(chunk_size, remaining)

            pred_dict = self.model.predict_future(ctx, num_steps=req_steps)

            if self.modality == "joint":
                fmri_pred = pred_dict.get("fmri")
                eeg_pred  = pred_dict.get("eeg")
                if fmri_pred is None or eeg_pred is None:
                    raise RuntimeError(
                        "_wolf_predict: joint 模态缺少 fmri 或 eeg 预测。"
                    )
                chunk_np = self._z_normalise_joint(fmri_pred, eeg_pred)
            else:
                if self.modality not in pred_dict:
                    raise RuntimeError(
                        f"模型未返回模态 '{self.modality}' 的预测 (_wolf_predict)。\n"
                        f"可用模态: {list(pred_dict.keys())}"
                    )
                pred = pred_dict[self.modality]  # [N, req_steps, 1]
                chunk_np = pred.squeeze(-1).detach().cpu().numpy().T  # [req_steps, N]
            trajectory[t_offset: t_offset + req_steps] = chunk_np[:req_steps]
            t_offset += req_steps

            # Always advance; the final ctx is the one returned.
            ctx = _advance_context(ctx, pred_dict)

        return trajectory, ctx

    def _wolf_pair_twin(
        self,
        x_base: np.ndarray,
        x_pert: np.ndarray,
        steps: int,
        wolf_context: Optional[HeteroData],
    ):
        """
        One Wolf period in twin mode.

        Both rollouts share the *same* pre-period context window, differing
        only in the last injected step (``x_base`` vs ``x_pert``).  After
        the base rollout the context is advanced via ``_advance_context`` so
        the next Wolf period continues from where the base trajectory left off.

        Resetting to ``base_graph`` every period (as plain ``rollout()`` does)
        causes ``x_cur`` to converge to the fixed point of the
        "inject → predict" map, making every trajectory yield the same
        biased LLE — the bug observed as λ=0.5559 for all 200 trajectories.
        """
        # Initialise context from base_graph on the first call.
        if wolf_context is None:
            wolf_context = self._trim_context(_clone_hetero_graph(self.base_graph))
            self._inject_x0_into_context(wolf_context, x_base)

        # --- Base rollout ---
        # Clone the current context so that the perturbed rollout below can
        # also start from the same (pre-advance) history.
        base_ctx = _clone_hetero_graph(wolf_context)
        self._inject_x0_into_context(base_ctx, x_base)
        traj_base, next_context = self._wolf_predict(base_ctx, steps)

        # --- Perturbed rollout ---
        # Uses the SAME pre-advance context but with x_pert as the last step.
        pert_ctx = _clone_hetero_graph(wolf_context)
        self._inject_x0_into_context(pert_ctx, x_pert)
        traj_pert, _ = self._wolf_predict(pert_ctx, steps)

        return traj_base[-1], traj_pert[-1], next_context

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _get_context_length(self) -> int:
        """Return the predictor's context_length (defaults to 200 if not found).

        200 is the standard context_length for the EnhancedMultiStepPredictor
        in TwinBrain V5 (see API.md §4.3 and training config model.context_length).
        Using it as the fallback keeps behaviour identical to the previous hard-coded
        ``_trim_context`` logic, which also defaulted to 200.
        """
        _predictor = getattr(getattr(self.model, "model", None), "predictor", None)
        return int(getattr(_predictor, "context_length", 200))

    @property
    def n_temporal_windows(self) -> int:
        """
        Number of overlapping context windows available in ``base_graph``.

        **Primary path** — full-context sliding windows (75 % overlap):

            stride      = max(1, context_length // 4)
            n_windows   = max(1, (T_primary - context_length) // stride + 1)

        This gives n_windows ≥ 2 only when ``T_primary > context_length +
        stride``, which requires more data than the model's full context window.

        **Fallback path** — prediction_steps-based stride (when T ≤
        context_length + stride and only 1 full-context window would be
        available):  use ``prediction_steps`` as the minimum stride to extract
        multiple shorter-context windows from the same recording.  Each window
        uses the oldest available slice of data:

            window 0  →  x[:, 0 : T, :]                    (full T steps)
            window 1  →  x[:, 0 : T−stride, :]             (T − stride steps)
            window k  →  x[:, 0 : T−k·stride, :]

        Predictions from shorter-context windows have lower initial quality but
        converge to full-context quality after
        ``(context_length − window_context) / prediction_steps`` autoregressive
        chunks.  For a 1000-step rollout this is a small fraction of the total
        trajectory.

        The fallback window count is capped at the number of genuinely distinct
        windows that ``_get_context_for_window`` can produce (i.e. windows
        whose ``end = T − k · stride > 0``).

        **Modality-aware**: For single-modality analysis (``fmri`` / ``eeg``),
        only the **primary** modality's temporal dimension ``T_primary``
        determines the window count.  Secondary modalities are handled
        gracefully at extraction time.

        For ``joint`` analysis, the minimum ``T`` across both ``fmri`` and
        ``eeg`` node types is used.

        Examples (context_length = 200, stride = 50, prediction_steps = 50):

            T_primary = 150  →  1 window   (T < context_length)
            T_primary = 200  →  4 windows  fallback: [0:200],[0:150],[0:100],[0:50]
            T_primary = 250  →  2 windows  primary: [50:250],[0:200]
            T_primary = 300  →  3 windows
            T_primary = 400  →  5 windows
            T_primary = 600  →  9 windows

        Examples (context_length = 37, stride = 9, prediction_steps = 17):

            T_primary = 200  →  19 windows  primary path (context fits many times)
        """
        ctx_len = self._get_context_length()
        stride = max(1, ctx_len // 4)

        if self.modality == "joint":
            # Joint mode: both fMRI and EEG must supply a full context window.
            min_t: float = float("inf")
            for nt in ("fmri", "eeg"):
                if nt in self.base_graph.node_types and hasattr(
                    self.base_graph[nt], "x"
                ):
                    min_t = min(min_t, int(self.base_graph[nt].x.shape[1]))
            if min_t == float("inf"):
                return 1
            T_primary = int(min_t)
        else:
            # Single-modality: only the primary modality determines window count.
            nt = self.modality
            if nt not in self.base_graph.node_types or not hasattr(
                self.base_graph[nt], "x"
            ):
                return 1
            T_primary = int(self.base_graph[nt].x.shape[1])

        # ── Primary path: full-context sliding windows ────────────────────────
        n_full = max(1, (T_primary - ctx_len) // stride + 1)
        if n_full > 1:
            return n_full

        # ── Fallback path: prediction_steps-based stride ─────────────────────
        # When T ≤ context_length + stride (only 1 full-context window), we can
        # still offer multiple shorter-context windows whose stride equals
        # prediction_steps.  This is only meaningful when prediction_steps <
        # context_length; otherwise there is no benefit over the single window.
        # Note: double-getattr is safe — getattr(None, attr, default) returns
        # default without raising AttributeError.
        _inner = getattr(self.model, "model", None)
        pred_steps: int = int(getattr(_inner, "prediction_steps", 0))
        if pred_steps > 0 and pred_steps < ctx_len and T_primary >= stride:
            n_pred = max(1, (T_primary - pred_steps) // pred_steps + 1)
            # Cap at the number of genuinely distinct windows _get_context_for_window
            # can provide: only windows where end = T - k*stride > 0 are unique;
            # beyond that the fallback in _get_context_for_window always returns
            # the same [0:T] slice.
            n_extractable = (T_primary - 1) // stride + 1  # ceil(T / stride)
            n_fallback = min(n_pred, n_extractable)
            if n_fallback > 1:
                return n_fallback

        return 1

    def _get_context_for_window(self, window_idx: int = 0) -> HeteroData:
        """
        Return a cloned HeteroData using the specified temporal window as context.

        Selects a ``context_length``-step slice from each node type's time series
        using an overlapping sliding-window scheme.  The stride between consecutive
        windows is ``max(1, context_length // 4)`` (75 % overlap), consistent with
        ``n_temporal_windows``::

          stride = max(1, context_length // 4)

          window_idx = 0  →  x[:, T-L : T, :]
          window_idx = 1  →  x[:, T-L-s : T-s, :]
          window_idx = k  →  x[:, T-L-k*s : T-k*s, :]   (L=context_length, s=stride)

        When ``window_idx`` would push the start before the beginning of the
        recording (``start < 0``), the earliest available ``context_length``-step
        slice is used instead of raising an error (graceful degradation).  This
        can happen for secondary modalities (e.g. EEG when analysing fMRI) that
        have a shorter stored time series than the primary modality.

        Args:
            window_idx: Non-negative integer selecting which historical window to
                        use (0 = most recent = current default behaviour).

        Returns:
            Cloned ``HeteroData`` with every node type's ``.x`` set to the
            requested window, ready for ``_inject_x0_into_context``.
        """
        ctx_len = self._get_context_length()
        stride = max(1, ctx_len // 4)
        context = _clone_hetero_graph(self.base_graph)

        for nt in list(context.node_types):
            if not hasattr(context[nt], "x"):
                continue
            nt_x = context[nt].x   # [N, T, C]
            T = int(nt_x.shape[1])
            # Overlapping sliding windows (stride = ctx_len // 4):
            #   window 0 → [T-L:T], window 1 → [T-L-s:T-s], ...
            end = T - window_idx * stride
            start = end - ctx_len
            if start < 0 or end <= 0:
                # Requested window extends before the start of the recording;
                # fall back to the oldest full-length window available.
                start = 0
                end = min(ctx_len, T)
                if window_idx > 0:
                    logger.debug(
                        "_get_context_for_window: '%s' T=%d, window_idx=%d is "
                        "out of range (ctx_len=%d, stride=%d, need T≥%d); "
                        "using earliest window [0:%d].",
                        nt, T, window_idx, ctx_len, stride,
                        ctx_len + window_idx * stride, end,
                    )
            context[nt].x = nt_x[:, start:end, :]

        return context

    def _trim_context(self, context: HeteroData) -> HeteroData:
        """
        Trim **all** modalities in ``context`` to ``predictor.context_length`` timesteps.

        Design rationale (计划书 §四/§十三):
            The graph cache serves as the **initial state** for the simulator.
            After initialization, the model self-iterates using its own predictions.
            Only the last ``context_length`` timesteps per modality are needed to
            seed the predictor — any older history is wasted encoder compute.

        Memory impact:
            For a typical graph where fMRI has T=200 and EEG has T=98 500:
            - Old behaviour (trim only fMRI): EEG still encoded at T=98 500.
              Peak encoder activation: [63, 98500, 128] × 2 (fast+slow Conv1d)
              = 6.34 GB → CUDA OOM on 8 GB GPU.
            - New behaviour (trim ALL modalities to context_length=200):
              EEG encoded at T=200.
              Peak activation: [63, 200, 128] × 2 = 12.9 MB → trivial.

        Scientific validity:
            ``EnhancedMultiStepPredictor.predict_next()`` uses the last
            ``context_length`` *encoded* timesteps.  The encoder's Conv1d
            (kernel_size=3, padding=same) preserves T, so trimming raw input to
            ``context_length`` steps produces exactly ``context_length`` encoded
            steps — a perfect match.  The most recent ``context_length`` raw
            steps carry the relevant brain-state history for the predictor.

        The original ``self.base_graph`` is never mutated — this operates on a
        clone returned by ``_clone_hetero_graph``.

        Returns:
            The same ``context`` object with every node type's ``.x`` replaced by
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
        # Trim ALL node types, not just the primary modality.
        # This is the key fix for EEG OOM: T_eeg can be orders of magnitude
        # larger than context_length (e.g. 98500 vs 200).
        for nt in list(context.node_types):
            # Guard: skip node types that lack a temporal .x attribute.
            # Auxiliary / metadata node types may appear in node_types without
            # .x (e.g. when the graph has extra metadata nodes after deepcopy).
            # Accessing .x on a NodeStorage without it raises:
            #   AttributeError: 'NodeStorage' object has no attribute 'x'
            if not hasattr(context[nt], "x"):
                continue
            nt_x = context[nt].x  # [N, T, C]
            if nt_x.shape[1] > _ctx_len:
                context[nt].x = nt_x[:, -_ctx_len:, :]
                logger.debug(
                    "_trim_context: trimmed '%s' from T=%d to T=%d",
                    nt, nt_x.shape[1], _ctx_len,
                )
        return context

    def _inject_x0_into_context(
        self,
        context: HeteroData,
        x0: Optional[np.ndarray],
    ) -> None:
        """
        Inject ``x0`` into the **last time step** of the primary modality's context.

        This gives each free-dynamics rollout a distinct starting point
        (计划书 §五: ``x0 = sample_random_state()``).  Without this injection,
        all n_init rollouts would start from identical graph-cache values and
        produce identical trajectories.

        **Single-modality mode** (fMRI / EEG):
            Writes ``x0`` into ``context[self.modality].x[:, -1, 0]``.

        **joint mode**:
            ``x0`` is a z-scored concatenated vector ``[z_fmri | z_eeg]`` from
            ``sample_random_state()``.  Each half is un-z-scored using the
            per-channel statistics pre-computed in ``__init__``, then written
            into the last step of the respective modality's context.  This maps
            the normalised joint state back to the raw representation stored in
            the graph cache.

        Args:
            context: Trimmed HeteroData clone (already detached from base_graph).
            x0:      Initial brain state, shape ``(n_regions,)``.  Silently
                     ignored when ``None`` or shape does not match ``n_regions``.
        """
        if x0 is None:
            return
        x0_np = np.asarray(x0, dtype=np.float32).flatten()
        if x0_np.shape[0] != self.n_regions:
            logger.warning(
                "_inject_x0_into_context: x0 shape %s does not match "
                "n_regions=%d; injection skipped.",
                x0_np.shape, self.n_regions,
            )
            return

        if self.modality == "joint":
            # Split joint z-scored x0 into fMRI and EEG parts
            fmri_x0_z = x0_np[:self.n_fmri_regions]  # z-score units
            eeg_x0_z  = x0_np[self.n_fmri_regions:]

            # Un-z-score to match the raw values stored in the graph context
            fmri_x0_raw = torch.from_numpy(fmri_x0_z * self._fmri_std + self._fmri_mean)
            eeg_x0_raw  = torch.from_numpy(eeg_x0_z  * self._eeg_std  + self._eeg_mean)

            ctx_fmri = context["fmri"].x  # [N_fmri, T_ctx, 1]
            if fmri_x0_raw.shape[0] == ctx_fmri.shape[0]:
                ctx_fmri[:, -1, 0] = fmri_x0_raw
            ctx_eeg = context["eeg"].x    # [N_eeg, T_ctx, 1]
            if eeg_x0_raw.shape[0] == ctx_eeg.shape[0]:
                ctx_eeg[:, -1, 0] = eeg_x0_raw
            return

        # Single-modality injection
        x0_t = torch.from_numpy(x0_np)
        ctx_x = context[self.modality].x  # [N, T_ctx, C], owned by clone
        if x0_t.shape[0] == ctx_x.shape[0]:
            # Replace the last time step with x0 so that the encoder sees the
            # desired initial brain state at t = T_ctx - 1.
            ctx_x[:, -1, 0] = x0_t

    # ── Private: TwinBrain twin-mode rollout ────────────────────────────────────

    def _rollout_with_twin(
        self,
        steps: int,
        stimulus: Optional[Stimulus],
        x0: Optional[np.ndarray] = None,
        context_window_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressive rollout using TwinBrainDigitalTwin.

        Initialization (计划书 §四/§十三):
            ``_get_context_for_window(context_window_idx)`` selects the initial
            context window from ``base_graph``.  ``window_idx=0`` reproduces the
            original behaviour (last ``context_length`` steps).  Higher indices
            select earlier historical segments, giving each trajectory a genuinely
            different context history.  This resolves the Wolf context-dilution
            bias where all trajectories shared the same ``context_length − 1``
            history steps (see AGENTS.md §Wolf上下文稀释).

            If ``x0`` is provided, it is injected into the last time step of the
            selected context window.
        """
        chunk_size: int = getattr(getattr(self.model, "model", None), "prediction_steps", 50)

        # Select the requested temporal window from base_graph.
        # window_idx=0 reproduces the previous _trim_context behaviour exactly.
        context = self._get_context_for_window(context_window_idx)

        # Inject x0 as the "current brain state" (last time step override).
        # This ensures 200 rollouts start from 200 different initial conditions.
        self._inject_x0_into_context(context, x0)

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
                pred_dict = result["perturbed"]  # {nt: [N, req_steps, 1]}
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

            # Roll ALL modality context windows forward using the full pred_dict.
            # This keeps every modality temporally aligned (EEG no longer lags fMRI).
            if t_offset < steps:
                context = _advance_context(context, pred_dict)

        return trajectory, times

    def _rollout_multi_stim_joint(
        self,
        steps: int,
        stimuli: List[Stimulus],
        x0: Optional[np.ndarray] = None,
        context_window_idx: int = 0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Joint fMRI+EEG autoregressive rollout for ``modality='joint'``.

        Handles both free dynamics (``stimuli=[]``) and any combination of
        fMRI/EEG stimuli (``stimuli=[...]``).  ``rollout()`` delegates here by
        wrapping a single ``Stimulus`` in a list.

        **Joint state**: ``[z_fmri | z_eeg]`` shape ``(steps, N_fmri+N_eeg)``.
        Each modality is per-channel z-normalised via ``_z_normalise_joint``
        using statistics pre-computed from ``base_graph``.  This puts fMRI BOLD
        and EEG on a common scale for distance-based analyses (Lyapunov, FNN).

        **Stimulation routing** (joint node space):
          - ``0 ≤ node < N_fmri``      → ``interventions={"fmri": ([node], δ)}``
          - ``N_fmri ≤ node < N_joint`` → ``interventions={"eeg": ([node-N_fmri], δ)}``
          - Multiple stimuli are partitioned into fMRI / EEG dicts and passed
            as a single ``simulate_intervention()`` call (one forward pass).
          - Out-of-range nodes raise ``ValueError`` immediately.

        **Temporal context**: ``context_window_idx`` selects which historical
        window from ``base_graph`` to use as the initial context, enabling
        trajectory diversity for Lyapunov / Wolf analysis.
        """
        chunk_size: int = getattr(
            getattr(self.model, "model", None), "prediction_steps", 50
        )
        context = self._get_context_for_window(context_window_idx)
        self._inject_x0_into_context(context, x0)

        trajectory = np.empty((steps, self.n_regions), dtype=np.float32)
        times = np.arange(steps, dtype=np.float32) * self.dt

        t_offset = 0
        while t_offset < steps:
            remaining = steps - t_offset
            req_steps = min(chunk_size, remaining)

            # Partition active stimuli into fMRI / EEG intervention dicts.
            fmri_node_deltas: Dict[int, float] = {}
            eeg_node_deltas:  Dict[int, float] = {}

            for stim in stimuli:
                node = stim.node
                if not (0 <= node < self.n_regions):
                    raise ValueError(
                        f"_rollout_multi_stim_joint: stimulus.node={node} 超出联合索引范围 "
                        f"[0, {self.n_regions})。"
                        f"fMRI: [0, {self.n_fmri_regions}), "
                        f"EEG: [{self.n_fmri_regions}, {self.n_regions})。"
                    )
                for i in range(req_steps):
                    if stim.is_active(t_offset + i):
                        v = stim.value(t_offset + i) * _STIM_AMP_TO_LATENT_SIGMA
                        if node < self.n_fmri_regions:
                            fmri_node_deltas[node] = max(fmri_node_deltas.get(node, 0.0), v)
                        else:
                            eeg_ch = node - self.n_fmri_regions
                            eeg_node_deltas[eeg_ch] = max(eeg_node_deltas.get(eeg_ch, 0.0), v)

            interventions: Dict[str, Tuple[List[int], float]] = {}
            if fmri_node_deltas:
                interventions["fmri"] = (
                    list(fmri_node_deltas.keys()),
                    float(np.mean(list(fmri_node_deltas.values()))),
                )
            if eeg_node_deltas:
                interventions["eeg"] = (
                    list(eeg_node_deltas.keys()),
                    float(np.mean(list(eeg_node_deltas.values()))),
                )

            if interventions:
                result = self.model.simulate_intervention(
                    baseline_data=context,
                    interventions=interventions,
                    num_prediction_steps=req_steps,
                )
                pred_dict = result["perturbed"]
            else:
                pred_dict = self.model.predict_future(context, num_steps=req_steps)

            fmri_pred = pred_dict.get("fmri")
            eeg_pred  = pred_dict.get("eeg")
            if fmri_pred is None or eeg_pred is None:
                raise RuntimeError(
                    "joint 模态要求模型同时返回 'fmri' 和 'eeg' 预测。\n"
                    f"实际返回: {list(pred_dict.keys())}\n"
                    "请确认模型以多模态联合训练，且 base_graph 包含两种节点。"
                )

            joint_chunk = self._z_normalise_joint(fmri_pred, eeg_pred)
            trajectory[t_offset: t_offset + req_steps] = joint_chunk[:req_steps]
            t_offset += req_steps

            if t_offset < steps:
                context = _advance_context(context, pred_dict)

        return trajectory, times

    def _rollout_multi_stim_with_twin(
        self,
        steps: int,
        stimuli: List[Stimulus],
        x0: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Autoregressive rollout with multiple simultaneous stimuli using TwinBrainDigitalTwin.

        All active stimuli in a chunk are packed into the ``interventions`` dict.
        Context trimming (same as ``_rollout_with_twin``) and x0 injection are
        applied to reduce encoder activation memory and support varied initial states.
        All modalities are advanced together after each prediction step.
        """
        chunk_size: int = getattr(getattr(self.model, "model", None), "prediction_steps", 50)

        context = self._trim_context(_clone_hetero_graph(self.base_graph))
        self._inject_x0_into_context(context, x0)

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
                pred_dict = result["perturbed"]  # {nt: [N, req_steps, 1]}
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

            # Advance ALL modality context windows together.
            if t_offset < steps:
                context = _advance_context(context, pred_dict)

        return trajectory, times

    # ── Convenience sampling ───────────────────────────────────────────────────

    def sample_random_state(
        self, rng: Optional[np.random.Generator] = None,
        noise_scale: float = _INITIAL_STATE_NOISE_SCALE,
        from_data: bool = False,
    ) -> np.ndarray:
        """
        采样随机初始脑状态。

        **单模态模式 (fMRI / EEG)**:

        *默认模式* ``from_data=False``:  从数据均值附近添加随机噪声
        （``mean + N(0, noise_scale) × std``）。适合刺激分析（小扰动、少瞬态）。

        *数据采样模式* ``from_data=True``（自由动力学推荐）:  从 ``base_graph``
        时序中随机抽取一个时间步 t，返回 ``data[:, t]`` 加上微量扰动。

        科学依据：用于自由动力学分析时，初始状态的**空间相关结构**至关重要。
        采用 ``mean + 各向同性噪声`` 会生成无空间相关的 x0（BOLD 激活在各脑区
        独立采样），而真实脑状态具有强空间相关性（功能连接）。模型编码器需要
        数百步才能将非相关初始状态"修正"为具有正确空间结构的状态，这一修正
        过程成为 PCA 的主成分（PC1 可高达 86% 方差），遮盖真实脑动力学。
        ``from_data=True`` 直接采样真实 BOLD 时间步，空间相关性与上下文历史
        一致，消除修正瞬态，使 PCA 反映真实动力学模态。

        **joint 模式**：分别为 fMRI 和 EEG 部分各自采样，再拼接成联合状态
        向量 ``[z_fmri | z_eeg]``，形状 ``(n_fmri_regions + n_eeg_regions,)``。
        joint 模式暂不支持 ``from_data=True``（两种模态时间轴可能不对齐）。

        检测方式：``base_graph[modality].x.min() < -0.1`` → z-scored；不裁剪。

        Args:
            rng:         随机数生成器；None → 使用 self.seed 重新创建。
            noise_scale: 噪声幅度（单位：数据标准差 σ）。
                         ``from_data=False`` 时：默认 0.3σ（小扰动）。
                         ``from_data=True`` 时：默认 0.1σ（仅用于避免轨迹重复）。
            from_data:   True → 从 base_graph 时序随机抽取时间步作为初始状态
                         （推荐用于自由动力学分析，可消除空间相关修正瞬态）。
                         False → 均值 + 各向同性噪声（旧行为，用于刺激分析）。
        """
        if rng is None:
            rng = np.random.default_rng(self.seed)

        if self.modality == "joint":
            # Joint mode: use noise-based sampling (fMRI/EEG time axes may differ).
            # When from_data is requested, fall back gracefully.
            _ns = noise_scale if not from_data else 0.1
            fmri_noise = rng.normal(0.0, _ns, self.n_fmri_regions).astype(np.float32)
            fmri_x0_z = fmri_noise  # z-score ≈ mean_z + noise (mean_z ≈ 0)

            eeg_noise = rng.normal(0.0, _ns, self.n_eeg_regions).astype(np.float32)
            eeg_x0_z = eeg_noise

            return np.concatenate([fmri_x0_z, eeg_x0_z])

        x_data = self.base_graph[self.modality].x  # [N, T, 1]
        data_np = x_data.squeeze(-1).cpu().numpy()   # [N, T]
        std_state  = np.maximum(data_np.std(axis=1), _STD_GUARD)  # [N]

        if from_data:
            # ── Data-sample mode: pick a random time step from actual BOLD/EEG ──
            # The selected column data[:, t] is a real brain state with full
            # spatial correlation structure, so the model encoder sees a context
            # that matches the injected x0 in distribution.  This eliminates the
            # long "spatial-structure correction" transient that dominates PC1
            # when using mean + isotropic-noise initial states.
            T_data = data_np.shape[1]
            t = int(rng.integers(0, T_data))
            x0 = data_np[:, t].copy().astype(np.float32)
            # Tiny perturbation (default 0.1σ) ensures each trajectory starts from
            # a slightly different state even when the same time step is sampled.
            _ns = noise_scale if noise_scale != _INITIAL_STATE_NOISE_SCALE else 0.1
            noise = rng.normal(0.0, _ns, self.n_regions).astype(np.float32)
            x0 = x0 + noise * std_state
        else:
            # ── Noise-perturb mode (original behaviour): mean + N(0, noise_scale×std) ─
            mean_state = data_np.mean(axis=1)            # [N]
            noise = rng.normal(0.0, noise_scale, self.n_regions).astype(np.float32)
            x0 = (mean_state + noise * std_state).astype(np.float32)

        # Only clip to [0, 1] for strictly non-negative (legacy-normalised) data.
        # Z-scored data (min < -0.1) must NOT be clipped.
        if self._state_bounds is not None:
            x0 = np.clip(x0, self._state_bounds[0], self._state_bounds[1])

        return x0


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
    pred_dict: Dict[str, torch.Tensor],
) -> HeteroData:
    """
    Advance the context window for ALL modalities present in ``pred_dict``.

    For each modality, the oldest ``chunk`` timesteps are dropped and the new
    predictions are appended, keeping the window length constant:

        new_x = concat(old_x[:, chunk:, :], pred[:, :, :], dim=1)

    Advancing all modalities together (not just the primary one) is required
    for temporal consistency: without it, e.g. EEG stays at initialization
    time while fMRI advances, creating an ever-growing temporal mismatch that
    degrades encoder representations over long rollouts.

    Args:
        context:   Current context HeteroData with ``x`` shape ``[N, T_ctx, 1]``
                   per node type.
        pred_dict: Dict mapping node-type name → prediction tensor
                   ``[N, chunk, 1]``.  Only modalities present in both
                   ``context.node_types`` and ``pred_dict`` are updated;
                   others are left unchanged.

    Returns:
        New context HeteroData with updated node features for all available
        modalities.
    """
    new_ctx = _clone_hetero_graph(context)
    for modality, pred in pred_dict.items():
        if modality not in context.node_types:
            continue
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
