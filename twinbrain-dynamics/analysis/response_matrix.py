"""
Response Matrix Computation
============================

量化 **刺激传播结构** (Stimulation Propagation Structure)。

定义：
  R[i, j] = response of node j when node i is stimulated

**WC 模式**：逐步积分，在稳态窗口（跳过暂态后）测量响应。
**TwinBrain 模式**：直接使用 model.simulate_intervention() 计算因果效应
  (causal_effect = perturbed − baseline)，一次 encoder 调用完成，无上下文漂移误差。

输出文件：outputs/response_matrix.npy
"""

import copy
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

# Stimulus amplitude [0,1] → latent-sigma multiplier (matches _STIM_AMP_TO_LATENT_SIGMA
# in brain_dynamics_simulator.py).
_STIM_AMP_TO_LATENT_SIGMA: float = 2.0


def compute_response_matrix(
    simulator: BrainDynamicsSimulator,
    n_nodes: Optional[int] = None,
    stim_amplitude: float = 0.5,
    stim_duration: int = 80,
    stim_frequency: float = 10.0,
    stim_pattern: str = "step",
    measure_window: int = 30,
    skip_transient: Optional[int] = None,
    pre_steps: int = 50,
    seed: int = 42,
    output_dir: Optional[Path] = None,
) -> np.ndarray:
    """
    计算完整的刺激响应矩阵 R[i, j]。

    R[i,j] 定义为：节点 j 响应节点 i 刺激的平均活动增量。

    **TwinBrain 模式（推荐）**：
    直接调用 ``model.simulate_intervention()``，使用相同 encoder 一次计算
    baseline 和 perturbed，取 ``causal_effect = perturbed − baseline`` 作为 R[i,:]。
    这比 rollout 模式更准确：消除了上下文漂移误差和自然动力学分量的干扰。

    **WC 模式（fallback）**：
    逐步积分，在稳态窗口（跳过暂态后）测量响应：
    ``R[i,j] = mean(stim_trajectory[skip:skip+window, j]) − baseline[j]``

    **为什么旧版 rollout 给出竖条纹**（仅 WC 模式）：
    ``simulate_intervention`` 存储 ``result["perturbed"]``（绝对预测），而非 causal_effect。
    绝对预测中包含巨大的自然动力学分量（对所有行 i 相同），使列结构主导 R，
    视觉上表现为竖条纹。在 TwinBrain 模式下直接取 causal_effect 可避免此问题。

    **注意：Twin 模式 R 仍可能出现块状/枢纽主导的竖条纹（真实现象）**：
    当网络存在少数枢纽区域（DMN、前额叶等）时，刺激任何节点都会通过图传播
    强烈影响这些枢纽，使枢纽列的响应与具体刺激位置无关。
    这是真实网络特性而非计算错误。可视化面板 B（行归一化）能揭示这些枢纽外
    的刺激特异性传播模式。

    Args:
        simulator:       BrainDynamicsSimulator 实例。
        n_nodes:         要刺激的节点数（None → 使用 simulator.n_regions）。
        stim_amplitude:  刺激幅度（Twin 模式：乘以 _STIM_AMP_TO_LATENT_SIGMA=2.0）。
        stim_duration:   WC 模式每个节点的刺激步数（Twin 模式忽略）。
        stim_frequency:  刺激频率（Hz，仅对 sine / square 模式有效）。
        stim_pattern:    WC 模式刺激模式（step / sine / square / ramp）。
        measure_window:  WC 模式稳态窗口步数（在 skip_transient 之后）。
        skip_transient:  WC 模式跳过暂态步数（None → auto stim_duration // 4）。
        pre_steps:       WC 模式基线步数（Twin 模式忽略）。
        seed:            随机种子（WC 模式 x0 采样）。
        output_dir:      保存 response_matrix.npy；None → 不保存。

    Returns:
        R: shape (n_nodes, simulator.n_regions)，响应矩阵。
    """
    if n_nodes is None:
        n_nodes = simulator.n_regions

    # Auto-compute skip_transient: at least 10 steps, at most stim_duration//2.
    # For StepStimulus with ramp_steps=10: need to skip the ramp-in.
    # For SinStimulus (bell envelope): peak is at middle of duration, so skip 25%.
    if skip_transient is None:
        skip_transient = max(10, stim_duration // 4)
    skip_transient = int(skip_transient)

    # Ensure the measurement window fits within the stim duration.
    available = max(1, stim_duration - skip_transient)
    effective_window = min(measure_window, available)
    if effective_window < measure_window:
        logger.warning(
            "measure_window=%d 超出可用稳态窗口 %d (stim_duration=%d, "
            "skip_transient=%d)，实际使用 %d 步。",
            measure_window, available, stim_duration, skip_transient, effective_window,
        )

    R = np.zeros((n_nodes, simulator.n_regions), dtype=np.float32)
    rng = np.random.default_rng(seed)

    # ── Branch: TwinBrainDigitalTwin mode ──────────────────────────────────────
    # In twin mode, the rollout-based approach (pre-phase → stim-phase) stores
    # the PERTURBED prediction (result["perturbed"]) rather than the pure causal
    # effect.  The perturbed prediction still contains the model's natural-dynamics
    # component, which is identical for all rows since all rows share the same
    # base_graph context.  This common component creates column-dominant structure
    # ("vertical stripes") in R that masks the stimulus-selective row patterns.
    #
    # Correct approach for twin mode: call simulate_intervention() directly on the
    # trimmed base_graph and use causal_effect = perturbed − baseline (computed in
    # one encoder pass, no context drift).  This mirrors compute_effective_connectivity()
    # in GraphNativeBrainModel and gives a clean row-varying response matrix.
    is_twin = getattr(simulator, "_is_twin", False)
    if is_twin:
        logger.info(
            "TwinBrain 模式：使用 simulate_intervention 直接计算因果效应（causal_effect）\n"
            "  → 避免 rollout 模式中自然动力学分量掩盖刺激特异性（竖条纹问题）\n"
            "  n_nodes=%d, delta=%.2f sigma (= stim_amplitude × %.1f)",
            n_nodes, stim_amplitude * _STIM_AMP_TO_LATENT_SIGMA, _STIM_AMP_TO_LATENT_SIGMA,
        )
        try:
            import torch
            # Trim base_graph ONCE before the n_nodes loop.
            # simulate_intervention() only READS from the context (it clones
            # internally when building h_perturbed), so we can safely reuse
            # the same trimmed context for all n_nodes rows.
            context = simulator._trim_context(copy.deepcopy(simulator.base_graph))
            modality = simulator.modality
            delta = float(stim_amplitude) * _STIM_AMP_TO_LATENT_SIGMA

            for i in range(n_nodes):
                result = simulator.model.simulate_intervention(
                    baseline_data=context,
                    interventions={modality: ([i], delta)},
                )
                effect = result["causal_effect"].get(modality)  # [N, steps, C]
                if effect is not None:
                    # Mean over prediction steps; squeeze last dim (C=1 for fMRI).
                    R[i] = effect.detach().squeeze(-1).mean(dim=1).cpu().numpy()

                if (i + 1) % max(1, n_nodes // 10) == 0:
                    logger.info("  %d/%d 节点完成 (twin模式)", i + 1, n_nodes)
        except Exception as exc:
            logger.warning(
                "TwinBrain 直接模式失败（%s: %s），回退到 rollout 模式。",
                type(exc).__name__, exc,
            )
            is_twin = False  # fall through to WC rollout path below

    # ── Branch: WC / fallback rollout mode ─────────────────────────────────────
    if not is_twin:
        # Sample ONE shared initial state for all rows.
        # Using the same x0 (hence the same equilibrium) for every row ensures that
        # R[i,j] reflects only the stimulus-propagation structure (W column i), not
        # the interaction between different random equilibria and stimulation.
        # Previously each row used a fresh x0, which caused row norms to vary
        # dramatically whenever the stimulated node's equilibrium sat near the [0,1]
        # boundary (tanh saturation), making the matrix non-comparable across rows.
        x0 = rng.random(simulator.n_regions).astype(np.float32)

        logger.info(
            "WC 模式：rollout 响应矩阵计算: n_nodes=%d, stim_duration=%d, pattern=%s, "
            "skip_transient=%d, measure_window=%d",
            n_nodes, stim_duration, stim_pattern, skip_transient, effective_window,
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

            # Baseline: mean across pre-stimulation phase (equilibrium = x0 in WC mode)
            baseline = result.pre_trajectory.mean(axis=0)

            # Steady-state response: skip the transient ramp-in, then average the
            # plateau.  This avoids the "all-rows-look-the-same" artefact that arises
            # when the stimulus is still small and propagation has not yet built up.
            win_start = skip_transient
            win_end = win_start + effective_window
            win_traj = result.stim_trajectory[win_start:win_end]
            if len(win_traj) == 0:
                win_traj = result.stim_trajectory  # fallback
            response = win_traj.mean(axis=0) - baseline

            R[i] = response

            if (i + 1) % max(1, n_nodes // 10) == 0:
                logger.info("  %d/%d 节点完成 (WC模式)", i + 1, n_nodes)

    col_mean = np.abs(R).mean(axis=0)
    stim_specificity = R.std(axis=1)

    logger.info(
        "✓ 响应矩阵完成。  全局均值=%.4f  最大绝对值=%.4f\n"
        "     列均值 mean=%.4f std=%.4f  (hub节点 ±2σ: %d个)\n"
        "     刺激特异性 mean=%.4f std=%.4f  (低特异性节点: %d个)\n"
        "     若仍出现竖条纹：\n"
        "       Twin模式 → 网络存在枢纽节点主导因果传播（真实现象），见行归一化面板B\n"
        "       WC 模式  → 增大 stim_duration 或 skip_transient 以采集稳态响应",
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
