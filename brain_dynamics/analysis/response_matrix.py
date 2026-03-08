"""
Response Matrix Computation
============================

量化 **刺激传播结构** (Stimulation Propagation Structure)。

定义：
  R[i, j] = response of node j when node i is stimulated

直接调用 ``model.simulate_intervention()`` 计算因果效应
(causal_effect = perturbed − baseline)，一次 encoder 调用完成，无上下文漂移误差。

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

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    _clone_hetero_graph,
)

logger = logging.getLogger(__name__)

# Stimulus amplitude [0,1] → latent-sigma multiplier (matches _STIM_AMP_TO_LATENT_SIGMA
# in brain_dynamics_simulator.py).
_STIM_AMP_TO_LATENT_SIGMA: float = 2.0


def compute_response_matrix(
    simulator: BrainDynamicsSimulator,
    n_nodes: Optional[int] = None,
    stim_amplitude: float = 0.5,
    output_dir: Optional[Path] = None,
    # Legacy WC-only params — completely ignored, kept only so old call sites do
    # not break.  Pass any non-default value and a warning is emitted.
    stim_duration: int = 80,
    stim_frequency: float = 10.0,
    stim_pattern: str = "step",
    measure_window: int = 30,
    skip_transient: Optional[int] = None,
    pre_steps: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """
    计算完整的刺激响应矩阵 R[i, j]。

    R[i,j] 定义为：节点 j 响应节点 i 刺激的平均活动增量（因果效应）。

    直接调用 ``model.simulate_intervention()``，使用相同 encoder 一次计算
    baseline 和 perturbed，取 ``causal_effect = perturbed − baseline`` 作为 R[i,:]。
    这比 rollout 模式更准确：消除了上下文漂移误差和自然动力学分量的干扰。

    **注意：R 可能出现枢纽主导的列结构（真实现象）**：
    当网络存在少数枢纽区域（DMN、前额叶等）时，刺激任何节点都会通过图传播
    强烈影响这些枢纽，使枢纽列的响应与具体刺激位置无关。
    这是真实网络特性。可视化面板 B（行归一化）能揭示枢纽外的刺激特异性传播模式。

    Args:
        simulator:       BrainDynamicsSimulator 实例。
        n_nodes:         要刺激的节点数（None → 使用 simulator.n_regions）。
        stim_amplitude:  刺激幅度（乘以 _STIM_AMP_TO_LATENT_SIGMA=2.0 转换为潜空间 σ）。
        output_dir:      保存 response_matrix.npy；None → 不保存。

    Returns:
        R: shape (n_nodes, simulator.n_regions)，响应矩阵。
    """
    # Warn if caller passes non-default WC-only params (they have no effect)
    _wc_defaults = dict(stim_duration=80, stim_frequency=10.0, stim_pattern="step",
                        measure_window=30, skip_transient=None, pre_steps=50, seed=42)
    _wc_actual   = dict(stim_duration=stim_duration, stim_frequency=stim_frequency,
                        stim_pattern=stim_pattern, measure_window=measure_window,
                        skip_transient=skip_transient, pre_steps=pre_steps, seed=seed)
    _changed = [k for k, v in _wc_actual.items() if v != _wc_defaults[k]]
    if _changed:
        logger.warning(
            "compute_response_matrix: 以下参数 %s 是 Wilson-Cowan 模式专用参数，"
            "已随 WC 模式一同移除，当前值被忽略。请移除这些参数。",
            _changed,
        )

    if n_nodes is None:
        n_nodes = simulator.n_regions

    R = np.zeros((n_nodes, simulator.n_regions), dtype=np.float32)

    context = simulator._trim_context(_clone_hetero_graph(simulator.base_graph))
    modality = simulator.modality
    delta = float(stim_amplitude) * _STIM_AMP_TO_LATENT_SIGMA
    is_joint = (modality == "joint")

    logger.info(
        "TwinBrain 响应矩阵: n_nodes=%d, delta=%.2f σ (amplitude=%.2f × %.1f)%s",
        n_nodes, delta, stim_amplitude, _STIM_AMP_TO_LATENT_SIGMA,
        " [joint 模式: 节点索引 0..N_fmri-1=fMRI, N_fmri..N_joint-1=EEG]" if is_joint else "",
    )

    for i in range(n_nodes):
        if is_joint:
            # Map joint node index to its physical modality and channel index.
            # Joint node space: [0 .. N_fmri-1] = fMRI regions,
            #                   [N_fmri .. N_fmri+N_eeg-1] = EEG channels.
            if i < simulator.n_fmri_regions:
                stim_mod = "fmri"
                stim_node = i
            else:
                stim_mod = "eeg"
                stim_node = i - simulator.n_fmri_regions
            result = simulator.model.simulate_intervention(
                baseline_data=context,
                interventions={stim_mod: ([stim_node], delta)},
            )
            # Build joint z-normalised response from both modalities' causal effects
            fmri_effect = result["causal_effect"].get("fmri")
            eeg_effect  = result["causal_effect"].get("eeg")
            if fmri_effect is not None and eeg_effect is not None:
                fmri_vals = fmri_effect.detach().squeeze(-1).mean(dim=1).cpu().numpy()
                eeg_vals  = eeg_effect.detach().squeeze(-1).mean(dim=1).cpu().numpy()
                # Z-normalise each modality using simulator's pre-computed stats
                fmri_z = (fmri_vals - simulator._fmri_mean) / simulator._fmri_std
                eeg_z  = (eeg_vals  - simulator._eeg_mean)  / simulator._eeg_std
                R[i]   = np.concatenate([fmri_z, eeg_z])
        else:
            result = simulator.model.simulate_intervention(
                baseline_data=context,
                interventions={modality: ([i], delta)},
            )
            effect = result["causal_effect"].get(modality)  # [N, steps, C]
            if effect is not None:
                # Mean over prediction steps; squeeze last dim (C=1 for fMRI).
                R[i] = effect.detach().squeeze(-1).mean(dim=1).cpu().numpy()

        if (i + 1) % max(1, n_nodes // 10) == 0:
            logger.info("  %d/%d 节点完成", i + 1, n_nodes)

    col_mean = np.abs(R).mean(axis=0)
    stim_specificity = R.std(axis=1)

    logger.info(
        "✓ 响应矩阵完成。  全局均值=%.4f  最大绝对值=%.4f\n"
        "     列均值 mean=%.4f std=%.4f  (hub节点 ±2σ: %d个)\n"
        "     刺激特异性 mean=%.4f std=%.4f  (低特异性节点: %d个)\n"
        "     若出现枢纽主导的列结构：这是真实网络特性，见行归一化面板B",
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

