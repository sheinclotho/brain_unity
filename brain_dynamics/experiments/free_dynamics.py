"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 确定可用的时序窗口数（见下文时序窗口机制）
2. 采样 n_init 个随机初始状态
3. 轨迹 i 使用窗口 i % n_windows 作为历史上下文（不同历史片段）
4. 将 x0 注入选定上下文的最后一个时间步
5. 运行 steps 步自回归预测，记录轨迹

输出文件：outputs/trajectories.npy

**时序窗口机制（解决 Wolf 上下文稀释偏差）**：

  以前所有轨迹共享同一个历史上下文（base_graph 最后 context_length 步），
  只有最后 1 步被不同 x0 覆盖。注意力机制将这 1 步扰动稀释在 L-1 个相同历史中，
  导致所有轨迹几乎相同（Wolf std ≈ 1.85e-05，见 AGENTS.md §Wolf上下文稀释偏差）。

  本模块使用 **滑窗策略**：步长 stride = context_length // 4（75% 重叠），
  从主模态时序 T_primary 中提取多个历史窗口：

    窗口 0：x[:, T-L:T, :]               （最近历史，原有行为）
    窗口 1：x[:, T-L-s:T-s, :]           （s = stride = L // 4）
    窗口 k：x[:, T-L-k*s:T-k*s, :]

  相比非重叠分块（要求 T ≥ k × L），滑窗方案只需 T > L + stride 即可产生第 2
  个窗口。对于典型 10 分钟 fMRI（300 TR、L=200、s=50），可得 3 个窗口，
  显著提升轨迹多样性。

  **模态感知**：窗口数仅由主模态（fmri/eeg）的时序长度决定。次要模态（如
  joint 模式中的 EEG）若时序较短，_get_context_for_window 会自动回退到
  最早可用窗口，不影响主模态的窗口计数。
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

    本模块使用 **滑窗策略**（stride = context_length // 4，75% 重叠）从主模态
    时序中提取多个历史窗口。只要主模态时序 ``T > context_length + stride``
    （即 ``T > 1.25 × context_length``），即可得到 ≥2 个窗口：

      轨迹 i 使用窗口 ``i % n_windows``，即 ``x[:, T-L-k*s:T-k*s, :]``。

    与以往要求 ``T ≥ n_windows × context_length``（严格非重叠分块）不同，
    滑窗方案能从更短的时序数据中提取更多有效历史上下文，大幅降低多样性对数据
    长度的要求。

    **模态感知**：窗口数仅由 **主模态**（fmri/eeg，非所有模态的最小值）决定，
    避免次要模态短时序错误瓶颈主模态窗口计数。

    ``n_temporal_windows=None``（默认）：自动使用 ``simulator.n_temporal_windows``。
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
                "  ⚠  n_temporal_windows=%d 超过可用滑窗数 %d"
                "（主模态时序 T 不足以支撑该数量的 stride=context_length//4 滑窗），"
                "实际使用 %d 个窗口。",
                n_temporal_windows, max_available, eff_windows,
            )

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, 预计输出大小=%.1f MiB",
        n_init, steps, n_regions, est_mb,
    )

    if eff_windows > 1:
        ctx_len = simulator._get_context_length()
        stride = max(1, ctx_len // 4)
        _nt = simulator.modality if simulator.modality != "joint" else "fmri"
        _T_primary = (
            int(simulator.base_graph[_nt].x.shape[1])
            if _nt in simulator.base_graph.node_types
            and hasattr(simulator.base_graph[_nt], "x")
            else 0
        )
        if _T_primary <= ctx_len:
            # Fallback path: prediction_steps-based stride gave us multiple
            # shorter-context windows.  Each window uses a different-length
            # initial context (window k starts with T - k*stride timesteps).
            logger.info(
                "  时序窗口: 使用 %d 个变长上下文窗口（回退模式）。\n"
                "  主模态 '%s' 时序 T=%d ≤ context_length=%d，"
                "使用 prediction_steps 步幅提取多个较短初始上下文：\n"
                "    窗口 0 → %d 步上下文（最完整）\n"
                "    窗口 1 → %d 步上下文\n"
                "    …\n"
                "    窗口 %d → %d 步上下文（最短）\n"
                "  初始上下文质量随窗口编号递减，但在约 %d 步后自回归预测会"
                "补充到完整 context_length 步，对长轨迹（steps=%d）影响极小。\n"
                "  轨迹 i → 窗口 i %% %d。",
                eff_windows,
                _nt, _T_primary, ctx_len,
                _T_primary,
                max(0, _T_primary - stride),
                eff_windows - 1, max(0, _T_primary - (eff_windows - 1) * stride),
                ctx_len,
                steps,
                eff_windows,
            )
        else:
            logger.info(
                "  时序窗口: 使用 %d 个滑窗上下文"
                "（context_length=%d，stride=%d，%.0f%% 重叠；"
                "轨迹 i → 窗口 i %% %d）。\n"
                "  每条轨迹从真正不同的历史时段出发，有效缓解上下文稀释效应。",
                eff_windows, ctx_len, stride, (1 - stride / ctx_len) * 100, eff_windows,
            )
    else:
        ctx_len = simulator._get_context_length()
        stride = max(1, ctx_len // 4)
        # Determine primary modality T for the diagnostic message.
        _nt = simulator.modality if simulator.modality != "joint" else "fmri"
        _T_primary = (
            int(simulator.base_graph[_nt].x.shape[1])
            if _nt in simulator.base_graph.node_types
            and hasattr(simulator.base_graph[_nt], "x")
            else 0
        )
        logger.info(
            "  时序窗口: 仅使用 1 个上下文窗口"
            "（主模态 '%s' 时序 T=%d，context_length=%d，stride=%d）。\n"
            "  当前多样性来源: %d 个不同随机初始状态 Uniform[0,1]^%d。\n"
            "  若 context_length（%d）远大于模型实际所需，请检查 config.yaml 中\n"
            "  v5_optimization.advanced_prediction.context_length 是否与\n"
            "  训练时一致（当前从 predictor.context_length=%d 读取）。",
            _nt, _T_primary, ctx_len, stride,
            n_init, n_regions,
            ctx_len, ctx_len,
        )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    log_interval = max(1, n_init // 10)

    # ── Compute context-end-aligned step indices for each temporal window ────
    # For temporal window k:
    #   context_window_k  = data[:, T - ctx - k*s : T - k*s]
    #   x0_natural        = data[:, T - k*s]       (step immediately after context)
    #
    # Injecting x0 = data[:, T - k*s] makes context + x0 a CONTIGUOUS BOLD
    # segment, so the model encoder sees internally consistent temporal structure
    # with zero spatial-structure mismatch.  This eliminates the long
    # "correction transient" that otherwise dominates PC1.
    #
    # For window 0: T - 0*s = T, clamped to T-1 (last available BOLD step).
    # The resulting x0 = context[-1] effectively re-uses the last context step,
    # which is identical to running with x0=None — the model immediately
    # produces a natural continuation without any correction.
    try:
        _stride_fd  = simulator._get_stride()   # canonical: max(1, ctx_len // 4)
        _nt_fd = simulator.modality if simulator.modality != "joint" else "fmri"
        _T_fd = (int(simulator.base_graph[_nt_fd].x.shape[1])
                 if _nt_fd in simulator.base_graph.node_types
                 and hasattr(simulator.base_graph[_nt_fd], "x")
                 else 0)
    except Exception:
        _stride_fd  = 50
        _T_fd       = 0

    def _x0_step_for_window(w_idx: int) -> Optional[int]:
        """Return the context-aligned step index for temporal window w_idx."""
        if _T_fd <= 0:
            return None  # no data available; fall back to random step
        step = _T_fd - w_idx * _stride_fd
        return int(max(0, min(step, _T_fd - 1)))

    for i in range(n_init):
        window_idx = i % eff_windows
        # Context-end-aligned initial state: x0 is the BOLD step that immediately
        # follows the context window being used.  This makes context+x0 a
        # temporally contiguous segment, so the model predicts naturally from the
        # first step, producing oscillatory/chaotic orbits rather than a long
        # correction drift toward the free-run attractor.
        x0_step = _x0_step_for_window(window_idx)
        x0 = simulator.sample_random_state(rng, from_data=True, step_idx=x0_step)
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
