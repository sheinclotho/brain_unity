"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 采样 n_init 个随机初始状态
2. 从每个初始状态进行长时间自由演化（steps 步）
3. 记录并返回所有状态轨迹

输出文件：outputs/trajectories.npy

GPU 加速：当 device 为 ``"cuda"`` 时（或 ``simulator.device`` 为 CUDA），
Wilson-Cowan 模式自动使用 ``BrainDynamicsSimulator.rollout_batched()``，
将所有 n_init 条轨迹并行在 GPU 上计算，并通过分块传输避免 OOM。
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Allow running as a standalone script or as an imported module
import sys
_HERE = Path(__file__).parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from simulator.brain_dynamics_simulator import (
    BrainDynamicsSimulator,
    _compute_chunk_steps,
    _resolve_device,
)

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
    chunk_size: Optional[int] = None,
) -> np.ndarray:
    """
    运行自由动力学实验。

    **GPU 加速**：在 Wilson-Cowan 模式（``model=None``）下，当 ``device`` 为
    ``"cuda"``（或 ``simulator.device`` 为 CUDA）时，自动调用
    ``BrainDynamicsSimulator.rollout_batched()``，将所有 n_init 条轨迹
    **并行**在 GPU 上计算，大幅缩短大规模实验的运行时间。

    内存管理：
      - 每个批次（``chunk_size`` 条轨迹）的轨迹数据在 GPU 上按步骤分块生成，
        每块计算完成后立即传回 CPU，避免 VRAM OOM。
      - ``chunk_size`` 控制 GPU 上同时处理的轨迹数；``None`` → 全部一次性处理。

    在**模型模式**下（TwinBrainDigitalTwin），因自回归推断需要逐条进行，
    仍沿用原有序列化流程。

    Args:
        simulator:   BrainDynamicsSimulator 实例。
        n_init:      随机初始状态数量（默认 200）。
        steps:       每条轨迹的模拟步数（默认 1000）。
        seed:        随机种子，确保可重复性。
        output_dir:  若指定，将结果保存为 trajectories.npy；
                     None → 不保存。
        device:      计算设备（``"cpu"``、``"cuda"``、``"auto"``）；
                     ``None`` → 使用 ``simulator.device``。
        chunk_size:  GPU 批量处理的轨迹数（``None`` → 一次性处理全部 n_init 条）。
                     对于 VRAM 不足的 GPU，可将此值减小（例如 32 或 64）。

    Returns:
        trajectories: shape (n_init, steps, n_regions)，所有轨迹。
    """
    rng = np.random.default_rng(seed)
    n_regions = simulator.n_regions

    # Resolve target device
    _device = _resolve_device(
        device if device is not None else getattr(simulator, "device", "cpu")
    )

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, device=%s, "
        "预计输出大小=%.1f MiB",
        n_init,
        steps,
        n_regions,
        _device,
        est_mb,
    )

    # ── WC mode: use GPU-accelerated batched rollout ──────────────────────────
    if not simulator._is_twin and not simulator._use_model:
        X0 = rng.random((n_init, n_regions)).astype(np.float32)  # (n_init, N)

        if chunk_size is None:
            # All trajectories at once
            logger.info("  → 使用 GPU 批量 rollout (device=%s, n_batch=%d)", _device, n_init)
            trajectories, _ = simulator.rollout_batched(
                X0, steps=steps, stimulus=None, device=_device
            )
        else:
            # Process in chunks to limit GPU VRAM usage
            n_chunks = (n_init + chunk_size - 1) // chunk_size
            logger.info(
                "  → 分块 GPU 批量 rollout (device=%s, chunk_size=%d, n_chunks=%d)",
                _device,
                chunk_size,
                n_chunks,
            )
            chunks: List[np.ndarray] = []
            for chunk_idx in range(n_chunks):
                lo = chunk_idx * chunk_size
                hi = min(lo + chunk_size, n_init)
                chunk_traj, _ = simulator.rollout_batched(
                    X0[lo:hi], steps=steps, stimulus=None, device=_device
                )
                chunks.append(chunk_traj)
                logger.info(
                    "  批次 %d/%d 完成 (%d 条轨迹)",
                    chunk_idx + 1,
                    n_chunks,
                    hi - lo,
                )
            trajectories = np.concatenate(chunks, axis=0)  # (n_init, steps, N)

    # ── Model mode: sequential (autoregressive context-window inference) ──────
    else:
        trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
        log_interval = max(1, n_init // 10)
        for i in range(n_init):
            x0 = rng.random(n_regions).astype(np.float32)
            traj, _ = simulator.rollout(x0=x0, steps=steps, stimulus=None)
            trajectories[i] = traj
            if (i + 1) % log_interval == 0:
                logger.info("  %d/%d 初始状态完成", i + 1, n_init)
            # Proactively release any cached (but unused) GPU memory so that
            # fragmentation from the previous rollout doesn't cause OOM on the next.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
