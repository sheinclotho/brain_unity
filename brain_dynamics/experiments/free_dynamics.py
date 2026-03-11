"""
Free Dynamics Experiment
========================

验证系统在 **无输入情况下** 的长期行为。

流程：
1. 确定可用的随机起始位置数量
2. 为每条轨迹随机采样一个等长上下文窗口（context_start ∈ [0, T - ctx_len]）
3. x0 = 上下文末步对应的数据值（上下文 + x0 在时序上连续）
4. 运行 steps 步自回归预测，记录轨迹

输出文件：outputs/trajectories.npy

**随机等长上下文窗口策略（替换旧的变长回退策略）**：

  所有轨迹的上下文长度完全相同（= context_length 步），区别仅在于起始位置。
  对每条轨迹 i，随机采样 context_start ∈ [0, max(0, T - ctx_len)]：

    轨迹 i → context_start_i = rand([0, T - ctx_len])
            → 上下文 = data[:, context_start_i : context_start_i + ctx_len]
            → x0     = data[:, context_start_i + ctx_len - 1]（时序连续）

  这与旧的"prediction_steps 步幅回退"策略有本质区别：
  旧策略产生 window0=T步, window1=T-s步, … 等 **不同长度** 的上下文，
  导致编码器的初始输入质量系统性不均等，干扰后续动力学分析。
  新策略确保所有轨迹经历相同质量的初始化。

  **当 T ≤ context_length 时**（如本次 T=200, ctx_len=200）：
  只有一个有效起始位置（context_start=0），上下文 = data[0:T]。
  100 条轨迹的多样性完全来自 x0 注入噪声（0.3σ 扰动）。
  在近临界系统（LLE ≈ 0）中，0.3σ 扰动足以驱散轨迹覆盖吸引子。

**多图上下文多样性策略（graph_paths 参数）**：

  当提供多个图缓存文件时（通过 `--graph 文件夹路径` 传入），
  每条轨迹从不同文件的 BOLD 时序作为初始上下文，彻底解决单文件
  T = context_length 时多样性不足的问题。

  图池分配规则：
    - 轨迹 i 使用 graph_paths[i % len(pool)] 对应的上下文
    - 图文件数 > 轨迹数：前 n_init 个文件各用一次，不循环
    - 图文件数 < 轨迹数：循环使用所有文件
    - 节点数不匹配的文件自动跳过并打印警告

  每次切换仅修改 simulator.base_graph[nt].x（CPU 张量），不加载到 GPU，
  不超过任何显存限制。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

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


# ── Multi-graph context pool helpers ──────────────────────────────────────────

def _load_graph_x_only(
    path: Path,
    required_nts: List[str],
    n_regions_primary: int,
    nt_primary: str,
    n_nodes_per_nt: Optional[Dict[str, int]] = None,
) -> Optional[Dict[str, "torch.Tensor"]]:
    """Load **only** the ``.x`` tensors from a graph-cache file on CPU.

    Returns a ``{node_type: tensor}`` dict on success, or ``None`` when the
    file is unreadable, not a ``HeteroData``, or any required modality's node
    count does not match the primary base-graph's node count for that type.

    Loading only ``.x`` data (no edge rebuilding) keeps memory use minimal
    and avoids the expensive Pearson-correlation edge-reconstruction step.
    Edges used during rollout always come from the *primary* graph; swapping
    only ``.x`` provides genuine BOLD-history diversity while preserving the
    structural connectivity of the primary subject.

    Args:
        path:              Path to the graph-cache ``.pt`` file.
        required_nts:      Node types to collect (must all be present).
        n_regions_primary: Expected node count for *nt_primary*.
        nt_primary:        The primary modality node type (e.g. ``'fmri'``).
        n_nodes_per_nt:    Expected node count for **every** required node type,
                           keyed by node-type string.  When provided, all
                           secondary modalities are validated against these
                           counts — this prevents CUDA index-out-of-bounds
                           (``srcIndex < srcSelectDimSize``) that occurs when a
                           secondary modality (e.g. ``'eeg'``) in a pool graph
                           has fewer nodes than the primary graph's edge_index
                           was built for.
    """
    try:
        from torch_geometric.data import HeteroData
        g = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(g, HeteroData):
            return None
        if nt_primary not in g.node_types:
            return None
        g_x = getattr(g[nt_primary], "x", None)
        if g_x is None or g_x.ndim != 3:
            return None
        if g_x.shape[0] != n_regions_primary:
            return None
        # Collect x-data for every node type also present in the primary graph.
        # CRITICAL: validate the node count for every secondary modality against
        # the expected count from the primary base-graph.  The primary graph's
        # edge_index was built for a specific N per modality; swapping in a
        # secondary graph with a different N causes edge indices to exceed the
        # feature-tensor dimension, triggering CUDA assertion
        # ``srcIndex < srcSelectDimSize`` during GNN message passing.
        x_data: Dict[str, torch.Tensor] = {}
        for nt in required_nts:
            if nt not in g.node_types:
                continue
            nt_x = getattr(g[nt], "x", None)
            if nt_x is None:
                continue
            # Validate node count for secondary modalities when expected counts
            # are provided.  Primary modality is already validated above.
            if n_nodes_per_nt is not None and nt != nt_primary:
                expected_n = n_nodes_per_nt.get(nt)
                if expected_n is not None and nt_x.shape[0] != expected_n:
                    logger.debug(
                        "_load_graph_x_only: skipping '%s' — node type '%s' "
                        "has N=%d but primary graph expects N=%d; "
                        "mismatched secondary modality would cause CUDA "
                        "index-out-of-bounds during GNN message passing.",
                        path.name, nt, nt_x.shape[0], expected_n,
                    )
                    return None
            # .clone() detaches from the loaded graph so the rest can be GC'd.
            x_data[nt] = nt_x.clone()
        return x_data if x_data else None
    except (FileNotFoundError, IsADirectoryError, OSError,
            RuntimeError, AttributeError, TypeError,
            Exception) as exc:  # noqa: BLE001 — include broad catch for torch unpickling
        logger.debug("_load_graph_x_only: failed to load '%s': %s", path, exc)
        return None


def _build_graph_pool(
    graph_paths: Optional[List[Path]],
    simulator: BrainDynamicsSimulator,
    nt_primary: str,
) -> List[Optional[Dict[str, "torch.Tensor"]]]:
    """Build the context-swap pool from *graph_paths*.

    Returns a list whose elements are either:
      * ``None``                    — use the primary base_graph (no swap)
      * ``dict[node_type, tensor]`` — x-data to temporarily substitute

    Index 0 is always ``None`` (primary graph already in *simulator*).
    Indices 1+ are loaded from ``graph_paths[1:]``.
    Files that fail to load or have mismatched node counts are skipped.
    """
    if not graph_paths or len(graph_paths) <= 1:
        return [None]  # single-graph mode: no swapping

    required_nts = [
        nt for nt in simulator.base_graph.node_types
        if hasattr(simulator.base_graph[nt], "x")
    ]
    # IMPORTANT: use the primary modality's node count (e.g. fmri=190),
    # NOT simulator.n_regions in joint mode (fmri+eeg, e.g. 253).
    # Multi-graph compatibility must be validated against the context modality
    # actually swapped during free dynamics initialisation.
    try:
        n_regions_primary = int(simulator.base_graph[nt_primary].x.shape[0])
    except (AttributeError, KeyError, TypeError) as exc:
        # Defensive fallback for malformed base_graph objects.
        logger.warning(
            "  多图上下文: 无法从主模态 '%s' 提取节点数，"
            "回退到 simulator.n_regions=%s 进行兼容校验。(%s: %s)",
            nt_primary,
            getattr(simulator, "n_regions", "unknown"),
            type(exc).__name__,
            exc,
        )
        n_regions_primary = int(simulator.n_regions)

    # Build per-node-type expected counts from the primary base_graph.
    # This is used by _load_graph_x_only to validate ALL modalities, not just
    # the primary.  Without this, a secondary pool graph with a different N_eeg
    # would be accepted but its swapped .x would be incompatible with the primary
    # graph's edge_index, causing CUDA ``srcIndex < srcSelectDimSize`` assertion.
    n_nodes_per_nt: Dict[str, int] = {}
    for nt in required_nts:
        try:
            n_nodes_per_nt[nt] = int(simulator.base_graph[nt].x.shape[0])
        except (AttributeError, KeyError, TypeError):
            pass

    pool: List[Optional[Dict[str, torch.Tensor]]] = [None]  # index 0 = primary

    n_ok = 0
    n_skip = 0
    for path in graph_paths[1:]:
        path = Path(path)  # normalise: accept both str and Path
        x_data = _load_graph_x_only(
            path, required_nts, n_regions_primary, nt_primary,
            n_nodes_per_nt=n_nodes_per_nt,
        )
        if x_data is not None:
            pool.append(x_data)
            n_ok += 1
        else:
            logger.warning(
                "  多图上下文: 跳过图文件 '%s' "
                "（无法加载、格式不符或任意模态节点数不匹配）。",
                path.name,
            )
            n_skip += 1

    if n_ok:
        logger.info(
            "  多图上下文池: 成功加载 %d 个额外图文件 x 数据（均在 CPU 内存，未占用显存），"
            "共 %d 个条目（含主图）；跳过 %d 个文件。",
            n_ok, len(pool), n_skip,
        )
    elif n_skip:
        logger.warning(
            "  多图上下文: 所有 %d 个额外图文件均无法加载，"
            "回退到单图模式。",
            n_skip,
        )

    return pool


def _swap_base_graph_x(
    simulator: BrainDynamicsSimulator,
    new_x_data: Dict[str, "torch.Tensor"],
) -> Dict[str, "torch.Tensor"]:
    """Temporarily replace ``simulator.base_graph[nt].x`` with *new_x_data*.

    Returns a dict of the original tensors so the caller can restore them.
    Only node types present in both *new_x_data* and ``base_graph`` are swapped.

    The replacement tensor is moved to the **same device** as the original so
    that downstream code (``_clone_hetero_graph``, model forward pass) never
    sees a mixed-device HeteroData.  Pool data is loaded on CPU by
    ``_load_graph_x_only``; moving it to the original device here (rather than
    relying on ``data.to(device)`` deep inside ``predict_future``) eliminates
    the root cause of CUDA index-assertion errors that arise when GPU edge
    indices are paired with CPU node features during intermediate tensor ops.
    """
    saved: Dict[str, torch.Tensor] = {}
    for nt, new_x in new_x_data.items():
        if nt in simulator.base_graph.node_types and hasattr(
            simulator.base_graph[nt], "x"
        ):
            orig_x = simulator.base_graph[nt].x
            saved[nt] = orig_x
            # Ensure pool data lands on the same device as the original tensor.
            simulator.base_graph[nt].x = new_x.to(orig_x.device)
    return saved


def _restore_base_graph_x(
    simulator: BrainDynamicsSimulator,
    saved_x: Dict[str, "torch.Tensor"],
) -> None:
    """Restore ``simulator.base_graph[nt].x`` from *saved_x*."""
    for nt, orig_x in saved_x.items():
        if nt in simulator.base_graph.node_types:
            simulator.base_graph[nt].x = orig_x


def _get_T_primary(simulator: BrainDynamicsSimulator, nt: str) -> int:
    """Return the current temporal length of ``simulator.base_graph[nt].x``."""
    try:
        if nt in simulator.base_graph.node_types and hasattr(
            simulator.base_graph[nt], "x"
        ):
            return int(simulator.base_graph[nt].x.shape[1])
    except (AttributeError, IndexError, TypeError):
        pass
    return 0


def run_free_dynamics(
    simulator: BrainDynamicsSimulator,
    n_init: int = 200,
    steps: int = 1000,
    seed: int = 42,
    output_dir: Optional[Path] = None,
    device: Optional[str] = None,
    n_temporal_windows: Optional[int] = None,
    graph_paths: Optional[List[Path]] = None,
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

    **多图上下文多样性（graph_paths）**：

    当 ``graph_paths`` 非 None 时，使用多个图文件的 BOLD 时序作为上下文，
    彻底解决单文件 T ≤ context_length 时多样性不足的问题。
    详见模块文档字符串。

    Args:
        simulator:           BrainDynamicsSimulator 实例（TwinBrainDigitalTwin 模式）。
        n_init:              随机初始状态数量（默认 200）。
        steps:               每条轨迹的模拟步数（默认 1000）。
        seed:                随机种子，确保可重复性。
        output_dir:          若指定，将结果保存为 trajectories.npy；None → 不保存。
        device:              保留参数（兼容性），实际设备由模型决定。
        n_temporal_windows:  使用的时序窗口数（None = 自动，1 = 禁用多窗口）。
        graph_paths:         可选：所有图缓存文件的路径列表（含主图 graph_paths[0]）。
                             提供时启用多图上下文多样性：轨迹 i 使用
                             ``graph_paths[i % len(pool)]`` 对应的 BOLD 数据作为
                             上下文。None → 单图模式（向后兼容默认值）。

    Returns:
        trajectories: shape (n_init, steps, n_regions)，所有轨迹。
    """
    rng = np.random.default_rng(seed)
    n_regions = simulator.n_regions

    # ── Determine primary-graph data dimensions ───────────────────────────────
    ctx_len = simulator._get_context_length()
    _nt = simulator.modality if simulator.modality != "joint" else "fmri"
    T_primary = _get_T_primary(simulator, _nt)

    est_mb = _estimate_memory_mb(n_init, steps, n_regions)
    logger.info(
        "自由动力学实验: n_init=%d, steps=%d, n_regions=%d, 预计输出大小=%.1f MiB",
        n_init, steps, n_regions, est_mb,
    )

    # ── Build multi-graph pool (CPU-only, no GPU overhead) ────────────────────
    graph_pool = _build_graph_pool(graph_paths, simulator, _nt)
    use_multi_graph = len(graph_pool) > 1

    # ── Random equal-length context window strategy (single-graph) ───────────
    # Compute baseline context params from the primary graph (used when no swap
    # is in effect, or as fallback for logging).
    if T_primary > 0:
        eff_ctx = min(ctx_len, T_primary)
        max_context_start = max(1, T_primary - eff_ctx + 1)
    else:
        eff_ctx = ctx_len
        max_context_start = 1

    # ── Log initialization strategy ───────────────────────────────────────────
    if use_multi_graph:
        logger.info(
            "  初始化策略: 多图上下文多样性"
            "（%d 个图文件，%d 条轨迹循环使用）。\n"
            "  每条轨迹使用不同图文件的时序作为初始上下文，提供最大初始化多样性。",
            len(graph_pool), n_init,
        )
    elif max_context_start > 1:
        logger.info(
            "  初始化策略: 随机等长上下文窗口"
            "（%d 条轨迹各自随机截取 %d 步历史，"
            "起始位置 ∈ [0, %d]，全部等长，无偏差）。\n"
            "  x0 = 上下文窗口末步对应的数据值（时序连续）。",
            n_init, eff_ctx, max_context_start - 1,
        )
    else:
        logger.info(
            "  初始化策略: 单一上下文窗口（主模态 '%s' T=%d ≤ context_length=%d）。\n"
            "  仅有一个等长上下文 [0:%d]，%d 条轨迹的多样性来自 x0 注入噪声。\n"
            "  提示: 若需要更多轨迹多样性，请提供更长的输入时序（T > %d）"
            "  或使用 --graph 文件夹路径（含多个图文件）。",
            _nt, T_primary, ctx_len, eff_ctx, n_init, ctx_len,
        )

    # Warn if caller passed n_temporal_windows (parameter now ignored in favour
    # of the random-start strategy, but we log so the caller knows).
    if n_temporal_windows is not None and int(n_temporal_windows) != 1:
        logger.warning(
            "  n_temporal_windows=%d 参数已忽略：当前使用随机等长上下文策略，"
            "轨迹多样性由 context_start 随机采样而非固定窗口数控制。",
            n_temporal_windows,
        )

    trajectories = np.empty((n_init, steps, n_regions), dtype=np.float32)
    log_interval = max(1, n_init // 10)

    for i in range(n_init):
        # ── Select context source for this trajectory ─────────────────────────
        saved_x: Dict[str, torch.Tensor] = {}
        if use_multi_graph:
            pool_idx = i % len(graph_pool)
            swap_data = graph_pool[pool_idx]
            if swap_data is not None:
                saved_x = _swap_base_graph_x(simulator, swap_data)

        # Compute per-trajectory context params (T may differ across graphs)
        if saved_x:
            T_cur = _get_T_primary(simulator, _nt)
            eff_ctx_i = min(ctx_len, T_cur) if T_cur > 0 else ctx_len
            max_cs_i = max(1, T_cur - eff_ctx_i + 1) if T_cur > 0 else 1
        else:
            T_cur = T_primary
            eff_ctx_i = eff_ctx
            max_cs_i = max_context_start

        # Random context start position — equal length for all trajectories
        context_start = int(rng.integers(0, max_cs_i))
        # x0 aligned to the last step of the selected context window:
        #   context covers [context_start : context_start + eff_ctx_i]
        #   → last step = context_start + eff_ctx_i - 1
        # Clamped to [0, T_cur-1] so we never request an out-of-range step.
        if T_cur > 0:
            natural_x0_step = context_start + eff_ctx_i - 1
            x0_step: int = min(natural_x0_step, T_cur - 1)
            x0 = simulator.sample_random_state(rng, from_data=True, step_idx=x0_step)
        else:
            # No data available — sample_random_state uses noise only (step_idx=None)
            x0 = simulator.sample_random_state(rng, from_data=True, step_idx=None)
        traj, _ = simulator.rollout(
            x0=x0, steps=steps, stimulus=None,
            context_start=context_start,
        )
        trajectories[i] = traj

        # ── Restore primary graph's x-data after rollout ──────────────────────
        if saved_x:
            _restore_base_graph_x(simulator, saved_x)

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
