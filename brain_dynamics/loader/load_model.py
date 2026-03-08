"""
Model Loader
============

加载训练完成的 TwinBrain V5 模型，返回 ``TwinBrainDigitalTwin`` 推理引擎实例。

设计原则:
- 使用 ``TwinBrainDigitalTwin.from_checkpoint()`` 加载，自动推断模型架构
- 自动查找检查点同目录的 ``config.yaml``（训练时同步生成），
  在 state_dict 键名不足以推断架构时作为备用架构信息传入
- 不允许修改模型结构
- 不参与任何梯度计算（所有推理在 ``torch.no_grad()`` 下执行）
- 禁止任何形式的 fallback：文件不存在或格式错误 → 抛出异常
- 图缓存加载辅助函数 ``load_graph_for_inference()`` 自动重建跨模态边
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch_geometric.data import HeteroData

logger = logging.getLogger(__name__)

# ── Ensure repo root is importable (models/, utils/ etc.) ────────────────────
# twinbrain-dynamics/loader/ → twinbrain-dynamics/ → repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _load_config_yaml(config_path: Path) -> Dict[str, Any]:
    """
    加载训练时生成的 ``config.yaml`` 配置快照。

    文件由训练管线自动保存到检查点同目录，包含模型架构参数
    （``model.hidden_channels``、``model.prediction_steps`` 等）以及
    数据模态列表，可在 state_dict 键名不足以推断架构时提供完整信息。

    Args:
        config_path: ``config.yaml`` 文件路径。

    Returns:
        解析后的 config dict；若文件不存在或无法解析则返回空 dict（非致命）。
    """
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        logger.info("读取训练配置: %s", config_path)
        model_sec = cfg.get("model", {})
        logger.info(
            "  model: hidden_channels=%s, prediction_steps=%s, type=%s",
            model_sec.get("hidden_channels"),
            model_sec.get("prediction_steps"),
            model_sec.get("type"),
        )
        return cfg
    except ImportError:
        logger.warning(
            "PyYAML 未安装，无法读取 config.yaml（pip install pyyaml）。"
            " 将仅依赖 state_dict 键名推断架构。"
        )
        return {}
    except Exception as exc:
        logger.warning("config.yaml 读取失败 (%s)，忽略。", exc)
        return {}


def load_trained_model(
    checkpoint_path: Union[str, Path],
    device: str = "cpu",
    subject_to_idx_path: Optional[Union[str, Path]] = None,
    config_path: Optional[Union[str, Path]] = None,
) -> "TwinBrainDigitalTwin":  # noqa: F821
    """
    加载训练完成的 TwinBrain V5 模型，返回 ``TwinBrainDigitalTwin``。

    加载策略（顺序执行）：

    1. 自动在检查点同目录查找 ``config.yaml``（训练时同步生成）；
       若 ``config_path`` 已显式指定，则优先使用。
    2. 将解析后的 config dict 传给
       ``TwinBrainDigitalTwin.from_checkpoint(config=...)``，
       用于在 state_dict 键名不足时辅助重建模型架构
       （如 ``model.hidden_channels``、``model.prediction_steps``）。
    3. 加载成功后冻结所有参数（``eval()`` 模式），可直接用于推理。

    Args:
        checkpoint_path:     检查点文件路径（``best_model.pt`` 或任意训练检查点）。
        device:              计算设备：``"cpu"``、``"cuda"`` 或 ``"auto"``。
                             ``"auto"`` 自动选择 CUDA（若可用）。
        subject_to_idx_path: ``subject_to_idx.json`` 路径；省略时自动在
                             检查点同目录查找。
        config_path:         训练配置文件路径（``config.yaml``）。省略时自动在
                             检查点同目录查找。若找不到则仅依赖 state_dict
                             键名推断架构。

    Returns:
        ``TwinBrainDigitalTwin`` — 推理就绪的数字孪生实例
        （``model.eval()``，参数已冻结）。

    Raises:
        FileNotFoundError: 检查点文件不存在。
        RuntimeError:      模型加载或架构推断失败（格式错误、缺少必要键等）。
        ImportError:       ``models`` 包无法导入（确认 PYTHONPATH 包含仓库根目录）。

    Notes:
        检查点必须是 TwinBrain 训练管线保存的标准格式（含 ``model_state_dict`` 键）。
        直接序列化的 ``nn.Module`` 或非标准 dict 不被支持；请使用
        ``outputs/<experiment>/best_model.pt``。
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"检查点文件不存在: {checkpoint_path}\n"
            "提示: 训练完成的模型通常位于 outputs/<experiment_name>/best_model.pt"
        )

    if checkpoint_path.suffix not in {".pt", ".pth"}:
        logger.warning("文件扩展名不常见 (%s)，尝试继续加载。", checkpoint_path.suffix)

    try:
        from models.digital_twin_inference import TwinBrainDigitalTwin
    except ImportError as exc:
        raise ImportError(
            f"无法导入 TwinBrainDigitalTwin: {exc}\n"
            f"请确保仓库根目录（{_REPO_ROOT}）在 PYTHONPATH 中，"
            "或从仓库根目录运行本脚本。"
        ) from exc

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Load config.yaml (generated alongside each model checkpoint) ──────────
    # Priority: explicit config_path > auto-detection in checkpoint directory
    if config_path is not None:
        _cfg_path = Path(config_path)
    else:
        _cfg_path = checkpoint_path.parent / "config.yaml"

    training_config = _load_config_yaml(_cfg_path)

    logger.info("加载 TwinBrain 检查点: %s  (device=%s)", checkpoint_path, device)

    try:
        twin = TwinBrainDigitalTwin.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            subject_to_idx_path=subject_to_idx_path,
            config=training_config if training_config else None,
        )
    except KeyError as exc:
        raise RuntimeError(
            f"检查点格式错误，缺少必要键 {exc}。\n"
            "TwinBrain V5 检查点必须包含 'model_state_dict' 键。\n"
            f"文件: {checkpoint_path}"
        ) from exc
    except (ValueError, RuntimeError) as exc:
        raise RuntimeError(
            f"模型架构推断失败: {exc}\n"
            "检查点的 state_dict 必须包含 'encoder.input_proj.*.weight' "
            "或 'subject_embed.weight'，\n"
            "或在检查点同目录提供 config.yaml 以辅助重建架构。\n"
            f"文件: {checkpoint_path}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"模型加载失败: {exc}\n文件: {checkpoint_path}"
        ) from exc

    n_params = sum(p.numel() for p in twin.model.parameters())
    logger.info(
        "✓ TwinBrainDigitalTwin 加载成功 "
        "(device=%s, n_params=%d, n_subjects=%d)",
        twin.device,
        n_params,
        twin.model.num_subjects,
    )
    return twin


def load_graph_for_inference(
    graph_path: Union[str, Path],
    device: str = "cpu",
    k_cross_modal: int = 5,
) -> HeteroData:
    """
    加载图缓存文件并重建所有图边，返回推理就绪的 ``HeteroData``。

    V5 图缓存（``outputs/graph_cache/*.pt``）**只存储节点特征**（``fmri.x``、
    ``eeg.x``）。图边不被存储，必须在推理时从节点时序动态重建：

    - 同模态边 ``('fmri', 'connects', 'fmri')``：从 fMRI 时序的 Pearson 相关
      矩阵计算（与训练时一致）。EEG 同模态边同理。
    - 跨模态边 ``('eeg', 'projects_to', 'fmri')``：随机 k-NN 映射（仅在
      fMRI + EEG 双模态下生成）。

    **如果跳过这一步（只有节点特征无边）：**
    ``GraphPredictionPropagator`` 和编码器的图卷积均退化为 passthrough，
    刺激节点 i 的效应无法传播到任何 j≠i，导致响应矩阵呈纯对角线形态。

    Args:
        graph_path:    图缓存文件路径（``{subject_id}_{task}_{hash}.pt``）。
        device:        将图数据移动到此设备（通常与模型设备一致）。
        k_cross_modal: 每个 EEG 电极保留的 fMRI ROI 邻居数（见 API.md §2.5）。

    Returns:
        ``HeteroData`` — 含同模态边 + 跨模态边的完整推理图。

    Raises:
        FileNotFoundError: 缓存文件不存在。
        RuntimeError:      文件加载失败或格式不符合 V5 规范。
        ImportError:       ``models.graph_native_mapper`` 无法导入。
    """
    graph_path = Path(graph_path)

    if not graph_path.exists():
        raise FileNotFoundError(
            f"图缓存文件不存在: {graph_path}\n"
            "提示: 图缓存位于 outputs/graph_cache/<subject_id>_<task>_<hash>.pt"
        )

    logger.info("加载图缓存: %s", graph_path)

    try:
        graph: HeteroData = torch.load(
            graph_path, map_location="cpu", weights_only=False
            # weights_only=False is required because HeteroData is not a plain tensor.
            # Security note: only load graph cache files from trusted sources (i.e.,
            # files generated by this project's own training pipeline).
        )
    except Exception as exc:
        raise RuntimeError(
            f"图缓存文件读取失败: {exc}\n文件: {graph_path}"
        ) from exc

    if not isinstance(graph, HeteroData):
        raise RuntimeError(
            f"文件不是 HeteroData 对象（实际类型: {type(graph).__name__}）。\n"
            "V5 图缓存必须是单个 torch_geometric.data.HeteroData 实例。\n"
            f"文件: {graph_path}"
        )

    # Validate required node types and attribute shapes
    for node_type in graph.node_types:
        x = getattr(graph[node_type], "x", None)
        if x is None:
            raise RuntimeError(
                f"节点类型 '{node_type}' 缺少 'x' 属性。\n"
                "V5 缓存要求每个节点类型有形状 [N, T, 1] 的 'x' 张量。"
            )
        if x.ndim != 3 or x.shape[-1] != 1:
            raise RuntimeError(
                f"节点类型 '{node_type}' 的 x 形状 {tuple(x.shape)} 不符合规范。\n"
                "期望形状: [N, T, 1]（3-D，最后维 C=1）。"
            )

    logger.info(
        "  节点类型: %s",
        {nt: tuple(graph[nt].x.shape) for nt in graph.node_types},
    )

    # ── Import mapper once for all edge-reconstruction steps ─────────────────
    try:
        from models.graph_native_mapper import GraphNativeBrainMapper
    except ImportError as exc:
        raise ImportError(
            f"无法导入 GraphNativeBrainMapper: {exc}\n"
            f"请确保仓库根目录（{_REPO_ROOT}）在 PYTHONPATH 中。"
        ) from exc

    mapper = GraphNativeBrainMapper(device="cpu")

    # ── Rebuild intra-modal edges (fMRI and/or EEG) ──────────────────────────
    # V5 caches store only node features; intra-modal FC edges must be
    # recomputed here so that the encoder's graph convolutions and the
    # GraphPredictionPropagator can propagate stimulation effects across
    # brain regions.  Without these edges both modules degrade to passthrough,
    # producing a perfectly diagonal response matrix (stimulating region i only
    # changes region i — no inter-regional propagation).
    _INTRA_MODAL_CFG = {
        "fmri": (mapper.threshold_fmri, mapper.k_nearest_fmri, "fMRI"),
        "eeg":  (mapper.threshold_eeg,  mapper.k_nearest_eeg,  "EEG"),
    }
    for nt, (thresh, k_nn, label) in _INTRA_MODAL_CFG.items():
        if nt not in graph.node_types:
            continue
        edge_type = (nt, "connects", nt)
        # Respect pre-existing valid edges (e.g. from training pipelines that
        # save edges in the cache).
        if edge_type in graph.edge_types:
            existing = getattr(graph[edge_type], "edge_index", None)
            if existing is not None and existing.shape[1] > 0:
                logger.info(
                    "  %s 同模态边已存在，保留缓存值 (E=%d)", label, existing.shape[1]
                )
                continue
        try:
            # graph[nt].x shape: [N, T, 1]  →  timeseries: [N, T] numpy
            ts = graph[nt].x.squeeze(-1).cpu().numpy()  # [N, T]
            if nt == "fmri":
                conn = mapper._compute_correlation_gpu(ts)
            else:
                conn = mapper._compute_eeg_connectivity(ts)
            ei, ea = mapper.build_graph_structure(
                conn, threshold=thresh, k_nearest=k_nn
            )
            graph[edge_type].edge_index = ei
            graph[edge_type].edge_attr  = ea
            logger.info(
                "  ✓ %s 同模态边重建完成 (N=%d, E=%d, avg_degree=%.1f)",
                label, ts.shape[0], ei.shape[1], ei.shape[1] / max(ts.shape[0], 1),
            )
        except (RuntimeError, ValueError, TypeError) as exc:
            logger.warning(
                "  ⚠ %s 同模态边重建失败: %s\n"
                "    编码器和传播器将以 passthrough 模式运行（响应矩阵可能退化为对角线）。",
                label, exc,
            )

    # ── Rebuild cross-modal edges if both modalities are present ─────────────
    if "eeg" in graph.node_types and "fmri" in graph.node_types:
        cross_ei, cross_ea = mapper.create_simple_cross_modal_edges(
            graph, k_cross_modal=k_cross_modal
        )
        graph["eeg", "projects_to", "fmri"].edge_index = cross_ei
        graph["eeg", "projects_to", "fmri"].edge_attr = cross_ea
        logger.info(
            "  ✓ 跨模态边重建完成 (k=%d, E_cross=%d)", k_cross_modal, cross_ei.shape[1]
        )
    elif "eeg" not in graph.node_types:
        logger.info("  缓存中无 EEG 节点，跳过跨模态边重建。")

    if device != "cpu":
        graph = graph.to(device)

    return graph
