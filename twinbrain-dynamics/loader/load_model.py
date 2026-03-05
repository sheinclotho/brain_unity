"""
Model Loader
============

加载训练完成的 TwinBrain 模型，冻结参数，切换为 evaluation 模式。

设计原则:
- 不允许修改模型结构
- 不参与任何梯度计算 (所有推理都在 torch.no_grad() 下执行)
- 若模型文件不存在或无法解析，返回 None（上层代码回退到 Wilson-Cowan 演化）
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def load_trained_model(
    model_path: str,
    device: str = "cpu",
) -> Optional[nn.Module]:
    """
    加载训练完成的 TwinBrain 模型。

    Args:
        model_path: 训练模型文件路径（.pt / .pth 格式的 PyTorch checkpoint）。
        device:     计算设备，"cpu" 或 "cuda"。

    Returns:
        已冻结参数、处于 eval 模式的 nn.Module；
        若加载失败则返回 None（调用方应回退到无模型的演化方式）。

    Notes:
        - checkpoint 格式支持:
            1. ``{"model": <state_dict>}``
            2. ``{"model_state_dict": <state_dict>}``
            3. 直接是 ``nn.Module`` 实例（整个对象被序列化）
            4. 普通 ``state_dict``（将被当作元数据，模型结构无法重建时返回 None）
        - 加载后自动调用 ``model.eval()`` 并 ``requires_grad_(False)``。
    """
    model_path = Path(model_path)

    if not model_path.exists():
        logger.error("模型文件不存在: %s", model_path)
        logger.info(
            "提示: 训练完成的模型通常位于 outputs/ 或 test_file3/sub-XX/results/"
        )
        return None

    if model_path.suffix not in {".pt", ".pth"}:
        logger.warning("扩展名不常见 (%s)，尝试继续加载。", model_path.suffix)

    logger.info("加载模型: %s  (device=%s)", model_path, device)

    try:
        obj = torch.load(model_path, map_location=device, weights_only=False)
    except Exception as exc:
        logger.error("文件读取失败: %s", exc)
        return None

    # ── Case 1: already a live nn.Module ──────────────────────────────────────
    if isinstance(obj, nn.Module):
        model = obj
        logger.info("  ✓ 直接加载为 nn.Module 实例")
        return _freeze_and_eval(model, device)

    # ── Case 2: dict checkpoint ───────────────────────────────────────────────
    if isinstance(obj, dict):
        # Try to extract the live module stored under common keys
        for key in ("model_obj", "module"):
            if key in obj and isinstance(obj[key], nn.Module):
                model = obj[key]
                logger.info("  ✓ 从 checkpoint['%s'] 提取 nn.Module", key)
                return _freeze_and_eval(model, device)

        # Log available metadata for diagnostics
        meta_keys = [k for k in obj if k not in ("model", "model_state_dict")]
        if meta_keys:
            logger.info("  Checkpoint 元数据键: %s", meta_keys)
            if "epoch" in obj:
                logger.info("    训练轮数: %s", obj["epoch"])
            if "best_loss" in obj:
                logger.info("    最佳损失: %.4f", obj["best_loss"])

        # Return a _StateWrapper that exposes the state dict for downstream use
        for key in ("model", "model_state_dict"):
            if key in obj:
                state_dict = obj[key]
                if isinstance(state_dict, dict):
                    logger.info(
                        "  ✓ 提取 state_dict（%d 个 tensor），包装为 StateWrapper",
                        len(state_dict),
                    )
                    return _StateWrapper(state_dict, device=device)

        # Whole dict treated as state_dict
        logger.warning(
            "Checkpoint 格式非标准，尝试将整个 dict 视为 state_dict。"
        )
        return _StateWrapper(obj, device=device)

    logger.error(
        "未知 checkpoint 格式 (%s)，无法加载模型。", type(obj).__name__
    )
    return None


# ── Internal helpers ──────────────────────────────────────────────────────────

def _freeze_and_eval(model: nn.Module, device: str) -> nn.Module:
    """Move model to device, switch to eval mode, freeze all parameters."""
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("  ✓ 模型已冻结 (eval 模式，%d 参数，device=%s)", n_params, device)
    return model


class _StateWrapper(nn.Module):
    """
    Thin wrapper that exposes a raw state_dict without the original model class.

    This is useful when the full model class (DynamicHeteroGNN etc.) is not
    importable in the dynamics analysis environment.  The wrapper stores the
    state dict so downstream code can inspect parameter shapes, but it cannot
    run a forward pass.

    ``can_forward`` is False; ``BrainDynamicsSimulator`` will fall back to the
    Wilson-Cowan integrator when it encounters this wrapper.
    """

    can_forward: bool = False

    def __init__(self, state_dict: Dict[str, Any], device: str = "cpu"):
        super().__init__()
        self.state_dict_raw = state_dict
        self._device = device
        logger.warning(
            "StateWrapper: 原始模型类不可用，前向传播已禁用。"
            " BrainDynamicsSimulator 将使用 Wilson-Cowan 回退模式。"
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError(
            "StateWrapper 不支持前向传播。"
            " 请确保完整的模型类可被导入，或使用无模型演化模式。"
        )

    @property
    def n_regions(self) -> int:
        """Try to infer n_regions from state_dict parameter shapes."""
        for name, tensor in self.state_dict_raw.items():
            if isinstance(tensor, torch.Tensor) and tensor.ndim >= 1:
                return int(tensor.shape[0])
        return 200  # default: Schaefer-200
