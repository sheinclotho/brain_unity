"""
TwinBrain — 生产模型模块
========================

仅导出主训练流程（main.py / val.py）实际使用的组件。

高级实验性模块（意识建模、预测编码、增强注意力等）已移至
``reference/`` 目录，附带完整说明文档（reference/README.md）。
"""

# 图原生基础模型
from .graph_native_system import (
    GraphNativeBrainModel,
    GraphNativeTrainer,
    GraphNativeDecoder,
)
from .graph_native_encoder import (
    GraphNativeEncoder,
    SpatialTemporalGraphConv,
)
from .graph_native_mapper import GraphNativeBrainMapper

# 优化模块
from .adaptive_loss_balancer import AdaptiveLossBalancer
from .eeg_channel_handler import EnhancedEEGHandler
from .advanced_prediction import EnhancedMultiStepPredictor

# 数字孪生推理引擎（V5.44）
from .digital_twin_inference import TwinBrainDigitalTwin

__all__ = [
    'GraphNativeBrainModel',
    'GraphNativeTrainer',
    'GraphNativeDecoder',
    'GraphNativeEncoder',
    'SpatialTemporalGraphConv',
    'GraphNativeBrainMapper',
    'AdaptiveLossBalancer',
    'EnhancedEEGHandler',
    'EnhancedMultiStepPredictor',
    'TwinBrainDigitalTwin',
]
