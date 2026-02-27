"""
Unity Integration Module
========================

Provides tools for running the TwinBrain WebSocket server and supporting
model inference, perturbation analysis, and brain state analysis.
"""

from .model_server import ModelServer

# Import server with error handling
try:
    from .realtime_server import BrainVisualizationServer
    SERVER_AVAILABLE = True
except ImportError:
    BrainVisualizationServer = None
    SERVER_AVAILABLE = False

__all__ = [
    'BrainVisualizationServer',
    'ModelServer',
    'SERVER_AVAILABLE',
]
