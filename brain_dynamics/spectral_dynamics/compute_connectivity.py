"""
Connectivity Matrix Extraction
================================

从 twinbrain-dynamics 流程的输出或图缓存中提取连接矩阵，
供后续谱分析使用。

提供三种连接矩阵：

1. **有效连接矩阵（EC）/ 响应矩阵 R**
   来源：``outputs/response_matrix.npy``（shape [N, N]）
   含义：刺激节点 i 时节点 j 的因果效应；近似系统 Jacobian。
   用于：E1–E4 的主要谱分析对象。

2. **功能连接矩阵（FC）**
   来源：``outputs/trajectories.npy``（shape [n_init, T, N]）→ Pearson 相关
   含义：轨迹节点间的时间相关性。
   用于：E6 随机网络对照（FC 比 EC 更容易从随机网络中获得）。

3. **结构连接矩阵（SC）**
   来源：图缓存中的边属性 ``graph['fmri', *, 'fmri'].edge_attr``（若可用）。
   含义：DTI 白质纤维束权重。
   用于：E4 中作为扰动基础的"真实结构"参照。

设计原则
--------
- 所有函数均返回 ``np.ndarray`` float32，形状 (N, N)。
- 若来源文件不存在或格式错误，抛出 ``FileNotFoundError`` / ``ValueError``，
  不允许静默回退到零矩阵。
- 模块本身不做任何模型推断，纯粹是数据读取与计算。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  有效连接矩阵（来自响应矩阵）
# ─────────────────────────────────────────────────────────────────────────────

def load_response_matrix(outputs_dir: Path) -> np.ndarray:
    """
    从 twinbrain-dynamics 输出目录加载响应矩阵。

    响应矩阵由 ``analysis/response_matrix.py`` 的 ``compute_response_matrix()``
    生成并保存为 ``response_matrix.npy``。

    Args:
        outputs_dir: ``outputs/`` 目录路径（含 ``response_matrix.npy``）。

    Returns:
        R: shape (N, N), float32。R[i, j] = 刺激节点 i 时节点 j 的效应。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError:         数组形状非方阵。
    """
    path = Path(outputs_dir) / "response_matrix.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"响应矩阵文件不存在: {path}\n"
            "请先运行 twinbrain-dynamics 流程步骤 6（响应矩阵计算）。"
        )
    R = np.load(path).astype(np.float32)
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"响应矩阵应为方阵，实际形状: {R.shape}")
    logger.info("加载响应矩阵: shape=%s, max=%.4f, min=%.4f", R.shape, R.max(), R.min())
    return R


# ─────────────────────────────────────────────────────────────────────────────
# 2.  功能连接矩阵（来自自由动力学轨迹）
# ─────────────────────────────────────────────────────────────────────────────

def compute_fc_from_trajectories(trajectories: np.ndarray) -> np.ndarray:
    """
    从自由动力学轨迹计算功能连接矩阵（Pearson 相关）。

    将所有轨迹的时序拼接后计算节点间时间相关性。

    Args:
        trajectories: shape (n_init, T, N)，来自 free_dynamics 输出。

    Returns:
        FC: shape (N, N), float32，对称矩阵，对角线为 1.0。
    """
    n_init, T, N = trajectories.shape
    # Reshape: (n_init * T, N) — treat all time points across all trajectories
    ts = trajectories.reshape(-1, N).astype(np.float64)
    # Subtract mean per region
    ts -= ts.mean(axis=0, keepdims=True)
    std = ts.std(axis=0)
    # Guard against zero-variance channels
    std = np.where(std < 1e-10, 1.0, std)
    ts /= std
    FC = (ts.T @ ts) / max(ts.shape[0] - 1, 1)
    FC = np.clip(FC, -1.0, 1.0).astype(np.float32)
    np.fill_diagonal(FC, 1.0)
    logger.info("计算功能连接矩阵: shape=%s, off-diag mean=%.4f",
                FC.shape, float(np.abs(FC[~np.eye(N, dtype=bool)]).mean()))
    return FC


def load_trajectories(outputs_dir: Path) -> np.ndarray:
    """
    从 twinbrain-dynamics 输出目录加载自由动力学轨迹。

    Args:
        outputs_dir: ``outputs/`` 目录路径（含 ``trajectories.npy``）。

    Returns:
        trajectories: shape (n_init, T, N)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError:         数组维度非 3D。
    """
    path = Path(outputs_dir) / "trajectories.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"轨迹文件不存在: {path}\n"
            "请先运行 twinbrain-dynamics 流程步骤 3（自由动力学实验）。"
        )
    traj = np.load(path)
    if traj.ndim != 3:
        raise ValueError(f"轨迹数组应为 3D (n_init, T, N)，实际形状: {traj.shape}")
    logger.info("加载轨迹: shape=%s", traj.shape)
    return traj.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  结构连接矩阵（来自图缓存边属性）
# ─────────────────────────────────────────────────────────────────────────────

def load_structural_connectivity(
    graph_cache_path: Path,
    edge_type: Optional[Tuple[str, str, str]] = None,
) -> np.ndarray:
    """
    从图缓存文件中提取结构连接矩阵（DTI/白质纤维束权重）。

    尝试从 ``graph['fmri', rel, 'fmri']`` 的边属性中重建稠密矩阵。
    若图中没有 fmri→fmri 边，则从 fMRI 时序的 Pearson 相关估计。

    Args:
        graph_cache_path: ``{subject}_{task}_{hash}.pt`` 图缓存文件路径。
        edge_type:         边类型三元组，如 ``('fmri', 'corr', 'fmri')``；
                           省略时自动查找第一个 fmri→fmri 边类型。

    Returns:
        SC: shape (N_fmri, N_fmri), float32。

    Raises:
        FileNotFoundError:  图缓存文件不存在。
        ImportError:         torch / torch_geometric 未安装。
        RuntimeError:        图缓存中未找到可用的 fMRI 节点。
    """
    graph_cache_path = Path(graph_cache_path)
    if not graph_cache_path.exists():
        raise FileNotFoundError(f"图缓存文件不存在: {graph_cache_path}")

    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:
        raise ImportError("需要 torch 和 torch_geometric：pip install torch torch_geometric") from exc

    data = torch.load(graph_cache_path, map_location="cpu", weights_only=False)
    if not isinstance(data, HeteroData):
        raise RuntimeError(f"图缓存应为 HeteroData，实际类型: {type(data).__name__}")

    if "fmri" not in data.node_types:
        raise RuntimeError(f"图缓存中未找到 'fmri' 节点类型。可用: {list(data.node_types)}")

    N = int(data["fmri"].x.shape[0])

    # Find fmri→fmri edges
    fmri_edges = [
        et for et in data.edge_types
        if et[0] == "fmri" and et[2] == "fmri"
    ]

    if edge_type is not None:
        if edge_type not in data.edge_types:
            raise ValueError(f"边类型 {edge_type} 不存在。可用: {list(data.edge_types)}")
        fmri_edges = [edge_type]

    if fmri_edges:
        et = fmri_edges[0]
        logger.info("从边类型 %s 构建结构连接矩阵", et)
        ei = data[et].edge_index.numpy()   # (2, E)
        SC = np.zeros((N, N), dtype=np.float32)
        if hasattr(data[et], "edge_attr") and data[et].edge_attr is not None:
            ea = data[et].edge_attr.numpy().ravel()
            SC[ei[0], ei[1]] = ea
        else:
            SC[ei[0], ei[1]] = 1.0
        logger.info("结构连接矩阵: shape=%s, nnz=%d", SC.shape, int((SC != 0).sum()))
        return SC
    else:
        # Fall back: compute FC from fMRI time series in the graph cache
        logger.warning(
            "图缓存中未找到 fmri→fmri 边，使用 fMRI 时序 Pearson 相关作为近似 SC。"
        )
        x = data["fmri"].x.squeeze(-1).numpy().T  # (T, N)
        x = x - x.mean(axis=0, keepdims=True)
        std = x.std(axis=0)
        std = np.where(std < 1e-10, 1.0, std)
        x /= std
        SC = (x.T @ x / max(x.shape[0] - 1, 1)).astype(np.float32)
        np.fill_diagonal(SC, 1.0)
        return SC


# ─────────────────────────────────────────────────────────────────────────────
# 4.  通用工具
# ─────────────────────────────────────────────────────────────────────────────

def symmetrize(W: np.ndarray) -> np.ndarray:
    """对称化矩阵：返回 (W + W.T) / 2。"""
    return (W + W.T) / 2.0


def normalize_spectral_radius(W: np.ndarray, target_rho: float = 0.95) -> np.ndarray:
    """
    谱半径归一化：缩放 W 使最大特征值绝对值 = target_rho。

    不改变特征向量方向（仅线性缩放），确保 Gramian 级数收敛。

    Args:
        W:           方阵 (N, N)。
        target_rho:  目标谱半径（< 1.0 保证 Gramian 收敛）。

    Returns:
        W_norm: 同形状，谱半径 = target_rho。
    """
    eigvals = np.linalg.eigvals(W)
    rho = float(np.abs(eigvals).max())
    if rho < 1e-10:
        logger.warning("矩阵谱半径近似为零，无法归一化。")
        return W.copy()
    return (W * target_rho / rho).astype(np.float32)


def participation_ratio(eigvals: np.ndarray) -> float:
    """
    计算特征值分布的参与率（谱有效维度）。

    定义：PR = (Σ|λ_k|)² / Σ(|λ_k|²)

    PR = N 表示所有特征值等强（白色谱）；
    PR = 1 表示只有一个主导特征值（完全低秩）。

    Args:
        eigvals: 特征值数组（可含复数）。

    Returns:
        PR: 参与率，范围 [1, N]。
    """
    mags = np.abs(eigvals).astype(np.float64)
    s1 = float(mags.sum())
    s2 = float((mags ** 2).sum())
    if s2 < 1e-30:
        return 1.0
    return float(s1 ** 2 / s2)
