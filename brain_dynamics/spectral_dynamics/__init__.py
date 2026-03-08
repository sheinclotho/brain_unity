"""
spectral_dynamics
=================

六个相互独立的实验模块 + 新增分析模块，验证"连接组谱结构 → 低维动力学 → 近临界行为"假设链。

模块说明
--------
compute_connectivity    从图缓存或预计算轨迹中提取功能/有效连接矩阵
e1_spectral_analysis    E1: 谱结构分析（特征值分布、谱有效维度、主导模式数）
e2_e3_modal_projection  E2+E3: 动力学模态投影与模态能量分布
e4_structural_perturbation  E4: 结构扰动实验（边重连、权重随机化、低秩截断）
e5_phase_diagram        E5: 耦合强度相图（稳定→振荡→混沌边界扫描）
e6_random_comparison    E6: 随机网络对照（ER、保度随机、权重混洗）
h_power_spectrum        H: 功率谱 / FFT 分析 + 脑节律频段标注 + 空间振荡模态
i_energy_constraint     I: 能量约束分岔实验（WC L1 投影 + LLE 扫描）
run_all                 一键运行所有实验的主入口

与 twinbrain-dynamics 的关系（合并后）
--------------------------------------
本包作为 **矩阵驱动分析层**，直接读取 twinbrain-dynamics 流程的输出：
  outputs/trajectories.npy     — shape (n_init, steps, n_regions)
  outputs/response_matrix.npy  — shape (n_regions, n_regions)

**共享实现（已统合）**：

以下功能曾在本包内多个文件中独立实现，现已统一由
``twinbrain-dynamics/analysis/wc_dynamics.py`` 提供：

+------------------------------------------+----------------------------+
| 旧位置（已删除）                          | 新统一位置                  |
+==========================================+============================+
| e4._wc_step                              | wc_dynamics.wc_step        |
| e5._wc_trajectories                      | wc_dynamics.wc_simulate    |
| e4._rosenstein_lle_on_wc / _simple_lle   | wc_dynamics.rosenstein_lle_on_wc |
| e5._rosenstein_from_twinbrain /          |                            |
|   _simple_rosenstein                     | wc_dynamics.rosenstein_lle_on_wc |
| i._rosenstein                            | wc_dynamics.rosenstein_lle_on_wc |
| i._project_energy_wc                     | wc_dynamics.project_energy_l1_bounded |
+------------------------------------------+----------------------------+

``h_power_spectrum.py`` 的内置 FFT 实现已删除；该模块现在只委托给
``twinbrain-dynamics/analysis/power_spectrum.py``（权威实现），
无 twinbrain 时抛出 ``ImportError``（有明确报错信息）。

sys.path 管理已统一移至本 ``__init__.py``。各子模块使用相对导入
``from analysis.wc_dynamics import ...``（在 _ensure_twinbrain_path() 调用后有效）。

批判性说明
----------
1. 连接矩阵层次：谱分析在"有效连接矩阵"（响应矩阵 R）上比结构 DTI
   更准确地反映动力学特性。本模块优先使用 R，DTI 结构矩阵作为备选。
2. 模态投影（E2/E3）仅在 R 近似于吸引子附近的系统 Jacobian 时物理意义明确。
   非线性 GNN 模型下此近似成立的充分条件是刺激幅度小（0.5σ 范围内）。
3. E5 相图在 Wilson-Cowan 框架下构建，而非直接修改 GNN 权重（后者需要
   重新训练）。WC 提供可控的基线，结构参数可以系统扫描。
4. 注意：原 B_LYA 模块（数据驱动线性映射近似 Lyapunov 谱）已移除。
   该功能与 dynamics_pipeline 的 DMD 谱分析（Phase 3e）功能完全重叠。
"""

from __future__ import annotations

import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# sys.path setup: ensure the parent brain_dynamics/ directory is on sys.path
# so that sub-modules can do:  from analysis.xxx import ...
# ─────────────────────────────────────────────────────────────────────────────

_BD_DIR = Path(__file__).resolve().parent.parent  # brain_dynamics/


def _ensure_brain_dynamics_path() -> bool:
    """
    Add ``brain_dynamics/`` to ``sys.path`` if not already present.

    Returns True if the directory exists and was added (or already present).
    """
    if not _BD_DIR.exists():
        return False
    bd_str = str(_BD_DIR)
    if bd_str not in sys.path:
        sys.path.insert(0, bd_str)
    return True


_TWINBRAIN_AVAILABLE = _ensure_brain_dynamics_path()
