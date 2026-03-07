"""
spectral_dynamics
=================

六个相互独立的实验模块，验证"连接组谱结构 → 低维动力学 → 近临界行为"假设链。

模块说明
--------
compute_connectivity    从图缓存或预计算轨迹中提取功能/有效连接矩阵
e1_spectral_analysis    E1: 谱结构分析（特征值分布、谱有效维度、主导模式数）
e2_e3_modal_projection  E2+E3: 动力学模态投影与模态能量分布
e4_structural_perturbation  E4: 结构扰动实验（边重连、权重随机化、低秩截断）
e5_phase_diagram        E5: 耦合强度相图（稳定→振荡→混沌边界扫描）
e6_random_comparison    E6: 随机网络对照（ER、保度随机、权重混洗）
run_all                 一键运行所有实验的主入口

与 twinbrain-dynamics 的接口
----------------------------
本模块作为 **独立分析层**，直接读取 twinbrain-dynamics 流程的输出：
  outputs/trajectories.npy     — shape (n_init, steps, n_regions)
  outputs/response_matrix.npy  — shape (n_regions, n_regions)

也可导入 twinbrain-dynamics 的分析函数（rosenstein_lyapunov 等）。

批判性说明
----------
1. 连接矩阵层次：谱分析在"有效连接矩阵"（响应矩阵 R）上比结构 DTI
   更准确地反映动力学特性。本模块优先使用 R，DTI 结构矩阵作为备选。
2. 模态投影（E2/E3）仅在 R 近似于吸引子附近的系统 Jacobian 时物理意义明确。
   非线性 GNN 模型下此近似成立的充分条件是刺激幅度小（0.5σ 范围内）。
3. E5 相图在 Wilson-Cowan 框架下构建，而非直接修改 GNN 权重（后者需要
   重新训练）。WC 提供可控的基线，结构参数可以系统扫描。
4. E4 扰动在 R 矩阵上进行，而非修改图缓存（避免模型推断前提被破坏）。
"""
