# dynamics_pipeline — 统一脑动力学分析流程

> **一个配置驱动的、分阶段的脑网络动力学测试流程。**
> 将 `twinbrain-dynamics`（模型驱动）和 `spectral_dynamics`（矩阵驱动）合并为单一管线。

---

## 快速开始

```bash
# 完整分析（默认 fMRI 模态）
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt

# 快速预实验（参数大幅缩减，适合验证流程）
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick

# 只运行指定阶段
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 3

# 带能量约束
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --energy-budget 0.3

# EEG 模态分析
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality eeg

# 双模态分析（fMRI + EEG 分别运行，结果存入 output/fmri/ 和 output/eeg/）
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality both

# 联合模态分析（fMRI + EEG 拼接为统一状态向量）
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --modality joint

# 直接运行（等价于 python -m dynamics_pipeline.run）
python dynamics_pipeline/run.py --model best_model.pt --graph graph.pt
```

---

## 模态选择

管线支持四种模态模式，通过 `--modality` 控制（配置键：`simulator.modality`）：

| 模态 | 说明 | 图缓存要求 | 输出结构 |
|------|------|-----------|---------|
| `fmri` | 分析 fMRI BOLD 流（默认） | 需要 `fmri` 节点 | `output_dir/` 平铺 |
| `eeg` | 分析 EEG 流 | 需要 `eeg` 节点 | `output_dir/` 平铺 |
| `both` | 对每种模态**独立运行完整管线** | 需要两种节点 | `output_dir/fmri/` + `output_dir/eeg/` |
| `joint` | 单次运行，使用 **fMRI+EEG 拼接状态向量** | 需要两种节点 | `output_dir/` 平铺 |

### `both` 模式

适用于**对比**两种模态的动力学特性。管线依次运行两遍（fMRI → EEG），每次使用独立的
模拟器实例，结果保存在各自子目录中。返回值格式为 `{"fmri": {...}, "eeg": {...}}`。

### `joint` 模式

适用于**统一**分析多模态脑状态。单次 `predict_future()` 调用同时获取 fMRI 和 EEG
预测；各自按 `base_graph` 统计量做逐通道 z-score 归一化后拼接为联合状态向量
`[z_fmri | z_eeg]`（维度 N_fmri + N_eeg），输出单一动力学指标集合。

**注意**：`joint` 模式下 Lyapunov 方法**自动强制为 `rosenstein`**。原因：Wolf/FTLE
依赖 [0,1] 状态空间裁剪，联合模态的 z-score 空间无上下界，裁剪会在 0 处产生虚假
吸引点，污染 LLE 估计。Rosenstein 不依赖状态空间边界，是唯一正确选择。

---

## 设计理念

### 我们在做什么？

我们要回答一个核心科学问题：**训练好的 TwinBrain 数字孪生模型展示了怎样的动力学行为？**

具体而言，我们测试五个假设：

| 假设 | 问题 | 对应分析 |
|------|------|---------|
| **H1** | 脑网络连接是否具有低秩谱结构？ | 特征值分解、参与比(PR)、谱半径 |
| **H2** | 动力学是否低维？ | D₂(关联维数)、K-Y 维度(线性化)、PCA n@90% |
| **H3** | 系统是否处于近临界状态？ | Rosenstein λ₁、DMD 谱半径 ρ |
| **H4** | 是否存在脑样振荡？ | 功率谱密度、DMD Hopf 分岔频率 |
| **H5** | 代谢能量约束是否维持临界性？ | L1 投影实验 |

### 为什么要合并两个模块？

原有两个模块存在显著重叠（Lyapunov 指数、功率谱、随机对照、PCA 均有双重实现），
且没有统一的假设评估和一致性检查。本管线消除冗余，用单一配置文件控制所有分析，
并在最后自动进行跨步骤一致性验证。

### 核心方法论：非线性与线性化分析的互补

本管线的动力学分析基于两类互补的方法：

**1. Rosenstein LLE (λ₁) — 非线性混沌指标（唯一权威来源）**

Rosenstein 方法在 N 维状态空间中寻找最近邻对，追踪它们的指数发散率。这是
纯粹的非线性分析，直接从轨迹数据工作，不做任何线性化假设。λ₁ 是系统是否
混沌的唯一可靠判据。

**2. DMD 线性化谱分析 — 结构性互补工具**

DMD 拟合最优线性转移算子 A: x(t+1) ≈ A·x(t)，其特征值 μ_i 给出系统在
吸引子附近的**线性化动力学**。这**不是**对非线性 Lyapunov 谱的替代，而是
一个结构性互补：

  - λ_DMD_i = ln|μ_i| 给出**线性化 Lyapunov 谱**
  - 线性化 K-Y 维度 = f(λ_DMD_1, ..., λ_DMD_N) 估计吸引子分形维度的线性近似
  - Hopf 对 (复共轭 μ = re^{±iθ}) 揭示振荡模态频率
  - 慢模态 (|Re(λ_ct)| < 0.05) 揭示长弛豫时间方向

**两者的关系是验证性的，不是等价性的：**

| 情景 | Rosenstein λ₁ | DMD λ_max | 解读 |
|------|---------------|-----------|------|
| 一致稳定 | < 0 | < 0 | 线性化近似有效 |
| 一致临界 | ≈ 0 | ≈ 0 | 系统接近分岔点 |
| 非线性主导 | > 0 | < 0 | 非线性折叠产生混沌，线性分析低估 |
| 线性不稳定 | > 0 | > 0 | 全局不稳定（需检查有界性） |

**非线性指数 Δ = |λ₁_Rosenstein - max(λ_DMD)| / |λ₁_Rosenstein|** 衡量线性化
近似的偏差程度。Δ > 1 表示非线性效应主导，此时 DMD 结果仅做参考。

### 吸引子维度估计

吸引子维度回答："系统的长期动力学实际占据了多少维度？"

本管线提供三个互补的维度估计：

1. **D₂ (Grassberger-Procaccia 关联维数)** — 非线性、从轨迹数据直接计算。
   测量吸引子的真实几何维度。D₂ = 3.2 表示类似 Lorenz 吸引子的低维混沌。

2. **K-Y_linear (DMD 线性化 Kaplan-Yorke 维度)** — 从 DMD 线性化 Lyapunov 谱
   计算。是线性近似下的吸引子维度估计。对弱非线性系统接近真实 K-Y 维度，
   对强非线性系统可能低估。

3. **PCA n@90%** — 解释 90% 方差所需的主成分数。这是吸引子维度的上界（线性
   嵌入维度 ≥ 分形维度）。

**科学上的优先序**：D₂ (非线性) > K-Y_linear (线性化) > PCA n@90% (线性上界)。
当三者一致时，维度估计可信；当 D₂ << PCA n@90% 时，表明吸引子具有分形结构。

### 关于 Wolf-GS Lyapunov 谱的历史说明

Wolf-GS 方法在 TwinBrain 架构下存在不可修复的上下文稀释偏差（所有 k 个指数
≈ 0.13，D_KY 恰好等于 k），已从管线中**移除**（非禁用）。相关的
`spectral_dynamics/b_lyapunov_spectrum.py` 模块和 `run_dynamics_analysis.py`
Step 15 均已删除。

DMD **不是** Wolf-GS 的"替代品"。DMD 提供的是互补的线性化分析视角，
Rosenstein LLE 是唯一的非线性混沌判据。

---

## 管线架构

```
Phase 1: Data Generation        → 轨迹 + 响应矩阵
    ↓
Phase 2: Network Structure      → 谱分析、社区、层次、模态能量
    ↓
Phase 3: Dynamics Characterisation → 稳定性、LLE(非线性)、DMD(线性化)、PSD、吸引子维度
    ↓
Phase 4: Statistical Validation  → 代替检验、随机对照、嵌入维度
    ↓
Phase 5: Advanced (optional)     → 刺激、能量约束、可控性、信息流
    ↓
Phase 6: Synthesis               → 非线性指数 + 假设评估 + 一致性检查 + 报告
```

---

## 各阶段详细说明

### Phase 1: 数据生成

| 步骤 | 分析 | 输出 | 科学目的 |
|------|------|------|---------|
| 1a | 自由动力学 | `trajectories.npy` (n_init × steps × N) | 从随机初态模拟自由演化 |
| 1b | 响应矩阵 | `response_matrix.npy` (N × N) | 每个节点受刺激后对所有节点的因果效应 |

**关键参数：**
- `n_init`: 独立轨迹数量（默认 100，quick 模式 20）
- `steps`: 每条轨迹的预测步数（默认 500，quick 模式 100）
- `seed`: 随机种子，保证可复现

**时间窗口机制：** 为避免所有轨迹共享相同的历史上下文（导致 Wolf 上下文稀释），
使用 75% 重叠的滑动窗口从时序数据中提取不同的上下文段，每条轨迹使用不同的
历史信息。

---

### Phase 2: 网络结构分析

从连接/响应矩阵中提取结构特征。优先使用响应矩阵（因果连接），若无则从
轨迹计算功能连接（FC = Pearson 相关矩阵）。

| 步骤 | 分析 | 关键指标 | 假设 | 解读 |
|------|------|---------|------|------|
| 2a | **谱分解** | ρ(W), PR, n_dominant, gap_ratio | H1 | ρ≈1 → 近临界；PR/N<0.3 → 低秩 |
| 2b | **社区检测** | Q, k, 社区大小 | H3 | Q>0.3 → 模块化组织 |
| 2c | **层次结构** | hierarchy_index, 树状图 | H3 | 多尺度层次组织 |
| 2d | **模态能量** | top5 累积能量, n@90% | H2 | <5 个模态解释 90% 能量 → 低维 |
| 2e | **可视化** | 连接热图（原始 + 社区排序） | — | 块状结构视觉确认 |

**指标说明：**
- **谱半径 ρ(W)**: W 的最大特征值模。ρ=1 是线性系统的稳定性边界。
- **参与比 PR**: PR = (Σ|λ|)² / Σ|λ|² ∈ [1, N]。衡量特征值分布的有效维度。
  PR/N < 0.3 表示少数特征值主导（低秩结构）。
- **谱隙 gap_ratio**: |λ₁|/|λ₂|。>1.5 表示清晰的主导模态。
- **模块度 Q**: Newman-Girvan 模块度。Q>0.3 表示显著的社区结构。

---

### Phase 3: 动力学特征化

从轨迹数据中提取动力学体制特征。

| 步骤 | 分析 | 关键指标 | 科学目的 |
|------|------|---------|---------|
| 3a | **稳定性分类** | fixed_point / limit_cycle / chaos | 基本动力学体制 |
| 3b | **吸引子分析** | k 个吸引子盆地分布 | 离散吸引子盆地识别（KMeans 聚类） |
| 3c | **轨迹收敛** | distance_ratio, label | 一般性吸引子行为检测 |
| 3d | **Lyapunov 指数** | λ (Rosenstein), regime | 非线性混沌/有序分类（**权威判据**） |
| 3e | **线性化谱 (DMD)** | ρ_DMD, n_slow, n_Hopf, K-Y_lin | 线性化动力学结构 |
| 3f | **功率谱** | f_dom (Hz), 频段功率 | 振荡结构 |
| 3g | **PCA 维度** | n@90%, 方差曲线 | 线性嵌入维度（上界） |
| 3h | **吸引子维度** | D₂, K-Y_lin, PCA n@90% | 吸引子分形维度估计 |

**核心方法说明：**

#### 3d: Rosenstein Lyapunov 指数 (λ₁) — 非线性混沌的唯一可靠判据

**为什么选 Rosenstein 而不是 Wolf？**
- Rosenstein：从轨迹数据直接计算，零额外模型调用，无上下文稀释
- Wolf：需要反复调用 `simulator.rollout()`，每次共享 199/200 的历史上下文
  → 所有轨迹看到相同稀释 → λ 一致偏正（虚假混沌）

**方法：** 在 N 维（或延迟嵌入后的 m 维）状态空间中寻找最近邻对，追踪它们
在时间上的对数距离增长率。三段采样（early/mid/late）提高鲁棒性。

**混沌体制分类：**

| 体制 | λ 范围 | 解读 |
|------|--------|------|
| `stable` | λ < -0.01 | 固定点吸引子 |
| `marginal_stable` | -0.01 ≤ λ < 0 | 边界稳定 |
| `edge_of_chaos` | 0 ≤ λ < 0.01 | 临界边缘 |
| `weakly_chaotic` | 0.01 ≤ λ < 0.1 | 弱混沌 |
| `strongly_chaotic` | λ ≥ 0.1 | 强混沌 |

#### 3e: 线性化谱分析 (DMD) — 结构互补，非混沌判据

**核心思想：** 从自由动力学轨迹中提取连续状态对 (x_t, x_{t+1})，用 Tikhonov
正则化最小二乘拟合最优线性转移算子 A。

**重要科学说明：** DMD 特征值 μ_i 是**线性化**动力学的特征值，不是非线性
Lyapunov 指数。λ_DMD_i = ln|μ_i| 给出的是线性化 Lyapunov 谱，对非线性系统
这是近似值。DMD **不能**检测非线性折叠/拉伸（混沌的核心机制），因此**不能**
作为混沌判据。混沌/有序的判断仅依赖 Rosenstein LLE (3d)。

**DMD 提供的互补信息：**
- ρ_DMD: 谱半径。ρ<1 → 局部稳定；ρ≈1 → 近临界；ρ>1 → 局部不稳定
- n_slow: 慢模态数量（|Re(λ)| < 0.05，弛豫时间 > 20 步的方向）
- n_Hopf: 振荡模态对数（复共轭特征值），频率由 Im(λ)/(2π) 给出
- 线性化 Lyapunov 谱: λ_DMD_1 ≥ ... ≥ λ_DMD_N（从 ln|μ_i| 计算）
- 线性化 K-Y 维度: 从 DMD 谱计算的 Kaplan-Yorke 维度

**验证框架：** 若 max(λ_DMD) ≈ λ₁_Rosenstein，说明线性化是好的近似；
若差异大（非线性指数 Δ >> 1），说明非线性效应主导，DMD 结论仅做参考。

#### 3h: 吸引子维度 — 三重估计

从三个独立视角估计吸引子有效维度：

| 方法 | 类型 | 强度 | 局限 |
|------|------|------|------|
| D₂ (关联维数) | 非线性 | 直接测量吸引子分形结构 | 需足够长的轨迹 |
| K-Y_linear | 线性化 | 利用 DMD 已有结果，零额外成本 | 对强非线性低估 |
| PCA n@90% | 线性 | 简单、鲁棒 | 上界，可能高估 |

---

### Phase 4: 统计验证

验证 Phase 3 结果的统计显著性。

| 步骤 | 分析 | 检验方法 | 科学目的 |
|------|------|---------|---------|
| 4a | **代替数据检验** | 相位随机化 / 时间洗牌 / AR(1) | 真实 LLE 是否显著高于线性基线？ |
| 4b | **随机对照** | 3 个谱半径 × 5 个种子的随机 W | 模型 vs 随机网络的动力学差异 |
| 4c | **嵌入维度** | FNN, D₂, Takens 检验 | 状态空间结构是否一致？ |
| 4d | **结构扰动** | 权重洗牌 / 度保持重连 | 连接结构是否驱动动力学？ |

**4a 代替数据检验说明：**

三种零假设：
- **相位随机化 (AAFT)**: 保留功率谱和振幅分布，破坏非线性相关 → H₀: 线性高斯过程
- **时间洗牌**: 破坏所有时间结构 → H₀: IID 过程
- **AR(1)**: 保留一阶自相关 → H₀: AR(1) 线性过程

19 个代替样本 → 秩检验 p < 0.05。z-score > 2 表示真实系统的非线性动力学显著
优于线性基线。

---

### Phase 5: 高级分析（可选）

默认大部分禁用，可通过配置启用。

| 步骤 | 分析 | 启用条件 | 计算量 | 科学目的 |
|------|------|---------|--------|---------|
| 5a | 虚拟刺激 | 默认启用 | 中 | 测试系统对外部输入的响应 |
| 5b | 能量约束 | `--energy-budget` | 低 | H5: 代谢约束 → 临界性 |
| 5c | 相图 | 手动启用 | 高 | g-扫描找到临界耦合强度 |
| 5d | 可控性 | 手动启用 | 中 | 识别关键控制节点 |
| 5e | 信息流 | 手动启用 | 高 | 转移熵有向因果分析 |
| 5f | 临界减速 | 手动启用 | 低 | 分岔早期预警信号 |

---

### Phase 6: 综合与一致性检查

自动执行以下检查：

1. **Rosenstein LLE vs DMD 一致性 + 非线性指数**
   - λ_DMD_max = ln(ρ_DMD) ← DMD 给出的线性化最大 Lyapunov 指数
   - 非线性指数 Δ = |λ₁_Rosenstein - λ_DMD_max| / |λ₁_Rosenstein|
   - Δ < 0.5: 线性化近似有效，DMD 结构分析结论可信
   - Δ > 1.0: 非线性效应主导，DMD 结论仅做参考
   - 符号不一致（λ₁ > 0 但 ρ < 1）: 非线性折叠在线性稳定系统中产生混沌

2. **代替检验交叉验证**
   - 代替检验显示 `is_nonlinear=True` 增强 LLE 分类的可信度
   - `is_nonlinear=False` 表明动力学可能可以用线性模型解释

3. **假设评估 (H1–H5)**
   - H2 现在综合 D₂、K-Y_linear、PCA n@90% 三重维度估计
   - 综合所有阶段结果，对每个假设给出 `SUPPORTED` / `NOT_SUPPORTED` / `INSUFFICIENT_DATA`

---

## 配置系统

所有参数通过 `config.yaml` 控制。CLI 参数覆盖配置文件值。

### 配置文件结构

```yaml
# 数据生成
data_generation:
  n_init: 100       # 轨迹数量
  steps: 500        # 每条轨迹步数
  seed: 42

# 网络结构（Phase 2）
network_structure:
  enabled: true
  spectral:
    enabled: true
  community:
    enabled: true
    k_range: [3, 4, 5, 6, 7, 8]

# 动力学特征化（Phase 3）
dynamics:
  lyapunov:
    enabled: true
    n_segments: 3          # 三段采样
    delay_embed_dim: 0     # FNN 维度（0=禁用）
  dmd_spectrum:
    enabled: true          # 线性化谱分析 (DMD)
  power_spectrum:
    enabled: true
  attractor_dimension:
    enabled: true          # D₂ + 线性化 K-Y 维度

# 统计验证（Phase 4）
validation:
  surrogate_test:
    enabled: true
    n_surrogates: 19       # p < 0.05
```

### Quick 模式预设

`--quick` 标志应用以下覆盖：

| 参数 | 默认 | Quick |
|------|------|-------|
| n_init | 100 | 20 |
| steps | 500 | 100 |
| n_surrogates | 19 | 9 |
| fnn_max_dim | 8 | 4 |
| corr_dim | true | false |

---

## 输出目录结构

### 单模态（`fmri` / `eeg` / `joint`）

```
outputs/dynamics_pipeline/
├── trajectories.npy              # Phase 1: 轨迹数据
├── response_matrix.npy           # Phase 1: 响应矩阵
├── structure/                    # Phase 2: 网络结构
│   ├── spectral_summary_*.json
│   ├── community_structure_*.json
│   └── connectivity_*.png
├── dynamics/                     # Phase 3: 动力学特征
│   ├── stability_metrics.json
│   ├── lyapunov_report.json
│   ├── jacobian_report.json
│   ├── power_spectrum_report.json
│   └── *.png
├── validation/                   # Phase 4: 统计验证
│   ├── surrogate_test.json
│   ├── analysis_comparison.json
│   └── embedding_dimension.json
├── advanced/                     # Phase 5: 高级分析
│   └── *.json / *.npy
├── plots/                        # 汇总图表
│   ├── trajectory_norms.png
│   ├── pca_trajectories.png
│   ├── lyapunov_histogram.png
│   └── basin_sizes.png
└── pipeline_report.json          # Phase 6: 综合报告
```

### 双模态（`both`）

```
outputs/dynamics_pipeline/
├── fmri/                         # fMRI 模态完整结果（同上结构）
│   ├── trajectories.npy
│   ├── structure/ ...
│   ├── dynamics/ ...
│   └── pipeline_report.json
└── eeg/                          # EEG 模态完整结果（同上结构）
    ├── trajectories.npy
    ├── structure/ ...
    ├── dynamics/ ...
    └── pipeline_report.json
```

---

## 已知问题与解决方案

### Wolf-GS Lyapunov 谱上下文稀释（已删除）

**问题：** TwinBrain 的 ST-GCN 编码器对完整上下文窗口做注意力加权。Wolf-GS 方法
在每个重正化周期调用 `rollout()` 时重置上下文，导致 k 个扰动方向共享 199/200 的
相同历史 → 所有 λ_i ≈ 常数（虚假结果）。

**解决：** Wolf-GS 相关代码已完全移除（`b_lyapunov_spectrum.py`、`Step 15`、
`B_LYA` 实验）。系统的线性化谱结构由 DMD 提供，非线性混沌判据由 Rosenstein 提供。

### DMD 对非线性系统的局限性

**问题：** DMD 是线性分析方法。对非线性系统，DMD 特征值仅反映吸引子附近的
线性化动力学，不能捕获折叠/拉伸等非线性机制。

**缓解措施：**
- 非线性指数 Δ 自动量化线性化偏差程度
- 当 Δ > 1 时，报告自动标注 DMD 结论"仅做参考"
- Rosenstein LLE 始终是混沌判据的权威来源
- D₂ (关联维数) 提供非线性的吸引子维度估计

### 混沌吸引子吸引效应

**问题：** 当初始多样性 < 0.087（0.3 × 随机基线）时，所有轨迹从相近的 x0 汇入
同一混沌吸引子 → distance_ratio 减小 → 误判为"稳定"。

**解决：** Rosenstein LLE 符号是权威指标。distance_ratio 下降可能是"汇入混沌
吸引子"（λ > 0）而非"收敛到固定点"（λ < 0）。代替数据检验（Phase 4）提供
独立的非线性性验证。

---

## 与旧管线的对应关系

| 旧步骤 (run_dynamics_analysis.py) | 新阶段 | 备注 |
|-----------------------------------|--------|------|
| Step 1-2: 加载模型/创建模拟器 | 初始化 | 移到 `run_pipeline()` 顶部，新增模态分发逻辑 |
| Step 3: 自由动力学 | Phase 1a | 新增 `n_temporal_windows` 参数 |
| Step 4: 吸引子分析 | Phase 3b | 移到动力学阶段 |
| Step 5: 虚拟刺激 | Phase 5a | 移到高级阶段，新增 joint 模态节点映射日志 |
| Step 6: 响应矩阵 | Phase 1b | 前移到数据生成阶段 |
| Step 7: 稳定性分析 | Phase 3a | 不变 |
| Step 8: 轨迹收敛 | Phase 3c | 不变 |
| Step 9: Lyapunov 指数 | Phase 3d | 配置驱动方法（默认 Rosenstein），joint 自动强制 Rosenstein，新增 `n_workers` |
| Step 10: 随机对照 | Phase 4b | 移到验证阶段 |
| Step 11: 代替检验 | Phase 4a | 移到验证阶段 |
| Step 12: 嵌入维度 | Phase 4c | D₂ 同时用于 Phase 3h 吸引子维度 |
| Step 13: 功率谱 | Phase 3f | 不变 |
| Step 14: Jacobian (DMD) | Phase 3e | 默认启用，重定位为线性化分析 |
| **Step 15: Wolf-GS 谱** | **已删除** | **上下文稀释不可修复** |
| Step 16: 能量约束 | Phase 5b | 不变 |
| 模态分发（fmri/eeg/both/joint）| `run_pipeline()` | 从旧管线完整移植，`_run_phases_for_modality()` 处理 `both` 模式 |

| 旧实验 (spectral_dynamics) | 新阶段 | 备注 |
|---------------------------|--------|------|
| A: 连接可视化 | Phase 2e | 不变 |
| B_E1: 谱分析 | Phase 2a | 不变 |
| C: 社区结构 | Phase 2b | 不变 |
| D: 层次结构 | Phase 2c | 默认禁用 |
| E2E3: 模态投影 | Phase 2d | 不变 |
| E4: 结构扰动 | Phase 4d | 默认禁用 |
| E5: 相图 | Phase 5c | 默认禁用 |
| E6: 随机对照 | Phase 4b | 合并到 twinbrain 随机对照 |
| F: PCA | Phase 3g | 不变 |
| **B_LYA: Lyapunov 谱** | **已删除** | **与 DMD (3e) 功能重叠** |
| H: 功率谱 | Phase 3f | 使用 twinbrain 实现 |
| I: 能量约束 | Phase 5b | 使用 twinbrain 实现 |

---

## 推荐工作流

### 第一次运行（探索性）

```bash
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
```

检查输出，确认模型加载正确，轨迹形状合理。

### 完整分析（单模态）

```bash
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \
    --n-init 200 --steps 1000
```

### 双模态对比分析

```bash
# 对比 fMRI 和 EEG 各自的动力学特性
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \
    --modality both --output outputs/both_modality
# 结果: outputs/both_modality/fmri/  和  outputs/both_modality/eeg/
```

### 联合多模态分析

```bash
# 使用 fMRI+EEG 拼接状态向量，分析整体脑动力学
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \
    --modality joint --output outputs/joint_modality
```

### 能量约束假设测试

```bash
# 先获取 E* 估计
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 5

# 然后用不同 E_budget 运行
for E in 0.2 0.4 0.7 1.0; do
    python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \
        --energy-budget $E --output outputs/energy_$E
done
```

比较不同 E_budget 下的 LLE 和 PCA 轨迹。预期：E ≈ E* 时 LLE ≈ 0（临界边缘）。
