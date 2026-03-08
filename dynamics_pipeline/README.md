# dynamics_pipeline — 统一脑动力学分析流程

> **一个配置驱动的、分阶段的脑网络动力学测试流程。**
> 将 `twinbrain-dynamics`（模型驱动）和 `spectral_dynamics`（矩阵驱动）合并为单一管线。

---

## 快速开始

```bash
# 完整分析
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt

# 快速预实验（参数大幅缩减，适合验证流程）
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick

# 只运行指定阶段
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --phases 1 3

# 带能量约束
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --energy-budget 0.3
```

---

## 设计理念

### 我们在做什么？

我们要回答一个核心科学问题：**训练好的 TwinBrain 数字孪生模型展示了怎样的动力学行为？**

具体而言，我们测试五个假设：

| 假设 | 问题 | 对应分析 |
|------|------|---------|
| **H1** | 脑网络连接是否具有低秩谱结构？ | 特征值分解、参与比(PR)、谱半径 |
| **H2** | 动力学是否低维？ | PCA 方差曲线、模态能量投影 |
| **H3** | 系统是否处于近临界状态？ | Lyapunov 指数、DMD 谱半径 |
| **H4** | 是否存在脑样振荡？ | 功率谱密度、频段分析 |
| **H5** | 代谢能量约束是否维持临界性？ | L1 投影实验 |

### 为什么要合并两个模块？

原有两个模块存在显著重叠（Lyapunov 指数、功率谱、随机对照、PCA 均有双重实现），
且没有统一的假设评估和一致性检查。本管线消除冗余，用单一配置文件控制所有分析，
并在最后自动进行跨步骤一致性验证。

### 关于 Wolf-GS Lyapunov 谱的重要说明

**Wolf-GS 方法在 TwinBrain 架构下存在不可修复的上下文稀释偏差。**

具体表现：所有 k 个 Lyapunov 指数几乎相同（如 0.126–0.133），D_KY 恰好等于 k，
跨轨迹标准差接近零。这是因为 TwinBrain 的 ST-GCN 编码器对 200 步上下文窗口做
注意力加权，单步扰动被 199 个相同历史步稀释为 ε/200 → 所有扰动方向看到相同
增长率。

**本管线用 DMD（Dynamic Mode Decomposition）谱分析替代 Wolf-GS。** DMD 直接从
自由动力学轨迹提取线性转移算子的特征值，零额外模型调用，无上下文稀释问题。

---

## 管线架构

```
Phase 1: Data Generation        → 轨迹 + 响应矩阵
    ↓
Phase 2: Network Structure      → 谱分析、社区、层次、模态能量
    ↓
Phase 3: Dynamics Characterisation → 稳定性、LLE、DMD 谱、PSD、PCA
    ↓
Phase 4: Statistical Validation  → 代替检验、随机对照、嵌入维度
    ↓
Phase 5: Advanced (optional)     → 刺激、能量约束、可控性、信息流
    ↓
Phase 6: Synthesis               → 假设评估 + 一致性检查 + 报告
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
| 3c | **轨迹收敛** | distance_ratio, label | 一般性吸引子行为检测（不依赖盆地结构） |
| 3d | **Lyapunov 指数** | λ (Rosenstein), regime | 混沌/有序分类 |
| 3e | **DMD 谱** | ρ_DMD, n_slow, n_Hopf, f_dom | 线性化动力学谱 |
| 3f | **功率谱** | f_dom (Hz), 频段功率 | 振荡结构 |
| 3g | **PCA 维度** | n@90%, 方差曲线 | 有效动力学维度 |

**核心方法说明：**

#### 3d: Rosenstein Lyapunov 指数 (λ)

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

#### 3e: DMD 谱分析（替代 Wolf-GS Lyapunov 谱）

**核心思想：** 从自由动力学轨迹中提取连续状态对 (x_t, x_{t+1})，用 Tikhonov
正则化最小二乘拟合最优线性转移算子 A。A 的特征值直接给出线性化 Lyapunov 谱。

**优势：**
- 零额外模型调用（直接使用已有轨迹）
- 无上下文稀释偏差
- 同时提供慢模态数量 (n_slow)、Hopf 分岔对数 (n_Hopf)、主导振荡频率 (f_dom)

**输出指标：**
- ρ_DMD: DMD 谱半径。ρ<1 → 稳定；ρ≈1 → 近临界；ρ>1 → 不稳定
- n_slow: |Re(λ)| < 0.05 的慢模态数量（对应弛豫时间 > 20 步的模态）
- n_Hopf: 虚部 |Im(λ)| > 0.01 的特征值对数量（振荡模态）
- f_dom: 主导振荡频率 (Hz/step)

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

1. **Rosenstein LLE vs DMD ρ 一致性**
   - λ > 0 应对应 ρ > 1（混沌）
   - λ < 0 应对应 ρ < 1（稳定）
   - 不一致 → 非线性效应可能将系统推过混沌边界

2. **代替检验交叉验证**
   - 代替检验显示 `is_nonlinear=True` 增强 LLE 分类的可信度
   - `is_nonlinear=False` 表明动力学可能可以用线性模型解释

3. **假设评估 (H1–H5)**
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
    enabled: true          # DMD 替代 Wolf-GS
  power_spectrum:
    enabled: true

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

---

## 已知问题与解决方案

### Wolf-GS Lyapunov 谱上下文稀释

**问题：** TwinBrain 的 ST-GCN 编码器对完整上下文窗口做注意力加权。Wolf-GS 方法
在每个重正化周期调用 `rollout()` 时重置上下文，导致 k 个扰动方向共享 199/200 的
相同历史 → 所有 λ_i ≈ 常数（虚假结果）。

**表现：** λ₁=0.133, λ₂=0.132, ..., λ₁₀=0.126, D_KY=10.00（恰好等于 k）。

**解决：** 本管线用 DMD 谱替代 Wolf-GS。DMD 直接从轨迹数据提取线性转移算子，
无需额外模型调用，无上下文稀释。

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
| Step 1-2: 加载模型/创建模拟器 | 初始化 | 移到 `run_pipeline()` 顶部 |
| Step 3: 自由动力学 | Phase 1a | 不变 |
| Step 4: 吸引子分析 | Phase 3b | 移到动力学阶段 |
| Step 5: 虚拟刺激 | Phase 5a | 移到高级阶段 |
| Step 6: 响应矩阵 | Phase 1b | 前移到数据生成阶段 |
| Step 7: 稳定性分析 | Phase 3a | 不变 |
| Step 8: 轨迹收敛 | Phase 3c | 不变 |
| Step 9: Lyapunov 指数 | Phase 3d | 固定为 Rosenstein |
| Step 10: 随机对照 | Phase 4b | 移到验证阶段 |
| Step 11: 代替检验 | Phase 4a | 移到验证阶段 |
| Step 12: 嵌入维度 | Phase 4c | 移到验证阶段 |
| Step 13: 功率谱 | Phase 3f | 不变 |
| Step 14: Jacobian (DMD) | Phase 3e | 默认启用 |
| **Step 15: Wolf-GS 谱** | **移除** | **用 DMD 替代** |
| Step 16: 能量约束 | Phase 5b | 不变 |

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
| B_LYA: Lyapunov 谱 | **移除** | **用 DMD 替代** |
| H: 功率谱 | Phase 3f | 使用 twinbrain 实现 |
| I: 能量约束 | Phase 5b | 使用 twinbrain 实现 |

---

## 推荐工作流

### 第一次运行（探索性）

```bash
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt --quick
```

检查输出，确认模型加载正确，轨迹形状合理。

### 完整分析

```bash
python -m dynamics_pipeline.run --model best_model.pt --graph graph.pt \
    --n-init 200 --steps 1000
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
