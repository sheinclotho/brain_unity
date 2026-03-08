# spectral_dynamics — 脑动力学谱分析模块

本目录对 TwinBrain GNN 训练完成的脑连接矩阵（响应矩阵 R 或功能连接矩阵 W）进行**谱分析与动力学实验**，验证六个关于大脑临界态的核心假设（H1–H5）。

---

## 一、快速开始

```python
from spectral_dynamics.run_all import run_all

# 假设你已经有了轨迹数据和响应矩阵（来自 run_dynamics_analysis.py 的输出）
summary = run_all(
    trajectories=trajectories,      # (n_traj, steps, N) float32
    response_matrix=response_matrix, # (N, N) float32
    output_dir=Path("outputs/spectral"),
)
```

或在 `run_dynamics_analysis.py` 中，谱分析已作为最后一步自动运行。

---

## 二、模块一览

| 文件 | 实验 | 假设 | 主要输出 |
|------|------|------|----------|
| `a_connectivity_visualization.py` | A：连接矩阵可视化 | H1 | 热图、度分布、图论指标 |
| `e1_spectral_analysis.py` | E1：谱分析 | H1 | 特征值分布、谱半径 |
| `e2_e3_modal_projection.py` | E2/E3：模态投影 | H2 | 模态能量、低秩近似 |
| `d_hierarchical_structure.py` | D：层次聚类 | H3 | 树状图、模块度 |
| `c_community_structure.py` | C：社区结构 | H3 | 图分区、模块度 |
| `e4_structural_perturbation.py` | E4：结构扰动 | H4 | 扰动敏感性、鲁棒性 |
| `e5_phase_diagram.py` | E5：相图 | H4 | 突触增益 g 扫描、分岔图 |
| `e6_random_comparison.py` | E6：随机对照 | H1–H4 | Erdos-Renyi / 随机矩阵对比 |
| `b_lyapunov_spectrum.py` | B：Lyapunov 谱 | H4 | 全谱 LLE、Kaplan-Yorke 维数 |
| `h_power_spectrum.py` | H：功率谱 | H2 | 1/f 谱、主频、谱斜率 |
| `i_energy_constraint.py` | I：**能量约束** | **H5** | L1 投影对比、稀疏度、LLE |
| `f_pca_attractor.py` | F：PCA 吸引子 | H4 | 4 面板轨迹图 |
| `run_all.py` | 汇总 | 所有 | `run_summary.json` |

---

## 三、各模块详解

### A — 连接矩阵可视化（a_connectivity_visualization.py）

**能解释什么**：大脑区域之间的连接结构长什么样？是否有枢纽节点（hub）？连接是否具有小世界特性？

**能证明什么**：若谱半径 ρ(W) ≈ 1，说明训练出的 GNN 的连接矩阵自然满足临界条件（理论上 ρ=1 是固定点→振荡的边界）。

**输出文件**：`connectivity_heatmap.png`、`degree_distribution.png`、`connectivity_metrics.json`

---

### E1 — 谱分析（e1_spectral_analysis.py）

**能解释什么**：连接矩阵的特征值如何分布？谱半径是多少？最大实部特征值是否接近 1？

**能证明什么**：H1 假设——GNN 学到的连接矩阵使系统处于边缘稳定态（谱半径接近 1）。与随机矩阵（Wigner 半圆律）对比，可证明 GNN 的连接结构不是随机的，而是有组织的。

**输出文件**：`spectral_analysis.json`、`eigenvalue_distribution.png`

---

### E2/E3 — 模态投影（e2_e3_modal_projection.py）

**能解释什么**：连接矩阵的特征模（principal modes）能捕获多少方差？前 k 个模式代表什么功能网络？

**能证明什么**：H2 假设——大脑动力学是低秩的（少数主模式支配整体动力学）。若前 5 个模态占总能量的 80% 以上，说明大脑的状态空间是低维的，即使有 200 个区域，实际自由度远小于 200。

**输出文件**：`modal_energy_*.json`、模态投影热图

---

### C/D — 社区与层次结构（c_community_structure.py, d_hierarchical_structure.py）

**能解释什么**：连接矩阵能否被分解为若干模块（功能网络）？这些模块的层次组织是什么？

**能证明什么**：H3 假设——大脑具有层次模块化结构，GNN 隐式学习了这种结构。模块度 Q > 0.3 是模块化显著的经验标准。

**输出文件**：树状图、社区热图、`community_metrics.json`

---

### E4 — 结构扰动（e4_structural_perturbation.py）

**能解释什么**：如果删除某些连接边（e.g., 切断枢纽节点），动力学如何变化？系统对结构扰动的敏感性是什么？

**能证明什么**：H4 假设的一部分——大脑处于临界态的系统对小扰动很敏感（类 butterfly effect），而远离临界态的系统是鲁棒的。临界态对应最大传播敏感性。

---

### E5 — 相图（e5_phase_diagram.py）

**能解释什么**：x(t+1) = tanh(g·W·x(t)) 中，随着突触增益 g 变化，系统经历什么相变？

**能证明什么**：G 扫描展示了三个相：g < g* → 固定点（静默），g ≈ g* → 临界振荡，g > g* → 混沌。若 GNN 提取的 W 对应 g* ≈ 1，说明系统被训练到临界点附近。

**输出文件**：`phase_diagram.png`（LLE vs g 的分岔图，含三色背景区域）

---

### E6 — 随机对照（e6_random_comparison.py）

**能解释什么**：GNN 学到的连接矩阵比随机矩阵"更特殊"在哪里？

**能证明什么**：通过与同维度随机矩阵（Erdos-Renyi、谱半径匹配的随机正交矩阵）比较，证明 GNN 的动力学特性不是随机的，而是由数据驱动得到的有意义结构。

---

### B — Lyapunov 谱（b_lyapunov_spectrum.py）

**能解释什么**：Wilson-Cowan 系统的全部 Lyapunov 指数，以及 Kaplan-Yorke（KY）维数（吸引子分形维数）。

**能证明什么**：LLE < 0 → 系统收缩到固定点；LLE ≈ 0 → 边界混沌（临界态）；LLE > 0 → 混沌。KY 维数衡量吸引子的复杂度。若 KY 维数在 1–5 之间，说明系统吸引子是低维的（神经科学意义：大脑状态空间的有效维度）。

**输出文件**：`lyapunov_spectrum.png`（谱柱图）、`lyapunov_spectrum.json`

---

### H — 功率谱（h_power_spectrum.py）

**能解释什么**：Wilson-Cowan 系统的神经振荡频率分布？是否存在 1/f 幂律谱？

**能证明什么**：H2 假设的频域版本——大脑活动的 1/f 噪声是临界态的频域特征（Bak et al. 1987；Linkenkaer-Hansen et al. 2001）。若功率谱斜率接近 -1（1/f），且有显著的低频主导，说明系统处于 1/f 临界态区域。

**输出文件**：`power_spectrum.png`（对数-对数谱 + 拟合斜率）

---

### I — 能量约束（i_energy_constraint.py）⭐ 新增

**假设（H5）**：大脑的有限代谢能量供应是维持其近临界态的机制之一。

**核心实验**：对 Wilson-Cowan 模型施加显式能量约束（L1 球投影），对比有/无约束的动力学差异。

#### 为什么是 L1 投影，不是均匀缩放？

| 方法 | 操作 | 科学意义 |
|------|------|----------|
| `g·F(x)` ❌ | 均匀缩放 | 拓扑不变，只是换了尺子 |
| `proj_E(F(x))` ✅ | L1 球投影 | 弱激活置零，强激活保留 → winner-takes-all |

L1 投影的核心性质（软阈值）：

```
x(t+1)_i = max(y_i - λ*, 0)
其中 λ* 使 mean(x) = E_budget
```

- 如果 `y_i < λ*`：神经元被**完全沉默**（代谢成本为零）
- 如果 `y_i > λ*`：神经元**继续放电**（但活动略有降低）
- 这是神经科学中"稀疏编码"（Olshausen & Field 1996）的计算实现

#### 约束激活率（projection_rate）是关键指标

| 激活率 | 含义 | 动力学状态 |
|--------|------|------------|
| ≈ 0% | 约束从不激活（E_budget 足够大） | 无约束 WC 动力学 |
| ≈ 50% | 约束约一半时间激活 | **临界边界** |
| ≈ 100% | 约束总是激活 | 严重压制，固定点 |

**假设预测**：

```
E_budget << E* → 固定点（LLE << 0，激活率 ≈ 100%）
E_budget ≈ E* → 近临界（LLE ≈ 0，激活率 ≈ 50%）  ← 假设支持点
E_budget >> E* → 无约束动力学（激活率 ≈ 0%）
```

其中 `E* = mean(|x_unconstrained|)` 是无约束系统的典型能量。

#### 对 GNN 进行同样实验（EnergyConstrainedSimulator）

在 `twinbrain-dynamics` 版中，`EnergyConstrainedSimulator` 将同样的 L1 投影应用于真实 GNN：

```bash
# 无约束（基线）
python run_dynamics_analysis.py --quick --model M.pt --graph G.pt \
    --output outputs/baseline

# 中等能量约束（E_budget = 0.7 × E*，自动标定）
python run_dynamics_analysis.py --quick --model M.pt --graph G.pt \
    --energy-budget 0.7 --output outputs/energy_0.7
```

关键区别：**约束后的状态会被注入 GNN 的上下文窗口**，确保下一步预测看到的是约束后的历史，而非假想的无约束历史。这才是真正的能量约束下的自我演化。

**输出文件**：
- `energy_constraint_{label}.json`：各条件 LLE / 振荡幅度 / 均值活动 / 约束激活率
- `energy_constraint_{label}.png`：3 行图（PCA 相空间 / 均值活动时序 / 汇总指标）

---

### F — PCA 吸引子（f_pca_attractor.py）

**能解释什么**：自由动力学在相空间中的形态？轨迹是否收敛到一个有限的吸引子？

**四面板输出**：2D 时间渐变轨迹 / 密度热图 / 3D PCA 吸引子 / 方差解释曲线（90% 阈值）。

**能证明什么**：若大量轨迹（从不同初始状态出发）最终收敛到同一区域，说明存在全局吸引子。若该吸引子是低维的（2–3 PC 占 90%+ 方差），与临界态的低维吸引子假设一致。

---

## 四、假设汇总

| 假设 | 内容 | 验证实验 | 核心指标 |
|------|------|----------|----------|
| **H1** | GNN 学到近临界连接结构 | E1, A, E6 | ρ(W) ≈ 1 |
| **H2** | 动力学是低维/低秩的 | E2/E3, H, F | 前 5 模态占 80%+ 能量；1/f 谱 |
| **H3** | 层次模块化结构 | C, D | 模块度 Q > 0.3 |
| **H4** | 系统处于混沌边缘（临界态） | E5, B, E4 | LLE ≈ 0；KY 维数 1–10 |
| **H5** | 有限能量约束维持临界态 | **I** | 适中 E_budget 使 LLE → 0；激活率 ≈ 50% |

---

## 五、run_all.py — 一键运行全部实验

```python
summary = run_all(
    trajectories=trajs,         # (n_traj, steps, N)
    response_matrix=R,          # (N, N) 来自 response_matrix 实验
    output_dir=Path("outputs"),
    experiments=["A", "B_LYA", "C", "D", "E2E3", "E4", "E5", "E6", "F", "H", "I"],
    seed=42,
)
# summary["hypotheses"] 包含所有假设的自动评估结果
```

输出 `run_summary.json` 包含：
- `results`：每个实验的完整数值结果
- `hypotheses`：H1–H5 的自动评估（supported / not_supported）

---

## 六、计算成本参考

| 实验 | N=190, 典型参数 | GPU 时间 | CPU 时间 |
|------|----------------|----------|----------|
| A (可视化) | — | < 1s | < 1s |
| E1 (谱) | 特征值分解 | < 1s | < 1s |
| E2/E3 (模态) | SVD | < 1s | < 1s |
| C/D (社区) | Louvain | < 5s | < 5s |
| E5 (相图) | 15 个 g 值 × 15 轨迹 × 200 步 | ~2s | ~10s |
| B (Lyapunov 谱) | Wolf 谱 | ~30s | ~5min |
| H (功率谱) | FFT | < 1s | < 1s |
| **I (能量约束)** | 4 条件 × 10 轨迹 × 200 步 | ~2s | ~10s |
| F (PCA) | PCA + 绘图 | < 5s | < 5s |

**GNN 版（run_dynamics_analysis.py）能量约束**：
- `--quick --energy-budget 0.7`：4,000 次 GNN 调用，GPU ~30–90s
- 建议先用 `--quick` 验证方向，再用完整参数运行

---

## 七、文献依据

| 理论 | 文献 | 本模块对应 |
|------|------|----------|
| 临界态最大化信息传输 | Shew et al. (2011) *J. Neurosci.* | E5, B, I |
| 稀疏编码 = L1 能量约束 | Olshausen & Field (1996) *Nature* | I |
| 神经编码的代谢成本 | Lennie (2003) *Curr. Biol.* | I |
| 1/f 神经噪声 = 临界态特征 | Linkenkaer-Hansen et al. (2001) | H |
| Lyapunov 指数与大脑临界 | Shew & Plenz (2013) *Neuroscientist* | B |
| 模块化网络与临界 | Rubinov et al. (2011) *PLoS Comp. Biol.* | C, D |
