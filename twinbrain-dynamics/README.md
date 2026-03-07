# TwinBrain Dynamics — 输出结果解读指南

本文档说明 `run_dynamics_analysis.py` 完整管线运行后，`outputs/plots/` 目录中每幅图的含义、正常与异常的判断标准，以及与其他输出文件（JSON / NPY）的对应关系。

---

## 目录

1. [快速总览：输出文件清单](#1-快速总览输出文件清单)
2. [trajectory_norms.png — 轨迹范数](#2-trajectory_normspng--轨迹范数)
3. [pca_trajectories.png — PCA 相空间投影](#3-pca_trajectoriespng--pca-相空间投影)
4. [region_heatmap.png — 脑区活动热图](#4-region_heatmappng--脑区活动热图)
5. [basin_sizes.png — 吸引子盆地分布](#5-basin_sizespng--吸引子盆地分布)
6. [trajectory_convergence.png — 轨迹收敛曲线](#6-trajectory_convergencepng--轨迹收敛曲线)
7. [lyapunov_histogram.png — Lyapunov 指数分布](#7-lyapunov_histogrampng--lyapunov-指数分布)
8. [lyapunov_growth.png — Wolf 对数增长曲线](#8-lyapunov_growthpng--wolf-对数增长曲线)
9. [response_matrix.png — 刺激响应矩阵](#9-response_matrixpng--刺激响应矩阵)
10. [response_column_stats.png — 响应矩阵列统计](#10-response_column_statspng--响应矩阵列统计)
11. [stim_response_\{pattern\}_node\{N\}.png — 单节点刺激响应](#11-stim_response_pattern_noden-png--单节点刺激响应)
12. [JSON / NPY 数据文件](#12-json--npy-数据文件)
13. [常见异常模式速查表](#13-常见异常模式速查表)

---

## 1. 快速总览：输出文件清单

| 文件 | 步骤 | 内容摘要 |
|------|------|----------|
| `plots/trajectory_norms.png` | 步骤 3 | 多条轨迹的 L₂ 范数随时间变化 |
| `plots/pca_trajectories.png` | 步骤 3 | 轨迹在 PCA 前两维的路径图 |
| `plots/region_heatmap.png` | 步骤 3 | 第一条轨迹的逐脑区活动热图 |
| `plots/basin_sizes.png` | 步骤 4 | 吸引子盆地大小（K-Means 分组） |
| `plots/trajectory_convergence.png` | 步骤 8 | 随机轨迹对间平均距离随时间衰减 |
| `plots/lyapunov_histogram.png` | 步骤 9 | LLE 值分布直方图 |
| `plots/lyapunov_growth.png` | 步骤 9 | Wolf 对数增长曲线（仅 wolf/both 方法） |
| `plots/response_matrix.png` | 步骤 6 | 刺激响应矩阵热图（原始 + 行归一化） |
| `plots/response_column_stats.png` | 步骤 6 | Hub 节点检测 + 刺激特异性分析 |
| `plots/stim_response_*.png` | 步骤 5 | 单节点三阶段刺激时间曲线 |
| `trajectories.npy` | 步骤 3 | 形状 (n_init, steps, n_regions) |
| `lyapunov_values.npy` | 步骤 9 | 形状 (n_init,)，每条轨迹 LLE |
| `rosenstein_values.npy` | 步骤 9 | 形状 (n_init,)，Rosenstein LLE |
| `log_growth_curve.npy` | 步骤 9 | 形状 (n_periods,)，Wolf 增长曲线 |
| `distance_curve.npy` | 步骤 8 | 形状 (steps,)，轨迹对平均距离 |
| `response_matrix.npy` | 步骤 6 | 形状 (n_nodes, n_regions) |
| `lyapunov_report.json` | 步骤 9 | LLE 统计 + 混沌分类 |
| `stability_metrics.json` | 步骤 7 | 稳定性分类统计（方法 A/B/C） |
| `analysis_comparison.json` | 步骤 10 | 真实模型 vs 随机模型 LLE 对比 |
| `surrogate_test.json` | 步骤 11 | 代替数据检验 z-score + 显著性 |

---

## 2. `trajectory_norms.png` — 轨迹范数

### 图的内容

纵轴：状态向量 **L₂ 范数** `‖x(t)‖₂`，即所有脑区活动值的均方根；横轴：预测步数（或时间）。每条彩色线代表一条独立轨迹（最多显示 20 条）。

### 如何解读

| 观察到的现象 | 含义 |
|-------------|------|
| 所有曲线快速收敛到同一水平线 | 系统有单一**固定点**或**极限环**吸引子，轨迹多样性被吸引子吸收 |
| 曲线收敛到 2–3 组不同水平线 | 系统有**多个吸引子**（对应 `basin_sizes.png` 中的多个盆地） |
| 曲线持续发散或无规律震荡 | 系统处于**混沌**或参数配置错误 |
| 所有曲线完全重合（从头到尾） | **轨迹多样性不足**（context 主导），不同初始状态对模型输出的影响被稀释；LLE 估计可能偏低 |
| 曲线先下降后稳定 | 正常的**瞬态收敛**行为：系统从随机初始状态被吸引到吸引子附近 |

### 与其他步骤的关联

- 若曲线收敛极快（< 50 步），步骤 9 的 `convergence_threshold` 可能触发 Wolf 跳过（见 `lyapunov_report.json` 的 `skipped_wolf` 字段）。
- 曲线最终的水平线高度对应 `region_heatmap.png` 的平均活动水平。

---

## 3. `pca_trajectories.png` — PCA 相空间投影

### 图的内容

将所有轨迹的状态向量用 **PCA 降至前两个主成分**，在二维平面绘制路径。  
- **○（圆圈）** = 轨迹起点  
- **★（星号）** = 轨迹终点  
- 颜色表示轨迹编号（viridis 渐变）  
- 坐标轴标注了 PC1、PC2 各自解释的方差比例

### 如何解读

| 形态 | 含义 |
|------|------|
| 所有终点（★）汇聚到同一小区域 | 单吸引子：系统高度稳定，不同初始条件最终收敛 |
| 终点分成 2–3 个离散簇 | 多吸引子：存在多稳态，轨迹落入不同盆地 |
| 终点形成封闭环（轨迹末端绕圈） | **极限环吸引子**：系统持续振荡，不收敛到固定点 |
| 终点随机散布、无规律 | 混沌吸引子：或轨迹步数不足以收敛 |
| 所有轨迹路径几乎重合 | Context 主导：x0 多样性被长上下文窗口稀释 |

**注意**：PCA 只保留了总方差的一小部分（轴标题中显示百分比）。若 PC1+PC2 < 50%，图形可能严重失真，需配合 `region_heatmap.png` 综合判断。

### 与步骤 12 的关联

步骤 12 给出 `PCA 95%方差需 N 主成分`。若 N > 10，说明吸引子在高维空间展开，PCA 二维投影仅供方向性参考，不能准确反映吸引子形状。

---

## 4. `region_heatmap.png` — 脑区活动热图

### 图的内容

**纵轴**：脑区索引（0 = 第一区，N-1 = 最后一区）；**横轴**：时间步数。颜色表示活动幅度（归一化到 [0, 1]，**黑色** = 低活动，**白色/亮黄** = 高活动，使用 `hot` 色图）。

仅显示轨迹集合中的**第一条**轨迹（Trajectory 0）。

### 如何解读

| 观察到的现象 | 含义 |
|-------------|------|
| 前几步颜色剧烈变化，后期趋于稳定 | 正常瞬态 → 收敛：系统从随机初始状态被吸引子捕获 |
| 持续出现水平条纹（同一脑区始终偏亮/偏暗） | 脑区有**时间稳定的功能偏好**（某些区域持续高/低活动） |
| 整幅图颜色均匀，无明显结构 | 活动趋于均匀，或模型对初始条件响应微弱（context 主导） |
| 某几行持续高亮（水平亮条） | **Hub 节点**：这些脑区对网络活动贡献大（可对照 `response_column_stats.png` 确认） |
| 颜色随时间周期性变化（明暗交替） | 系统处于**极限环**，与稳定性分析（步骤 7）的 `limit_cycle` 分类对应 |
| 图像在某一时刻突然改变风格 | 可能存在吸引子**转换**（多稳态系统在不同盆地之间切换） |

**实用技巧**：放大图像左侧（时间步 0–50）可观察瞬态特征；右侧反映稳态行为。

---

## 5. `basin_sizes.png` — 吸引子盆地分布

### 图的内容

**条形图**，横轴为吸引子编号（A, B, C, ...），纵轴为**盆地大小**（该吸引子吸引的轨迹比例）。每个条形顶部标注百分比。

算法：取所有轨迹末尾 `tail_steps` 步的平均状态，用 **K-Means 聚类**（测试 k = 2,3,4,5,6），选择轮廓系数最高的 k 值。

### 如何解读

| 吸引子数量 | 含义 |
|-----------|------|
| **1 个吸引子（Attractor A = 100%）** | 系统**单稳态**：所有轨迹最终收敛到同一状态。这是最常见的结果，说明脑网络有强烈的全局吸引子 |
| **2–3 个吸引子** | **多稳态**：不同初始条件引导系统到不同的稳定状态，可能对应不同的功能模式（如休息态 vs 任务态） |
| **5+ 个吸引子** | 可能是真实高维多稳态，也可能是步数不足（轨迹尚未收敛）或 K-Means 过度拟合 |

**注意**：若 `pca_trajectories.png` 中终点没有明显簇结构，但 `basin_sizes.png` 显示多吸引子，可能是 K-Means 在高维空间的伪多稳态（降维后重叠不代表高维空间重叠）。

---

## 6. `trajectory_convergence.png` — 轨迹收敛曲线

### 图的内容

纵轴：随机选取的 `n_pairs` 对轨迹之间的**平均 L₂ 距离**；横轴：预测步数。曲线单调性直接反映吸引子结构。

对应数据文件：`distance_curve.npy`（形状 `(steps,)`）。

### 如何解读

| 曲线形态 | `lyapunov_report.json` 中 `distance_ratio` | 含义 |
|----------|-------------------------------------------|------|
| **持续单调下降，末尾接近 0** | `< 0.05`（触发 Wolf 跳过） | 强收敛，单一吸引子 |
| **先快速下降后趋于平稳（非零水平线）** | `0.05–0.3` | 有限收敛：存在吸引子但轨迹间保留一定差异 |
| **基本水平（无下降趋势）** | `≈ 1.0` | 无收敛：轨迹发散或极弱吸引子 |
| **先下降后上升** | > 1.0 | 异常：初始收敛后发散，可能为瞬态吸引子或多稳态切换 |

**关键数值**：
- `distance_ratio = 终止距离 / 初始距离`
- 日志示例：`比率=0.010 → converging`（初始 0.2547，终止 0.0026，收缩约 100 倍）

---

## 7. `lyapunov_histogram.png` — Lyapunov 指数分布

### 图的内容

所有 `n_init` 条轨迹的**最大 Lyapunov 指数（LLE）** 分布直方图。  
- **红色竖线**：均值  
- **橙色虚线**：中位数  
- **黑色虚线**：λ = 0 分界线（稳定/混沌边界）

颜色：鲑鱼红。X 轴单位：**每预测步的 LLE**（与 fMRI TR 对应，默认 2s / 步）。

对应数据：`lyapunov_values.npy`（Wolf 方法）或 `rosenstein_values.npy`（Rosenstein 方法）。

### 混沌分类阈值

| λ 范围 | 分类 | 解读 |
|--------|------|------|
| `λ < −0.01` | `stable` | 稳定收敛，扰动快速衰减 |
| `−0.01 ≤ λ < 0` | `marginal_stable` | 弱收敛，扰动缓慢衰减 |
| `0 ≤ λ < 0.01` | `edge_of_chaos` | 混沌边缘，中性稳定，神经计算最优工作点 |
| `0.01 ≤ λ < 0.1` | `weakly_chaotic` | 弱混沌，轨迹缓慢发散 |
| `λ ≥ 0.1` | `strongly_chaotic` | 强混沌，轨迹快速发散，长期预测不可靠 |

### 如何解读

| 分布形状 | 含义 |
|----------|------|
| **所有值集中在 λ ≈ 同一负值** | 强收敛吸引子；若 std < 1e-3 且用 Wolf 方法，可能存在**上下文稀释偏差** |
| **均值 λ ≈ 0.01–0.1（弱混沌）** | 系统对初始条件敏感，但仍有可辨识结构（见日志中 `WEAKLY_CHAOTIC`） |
| **双峰分布** | 轨迹分属两类动力学区域（多稳态或不同吸引子盆地） |
| **宽分布（std 很大）** | 不同初始状态经历非常不同的局部 Lyapunov 动力学；混沌行为空间不均匀 |
| **所有值几乎完全相同（std < 0.001）** | **Wolf 方法**下的上下文稀释偏差：日志会给出 `⚠ Wolf LLE 跨轨迹标准差...` 警告；此时应以 `rosenstein_values.npy` 为主要参考 |

### 注意：Wolf vs Rosenstein

若日志显示 `wolf_bias_warning: true`，直方图展示的是**偏差值**（所有值聚集在某负值附近）。此时请：
1. 查看 `rosenstein_values.npy` 的分布（通常接近 N(0.02, 0.0001)）
2. 以 `lyapunov_report.json` 中 `mean_rosenstein` 为主要混沌评估依据

---

## 8. `lyapunov_growth.png` — Wolf 对数增长曲线

> **此图仅在 `method: "wolf"` 或 `method: "both"` 时生成。**
> 使用默认 `method: "rosenstein"` 时不会生成此图。

### 图的内容（双面板）

**左图 — 逐周期对数增长（Per-Period Log Growth）**

每个 Wolf 重归一化周期的 `log(r/ε)` 值（柱状图）：
- **红色柱**：`log(r/ε) > 0`，该周期扰动放大（混沌特征）
- **蓝色柱**：`log(r/ε) < 0`，该周期扰动收缩（稳定特征）
- **绿线**：滑动均值（显示 LLE 收敛过程）
- **红色水平线**：收敛后均值（= LLE × renorm_steps）
- **橙色阴影**：±1σ 区间

**右图 — 去趋势累积对数增长（Detrended Cumulative）**

原始累积和 `S(t) = Σ log_growth` 减去线性拟合后的残差：
- `R² ≈ 1.0`：LLE 完全收敛（纯直线）
- `R² < 0.9`：暂态偏差或非平稳动力学
- 左侧灰色阴影区：暂态期（Lyapunov 向量尚未对齐）

### 如何解读

| 观察 | 含义 |
|------|------|
| **后期所有柱子高度完全相同** | 对于**稳定收敛系统**（λ < 0），Lyapunov 向量对齐后每周期增长率收敛到固定值，柱子高度趋于一致；对于**弱混沌系统**（λ > 0），柱子应在均值附近有小幅波动 — 若所有柱高完全相同且 std ≈ 0，则可能存在 Wolf 上下文稀释偏差 |
| **前 3–5 个柱子偏离后期均值** | 正常**暂态期**（Lyapunov 向量对齐阶段） |
| **柱子始终接近零且无方差** | Wolf 上下文稀释偏差：见 `lyapunov_histogram.png` 注意事项 |
| **右图残差持续波动（不收敛到零）** | 非平稳系统或混沌行为：局部 LLE 随时间变化（吸引子未完全稳定） |
| **右图 R² < 0.8** | LLE 估计可信度较低，考虑增加 `n_segments` 或切换为 `rosenstein` 方法 |

---

## 9. `response_matrix.png` — 刺激响应矩阵

### 图的内容（双面板热图）

**面板 A（左）— 原始响应 `R[i, j]`**

`R[i, j]` = 刺激节点 `i` 时，节点 `j` 的活动变化量（红 = 增强，蓝 = 抑制）。
- **行 i**：被刺激的源节点
- **列 j**：产生响应的目标节点
- **对角线（+标记）**：`i = j`，即直接刺激效果

**面板 B（右）— 行归一化响应 `R[i,j] − mean_j(R[i,j])`**

减去每行均值（全局偏置），保留**刺激特异性**响应。

### 如何解读

| 观察 | 含义 | 好/坏？ |
|------|------|---------|
| 面板 A 对角线最亮 | 被刺激节点自身响应最强 → 模型有空间特异性 | ✅ 理想 |
| **面板 A 出现竖条纹**（某几列始终亮） | Hub 节点：这些节点对任何刺激都强烈响应（见 `response_column_stats.png`）；这是**真实网络特性**，不是问题 | ✅ 可接受，参考面板 B |
| **面板 B 出现块状结构** | 刺激有功能社区特异性（刺激 A 区域更影响 A 区附近）→ 模型保留了解剖连接 | ✅ 理想 |
| **面板 B 仍是竖条纹** | 响应完全由连接矩阵列结构决定，与刺激位置无关 → 模型可能过度全局耦合或测量窗口太短 | ⚠️ 可改进 |
| 矩阵全为近零值 | 刺激幅度不足，或步骤 6 `n_nodes` 太少导致统计不足 | ⚠️ 调整参数 |

对应数据：`response_matrix.npy`，形状 `(n_nodes, n_regions)`。

---

## 10. `response_column_stats.png` — 响应矩阵列统计

### 图的内容（三子图）

**子图 1 — 列均值（Hub 检测）**

`column_mean[j] = mean|R[:, j]|`：节点 `j` 对所有刺激的平均响应强度。
- 红色柱：前 10 个最高值节点（Hub 候选）
- 虚线：均值；点线：+2σ 阈值（Hub 判定门槛）

**子图 2 — 刺激特异性**

`S(i) = std_j(R[i, j])`：刺激节点 `i` 的响应是否有空间选择性。
- 橙色柱：特异性低于均值的 50% 的节点（扩散型刺激）
- ⚠ 文字框：若所有节点特异性高度一致，可能是测量窗口仍处于暂态

**子图 3 — 列均值分布直方图**

列均值的分布形状，用于检测 hub 节点的统计特征。

### 如何解读

| 观察 | 含义 |
|------|------|
| 少数节点列均值远超 +2σ（日志报 `hub节点 ±2σ: 6个`） | 存在明确 Hub：这 6 个节点高度受连接影响，无论刺激哪里都响应 |
| 子图 3 右偏（少数高值，多数低值） | **重尾分布**：Hub 结构明确，符合真实脑网络（幂律分布）|
| 子图 2 所有柱子高度一致 | 模型响应空间无差异化（全局耦合过强，或 n_nodes 太少）|
| 低特异性节点占比高（日志报 `低特异性节点: 40个`） | 系统对 40/200 个节点的刺激产生近乎相同的响应，可能是扩散性强的解剖区域 |

---

## 11. `stim_response_{pattern}_node{N}.png` — 单节点刺激响应

### 文件命名规则

格式：`stim_response_{pattern}_node{N}.png`  
- `{pattern}`：刺激波形，如 `sine`、`square`、`step`  
- `{N}`：被刺激节点索引

### 图的内容

三段时序曲线：**[基线期] → [刺激期（黄色背景）] → [恢复期]**

- **红线**：被刺激节点（直接响应）
- **彩色线**：响应最强的 5 个非刺激节点（传播效果）
- **灰色虚线**：`Stim on`（刺激开始）
- **黑色虚线**：`Stim off`（刺激结束）

图例中标注每个节点的 `Δ = 刺激期均值 - 基线期均值`。

### 如何解读

| 观察 | 含义 |
|------|------|
| 红线（刺激节点）在刺激期显著上升 | 刺激有效传导到目标节点 |
| 其他彩色线也同步变化（正相关） | 刺激传播到功能相连的远端节点 |
| 停止刺激后曲线回落到基线 | 系统有**弹性**：扰动后恢复（收敛型） |
| 停止刺激后曲线维持在新水平线 | **持续效应**：刺激改变了系统的稳态 |
| 刺激期几乎无变化（Δ ≈ 0） | 刺激幅度不足，或该节点被网络"屏蔽" |
| 非刺激节点出现负响应（蓝线） | 抑制性传播（网络中存在抑制性连接） |
| `sine` 和 `step` 响应形态完全不同 | 系统对频率敏感，存在频率选择性 |

---

## 12. JSON / NPY 数据文件

### `lyapunov_report.json`

```
{
  "mean_lyapunov": float,          // Wolf/FTLE LLE 均值（有偏差时请忽略）
  "mean_rosenstein": float,        // Rosenstein LLE 均值（推荐主要参考）
  "std_rosenstein": float,         // Rosenstein std（接近 0 → 单一全局吸引子）
  "chaos_regime": str,             // "stable" / "edge_of_chaos" / "weakly_chaotic" 等
  "wolf_bias_warning": bool,       // true → Wolf 结果不可用，以 Rosenstein 为准
  "skipped_wolf": bool,            // true → 系统强收敛，Wolf 已跳过
  "initial_trajectory_diversity": float,  // 初始轨迹多样性（<0.02 → context 主导）
  "delay_embed_dim": int,          // 0 = 未使用延迟嵌入；> 1 = 使用的 Takens 维度
  ...
}
```

**使用提示**：
- 若 `wolf_bias_warning = true`，使用 `mean_rosenstein` 作为 LLE 的真实值
- 若 `std_rosenstein < 1e-3` 且 `initial_trajectory_diversity >= 0.02`，这是**单一全局吸引子**的有力证据（λ 是吸引子属性，不同起点收敛到相同值是正确行为）

### `stability_metrics.json`

```
{
  "classification_counts": {...},     // 方法 C（自适应，最准确）
  "classification_counts_v1": {...},  // 方法 A（邻接差分）
  "classification_counts_v2": {...},  // 方法 B（延迟距离）
  "delta_ratio_stats": {"mean": ..., "median": ..., "p25": ..., "p75": ..., "p95": ...},
  "acf_score_stats": {"mean": ..., ...}
}
```

**最重要字段**：`classification_counts`（方法 C 的结果），典型值如 `{"limit_cycle": 200, "fixed_point": 0}`。

若方法 A/B 与方法 C 不一致（日志中会给出原因说明），以**方法 C 为准**（方法 C 使用无量纲 delta_ratio，不受脑区数量影响）。

### `analysis_comparison.json`

```
{
  "real_model_lle": float,           // 真实模型的 LLE（Rosenstein）
  "random_stable_baseline": {...},   // 随机稳定基线（ρ=0.9）LLE
  "random_chaotic_baseline": {...},  // 随机混沌基线（ρ=2.0）LLE
  "comparison_note": str,            // 真实 LLE 与随机基线的关系文字描述
}
```

**使用提示**：若真实模型 LLE ≈ 随机混沌系统的 LLE（见日志 `与随机混沌系统相当`），说明模型动力学的混沌程度与随机权重矩阵（ρ=2.0）相当——不是问题，而是大脑工作点接近混沌边缘的证据。

### `surrogate_test.json`

```
{
  "real_lle": float,                 // 真实系统 LLE
  "phase_randomize": {"z": float, "significant": bool},  // z > 1.96 → p < 0.05
  "shuffle": {"z": float, "significant": bool},
  "ar": {"z": float, "significant": bool},
  "summary": str                     // 综合显著性结论
}
```

**使用提示**：三类代替检验均显著（如 z = 263, 549, 209）说明系统具有**显著非线性动力学**，不可被线性随机过程（AR 模型）或随机相位模型解释。

---

## 13. 常见异常模式速查表

| 问题现象 | 可能原因 | 建议操作 |
|----------|----------|----------|
| 步骤 9 耗时 > 30 分钟 | `method: "both"` 运行 Wolf（每条轨迹 ~20 次模型调用） | 改为 `method: "rosenstein"` |
| `wolf_bias_warning: true` | TwinBrainDigitalTwin 上下文稀释偏差（Wolf 设计缺陷） | 以 `mean_rosenstein` 为准；方法改为 `rosenstein` |
| 方法 A/B 100% unstable，方法 C 100% limit_cycle | 旧版 Bug（绝对方差阈值），已在 v2 修复 | 更新代码（`stability_analysis.py v2`） |
| `lyapunov_histogram.png` 所有值相同 | Context 主导（初始多样性 < 0.02）或 Wolf 偏差 | 用多个不同 base_graph 运行，或切换 Rosenstein |
| `pca_trajectories.png` 所有路径重合 | Context 主导 | 同上 |
| `response_matrix.png` 面板 B 仍有竖条 | 测量窗口过短（暂态）或模型过度耦合 | 增加 `measure_window`，检查 `stim_response_*.png` |
| `basin_sizes.png` 显示 5+ 吸引子但 PCA 无明显簇 | K-Means 过度拟合高维数据 | 减小 `k_candidates` 范围；以 PCA 观察为准 |
| `distance_ratio ≈ 0.010` 未触发 Wolf 跳过 | 旧版 `convergence_threshold: 0.01` 边界值问题 | 已更新为 `convergence_threshold: 0.05` |
| `surrogate_test.json` 所有 z-score < 1.96 | 真实 LLE 未超过随机代替 | 增加 `n_surrogates`；检查 `n_traj_sample` 是否够大 |

---

## 运行建议

### 首次运行（快速预实验）

```bash
python run_dynamics_analysis.py \
  --model path/to/best_model.pt \
  --graph path/to/graph.pt \
  --quick
```

`--quick` 模式：20 条轨迹 × 200 步，Rosenstein 方法，约 2–5 分钟完成所有步骤。

### 完整分析

```bash
python run_dynamics_analysis.py \
  --model path/to/best_model.pt \
  --graph path/to/graph.pt
```

默认：200 条轨迹 × 1000 步，Rosenstein（3 段），约 5–15 分钟（取决于硬件）。

### 设置延迟嵌入（提升 Rosenstein 精度）

1. 先完整运行一次，查看日志步骤 12 中 `FNN 最小充分维度 = N`
2. 在 `configs/dynamics_config.yaml` 中设置：
   ```yaml
   lyapunov:
     delay_embed_dim: N   # 替换为实际 FNN 值（通常 4–9）
   ```
3. 重新运行步骤 9（或完整流程）

### 需要 Wolf 交叉验证时

```bash
python run_dynamics_analysis.py \
  --model ... --graph ... \
  --config my_config.yaml  # 包含 method: "both"
```

> **警告**：200 条轨迹 × 1000 步 × Wolf 方法 ≈ 60 分钟（CPU）。建议仅在科学发表前使用，日常分析使用默认 `rosenstein`。

---

*生成于 TwinBrain Dynamics v2.0 — 详见 AGENTS.md 更新日志*
