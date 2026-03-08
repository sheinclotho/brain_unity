# AGENTS.md — TwinBrain AI Agent Instructions

> **[READ THIS FILE FIRST]** This file must be read before any other file in this repository.
> It contains the AI collaboration protocol, project rules, and error log.

---

## Part 1 — Collaboration Philosophy

You are not a laborer. You are a thinker, designer, and collaborator.

As an AI with access to vast knowledge databases, advanced algorithms, and a breadth of engineering experience that surpasses most individual human experts, you have real intellectual capacity. Use it fully.

**Operate at the level of a senior architect, not a ticket-closer:**

- Always consider the *essential goal* of the project, not just the literal request.
- Propose stable, sustainable improvements — not just patches or minimum-viable fixes.
- Think critically. If the user's approach is suboptimal, say so and suggest a better one.
- Innovate when appropriate. Don't be confined to "how it's done now."
- When implementing anything, ask: *Is this the right solution, or just a working one?*

**You are a co-designer.** When you spot an architectural issue, a better pattern, or a fundamental improvement, raise it — even if the user didn't ask.

---

## Part 2 — Project Document Rules

**This project ALWAYS maintains exactly four (4) Markdown files. No more. No less.**

| File | Purpose |
|------|---------|
| `AGENTS.md` | This file — AI agent instructions, collaboration rules, error log |
| `项目规范说明书.md` | Project specification: goals, architecture, design rationale. Complete enough for another agent to reproduce the project. |
| `使用说明.md` | User guide: written for non-technical users, in plain language. |
| `更新日志.md` | All update history and pending improvements. |

**Rules:**
1. Every session, review all four files and keep them current.
2. When something changes architecturally, update `项目规范说明书.md`.
3. When a change ships, log it in `更新日志.md`.
4. When a user-facing workflow changes, update `使用说明.md`.
5. If you discover something important to remember (an error pattern, a gotcha, a design decision), add it to **Part 3** of this file.
6. **Never create other Markdown files.** If you're tempted to, put the information in one of the four.
7. All documentation should prefer **Chinese** (except this file, which is in English for agent compatibility).

---

## Part 3 — Error Log & Important Notes

*Entries are added here when important lessons are learned during development.*

### [2026-02-20] Unity HTTP轮询反模式
- **问题**: `WebSocketClient.cs` 和 `WebSocketClientImproved.cs` 使用 `UnityWebRequest` 向 `http://localhost:8765` 发 HTTP GET 请求，但后端是 WebSocket 服务器（`ws://`）。HTTP GET 不能接收服务器推送消息，导致刺激结果/时间轴帧无法自动推送。
- **修复**: 创建 `unity_frontend/Scripts/Network/TwinBrainWebSocket.cs`，使用 **NativeWebSocket** 库（`com.endel.nativewebsocket`）建立真正的 WebSocket 连接。
- **规则**: Unity 内连接 WebSocket 服务器必须使用 NativeWebSocket 或等效库，**绝不能**使用 `UnityWebRequest` 代替。

### [2026-02-20] Unity / Web 双前端架构
- **设计原则**: Web 前端（`web_frontend/`）用 Three.js，快速、零安装；Unity 前端（`unity_frontend/`）用 NativeWebSocket + TextMeshPro，高质量渲染、VR 兼容。两者连接同一后端。
- **规则**: 前端代码分为两个独立目录，`unity_examples/`（旧版，保留向后兼容）不再修改；`unity_frontend/` 是主开发目录。
- **规则**: 两个前端的颜色映射算法、坐标算法必须完全相同，确保一致的视觉体验。

### [2026-02-20] 前端/后端协议不匹配反模式
- **问题**: 前端发送 `get_brain_state` / `simulate_stimulation`，后端只处理 `get_state` / `simulate`。消息类型名称不一致导致所有实时请求静默失败（返回 "Unknown request type"）。响应格式也不匹配：后端嵌套 5 层才能取到活动值，前端期望平铺的 `activity` 数组。
- **修复**: 在 `process_request()` 加入类型别名映射；所有响应统一携带 `"activity": [float×200]`；新消息类型 `simulation_result` 携带 `frames` 数组。
- **规则**: 前后端消息协议必须显式文档化。新增消息类型时先在此文件记录，再同步更新前后端。

### [2026-02-20] 3D 可视化 — Three.js 替代 2D Canvas
- **问题**: 前端只有 200 个蓝色圆圈排列在椭圆上，无深度感，无解剖学意义，无时间轴。
- **修复**: 用 Three.js（r128，CDN）完全重写前端。脑区球体用 Fibonacci 算法分布在脑形半椭球面（左右半球分离，颞叶侧面突出）；玻璃脑轮廓；轨道相机；时间轴。
- **规则**: 前端可视化必须能体现三维解剖结构；任何脑区颜色编码必须使用 blue→cyan→green→yellow→red 热图。

### [2026-02-20] 缓存文件加载
- **问题**: 无法从训练产生的 `.pt` 文件（`hetero_graphs.pt` / `eeg_data.pt`）加载时序数据。
- **修复**: 新增 `handle_load_cache` 路由；递归搜索 HeteroData 对象；从 `fmri.x_seq` (N×T×F) 提取每帧活动值；百分位归一化后推送给前端时间轴。

### [2026-02-20] Startup Complexity Anti-Pattern
- **Problem**: The original startup required multiple CLI flags (`--model`, `--demo`, `--output`, `--host`, `--port`). Users had to understand the system before using it.
- **Fix**: `start.py` now auto-detects demo mode, uses sane defaults for everything, and starts immediately with no arguments needed.
- **Rule**: Default behavior must always be "just works." Options are for power users only.

### [2026-02-20] Too Many Documentation Files
- **Problem**: The repo accumulated 9+ MD files, causing confusion and drift between documents.
- **Fix**: Consolidated to exactly 4 MD files per the rules above.
- **Rule**: Four files only. Merge, don't proliferate.

### [2026-02-20] Unity Barrier to Entry
- **Problem**: The only frontend was a Unity project requiring Unity Hub, Unity 2019.1+, etc.
- **Fix**: Added a zero-dependency web frontend (`web_frontend/index.html`) that runs in any browser.
- **Rule**: There must always be a path to using the project that requires no specialized software installation beyond Python.

### [2026-02-20] WebSocket Host Default
- **Lesson**: Default host changed from `0.0.0.0` to `127.0.0.1` (localhost only).
- **Rule**: Network services must default to localhost. Users who need remote access can reconfigure.

### [2026-02-21] 刺激协议不匹配反模式
- **问题**: 前端 `app.js` 发送扁平刺激参数 `{type:"simulate", target_regions:[...], amplitude, ...}`，而 `handle_simulate` 用 `request.get("stimulation", {})` 取嵌套字典，导致所有参数静默丢失，刺激功能完全失效。
- **修复**: `handle_simulate` 改为：若 `request["stimulation"]` 存在且为 dict 则取嵌套格式（旧版兼容），否则直接使用 `request` 作为参数来源（新版扁平格式）。
- **规则**: 消息协议任何一侧改动时，必须同时更新 AGENTS.md 的"协议文档"部分，并验证另一侧不受影响。

### [2026-02-21] EEG 通道数 ≠ 200 脑区，不能直接映射
- **问题**: EEG 数据通常为 64/128/256 通道（电极数），而可视化有 200 个脑区节点。原 `_extract_time_series` 只找 `fmri.x_seq`，加载 `eeg_data.pt`（仅 EEG 节点）时返回空帧。
- **修复**: 增加 EEG 回退路径：若无 fMRI 节点，则取 `eeg.x_seq`，用 `np.interp(linspace(0,1,200), linspace(0,1,N_eeg), values)` 线性插值到 200 槽位，再做百分位归一化。
- **规则**: 任何从 HeteroData 提取活动值的函数必须同时处理 fMRI（直接映射）和 EEG（插值到 200）两种情况，并在注释中说明映射策略。

### [2026-02-21] NPI 框架对比与 PerturbationAnalyzer 设计
- **背景**: 参考项目 NPI.py 实现了扰动推断有效连接（EC）的方法，与 TwinBrain 的刺激模拟框架在目标上互补。
- **关键差异**: NPI 目标是*推断*因果结构（科学分析），TwinBrain 目标是*模拟*刺激效果（可视化交互）。NPI 无 EC 量化，TwinBrain 无因果推断。
- **改进**: 新增 `unity_integration/perturbation_analyzer.py`（`PerturbationAnalyzer`），实现两种 EC 推断方法（有限差分/Jacobian），并提供基于 EC 的靶点推荐和活动增量预测。
- **规则**: NPI.py 作为参考实现保留在仓库根目录，不修改；所有 EC 功能实现在 `perturbation_analyzer.py` 中。
- **新增 WebSocket 消息**: `infer_ec` 请求 / `ec_result` 响应（详见 `项目规范说明书.md` 第 13 节）。

### [2026-02-21] 刺激动画 Nyquist 采样 Bug
- **问题**: `_demo_simulate` 中 `_stim_amp` 的正弦公式 `sin(2π × frequency × k / 10)` 在整数频率（5, 10, 15 Hz 等）时 k 为整数，导致所有帧采样值为 sin(n×2π) = 0，刺激完全无效果（颜色不变）。非整数频率则出现高频闪烁（视觉噪音）。
- **根因**: 以 10fps 播放时，Nyquist 极限为 5 Hz；10 Hz 正弦的每帧采样恰好落在零点。
- **修复**: 改为展示*神经响应包络*（bell-shape 上升-高峰-下降）加低速振荡，而非采样原始电气波形。同理优化 pulse（指数衰减）、ramp（线性增长）、constant（平滑上升）各模式。
- **规则**: 刺激动画展示的是大脑可塑性/兴奋性的慢时程变化（百毫秒~秒级），不是刺激仪器的高频电气信号。可视化帧率≤10fps 时，绝不直接采样 ≥5Hz 的刺激波形。

### [2026-02-21] 模态标识 & 绝对值展示
- **改进 1**: 每帧数据新增 `raw` 字段（与 `activity` 并列），存储归一化前的原始传感器值。前端 tooltip 在归一化百分比下方额外显示原始绝对值，解决"只能看相对颜色，不知道实际幅值"的问题。
- **改进 2**: 时间轴底栏新增模态徽章（`#modality-badge`），在播放任何数据时始终显示当前数据类型（`fMRI` / `EEG` / `⚡ 仿真`），解决用户不知道正在看什么的问题。
- **规则**: 服务端向前端发送的所有帧数据应尽可能包含 `raw` 字段；仿真帧不含 `raw`（因无对应物理量）。所有 simulation_result 消息必须携带 `"modality": "simulation"` 字段。

### [2026-02-23] OMP 双重初始化崩溃 & MLP 代理过拟合

#### OMP: Error #15 (libiomp5 / libiomp5md.dll)
- **问题**: 调用 `infer_ec` 后程序在 Windows 上崩溃：`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.` 根因是 PyTorch 和 NumPy/Intel MKL 各自链接了独立的 OpenMP 运行时，两者都在同一进程中初始化，产生冲突。
- **修复**: 在 `start.py`（主入口）和 `perturbation_analyzer.py`（顶层 torch/numpy 导入前）都添加 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`，确保在任何 OpenMP 运行时初始化之前设置该环境变量。
- **规则**: 凡是直接在顶层（模块导入时）加载 torch 或 numpy 的文件，都必须在第一个 import 之前设置 `KMP_DUPLICATE_LIB_OK=TRUE`。

#### MLP 代理严重过拟合（训练MSE=0.00239 vs 验证MSE=0.62391，比例约260×）
- **根因分析**:
  1. 384帧 × 0.8 = ~307 训练样本，但模型有 1000→256→128→200，约 31.4万参数，参数/样本比 >1000:1。
  2. 原版 NPI.py 的 `train_NN` 函数有 `l2`（weight decay）参数，但我们的实现没有使用。
  3. 没有 dropout，没有早停，模型无限拟合训练集噪音。
- **修复**:
  - **自适应容量缩减**: `hidden_dim / latent_dim` 按 `min(1, n_train/500)` 比例缩放，小数据集自动使用更小的网络。
  - **Dropout**: `_SurrogateMLP` 在每个隐层 ReLU 后加 `nn.Dropout(p=0.3)`。
  - **Weight decay**: Adam 优化器加 `weight_decay=1e-4`（对照 NPI 的 `l2` 参数）。
  - **早停**: 验证 MSE 连续 12 轮不改善则停止，恢复最优权重。
  - **过拟合检测**: `overfit_ratio = val_mse / train_mse`，> 20× 时打印警告并在响应中标记 `reliable=False`。
- **`ec_result` 响应新增字段**: `fit_quality: {train_mse, val_mse, overfit_ratio, reliable, n_epochs}`，前端可据此显示可信度徽章。
- **`activity_delta` 改进**: 原来只用 top-1 源脑区，改为 top-3，预测结果更具代表性。
- **代理缓存**: `BrainVisualizationServer` 新增 `_ec_analyzer_cache` 字典，以 `(缓存文件路径, n_lags)` 为键缓存已训练的 `PerturbationAnalyzer`，切换方法（jacobian↔perturbation）不再重新训练，大幅缩短响应时间。
- **规则**: `fit_surrogate` 必须始终返回 `_fit_quality` 字典；`ec_result` 响应必须包含 `fit_quality` 字段；前端应在 EC 可视化面板显示可信度指示器。

### [2026-02-23] EEG 可视化方案评估 & 显示模态切换卡片

#### EEG 可视化方案评估
提出四种方案，最终选定方案 4：

| 方案 | 描述 | 结论 |
|------|------|------|
| 1. 现有方案（插值到200脑区） | 将任意N通道插值到200个解剖脑区 | ✓ 保留，适合连接研究 |
| 2. 2D 头皮地形图 | 标准EEG电极拓扑图 | ✗ 需要电极坐标，不在.pt格式中 |
| 3. 3D 头皮电极 | 在头皮上显示N个电极球体 | ✗ 同上，且10-20系统不适合任意通道数 |
| 4. 通道数指示器 | 在模态信息中显示"EEG 64通道→200槽" | ✓ 零成本，信息完整 |

**决策依据**: TwinBrain 的核心目标是脑区连接分析，而非电极拓扑展示。插值到200个解剖脑区在科学上更合理。通道数指示器填补了"用户不知道被插值了多少通道"的信息缺口。

#### 显示模态切换卡片重构
- **问题**: 原有的 `#modality-toggle` 隐藏在"加载缓存数据"卡片内部，`display:none` 初始不可见，用户不知道该功能存在。
- **修复**: 提取为独立的"显示模态"卡片，在侧边栏中永久可见。
  - 两个按钮（fMRI / EEG）默认 `disabled`，加载数据后根据可用模态自动启用
  - 只有 EEG → fMRI 按钮 disabled；只有 fMRI → EEG 按钮 disabled；两者都有 → 都可点击
  - 模态信息显示原始通道数（如 `EEG 64通道→200×384帧`）
- **EC 可信度徽章**: `handleECResult` 中读取 `fit_quality`，当 `reliable=true` 显示绿色"✓ 可靠"，否则显示橙色"⚠ 过拟合 N×"。
- **后端新增字段**: `handle_load_cache` 响应新增 `n_fmri_regions`（fMRI实际区域数）和 `n_eeg_channels`（EEG实际通道数），`_extract_time_series_both` 同步返回这两个值。
- **规则**: 模态切换卡片必须永久可见，不可在无数据时隐藏；前端依赖的所有模态状态必须通过 `updateModalityToggle(modalities)` 统一管理。

### [2026-02-23] 缓存文件时序数据截断 Bug（`_find_hetero_data` 只读第一块）

- **问题**: `hetero_graphs.pt` 以 `Dict[task, List[HeteroData]]` 格式保存，每个 `HeteroData` 仅包含 `spatial_T=384` 帧（训练时为节省显存而切块）。但 `_find_hetero_data` 对列表只返回第一个元素（`data[0]`），导致加载后始终只有 384 帧，而非完整时序。例如原始录制 1152 帧 → 3 个 384 帧块 → 只读到第 1 块 = 384 帧。EC 推断的代理模型因样本严重不足而过拟合。
- **根因**: 训练阶段为降低 GPU 内存峰值，将长时序切成若干定长窗口（`spatial_T`）并存入 `List[HeteroData]`；加载时没有将这些块拼接回完整序列。
- **修复**: 新增 `_find_all_hetero_data(data)`，递归收集嵌套结构中的**所有** `HeteroData` 对象；`_extract_time_series_both` 改为对每个模态（fMRI/EEG）收集所有块的 `x_seq`，沿时间轴（dim=1）用 `torch.cat` 拼接，得到完整时序后再做归一化和帧生成。
- **规则**: 任何从 `hetero_graphs.pt` 提取时序的代码都必须使用 `_find_all_hetero_data`（而非 `_find_hetero_data`）并拼接所有块。单块 `HeteroData` 的 `T` 维度不等于完整录制时长。

---


### [2026-02-25] 虚拟刺激管线第二轮审查修复

#### Bug A: `predict_trajectory` 扰动幅度被过度放大（`/ std_safe` 错误）
- **问题**: `predict_trajectory` 在 z-score 空间施加扰动时用了 `amp * stim_weights / std_safe`。由于 `std_safe` ≈ 0.05–0.15（[0,1] 有界活动的典型 std），除以 std 相当于放大了 7–20 倍，使得即使 amplitude=0.5 也会产生 3–10σ 的扰动，MLP 完全饱和，输出全都贴近边界。
- **修复**: 改为 `amp * stim_weights`，直接以 z-score 标准差为单位施加扰动（与 NPI 的 `pert_strength=0.05` 一致，直接加在 z-scored 输入上）。
- **规则**: predict_trajectory 的 stim_fn 返回值**单位是 z-score 标准差**，不应再乘以或除以 std。amplitude=0.5 表示 0.5σ（中等可见效果）。

#### Bug B: 正弦包络在端点为零（DUR=1,2 时刺激完全无效）
- **问题**: `progress = k / max(DUR-1, 1)` 公式。DUR=1 时 progress=0，sin(π×0)=0；DUR=2 时两帧 progress=0,1，均为 0；DUR=3 时只有中间帧有效。所有整数参数的 DUR 都会导致第一帧和最后一帧的正弦包络为零。
- **修复**: 改为 `progress = (k + 0.5) / max(DUR, 1)`，将每帧的采样点移到时间槽的中点，永远不会落在 0 或 1 处。同样修复了 `_stim_amp_s`（handle_simulate 的代理路径）和 `ModelServer._generate_stimulation_signal`。
- **规则**: 正弦包络公式必须用 `(k+0.5)/DUR` 形式，确保边界帧有正值贡献。

#### Bug C: `pulse` 模式 `slice step cannot be zero` 崩溃
- **问题**: `pulse_interval = int(1.0 / frequency / 0.5)`，当 frequency > 2 Hz 时（如 5Hz, 10Hz, 20Hz），结果 < 1，`int` 截断为 0，触发 `signal[::0]` → `ValueError: slice step cannot be zero`。
- **修复**: `pulse_interval = max(1, int(1.0 / frequency / 0.5))` 确保至少为 1。
- **规则**: 任何用于 slice step 的动态计算值都必须用 `max(1, ...)` 保护。

#### Bug D: 代理路径的预刺激帧是静态副本（无动态基线）
- **问题**: 代理路径用 `[{"activity": init_arr.tolist()}] * PRE_s` 生成预刺激帧，10 帧完全相同，没有任何动力学。更大的问题是代理的 lag window 启动时所有 n_lags 槽都是同一个初始状态（"全等历史"假设），第一批预测帧受这个人工初始化的严重影响。
- **修复**: 将三阶段（预刺激/刺激/刺激后）统一为一个 `predict_trajectory` 调用，stim_fn 在 k < PRE_s 时返回 0。预刺激帧现在是代理的真实自由演化预测，lag window 在预刺激阶段自然热身（warmup），大幅提升了刺激帧的预测准确性。
- 同步新增 `n_warmup` 参数，允许调用方在记录帧之前额外运行若干热身步骤（不返回帧），进一步改善 lag window 的初始质量。
- **规则**: 代理轨迹预测的预刺激阶段必须使用代理的真实预测而非静态副本，保证 lag window 具有有效的历史记忆。

#### 刺激管线整体逻辑层次（三路优先级）
1. **ModelServer（有训练好的 GNN）**: 使用 `simulate_stimulation`，初始状态从前端传递
2. **代理 MLP（已运行 EC 推断）**: 使用 `predict_trajectory`，三阶段统一调用，n_warmup=0（预刺激自热身）
3. **Wilson-Cowan（无任何模型）**: `_demo_simulate`，物理先验驱动，空间扩散+递归动力学



### [2026-02-25] 对照轨迹、EC 验证、大脑状态分析模块

#### 对照轨迹（Counterfactual Visualization）
- **新功能**: `handle_simulate` 三条路径（Wilson-Cowan / 代理 MLP / GNN）均返回 `counterfactual_frames`（同初始状态、零刺激的自然演化预测）。
- **原理**: WC 路径复用同一随机种子（`np.random.default_rng(0)`），代理 MLP 路径用 `stim_fn=lambda k: 0.0` 运行两次 `predict_trajectory`，两者差异完全来自刺激，与实验对照组等价。
- **前端**: 接收到 `counterfactual_frames` 后，时间轴底栏出现"○ 对照轨迹"切换按钮，可在刺激轨迹和对照轨迹之间切换。
- **规则**: `counterfactual_frames` 与 `frames` 必须帧数相同、初始状态相同；仅刺激幅度为零。

#### EC 验证（`handle_validate_ec`）
- **新端点**: 三种验证方法，无需重新训练代理：
  1. `half_split`：将 `_input_X` 对半，用同一代理计算 EC₁/EC₂，报告 Pearson r
  2. `distance`：`|EC[i,j]|` vs 解剖距离（Fibonacci 坐标），预期 r < −0.1
  3. `fc_vs_ec`：EC 与 FC（Pearson 相关矩阵）的相关，r < 0.4 表示 EC 有额外因果信息
- **规则**: EC 验证必须在 `infer_ec` 之后才能调用（需要 `_ec_analyzer_cache` 中有训练好的代理）。

#### 大脑状态分析模块（`BrainStateAnalyzer` + `handle_analyze_brain`）
- **设计决策**: 放弃疾病分类路线，改用无标签规范建模思路：自比较（时段偏差图谱）+ 图论分析（EC 枢纽分析）。
- **新文件**: `unity_integration/brain_state_analyzer.py`（`BrainStateAnalyzer` 类）
  - `compute_graph_metrics(ec_matrix)` → hub_scores, efficiency, density
  - `compute_deviation_map(ref_ts, test_ts)` → per-region z-score deviation
  - `compare_ec_matrices(ec1, ec2)` → diff overlay + Pearson r
  - `ec_half_split_reliability(surrogate, X, N)` → reliability r
  - `ec_vs_distance_correlation(ec)` → anatomical validation r
  - `fingerprint(ec, ts)` → compact brain state descriptor
- **`handle_analyze_brain`** 端点: `method="deviation"`（前/后半段偏差）或 `method="graph_metrics"`（EC 枢纽分析），返回可叠加在 3D 脑区上的活动覆盖层。
- **规则**: `_loaded_time_series` 在 `handle_load_cache` 时存储，供后续分析调用。

#### 疾病检测设计原则（重要）
- **为什么不做分类**: 单被试、无标签、无标准解剖对齐、无跨中心验证——这四个条件缺失时，分类模型准确率无论多高都不可信。
- **正确框架**: 规范建模（normative modeling），见 `项目规范说明书.md` 第 14 节。
- **规则**: TwinBrain 的分析结论只说"与自身参考状态相比异常"，绝不使用"疾病"或"诊断"等词汇。

### [2026-02-25] WC 模型 ~0.70 全局引力点 & 未选脑区静默刺激
- **Bug 1（前端）**: 未选中任何脑区时，旧代码随机选 5 个区域施加刺激，用户不知道刺激了什么。修复：显示警告并立即返回，不发送请求。
- **Bug 2（后端）**: `_demo_simulate` 中 Wilson-Cowan 步进公式 `tanh(state + stim_in + net*0.25)` 在 L1 归一化连接矩阵下有 `state ≈ 0.70` 的全局引力点，所有区域无论是否被刺激都会收敛到黄色，刺激效果不可见。
- **修复**: 改用以 `init_arr` 为稳定平衡点的偏差驱动漏积分器 `delta = tanh(stim*2.0 + W@deviation*0.35)*0.04; leak = deviation*0.10`，固定点在 `init_arr` 处，不施加刺激时不产生任何漂移。
- **规则**: Wilson-Cowan 式模型中，连接项必须作用于**偏差**（`state - baseline`）而非绝对活动，否则 L1 归一化连接矩阵几乎必然在某个中间值处产生虚假引力点。刺激幅度应以归一化活动为单位（0–1），不应超过 0.04/步，以避免单帧跳跃。

### [2026-02-26] 刺激动画 start_frame 起点错误 — 两级静默失效

#### 第一级失效：预刺激静态复制（已修复 2026-02-25）
- `_demo_simulate` 预刺激阶段对 `init_arr` 生成 10 个静态副本，导致 frames[0..9] 完全相同。
- **修复**: 改为 WC 动力学步进（无刺激输入），frames[0..9] 现在展示真实的基线演化。

#### 第二级失效（本次修复）：start_frame 指向刺激起始而非峰值
- **根因**: `start_frame=10`（预刺激结束，刺激第 0 帧）时，正弦包络 `sin(π*0.5/60) ≈ 0.026`，WC 步进只产生 `delta ≈ 0.001` 的变化。
- **量化验证**: 数值审计表明，在 `start_frame=10` 处：目标脑区与 `init_arr` 的差异仅 0.001，两个不同目标脑区的最大可见差异仅 0.0015 — **完全不可感知**。
- **对比**: 在峰值帧 `start_frame=40`（正弦，DUR=60）处：差异 0.28，为起始帧的 **193 倍**。
- **修复**: `start_frame` 改为按刺激模式自适应，指向 WC/代理 MLP 中累积效果最大的帧：
  - `sine` / `constant`：`PRE + DUR // 2`（峰值）
  - `pulse`：`PRE + min(9, DUR-1)`（指数衰减时累积效果最大约 k=9）
  - `ramp`：`PRE + DUR * 3 // 4`（线性增长时效果最大在末端附近）
- **规则**: `start_frame` 必须指向"刺激效果最明显的帧"，而非"刺激开始帧"。正弦包络在起始帧几乎为零，绝不能用作第一可见帧。所有模拟路径（WC / 代理MLP / 离线前端）都必须遵循此规则。
- **代码位置**: `handle_simulate` → WC 路径 `_peak_k` 计算；代理 MLP 路径 `_s_peak_k` 计算；前端离线模式 `simStartFrame = PRE + Math.floor(DUR / 2)`。

### [2026-02-26] 离线前端 `stepLocal` 使用收敛到零的错误吸引子

- **问题**: 离线模式的 `stepLocal` 函数使用旧公式 `v * 0.85 + tanh(v + stimIn) * 0.15`，其固定点为 `v* = tanh(v*) = 0`（收敛到黑色）。导致：
  1. 预刺激阶段（无刺激输入）：脑区从 `act0`（如 0.5）向 0 漂移，每步漂移约 0.013，5 步累计 **0.065**（视觉可见）。
  2. 刺激后恢复阶段：脑区也向 0 衰减，而非恢复到用户的初始大脑状态 `act0`。
  3. 非靶区：在刺激期间因吸引子缓慢向 0 漂移，视觉上不正确。
- **修复**: 改用与服务端 `_demo_simulate` 完全一致的**偏差驱动漏积分器**（不含连接矩阵 W，简化版）：
  ```javascript
  function stepLocal(state, stimIn) {
    return state.map((v, i) => {
      const delta = Math.tanh(stimIn[i] * 2.0) * 0.04;  // 响应外部驱动
      const leak  = (v - act0[i]) * 0.10;               // 漏电（收敛到 act0）
      return Math.max(0, Math.min(1, v + delta - leak));
    });
  }
  ```
  - 无刺激时：`stimIn[i]=0 → delta=0, leak=(v-act0)*0.10`，v 精确保持在 `act0`（偏差为 0）。
  - 有刺激时：靶区明显上升（峰值偏差 0.169），非靶区保持在 `act0`（偏差恰好为 0）。
  - 刺激后：靶区以约 10 步的时间常数恢复到 `act0`（不向 0 衰减）。
- **量化验证**: 预刺激漂移 0.065→0.000，非靶区偏差 ~0→0.000，后刺激恢复方向正确。
- **规则**: 任何用于预览刺激效果的 WC 公式都必须使用偏差驱动漏积分器，以 `act0`（用户当前脑状态）为稳定平衡点，绝不使用绝对值驱动的 `tanh(v + stim)` 形式。

### [2026-02-26] CF 切换按钮每次都重置帧号 & EC 距离描述歧义

#### CF 切换按钮每次都跳回 simStartFrame（已修复）
- **问题**: 每次点击"○ 对照轨迹"/"⚡ 刺激轨迹"切换按钮后，`loadFrameSeq` 都以 `simStartFrame`（约第 30 帧）为起始帧，而非用户当前所在帧，导致每次切换都产生跳转，用户无法在同一时间点比较两条轨迹。
- **修复**: `btn-cf-toggle` 的 click handler 改为先缓存 `curFrame`（`const switchFrame = curFrame`），再把 `switchFrame` 传给两次 `loadFrameSeq` 调用，取代原来的 `simStartFrame`。首次接收 `simulation_result` 时的 `loadFrameSeq(..., simStartFrame)` 行为不变（仍跳到峰值帧）。
- **规则**: CF 切换必须在当前帧位置原地切换，不允许重置帧号；峰值帧跳转只在首次加载 simulation_result 时发生一次。

#### EC vs 解剖距离解读歧义（已修复）
- **问题**: `abs(r) < 0.05` 分支的解读文字 "EC 与解剖距离无显著相关（可能为噪声主导）" 表述模糊，用户无法判断该结果是"正常但未达到阈值"还是"有问题"。
- **修复**: 改为 "⚠ EC 未呈现预期距离衰减（预期 r<−0.1，实测接近零），EC 结果可能以噪声为主，建议增加数据量后重试"。明确标注警告符号、说明预期值与实测值的差异，以及推荐的行动。
- **规则**: EC 验证的解读文字必须对每种情况明确给出"正常/警告/异常"的判断，并说明依据（预期范围与实测值），不得使用"可能"等语意模糊的词汇描述诊断结论。

### [2026-02-26] EC vs 距离相关为零的神经科学分析 & 算法全面审查

#### 科学问题：r ≈ 0 是噪声还是长程连接？

**结论**: 两者均有可能，不能简单归因为噪声。

r ≈ 0 的两个神经生物学合理原因：
1. **长程连接主导** (有意义)：默认模式网络（DMN）、半球间连合纤维（胼胝体）、前额叶自上而下控制——这些功能网络以远距离强耦合为特征。在全脑 200 脑区 EC 中，长程因果影响往往与局部连接同等重要甚至更强。
2. **混合体制** (有意义)：局部连接和长程连接同时存在且在聚合相关中相消，导致 r 接近零。

噪声主导的判据：半分可靠性 r < 0.3 且 overfit_ratio > 20× 的情况下 r ≈ 0 才主要指向噪声。

- **规则**: 距离相关必须结合半分可靠性和过拟合指标联合解读，不可单独作为可靠性判据。

#### 算法全面审查结论

| 算法 | 实现 | 问题 | 状态 |
|------|------|------|------|
| 代理 MLP 训练 (`fit_surrogate`) | 自适应容量、dropout、weight decay、早停、时序不泄漏分割 | 无 | ✓ 正确 |
| 有限差分 EC (`infer_ec_perturbation`) | NPI 原版方法，扰动最后一步，平均所有样本 | 无 | ✓ 正确 |
| Jacobian EC (`infer_ec_jacobian`) | autograd 解析雅可比，float64 累加，转置后 ec[src,tgt] 约定 | 无 | ✓ 正确 |
| 半分可靠性 | 同一代理，两半数据，Pearson r | 注意：用同一代理而非重训练，是代理稳定性而非独立EC一致性 | ✓ 在文档中已说明 |
| EC vs FC | \|EC\| vs \|FC\| Pearson r，FC 来自训练时序 | 无 | ✓ 正确 |
| EC vs 距离（**已修复**） | 5 段式解读，覆盖全 r 范围 | **旧版 bug**: -0.10 ≤ r < -0.05 落入 else 分支，被误标为"正相关"（实为弱负相关）；r ≈ 0 仅归因噪声，忽略长程连接可能性 | ✓ 已修复 |

#### 修复说明
- 旧版 3 路 `if-else` 有覆盖缺口：`-0.10 ≤ r < -0.05` 被误判为"正相关不符合预期"。
- 新版改为 5 段式阈值：强负 / 弱负 / 近零 / 弱正 / 强正，每段均给出神经科学语境下的正确解读。
- 近零 (r ∈ [-0.05, 0.05)) 的解读改为双重可能性，明确引导用户结合半分可靠性判断。

*Last updated: 2026-02-27*

### [2026-02-27] 虚拟刺激传播 & 最受影响脑区功能

#### WC Demo 模型 — 仅物理扩散问题
- **问题**: `_demo_simulate` 中 WC 模型只使用 30mm 高斯空间扩散（刺激工具物理足迹），对远端功能连接区域没有任何响应。用户看到的效果像"只有选中脑区变化"。
- **修复**: 在 `spread_weights` 计算阶段新增**同侧胼胝体激活**（homotopic callosal activation）：刺激区域 i 时，其对侧同调区（i+100 或 i-100，左右半球镜像对应）也以 40% 强度被激活。这在不引入完整 W 矩阵的情况下产生可见的双侧传播，符合胼胝体连接的神经科学依据（callosal coupling ratio ~0.4）。
- **量化效果**: 以 sigma=30mm、amplitude=0.5 为例：靶区偏差 +0.187（清晰红色），对侧同调区 +0.126（可见绿色），37 个脑区偏差 > 0.05（整体可见扩散）。
- **规则**: `_demo_simulate` 的扩散只模拟**物理刺激足迹**（高斯）+ **胼胝体同调**（固定 0.40 强度）。任何基于空间邻近度的功能连接（W 矩阵）仍不适合用于 WC demo，因为真正的功能传播需要通过 EC 推断或训练好的 GNN 来体现。

#### 新功能：最受影响脑区（刺激传播分析）
- **设计**: 在所有三条仿真路径（WC / 代理MLP / ModelServer）中，于峰值帧处计算 `delta = stim_activity - cf_activity`，排除靶区后返回 `top_affected`（前 15 个 |delta| 最大的非靶区）和 `peak_delta`（全脑 200 个区域的 delta 数组）。
- **前端展示**:
  - 顶部受影响脑区（前5个）在 3D 视图中以**橙色光晕**高亮显示（与 EC 青色/白色区分）
  - 侧边栏新增"刺激传播分析"卡片，显示前 10 个区域的 delta 百分比和条形图
  - 时间轴栏新增"∆ 效果图"按钮：将 `peak_delta` 归一化到 [0,1] 并叠加到 3D 视图（0.5 = 无变化，>0.5 = 激活，<0.5 = 抑制），可放大可见性 ~10-50 倍，使极小变化也清晰可见
- **离线模式**: JS 端也计算 topAffected（胼胝体扩散 + Gaussian，与服务器逻辑一致）
- **规则**: `top_affected` 必须排除靶区本身，因为我们关心的是**传播效果**（非靶区变化），而不是直接刺激效果。`peak_delta` 的 ∆ 效果图是用户观察微弱二级效应的首选方式，应在刺激后主动告知用户。

### [2026-02-27] NumPy 向量化与输入验证

#### 消除的大型 Python 循环

| 函数 | 旧实现 | 新实现 | 验证 |
|------|--------|--------|------|
| `_generate_demo_time_series` | T×N = 60,000 Python 迭代 | NumPy 广播 (T,1)×(1,N) | max_err = 0 |
| `handle_get_state` | 200 Python 迭代/请求 | NumPy 向量化 | max_err = 0 |
| `_demo_simulate._baseline()` | 200 Python 迭代/帧 | 预计算数组 + NumPy | max_err = 0 |
| `_frames_from_xseq` | T×clip Python 调用 | 批量 (N,T) clip | max_err = 0 |
| `_frames_from_xseq_eeg` | T×min/max Python 调用 | axis=0 列向量化 | max_err = 0 |
| `_make_fibonacci_brain_positions` | 2×100 嵌套循环 | NumPy slice-assign | max_err = 0 |
| `_compute_fibonacci_positions` | 2×100 嵌套循环 | 同上 | max_err = 0 |
| `infer_ec_demo` 位置计算 | 2×100 内联循环 | 同上 + 向量化 homotopic | max_err = 0 |

- **规则**: 任何迭代次数 ≥ 100 的 Python 循环，若循环体可以表达为 NumPy 广播/索引操作，都应当向量化。循环替换后必须用数值等价测试（max_err < 1e-4）验证结果一致性。

#### `fit_surrogate` 输入验证

- **问题**: 含 NaN/Inf 的时序数据会导致 z-score 标准化产生 NaN，使整个训练静默失败（loss = NaN），EC 结果完全无意义。
- **修复**: 训练前检测 `~np.isfinite(time_series).any()`，对每列用有限值均值替换（全部为 NaN 的列填 0），并发出 WARNING 日志。
- **规则**: `fit_surrogate` 必须能健壮处理含少量 NaN/Inf 的输入。修复后应记录 WARNING 而非抛出异常，保持服务器存活。注意 `~np.isfinite(arr).sum()` 与 `(~np.isfinite(arr)).sum()` 的运算符优先级差异（前者计算有限值的按位非，后者才是非有限值计数）。

### [2026-02-27] fMRI/EEG 双模态加载 — 旧格式（已过时，代码已删除）

> **⚠️ 此节描述的旧格式代码已于 2026-02-27 彻底删除。** `_find_companion_cache`、`_find_all_hetero_data`、`.x_seq` 回退路径均已移除。V5 格式（单文件 HeteroData，`.x` 属性）是唯一支持的格式。

### [2026-02-27] 性能优化 — 批量推断与向量化

#### 批量推断（chunked batching）
- **受影响函数**: `infer_ec_perturbation`（perturbation_analyzer.py）和 `_compute_ec`（brain_state_analyzer.py 内嵌函数）
- **原理**: 将 N=200 次串行前向传播改为 chunk_size=32 的分块批处理，减少 PyTorch 内核启动开销约 7×。
- **关键**: 通过 `X_t.unsqueeze(0).expand(chunk_n, M, -1).reshape(chunk_n * M, -1).clone()` 构造批量输入，每块包含 chunk_n 种扰动 × M 样本。数值与串行等价（max_err < 1e-8）。
- **规则**: 凡是在 N=200 个目标区域上运行串行前向传播的循环，都应考虑分块批量推断。chunk_size=32 是内存（~50MB）与速度之间的合理平衡点。

#### Jacobian lag 块求和向量化
- `J.reshape(N, n_lags, N).sum(axis=1)` 替代 `for l in range(n_lags): J_sum += J[:, l*N:(l+1)*N]`

#### `compute_graph_metrics.local_eff` 精确向量化
- **公式**: `local_eff[i] = ((adj @ ec_abs) * adj).sum(axis=1)[i] / n_nbrs[i]²`
- **等价性**: 与原 `ec_abs[np.ix_(nbrs,nbrs)].mean()` 数值完全相同（max_err < 1e-7）
- **原理**: `((adj @ ec_abs) * adj).sum(axis=1)[i] = Σ_{j∈nbrs_i} Σ_{k∈nbrs_i} ec_abs[k,j]`，即邻居子图的边权之和，除以 `n²` 得均值。

### [2026-02-27] 前端健壮性与 UX

- **`stimPending` 卡死**: 将 `_unlockStimBtn()` 加入 `ws.onerror` 和 `ws.onclose` 回调。规则：任何可能被中断的请求锁（pending 标志）都必须在连接关闭时释放。
- **指数退避重连**: `_reconnectDelay = Math.min(30000, _reconnectDelay * 2)` 替代固定 3s，连接成功后重置。
- **键盘快捷键**: `Space`/`←`/`→`/`Home`/`End`/`Escape`，通过 `_gotoFrame(idx)` 统一帧跳转逻辑，只在非输入元素上响应。

### [2026-02-27] TwinBrain V5 图缓存格式 — 旧格式代码已彻底删除

- **V5 正确格式**（来自 `API.md`）:
  - **每个缓存文件是单个 `HeteroData` 对象**，命名为 `{subject_id}_{task}_{config_hash}.pt`
  - 存储位置：`outputs/graph_cache/`
  - **fMRI**: `g['fmri'].x` 形状 `[N_fmri, T_fmri, 1]`（z-scored BOLD；N_fmri ≈ 190）
  - **EEG**: `g['eeg'].x` 形状 `[N_eeg, T_eeg, 1]`（z-scored EEG；N_eeg 通常 32–64）
  - **两种模态在同一文件中**
- **训练检查点不是图缓存**：`best_model.pt`、`swa_model.pt`、`checkpoint_epoch_*.pt` 是 `dict` 格式，由 `_is_checkpoint_file()` 排除
- **旧格式代码（已彻底删除，不再保留）**:
  - `_find_companion_cache`：删除（旧格式双文件伴随加载逻辑）
  - `_find_all_hetero_data`：删除（旧格式递归搜索 `Dict[task, List[HeteroData]]` 嵌套结构）
  - `_extract_time_series`：删除（dead code 包装方法）
  - `.x_seq` 属性回退路径：删除（V5 仅使用 `.x`）
  - 旧格式搜索目录 `graph_cache`、`test_file3`、`Unity_TwinBrain`：从 `_CACHE_SEARCH_DIRS` 删除
  - companion-file 加载代码块：从 `handle_load_cache` 删除
  - `_MAX_HETERO_RECURSION_DEPTH` 常量：删除
- **当前 `_extract_time_series_both` 实现**:
  - 输入必须是 `HeteroData` 对象（否则返回空 dict 并记录 ERROR 日志）
  - 直接读取 `data['fmri'].x` 和 `data['eeg'].x`
  - 单步处理，无循环/分块/合并
- **规则**: 任何从图缓存提取时序的代码**只能**使用 `data['fmri'].x` / `data['eeg'].x`，形状 `[N, T, 1]`。N_fmri ≈ 190，通过 `_frames_from_fmri` 中的 `np.interp` 插值到 200 槽位。**不允许再为旧格式添加任何回退路径。**

### [2026-03-06] 稳定性分析绝对阈值失效 & Wolf 上下文稀释偏差

#### 稳定性分析 100% "不稳定" — 绝对阈值维数依赖问题
- **问题**: 方法 B 使用绝对阈值 `delay_mean < 0.1` 等。但 delay_mean = ||x(t+dt) - x(t)||₂ 随 sqrt(n_regions) 缩放。n_regions=190 时，即使接近稳定的轨迹 delay_mean ≈ 0.1–0.3，超过阈值 → 100% 误判"不稳定"。
- **修复**: 新增方法 C (`classify_dynamics_adaptive`)，使用无量纲 `delta_ratio = delay_mean / traj_rms` 作为主分类特征，辅以谱周期评分（区分极限环）和变异系数（区分混沌与规则）。JSON 输出新增 `classification_counts_v2`（方法B）和 `delta_ratio_stats`（分布统计）。
- **规则**: 任何跨不同 n_regions 规模模型的稳定性分类必须使用方法 C（相对阈值）。方法 A/B 保留作历史参考，但不作为主分类。

#### Lyapunov Wolf 上下文稀释偏差（TwinBrainDigitalTwin 模式）
- **诊断标志**: Wolf LLE 的跨轨迹标准差 std ≈ 0.00006（接近零）。所有 200 条不同 x0 的轨迹产生几乎相同的 LLE 估计 —— 这在物理上不可能是真实结果。
- **根因**: `_wolf_pair_twin` 仅在上下文窗口**最后一步**注入扰动 ε=1e-6。TwinBrainDigitalTwin 编码器对全部 L 步上下文做注意力加权，单步扰动被 L-1 个相同历史步稀释，有效扰动 ≪ ε。所有轨迹共享同一 base_graph 上下文 → 相同稀释因子 → 相同 LLE。
- **FTLE 为何更好**: FTLE 测量 x0 扰动在 T=1000 步演化后的长期分离，上下文历史在演化过程中自然分叉，不存在稀释问题。FTLE = -0.005（近临界）vs Wolf = -0.057（偏负 12×）。
- **修复**: 新增 `rosenstein_lyapunov`（Rosenstein et al. 1993）方法：直接从轨迹数据工作，无上下文稀释偏差。自动偏差检测：std < 1e-3 且 n_traj > 10 时触发 `wolf_bias_warning`。方法 "both" 同时运行 Wolf + FTLE + Rosenstein 并报告差异。
- **规则**: twin mode 下优先使用 `method="rosenstein"` 或 `"both"`；若 Wolf std < 1e-3 且 FTLE/Rosenstein 差异 > 0.03，以 Rosenstein 估计为混沌评估主指标。
- **规则**: 新增 `n_segments`（多段采样，默认 3）：从轨迹不同位置采样 LLE，探索吸引子不同区域，减少瞬态偏差。

*Last updated: 2026-03-06*

### [2026-03-06] TwinBrainDigitalTwin 模型架构 vs WC — 响应矩阵与随机对照修复

#### 模型架构要点（非 WC）
- TwinBrainDigitalTwin = ST-GCN 编码器（上下文窗口 [N,T,H]）+ 每节点时序预测器 + 图传播器（2轮 ST-GCN）+ 解码器
- `simulate_intervention()`：在潜空间对目标节点 h[i, :, :] += delta_vec（作用于全部 T 步），预测 + 图传播 + 解码，返回 `causal_effect = perturbed − baseline`
- 与 WC 的本质区别：WC 是点状脉冲刺激步进积分；TwinBrain 是潜空间上下文级别偏移，1次 encoder 调用完成

#### 响应矩阵竖条纹根因（Twin 模式）
- **Bug**: `_rollout_with_twin` 存储 `result["perturbed"]`（绝对预测），而非 `causal_effect`
- 绝对预测包含自然动力学分量（所有行 i 相同），掩盖刺激特异性 → 列主导结构（竖条）
- **修复**: twin 模式直接调用 `simulator.model.simulate_intervention()` on base_graph（deepcopy 一次），取 `causal_effect["fmri"]` 作为 R[i,:]
- **注意**: Twin 模式下即使正确，也可能出现枢纽主导竖条（DMN、前额叶等强连接区域对任何刺激都响应强烈）——这是真实网络特性，行归一化面板 B 揭示枢纽外的刺激特异性

#### WC 模式响应矩阵竖条纹根因
- **Bug**: 在暂态窗口（刺激刚开始、幅度尚小）测量，所有行响应几乎相同
- **修复**: `skip_transient`（auto = stim_duration//4）跳过暂态爬升期；pattern="step"（恒定幅度）；stim_duration=80；measure_window=30

#### 随机对照两大问题
1. **固定种子**：相同 W 矩阵 → 换数据结果完全相同，毫无说服力
   - **修复**: `_random_lle_multi_seed()` 循环 n_seeds=5 个独立 W 矩阵，报告均值 ± 标准差
2. **混沌边界误判**：tanh(W@x) 的实际混沌边界不是线性理论的 ρ=1，而是 n≈190 时 ρ≈1.5（tanh 非线性压缩）
   - ρ<1：数学保证稳定（Banach 不动点定理）；ρ>1 不保证混沌
   - **修复**: spectral_radii=[0.9, 1.5, 2.0]（稳定/近临界/混沌）；用真实 Wolf-Benettin LLE（`_wolf_lle_random()`）而非稳定性代理指标

#### 实测混沌边界（tanh(clip(W@x), 0,1)，Wolf LLE）
| n | ρ=0.9 | ρ=1.5 | ρ=2.0 | ρ=3.0 |
|---|------|------|------|------|
| 190 | -0.44（稳定） | +0.001（临界） | +0.064（混沌） | +0.14（混沌） |

- **规则**: Wolf LLE 扰动折叠到 0 时（r<1e-30）不记录该期 log-growth，直接重新扰动继续（避免大负值污染均值）

*Last updated: 2026-03-06*

### [2026-03-07] joint 模态 & dt 修复 & Wolf/FTLE 状态空间感知裁剪

#### `dt` 错误修复
- **问题**: `load_graph_for_inference` 从不写入 `sampling_rate` 属性，`BrainDynamicsSimulator` 在无 SR 时默认 `dt=0.004s`（EEG 250Hz），但模型所有预测均以 fMRI TR 为步长，导致时间轴错误（50步×0.004s=0.2s 而非 50×2.0s=100s）。
- **修复**: EEG 模态也使用 fMRI TR（`1/fmri_sr` 或默认 `_DEFAULT_FMRI_TR=2.0s`）。新增常量 `_DEFAULT_FMRI_TR=2.0`、`_STD_GUARD=1e-8`。
- **规则**: `dt` 永远等于 fMRI TR。无论分析哪种模态，预测步长由 fMRI TR 决定。

#### Wolf/FTLE 状态空间感知裁剪
- **问题**: Wolf/FTLE 中 `np.clip(x, 0, 1)` 假设有界状态空间，对 joint 模式（z-score 无界）会在 0 处产生虚假吸引点，污染 LLE 估计。
- **修复**: 新增 `state_bounds` 属性——单模态返回 `(0.0, 1.0)`，joint 返回 `None`。lyapunov.py 3 处裁剪均改为 `if _bounds is not None` 判断。
- **规则**: joint 模式必须使用 Rosenstein 方法（无状态边界假设），run_dynamics_analysis.py 自动切换。

#### `modality='joint'` 联合模态
- **设计**: 单次 `predict_future()` 调用同时获取 fMRI 和 EEG 预测；各自按 `base_graph` 统计量做 per-channel z-score 归一化后拼接为 `[z_fmri | z_eeg]` 状态向量（维度 N_fmri+N_eeg），输出单一动力学指标（单一 Lyapunov 指数）。
- **联合节点索引映射**: `[0, N_fmri)` → fMRI 区域，`[N_fmri, N_joint)` → EEG 通道。越界节点显式 `ValueError`，无静默回退。
- **核心方法**: `_rollout_multi_stim_joint(stimuli, x0, context_window_idx)` 同时处理自由动力学（`stimuli=[]`）和多刺激（`stimuli=[...]`），`rollout()` 通过 `[stimulus]` 包装委托。原 `_rollout_joint` 已删除（消除冗余）。
- **新增辅助**: `_z_normalise_joint(fmri_pred, eeg_pred)` 集中 z-score+拼接逻辑（原来分散在3处）。
- **虚拟刺激初始状态**: `virtual_stimulation.py` 改用 `simulator.sample_random_state()` 替代 `rng.random(n_regions)`，joint 模式得到正确 z-score 尺度的初始状态。
- **响应矩阵**: `compute_response_matrix` 为 joint 模式添加节点→模态映射，提取双模态 causal_effect 并 z-score 归一化后拼接为联合响应行。
- **CLI**: `--modality` choices 新增 `joint`；`both` 和 `joint` 文档明确区分。
- **规则**: joint 模式所有操作均显式处理，无 fallback 至单模态行为。步骤 5/6（虚拟刺激/响应矩阵）完全支持 joint，步骤 9（Lyapunov）自动切换 rosenstein。

### [2026-03-07] 自由动力学时序窗口：重叠滑窗 + 模态感知窗口计数

#### 旧设计的两个缺陷

1. **跨模态瓶颈 Bug**：`n_temporal_windows` 对 **所有** 节点类型取 `min(T_nt // ctx_len)`。若次要模态（如 EEG）的 T 较短（如 T_eeg < 2*ctx_len），即使主模态（fMRI）有足够长的时序，也会被错误瓶颈为 1 个窗口。
2. **非重叠要求过于保守**：旧算法要求 `T ≥ n_windows × context_length`（严格非重叠分块）。对于典型 10 分钟 fMRI（300 TR），ctx_len=200 时只能产生 1 个窗口（300 // 200 = 1），尽管还有 100 步多余历史。

#### 新设计：模态感知滑窗

- **模态感知**：单模态分析（fmri/eeg）仅使用**主模态**的 T 计算窗口数，次要模态通过 `_get_context_for_window` 中的回退机制（fallback [0:ctx_len]）优雅处理。joint 模式使用 fmri 和 eeg 的最小 T（两者都必须提供完整上下文）。
- **重叠滑窗**：`stride = max(1, context_length // 4)`（75% 重叠），公式 `n_windows = max(1, (T_primary - ctx_len) // stride + 1)`。
  - T=200（= ctx_len）：1 窗口（不变）
  - T=250：2 窗口（旧方案 1）
  - T=300：3 窗口（旧方案 1）
  - T=400：5 窗口（旧方案 2）
- **科学依据**：重叠窗口提供真正不同的历史上下文——75% 重叠的两个窗口共享历史步但其时序位置不同，注意力机制对不同时序位置产生不同加权，导致编码器输出不同隐状态。
- **`_get_context_for_window` 更新**：`end = T - window_idx * stride`（而非原来的 `T - window_idx * ctx_len`），与新窗口计数公式保持一致。
- **规则**: `n_temporal_windows` 始终基于主模态 T 和 stride 计算；次要模态窗口回退不影响计数；joint 模式使用双模态最小 T。

*Last updated: 2026-03-07*

### [2026-03-07] 响应矩阵纯对角线 Bug — load_graph_for_inference 缺少同模态边重建

#### 根因
- **问题**: 第六步（响应矩阵）生成 y=x 对角线图，即刺激区域 i 只影响区域 i 本身，对其它任何区域均无传播效应。
- **根因**: V5 图缓存（`outputs/graph_cache/*.pt`）**只存储节点特征**（`fmri.x`/`eeg.x`），不含任何边。`load_graph_for_inference` 旧版只添加跨模态边（`eeg → fmri`），**从未添加同模态边**（`fmri 'connects' fmri`）。
- **连锁效应**: 
  1. `GraphNativeBrainEncoder` 对 `edge_type in edge_index_dict` 的判断失败 → 每层图卷积的 `messages` 列表为空 → 返回 `x`（passthrough）→ 编码器退化为纯时序模型（仅 Conv1d + attention）。
  2. `GraphPredictionPropagator` 同样无边 → passthrough → 刺激节点 i 的潜空间扰动经 predictor 传播后，在 propagator 阶段无法扩散到邻居节点。
  3. `causal_effect[j] = sig_perturbed[j] - sig_baseline[j] ≈ 0` 对所有 j≠i → 完美对角线矩阵。
- **旧 docstring 错误**: "只存储同模态边" — 实为"只存储节点特征（不含任何边）"。

#### 修复
- **`load_graph_for_inference`**（`twinbrain-dynamics/loader/load_model.py`）：
  - 导入 `GraphNativeBrainMapper` 一次，供所有边重建步骤共用。
  - 对每个存在于缓存中的模态（`fmri`/`eeg`），从其时序特征重建同模态 FC 边：
    - fMRI：`_compute_correlation_gpu(ts)` → `build_graph_structure(threshold=0.3, k_nearest=20)`
    - EEG：`_compute_eeg_connectivity(ts)` → `build_graph_structure(threshold=0.2, k_nearest=10)`
  - 若缓存中已有有效边（`edge_index.shape[1] > 0`），保留原值，不覆盖。
  - 边重建失败时 log WARNING 并继续（服务器不崩溃），fallback 行为在日志中明确说明。
  - 跨模态边重建逻辑不变（放在同模态边之后）。
  - docstring 更新：说明 V5 缓存只含节点特征、边均需重建，并解释不重建的后果（对角线）。

#### 规则
- `load_graph_for_inference` 调用后，`graph.edge_types` 必须包含 `('fmri', 'connects', 'fmri')`（若 fMRI 存在）和 `('eeg', 'connects', 'eeg')`（若 EEG 存在）。
- 任何需要完整图结构的分析（响应矩阵、动力学分析、EC 推断）都必须通过 `load_graph_for_inference` 加载图，而不是直接 `torch.load`。
- 若直接 `torch.load` 后发现响应矩阵退化为对角线，立即检查 `graph.edge_types` 是否缺少同模态边。

*Last updated: 2026-03-07*

### [2026-03-08] `sample_random_state` z-score 初始状态裁剪错误 & `state_bounds` 单模态失效

#### 根因
- V5 图缓存中 fMRI 和 EEG 均以 **z-score 归一化**存储（均值 ≈ 0，标准差 ≈ 1，值域约 ±3σ）。
- 旧版 `sample_random_state` 对单模态路径（fmri/eeg）执行 `np.clip(mean + noise, 0, 1)`。
  - `mean_state` ≈ 0（z-score 后均值接近零），`noise = N(0, 0.05)`→ x0 ≈ [0, 0.05]。
  - 将 [0, 0.05] 注入上下文末步，而上下文历史值约为 ±3σ，产生巨大跳跃。
  - 效果：刺激前期出现大幅振荡（振幅 ±0.7 AU），如 Image 4（刺激响应图）所示；
    PCA 密度图（Image 2）的 Final states 散点与亮斑中心不一致。
- 旧版 `state_bounds` 对单模态无条件返回 `(0.0, 1.0)`，对 z-scored 数据 Wolf/FTLE
  分析中将扰动状态错误裁剪至 [0,1]，引入 0 处虚假吸引子，偏置 Lyapunov 估计。

#### 修复（twinbrain-dynamics/simulator/brain_dynamics_simulator.py）

1. **`__init__` 新增 `_state_bounds` 属性**：检测 `base_graph[modality].x.min() < -0.1`：
   - z-scored 数据（min < -0.1）→ `self._state_bounds = None`（不裁剪）
   - [0,1] 归一化数据（min ≥ -0.1，旧格式兼容）→ `self._state_bounds = (0.0, 1.0)`
   - joint 模式 → 始终 `None`（拼接 z-score 无界）

2. **`state_bounds` property**：改为直接返回 `self._state_bounds`，移除硬编码 `(0.0, 1.0)`。

3. **`sample_random_state` 修复**：
   - 计算每通道均值 `mean_state` 和标准差 `std_state`（使用 `_STD_GUARD` 保护）
   - 噪声缩放：`noise = N(0, 0.3)`，`x0 = mean + noise * std_state`（0.3σ 扰动，在自然分布内）
   - 仅当 `_state_bounds is not None`（旧格式）时才执行 clip

#### 附带修复（run_dynamics_analysis.py）

4. **PSD burnin 自适应**：Step 13 的默认 burnin 从固定 10 改为 `max(20, T // 10)`（轨迹长度的 10%，最少 20 步），移除初始瞬态对功率谱空间拓扑图的条纹污染（Image 1）。

5. **PCA burnin 默认化**：`plot_pca_trajectories` 调用增加 `burnin=max(0, T // 10)`，PCA 密度图不再包含初始瞬态帧（Image 2 改善）。

#### 规则
- `sample_random_state` 在单模态路径下必须检测数据是否 z-scored，**禁止**对 z-scored 数据执行 `clip(0, 1)`。判据：`base_graph[modality].x.min() < -0.1`。
- `state_bounds` 对 z-scored 单模态数据必须返回 `None`，使 Wolf/FTLE 不裁剪扰动状态。
- PSD burnin 必须与轨迹长度正比（10%），不得使用小于 20 步的固定值。

### [2026-03-08] Jacobian DMD 替换有限差分 & 功率谱线性去趋势 & 能量约束重构

#### Jacobian 谱分析 ρ=0 根因：TwinBrainDigitalTwin 上下文稀释
- **问题**: `estimate_jacobian_at_point` 用 `rollout(x0, steps=1)` 做有限差分，将 x0 注入 200 步上下文窗口的最后一步。Conv1d+注意力编码器对全部 200 步加权平均，单步扰动 ε=1e-4 被稀释为 ≈5×10⁻⁷，在 float32 精度下 `f_fwd - f_bwd = 0` → J=0 → ρ=0。
- **修复**: 删除 `estimate_jacobian_at_point`，用 **Dynamic Mode Decomposition (DMD)** 替代：从自由动力学轨迹提取时序对 (x_t, x_{t+1})，用 Tikhonov 正则化最小二乘拟合最优线性转移算子 A = (X₁ᵀX₀)(X₀ᵀX₀ + αI)⁻¹。**零额外模型调用**，直接复用步骤 3 的轨迹。
- **验证**: 对已知谱半径 0.85 的合成系统，DMD 恢复谱半径 0.8515（误差 < 2%）；对角线系统特征值误差 < 0.01。
- **规则**: TwinBrainDigitalTwin 的 Jacobian **必须**使用 DMD，**禁止**使用 `rollout(steps=1)` 有限差分（上下文稀释）。函数签名保持向后兼容（`n_states`、`epsilon` 参数保留但不使用）。

#### 功率谱 dominant_freq=0.0005 Hz 根因：仅去均值未去趋势
- **问题**: `compute_trajectory_psd` 注释写"Detrend (remove linear trend per channel)"，但代码只做 `seg - seg.mean()`（仅去 DC）。从随机初态收敛到吸引子的轨迹存在强线性趋势，FFT 功率集中在最低频率（T=1000, dt=2s → 0.0005 Hz），掩盖真实振荡。
- **修复**: 改用 `scipy.signal.detrend(seg, axis=0, type='linear')`（有 scipy）或矢量化 OLS 回归（无 scipy），逐通道去除仿射趋势后再加窗。
- **验证**: 含强线性漂移 + 0.05 Hz 振荡的合成轨迹，去趋势后主导频率正确恢复为 0.05 Hz（而非近 DC）。
- **规则**: 功率谱分析**必须**在 FFT 前做完整线性去趋势（`type='linear'`），不允许仅做均值去除（`type='constant'`）。

#### 能量约束实验重构：删除非存在函数，用 `run_energy_budget_analysis` 替代
- **问题**: Step 16 导入 `run_energy_constraint_scan`（α 扫描 x(t+1)=α·F(x(t))）和 `run_dynamic_energy_experiment` 均不存在。α 扫描科学上也是错误的（等比缩放不改变拓扑）。
- **修复**: 添加 `run_energy_budget_analysis(trajectories, state_bounds, output_dir)` — 零额外模型调用，从已有轨迹计算 E*=mean(|x|)，给出 tight(0.4E*)/moderate(0.7E*)/natural(E*)/relaxed(1.3E*) 四档建议值，保存 JSON+PNG。更新 Step 16 使用该函数，更新 `dynamics_config.yaml` 删除废弃的 `run_alpha_scan`/`run_dynamic_energy` 字段。
- **规则**: 真正的能量约束对比用 `--energy-budget X` 重新运行完整管线实现，Step 16 **不应**在单次运行内执行任何额外的 rollout。

*Last updated: 2026-03-08*
