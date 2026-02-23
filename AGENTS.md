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

---

*Last updated: 2026-02-21*
