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

---

*Last updated: 2026-02-20*
