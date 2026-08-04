# CANNBot-Insight

LLM 编码 Agent 的 Session 级可观测工具。辅助长上下文分析、模型幻觉问题治理，以及 Agent Session 中上下文窗口增长的监控与优化。

**[English Documentation](README.md)**

## 功能介绍

导入 opencode sessions.db 或 Claude Code JSONL 日志，逐轮分析 Agent Session：

- **Token 与费用** — 每轮 token 五项拆解柱状图，按模型上下文窗口显示占比；根据使用量和模型定价估算费用
- **上下文增长** — 按 subagent session 展示上下文增长曲线；动画回放 context window 变化过程，含 subagent 生成/消亡标记；`/compact` 压缩标记与上下文下降标注，支持多次压缩
- **上下文治理** — 查看 LLM 输入上下文组成：可见消息 + 稳定的 "System (hidden)" 开销；输入窗口在每个 `/compact` 边界正确截断
- **Subagent 追踪** — 识别 subagent session 与 dispatch→response 链路；subagent turn 在 Turns 时间线内联展示（带徽章），数量显示在 Overview 卡片
- **Skill 事件** — 跟踪每轮 skill load/invoke/use 事件
- **Skill 全文（SKILL.md）** — Skill Summary 与 Skills per Agent 每个 skill 均可点击，取原生 Skill 工具注入内容或 SKILL.md 读取结果重建全文，支持下载
- **概念追踪** — 跨轮次关键词搜索，查看传播链路和 DAG 图；搜索结果支持按 source（用户/模型/子agent 等）、thinking、tools 二次过滤
- **文件读取分析** — 分析文件读取冗余，检测重复和不必要读取
- **文件内容复卷（循迹复卷）** — 每个文件行一个按钮，按时间戳逐行重建该文件全部内容（最后写入胜），未采集到的行标注 `--line N not found --`，支持下载
- **重编汇册（Gather & Rebuild Directory）** — 一键按时间戳逐行重建本会话工作目录下全部被读/写文件，保留路径树；打包下载 zip（纯 JS STORE，无第三方依赖）
- **Session 对比** — 对比两个 session 的 token、费用、耗时、工具调用和 subagent
- **Audit** — 粘贴或从文件导入 workflow 分析 JSON（由 Audit 提示词 + Claude Code 产出），渲染实际流程框图 + 每节点问题、G/S 八维 skill 质量看板、优化优先级。证据里的 `§N` turn 引用可点击跳转到对应 turn；并行分支（同 `parallel` id）渲染为同一行；耗时最长的 3 个节点标红；支持从文件导入 JSON，也可把渲染好的分析导出回文件。生成状态跨 tab 切换不丢失（可恢复）。**v4**：agent 中心三维度审计（功能完成 / 开发效率 / 开发质量），先确定性提取每个 agent 的输入/输出/信封，再由 LLM 评完成度与质量、效率维度由信封确定性计算，渲染 agent 树。Audit tab 拆两个子 tab：**Workflow audit**（上述 v1-v4 分析）与 **Skill audit**——对当前 session 跑 skill-eval 的 `audit`，对账真实执行 vs 每个被调用 skill 的 SKILL.md（不 re-run），以及 vs 每个被 dispatch 的 agent 的 `.md`（从本地 `AGENTS_SCAN_ROOT` 扫描，多插件按 session 覆盖率消歧），以及 vs 主 agent 的 workflow 级 SKILL.md（`--kind root`——切顶层主 agent 作用域审主 agent 编排；主 agent 通常只 dispatch 子 agent、不 invoke skill，故 workflow 声明从 session 首条 user turn 的注入系统提示恢复；仅当首条 user turn 内容 ≥500 字时显示该目标），按 target 出 summary + 原生报告视图（非 iframe）：五态 verdict 徽章、findings 可过滤（按 verdict / 类别 / 方法 / 指令或证据文本）、多 transcript 指令级聚合表、按指令可折叠分组（FAIL 优先排序）、verdict/method 图例、保留"在新页打开原始 HTML"逃生口。运行中有实时进度条 + 百分比（解析 skill-eval on_progress 输出）；结果跨 tab 切换不丢失；需装 `skill-eval` + `claude` CLI；agent 对账需 `AGENTS_SCAN_ROOT` env（默认自动探测 skills-dev 仓库根）。

## 方式一：Web UI

**需要 Node.js >= 20.x**（v18.19.x 无法安装 better-sqlite3 / Prisma 6）。如果有 nvm，`start.sh` 会自动切换到 Node 20 LTS。

日志文件位置：
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

Linux系统（Windows系统请使用start.bat代替）：
```bash
./start.sh              # 自动安装 + 迁移 + 启动 Web UI，端口 21025
./start.sh -u           # 更新依赖 + 迁移 + 启动 Web UI
./start.sh -f           # 清除 .next 缓存，重新编译
```

浏览器打开 `http://localhost:21025`。导入日志文件后，点击 session 进入 9 个分析 Tab。

Web UI 还支持：导出 session 为独立 SQLite 或层级 Markdown；上传 session 到 CANNBay（带提交信息对话框）。CANNBay master 最多保留最新 20 个 session（更旧的自动归档到按月命名的 `archive-YYYY-MM` 分支）；上传用持久本地镜像，每次只 fetch 增量。

## 方式二：CLI 上传 + Web 分析

适用于 SSH 远程服务器、Web IDE 等无浏览器环境。CLI 一步完成导入和上传，之后在 Web UI 上分析。

日志文件位置：
- opencode: `~/.local/share/opencode/sessions.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

```bash
# 从源文件一步上传（源类型根据文件自动识别）
npx tsx src/cli/index.ts upload --file ./sessions.db           # 多个 session 时交互式选择
npx tsx src/cli/index.ts upload --file ./logs/                 # Claude JSONL（目录）
```

上传后会交互式填写提交信息。后端自动启动，上传完成后自动关闭。

上传后在 Web UI 上查看分析：导入时点击 **CANNBay** 按钮，直接从仓库选择 DB 文件导入，无需手动下载。
