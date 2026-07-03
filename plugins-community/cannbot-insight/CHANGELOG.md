# Changelog

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

## [v0.96] - 2026-06-29

### Changed
- README 面向用户重写 — 功能介绍 + 方式一（Web UI）+ 方式二（CLI 上传 + Web 分析），日志路径说明，CANNBay 直接导入（无需手动下载）
- README 去除技术栈/运行测试/许可等开发者信息（保留在 CLAUDE.md）

## [v0.95] - 2026-06-29

### Added
- CANNBay 上传描述对话框 — Web UI 上传按钮改为弹出对话框，预填模板（提交人/内容描述/问题说明/日志路径/备注），sourcePath 自动填充
- CLI upload 命令 — `upload --file <path>` 一步完成导入+上传，支持 opencode .db 和 Claude JSONL
- 源类型自动识别 — 根据文件扩展名自动选择 adapter，无需 `--source` 参数
- 多 session 交互式选择 — 源文件包含多个 session 时自动列出并让用户选择
- 交互式描述填写 — CLI 上传时交互式填写提交信息（与 Web UI 对话框一致），从 session 数据自动填充默认值
- 后端自动管理 — CLI upload 命令自动启动后端（如未运行），上传完成后自动关闭
- TUI 上传入口 — SessionList 新增 'u' 键上传，ConfirmDialog 确认

### Changed
- Upload API commit message — 使用用户描述作为提交信息（不再使用 "Add session <taskId>"）
- README 简化 — 去除 beta/内部功能描述，中英文一致，聚焦核心功能

## [v0.76] - 2026-06-19

### Changed
- TurnContextPanel 新增 "Cached" 类别 — cacheReadTokens + cacheWriteTokens 单独展示为黄色 Cached 区段，不再混入 "Other context"
- ContextInfo 接口增加 cacheReadTokens/cacheWriteTokens 字段
- deltaTokens/unclassifiedTokens 计算排除 cacheInputTokens，避免 "Other context" 虚大

## [v0.75] - 2026-06-19

### Changed
- Input 统一 — TokenBarChart/Overview/ContextTracker 不再分开显示 Input/Cache Read/Cache Write，统一为 "Input" (= inputTokens + cacheReadTokens + cacheWriteTokens)
- inputMessagesTokens 公式从 `inputTokens + cacheReadTokens` 改为 `inputTokens + cacheReadTokens + cacheWriteTokens`（cache_write 也是模型输入的一部分）
- Skill injection system turns 合入前一个 assistant turn 的 Skill tool_call resultJson，不再作为独立 turn 显示

## [v0.74] - 2026-06-19

### Fixed
- Claude Code JSONL 流式行合并 — 同一 API 调用的 thinking/text/tool_use 多行合并为一个 turn，消除 context 0%跳变和重复 usage
- context-window-config 增加 qwen3.7-max (1M) — 修复 qwen3.7-max 被 includes fallback 匹配到 qwen3→128K 导致 contextPct>200%
- DB 修复 — 已入库 qwen3.7-max turns 的 contextWindowPct 用 1M 重新计算（207%→25.9%）

### Changed
- claude-jsonl adapter: 连续 assistant 行合并为 AssistantGroup，取最后一行含 cache 字段的 usage 数据
- context-window-config: 新增 qwen3-235b、qwen3.7-max 条目

## [v0.73] - 2026-06-19

### Added
- Claude Code subagent 导入 — listSubagentSessions 路径修正，Agent 工具 dispatch 检测，toolUseId→subagentSessionId 精确映射
- Claude Code 版本号提取 — JSONL line.version 字段解析，Overview 显示 "Claude Code v2.1.143"
- 费用估算 — Claude Code JSONL 无 cost_usd 字段，从 token 使用量和模型定价估算（Opus/Sonnet/Haiku）
- 耗时推断 — Claude Code JSONL 无 duration_ms，从相邻轮次 timestamp 差值推算 latency
- 单条 session 删除 — 每行 TrashIcon 按钮 + 确认对话框，仅删除 Insight DB 数据，原始文件不受影响
- JSONL 一键导入 — 单个 .jsonl 文件跳过 session 选择步骤，直接导入
- import-file API 返回 query 字段 — 直接导入场景下 ImportHistory 正确记录首条查询内容

### Fixed
- Skill 工具检测大小写 — Claude Code 用 "Skill"（大写 S），改为 toLowerCase() 匹配
- Skill 名称提取 — Claude Code args.skill 格式支持
- Tokens 负数 — Overview 计算 totalTokens - cacheReadTokens 在 claude-code 场景产生负值，改为直接显示 totalTokens
- totalTokens 含缓存 — claude-jsonl mapUsage total 从 input+output 改为 input+output+cacheRead+cacheWrite
- Latency 累加范围 — 仅累加 assistant turns 的 latency，排除 user turns
- 删除确认文案 — 从"不可撤销/永久删除"改为"原始文件不受影响，可随时重新导入"
- ImportHistory 列顺序 — File Path → Query → Import Time → Type → Status → 删除按钮
- 删除图标统一 — ImportHistory 与 SessionList 均使用 TrashIcon

### Changed
- RawInteraction 增加 subagent_type 字段
- TurnRow 增加 subagentType 字段（Prisma 写入前剔除）
- bridge-builder 支持 Agent 工具名 + toolUseIdToSubagentSessionId 参数
- execution-split 从 turn.subagentType 优先提取 subagent 类型

## [v0.72] - 2026-06-18

### Added
- Node.js 版本校验 — start.sh 启动时检查 Node >= 20.x，v18.19.x 无法安装 better-sqlite3 / Prisma 6
- nvm 自动切换 — 有 nvm 时自动安装 Node 20 LTS 并切换，无 nvm 时提示安装指引
- export-service 测试用例 — 10 个测试覆盖 INSERT 列名/占位符/run() 参数三方匹配、DDL 可执行性、Session 30-vs-31 回归守卫

### Fixed
- Upload to CANNBay 失败 — export-service Session INSERT 31 列名 / 30 占位符不匹配（Turn 同类 bug 33/32 一并修复）
- Import 完成后需手动点 Done — 导入完成后 1.5 秒自动跳回主页并刷新

### Changed
- .gitignore 新增 `.claude/`（per-user 设置）、`package-lock.json`（平台相关生成文件）、`docs/dev-log.md` 和 `docs/report.md`（临时开发日志）
- 从 git 追踪移除 `package-lock.json`、`docs/dev-log.md`、`docs/report.md`

## [v0.69] - 2026-06-17

### Added
- inputMessagesJson 重建包含 prior assistant turns 的 tool_call args + result（按正确顺序嵌入）
- LlmContextView 和 TurnContextPanel 展开 assistant 消息时显示 tool calls（橙色 badge + args/result + token 数）
- System (hidden) 拆为稳定值 + Other context delta，数字可闭合
- computeSystemOverhead 计算 now 包含 prior tool call args tokens

### Fixed
- 8.1kt context gap — prior assistant tool_call argsJson 未计入 visible tokens
- ImportHistory 单条删除无论 DB 是否存在都移除 localStorage 条目
- LlmOutputView badge 直接显示 DB 原始值不再拆分估算
- SkillEventList token 改为 skill tool call args+result 估算
- TurnContextPanel System (hidden) 与 LlmContextView 使用一致逻辑
- LlmContextView 删除 per-category 百分比和底部 Total 行

## [v0.68] - 2026-06-17

### Changed
- LlmOutputView badge 直接显示 DB 原始值 `reasoningTokens + outputTokens`，不再拆分估算 tool args
- LlmContextView 删除 per-category 百分比和底部 Total 汇总行（冗余）
- SkillEventList token 改为 skill 相关 tool call 的 args+result 估算（与 Tool Calls 区段一致）
- TurnContextPanel System (hidden) 使用 API 返回的稳定 systemOverheadTokens（与中间 LlmInput 一致）

### Fixed
- ImportHistory 单条删除：无论 DB session 是否存在都移除 localStorage 条目（解决先删 session 后删 import 卡死的问题）
- LlmOutputView text 区段增加 labeled header（text badge + token 数）

## [v0.67] - 2026-06-17

### Added
- LLM Input 自动展开 — 可见 token < 6000 时全部消息默认展开显示完整内容
- TokenBarChart 按模型上下文窗口显示 Input/Output/Tool Calls 三段占比
- Overview 合并 Timing & Model + Token Usage 为一张卡片
- System (hidden) 稳定值 — 从首轮 assistant turn 计算，per-agent（root/subagent 各自独立）
- Tool Calls 标题格式 `Tool Calls (N, M skill)`
- SkillEventList 显示单条 token 数量

### Fixed
- inputMessagesJson 重建不再截断消息内容（之前硬编码 200 字符截断，前端永远看不到完整内容）
- inputMessagesJson 重建按 agent 隔离 — subagent 只用同 subagentSessionId 的 prior turns
- inputMessagesCount 按作用域计算 — subagent 只计算同 subagentSessionId 的 prior turns
- computeSystemOverhead per-agent 计算 — subagent 使用自身首轮 assistant，不再混入 root 数据
- subagent turn 选中时 root context 使用 rootTurn 数据（不再被 subagent 详情覆盖）
- LlmContextView 消息按钮行去掉摘要预览（CSS truncate 截断导致看不全），改为 role + token 数
- setHighlightSubagentSessionId → setHighlightSubagentTurnId（构建失败修复）

## [v0.66] - 2026-06-17

### Added
- LlmContextView 分离 visible messages 与 hidden system context
- TurnDetail 概览卡片合并（Token Usage + Timing & Model）
- TokenBarChart 三段 Input/Output/Tool Calls 显示

## [v0.65] - 2026-06-17

### Added
- start.sh `-f` flag — 清除 .next 缓存重新编译（Turbopack 缓存不生效时使用）

### Fixed
- `contextWindowPct` 和 `inputMessagesTokens` 现在包含 cacheReadTokens（真实 LLM 输入，不含 cache 时数据严重失真）
- TurnDetail "in: XXk" 显示模型真实输入（inputTokens + cacheReadTokens）
- ContextTracker / ContextReplay / GrowthChart 使用真实输入作为 context size
- subagent turn Context panel 加载 inputMessagesJson（之前只为 root turn 加载）
- inputMessagesJson 重建添加每条消息的 estimated tokenCount（之前显示 0t 0%）
- inputMessagesJson 重建逻辑修复：subagent turn 使用全局 prior turns 而非仅 subagent 内部 turns

## [v0.64] - 2026-06-16

### Added
- ESLint `no-mixed-operators` 规则 — 拦截 `&&`/`||`/`??` 混用，防止运算符优先级歧义导致的运行时解析错误

### Fixed
- Import Browse 输入文件路径后点击 Browse 打开父目录（之前停留在 input 步骤无反馈）
- LocalFileImport `??`/`||` 混用加括号（Next.js 解析器要求）
- CompareView `&&`/`||` 混用加括号

## [v0.63] - 2026-06-16

### Added
- Session 导出 — 导出 session 为独立 SQLite 文件 (`cannbot_session_<taskId>.db`)，Web UI 导出按钮 + CLI `cbin export` 命令
- CANNBot Insight 导入适配器 — 导出的 SQLite 文件可重新导入回 Insight（`import --source cannbot-insight`）
- 复合唯一键 `@@unique([taskId, framework])` — 同一 session 从不同源（opencode-db vs cannbot-insight）并存为两条记录
- Session Compare 改用 sessionId（Prisma cuid）选取和对比，支持同一 taskId 不同源的 session 对比
- 所有 observe API 端点新增可选 `framework` 参数用于多源 session 精确查找
- session API 端点新增 `sessionId` 参数用于 compare 页直接 cuid 查找
- ImportHistory / SessionList 新增 CANNBot 紫色 badge 显示 cannbot-insight 类型
- Agent Timeline 图表 — token heatmap + 点击导航到对应 Turn
- Agent firstPrompt — execution 列表显示每个 agent 的首条用户提示词

### Fixed
- start.sh 自动创建 `.env` 文件（修复首次运行 Prisma migration 报 `DATABASE_URL` 缺失，Issue #1）
- File Reads 点击 Agent 跳转到对应 Turn
- Workflow phase 渲染使用 unique phaseSequence 防止重复

## [v0.62] - 2026-06-16

### Fixed
- Subagent Tab 跳转和名称显示问题 — 使用 agentName 替代 subagentName
- computeEndPct 提升到组件级别（避免在 renderTurns 内重复创建闭包）

## [v0.61] - 2026-06-16

### Added
- Context panel 双指标条 — 显示 start→end pct（上下文窗口起始和结束占比）

## [v0.60] - 2026-06-16

### Added
- Import 性能优化（23.5s → 1.6s）— batch write + 并发导入
- Import Session 对话框默认路径 `/`
- start.sh `-k` flag — 杀掉占用 21025 端口的进程
- start.sh 自动打开浏览器（WSL: cmd.exe / macOS: open / Linux: xdg-open）

### Fixed
- WSL 自动打开浏览器 — 通过 cmd.exe 全路径（interop 不在 PATH 时也能工作）

## [v0.59] - 2026-06-16

### Added
- CLI 功能对齐 — Interactions tab、search、ToolCalls 统计、detail 增强

## [v0.58] - 2026-06-16

### Added
- CLI 功能对齐 — Overview cards、Turn detail、Trace tab、Context trend、Subagent hierarchy

## [v0.57] - 2026-06-16

### Added
- CLI WorkflowTab 交互 — cursor 导航、展开/折叠、context 显示

## [v0.56] - 2026-06-16

### Added
- Workflow tab 右侧 Context panel
- Token 拆分为两个卡片 — Tokens + Cache Read

## [v0.55] - 2026-06-16

### Added
- Context Replay 动画 — subagent spawn/death 标记可视化

## [v0.54] - 2026-06-16

### Added
- WorkflowTreeView 重设计
- opencode-db adapter 重构
- TurnContextPanel 组件

### Fixed
- subagent 显示使用 agentName

## [v0.51] - 2026-06-16

### Added
- TurnTimeline 重设计 — root turns 主时间线 + subagent 并行车道

## [v0.50] - 2026-06-16

### Fixed
- Workflow phase 去重 — 同 phaseIndex 不创建重复 phase
- strip `<thinking>` 标签从 phase 名称

## [v0.49] - 2026-06-16

### Fixed
- Workflow phases 填补 turn 间隙 — phase 边界之间 turns 连续分配

## [v0.48] - 2026-06-16

### Added
- Skill badge 显示 skill 名称（而非数量）
- Overview 会话摘要 + agents/tools 两列布局

### Fixed
- Context page RangeError — allSessions unchecked 时 GrowthChart 空 allPoints 处理

## [v0.45] - 2026-06-16

### Added
- Turns tab 左右面板独立滚动 + turn 切换自动滚动到顶部

## [v0.44] - 2026-06-16

### Added
- Step node 展开 subagent turns（thinking/tool calls/skills）
- turns API subagentSessionId filter + includeDetail param
- Prep phase turn range 包含所有 root turns

## [v0.43] - 2026-06-15

### Added
- Workflow 准备阶段 — 阶段一之前的 turns 自动归入"准备阶段"(phaseIndex=0)，包含 content 摘要
- WorkflowTurnNode 类型 — 新增 turn 类型节点，用于展示纯文本 turn（无 subagent dispatch）

## [v0.42] - 2026-06-15

### Added
- Session Compare Turn-by-Turn tab — 每轮 turn 左右对比，显示完整提示词/输入/输出/thinking 内容
- Turns API `includeContent=true` 参数 — compare 页面切换 Turns tab 时按需加载 turn content
- Compare 页面 tab 切换（Overview / Turn-by-Turn）
- 首页 Compare 入口提示 — 显示"Select 2 sessions to compare"/"1 selected — select 1 more"

## [v0.41] - 2026-06-15

### Added
- Tool Calls 卡片显示 token 消耗（total/in/out + latency）
- opencode-db 适配器 N+1 优化 — listSessions 从 ~200+ 次查询减少到 5-6 次批量查询
- opencode-db 适配器 readSession N+1 优化 — 从 N×3 次 part 查询减少到 3 次批量查询
- 导入计时日志 — console.log 打印各阶段耗时便于排查

### Changed
- LlmOutputView 始终传递完整 content — 不再因 content > 10K 而隐藏

## [v0.40] - 2026-06-15

### Added
- Thinking 内容显示 — opencode-db 适配器 reasoning 内容用 `<thinking>` 标签包裹，Web UI 自动识别并折叠展示
- Tool call 截断提升 — argsJson/resultJson 显示截断从 500 字符提升至 2000 字符，标注总长度

### Fixed
- LlmOutputView 始终传递完整 content — 不再因 content > 10K 而隐藏 thinking/输出内容
- isLongContent 警告移除 — 内容不再被抑制，由组件内滚动条自然处理

## [v0.39] - 2026-06-15

### Fixed
- 导入事务超时 — $transaction timeout 从默认 5s 提升至 60s，maxWait 从 2s 提升至 30s，解决大 session 导入报错 "Transaction already closed"

## [v0.38] - 2026-06-15

### Added
- Web UI 目录浏览器 — Import Session 对话框新增 Browse 按钮，交互式浏览目录选择文件
- API 端点 `/api/ingest/browse-directory` — 服务端目录列表，自动标记可导入文件（.db/.jsonl）及 importableType
- CLI FilePicker 组件 — TUI ImportPanel 输入目录路径后显示目录浏览器

## [v0.37] - 2026-06-15

### Changed
- 导入性能优化 — create → createMany + $transaction 批量写入，单 session 35-100x 加速
- 新建路径：8 条 createMany 替代 ~104 条 create（turns/toolCalls/skillEvents/bridges/executions/executionSkills/sessionSkills + session）
- 增量路径：3 条 createMany 替代逐条 merge 写入
- batchCreateMany 分批保护（500 条/批），防止 SQLite 变量限制溢出

## [v0.36] - 2026-06-15

### Added
- CLI 前端迭代5 — ImportPanel 交互修复（filePath/source 可输入）、SessionList 搜索修复、TUI 快捷键（Space/c/i/d）
- analyze 命令 — AI workflow 分析（--base-url/--api-key/--model/--json）
- DataTable 标记对比功能（◉ prefix + markedIds/idKey）
- start.sh `-c` flag — 后台启动后端并等待就绪

### Fixed
- ContextTab 移除重复 MODEL_CONTEXT_WINDOWS，使用共享 context-window-config
- App.tsx 传递 onCompare/onImport/onDelete，delete 有 ConfirmDialog

## [v0.34] - 2026-06-14

### Added
- CLI 前端迭代3 — TUI 交互模式（Ink + React），SessionList/SessionDetail/TurnDetail/CompareView/ImportPanel 五屏
- StatusBar/KeyBar/DataTable/MetricCards/TextInput 等自建组件

## [v0.33] - 2026-06-13

### Added
- CLI 前端迭代2 — 核心命令实现（sessions/session/turn/search/compare/stats/import/delete/config）

## [v0.32] - 2026-06-12

### Added
- CLI 前端迭代1 — 地基模块（Commander.js 入口、API client 15 端点、类型定义、hooks）

## [v0.30] - 2025-06-14

### Added
- Session Compare — 首页选取两个 session 对比（token、费用、耗时、工具调用、subagent）

## [v0.20] - 2025-06-14

### Added
- Claude Code JSONL 适配器 — 导入 Claude Code session JSONL 文件
- 多适配器架构（opencode-db + claude-jsonl）

## [v0.15] - 2025-06-07

### Added
- Context 📊 Tab — 上下文追踪，按 subagent session 展示增长曲线
- AI Workflow (beta) — LLM 驱动的 workflow 分析

## [v0.13] - 2025-06-05

### Added
- Trace 🔍 Tab — 概念溯源追踪，关键词搜索、传播链路、DAG 图

## [v0.12] - 2025-06-04

### Added
- Interactions 页面重构 + 时间轴

## [v0.09] - 2025-06-01

### Added
- Workflow ✦ Tab — phase-split 核心算法 + WorkflowTreeView
- Subagents Top3 消耗标识

## [v0.01] - 2025-05-24

### Added
- 初始版本：Opencode DB 导入、Turn 级存储、Token 五项拆解
- 上下文治理、Subagent 追踪、Skill 事件、交互链路图
- Web UI（9 个 Tab）
