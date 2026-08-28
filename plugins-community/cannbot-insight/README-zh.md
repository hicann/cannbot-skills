# CANNBot-Insight

LLM 编码 Agent 的 Session 级可观测工具。辅助长上下文分析、模型幻觉问题治理，以及 Agent Session 中上下文窗口增长的监控与优化。

**[English Documentation](README.md)**

## 功能介绍

导入 SQLite（opencode.db / CANNBot-Insight 归档，自动识别）或 JSONL（Claude 原生 / cpx proxy 捕获）日志，逐轮分析 Agent Session：

- **Token 与费用** — 每轮 token 五项拆解柱状图，按模型上下文窗口显示占比；根据使用量和模型定价估算费用
- **上下文增长** — 按 subagent session 展示上下文增长曲线；动画回放 context window 变化过程，含 subagent 生成/消亡标记；`/compact` 压缩标记与上下文下降标注，支持多次压缩
- **上下文治理** — 查看 LLM 输入上下文组成：可见消息 + 稳定的 "System (hidden)" 开销；输入窗口在每个 `/compact` 边界正确截断
- **Subagent 追踪** — 识别 subagent session 与 dispatch→response 链路；subagent turn 在 Turns 时间线内联展示（带徽章），数量显示在 Overview 卡片
- **Turn 快速跳转** — 在 Turns 时间线输入 turn 编号（如 `#459`）回车即可直达该轮次并滚动到视图；subagent turn 自动展开祖先 dispatch lane
- **Wire 轮次（proxy 捕获）** — 直接读 cpx 捕获文件逐轮重组：每个 wire round 展示完整输入（累积 messages，本轮新增高亮）与输出（response 原文），与 DB 重建的 LLM Input 对照；tab 默认隐藏，`./start.sh -a` 启动才显示
- **密钥脱敏（proxy 捕获）** — API key 永不落盘：在唯一落盘咽喉对敏感字段与各厂家键形（`sk-ant-` / `AIza` / `gsk_` / `LTAI` / `Bearer` 凭据 / env 回显 / URL query）统一打码（`前4…后4`），cpx 启动日志同源掩码
- **Round-Pair 切分（proxy 捕获）** — proxy session 的 Turns 按每轮 输入/输出 成对切分（时间线带 输入/输出 徽章）：输入 turn 逐条展示本轮新增的 verbatim wire 消息，输出 turn 的 LLM Input 存储并直接返回原始累积请求，wire 顺序（含 tool_result 与 system 注入的先后）精确保留，无需重建
- **Skill 事件** — 跟踪每轮 skill load/invoke/use 事件
- **Skill 覆盖度（proxy 捕获）** — 仅 proxy 捕获会话展示：wire verbatim 系统提示里的可用 skills 全集（Claude Code 列表格式 / opencode `<available_skills>` XML）对照会话实际使用，未使用 skills 置顶强调（圆环中心计数 + 首位排序），已使用的按 调用 / 仅加载 / 子代理派发 / 全集外 区分；原生会话（claude-code / opencode）不显示该看板
- **proxy 捕获分类** — 统一分类器（`src/lib/ingest/proxy-classify.ts`）作为 proxy 判定单一权威源：信号优先级 `cc-session-meta`（文件级权威）> 行级 `source` 标记 > wire 指纹（顶层 `system`/`tools`，兜底早期无标记产物）；agent 归属覆盖 claude-code / opencode / codex；一切 proxy 捕获（无论 agent 归属）均按 claude 格式 jsonl 处理 wire 轮次 / full-context / 增量刷新
- **Skill 全文（SKILL.md）** — Skill Summary 与 Skills per Agent 每个 skill 均可点击，取原生 Skill 工具注入内容或 SKILL.md 读取结果重建全文，支持下载
- **概念追踪** — 跨轮次关键词搜索，查看传播链路和 DAG 图；搜索结果支持按 source（用户/模型/子agent 等）、thinking、tools 二次过滤；点击结果的 View Turn 跳转到 Turns tab，自动展开并高亮关键词至匹配位置（内容 / 工具参数 / 结果 / 错误）
- **文件读取分析** — 分析文件读取冗余，检测重复和不必要读取
- **文件内容复卷（循迹复卷）** — 每个文件行一个按钮，按时间戳逐行重建该文件全部内容（最后写入胜），未采集到的行标注 `--line N not found --`，支持下载
- **重编汇册（Gather & Rebuild Directory）** — 一键按时间戳逐行重建本会话工作目录下全部被读/写文件，保留路径树；打包下载 zip（纯 JS STORE，无第三方依赖）
- **Session 对比** — 对比两个 session 的 token、费用、耗时、工具调用和 subagent
- **Audit** — 粘贴或从文件导入 workflow 分析 JSON（由 Audit 提示词 + Claude Code 产出），渲染实际流程框图 + 每节点问题、G/S 八维 skill 质量看板、优化优先级。证据里的 `§N` turn 引用可点击跳转到对应 turn；并行分支（同 `parallel` id）渲染为同一行；耗时最长的 3 个节点标红；支持从文件导入 JSON，也可把渲染好的分析导出回文件。生成状态跨 tab 切换不丢失（可恢复）。**v4**：agent 中心三维度审计（功能完成 / 开发效率 / 开发质量），先确定性提取每个 agent 的输入/输出/信封，再由 LLM 评完成度与质量、效率维度由信封确定性计算，渲染 agent 树。agent 树默认隐藏全部通过的子 agent（勾选开关才显示）；跨 agent 问题与优化优先级每项都给出证据（`agent:<name> #turn`，可点击跳转），审计头部展示本次审计消耗的 token 数。Audit tab 拆两个子 tab：**Workflow audit**（上述 v1-v4 分析）与 **Skill audit**——对当前 session 跑 sift 的 `audit`，对账真实执行 vs 每个被调用 skill 的 SKILL.md（不 re-run），以及 vs 每个被 dispatch 的 agent 的 `.md`（从本地 `AGENTS_SCAN_ROOT` 扫描，多插件按 session 覆盖率消歧），以及 vs 主 agent 的 workflow 级 SKILL.md（`--kind root`——切顶层主 agent 作用域审主 agent 编排；主 agent 通常只 dispatch 子 agent、不 invoke skill，故 workflow 声明从 session 首条 user turn 的注入系统提示恢复；仅当首条 user turn 内容 ≥500 字时显示该目标），按 target 出 summary + 原生报告视图（非 iframe）：五态 verdict 徽章、findings 可过滤（按 verdict / 类别 / 方法 / 指令或证据文本）、多 transcript 指令级聚合表、按指令可折叠分组（FAIL 优先排序）、verdict/method 图例、保留"在新页打开原始 HTML"逃生口。运行中有实时进度条 + 百分比（解析 sift on_progress 输出）；结果跨 tab 切换不丢失；需装 `sift` + `claude` CLI；agent 对账需 `AGENTS_SCAN_ROOT` env（默认自动探测 skills-dev 仓库根）。

## 方式零：tgz 安装（推荐）

无需克隆源码、无需手工配环境、无需 npm registry。从分发的 `.tgz` 包文件安装。

**安装**（从维护者 / GitCode Release 获取 `cannbot-insight-1.83.0.tgz` 后）：

```bash
npm install -g ./cannbot-insight-1.83.0.tgz
cannbot-insight            # 首次：自动 build + migrate（约 30s-2min），随后启动 + 开浏览器
```

运行时依赖（next/prisma/better-sqlite3 等）在安装时自动从公共 npmjs 拉取，只有包本体在私有 `.tgz` 里。首次安装会编译 `better-sqlite3` 原生模块（约 1-2 分钟，需 `python3`/`make`/`g++` 或 prebuild 命中）。需 Node.js >= 20。

所有可写状态集中在 `~/.cannbot-insight/`（SQLite 库 + `.next/` 构建缓存），安装包目录保持只读。默认端口 21025（被占则自动找空闲端口）。

| 标志 | 作用 |
|------|------|
| `-a`, `--advanced` | 显示高级 tab（wireRounds/replay） |
| `-k`, `--kill` | 杀掉占用 21025 端口的进程，复用该端口 |
| `-f`, `--fresh` | 清空 `.next` 构建缓存重新构建 |

CLI 子命令直接透传（同方式二，但无需 `cd`/`npx tsx` 前缀）：

```bash
cannbot-insight upload --file ~/.local/share/opencode/opencode.db --list
cannbot-insight sessions
cannbot-insight --help
```

**需 Node.js >= 20。** Python 3 为可选项（自动探测）：存在则启动 smart-agent 后端（breather/v2 分析，端口 21026）；缺失则静默降级，其余功能正常。

## 方式一：Web UI

**需要 Node.js >= 20.x**（v18.19.x 无法安装 better-sqlite3 / Prisma 6）。如果有 nvm，`start.sh` 会自动切换到 Node 20 LTS。

日志文件位置：
- opencode: `~/.local/share/opencode/opencode.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

Linux系统（Windows系统请使用start.bat代替）：
```bash
./start.sh              # 自动安装 + 迁移 + 启动 Web UI，端口 21025
./start.sh -u           # 更新依赖 + 迁移 + 启动 Web UI
./start.sh -f           # 清除 .next 缓存，重新编译
```

浏览器打开 `http://localhost:21025`。导入日志文件后，点击 session 进入 9 个分析 Tab。

Web UI 还支持：导出 session 为独立 SQLite 或层级 Markdown；上传 session 到 CANNBay v2（atomgit 数据集仓，带提交信息对话框）。上传格式为 proxy 捕获的 claude-jsonl 文件夹（主会话 + subagents），opencode 会话自动从 DB 导出 jsonl；上传前强制数据治理（密钥清洗 + 残留熔断，密钥不出公开仓）。CANNBay 列表/下载走 partial clone（`--filter=blob:none`）持久镜像：列表只读元数据秒级返回，导入单条才按需下载 blob，千级会话仓库也不变慢。旧版 .db 快照上传/解析（gitcode CANNBay）后端保留，用于读取历史数据，界面不再展示。 所有上传/捕获/导出 jsonl 同时携带声明式 `x_cannbay` 扩展命名空间（信封冻结 + `{schema, version, data}` 口袋，规范见 docs/cannbay-schema-spec.md），并修复 6 项导出丢失（reasoningTokens/ttftMs/ToolCall 明细/framework/模型参数/wire 状态码）；列清单对账 round-trip 集成测试证明 DB → 导出 → 再导入零丢失。

## 方式二：CLI 上传 + Web 分析

适用于 SSH 远程服务器、Web IDE 等无浏览器环境。CLI 一步完成导入和上传，之后在 Web UI 上分析。

日志文件位置：
- opencode: `~/.local/share/opencode/opencode.db`
- Claude Code: `~/.claude/projects/<hash>/sessions/<id>.jsonl`，也可指定目录自动扫描

### 一次性初始化（新环境首次）

```bash
cd cannbot-insight
npm install
echo 'DATABASE_URL="file:./dev.db"' > .env
npx prisma generate
npx prisma migrate deploy
```

约 1 分钟，之后无需重复（除非依赖更新）。CLI 上传命令会自动拉起后端，无需手动跑 `start.sh`。

### 上传 Session

**opencode：**

```bash
# 列出可选 session
npx tsx src/cli/index.ts upload --file ~/.local/share/opencode/opencode.db --list

# 指定 session-id 上传（非交互，适合脚本）
npx tsx src/cli/index.ts upload \
  --file ~/.local/share/opencode/opencode.db \
  --session-id <session-id> \
  --description "描述" \
  --yes --json

# 交互式选择 session 并填写描述
npx tsx src/cli/index.ts upload --file ~/.local/share/opencode/opencode.db
```

**Claude Code：**

```bash
# 列出可选 session（指定 projects 目录，自动递归扫描 .jsonl）
npx tsx src/cli/index.ts upload --file ~/.claude/projects/ --list

# 指定 session-id 上传（session-id = .jsonl 文件名，不含扩展名）
npx tsx src/cli/index.ts upload \
  --file ~/.claude/projects/ \
  --session-id <uuid> \
  --description "描述" \
  --yes --json

# 也可直接指定单个 .jsonl 文件
npx tsx src/cli/index.ts upload --file ~/.claude/projects/<hash>/<uuid>.jsonl --list
```

| 参数 | 说明 |
|------|------|
| `--file <path>` | 源路径：opencode `.db` 文件 / Claude Code `.jsonl` 文件或目录，源类型自动推断 |
| `--session-id <id>` | 指定上传哪个 session（跳过交互选择） |
| `--description <text>` | 提交描述（跳过交互填写） |
| `--yes` | 跳过确认提示 |
| `--json` | JSON 输出（非交互，适合脚本） |
| `--list` | 仅列出可选 session，不上传 |
| `--framework <name>` | 框架类型（自动推断：opencode-db → opencode，claude-jsonl → claude-code） |

上传后在 Web UI 上查看分析：导入时点击 **CANNBay** 按钮，直接从仓库选择会话文件夹导入（按需下载该会话文件），无需手动 clone。

## 方式三：零依赖导出 Session DB

从 opencode.db 或 Claude Code JSONL 提取单个 session（含子会话）为独立文件。**零依赖**——只需 Node.js >= 22（内置 `node:sqlite`），无需 `npm install`、无需 Prisma、无需后端。根据 `--file` 自动识别框架：`.db` → opencode，`.jsonl`/目录 → Claude Code。

```bash
# 列出可选 session（自动检测源）
node export-db.mjs --list
node export-db.mjs -f ~/.claude/projects/ --list

# 交互式选择 session 并导出
node export-db.mjs
node export-db.mjs -f ~/.claude/projects/

# 指定 session-id 导出
node export-db.mjs -s ses_xxx                          # opencode
node export-db.mjs -f ~/.claude/projects/ -s <uuid>     # Claude Code

# 指定输出路径
node export-db.mjs -s ses_xxx -o ~/my-session.db
node export-db.mjs -f ~/.claude/projects/ -s <uuid> -o ~/my-session.jsonl

# JSON 输出（脚本友好）
node export-db.mjs -s ses_xxx --json
```

未指定 `-o` 时，默认输出到脚本同目录下：
- opencode → `dbfile/session_<id>.db`
- Claude Code → `jsonlfile/<id>.jsonl`

导出的文件可通过方式二的 `upload --file` 命令重新导入 Insight 或上传 CANNBay：

```bash
npx tsx src/cli/index.ts upload --file /abs/path/to/session_xxx.db  --session-id ses_xxx --yes --json
npx tsx src/cli/index.ts upload --file /abs/path/to/<uuid>.jsonl    --session-id <uuid>  --yes --json
```

| 参数 | 说明 |
|------|------|
| `-f, --file <path>` | 源路径（默认自动检测：opencode.db 或 ~/.claude/projects/） |
| `-s, --session-id <id>` | 指定导出哪个 session（跳过交互选择） |
| `-o, --output <path>` | 输出路径（默认 `dbfile/` 或 `jsonlfile/`） |
| `-l, --list` | 仅列出可选 session，不导出 |
| `-j, --json` | JSON 输出 |
| `-h, --help` | 显示帮助 |
