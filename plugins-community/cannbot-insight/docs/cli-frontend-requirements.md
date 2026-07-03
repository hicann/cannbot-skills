# CANNBot-Insight CLI 前端需求文档

> **版本**: v1.1  
> **日期**: 2025-06-14  
> **作者**: Guan Xinghua  
> **PoC 验证**: Ink v7.0.6 + React 19.2.7 + ESM + tsx  
> **状态**: PoC 验证后更新

---

## 1. 背景与目标

### 1.1 现状

CANNBot-Insight 目前只有一个 Web 前端（Next.js 16 + shadcn/ui），提供 3 个页面、15 个 API 端点，覆盖 Session 导入、观测、对比、AI 分析等功能。

### 1.2 痛点

| 痛点 | 说明 |
|------|------|
| **必须启动浏览器** | 在 SSH 远程服务器、无桌面环境下无法使用 |
| **不适合脚本化** | 无法在 CI/CD、cron 任务中自动查询和分析 |
| **信息密度低** | Web UI 的卡片和图表在快速排查问题时不如终端高效 |
| **开发者偏好** | 目标用户（LLM Agent 开发者）习惯终端工作流 |

### 1.3 目标

> **后端零改动，新增一套纯 CLI 前端，复用现有 15 个 API 端点。**

- 🖥️ 提供两种模式：**TUI 交互模式**（Terminal UI）+ **纯命令行模式**（单次命令）
- 🔧 后端 API 完全不变，CLI 作为 API 客户端调用
- 📦 可独立安装，也可作为 `npm script` 使用

---

## 2. 竞品参考

| 工具 | 模式 | 值得借鉴 |
|------|------|----------|
| **k9s** | TUI | 表格+详情面板、快捷键导航、资源树 |
| **lazygit** | TUI | 分栏布局、左右面板切换、内联 diff |
| **htop/btop** | TUI | 实时刷新、ASCII 柱状图、颜色编码 |
| **mitmproxy** | 双模 | Web + CLI 共存、flow 列表+详情钻取 |
| **SimonW's llm** | 纯命令 | `llm logs` 列表、`llm logs -r ID` 详情、管道友好 |
| **gh CLI** | 纯命令 | `gh pr list` → `gh pr view N` 列表+详情范式 |
| **LiteLLM** | 双模 | CLI proxy + Web dashboard 并行 |

**核心 UX 范式**：`list → select → detail`（列表浏览 → 选择 → 详情钻取）

---

## 3. 技术方案

### 3.1 技术选型

| 方案 | 推荐 | 说明 |
|------|------|------|
| **Ink v7** (React for CLI) | ✅ 首选 | React 19 + ESM，与现有 React 技术栈统一；第三方组件自实现 |
| **Blessed/Blessed-Contrib** | ❌ | 成熟但非 React 范式，学习成本高 |
| **Terminal-kit** | ❌ | 低级 API，开发效率低 |
| **纯 Commander.js** | ⚡ 备选 | 仅做纯命令模式，不做 TUI |

**推荐：Ink v7 + 自实现组件**（PoC 验证：第三方 Ink 组件不兼容 v7，全部自实现）
- `DataTable` — 自写表格（string-width 视觉宽度对齐）
- `SelectInput` — 自写选择菜单（<50 行）
- `TextInput` — 自写输入框（30 行）
- `Spinner` — 自写加载动画（10 行）
- Ink 内置 `<Box>` — 布局容器
- chalk.bold — 标题着色（替代 ink-big-text）

> **ESM 要求**: Ink v7 + yoga-layout 使用 top-level await，package.json 必须设 `"type": "module"`，运行工具用 tsx（不是 ts-node）

#### 核心依赖版本（PoC 验证锁定）

| 包名 | 版本 | 用途 |
|------|------|------|
| `ink` | ^7.0.6 | React CLI 渲染引擎 |
| `react` | ^19.2.7 | UI 框架 |
| `tsx` | latest | TypeScript ESM 运行工具（替代 ts-node） |
| `commander` | latest | 命令行参数解析（纯命令模式） |
| `chalk` | latest | 终端颜色输出 |
| `string-width` | latest | 中文/CJK 字符视觉宽度计算 |
| `cli-truncate` | latest | 视觉宽度安全的字符串截断 |

> **重要**: 不要安装 ink-table、ink-select、ink-spinner、ink-text-input、ink-big-text 等第三方 Ink 组件库——它们均不兼容 Ink v7，必须全部自实现。

#### 测试依赖

| 包名 | 用途 |
|------|------|
| `ink-testing-library` | Ink 组件测试（完全兼容 v7，提供 `lastFrame`） |

> **Ink v7 render() 限制**: Ink v7 原生 `render()` 返回对象不包含 `lastFrame` / `frames` / `output` 属性。测试时必须使用 `ink-testing-library` 的 `render()`，它提供 `lastFrame()` 方法获取最近一帧输出。

### 3.2 架构设计

```
┌──────────────────────────────────────────────┐
│              CANNBot-Insight                 │
├──────────────────┬───────────────────────────┤
│   Web Frontend   │      CLI Frontend         │
│   (Next.js)      │      (Ink / Commander)    │
│                  │                           │
│   /page.tsx      │   cli/index.tsx           │
│   /session/...   │   cli/commands/           │
│   /compare       │   cli/components/         │
│                  │   cli/hooks/              │
├──────────────────┴───────────────────────────┤
│              HTTP API Client                 │
│           (shared API client lib)            │
├──────────────────────────────────────────────┤
│              Backend (不变)                   │
│         /api/observe/*  /api/ingest/*        │
│         /api/ai/*       /api/config/*        │
├──────────────────────────────────────────────┤
│              Prisma + SQLite                 │
└──────────────────────────────────────────────┘
```

### 3.3 两种运行模式

#### 模式 A：TUI 交互模式（主模式）

```bash
$ cannbot-insight tui
# 或简写
$ cannbot-insight
```

启动全屏终端界面，类似 k9s/lazygit 的体验。

#### 模式 B：纯命令行模式（脚本友好）

```bash
# 列表
$ cannbot-insight sessions
$ cannbot-insight sessions --user guanxinghua --limit 20

# 详情
$ cannbot-insight session <taskId>
$ cannbot-insight session <taskId> --tab turns
$ cannbot-insight session <taskId> --tab workflow --format tree

# 搜索
$ cannbot-insight search <taskId> --keyword "bug"

# 导入
$ cannbot-insight import --source opencode-db --file ./sessions.db
$ cannbot-insight import --source claude-jsonl --dir ./logs/

# 对比
$ cannbot-insight compare <taskId1> <taskId2>

# 统计
$ cannbot-insight stats
$ cannbot-insight stats --session <taskId>

# 管道友好
$ cannbot-insight sessions --json | jq '.[] | .totalTokens'
```

---

## 4. 功能映射

### 4.1 Web → CLI 完整映射表

| Web 功能 | API 端点 | CLI TUI | CLI 命令 |
|----------|----------|---------|----------|
| **首页 — Session 列表** | `GET /api/observe/data` | Session 列表表格 | `sessions` |
| **首页 — 全局统计卡片** | `GET /api/observe/stats` | 顶部状态栏 | `stats` |
| **首页 — 文件导入** | `POST /api/ingest/import-file` | 导入面板 | `import` |
| **首页 — 会话预览** | `POST /api/ingest/import-file/sessions` | 导入预览列表 | `import --list` |
| **首页 — 删除会话** | `DELETE /api/ingest/delete-session` | 删除确认对话框 | `delete` |
| **首页 — Session 对比选择** | — | 多选 + Enter | `compare <id1> <id2>` |
| **Session — Overview Tab** | `GET /api/observe/stats?taskId=X` + `GET /api/observe/executions` | Overview 面板 | `session <id> --tab overview` |
| **Session — Turns Tab** | `GET /api/observe/session/turns` | Turn 时间线列表 | `session <id> --tab turns` |
| **Session — Workflow Tab** | `GET /api/observe/session/workflow` | ASCII 树形图 | `session <id> --tab workflow` |
| **Session — Trace Tab** | `GET /api/observe/session/turns/search` | 搜索结果面板 | `search <id> --keyword X` |
| **Session — Subagents Tab** | `GET /api/observe/executions` | Subagent 卡片列表 | `session <id> --tab subagents` |
| **Session — Skills Tab** | `GET /api/observe/session` (skills) | Skill 事件列表 | `session <id> --tab skills` |
| **Session — Interactions Tab** | `GET /api/observe/session/bridges` | 交互桥接表格 | `session <id> --tab bridges` |
| **Session — AI Workflow** | `POST /api/ai/analyze-workflow` | AI 分析输出 | `session <id> --ai-analyze` |
| **Session — Context Tab** | `GET /api/observe/session/turns` (contextWindowPct) | 上下文增长表 | `session <id> --tab context` |
| **Session — Turn 详情** | `GET /api/observe/session/turns/[turnId]` | Turn 详情面板 | `turn <taskId> <turnId>` |
| **Compare 页面** | 多次 observe API | 并排对比表格 | `compare <id1> <id2>` |
| **Context Window 配置** | `GET /api/config/context-windows` | 配置查看 | `config context-windows` |

### 4.2 TUI 布局设计

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 CANNBot-Insight v0.31          Sessions: 42  Cost: ¥12  │  ← 顶部状态栏
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─── Sessions ──────────────────────────────────────────┐  │
│  │ # │ Date       │ User    │ Turns │ Tokens │ Cost     │  │  ← 主表格
│  │ 1 │ 06-14 10:30│ guan... │  25   │  150K  │ ¥3.20   │  │
│  │ 2 │ 06-14 09:15│ guan... │  12   │   80K  │ ¥1.50   │  │
│  │ 3 │ 06-13 18:00│ guan... │  45   │  320K  │ ¥8.10   │  │
│  │ * │ ...        │         │       │        │         │  │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌─── Preview ──────────────────────────────────────────┐   │  ← 底部预览
│  │ Session: abc123...                                    │   │
│  │ First Query: "帮我实现 xxx 功能"                        │   │
│  │ Model: claude-3.5-sonnet  Duration: 45m              │   │
│  │ Tools: 120  Skills: 8  Subagents: 3                  │   │
│  └───────────────────────────────────────────────────────┘  │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ ↑↓ Navigate │ Enter: Detail │ i: Import │ d: Delete │ q: Quit │  ← 快捷键
└─────────────────────────────────────────────────────────────┘
```

### 4.3 TUI Session 详情页

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Session: abc123  │  claude-3.5-sonnet  │  45min  │ ¥3.20│
├─────────────────────────────────────────────────────────────┤
│ [Overview] [Turns] [Workflow] [Trace] [Subs] [Skills] [Ctx] │  ← Tab 切换
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Overview:                                                  │
│  ┌──────────┬──────────┬──────────┬──────────┐             │
│  │ Tokens   │ Cost     │ Duration │ LLM Calls│             │  ← 指标卡片
│  │  150K    │ ¥3.20   │   45m    │   25     │             │
│  └──────────┴──────────┴──────────┴──────────┘             │
│                                                             │
│  Token Breakdown:                                           │
│  Input:    ████████████░░░░  60K (40%)                     │  ← ASCII 柱状图
│  Output:   ██████░░░░░░░░░░  30K (20%)                     │
│  Reasoning:████████████████  80K (53%)                     │
│  Cache:    ███░░░░░░░░░░░░░  15K (10%)                     │
│                                                             │
│  Executions:                                                │
│  ┌────────────────┬───────┬──────┬───────┐                 │
│  │ Agent          │Tokens │ Cost │ Tools │                 │
│  │ root           │ 100K  │ ¥2.1 │  80   │                 │
│  │ subagent:debug │  50K  │ ¥1.1 │  40   │                 │
│  └────────────────┴───────┴──────┴───────┘                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│ Tab: Switch │ Enter: Drill-down │ Esc: Back │ q: Quit       │
└─────────────────────────────────────────────────────────────┘
```

### 4.4 Workflow Tab — ASCII 树形图

```
┌─ Workflow ──────────────────────────────────────────────────┐
│                                                             │
│  📁 Phase 1: 需求分析                                        │
│  ├── [Turn 1] user → assistant  (12K tokens, 5.2s)         │
│  │   └── 🔧 read_file × 3                                  │
│  ├── [Turn 2] user → assistant  (8K tokens, 3.1s)          │
│  │   ├── 🔧 search_files × 2                               │
│  │   └── 🔧 terminal × 1                                   │
│  │                                                          │
│  📁 Phase 2: 代码实现                                        │
│  ├── [Turn 3] user → assistant  (25K tokens, 12.5s)        │
│  │   ├── 🔧 write_file × 2                                 │
│  │   ├── 🔧 patch × 4                                      │
│  │   └── 🤖 → subagent:test-runner                         │
│  │       ├── [Turn 3.1]  (8K tokens, 4.2s)                 │
│  │       └── [Turn 3.2]  (6K tokens, 3.0s)                 │
│  │                                                          │
│  ✅ Checkpoint: 功能完成                                     │
│  │                                                          │
│  📁 Phase 3: 测试验证                                        │
│  ├── [Turn 4] user → assistant  (15K tokens, 8.0s)         │
│  │   ├── 🔧 terminal (npm test)                            │
│  │   └── 🔧 patch × 2                                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 4.5 Context Tab — 上下文增长追踪

```
┌─ Context Window Usage ──────────────────────────────────────┐
│                                                             │
│  Turn │ Tokens │ Context% │ Bar                            │
│  ─────┼────────┼──────────┼────────────────────────────────│
│    1  │  12K   │   6%     │ ██                             │
│    2  │  20K   │  10%     │ ███                            │
│    3  │  45K   │  22%     │ ██████                         │
│    4  │  80K   │  40%     │ ████████████                   │
│    5  │ 120K   │  60%     │ ██████████████████             │
│    6  │ 160K   │  80%     │ ████████████████████████ ⚠️    │
│    7  │ 195K   │  97%     │ ██████████████████████████🔴   │
│                                                             │
│  ⚠️ Turn 6: Context > 80%, approaching limit                │
│  🔴 Turn 7: Context > 95%, critical!                        │
│                                                             │
│  Model: claude-3.5-sonnet (200K window)                     │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. 命令设计

### 5.1 命令树

```
cannbot-insight (cbin)
├── tui                          # 启动 TUI 交互模式（默认）
├── sessions                     # 列出所有 session
│   ├── --user <name>            # 按用户过滤
│   ├── --limit <n>              # 限制数量
│   ├── --subagent               # 包含 subagent
│   └── --json                   # JSON 输出
├── session <taskId>             # 查看 session 详情
│   ├── --tab <name>             # 指定 tab (overview|turns|workflow|subagents|skills|bridges|context)
│   ├── --format <fmt>           # 输出格式 (table|tree|json|ascii)
│   └── --json                   # JSON 输出
├── turn <taskId> <turnId>       # 查看 turn 详情
│   └── --json                   # JSON 输出
├── search <taskId>              # 搜索 session 内 turn
│   ├── --keyword <kw>           # 搜索关键词
│   └── --json                   # JSON 输出
├── compare <id1> <id2>          # 对比两个 session
│   └── --json                   # JSON 输出
├── stats                        # 全局统计
│   ├── --session <taskId>       # 指定 session 统计
│   └── --json                   # JSON 输出
├── import                       # 导入 session
│   ├── --source <type>          # opencode-db | claude-jsonl
│   ├── --file <path>            # 文件路径
│   ├── --dir <path>             # 目录路径（递归扫描）
│   └── --list                   # 仅列出可导入的 session
├── delete                       # 删除 session
│   ├── --session <taskId>       # 删除指定 session
│   └── --all                    # 删除全部
├── config                       # 配置管理
│   └── context-windows          # 查看上下文窗口配置
└── --server <url>               # 指定后端地址 (默认 http://localhost:21025)
```

### 5.2 纯命令模式输出示例

```bash
$ cannbot-insight sessions --limit 5

  #   Date              User        Turns   Tokens    Cost      Model
  1   06-14 10:30       guanxing..  25      150.2K    ¥3.20     claude-3.5-sonnet
  2   06-14 09:15       guanxing..  12       80.5K    ¥1.50     glm-5
  3   06-13 18:00       guanxing..  45      320.0K    ¥8.10     deepseek-v4
  4   06-13 14:20       guanxing..   8       45.3K    ¥0.90     claude-3.5-sonnet
  5   06-12 22:10       guanxing..  33      210.7K    ¥5.40     gpt-4o

  Total: 42 sessions | 2.1M tokens | ¥56.30 cost
```

```bash
$ cannbot-insight stats

  📊 Global Statistics
  ──────────────────────────────
  Sessions:       42
  Total Tokens:   2,100,450
  Total Cost:     ¥56.30
  Avg Latency:    8.2s
  Total Duration: 18h 32m
```

```bash
$ cannbot-insight compare abc123 def456

  ┌─────────────┬──────────────┬──────────────┐
  │ Metric      │ abc123       │ def456       │
  ├─────────────┼──────────────┼──────────────┤
  │ Tokens      │ 150.2K       │ 80.5K        │
  │ Cost        │ ¥3.20        │ ¥1.50        │
  │ Duration    │ 45m          │ 22m          │
  │ Turns       │ 25           │ 12           │
  │ Tool Calls  │ 120          │ 45           │
  │ Subagents   │ 3            │ 1            │
  │ Skills      │ 8            │ 4            │
  │ Model       │ claude-3.5   │ glm-5        │
  └─────────────┴──────────────┴──────────────┘

  Winner: def456 (lower cost, fewer tokens, faster)
```

---

## 6. 快捷键设计（TUI 模式）

### 6.1 全局快捷键

| 键 | 功能 |
|----|------|
| `q` / `Ctrl+C` | 退出 |
| `?` / `F1` | 帮助 |
| `/` | 搜索/过滤 |
| `r` | 刷新数据 |
| `Tab` | 切换面板焦点 |
| `j` / `↓` | 向下 |
| `k` / `↑` | 向上 |

### 6.2 Session 列表页

| 键 | 功能 |
|----|------|
| `Enter` | 进入 Session 详情 |
| `Space` | 选中（用于对比） |
| `c` | 对比选中的两个 Session |
| `i` | 导入面板 |
| `d` | 删除选中 Session |
| `D` | 删除全部 |
| `s` | 切换排序列 |
| `f` | 过滤（按用户/模型） |

### 6.3 Session 详情页

| 键 | 功能 |
|----|------|
| `1-7` | 切换 Tab |
| `[` / `]` | 上一个/下一个 Tab |
| `Enter` | 钻取详情（Turn/Execution） |
| `Esc` | 返回列表 |
| `a` | AI 分析 |
| `y` | 复制当前 Session ID |

---

## 7. 目录结构

```
src/
├── app/                          # Web 前端（不变）
│   ├── page.tsx
│   ├── session/[taskId]/
│   ├── compare/
│   └── api/
├── lib/                          # 共享库（不变）
│   ├── shared/types.ts
│   ├── db.ts
│   ├── ingest/
│   └── ...
└── cli/                          # ✨ 新增 CLI 前端
    ├── index.ts                  # CLI 入口（commander 路由）
    ├── client.ts                 # API 客户端（封装 HTTP 调用）
    ├── config.ts                 # CLI 配置（server URL、颜色主题）
    │
    ├── commands/                 # 纯命令模式
    │   ├── sessions.ts           # sessions 命令
    │   ├── session.ts            # session <id> 命令
    │   ├── turn.ts               # turn 命令
    │   ├── search.ts             # search 命令
    │   ├── compare.ts            # compare 命令
    │   ├── stats.ts              # stats 命令
    │   ├── import.ts             # import 命令
    │   ├── delete.ts             # delete 命令
    │   └── config.ts             # config 命令
    │
    ├── tui/                      # TUI 交互模式
    │   ├── App.tsx               # TUI 根组件（Ink）
    │   ├── screens/
    │   │   ├── SessionList.tsx   # Session 列表屏
    │   │   ├── SessionDetail.tsx # Session 详情屏
    │   │   ├── TurnDetail.tsx    # Turn 详情屏
    │   │   ├── CompareView.tsx   # 对比屏
    │   │   ├── ImportPanel.tsx   # 导入面板
    │   │   └── HelpScreen.tsx    # 帮助屏
    │   ├── tabs/
    │   │   ├── OverviewTab.tsx   # Overview Tab
    │   │   ├── TurnsTab.tsx      # Turns Tab
    │   │   ├── WorkflowTab.tsx   # Workflow Tab (ASCII tree)
    │   │   ├── SubagentsTab.tsx  # Subagents Tab
    │   │   ├── SkillsTab.tsx     # Skills Tab
    │   │   ├── BridgesTab.tsx    # Interactions Tab
    │   │   └── ContextTab.tsx    # Context Tab
    │   └── components/
│       ├── StatusBar.tsx     # 顶部状态栏
│       ├── KeyBar.tsx        # 底部快捷键提示
│       ├── DataTable.tsx     # 通用数据表格
│       ├── MetricCards.tsx   # 指标卡片行
│       ├── AsciiBar.tsx      # ASCII 柱状图
│       ├── TreeView.tsx      # ASCII 树形图
│       ├── TabBar.tsx        # Tab 切换栏
│       ├── ConfirmDialog.tsx # 确认对话框
│       ├── Spinner.tsx       # 自写加载动画（替代 ink-spinner）
│       ├── TextInput.tsx     # 自写输入框（替代 ink-text-input）
    │
    ├── hooks/                    # TUI 自定义 hooks
    │   ├── useApi.ts             # API 请求 hook
    │   ├── useKeyboard.ts        # 键盘事件 hook
    │   ├── useNavigation.ts      # 屏幕导航 hook
    │   └── useTable.ts           # 表格排序/翻页 hook
    │
    └── utils/                    # CLI 工具函数
        ├── format.ts             # 数字/时间/货币格式化
        ├── colors.ts             # 终端颜色主题
        └── table.ts              # 表格对齐/截断
```

---

## 8. 分期计划

### Phase 1：纯命令模式（MVP）

**目标**：可在脚本和终端中快速查询数据

| 任务 | 对应 API | 优先级 |
|------|----------|--------|
| CLI 入口 + Commander 框架 | — | P0 |
| API Client 封装 | 全部 15 个 | P0 |
| `sessions` 列表命令 | `/api/observe/data` | P0 |
| `session <id>` 详情命令 | `/api/observe/session` + `stats` + `executions` | P0 |
| `stats` 全局统计 | `/api/observe/stats` | P0 |
| `turn <id>` 详情 | `/api/observe/session/turns/[turnId]` | P1 |
| `search` 搜索 | `/api/observe/session/turns/search` | P1 |
| `compare` 对比 | 多次 observe API | P1 |
| `import` 导入 | `/api/ingest/import-file` | P1 |
| `delete` 删除 | `/api/ingest/delete-session` | P2 |
| `--json` 输出模式 | — | P1 |
| `--server` 参数 | — | P0 |

**预估工作量**：3-5 天

### Phase 2：TUI 交互模式

**目标**：全屏终端 UI，k9s/lazygit 风格体验

| 任务 | 优先级 |
|------|--------|
| Ink 框架 + App 壳 + 屏幕导航（ESM + tsx） | P0 |
| StatusBar + KeyBar 组件 | P0 |
| Session 列表屏（表格、排序、过滤） | P0 |
| Session 详情屏（Tab 切换） | P0 |
| Overview Tab（指标卡片 + ASCII 柱状图） | P0 |
| Turns Tab（时间线列表 + 详情钻取） | P0 |
| Workflow Tab（ASCII 树形图） | P1 |
| Subagents Tab（卡片列表） | P1 |
| Context Tab（上下文增长表） | P1 |
| Import 面板 | P2 |
| Compare 屏 | P2 |
| Skills / Bridges Tab | P2 |
| AI Workflow 分析触发 | P2 |

**预估工作量**：7-10 天

### Phase 3：增强特性

| 特性 | 说明 |
|------|------|
| **实时刷新** | `--watch` 模式，自动轮询新 session |
| **管道集成** | stdin 管道输入 session ID |
| **主题配置** | 支持暗色/亮色终端主题 |
| **导出** | 导出为 CSV/Markdown/HTML |
| **Shell 补全** | bash/zsh/fish 自动补全 |
| **Alias** | `cbin` 短命令别名 |
| **多服务器** | 配置文件中管理多个后端地址 |

---

## 9. 技术要点

### 9.1 API Client 设计

```typescript
// src/cli/client.ts
export class InsightClient {
  constructor(private baseUrl: string = 'http://localhost:21025') {}

  // Sessions
  async listSessions(opts: { page?: number; pageSize?: number; user?: string }) { ... }
  async getSession(taskId: string) { ... }
  async getStats(taskId?: string) { ... }
  async getExecutions(taskId: string) { ... }
  async getTurns(taskId: string, opts?: { isSubagent?: boolean; role?: string }) { ... }
  async getTurnDetail(turnId: string) { ... }
  async searchTurns(taskId: string, keyword: string) { ... }
  async getWorkflow(taskId: string) { ... }
  async getBridges(taskId: string) { ... }

  // Ingest
  async importSession(source: string, filePath: string) { ... }
  async listImportableSessions(source: string, filePath: string) { ... }
  async deleteSession(taskId?: string, deleteAll?: boolean) { ... }

  // AI
  async analyzeWorkflow(taskId: string, provider: AIProvider) { ... }

  // Config
  async getContextWindows(model?: string) { ... }
}
```

### 9.2 共享类型复用

CLI 直接复用 `src/lib/shared/types.ts` 中的类型定义，无需重新定义：
- `SessionListItem`
- `TokenUsage`
- `ToolCallInfo`
- `RawInteraction`

> CLI 专用 API response 类型（字段结构与 shared 内部类型不同）在 `src/cli/types.ts` 中定义，加 `Api` 前缀与 shared 区分。重叠子结构（如 TokenUsage）从 shared 导入复用。

### 9.3 版本号来源

CLI 版本号从 `src/lib/version.ts` 导入，不硬编码：

```typescript
import { VERSION, VERSION_DISPLAY } from '@/lib/version';
// Commander: program.version(VERSION)
// StatusBar: <Text> {VERSION_DISPLAY}</Text>
```

### 9.4 TUI stdin 原始模式切换

TUI 需要接管键盘输入，必须将 stdin 切换到原始模式。生命周期管理：

```typescript
// 进入 TUI 前
process.stdin.resume();
process.stdin.setRawMode(true);

// Ink render 配置
render(<App />, {
  exitOnCtrlC: false,      // 禁用 Ink 默认 Ctrl+C 退出（自己处理）
  patchConsole: false,     // 不 patch console（避免干扰 API 日志）
});

// 退出 TUI 时恢复
instance.unmount();
process.stdin.setRawMode(false);
process.stdin.pause();
```

> **关键点**: `exitOnCtrlC: false` 让我们自己处理退出逻辑（清理状态、恢复 stdin）；`patchConsole: false` 避免 Ink 劫持 console.log 导致 API 调试信息丢失。

### 9.5 中文宽度处理

中文字符在终端占 2 列宽，Ink `<Text width={N}>` 按字符数计算导致错位。解决方案：

```typescript
import stringWidth from 'string-width';

export function padEndVisual(str: string, width: number): string {
  const visualWidth = stringWidth(str);
  const padding = width - visualWidth;
  if (padding <= 0) return truncateVisual(str, width);
  return str + ' '.repeat(padding);
}

export function truncateVisual(str: string, maxVisualWidth: number): string {
  if (!str) return '—';
  if (stringWidth(str) <= maxVisualWidth) return str;
  let result = '';
  let currentWidth = 0;
  for (const char of str) {
    const charWidth = stringWidth(char);
    if (currentWidth + charWidth > maxVisualWidth - 1) {
      return result + '…';
    }
    result += char;
    currentWidth += charWidth;
  }
  return result;
}
```

### 9.6 格式化示例

```typescript
// src/cli/utils/format.ts
export function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

export function formatCost(cost: number): string {
  return `¥${cost.toFixed(2)}`;
}

export function formatDuration(seconds: number): string {
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(0)}m`;
  return `${seconds.toFixed(1)}s`;
}

export function formatDate(iso: string): string {
  const d = new Date(iso);
  return `${(d.getMonth()+1).toString().padStart(2,'0')}-${d.getDate().toString().padStart(2,'0')} ${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
}
```

---

## 10. 安装与使用

### 10.1 安装

```bash
# 全局安装
npm install -g cannbot-insight

# 或 npx 直接使用
npx cannbot-insight sessions
```

### 10.2 配置

```bash
# 首次使用，配置后端地址
cannbot-insight config set server http://localhost:21025

# 或环境变量
export CANNBOT_SERVER=http://192.168.1.100:21025
```

### 10.3 package.json 脚本

```json
{
  "type": "module",
  "scripts": {
    "cli": "tsx src/cli/index.ts",
    "tui": "tsx src/cli/index.ts tui"
  }
}
```

> **ESM + tsx**: package.json `"type": "module"` 是 Ink v7 的硬性要求（yoga-layout 使用 top-level await，CJS 模式报错）。运行工具必须用 tsx，不能用 ts-node。

---

## 11. 风险与决策

| 风险 | 影响 | 应对 |
|------|------|------|
| Ink 在 Windows Terminal 兼容性 | 部分 ANSI 转义码不支持 | 测试 Windows Terminal + WSL，降级渲染 |
| TUI 大表格性能 | 1000+ session 列表卡顿 | 虚拟滚动 + 分页加载 |
| 后端需要运行 | CLI 不能独立工作 | 文档明确说明依赖关系，提供 `cbin server` 启动命令 |
| 命令与 npm script 冲突 | `npx cannbot-insight` 可能冲突 | 提供 `cbin` 短别名 |
| Ink v7 第三方组件不兼容 | ink-table/ink-spinner 等不兼容 v7 | 全部自实现（Spinner 10行、TextInput 30行、DataTable 已有设计） |

---

## 12. 成功标准

| 指标 | 目标 |
|------|------|
| API 覆盖率 | 15/15 端点全部覆盖 |
| 命令模式可用性 | 所有命令可在 3 秒内返回结果 |
| TUI 可用性 | Session 列表 → 详情 → 返回，全程无需鼠标 |
| 管道友好 | 所有命令支持 `--json` 输出 |
| 零后端改动 | 后端代码无任何修改 |
| 测试覆盖 | CLI 命令单元测试 > 80% |
