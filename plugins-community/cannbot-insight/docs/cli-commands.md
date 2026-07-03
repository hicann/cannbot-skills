# CLI 命令参考文档

> **版本**: v0.36  
> **更新日期**: 2026-06-15  
> **技术栈**: Ink v7.0.6 + React 19.2.7 + Commander.js + ESM

---

## 目录

- [快速开始](#快速开始)
- [全局选项](#全局选项)
- [TUI 交互模式](#tui-交互模式)
- [命令参考](#命令参考)
  - [sessions - 列出会话](#sessions)
  - [session - 查看会话详情](#session)
  - [stats - 统计信息](#stats)
  - [compare - 对比会话](#compare)
  - [search - 搜索轮次](#search)
  - [turn - 查看轮次详情](#turn)
  - [import - 导入会话](#import)
  - [delete - 删除会话](#delete)
  - [config - 配置管理](#config)
  - [analyze - AI 工作流分析](#analyze)
- [环境变量](#环境变量)
- [配置文件](#配置文件)
- [高级用法](#高级用法)

---

## 快速开始

### 安装依赖

```bash
npm install
```

### 启动 TUI 模式

```bash
# 方式 1：直接使用 tsx 运行
npx tsx src/cli/index.ts tui

# 方式 2：使用 npm script
npm run cli tui
```

### 使用命令模式

```bash
# 列出所有会话
npx tsx src/cli/index.ts sessions

# 查看统计
npx tsx src/cli/index.ts stats

# 搜索特定内容
npx tsx src/cli/index.ts search <taskId> --keyword "implement login"
```

---

## 全局选项

所有命令都支持以下全局选项：

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--server <url>` | Insight API 服务器地址 | `http://localhost:21025` |
| `--timeout <ms>` | API 请求超时时间（毫秒） | `30000` |
| `--help` | 显示帮助信息 | - |
| `--version` | 显示版本号 | - |

**示例**：

```bash
# 连接到远程服务器
npx tsx src/cli/index.ts sessions --server http://192.168.1.100:21025

# 设置更长的超时时间
npx tsx src/cli/index.ts compare task1 task2 --timeout 60000
```

---

## TUI 交互模式

TUI（Terminal User Interface）模式提供全屏交互界面，基于 Ink v7 + React 19 构建。

### 启动

```bash
npx tsx src/cli/index.ts tui
```

### 屏幕导航

TUI 包含 6 个主要屏幕：

1. **Session 列表屏** - 浏览所有会话
2. **Session 详情屏** - 查看单个会话的详细信息
3. **Turn 详情屏** - 查看单轮对话的完整内容
4. **对比屏** - 并排对比两个会话
5. **导入面板** - 从外部数据源导入会话（TUI 模式可交互输入路径）
6. **帮助屏** - 显示快捷键说明

### 快捷键

#### 全局快捷键

| 快捷键 | 功能 |
|--------|------|
| `q` | 退出 TUI |
| `Ctrl+C` | 退出 TUI |
| `?` | 显示帮助 |
| `Esc` | 返回上一屏 / 关闭搜索 |

#### Session 列表屏快捷键

| 快捷键 | 功能 |
|--------|------|
| `↑↓` | 上下移动选择 |
| `Enter` | 打开会话详情 |
| `n` | 下一页 |
| `p` | 上一页 |
| `/` | 进入/退出搜索模式 |
| `Space` | 标记会话用于对比 |
| `c` | 对比两个已标记的会话 |
| `i` | 打开导入面板 |
| `d` | 删除当前选中的会话 |
| `r` | 刷新数据 |

#### 导航快捷键

| 快捷键 | 功能 |
|--------|------|
| `↑↓` | 向上/向下移动 |
| `Enter` | 选择 / 进入详情 |
| `Esc` | 返回 / 取消 |
| `Tab` | 切换 Tab（Session 详情屏）

### Session 详情 Tab 说明

#### 1. Overview Tab
显示会话概览信息：
- 基本信息（ID、模型、用户、时间范围）
- Token 统计（总输入/输出、缓存命中率）
- 成本统计
- 性能指标（总耗时、平均轮次耗时）
- 工具调用统计
- Subagent 列表

#### 2. Turns Tab
显示对话轮次列表：
- 每轮的角色（user/assistant）
- 内容摘要
- Token 使用量
- 耗时
- 工具调用数量
- 支持分页浏览（每页 20 条）

#### 3. Workflow Tab
显示工作流信息：
- 阶段划分
- 工具调用序列
- 并行执行的任务

#### 4. Subagents Tab
显示子代理信息：
- 子代理 ID
- 执行状态
- Token 使用量
- 耗时
- 工具调用统计

#### 5. Skills Tab
显示技能调用信息：
- 加载的技能列表
- 技能调用次数
- 技能执行耗时

#### 6. Bridges Tab
显示跨会话桥接信息：
- 父会话 → 子会话的关系
- 桥接类型（subagent/session_spawn）
- 桥接时间

#### 7. Context Tab
显示上下文使用情况：
- 每轮的 context window 使用百分比
- 上下文增长趋势图（ASCII 柱状图）
- 警告：当使用率 > 80% 时高亮显示

---

## 命令参考

### sessions

列出所有会话，支持过滤和分页。

#### 语法

```bash
npx tsx src/cli/index.ts sessions [选项]
```

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--limit <n>` | 限制返回数量 | `20` |
| `--offset <n>` | 跳过前 N 条记录 | `0` |
| `--user <name>` | 按用户过滤 | - |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 列出前 20 个会话
npx tsx src/cli/index.ts sessions

# 列出 50 个会话，跳过前 20 个
npx tsx src/cli/index.ts sessions --limit 50 --offset 20

# 只列出特定用户的会话
npx tsx src/cli/index.ts sessions --user guanxinghua

# 输出 JSON 格式，便于程序处理
npx tsx src/cli/index.ts sessions --json

# 结合 jq 进行高级过滤
npx tsx src/cli/index.ts sessions --json | jq '.[] | select(.totalTokens > 100000)'
npx tsx src/cli/index.ts sessions --json | jq '.[] | {taskId, model, totalCost}'
```

#### 输出格式

**表格模式**（默认）：

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ #  │ Task ID              │ Model              │ User        │ Turns │ Tokens   │ Cost   │ Time              │
├────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ 1  │ task_abc123          │ claude-3.5-sonnet  │ guanxinghua │ 25    │ 150.2K   │ ¥3.20  │ 06-14 10:30:15    │
│ 2  │ task_def456          │ gpt-4o             │ guanxinghua │ 12    │ 80.5K    │ ¥1.50  │ 06-14 09:15:22    │
│ 3  │ task_ghi789          │ claude-3.5-sonnet  │ zhangsan    │ 45    │ 320.0K   │ ¥8.10  │ 06-13 18:00:45    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
  显示 1-3 条，共 150 条记录
  总 tokens: 550.7K
  总成本: ¥12.80
```

**JSON 模式**：

```json
[
  {
    "taskId": "task_abc123",
    "model": "claude-3.5-sonnet",
    "user": "guanxinghua",
    "turns": 25,
    "totalTokens": 150200,
    "totalCost": 3.20,
    "startTime": "2026-06-14T10:30:15.000Z",
    "endTime": "2026-06-14T10:45:30.000Z"
  },
  ...
]
```

---

### session

查看单个会话的详细信息。

#### 语法

```bash
npx tsx src/cli/index.ts session <taskId> [选项]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `taskId` | 会话 ID（必填） |

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 查看会话详情
npx tsx src/cli/index.ts session task_abc123

# 输出 JSON 格式
npx tsx src/cli/index.ts session task_abc123 --json
```

#### 输出格式

**表格模式**（默认）：

```
Session: task_abc123
══════════════════════════════════════════════════════════════

基本信息
────────────────────────────────────────────────────────────
  模型:       claude-3.5-sonnet
  用户:       guanxinghua
  开始时间:   2026-06-14 10:30:15
  结束时间:   2026-06-14 10:45:30
  持续时间:   15m 15s

Token 统计
────────────────────────────────────────────────────────────
  总 tokens:      150,200
  输入 tokens:    80,500  (53.6%)
  输出 tokens:    45,300  (30.2%)
  缓存 tokens:    24,400  (16.2%)
  缓存命中率:     35.1%

成本统计
────────────────────────────────────────────────────────────
  总成本:         ¥3.20
  每千 tokens:    ¥0.021

性能指标
────────────────────────────────────────────────────────────
  总轮次:         25
  总耗时:         915,000 ms
  平均轮次耗时:   36,600 ms

工具调用统计
────────────────────────────────────────────────────────────
  总调用次数:     120
  成功:           115  (95.8%)
  失败:           5    (4.2%)

Subagent 列表
────────────────────────────────────────────────────────────
┌────────────────────────────────────────────────────────────┐
│ Subagent ID        │ Status │ Tokens │ Cost  │ Time       │
├────────────────────────────────────────────────────────────┤
│ subagent_xyz789    │ ✓      │ 45.2K  │ ¥0.95 │ 5m 30s     │
│ subagent_def456    │ ✓      │ 32.1K  │ ¥0.68 │ 3m 15s     │
└────────────────────────────────────────────────────────────┘
```

**JSON 模式**：

```json
{
  "taskId": "task_abc123",
  "model": "claude-3.5-sonnet",
  "user": "guanxinghua",
  "startTime": "2026-06-14T10:30:15.000Z",
  "endTime": "2026-06-14T10:45:30.000Z",
  "totalTokens": 150200,
  "totalInputTokens": 80500,
  "totalOutputTokens": 45300,
  "totalCacheTokens": 24400,
  "cacheHitRate": 0.351,
  "totalCost": 3.20,
  "totalLatencyMs": 915000,
  "turns": 25,
  "avgTurnLatencyMs": 36600,
  "toolCalls": 120,
  "toolCallSuccess": 115,
  "toolCallFailed": 5,
  "subagents": [
    {
      "subagentId": "subagent_xyz789",
      "status": "completed",
      "tokens": 45200,
      "cost": 0.95,
      "latencyMs": 330000
    },
    ...
  ]
}
```

---

### stats

查看统计信息，支持全局统计和单会话统计。

#### 语法

```bash
npx tsx src/cli/index.ts stats [选项]
```

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--session <taskId>` | 查看指定会话的统计 | - |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 查看全局统计
npx tsx src/cli/index.ts stats

# 查看指定会话的统计
npx tsx src/cli/index.ts stats --session task_abc123

# 输出 JSON 格式
npx tsx src/cli/index.ts stats --json
```

#### 输出格式

**全局统计**（表格模式）：

```
全局统计
══════════════════════════════════════════════════════════════

  总会话数:       150
  总轮次:         3,450
  
Token 统计
────────────────────────────────────────────────────────────
  总 tokens:      12,500,000
  总输入 tokens:  7,200,000  (57.6%)
  总输出 tokens:  3,800,000  (30.4%)
  总缓存 tokens:  1,500,000  (12.0%)
  平均缓存命中率: 28.3%

成本统计
────────────────────────────────────────────────────────────
  总成本:         ¥285.50
  平均每会话:     ¥1.90
  平均每千 tokens: ¥0.023

性能指标
────────────────────────────────────────────────────────────
  总耗时:         85,500,000 ms (23.75 小时)
  平均会话耗时:   570,000 ms (9.5 分钟)
  平均轮次耗时:   24,783 ms

模型使用分布
────────────────────────────────────────────────────────────
  claude-3.5-sonnet:  85 会话  (56.7%)
  gpt-4o:             45 会话  (30.0%)
  glm-5:              20 会话  (13.3%)
```

**单会话统计**（表格模式）：

```
会话统计: task_abc123
══════════════════════════════════════════════════════════════

  轮次:           25

Token 统计
────────────────────────────────────────────────────────────
  总 tokens:      150,200
  输入 tokens:    80,500  (53.6%)
  输出 tokens:    45,300  (30.2%)
  缓存 tokens:    24,400  (16.2%)
  缓存命中率:     35.1%

成本统计
────────────────────────────────────────────────────────────
  总成本:         ¥3.20
  每千 tokens:    ¥0.021

性能指标
────────────────────────────────────────────────────────────
  总耗时:         915,000 ms (15.25 分钟)
  平均轮次耗时:   36,600 ms

工具调用统计
────────────────────────────────────────────────────────────
  总调用次数:     120
  成功:           115  (95.8%)
  失败:           5    (4.2%)
```

---

### compare

并排对比两个会话的各项指标。

#### 语法

```bash
npx tsx src/cli/index.ts compare <taskId1> <taskId2> [选项]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `taskId1` | 第一个会话 ID（必填） |
| `taskId2` | 第二个会话 ID（必填） |

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 对比两个会话
npx tsx src/cli/index.ts compare task_abc123 task_def456

# 输出 JSON 格式
npx tsx src/cli/index.ts compare task_abc123 task_def456 --json
```

#### 输出格式

**表格模式**（默认）：

```
会话对比
══════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────┐
│ 指标               │ task_abc123      │ task_def456      │
├────────────────────────────────────────────────────────────┤
│ 模型               │ claude-3.5-sonnet│ gpt-4o           │
│ 用户               │ guanxinghua      │ guanxinghua      │
├────────────────────────────────────────────────────────────┤
│ 轮次               │ 25               │ 12               │
│ 总 tokens          │ 150.2K           │ 80.5K            │
│ 输入 tokens        │ 80.5K (53.6%)    │ 45.2K (56.1%)    │
│ 输出 tokens        │ 45.3K (30.2%)    │ 25.8K (32.0%)    │
│ 缓存 tokens        │ 24.4K (16.2%)    │ 9.5K (11.8%)     │
│ 缓存命中率         │ 35.1%            │ 20.1%            │
├────────────────────────────────────────────────────────────┤
│ 总成本             │ ¥3.20            │ ¥1.50            │
│ 每千 tokens        │ ¥0.021           │ ¥0.019           │
├────────────────────────────────────────────────────────────┤
│ 总耗时             │ 15m 15s          │ 8m 30s           │
│ 平均轮次耗时       │ 36.6s            │ 42.5s            │
├────────────────────────────────────────────────────────────┤
│ 工具调用次数       │ 120              │ 45               │
│ 工具成功率         │ 95.8%            │ 97.8%            │
├────────────────────────────────────────────────────────────┤
│ Subagent 数量      │ 2                │ 0                │
└────────────────────────────────────────────────────────────┘

差异分析
────────────────────────────────────────────────────────────
  task_abc123 比 task_def456:
    ✓ 多 13 轮对话 (+108%)
    ✓ 多使用 69.7K tokens (+87%)
    ✗ 多花费 ¥1.70 (+113%)
    ✗ 多耗时 6m 45s (+79%)
    ✓ 缓存命中率更高 (+15.0%)
```

**JSON 模式**：

```json
{
  "session1": {
    "taskId": "task_abc123",
    "model": "claude-3.5-sonnet",
    "turns": 25,
    "totalTokens": 150200,
    "totalCost": 3.20,
    "totalLatencyMs": 915000
  },
  "session2": {
    "taskId": "task_def456",
    "model": "gpt-4o",
    "turns": 12,
    "totalTokens": 80500,
    "totalCost": 1.50,
    "totalLatencyMs": 510000
  },
  "comparison": {
    "turnsDiff": 13,
    "turnsDiffPercent": 108,
    "tokensDiff": 69700,
    "tokensDiffPercent": 87,
    "costDiff": 1.70,
    "costDiffPercent": 113,
    "latencyDiffMs": 405000,
    "latencyDiffPercent": 79
  }
}
```

---

### search

在指定会话中搜索轮次内容。

#### 语法

```bash
npx tsx src/cli/index.ts search <taskId> --keyword <text> [选项]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `taskId` | 会话 ID（必填） |

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--keyword <text>` | 搜索关键词（必填） | - |
| `--limit <n>` | 限制返回结果数量 | `50` |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 搜索包含 "implement login" 的轮次
npx tsx src/cli/index.ts search task_abc123 --keyword "implement login"

# 搜索包含 "bug" 的轮次，限制 20 条结果
npx tsx src/cli/index.ts search task_abc123 --keyword "bug" --limit 20

# 输出 JSON 格式
npx tsx src/cli/index.ts search task_abc123 --keyword "error" --json
```

#### 输出格式

**表格模式**（默认）：

```
搜索结果: task_abc123
关键词: "implement login"
══════════════════════════════════════════════════════════════

找到 5 条匹配结果

┌────────────────────────────────────────────────────────────┐
│ Turn │ Role      │ Content Preview                    │ Time              │
├────────────────────────────────────────────────────────────┤
│ 3    │ user      │ Please implement login feature...  │ 10:32:15          │
│ 4    │ assistant │ I'll implement the login feature...│ 10:32:45          │
│ 8    │ user      │ The login implementation looks...  │ 10:38:20          │
│ 12   │ assistant │ Updated login handler with error...│ 10:42:10          │
│ 18   │ user      │ Test the login flow...             │ 10:48:30          │
└────────────────────────────────────────────────────────────┘
```

**JSON 模式**：

```json
{
  "taskId": "task_abc123",
  "keyword": "implement login",
  "totalMatches": 5,
  "matches": [
    {
      "turnIndex": 3,
      "role": "user",
      "content": "Please implement login feature with OAuth support",
      "timestamp": "2026-06-14T10:32:15.000Z",
      "matchScore": 0.95
    },
    ...
  ]
}
```

---

### turn

查看指定会话中某一轮次的完整详情。

#### 语法

```bash
npx tsx src/cli/index.ts turn <taskId> <turnId> [选项]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `taskId` | 会话 ID（必填） |
| `turnId` | 轮次 ID（必填） |

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 查看某一轮次的详情
npx tsx src/cli/index.ts turn task_abc123 turn_007

# 输出 JSON 格式
npx tsx src/cli/index.ts turn task_abc123 turn_007 --json
```

#### 输出格式

**表格模式**（默认）：

```
Turn #7 in session task_abc123
══════════════════════════════════════════════════════════════

基本信息
────────────────────────────────────────────────────────────
  角色:       assistant
  模型:       claude-3.5-sonnet
  时间:       2026-06-14 10:35:22
  耗时:       8,200 ms
  完成原因:   end_turn

Token 统计
────────────────────────────────────────────────────────────
  总 tokens:      12,500
  输入 tokens:    8,000  (64.0%)
  输出 tokens:    3,200  (25.6%)
  推理 tokens:    1,300  (10.4%)

上下文使用
────────────────────────────────────────────────────────────
  输入消息数:     15
  输入 token 数:  8,000
  上下文占比:     4.0%

工具调用 (3 次)
────────────────────────────────────────────────────────────
  1. read_file     ✓ success   120ms
  2. write_file    ✓ success   350ms
  3. terminal      ✓ success   2,100ms
```

---

### import

从外部数据源导入会话。

#### 语法

```bash
npx tsx src/cli/index.ts import [选项]
```

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--source <type>` | 数据源类型: opencode-db / claude-jsonl | - |
| `--file <path>` | 文件路径 | - |
| `--dir <path>` | 目录路径（递归扫描） | - |
| `--list` | 仅列出可导入的会话 | `false` |
| `--session-id <id>` | 导入指定会话 | - |
| `--all` | 导入所有会话 | `false` |
| `--yes` | 跳过确认提示 | `false` |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 列出可导入的会话（Opencode DB）
npx tsx src/cli/index.ts import --source opencode-db --file ./sessions.db --list

# 列出可导入的会话（Claude JSONL 目录）
npx tsx src/cli/index.ts import --source claude-jsonl --dir ./claude-sessions/ --list

# 导入指定会话
npx tsx src/cli/index.ts import --source opencode-db --file ./sessions.db --session-id ses_001

# 导入所有会话（跳过确认）
npx tsx src/cli/index.ts import --source opencode-db --file ./sessions.db --all --yes

# 交互式选择导入
npx tsx src/cli/index.ts import --source claude-jsonl --dir ./logs/
```

#### 输出格式

**列表模式**（`--list`）：

```
可导入的会话 (opencode-db: ./sessions.db)
══════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────┐
│ #  │ Session ID    │ First Query        │ Turns │ Model   │
├────────────────────────────────────────────────────────────┤
│ 1  │ ses_001       │ implement login... │ 25    │ claude  │
│ 2  │ ses_002       │ fix database...    │ 12    │ gpt-4o  │
│ 3  │ ses_003       │ refactor code...   │ 45    │ claude  │
└────────────────────────────────────────────────────────────┘
  共 3 个会话可导入
```

**导入结果**：

```
导入完成
────────────────────────────────────────────────────────────
  ✓ ses_001 — 25 turns imported
  ✓ ses_002 — 12 turns imported
  ⏭ ses_003 — skipped (already exists)
  
  成功: 2  跳过: 1  失败: 0
```

---

### delete

删除已导入的会话。

#### 语法

```bash
npx tsx src/cli/index.ts delete [选项]
```

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--session <taskId>` | 删除指定会话 | - |
| `--all` | 删除所有会话 | `false` |
| `--yes` | 跳过确认提示 | `false` |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 删除指定会话（会提示确认）
npx tsx src/cli/index.ts delete --session task_abc123

# 删除指定会话（跳过确认）
npx tsx src/cli/index.ts delete --session task_abc123 --yes

# 删除所有会话（跳过确认）
npx tsx src/cli/index.ts delete --all --yes
```

#### 输出格式

```
⚠️  即将删除会话: task_abc123
  模型: claude-3.5-sonnet
  轮次: 25
  Tokens: 150.2K

确认删除？(y/N) y

✓ 已删除 1 个会话
```

---

### config

管理 CLI 配置。

#### 语法

```bash
npx tsx src/cli/index.ts config <subcommand> [参数]
```

#### 子命令

| 子命令 | 说明 |
|--------|------|
| `config get <key>` | 获取配置项 |
| `config set <key> <value>` | 设置配置项 |
| `config list` | 列出所有配置 |
| `config reset` | 重置为默认配置 |

#### 示例

```bash
# 查看所有配置
npx tsx src/cli/index.ts config list

# 获取服务器地址
npx tsx src/cli/index.ts config get server

# 设置服务器地址
npx tsx src/cli/index.ts config set server http://192.168.1.100:21025

# 设置超时时间
npx tsx src/cli/index.ts config set timeout 60000

# 设置主题
npx tsx src/cli/index.ts config set theme dark

# 重置为默认配置
npx tsx src/cli/index.ts config reset
```

#### 输出格式

**config list**：

```
CLI 配置 (~/.cannbot-insight/config.json)
══════════════════════════════════════════════════════════════

  server:      http://localhost:21025
  timeout:     15000
  theme:       dark
  authToken:   (未设置)

快捷键:
  quit:        q
  help:        ?
  search:      /
  refresh:     r
  navigateUp:  k
  navigateDown: j
```

**config get**：

```
server = http://localhost:21025
```

**config set**：

```
✓ Set server = http://192.168.1.100:21025
```

---

### analyze

使用 AI 对指定会话进行工作流分析。

#### 语法

```bash
npx tsx src/cli/index.ts analyze <taskId> --base-url <url> --api-key <key> [选项]
```

#### 参数

| 参数 | 说明 |
|------|------|
| `taskId` | 会话 ID（必填） |

#### 选项

| 选项 | 说明 | 默认值 |
|------|------|--------|
| `--base-url <url>` | AI Provider API 地址（必填） | - |
| `--api-key <key>` | AI Provider API Key（必填） | - |
| `--model <model>` | AI 模型名称 | `gpt-4o-mini` |
| `--json` | 输出 JSON 格式 | `false` |

#### 示例

```bash
# 使用 OpenAI 分析工作流
npx tsx src/cli/index.ts analyze task_abc123 --base-url https://api.openai.com/v1 --api-key sk-...

# 使用自定义模型
npx tsx src/cli/index.ts analyze task_abc123 --base-url https://api.openai.com/v1 --api-key sk-... --model gpt-4o

# 输出 JSON 格式
npx tsx src/cli/index.ts analyze task_abc123 --base-url https://api.openai.com/v1 --api-key sk-... --json
```

#### 输出格式

**表格模式**（默认）：

```
AI Workflow Analysis: task_abc123
══════════════════════════════════════════════════════════════

  Provider:  https://api.openai.com/v1
  Model:     gpt-4o-mini

Workflow Analysis Result
══════════════════════════════════════════════════════════════

  Phases:       3
  Steps:        12
  Checkpoints:  1
  Active Time:  8m 30s
  Wait Time:    2m 15s
  Active %:     79%

Phase 0: analysis
  Duration: 2m 30s │ Tokens: 15.0K │ Cost: $0.03 │ Tool Calls: 5

  Step 0: Read files (completed)
    Duration: 45s │ Tokens: 5.0K
  Step 1: Analyze code (completed)
    Duration: 1m 45s │ Tokens: 10.0K
```

---

## 环境变量

可以通过环境变量配置 CLI 行为：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `CANNBOT_SERVER` | Insight API 服务器地址 | `http://localhost:21025` |
| `CANNBOT_TIMEOUT` | API 请求超时时间（毫秒） | `30000` |
| `CANNBOT_TOKEN` | 认证 Token（预留） | - |

**示例**：

```bash
# 设置环境变量
export CANNBOT_SERVER=http://192.168.1.100:21025
export CANNBOT_TIMEOUT=60000

# 使用环境变量
npx tsx src/cli/index.ts sessions
```

---

## 配置文件

CLI 支持通过配置文件进行持久化设置。

### 配置文件位置

```
~/.cannbot-insight/config.json
```

### 配置文件格式

```json
{
  "server": "http://localhost:21025",
  "timeout": 30000,
  "theme": "dark",
  "keybindings": {
    "quit": "q",
    "help": "?",
    "search": "/",
    "refresh": "r",
    "navigateUp": "k",
    "navigateDown": "j",
    "enter": "Enter",
    "tabSwitch": "Tab"
  }
}
```

### 配置优先级

配置优先级从高到低：

1. **命令行参数** - `--server`, `--timeout`
2. **环境变量** - `CANNBOT_SERVER`, `CANNBOT_TIMEOUT`
3. **配置文件** - `~/.cannbot-insight/config.json`
4. **默认值** - 内置默认配置

---

## 高级用法

### 结合 jq 进行数据处理

```bash
# 提取所有会话的 taskId 和 model
npx tsx src/cli/index.ts sessions --json | jq '.[] | {taskId, model}'

# 筛选成本超过 ¥5 的会话
npx tsx src/cli/index.ts sessions --json | jq '.[] | select(.totalCost > 5)'

# 计算平均 token 使用量
npx tsx src/cli/index.ts sessions --json | jq '[.[].totalTokens] | add / length'

# 按模型分组统计
npx tsx src/cli/index.ts sessions --json | jq 'group_by(.model) | map({model: .[0].model, count: length})'
```

### 脚本化使用

```bash
#!/bin/bash
# batch-analyze.sh - 批量分析高成本会话

# 获取成本超过 ¥5 的会话
HIGH_COST_SESSIONS=$(npx tsx src/cli/index.ts sessions --json | jq -r '.[] | select(.totalCost > 5) | .taskId')

# 逐个分析
for TASK_ID in $HIGH_COST_SESSIONS; do
  echo "Analyzing $TASK_ID..."
  npx tsx src/cli/index.ts session "$TASK_ID" --json > "analysis_${TASK_ID}.json"
done

echo "Analysis complete. Check analysis_*.json files."
```

### 与 Web UI 配合使用

```bash
# 1. 启动 Web UI
./start.sh

# 2. 在另一个终端使用 CLI 快速查找
npx tsx src/cli/index.ts sessions --user guanxinghua --limit 10

# 3. 找到感兴趣的会话后，在浏览器中查看详情
# 访问 http://localhost:21025/session/task_abc123
```

### 性能优化

```bash
# 增加超时时间（处理大型会话）
npx tsx src/cli/index.ts session task_large_session --timeout 120000

# 限制返回数量（快速预览）
npx tsx src/cli/index.ts sessions --limit 10

# 使用 JSON 模式避免表格渲染开销
npx tsx src/cli/index.ts sessions --json | head -20
```

---

## 故障排查

### 常见错误

#### 1. 连接失败

```
Error: Failed to connect to http://localhost:21025
```

**解决方案**：
- 确认 Web UI 已启动：`./start.sh`
- 检查端口是否正确：`lsof -i :21025`
- 使用 `--server` 指定正确的服务器地址

#### 2. 超时错误

```
Error: Request timeout after 30000ms
```

**解决方案**：
- 增加超时时间：`--timeout 60000`
- 检查网络连接
- 确认服务器负载正常

#### 3. 会话不存在

```
Error: Session not found: task_xyz789
```

**解决方案**：
- 确认 taskId 正确
- 使用 `sessions` 命令查看所有可用会话
- 检查会话是否已导入

#### 4. ESM 模块错误

```
Error: Cannot use import statement outside a module
```

**解决方案**：
- 确认 `package.json` 中有 `"type": "module"`
- 使用 `tsx` 而不是 `node` 运行
- 确认使用 ESM 导入语法

### 调试技巧

```bash
# 启用详细日志
DEBUG=* npx tsx src/cli/index.ts sessions

# 检查服务器连接
curl http://localhost:21025/api/observe/data

# 查看 CLI 版本
npx tsx src/cli/index.ts --version

# 查看帮助
npx tsx src/cli/index.ts --help
npx tsx src/cli/index.ts sessions --help
```

---

## 附录

### 版本历史

| 版本 | 日期 | 主要变更 |
|------|------|----------|
| v0.36 | 2026-06-15 | analyze 命令；TUI 搜索/导入/对比/删除交互修复；9 个新测试；ContextTab 统一配置 |
| v0.35 | 2026-06-15 | turn/import/delete/config 命令；33 新测试 |
| v0.34 | 2026-06-14 | TUI 交互模式（10 组件 + 4 Hooks + 6 屏幕 + 7 Tabs） |
| v0.33 | 2026-06-14 | 核心命令实现（sessions/session/stats/compare/search） |
| v0.32 | 2026-06-14 | CLI 地基模块（Client/Types/Errors/Config/Utils） |

### 相关文档

- [CLI 设计文档](cli-frontend-design-complete.md) - 完整的技术设计和实现细节
- [AGENTS.md](../AGENTS.md) - 项目级 AI 工具配置
- [README-zh.md](../README-zh.md) - 项目总体说明

### 反馈与支持

如有问题或建议，请联系项目维护者。
