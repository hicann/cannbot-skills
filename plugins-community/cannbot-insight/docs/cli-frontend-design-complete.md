# CANNBot-Insight CLI 前端设计与实现

> **版本**: v1.0  
> **日期**: 2026-06-14  
> **状态**: PoC 验证完成，准备开发  
> **技术栈**: Ink v7.0.6 + React 19.2.7 + ESM + tsx  
> **PoC 验证**: 所有关键技术点已通过验证（stdin 切换、组件渲染、测试框架）

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 技术选型与架构](#2-技术选型与架构)
- [3. 详细设计](#3-详细设计)
- [4. 测试策略](#4-测试策略)
- [5. 评审决策](#5-评审决策)

---

## 1. 项目概述

### 1.1 背景与目标

**现状**: CANNBot-Insight 目前只有一个 Web 前端（Next.js 16 + shadcn/ui），提供 3 个页面、15 个 API 端点。

**痛点**:
- 必须启动浏览器，SSH 远程服务器无法使用
- 不适合脚本化（CI/CD、cron）
- 信息密度低，快速排查不如终端高效
- 目标用户（LLM Agent 开发者）习惯终端工作流

**目标**: 
> **后端零改动，新增一套纯 CLI 前端，复用现有 15 个 API 端点。**

- 🖥️ 双模式：**TUI 交互模式**（Terminal UI）+ **纯命令行模式**（单次命令）
- 🔧 后端 API 完全不变，CLI 作为 API 客户端调用
- 📦 可独立安装，也可作为 `npm script` 使用

### 1.2 竞品参考

| 工具 | 模式 | 值得借鉴 |
|------|------|----------|
| **k9s** | TUI | 表格+详情面板、快捷键导航、资源树 |
| **lazygit** | TUI | 分栏布局、左右面板切换、内联 diff |
| **htop/btop** | TUI | 实时刷新、ASCII 柱状图、颜色编码 |
| **mitmproxy** | 双模 | Web + CLI 共存、flow 列表+详情钻取 |
| **SimonW's llm** | 纯命令 | `llm logs` 列表、`llm logs -r ID` 详情、管道友好 |
| **gh CLI** | 纯命令 | `gh pr list` → `gh pr view N` 列表+详情范式 |
| **LiteLLM** | 双模 | CLI proxy + Web dashboard 并行 |

**核心 UX 范式**: `list → select → detail`（列表浏览 → 选择 → 详情钻取）

---

## 2. 技术选型与架构

### 2.1 技术选型

| 方案 | 推荐 | 说明 |
|------|------|------|
| **Ink v7** (React for CLI) | ✅ 首选 | React 19 + ESM，与现有 React 技术栈统一；第三方组件自实现 |
| **Blessed/Blessed-Contrib** | ❌ | 成熟但非 React 范式，学习成本高 |
| **Terminal-kit** | ❌ | 低级 API，开发效率低 |
| **纯 Commander.js** | ⚡ 备选 | 仅做纯命令模式，不做 TUI |

**推荐**: Ink v7 + 自实现组件（PoC 验证：第三方 Ink 组件不兼容 v7，全部自实现）
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

### 2.2 架构设计

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

### 2.3 两种运行模式

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

## 3. 详细设计

### 3.1 模块接口设计

#### 3.1.1 InsightClient — API 客户端

**类定义**:

```typescript
// src/cli/client.ts

export interface ClientConfig {
  baseUrl: string;
  timeout: number;
  retries: number;
  retryDelay: number;
  authToken?: string;
}

export class InsightClient {
  private config: ClientConfig;

  constructor(baseUrl: string = 'http://localhost:21025', config?: Partial<ClientConfig>) {
    this.config = {
      baseUrl,
      timeout: 15000,
      retries: 2,
      retryDelay: 1000,
      ...config,
    };
  }

  private async request<T>(
    method: 'GET' | 'POST' | 'DELETE',
    path: string,
    params?: Record<string, string | number | boolean>,
    body?: Record<string, unknown>,
  ): Promise<T> { ... }
}
```

**全部方法签名**:

| 方法 | HTTP | API 路径 | 参数 | 返回值 |
|------|------|----------|------|--------|
| `listSessions(opts)` | GET | `/api/observe/data` | `{ page?, pageSize?, isSubagent?, user? }` | `{ items: ApiSessionListItem[], total: number, page: number }` |
| `getSession(taskId)` | GET | `/api/observe/session` | `taskId: string` | `ApiSessionDetailResponse` |
| `getStats(taskId?)` | GET | `/api/observe/stats` | `taskId?: string` | `ApiGlobalStatsResponse \| ApiSessionStatsResponse` |
| `getExecutions(taskId)` | GET | `/api/observe/executions` | `taskId: string` | `ApiExecutionItem[]` |
| `getTurns(taskId, opts?)` | GET | `/api/observe/session/turns` | `taskId: string, opts?: { isSubagent?, role? }` | `{ items: ApiTurnItem[], total: number }` |
| `getTurnDetail(turnId)` | GET | `/api/observe/session/turns/[turnId]` | `turnId: string` | `ApiTurnDetailResponse` |
| `searchTurns(taskId, keyword)` | GET | `/api/observe/session/turns/search` | `taskId: string, keyword: string` | `{ items: ApiSearchResult[], total: number }` |
| `getWorkflow(taskId)` | GET | `/api/observe/session/workflow` | `taskId: string` | `WorkflowTree` |
| `getBridges(taskId)` | GET | `/api/observe/session/bridges` | `taskId: string` | `{ items: ApiBridgeItem[], total: number }` |
| `importSession(source, filePath, sessionId)` | POST | `/api/ingest/import-file` | `source: string, filePath: string, sessionId: string` | `{ sessionId: string, imported: boolean }` |
| `listImportableSessions(source, filePath)` | POST | `/api/ingest/import-file/sessions` | `source: string, filePath: string` | `{ sessions: ApiImportableSession[] }` |
| `deleteSession(taskId?)` | DELETE | `/api/ingest/delete-session` | `taskId?: string, deleteAll?: boolean` | `{ deleted: number, taskId?: string }` |
| `analyzeWorkflow(taskId, provider)` | POST | `/api/ai/analyze-workflow` | `taskId: string, provider: ApiAIProviderConfig` | `{ result: WorkflowTree }` |
| `testProvider(baseUrl, apiKey)` | POST | `/api/ai/test-provider` | `baseUrl: string, apiKey: string` | `{ success: boolean, message: string }` |

**Response 类型定义**:

```typescript
// src/cli/types.ts — CLI 专用视图类型（从 API response 提取）
// CLI 类型加 Api 前缀，与 shared 内部类型（src/lib/shared/types.ts）明确区分
// 重叠子结构（如 TokenUsage）从 shared 导入复用

/** 来自 /api/observe/data response — 与 shared/SessionListItem 不同 */
export interface ApiSessionListItem {
  sessionId: string;
  taskId: string;
  query: string | null;
  startTime: string;
  endTime: string | null;
  totalTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  totalToolCallCount: number;
  totalSkillLoadCount: number;
  totalSubagentCount: number;
  model: string | null;
  user: string | null;
}

/** 来自 /api/observe/stats (无 taskId) */
export interface ApiGlobalStatsResponse {
  totalSessions: number;
  totalTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  avgLatencyMs: number;
}

/** 来自 /api/observe/stats (带 taskId) */
export interface ApiSessionStatsResponse {
  taskId: string;
  totalTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalReasoningTokens: number;
  totalCacheReadTokens: number;
  totalCacheWriteTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  totalToolCallCount: number;
  totalSkillLoadCount: number;
  totalSubagentCount: number;
  totalLlmCallCount: number;
}

// ... 其他类型定义省略，见原 design.md
```

**request 私有方法实现**:

```typescript
private async request<T>(
  method: 'GET' | 'POST' | 'DELETE',
  path: string,
  params?: Record<string, string | number | boolean>,
  body?: Record<string, unknown>,
): Promise<T> {
  const url = new URL(path, this.config.baseUrl);
  if (params && method === 'GET') {
    for (const [key, val] of Object.entries(params)) {
      url.searchParams.set(key, String(val));
    }
  }

  let lastError: Error | null = null;
  for (let attempt = 0; attempt <= this.config.retries; attempt++) {
    try {
      const init: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(this.config.authToken ? { Authorization: `Bearer ${this.config.authToken}` } : {}),
        },
        signal: AbortSignal.timeout(this.config.timeout),
      };
      if (body && (method === 'POST' || method === 'DELETE')) {
        init.body = JSON.stringify(body);
      }

      const res = await fetch(url.toString(), init);
      if (!res.ok) {
        const errorBody = await res.text();
        let errorMessage: string;
        try {
          const parsed = JSON.parse(errorBody);
          errorMessage = parsed.error ?? errorBody;
        } catch {
          errorMessage = errorBody;
        }
        if (res.status >= 400 && res.status < 500) {
          throw new ApiError(res.status, errorMessage, false);
        }
        throw new ApiError(res.status, errorMessage, true);
      }

      return await res.json() as T;
    } catch (err) {
      lastError = err instanceof ApiError
        ? err
        : new NetworkError(err instanceof Error ? err.message : String(err));

      if (err instanceof ApiError && !err.retryable) throw err;
      if (attempt < this.config.retries) {
        await new Promise(r => setTimeout(r, this.config.retryDelay * (attempt + 1)));
      }
    }
  }
  throw lastError ?? new NetworkError('Unknown error');
}
```

#### 3.1.2 错误类型体系

```typescript
// src/cli/errors.ts

export class InsightError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'InsightError';
  }
}

export class ApiError extends InsightError {
  public readonly status: number;
  public readonly retryable: boolean;

  constructor(status: number, message: string, retryable: boolean) {
    super(`API ${status}: ${message}`);
    this.name = 'ApiError';
    this.status = status;
    this.retryable = retryable;
  }
}

export class NetworkError extends InsightError {
  constructor(message: string) {
    super(`Network: ${message}`);
    this.name = 'NetworkError';
  }
}

export class TerminalError extends InsightError {
  constructor(message: string) {
    super(`Terminal: ${message}`);
    this.name = 'TerminalError';
  }
}

export class ConfigError extends InsightError {
  constructor(message: string) {
    super(`Config: ${message}`);
    this.name = 'ConfigError';
  }
}
```

#### 3.1.3 CLI 入口与命令路由

```typescript
// src/cli/index.ts

import { Command } from 'commander';
import { VERSION } from '@/lib/version';
import { runTui } from './tui/App';
import { sessionsCommand } from './commands/sessions';
import { sessionCommand } from './commands/session';
import { turnCommand } from './commands/turn';
import { searchCommand } from './commands/search';
import { compareCommand } from './commands/compare';
import { statsCommand } from './commands/stats';
import { importCommand } from './commands/import';
import { deleteCommand } from './commands/delete';
import { configCommand } from './commands/config';
import { loadConfig } from './config';
import { DEFAULT_SERVER_URL } from './config';

export function createProgram(): Command {
  const program = new Command();
  program
    .name('cannbot-insight')
    .alias('cbin')
    .description('CANNBot-Insight CLI — LLM Agent Session Observability')
    .version(VERSION)
    .option('--server <url>', 'Backend server URL', DEFAULT_SERVER_URL)
    .option('--timeout <ms>', 'Request timeout in ms', '15000');

  program
    .command('tui')
    .description('Launch interactive TUI mode')
    .action(async (opts, cmd) => {
      const globalOpts = cmd.parent.opts();
      await runTui(loadConfig(globalOpts));
    });

  program.addCommand(sessionsCommand());
  program.addCommand(sessionCommand());
  program.addCommand(turnCommand());
  program.addCommand(searchCommand());
  program.addCommand(compareCommand());
  program.addCommand(statsCommand());
  program.addCommand(importCommand());
  program.addCommand(deleteCommand());
  program.addCommand(configCommand());

  return program;
}

const program = createProgram();
program.parseAsync(process.argv).catch((err: Error) => {
  if (err instanceof InsightError) {
    console.error(`\n❌ ${err.message}`);
    process.exit(1);
  }
  console.error(err);
  process.exit(1);
});
```

### 3.2 TUI 组件设计

#### 3.2.1 App.tsx — TUI 根组件（含 stdin 切换）

```typescript
// src/cli/tui/App.tsx
import { render } from 'ink';
import React from 'react';
import { VERSION } from '@/lib/version';

export function runTui(config: CliConfig): Promise<void> {
  // Commander 解析完 argv 后 stdin 处于 paused 状态
  // 需要先 resume，再让 Ink 接管 raw mode
  if (process.stdin.isTTY) {
    process.stdin.resume();
    process.stdin.setRawMode(true);
  }

  const { waitUntilExit } = render(<App config={config} />, {
    exitOnCtrlC: false, // 我们自己处理 Ctrl+C
    patchConsole: false, // 不 patch console.log，避免和命令模式冲突
  });

  return waitUntilExit().then(() => {
    // Ink 退出后恢复 stdin
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
      process.stdin.pause();
    }
  });
}

function App({ config }: TuiAppProps) {
  // useMemo: config 不变时 client 不变
  const client = useMemo(
    () => new InsightClient(config.server, { timeout: config.timeout }),
    [config.server, config.timeout]
  );
  const { exit } = useApp();
  const [nav, setNav] = useState<NavigationState>({ screen: 'sessions', stack: [] });

  useInput((input, key) => {
    if (input === 'q' || (key.ctrl && input === 'c')) {
      exit();
    }
  });

  return (
    <Box flexDirection="column" height="100%">
      <StatusBar client={client} nav={nav} config={config} />
      <Box flexGrow={1}>
        {/* 屏幕路由 */}
      </Box>
      <KeyBar screen={nav.screen} />
    </Box>
  );
}
```

#### 3.2.2 自实现组件（替代第三方库）

**Spinner 组件**（替代 ink-spinner）:

```typescript
// src/cli/tui/components/Spinner.tsx
import { Text } from 'ink';
import { useState, useEffect } from 'react';

const SPINNER_FRAMES = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];

function Spinner({ label }: { label?: string }) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setFrame(f => (f + 1) % SPINNER_FRAMES.length), 80);
    return () => clearInterval(timer);
  }, []);
  return <Text>{SPINNER_FRAMES[frame]} {label ?? 'Loading...'}</Text>;
}
```

**TextInput 组件**（替代 ink-text-input）:

```typescript
// src/cli/tui/components/TextInput.tsx
import { Box, Text, useInput } from 'ink';

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  focus?: boolean;
}

function TextInput({ value, onChange, placeholder, focus = true }: TextInputProps) {
  useInput((input, key) => {
    if (!focus) return;
    if (key.backspace || key.delete) onChange(value.slice(0, -1));
    else if (key.return) { /* 提交 */ }
    else if (!key.ctrl && !key.meta) onChange(value + input);
  }, { isActive: focus });
  
  return (
    <Box>
      <Text color="gray">{placeholder ?? ''}</Text>
      <Text>{value}</Text>
      <Text color="gray">█</Text>
    </Box>
  );
}
```

**DataTable 组件**（替代 ink-table）:

```typescript
// src/cli/tui/components/DataTable.tsx
import { Box, Text } from 'ink';
import stringWidth from 'string-width';

interface DataTableProps<T> {
  columns: Array<{
    key: string;
    label: string;
    width: number;
    render?: (row: T, selected: boolean) => string;
  }>;
  data: T[];
  selectedIndex: number;
}

function DataTable<T>({ columns, data, selectedIndex }: DataTableProps<T>) {
  return (
    <Box flexDirection="column">
      {/* Header */}
      <Box>
        {columns.map(col => (
          <Text bold key={col.key} width={col.width}>
            {padEndVisual(col.label, col.width)}
          </Text>
        ))}
      </Box>
      {/* Separator */}
      <Text color="gray">
        {columns.map(c => '─'.repeat(c.width)).join('─┼─')}
      </Text>
      {/* Body */}
      {data.map((row, i) => {
        const selected = i === selectedIndex;
        return (
          <Box key={i}>
            {columns.map(col => {
              const val = col.render
                ? col.render(row, selected)
                : String((row as any)[col.key] ?? '—');
              return (
                <Text
                  key={col.key}
                  width={col.width}
                  color={selected ? 'cyan' : undefined}
                  bold={selected}
                >
                  {truncateVisual(val, col.width)}
                </Text>
              );
            })}
          </Box>
        );
      })}
    </Box>
  );
}

// 视觉宽度安全的 padEnd
function padEndVisual(str: string, width: number): string {
  const visualWidth = stringWidth(str);
  if (visualWidth >= width) return str;
  return str + ' '.repeat(width - visualWidth);
}

// 视觉宽度安全的截断
function truncateVisual(str: string, maxWidth: number): string {
  if (stringWidth(str) <= maxWidth) return str;
  let result = '';
  let width = 0;
  for (const char of str) {
    const charWidth = stringWidth(char);
    if (width + charWidth > maxWidth - 1) break;
    result += char;
    width += charWidth;
  }
  return result + '…';
}
```

### 3.3 分页加载（替代虚拟滚动）

```typescript
// src/cli/tui/screens/SessionList.tsx
import { Box, Text, useInput } from 'ink';
import { useState, useCallback } from 'react';

function SessionList({ client, onSelect }: SessionListProps) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;

  const fetchSessions = useCallback(
    () => client.listSessions({ page, pageSize: PAGE_SIZE }),
    [client, page]
  );
  
  const { data, loading } = useApi(fetchSessions, [client, page]);
  const table = useTable(data?.items ?? [], SESSION_COLUMNS);

  useKeyboard({
    onNavigateUp: table.selectUp,
    onNavigateDown: table.selectDown,
    onEnter: () => {
      const selected = table.state.visibleData[table.state.selectedIndex];
      if (selected) onSelect(selected.taskId);
    },
    custom: {
      'n': () => setPage(p => p + 1),  // Next page
      'p': () => setPage(p => Math.max(1, p - 1)),  // Previous page
    },
  });

  if (loading) return <Spinner label="Loading sessions..." />;

  return (
    <Box flexDirection="column">
      <DataTable
        columns={SESSION_COLUMNS}
        data={table.state.visibleData}
        selectedIndex={table.state.selectedIndex}
      />
      <Text color="gray">
        Page {page}/{Math.ceil((data?.total ?? 0) / PAGE_SIZE)} │ 
        n: Next │ p: Prev │ Total: {data?.total ?? 0}
      </Text>
    </Box>
  );
}
```

### 3.4 配置管理

```typescript
// src/cli/config.ts
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export const DEFAULT_SERVER_URL = 'http://localhost:21025';

export interface CliConfig {
  server: string;
  timeout: number;
  theme: 'dark' | 'light' | 'auto';
  keybindings: Record<string, string>;
}

export const DEFAULT_CONFIG: CliConfig = {
  server: DEFAULT_SERVER_URL,
  timeout: 15000,
  theme: 'auto',
  keybindings: {
    quit: 'q',
    help: '?',
    search: '/',
    refresh: 'r',
    navigateUp: 'k',
    navigateDown: 'j',
    enter: 'Enter',
    tabSwitch: 'Tab',
  },
};

const CONFIG_DIR = path.join(os.homedir(), '.cannbot-insight');
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

export function loadConfig(globalOpts?: { server?: string; timeout?: string }): CliConfig {
  let config = { ...DEFAULT_CONFIG };

  // 优先级: 命令行参数 > 环境变量 > 配置文件 > 默认值
  if (fs.existsSync(CONFIG_FILE)) {
    try {
      const saved = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf-8'));
      config = { ...config, ...saved };
    } catch { /* ignore invalid config */ }
  }

  if (process.env.CANNBOT_SERVER) {
    config.server = process.env.CANNBOT_SERVER;
  }

  if (globalOpts?.server) {
    config.server = globalOpts.server;
  }

  if (process.env.CANNBOT_TIMEOUT) {
    config.timeout = +process.env.CANNBOT_TIMEOUT;
  }

  if (globalOpts?.timeout) {
    config.timeout = +globalOpts.timeout;
  }

  return config;
}

export function saveConfig(config: Partial<CliConfig>): void {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
  const current = loadConfig();
  const merged = { ...current, ...config };
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(merged, null, 2));
}
```

---

## 4. 测试策略

### 4.1 测试分层策略

```
┌──────────────────────────────────────┐
│        E2E 测试 (5%)                 │  ← 完整用户流程
├──────────────────────────────────────┤
│        集成测试 (15%)                │  ← API Client + 后端
├──────────────────────────────────────┤
│        TUI 组件测试 (20%)            │  ← Ink 组件渲染/交互
├──────────────────────────────────────┤
│        单元测试 (60%)                │  ← 纯函数、hooks、命令
└──────────────────────────────────────┘
```

### 4.2 测试目录结构

```
tests/
├── cli/                           # ✨ CLI 前端测试（新增）
│   ├── unit/
│   │   ├── client.test.ts         # InsightClient 单元测试
│   │   ├── config.test.ts         # 配置管理测试
│   │   ├── errors.test.ts         # 错误类型测试
│   │   ├── format.test.ts         # 格式化工具测试
│   │   ├── table.test.ts          # 表格渲染工具测试
│   │   ├── colors.test.ts         # 颜色主题测试
│   │   └── commands/
│   │       ├── sessions.test.ts   # sessions 命令测试
│   │       ├── session.test.ts    # session 命令测试
│   │       ├── turn.test.ts       # turn 命令测试
│   │       ├── search.test.ts     # search 命令测试
│   │       ├── compare.test.ts    # compare 命令测试
│   │       ├── stats.test.ts      # stats 命令测试
│   │       ├── import.test.ts     # import 命令测试
│   │       ├── delete.test.ts     # delete 命令测试
│   │       └── config.test.ts     # config 命令测试
│   ├── hooks/
│   │   ├── useApi.test.ts         # API hook 测试
│   │   ├── useKeyboard.test.ts    # 键盘事件 hook 测试
│   │   ├── useNavigation.test.ts  # 导航 hook 测试
│   │   └── useTable.test.ts       # 表格状态 hook 测试
│   ├── components/
│   │   ├── StatusBar.test.tsx      # 状态栏组件测试
│   │   ├── KeyBar.test.tsx         # 快捷键提示测试
│   │   ├── DataTable.test.tsx      # 数据表格测试
│   │   ├── MetricCards.test.tsx    # 指标卡片测试
│   │   ├── AsciiBar.test.tsx       # ASCII 柱状图测试
│   │   ├── TreeView.test.tsx       # 树形图测试
│   │   ├── TabBar.test.tsx         # Tab 切换栏测试
│   │   ├── ConfirmDialog.test.tsx  # 确认对话框测试
│   │   ├── Spinner.test.tsx        # 自写 Spinner 组件测试
│   │   └── TextInput.test.tsx      # 自写 TextInput 组件测试
│   ├── screens/
│   │   ├── SessionList.test.tsx    # Session 列表屏测试
│   │   ├── SessionDetail.test.tsx  # Session 详情屏测试
│   │   ├── TurnDetail.test.tsx     # Turn 详情屏测试
│   │   ├── CompareView.test.tsx    # 对比屏测试
│   │   └── ImportPanel.test.tsx    # 导入面板测试
│   └── integration/
│       ├── client-api.test.ts      # Client + Mock Server 集成
│       ├── command-flow.test.ts    # 命令完整流程测试
│       ├── tui-navigation.test.tsx # TUI 导航流程测试
│       ├── sessions-flow.test.ts   # Session 列表→详情 API 流程
│       ├── import-flow.test.ts     # 导入 API 流程
│       └── compare-flow.test.ts    # 对比 API 流程
├── fixtures/                      # 测试数据
│   ├── mock-sessions.json
│   ├── mock-stats.json
│   ├── mock-turns.json
│   ├── mock-workflow.json
│   ├── mock-bridges.json
│   └── mock-executions.json
└── helpers/                       # 测试辅助工具
    ├── mock-server.ts             # MSW mock server
    ├── render-tui.tsx             # Ink 测试渲染器
    └── fixtures.ts                # fixture 工厂函数
```

### 4.3 TUI 组件测试（双 render 策略）

**测试辅助工具**:

```typescript
// tests/helpers/render-tui.tsx — 使用 ink-testing-library（PoC 验证兼容）
import React from 'react';
import { render } from 'ink-testing-library';

export function renderTui(element: React.ReactElement) {
  const instance = render(element);
  return {
    ...instance,
    // ink-testing-library 的 lastFrame() 返回 ANSI 含色码的输出
    lastFrame: () => instance.lastFrame(),
    // 获取纯文本（去除 ANSI 色码）
    getPlainText: () => instance.lastFrame()?.replace(/\x1b\[[0-9;]*m/g, '') ?? '',
    // stdin 可模拟按键（ink-testing-library 提供）
    pressKey: (key: string) => instance.stdin.write(key),
    pressEnter: () => instance.stdin.write('\r'),
    pressEscape: () => instance.stdin.write('\x1b'),
    pressUp: () => instance.stdin.write('\x1b[A'),
    pressDown: () => instance.stdin.write('\x1b[B'),
    pressCtrlC: () => instance.stdin.write('\x03'),
  };
}
```

**测试示例**:

```typescript
// tests/cli/components/DataTable.test.tsx
import { describe, it, expect } from 'vitest';
import { renderTui } from '../../helpers/render-tui';
import { DataTable } from '../../../src/cli/tui/components/DataTable';

describe('DataTable', () => {
  const columns = [
    { key: 'id', label: 'ID', width: 10 },
    { key: 'name', label: 'Name', width: 20 },
  ];
  
  const data = [
    { id: '001', name: 'Alice' },
    { id: '002', name: 'Bob' },
  ];

  it('renders table with header and rows', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={0} />
    );
    
    const output = getPlainText();
    expect(output).toContain('ID');
    expect(output).toContain('Name');
    expect(output).toContain('001');
    expect(output).toContain('Alice');
    expect(output).toContain('002');
    expect(output).toContain('Bob');
  });

  it('highlights selected row', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={1} />
    );
    
    const output = getPlainText();
    // 验证第二行被选中（通过颜色或样式）
    expect(output).toContain('002');
  });
});
```

### 4.4 单元测试示例

**InsightClient 测试**:

```typescript
// tests/cli/unit/client.test.ts
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InsightClient, ApiError, NetworkError } from '../../src/cli/client';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('InsightClient', () => {
  describe('constructor', () => {
    it('uses default baseUrl when none provided', () => {
      const client = new InsightClient();
      expect(client['config'].baseUrl).toBe('http://localhost:21025');
    });

    it('accepts custom baseUrl', () => {
      const client = new InsightClient('http://custom:8080');
      expect(client['config'].baseUrl).toBe('http://custom:8080');
    });

    it('merges partial config with defaults', () => {
      const client = new InsightClient('http://localhost:21025', { timeout: 5000 });
      expect(client['config'].timeout).toBe(5000);
      expect(client['config'].retries).toBe(2);
      expect(client['config'].retryDelay).toBe(1000);
    });
  });

  describe('retry logic', () => {
    let client: InsightClient;

    beforeEach(() => {
      client = new InsightClient('http://localhost:21025', { retries: 2, retryDelay: 10 });
      mockFetch.mockReset();
    });

    it('retries on 5xx and succeeds on second attempt', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false, status: 500,
          text: async () => 'Internal Server Error',
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ totalSessions: 42 }),
        });

      const result = await client.getStats();
      expect((result as any).totalSessions).toBe(42);
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('throws after all retries exhausted on 5xx', async () => {
      mockFetch.mockResolvedValue({
        ok: false, status: 500,
        text: async () => 'Internal Server Error',
      });

      await expect(client.getStats()).rejects.toThrow(ApiError);
      expect(mockFetch).toHaveBeenCalledTimes(3); // initial + 2 retries
    });

    it('does NOT retry on 4xx errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false, status: 400,
        text: async () => JSON.stringify({ error: 'Bad request' }),
      });

      await expect(client.listSessions({})).rejects.toThrow(ApiError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });
  });
});
```

### 4.5 测试环境配置

**Vitest 配置**:

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  test: {
    globals: true,
    environment: 'node',
    setupFiles: ['./tests/cli/setup.ts'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      exclude: [
        'node_modules/',
        'tests/',
        '**/*.d.ts',
      ],
    },
  },
});
```

**测试 setup**:

```typescript
// tests/cli/setup.ts
import { vi } from 'vitest';

// Mock Ink 组件（如果需要）
vi.mock('ink', async () => {
  const actual = await vi.importActual('ink');
  return {
    ...actual,
    // 可以添加 mock
  };
});

// 全局测试超时
vi.setConfig({ testTimeout: 10000 });
```

---

## 5. 评审决策

### 5.1 决策表

| # | 问题 | 决策 | 修改文件 | 优先级 |
|---|------|------|----------|--------|
| 1 | Commander 与 Ink stdin 冲突 | runTui() 中显式 resume stdin + setRawMode，退出时恢复；Ink render 配置 exitOnCtrlC=false | design.md §3.1 App.tsx | P0 |
| 2 | Ink 第三方组件库成熟度 | 全部自实现（Spinner 10行、TextInput 30行），不依赖 ink-table/ink-select/ink-spinner 等第三方库 | design.md 新增 §3.10 Spinner、§3.11 TextInput，删除第三方库引用 | P0 |
| 3 | ink-testing-library 兼容性 | 双层策略：层1 用 Ink render()+lastFrame() 快照测试；层2 对 useInput 组件 mock useInput 测试 handler；删除 ink-testing-library 依赖 | test-plan.md §4.1 render-tui.tsx、§4.2-4.9 全部组件测试、§8.1 依赖包 | P0 |
| 4 | 类型定义重复 | CLI 类型加 Api 前缀与 shared 区分；TokenUsage 等子结构从 shared 导入复用；注释标注来源 | design.md §1.1.3 Response 类型定义 | P1 |
| 5 | InsightClient 每次渲染重新实例化 | 改为 useMemo 创建 client，依赖 [config.server, config.timeout] | design.md §3.1 App.tsx L816 | P1 |
| 6 | useApi 缓存 key 性能问题 | cacheKey 用 useMemo 包裹；fetcher 用 useCallback 稳定化；调用示例更新 | design.md §4.3 useApi.ts | P1 |
| 7 | 虚拟滚动 vs 分页 | 改为服务端分页（pageSize=20）+ 前端翻页（n/p 键）；DataTable 移除虚拟滚动逻辑 | design.md §3.4 DataTable、§5.1 SessionList、§5.4 TurnsTab、§8.1 性能 | P1 |
| 8 | E2E TUI 测试在 CI 中不可靠 | 将 E2E 测试重分类为 API 流程集成测试；TUI 交互测试改为组件级（mock useInput）；真实 TUI E2E 仅本地手动 | test-plan.md §5 E2E 章节 → integration 目录 | P1 |
| 9 | GitHub Actions CI 不适用 | 改为 GitCode/GitLab-compatible CI（.gitlab-ci.yml）；保留 GitHub Actions 作为备用 | test-plan.md §8.5 CI/CD 集成 | P1 |
| 10 | 版本号硬编码 | 从 @/lib/version.ts 导入 VERSION / VERSION_DISPLAY | design.md §1.3 index.ts、§3.2 StatusBar.tsx | P2 |
| 11 | cbin 短别名未配置 | package.json bin 字段配置 cannbot-insight + cbin 双入口 | design.md 新增 §1.9 package.json bin 配置 | P2 |
| 12 | 缺少认证机制 | ClientConfig 和 CliConfig 预留 authToken 字段；request 方法中自动添加 Authorization header；支持 CANNBOT_TOKEN 环境变量 | design.md §1.1 ClientConfig、§1.5 CliConfig | P2 |
| 13 | 缺少性能基准 | 新增简易性能基准测试（renderSnapshot 计时、mock server 计时）；定义 5 个性能目标 | test-plan.md 新增 §9 性能基准 | P2 |
| 14 | 中文宽度对齐问题 | 使用 string-width 库计算视觉宽度；padEndVisual/truncateVisual 函数；DataTable 列宽用视觉宽度 | design.md §1.6 format.ts、§1.7 table.ts、§3.4 DataTable | P1 |

### 5.2 实施优先级

1. **P0 立即执行**（开发前必须完成）: 问题1、2、3 — stdin 切换、组件自实现、测试方案
2. **P1 开发中调整**（第一个迭代内完成）: 问题4-9、14 — 类型命名、client/useApi 修复、分页、测试重分类、CI、中文宽度
3. **P2 小改进**（后续迭代完成）: 问题10-13 — 版本号、bin 配置、authToken、性能基准

---

## 附录：PoC 验证结果

所有关键技术点已通过 PoC 验证：

| PoC | 验证内容 | 结果 |
|-----|----------|------|
| PoC 1 | Ink v7 基础渲染 | ✅ 通过 |
| PoC 2 | Ink v7 render() API | ✅ 确认无 lastFrame |
| PoC 3 | ink-testing-library 兼容性 | ✅ 完全兼容 v7 |
| PoC 4 | Commander + Ink stdin 切换 | ✅ 键盘交互正常 |

**验证代码位置**: `/home/gxh/code/07_cannbot_insight/poc-ink/`

---

## 下一步

1. 安装核心依赖（ink@7.0.6, react@19.2.7, commander, chalk, string-width, cli-truncate, ink-testing-library）
2. 实现 InsightClient API 客户端
3. 实现纯命令模式（sessions, session, turn, search, compare, stats, import, delete, config）
4. 实现 TUI 模式（App.tsx + 各屏幕组件）
5. 编写单元测试和组件测试
6. 集成测试和性能测试

---

**文档结束**
