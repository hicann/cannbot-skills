# CANNBot-Insight CLI 前端详细设计文档

> **版本**: v1.2  
> **日期**: 2026-06-14  
> **基于**: [CLI 前端需求文档 v1.0](../../cli-frontend-requirements.md)  
> **PoC 验证**: Ink v7.0.6 + React 19.2.7 + ESM + tsx  
> **变更**: 基于 PoC 验证结果全面刷新，确认 Ink v7 API、ESM 要求、自实现组件、中文宽度方案

---

## 1. 模块接口设计

### 1.1 InsightClient — API 客户端

#### 1.1.1 类定义

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

#### 1.1.2 全部方法签名

| 方法 | HTTP | API 路径 | 参数 | 返回值 |
|------|------|----------|------|--------|
| `listSessions(opts)` | GET | `/api/observe/data` | `{ page?, pageSize?, isSubagent?, user? }` | `{ items: SessionListItem[], total: number, page: number }` |
| `getSession(taskId)` | GET | `/api/observe/session` | `taskId: string` | `SessionDetailResponse` |
| `getStats(taskId?)` | GET | `/api/observe/stats` | `taskId?: string` | `GlobalStatsResponse | SessionStatsResponse` |
| `getExecutions(taskId)` | GET | `/api/observe/executions` | `taskId: string` | `ExecutionsResponse` |
| `getTurns(taskId, opts?)` | GET | `/api/observe/session/turns` | `taskId: string, opts?: { isSubagent?, role? }` | `{ items: TurnItem[], total: number }` |
| `getTurnDetail(turnId)` | GET | `/api/observe/session/turns/[turnId]` | `turnId: string` | `TurnDetailResponse` |
| `searchTurns(taskId, keyword)` | GET | `/api/observe/session/turns/search` | `taskId: string, keyword: string` | `{ items: SearchResult[], total: number }` |
| `getWorkflow(taskId)` | GET | `/api/observe/session/workflow` | `taskId: string` | `WorkflowTree` |
| `getBridges(taskId)` | GET | `/api/observe/session/bridges` | `taskId: string` | `{ items: BridgeItem[], total: number }` |
| `importSession(source, filePath, sessionId)` | POST | `/api/ingest/import-file` | `source: string, filePath: string, sessionId: string` | `{ sessionId: string, imported: boolean }` |
| `listImportableSessions(source, filePath)` | POST | `/api/ingest/import-file/sessions` | `source: string, filePath: string` | `{ sessions: ImportableSession[] }` |
| `deleteSession(taskId?)` | DELETE | `/api/ingest/delete-session` | `taskId?: string, deleteAll?: boolean` | `{ deleted: number, taskId?: string }` |
| `analyzeWorkflow(taskId, provider)` | POST | `/api/ai/analyze-workflow` | `taskId: string, provider: AIProviderConfig` | `{ result: WorkflowTree }` |
| `testProvider(baseUrl, apiKey)` | POST | `/api/ai/test-provider` | `baseUrl: string, apiKey: string` | `{ success: boolean, message: string }` |

#### 1.1.3 Response 类型定义

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

/** 来自 /api/observe/session response */
export interface ApiSessionDetailResponse {
  sessionId: string;
  taskId: string;
  label: string | null;
  query: string | null;
  framework: string | null;
  startTime: string | null;
  endTime: string | null;
  totalTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalReasoningTokens: number;
  totalCacheReadTokens: number;
  totalCacheWriteTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  totalToolCallCount: number;
  totalLlmCallCount: number;
  totalSkillLoadCount: number;
  totalSubagentCount: number;
  model: string | null;
  user: string | null;
  sourcePath: string | null;
  agents: ExecutionItem[];
  skills: SkillSummary[];
}

/** 来自 /api/observe/executions response */
export interface ApiExecutionItem {
  executionId: string;
  agentName: string | null;
  isSubagent: boolean;
  parentExecutionId: string | null;
  tokens: number;
  cost: number;
  toolCallCount: number;
  skillLoadCount: number;
  model: string | null;
  createdAt: string;
  latencyMs: number;
}

/** 来自 /api/observe/session (skills) */
export interface ApiSkillSummary {
  skillName: string;
  version: number | null;
  invocationCount: number;
}

/** 来自 /api/observe/session/turns response */
export interface ApiTurnItem {
  turnId: string;
  turnIndex: number;
  role: string;
  contentSummary: string | null;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  parentExecutionId: string | null;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  inputMessagesCount: number | null;
  inputMessagesTokens: number | null;
  contextWindowPct: number | null;
  latencyMs: number;
  createdAt: string;
  completedAt: string | null;
  model: string | null;
  finishReason: string | null;
  toolCalls: TurnToolCall[];
  skillEvents: TurnSkillEvent[];
}

/** Turn 内嵌 ToolCall */
export interface ApiTurnToolCall {
  toolCallId: string;
  toolName: string;
  state: string;
  durationMs: number | null;
}

/** Turn 内嵌 SkillEvent */
export interface ApiTurnSkillEvent {
  skillName: string;
  eventType: string;
  success: boolean | null;
}

/** 来自 /api/observe/session/turns/[turnId] response */
export interface ApiTurnDetailResponse {
  turnId: string;
  sessionId: string;
  turnIndex: number;
  role: string;
  content: string | null;
  contentJson: string | null;
  contentSummary: string | null;
  inputMessagesJson: string | null;
  inputMessagesCount: number | null;
  inputMessagesTokens: number | null;
  contextWindowPct: number | null;
  agentName: string | null;
  subagentName: string | null;
  subagentSessionId: string | null;
  isSubagent: boolean;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  latencyMs: number;
  ttftMs: number | null;
  createdAt: string;
  completedAt: string | null;
  model: string | null;
  modelId: string | null;
  providerId: string | null;
  finishReason: string | null;
  toolCalls: TurnDetailToolCall[];
  skillEvents: TurnDetailSkillEvent[];
}

export interface ApiTurnDetailToolCall {
  id: string;
  toolCallId: string;
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
  errorType: string | null;
  errorMessage: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
  dispatchBridgeId: string | null;
  isSkillRelated: boolean | null;
}

export interface ApiTurnDetailSkillEvent {
  id: string;
  skillName: string;
  skillVersion: number | null;
  eventType: string;
  success: boolean | null;
  errorMessage: string | null;
  argsJson: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number | null;
}

/** 来自 /api/observe/session/turns/search response */
export interface ApiSearchResult {
  turnId: string;
  turnIndex: number;
  role: string;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  contentSummary: string | null;
  matchContext: string;
  matchField: 'content' | 'contentSummary';
  createdAt: string;
  hasDispatchBridge: boolean;
}

/** 来自 /api/observe/session/bridges response */
export interface ApiBridgeItem {
  bridgeId: string;
  dispatchExecutionId: string | null;
  dispatchTurnId: string | null;
  dispatchToolCallId: string | null;
  dispatchContent: string | null;
  dispatchTimestamp: string | null;
  responseExecutionId: string | null;
  responseTurnId: string | null;
  responseContent: string | null;
  responseTimestamp: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  subagentTokens: number | null;
  subagentLatencyMs: number | null;
}

export interface ApiAIProviderConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

/** 来自 /api/ingest/import-file/sessions response */
export interface ApiImportableSession {
  id: string;
  createdAt: string | null;
  firstQuery: string | null;
  turnCount: number;
  model: string | null;
}
```

#### 1.1.4 `request` 私有方法实现

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

### 1.2 错误类型体系

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

### 1.3 CLI 入口与命令路由

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

### 1.4 命令模块接口

每个命令模块导出工厂函数，返回 `Command` 实例：

```typescript
// src/cli/commands/sessions.ts

export function sessionsCommand(): Command {
  const cmd = new Command('sessions');
  cmd
    .description('List all sessions')
    .option('--user <name>', 'Filter by user')
    .option('--limit <n>', 'Limit number of results', '20')
    .option('--subagent', 'Include subagent sessions')
    .option('--json', 'Output as JSON')
    .action(async (opts, cmd) => {
      const globalOpts = cmd.parent.opts();
      const client = new InsightClient(globalOpts.server, { timeout: +globalOpts.timeout });
      const result = await client.listSessions({
        pageSize: +opts.limit,
        user: opts.user,
        isSubagent: opts.subagent ? 'true' : undefined,
      });
      if (opts.json) {
        console.log(JSON.stringify(result.items, null, 2));
        return;
      }
      renderSessionTable(result.items, result.total);
    });
  return cmd;
}
```

所有命令模块遵循相同模式：

| 模块 | 导出函数 | 核心逻辑 |
|------|----------|----------|
| `sessions.ts` | `sessionsCommand()` | 调用 `client.listSessions()` → 表格/JSON |
| `session.ts` | `sessionCommand()` | 调用 `client.getSession()` + `getStats()` + `getExecutions()` → 多 tab |
| `turn.ts` | `turnCommand()` | 调用 `client.getTurnDetail()` → JSON/表格 |
| `search.ts` | `searchCommand()` | 调用 `client.searchTurns()` → 搜索结果 |
| `compare.ts` | `compareCommand()` | 并行调用 2 次 `getSession()` + `getStats()` → 对比表格 |
| `stats.ts` | `statsCommand()` | 调用 `client.getStats()` → 统计输出 |
| `import.ts` | `importCommand()` | 调用 `client.listImportableSessions()` 或 `importSession()` |
| `delete.ts` | `deleteCommand()` | 调用 `client.deleteSession()` → 确认提示 |
| `config.ts` | `configCommand()` | 管理 `~/.cannbot-insight/config.json` |

### 1.5 配置管理

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
  authToken?: string;
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

  if (process.env.CANNBOT_TOKEN) {
    config.authToken = process.env.CANNBOT_TOKEN;
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

### 1.6 格式化工具

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

export function formatDuration(ms: number): string {
  const seconds = ms / 1000;
  if (seconds >= 3600) return `${(seconds / 3600).toFixed(1)}h`;
  if (seconds >= 60) return `${(seconds / 60).toFixed(0)}m`;
  return `${seconds.toFixed(1)}s`;
}

export function formatDate(iso: string | null): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return `${(d.getMonth() + 1).toString().padStart(2, '0')}-${d.getDate().toString().padStart(2, '0')} ${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`;
}

export function formatPercent(pct: number | null): string {
  if (pct === null) return '—';
  return `${pct.toFixed(1)}%`;
}

export function truncate(str: string | null, maxLen: number): string {
  if (!str) return '—';
  if (str.length <= maxLen) return str;
  return str.substring(0, maxLen - 1) + '…';
}

// 中文宽度相关工具（PoC 验证：中文字符占 2 列宽，需要视觉宽度计算）
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

### 1.7 表格渲染工具

```typescript
// src/cli/utils/table.ts

import { padEndVisual, truncateVisual } from './format';

export interface TableColumn<T> {
  key: string;
  label: string;
  width: number; // 视觉宽度（中文字符占 2 列）
  render?: (row: T) => string;
}

export function renderTable<T>(columns: TableColumn<T>[], rows: T[]): string {
  const header = columns.map(c => padEndVisual(c.label, c.width)).join(' │ ');
  const separator = columns.map(c => '─'.repeat(c.width)).join('─┼─');
  const body = rows.map(row =>
    columns.map(c => {
      const val = c.render ? c.render(row) : String((row as Record<string, unknown>)[c.key] ?? '—');
      return truncateVisual(val, c.width);
    }).join(' │ ')
  ).join('\n');
  return `${header}\n${separator}\n${body}`;
}
```

### 1.8 颜色主题

```typescript
// src/cli/utils/colors.ts

import chalk from 'chalk';

export interface ColorTheme {
  header: chalk.Chalk;
  accent: chalk.Chalk;
  muted: chalk.Chalk;
  success: chalk.Chalk;
  warning: chalk.Chalk;
  error: chalk.Chalk;
  highlight: chalk.Chalk;
  selected: chalk.Chalk;
  border: chalk.Chalk;
}

export const DARK_THEME: ColorTheme = {
  header: chalk.bold.cyan,
  accent: chalk.cyan,
  muted: chalk.gray,
  success: chalk.green,
  warning: chalk.yellow,
  error: chalk.red.bold,
  highlight: chalk.white.bold,
  selected: chalk.bgCyan.black,
  border: chalk.gray,
};

export const LIGHT_THEME: ColorTheme = {
  header: chalk.bold.blue,
  accent: chalk.blue,
  muted: chalk.dim,
  success: chalk.green,
  warning: chalk.yellow,
  error: chalk.red.bold,
  highlight: chalk.black.bold,
  selected: chalk.bgBlue.white,
  border: chalk.dim,
};

export function getTheme(configTheme: 'dark' | 'light' | 'auto'): ColorTheme {
  if (configTheme === 'auto') {
    const term = process.env.TERM ?? '';
    const colorScheme = process.env.COLORSCHEME ?? '';
    if (term.includes('dark') || colorScheme === 'dark') return DARK_THEME;
    return DARK_THEME; // 终端默认深色
  }
  return configTheme === 'dark' ? DARK_THEME : LIGHT_THEME;
}
```

---

## 2. 数据流设计

### 2.1 API Client 请求封装

```
Command / TUI Screen
       │
       ▼
   InsightClient.request<T>()
       │
       ├── 构建 URL (baseUrl + path + query params)
       ├── 构建 RequestInit (method, headers, body, signal)
       │
       ▼
   fetch() → Response
       │
       ├── res.ok → res.json() → T
       │
       ├── 4xx → ApiError(status, message, retryable=false) → 直接抛出
       │
       ├── 5xx → ApiError(status, message, retryable=true) → 重试
       │
       ├── timeout / network → NetworkError → 重试
       │
       ▼
   重试循环 (retries=2, delay=1000×attempt)
       │
       ├── 成功 → 返回 T
       ├── 失败（重试耗尽）→ 抛出 lastError
```

### 2.2 错误处理分级

| 级别 | 错误类型 | 处理方式 | 例子 |
|------|----------|----------|------|
| 1 | `ApiError(4xx)` | 不重试，直接报错退出 | 404 session 不存在、400 参数错误 |
| 2 | `ApiError(5xx)` | 重试 2 次，失败后报错 | 500 服务器内部错误 |
| 3 | `NetworkError` | 重试 2 次，失败后提示检查 server URL | ECONNREFUSED、timeout |
| 4 | `TerminalError` | 不重试，提示降级方案 | 终端不支持 ANSI |
| 5 | `ConfigError` | 不重试，提示修复配置 | 配置文件格式错误 |

### 2.3 命令模式数据流

```
用户输入命令 → Commander 解析 → 命令 action
  → InsightClient 方法调用 → API 请求 → 返回数据
  → --json 模式: console.log(JSON.stringify(data))
  → 表格模式: renderTable() / 自定义渲染 → console.log()
```

### 2.4 TUI 模式数据流

```
App.tsx 初始化 → InsightClient 实例
  → runTui()：stdin resume + setRawMode → Ink v7 原生 render() → waitUntilExit
  → useApi hook → fetch + loading/error/data 状态
  → TUI 组件渲染 → useKeyboard hook → 用户输入
  → 导航事件 → useNavigation hook → 屏幕切换
  → 刷新事件 → 重新 fetch → 更新组件
  → 退出：stdin setRawMode(false) + pause

注意：Ink v7 原生 render() 返回 rerender/unmount/waitUntilExit/waitUntilRenderFlush/cleanup/clear
     没有 lastFrame/frames/output（测试用 ink-testing-library 的 render()）
```

---

## 3. TUI 组件设计

### 3.1 App.tsx — TUI 根组件

```typescript
// src/cli/tui/App.tsx

import React, { useState, useMemo } from 'react';
import { Box, Text, useApp, useInput, render } from 'ink';
import { VERSION_DISPLAY } from '@/lib/version';
import { InsightClient } from '../client';
import { CliConfig } from '../config';
import { NavigationState, Screen } from '../hooks/useNavigation';
import { StatusBar } from '../components/StatusBar';
import { KeyBar } from '../components/KeyBar';
import { SessionList } from '../screens/SessionList';
import { SessionDetail } from '../screens/SessionDetail';
import { TurnDetail } from '../screens/TurnDetail';
import { CompareView } from '../screens/CompareView';
import { ImportPanel } from '../screens/ImportPanel';
import { HelpScreen } from '../screens/HelpScreen';

export interface TuiAppProps {
  config: CliConfig;
}

export function runTui(config: CliConfig): Promise<void> {
  if (process.stdin.isTTY) {
    process.stdin.resume();
    process.stdin.setRawMode(true);
  }

  const { waitUntilExit } = render(<App config={config} />, {
    exitOnCtrlC: false,
    patchConsole: false,
  });

  return waitUntilExit().then(() => {
    if (process.stdin.isTTY) {
      process.stdin.setRawMode(false);
      process.stdin.pause();
    }
  });
}

function App({ config }: TuiAppProps) {
  const client = useMemo(
    () => new InsightClient(config.server, { timeout: config.timeout }),
    [config.server, config.timeout]
  );
  const { exit } = useApp();
  const [nav, setNav] = useState<NavigationState>({ screen: 'sessions', stack: [] });
  const [compareIds, setCompareIds] = useState<string[]>([]);

  useInput((input, key) => {
    if (input === 'q' || (key.ctrl && input === 'c')) {
      exit();
    }
  });

  const currentScreen = nav.screen;

  return (
    <Box flexDirection="column" height="100%">
      <StatusBar client={client} nav={nav} config={config} />
      <Box flexGrow={1}>
        {currentScreen === 'sessions' && (
          <SessionList client={client} onSelect={(id) => setNav({ screen: 'session', stack: [...nav.stack, nav], taskId: id })} />
        )}
        {currentScreen === 'session' && nav.taskId && (
          <SessionDetail client={client} taskId={nav.taskId} onBack={() => { const prev = nav.stack[nav.stack.length - 1]; setNav(prev ?? { screen: 'sessions', stack: [] }); }} />
        )}
        {currentScreen === 'turn' && nav.turnId && (
          <TurnDetail client={client} turnId={nav.turnId} onBack={() => setNav({ ...nav, screen: 'session' })} />
        )}
        {currentScreen === 'compare' && (
          <CompareView client={client} taskIds={compareIds} onBack={() => setNav({ screen: 'sessions', stack: [] })} />
        )}
        {currentScreen === 'import' && (
          <ImportPanel client={client} onBack={() => setNav({ screen: 'sessions', stack: [] })} />
        )}
        {currentScreen === 'help' && (
          <HelpScreen onBack={() => { const prev = nav.stack[nav.stack.length - 1]; setNav(prev ?? { screen: 'sessions', stack: [] }); }} />
        )}
      </Box>
      <KeyBar screen={currentScreen} />
    </Box>
  );
}
```

### 3.2 StatusBar — 顶部状态栏

```typescript
// src/cli/tui/components/StatusBar.tsx

interface StatusBarProps {
  client: InsightClient;
  nav: NavigationState;
  config: CliConfig;
}

function StatusBar({ client, nav, config }: StatusBarProps) {
  const { data: stats } = useApi(() => client.getStats(), [client]);
  return (
    <Box borderStyle="single" paddingX={1}>
      <Text bold color="cyan">🤖 CANNBot-Insight</Text>
      <Text> {VERSION_DISPLAY}</Text>
      <Text> │ Sessions: {stats?.totalSessions ?? '…'}</Text>
      <Text> │ Cost: {formatCost(stats?.totalCost ?? 0)}</Text>
      <Text> │ {nav.screen !== 'sessions' ? `← ${nav.screen}` : ''}</Text>
    </Box>
  );
}
```

### 3.3 KeyBar — 底部快捷键提示

```typescript
// src/cli/tui/components/KeyBar.tsx

const KEY_HINTS: Record<Screen, string[]> = {
  sessions: ['↑↓ Navigate', 'Enter: Detail', 'Space: Select', 'c: Compare', 'i: Import', 'd: Delete', 'q: Quit'],
  session: ['1-7: Switch Tab', '[/]: Prev/Next Tab', 'Enter: Drill-down', 'a: AI Analyze', 'y: Copy ID', 'Esc: Back', 'q: Quit'],
  turn: ['Esc: Back', 'q: Quit'],
  compare: ['Esc: Back', 'q: Quit'],
  import: ['↑↓ Select', 'Enter: Import', 'Esc: Back', 'q: Quit'],
  help: ['Esc: Back', 'q: Quit'],
};

interface KeyBarProps { screen: Screen; }

function KeyBar({ screen }: KeyBarProps) {
  const hints = KEY_HINTS[screen] ?? [];
  return (
    <Box borderStyle="single" paddingX={1}>
      <Text color="gray">{hints.join(' │ ')}</Text>
    </Box>
  );
}
```

### 3.4 DataTable — 通用数据表格组件

```typescript
// src/cli/tui/components/DataTable.tsx

interface DataTableProps<T> {
  columns: Array<{
    key: string;
    label: string;
    width: number;
    render?: (row: T, selected: boolean) => string;
  }>;
  data: T[];
  selectedIndex: number;
  onSelect?: (index: number) => void;
}

function DataTable<T>({ columns, data, selectedIndex, onSelect }: DataTableProps<T>) {
  return (
    <Box flexDirection="column">
      {/* Header row */}
      <Box>
        {columns.map(col => (
          <Text bold key={col.key} width={col.width}>{col.label}</Text>
        ))}
      </Box>
      {/* Separator */}
      <Text color="gray">{columns.map(c => '─'.repeat(c.width)).join('─┼─')}</Text>
      {/* Body rows */}
      {data.map((row, i) => {
        const selected = i === selectedIndex;
        return (
          <Box key={i}>
            {columns.map(col => {
              const val = col.render
                ? col.render(row, selected)
                : String((row as Record<string, unknown>)[col.key] ?? '—');
              return (
                <Text
                  key={col.key}
                  width={col.width}
                  color={selected ? 'cyan' : undefined}
                  bold={selected}
                >
                  {val}
                </Text>
              );
            })}
          </Box>
        );
      })}
    </Box>
  );
}
```

### 3.5 MetricCards — 指标卡片行

```typescript
// src/cli/tui/components/MetricCards.tsx

interface MetricCardProps {
  metrics: Array<{ label: string; value: string; color?: string }>;
}

function MetricCards({ metrics }: MetricCardProps) {
  return (
    <Box gap={1}>
      {metrics.map(m => (
        <Box borderStyle="round" paddingX={1} key={m.label}>
          <Text bold color={m.color ?? 'cyan'}>{m.value}</Text>
          <Text color="gray"> {m.label}</Text>
        </Box>
      ))}
    </Box>
  );
}
```

### 3.6 AsciiBar — ASCII 柱状图

```typescript
// src/cli/tui/components/AsciiBar.tsx

interface AsciiBarProps {
  label: string;
  value: number;
  max: number;
  width: number; // bar 宽度（字符数）
  color?: string;
  warningThreshold?: number;
  criticalThreshold?: number;
}

function AsciiBar({ label, value, max, width, color, warningThreshold, criticalThreshold }: AsciiBarProps) {
  const pct = Math.min(value / max, 1);
  const filled = Math.round(pct * width);
  const empty = width - filled;
  const bar = '█'.repeat(filled) + '░'.repeat(empty);

  let barColor = color ?? 'green';
  if (criticalThreshold && pct >= criticalThreshold) barColor = 'red';
  else if (warningThreshold && pct >= warningThreshold) barColor = 'yellow';

  const pctStr = `${(pct * 100).toFixed(1)}%`;
  return (
    <Box>
      <Text bold width={12}>{label}</Text>
      <Text color={barColor}>{bar}</Text>
      <Text> {formatTokens(value)} ({pctStr})</Text>
    </Box>
  );
}
```

### 3.7 TreeView — ASCII 树形图

```typescript
// src/cli/tui/components/TreeView.tsx

interface TreeNode {
  label: string;
  icon?: string;
  children?: TreeNode[];
  detail?: string;
}

interface TreeViewProps {
  tree: TreeNode[];
  indent?: number;
}

function TreeView({ tree, indent = 0 }: TreeViewProps) {
  return (
    <Box flexDirection="column">
      {tree.map((node, i) => {
        const isLast = i === tree.length - 1;
        const prefix = indent === 0 ? '' : (isLast ? '└── ' : '├── ');
        const childPrefix = indent === 0 ? '' : (isLast ? '    ' : '│   ');

        return (
          <Box flexDirection="column" key={i}>
            <Box>
              <Text color="gray">{prefix}</Text>
              {node.icon && <Text>{node.icon} </Text>}
              <Text bold>{node.label}</Text>
              {node.detail && <Text color="gray"> {node.detail}</Text>}
            </Box>
            {node.children && (
              <TreeView tree={node.children} indent={indent + 1} />
            )}
          </Box>
        );
      })}
    </Box>
  );
}
```

### 3.8 TabBar — Tab 切换栏

```typescript
// src/cli/tui/components/TabBar.tsx

interface TabBarProps {
  tabs: string[];
  activeIndex: number;
  onSwitch: (index: number) => void;
}

function TabBar({ tabs, activeIndex }: TabBarProps) {
  return (
    <Box>
      {tabs.map((tab, i) => (
        <Box key={i} paddingX={1}>
          <Text
            bold={i === activeIndex}
            color={i === activeIndex ? 'cyan' : 'gray'}
            underline={i === activeIndex}
          >
            [{tab}]
          </Text>
        </Box>
      ))}
    </Box>
  );
}
```

### 3.9 ConfirmDialog — 确认对话框

```typescript
// src/cli/tui/components/ConfirmDialog.tsx

interface ConfirmDialogProps {
  message: string;
  onConfirm: () => void;
  onCancel: () => void;
}

function ConfirmDialog({ message, onConfirm, onCancel }: ConfirmDialogProps) {
  useInput((input) => {
    if (input === 'y' || input === 'Y') onConfirm();
    if (input === 'n' || input === 'N' || input === 'Escape') onCancel();
  });

  return (
    <Box borderStyle="double" padding={1} flexDirection="column">
      <Text bold color="yellow">{message}</Text>
      <Text color="gray">y: Confirm │ n: Cancel</Text>
    </Box>
  );
}
```

### 3.10 Spinner — 自写加载动画

> **PoC 验证**: ink-spinner 不兼容 Ink v7，自写仅需 10 行代码。

```typescript
// src/cli/tui/components/Spinner.tsx

const SPINNER_FRAMES = ['⠋','⠙','⠹','⠸','⠼','⠴','⠦','⠧','⠇','⠏'];

interface SpinnerProps {
  label?: string;
}

function Spinner({ label }: SpinnerProps) {
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const timer = setInterval(() => setFrame(f => (f + 1) % SPINNER_FRAMES.length), 80);
    return () => clearInterval(timer);
  }, []);
  return <Text>{SPINNER_FRAMES[frame]} {label ?? 'Loading...'}</Text>;
}
```

### 3.11 TextInput — 自写输入框

> **PoC 验证**: ink-text-input 不兼容 Ink v7，自写仅需 30 行代码。

```typescript
// src/cli/tui/components/TextInput.tsx

interface TextInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  focus?: boolean;
  onSubmit?: (value: string) => void;
}

function TextInput({ value, onChange, placeholder, focus = true, onSubmit }: TextInputProps) {
  useInput((input, key) => {
    if (!focus) return;
    if (key.backspace || key.delete) onChange(value.slice(0, -1));
    else if (key.return) onSubmit?.(value);
    else if (!key.ctrl && !key.meta) onChange(value + input);
  }, { isActive: focus });
  return (
    <Box>
      {value.length === 0 && placeholder && <Text color="gray">{placeholder}</Text>}
      <Text>{value}</Text>
      <Text color="gray">█</Text>
    </Box>
  );
}
```

---

## 4. 状态管理设计

### 4.1 屏幕导航栈

```typescript
// src/cli/hooks/useNavigation.ts

export type Screen = 'sessions' | 'session' | 'turn' | 'compare' | 'import' | 'help';

export interface NavigationState {
  screen: Screen;
  stack: NavigationState[];
  taskId?: string;
  turnId?: string;
  tab?: string;
}

export function useNavigation(): {
  nav: NavigationState;
  navigate: (screen: Screen, params?: Record<string, string>) => void;
  goBack: () => void;
} {
  const [nav, setNav] = useState<NavigationState>({ screen: 'sessions', stack: [] });

  const navigate = (screen: Screen, params?: Record<string, string>) => {
    setNav(prev => ({
      screen,
      stack: [...prev.stack, prev],
      ...params,
    }));
  };

  const goBack = () => {
    const prev = nav.stack[nav.stack.length - 1];
    if (prev) setNav(prev);
  };

  return { nav, navigate, goBack };
}
```

### 4.2 选中状态管理

```typescript
// src/cli/hooks/useTable.ts

export interface TableState<T> {
  selectedIndex: number;
  sortKey: string;
  sortDir: 'asc' | 'desc';
  filterText: string;
  data: T[];
  visibleData: T[];
}

export function useTable<T>(
  data: T[],
  columns: Array<{ key: string; sortable?: boolean; filterable?: boolean }>
): {
  state: TableState<T>;
  selectUp: () => void;
  selectDown: () => void;
  sortBy: (key: string) => void;
  filter: (text: string) => void;
  select: (index: number) => void;
} {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [sortKey, setSortKey] = useState(columns[0]?.key ?? '');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [filterText, setFilterText] = useState('');

  const visibleData = useMemo(() => {
    let filtered = data;
    if (filterText) {
      filtered = data.filter(row =>
        columns.some(col => {
          if (!col.filterable) return false;
          const val = String((row as Record<string, unknown>)[col.key] ?? '');
          return val.toLowerCase().includes(filterText.toLowerCase());
        })
      );
    }
    const sorted = [...filtered].sort((a, b) => {
      const aVal = (a as Record<string, unknown>)[sortKey];
      const bVal = (b as Record<string, unknown>)[sortKey];
      const cmp = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return sortDir === 'asc' ? cmp : -cmp;
    });
    return sorted;
  }, [data, sortKey, sortDir, filterText, columns]);

  const selectUp = () => setSelectedIndex(Math.max(0, selectedIndex - 1));
  const selectDown = () => setSelectedIndex(Math.min(visibleData.length - 1, selectedIndex + 1));
  const sortBy = (key: string) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc');
    else { setSortKey(key); setSortDir('desc'); }
  };
  const filter = (text: string) => {
    setFilterText(text);
    setSelectedIndex(0);
  };

  return {
    state: { selectedIndex, sortKey, sortDir, filterText, data, visibleData },
    selectUp, selectDown, sortBy, filter,
    select: setSelectedIndex,
  };
}
```

### 4.3 API 数据缓存策略

```typescript
// src/cli/hooks/useApi.ts

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

const cache = new Map<string, CacheEntry<unknown>>();

const DEFAULT_TTL = 30_000; // 30 秒缓存

export function useApi<T>(
  fetcher: () => Promise<T>,
  deps: unknown[],
  ttl: number = DEFAULT_TTL,
): {
  data: T | null;
  loading: boolean;
  error: Error | null;
  refresh: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  const cacheKey = useMemo(() => JSON.stringify(deps), [deps]);

  const fetchData = useCallback(async () => {
    // 检查缓存
    const cached = cache.get(cacheKey) as CacheEntry<T> | undefined;
    if (cached && Date.now() - cached.timestamp < cached.ttl) {
      setData(cached.data);
      setLoading(false);
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const result = await fetcher();
      setData(result);
      cache.set(cacheKey, { data: result, timestamp: Date.now(), ttl });
    } catch (err) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  }, [fetcher, cacheKey]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const refresh = useCallback(() => {
    cache.delete(cacheKey);
    fetchData();
  }, [cacheKey, fetchData]);

  return { data, loading, error, refresh };
}
```

**缓存 TTL 策略**：

| 数据类型 | TTL | 说明 |
|----------|-----|------|
| Session 列表 | 30s | 定期刷新，可能新增 session |
| Session 详情 | 60s | 详情相对稳定 |
| Stats | 30s | 统计可能变化 |
| Workflow/Bridges | 120s | 计算型数据，极少变化 |
| Turn 详情 | 无缓存 | 每次进入重新加载 |

### 4.4 键盘事件处理

```typescript
// src/cli/hooks/useKeyboard.ts

import { useInput } from 'ink';

export interface KeyboardHandler {
  onNavigateUp?: () => void;
  onNavigateDown?: () => void;
  onEnter?: () => void;
  onEscape?: () => void;
  onRefresh?: () => void;
  onSearch?: () => void;
  onHelp?: () => void;
  onTabSwitch?: (direction: 'prev' | 'next') => void;
  onDigit?: (n: number) => void;
  custom?: Record<string, () => void>;
}

export function useKeyboard(handler: KeyboardHandler): void {
  useInput((input, key) => {
    if (key.upArrow || input === 'k') handler.onNavigateUp?.();
    else if (key.downArrow || input === 'j') handler.onNavigateDown?.();
    else if (key.return) handler.onEnter?.();
    else if (key.escape) handler.onEscape?.();
    else if (input === 'r') handler.onRefresh?.();
    else if (input === '/') handler.onSearch?.();
    else if (input === '?' || input === 'F1') handler.onHelp?.();
    else if (input === '[') handler.onTabSwitch?.('prev');
    else if (input === ']') handler.onTabSwitch?.('next');
    else if (input >= '1' && input <= '7') handler.onDigit?.(+input);
    else if (handler.custom?.[input]) handler.custom[input]();
  });
}
```

---

## 5. TUI Screen 设计

### 5.1 SessionList — Session 列表屏

```typescript
// src/cli/tui/screens/SessionList.tsx

interface SessionListProps {
  client: InsightClient;
  onSelect: (taskId: string) => void;
}

const SESSION_COLUMNS: TableColumn<SessionListItem>[] = [
  { key: '#', label: '#', width: 3, render: (_, __, idx) => String(idx + 1) },
  { key: 'startTime', label: 'Date', width: 14, render: (r) => formatDate(r.startTime) },
  { key: 'user', label: 'User', width: 12, render: (r) => truncateVisual(r.user, 12) },
  { key: 'query', label: 'Query', width: 30, render: (r) => truncateVisual(r.query ?? '—', 30) },
  { key: 'model', label: 'Model', width: 16, render: (r) => truncateVisual(r.model ?? '—', 16) },
  { key: 'totalTokens', label: 'Tokens', width: 8, render: (r) => formatTokens(r.totalTokens) },
  { key: 'totalCost', label: 'Cost', width: 8, render: (r) => formatCost(r.totalCost) },
  { key: 'totalToolCallCount', label: 'Tools', width: 6, render: (r) => String(r.totalToolCallCount) },
];

function SessionList({ client, onSelect }: SessionListProps) {
  const [page, setPage] = useState(1);
  const PAGE_SIZE = 20;
  const fetchSessions = useCallback(() => client.listSessions({ page, pageSize: PAGE_SIZE }), [client, page]);
  const { data, loading, error, refresh } = useApi(fetchSessions, [client, page]);
  const table = useTable(data?.items ?? [], SESSION_COLUMNS);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [showImport, setShowImport] = useState(false);

  useKeyboard({
    onNavigateUp: table.selectUp,
    onNavigateDown: table.selectDown,
    onEnter: () => {
      const selected = table.state.visibleData[table.state.selectedIndex];
      if (selected) onSelect(selected.taskId);
    },
    onRefresh: refresh,
    onSearch: () => { /* 进入搜索过滤模式 */ },
    custom: {
      ' ': () => { /* Space: toggle compare selection */ },
      'c': () => { /* Compare selected sessions */ },
      'i': () => setShowImport(true),
      'd': () => { /* Delete confirmation */ },
      's': () => table.sortBy(SESSION_COLUMNS[table.state.selectedIndex % SESSION_COLUMNS.length].key),
      'f': () => { /* Filter prompt */ },
      'n': () => setPage(p => p + 1),
      'p': () => setPage(p => Math.max(1, p - 1)),
    },
  });

  if (loading) return <Spinner label="Loading sessions..." />;
  if (error) return <Text color="red">Error: {error.message}</Text>;

  return (
    <Box flexDirection="column">
      <DataTable
        columns={SESSION_COLUMNS}
        data={table.state.visibleData}
        selectedIndex={table.state.selectedIndex}
      />
      <Text color="gray">Page {page}/{Math.ceil((data?.total ?? 0) / PAGE_SIZE)} │ n: Next │ p: Prev │ Total: {data?.total ?? 0}</Text>
      {/* Preview */}
      {table.state.visibleData[table.state.selectedIndex] && (
        <PreviewPanel session={table.state.visibleData[table.state.selectedIndex]} />
      )}
    </Box>
  );
}
```

### 5.2 SessionDetail — Session 详情屏

```typescript
// src/cli/tui/screens/SessionDetail.tsx

const TAB_NAMES = ['Overview', 'Turns', 'Workflow', 'Subagents', 'Skills', 'Bridges', 'Context'] as const;
type TabName = typeof TAB_NAMES[number];

interface SessionDetailProps {
  client: InsightClient;
  taskId: string;
  onBack: () => void;
}

function SessionDetail({ client, taskId, onBack }: SessionDetailProps) {
  const [activeTab, setActiveTab] = useState<number>(0);
  const { data: session } = useApi(() => client.getSession(taskId), [client, taskId]);
  const { data: stats } = useApi(() => client.getStats(taskId), [client, taskId], 60000);

  useKeyboard({
    onEscape: onBack,
    onTabSwitch: (dir) => {
      setActiveTab(dir === 'next'
        ? Math.min(activeTab + 1, TAB_NAMES.length - 1)
        : Math.max(activeTab - 1, 0)
      );
    },
    onDigit: (n) => {
      if (n >= 1 && n <= TAB_NAMES.length) setActiveTab(n - 1);
    },
    custom: {
      'a': () => { /* Trigger AI analysis */ },
      'y': () => { /* Copy taskId */ },
    },
  });

  if (!session || !stats) return <Spinner label="Loading session..." />;

  const tabContent = (() => {
    switch (TAB_NAMES[activeTab]) {
      case 'Overview': return <OverviewTab session={session} stats={stats} />;
      case 'Turns': return <TurnsTab client={client} taskId={taskId} />;
      case 'Workflow': return <WorkflowTab client={client} taskId={taskId} />;
      case 'Subagents': return <SubagentsTab client={client} taskId={taskId} />;
      case 'Skills': return <SkillsTab session={session} />;
      case 'Bridges': return <BridgesTab client={client} taskId={taskId} />;
      case 'Context': return <ContextTab client={client} taskId={taskId} />;
    }
  })();

  return (
    <Box flexDirection="column">
      {/* Header */}
      <Box paddingX={1}>
        <Text bold color="cyan">Session: {truncate(taskId, 12)}</Text>
        <Text> │ {session.model ?? '—'} │ {formatDuration(stats.totalLatencyMs)} │ {formatCost(stats.totalCost)}</Text>
      </Box>
      <TabBar tabs={TAB_NAMES} activeIndex={activeTab} onSwitch={setActiveTab} />
      <Box flexGrow={1}>{tabContent}</Box>
    </Box>
  );
}
```

### 5.3 OverviewTab

```typescript
// src/cli/tui/tabs/OverviewTab.tsx

interface OverviewTabProps {
  session: SessionDetailResponse;
  stats: SessionStatsResponse;
}

function OverviewTab({ session, stats }: OverviewTabProps) {
  return (
    <Box flexDirection="column" gap={1}>
      <MetricCards metrics={[
        { label: 'Tokens', value: formatTokens(stats.totalTokens), color: 'cyan' },
        { label: 'Cost', value: formatCost(stats.totalCost), color: 'yellow' },
        { label: 'Duration', value: formatDuration(stats.totalLatencyMs), color: 'green' },
        { label: 'LLM Calls', value: String(stats.totalLlmCallCount), color: 'blue' },
      ]} />

      <Text bold>Token Breakdown:</Text>
      <AsciiBar label="Input" value={stats.totalInputTokens} max={stats.totalTokens} width={20} />
      <AsciiBar label="Output" value={stats.totalOutputTokens} max={stats.totalTokens} width={20} />
      <AsciiBar label="Reasoning" value={stats.totalReasoningTokens} max={stats.totalTokens} width={20} />
      <AsciiBar label="Cache Read" value={stats.totalCacheReadTokens} max={stats.totalTokens} width={20} />

      <Text bold>Executions:</Text>
      <DataTable
        columns={EXECUTION_COLUMNS}
        data={session.agents}
        selectedIndex={0}
      />
    </Box>
  );
}
```

### 5.4 TurnsTab

```typescript
// src/cli/tui/tabs/TurnsTab.tsx

interface TurnsTabProps {
  client: InsightClient;
  taskId: string;
}

const TURN_COLUMNS = [
  { key: 'turnIndex', label: '#', width: 4, render: (r: TurnItem) => String(r.turnIndex) },
  { key: 'role', label: 'Role', width: 10, render: (r: TurnItem) => r.isSubagent ? `🤖 ${r.role}` : r.role },
  { key: 'contentSummary', label: 'Content', width: 30, render: (r: TurnItem) => truncate(r.contentSummary ?? '—', 30) },
  { key: 'model', label: 'Model', width: 14, render: (r: TurnItem) => truncate(r.model ?? '—', 14) },
  { key: 'totalTokens', label: 'Tokens', width: 8, render: (r: TurnItem) => formatTokens(r.totalTokens) },
  { key: 'latencyMs', label: 'Latency', width: 8, render: (r: TurnItem) => formatDuration(r.latencyMs) },
  { key: 'toolCalls', label: 'Tools', width: 6, render: (r: TurnItem) => String(r.toolCalls?.length ?? 0) },
];

function TurnsTab({ client, taskId }: TurnsTabProps) {
  const { data, loading, refresh } = useApi(() => client.getTurns(taskId), [client, taskId]);
  const table = useTable(data?.items ?? [], TURN_COLUMNS);

  useKeyboard({
    onNavigateUp: table.selectUp,
    onNavigateDown: table.selectDown,
    onEnter: () => {
      const turn = table.state.visibleData[table.state.selectedIndex];
      if (turn) navigate('turn', { turnId: turn.turnId });
    },
    onRefresh: refresh,
  });

  if (loading) return <Spinner label="Loading turns..." />;

  return <DataTable columns={TURN_COLUMNS} data={table.state.visibleData} selectedIndex={table.state.selectedIndex} />;
}
```

### 5.5 WorkflowTab — ASCII 树形图

```typescript
// src/cli/tui/tabs/WorkflowTab.tsx

interface WorkflowTabProps {
  client: InsightClient;
  taskId: string;
}

function workflowToTree(workflow: WorkflowTree): TreeNode[] {
  return workflow.phases.map(phase => ({
    label: `Phase ${phase.phaseIndex}: ${phase.phaseName}`,
    icon: '📁',
    detail: `${formatTokens(phase.totalTokens)} tokens, ${formatDuration(phase.durationMs)}, ¥${phase.totalCost.toFixed(2)}`,
    children: phase.children.map(child => {
      if (child.type === 'step') {
        const step = child as WorkflowStepNode;
        return {
          label: `[Turn] ${step.stepLabel}`,
          icon: step.subagentSessionId ? '🤖' : '🔧',
          detail: `${formatTokens(step.totalTokens)} tokens, ${formatDuration(step.durationMs)}`,
        };
      }
      if (child.type === 'checkpoint') {
        const cp = child as WorkflowCheckpointNode;
        return {
          label: `Checkpoint: ${cp.checkpointLabel}`,
          icon: '✅',
          detail: cp.waitTimeMs ? `wait ${formatDuration(cp.waitTimeMs)}` : '',
        };
      }
      if (child.type === 'parallel-group') {
        const pg = child as WorkflowParallelGroupNode;
        return {
          label: `Parallel: ${pg.label}`,
          icon: '⚡',
          detail: `${formatDuration(pg.totalDurationMs)}`,
          children: pg.steps.map(step => ({
            label: step.stepLabel,
            detail: `${formatTokens(step.totalTokens)} tokens`,
          })),
        };
      }
      return { label: 'unknown' };
    }),
  }));
}

function WorkflowTab({ client, taskId }: WorkflowTabProps) {
  const { data, loading } = useApi(() => client.getWorkflow(taskId), [client, taskId]);

  if (loading) return <Spinner label="Loading workflow..." />;
  if (!data) return <Text color="gray">No workflow data</Text>;

  const tree = workflowToTree(data);

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">Workflow — {data.summary.totalPhases} phases, {data.summary.totalSteps} steps</Text>
      <TreeView tree={tree} />
    </Box>
  );
}
```

### 5.6 ContextTab — 上下文增长追踪

```typescript
// src/cli/tui/tabs/ContextTab.tsx

interface ContextTabProps {
  client: InsightClient;
  taskId: string;
}

function ContextTab({ client, taskId }: ContextTabProps) {
  const { data, loading } = useApi(() => client.getTurns(taskId), [client, taskId]);

  if (loading) return <Spinner label="Loading context data..." />;
  if (!data) return <Text color="gray">No data</Text>;

  const turns = data.items.filter(t => t.contextWindowPct !== null);
  const maxPct = Math.max(...turns.map(t => t.contextWindowPct ?? 0));

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">Context Window Usage</Text>
      {turns.map(turn => {
        const pct = turn.contextWindowPct ?? 0;
        let icon = '';
        if (pct >= 95) icon = '🔴';
        else if (pct >= 80) icon = '⚠️';
        return (
          <AsciiBar
            key={turn.turnId}
            label={`Turn ${turn.turnIndex}`}
            value={turn.inputMessagesTokens ?? 0}
            max={128000}
            width={25}
            warningThreshold={0.8}
            criticalThreshold={0.95}
          />
        );
      })}
      {/* Warning messages */}
      {turns.filter(t => (t.contextWindowPct ?? 0) >= 80).map(t => (
        <Text key={t.turnId} color={(t.contextWindowPct ?? 0) >= 95 ? 'red' : 'yellow'}>
          {(t.contextWindowPct ?? 0) >= 95 ? '🔴' : '⚠️'} Turn {t.turnIndex}: Context {(t.contextWindowPct ?? 0).toFixed(1)}%
        </Text>
      ))}
    </Box>
  );
}
```

---

## 6. 错误处理策略

### 6.1 命令模式错误处理

```typescript
// src/cli/commands/中的统一错误处理

async function safeAction(action: () => Promise<void>): Promise<void> {
  try {
    await action();
  } catch (err) {
    if (err instanceof ApiError) {
      if (err.status === 404) {
        console.error(`❌ Not found: ${err.message}`);
      } else if (err.status === 400) {
        console.error(`❌ Bad request: ${err.message}`);
      } else {
        console.error(`❌ Server error: ${err.message}`);
      }
    } else if (err instanceof NetworkError) {
      console.error(`❌ Network error: ${err.message}`);
      console.error(`   Check: --server URL is correct and backend is running`);
    } else if (err instanceof InsightError) {
      console.error(`❌ ${err.message}`);
    } else {
      console.error(err);
    }
    process.exit(1);
  }
}
```

### 6.2 TUI 模式错误处理

```typescript
// TUI 中错误显示为内联消息

function ErrorBanner({ error: Error | null }) {
  if (!error) return null;
  const color = error instanceof NetworkError ? 'yellow' : 'red';
  const prefix = error instanceof NetworkError ? '⚠️' : '❌';
  return (
    <Box borderStyle="single" borderColor={color} paddingX={1}>
      <Text color={color}>{prefix} {error.message}</Text>
    </Box>
  );
}
```

### 6.3 终端兼容性检查

```typescript
// src/cli/utils/terminal.ts

export function checkTerminalSupport(): { supported: boolean; warnings: string[] } {
  const warnings: string[] = [];
  let supported = true;

  const minWidth = 80;
  const minHeight = 24;
  const cols = process.stdout.columns ?? 80;
  const rows = process.stdout.rows ?? 24;

  if (cols < minWidth) {
    warnings.push(`Terminal width ${cols} < minimum ${minWidth}. Tables may be truncated.`);
  }
  if (rows < minHeight) {
    warnings.push(`Terminal height ${rows} < minimum ${minHeight}. TUI may not display properly.`);
  }

  if (!process.stdout.isTTY) {
    warnings.push('Not running in a TTY. TUI mode requires interactive terminal.');
    supported = false;
  }

  const term = process.env.TERM ?? '';
  if (!term) {
    warnings.push('$TERM not set. Colors may not work.');
  }

  return { supported, warnings };
}
```

---

## 7. 配置管理

### 7.1 配置优先级

```
命令行参数 (--server, --timeout)
  ↓ 覆盖
环境变量 (CANNBOT_SERVER, CANNBOT_TIMEOUT)
  ↓ 覆盖
配置文件 (~/.cannbot-insight/config.json)
  ↓ 覆盖
默认值 (DEFAULT_SERVER_URL=http://localhost:21025)
```

### 7.2 配置文件结构

```json
// ~/.cannbot-insight/config.json
{
  "server": "http://192.168.1.100:21025",
  "timeout": 30000,
  "theme": "dark",
  "keybindings": {
    "quit": "q",
    "help": "?",
    "refresh": "r"
  }
}
```

### 7.3 config 命令实现

```typescript
// src/cli/commands/config.ts

export function configCommand(): Command {
  const cmd = new Command('config');
  cmd.description('Configuration management');

  cmd
    .command('set <key> <value>')
    .description('Set a config value')
    .action((key, value) => {
      saveConfig({ [key]: value });
      console.log(`✅ Set ${key} = ${value}`);
    });

  cmd
    .command('get <key>')
    .description('Get a config value')
    .action((key) => {
      const config = loadConfig();
      console.log(config[key as keyof CliConfig] ?? 'Not set');
    });

  cmd
    .command('list')
    .description('List all config values')
    .action(() => {
      const config = loadConfig();
      console.log(JSON.stringify(config, null, 2));
    });

  cmd
    .command('context-windows')
    .description('View context window configuration')
    .action(async (_, cmd) => {
      const globalOpts = cmd.parent.parent.opts();
      const client = new InsightClient(globalOpts.server);
      const stats = await client.getStats();
      console.log('Context Window Sizes (from MODEL_CONTEXT_WINDOWS):');
      console.log('  claude-3.5-sonnet: 200K');
      console.log('  claude-3-opus: 200K');
      console.log('  gpt-4o: 128K');
      console.log('  glm-5: 128K (default)');
      console.log('  ...');
    });

  return cmd;
}
```

---

## 8. 性能考虑

### 8.1 分页加载

> **评审决策**: 虚拟滚动在 Ink 环境下收益有限（只减少 React reconcile 成本，不减少终端 I/O），改为服务端分页 + 前端翻页。

DataTable 采用分页模式（每页 20 条），通过 n/p 键翻页：

```typescript
// SessionList 分页示例
const [page, setPage] = useState(1);
const PAGE_SIZE = 20;

const { data } = useApi(
  useCallback(() => client.listSessions({ page, pageSize: PAGE_SIZE }), [client, page]),
  [client, page]
);

useKeyboard({
  custom: {
    'n': () => setPage(p => p + 1),
    'p': () => setPage(p => Math.max(1, p - 1)),
  },
});

// DataTable 不再需要虚拟滚动，每页最多 20 行
<Text color="gray">Page {page}/{Math.ceil((data?.total ?? 0) / PAGE_SIZE)} │ n: Next │ p: Prev │ Total: {data?.total ?? 0}</Text>
```

### 8.2 数据缓存

- `useApi` hook 内置 Map 缓存，按 deps 序列化作为 key
- Session 列表 30s TTL，Session 详情 60s TTL
- `r` 键触发 `refresh()` 清除缓存并重新 fetch
- Turn 详情不缓存，每次进入重新加载

### 8.3 懒加载策略

- Session 列表页：每页 20 条（`pageSize: 20`），n/p 键翻页
- Tab 内容：仅在切换到该 Tab 时才 fetch 数据
- Workflow/Context 数据：只在 Tab 被选中时请求
- Turn 详情：只在 Enter 钻取时请求

```typescript
// SessionDetail 中 Tab 懒加载
function SessionDetail({ client, taskId }) {
  const [activeTab, setActiveTab] = useState(0);
  // Overview 和 Turns 在首次渲染时 fetch
  // 其他 Tab 在切换时才 fetch
  switch (TAB_NAMES[activeTab]) {
    case 'Overview': return <OverviewTab ... />; // 已有数据
    case 'Turns': return <TurnsTab ... />;       // 已有数据
    case 'Workflow': return <WorkflowTab ... />;  // 按需 fetch
    case 'Subagents': return <SubagentsTab ... />;// 按需 fetch
    case 'Context': return <ContextTab ... />;    // 按需 fetch
  }
}
```

### 8.4 网络请求优化

- 并行请求：Session 详情页首次加载时并行 fetch `getSession()` + `getStats()`
- Compare 命令：并行 fetch 两个 session 的数据
- Import 预览：先 `listImportableSessions()` 再选择性 `importSession()`

```typescript
// 并行请求示例
const [session, stats] = await Promise.all([
  client.getSession(taskId),
  client.getStats(taskId),
]);
```

### 8.5 中文宽度对齐

CJK 字符（中文、日文、韩文）在终端中占 2 列宽，而 ASCII 字符占 1 列宽。Ink 的 `<Text width={N}>` 按 N 个字符位计算，不区分 CJK 双宽字符，导致含中文的列错位。

**PoC 验证结果**: 表格渲染时中文列对齐有偏差，`width` 属性按字符数而非视觉宽度计算。

**解决方案**:

1. **格式化层**: `truncateVisual()` 和 `padEndVisual()` 使用 `string-width` 库计算视觉宽度
2. **表格层**: `renderTable()` 使用视觉宽度而非字符长度
3. **TUI 层**: DataTable 组件中 `<Text width={N}>` 仍按 Ink 规则，但列定义的 `width` 值需要考虑中文占比；关键列使用英文标签
4. **依赖**: `string-width` + `cli-truncate`（Sindre Sorhus 维护的轻量 ESM 库）

### 8.6 Ink v7 环境要求

**PoC 验证**: Ink 版本为 v7.0.6（不是 v5），React 19.2.7。

| 要求 | 说明 |
|------|------|
| ESM | package.json `"type": "module"`（yoga-layout 使用 top-level await，CJS 报错） |
| 运行工具 | tsx（不是 ts-node） |
| render() API | Ink v7 原生 render() 没有 lastFrame/frames/output；只有 rerender/unmount/waitUntilExit/waitUntilRenderFlush/cleanup/clear |
| 测试 render | ink-testing-library 的 render() 有 lastFrame + stdin（PoC 验证兼容 v7）— 测试用 ink-testing-library，生产用 Ink 原生 render() |
| 第三方组件 | ink-table/ink-select/ink-spinner/ink-text-input 等均不兼容 v7，全部自实现 |
| 中文宽度 | `<Text width={N}>` 按字符数而非视觉宽度计算，需要 string-width + padEndVisual/truncateVisual |
| 版本号 | 从 `@/lib/version.ts` 导入 VERSION/VERSION_DISPLAY，不硬编码 |

---

## 9. 命令模式渲染示例

### 9.1 sessions 命令渲染

```typescript
// src/cli/commands/sessions.ts — 表格渲染

function renderSessionTable(items: SessionListItem[], total: number): void {
  const columns = [
    { key: '#', label: '#', width: 4 },
    { key: 'startTime', label: 'Date', width: 14 },
    { key: 'user', label: 'User', width: 12 },
    { key: 'query', label: 'Query', width: 30 },
    { key: 'model', label: 'Model', width: 16 },
    { key: 'totalTokens', label: 'Tokens', width: 8 },
    { key: 'totalCost', label: 'Cost', width: 8 },
    { key: 'totalToolCallCount', label: 'Tools', width: 6 },
  ];

  const rows = items.map((item, i) => ({
    '#': String(i + 1),
    startTime: formatDate(item.startTime),
    user: truncateVisual(item.user, 12),
    query: truncateVisual(item.query ?? '—', 30),
    model: truncateVisual(item.model ?? '—', 16),
    totalTokens: formatTokens(item.totalTokens),
    totalCost: formatCost(item.totalCost),
    totalToolCallCount: String(item.totalToolCallCount),
  }));

  console.log(renderTable(columns, rows));
  console.log(`\n  Total: ${total} sessions | ${formatTokens(items.reduce((s, i) => s + i.totalTokens, 0))} tokens | ${formatCost(items.reduce((s, i) => s + i.totalCost, 0))} cost`);
}
```

### 9.2 compare 命令渲染

```typescript
// src/cli/commands/compare.ts

async function compareAction(id1: string, id2: string, opts: any, cmd: Command) {
  const globalOpts = cmd.parent.opts();
  const client = new InsightClient(globalOpts.server);

  const [session1, session2] = await Promise.all([
    client.getSession(id1),
    client.getSession(id2),
  ]);

  const [stats1, stats2] = await Promise.all([
    client.getStats(id1),
    client.getStats(id2),
  ]);

  if (opts.json) {
    console.log(JSON.stringify({ s1: { session: session1, stats: stats1 }, s2: { session: session2, stats: stats2 } }, null, 2));
    return;
  }

  const metrics = [
    { label: 'Tokens', v1: formatTokens(stats1.totalTokens), v2: formatTokens(stats2.totalTokens) },
    { label: 'Cost', v1: formatCost(stats1.totalCost), v2: formatCost(stats2.totalCost) },
    { label: 'Duration', v1: formatDuration(stats1.totalLatencyMs), v2: formatDuration(stats2.totalLatencyMs) },
    { label: 'Turns', v1: String(stats1.totalLlmCallCount), v2: String(stats2.totalLlmCallCount) },
    { label: 'Tools', v1: String(stats1.totalToolCallCount), v2: String(stats2.totalToolCallCount) },
    { label: 'Subagents', v1: String(stats1.totalSubagentCount), v2: String(stats2.totalSubagentCount) },
    { label: 'Model', v1: session1.model ?? '—', v2: session2.model ?? '—' },
  ];

  console.log(renderCompareTable(id1, id2, metrics));
}
```
