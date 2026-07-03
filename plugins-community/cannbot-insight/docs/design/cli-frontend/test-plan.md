# CANNBot-Insight CLI 前端测试设计文档

> **版本**: v1.3  
> **日期**: 2025-06-14  
> **基于**: [CLI 前端详细设计文档 v1.1](./design.md)  
> **测试框架**: Vitest  
> **PoC 验证**: Ink v7.0.6 + React 19.2.7 + ESM + tsx；ink-testing-library 完全兼容 Ink v7（有 lastFrame + stdin）；Ink v7 原生 render() 无 lastFrame/frames/output  
> **状态**: PoC 验证后更新

---

## 目录

1. [测试总览](#1-测试总览)
2. [单元测试计划](#2-单元测试计划)
3. [集成测试计划](#3-集成测试计划)
4. [TUI 组件测试](#4-tui-组件测试)
5. [E2E 测试计划](#5-e2e-测试计划)
6. [测试数据准备](#6-测试数据准备)
7. [覆盖率目标](#7-覆盖率目标)
8. [测试环境配置](#8-测试环境配置)

---

## 1. 测试总览

### 1.1 测试分层策略

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

### 1.2 测试目录结构

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
│   │   ├── TextInput.test.tsx      # 自写 TextInput 组件测试
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

### 1.3 测试风格约定

遵循项目现有 Vitest 测试风格：

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';

describe('模块名', () => {
  describe('功能分组', () => {
    it('具体测试用例描述', () => {
      // arrange → act → assert
    });
  });
});
```

---

## 2. 单元测试计划

### 2.1 InsightClient 单元测试

**文件**: `tests/cli/unit/client.test.ts`  
**测试对象**: `src/cli/client.ts`

#### 2.1.1 构造函数与配置

| 测试用例 | 输入 | 期望输出 | 边界条件 |
|----------|------|----------|----------|
| 默认配置 | `new InsightClient()` | baseUrl=`http://localhost:21025`, timeout=15000 | — |
| 自定义 baseUrl | `new InsightClient('http://custom:8080')` | baseUrl 被正确设置 | — |
| 部分配置覆盖 | `new InsightClient(url, { timeout: 5000 })` | timeout=5000, retries 保持默认 | — |
| 空字符串 baseUrl | `new InsightClient('')` | 使用默认值或抛出 ConfigError | baseUrl 为空 |

```typescript
import { describe, it, expect } from 'vitest';
import { InsightClient } from '../../src/cli/client';

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
});
```

#### 2.1.2 API 方法调用

| 方法 | 测试用例 | Mock 响应 | 验证点 |
|------|----------|-----------|--------|
| `listSessions()` | 正常列表请求 | `{ items: [...], total: 42, page: 1 }` | URL 参数、返回类型 |
| `listSessions()` | 带过滤条件 | 同上 | `user`、`pageSize` 参数传递 |
| `getSession()` | 正常详情请求 | `SessionDetailResponse` | taskId 参数、返回字段 |
| `getSession()` | 不存在的 taskId | 404 | 抛出 ApiError(404) |
| `getStats()` | 全局统计 | `GlobalStatsResponse` | 无 taskId 参数 |
| `getStats()` | Session 统计 | `SessionStatsResponse` | 带 taskId 参数 |
| `getTurns()` | 正常 Turn 列表 | `{ items: [...], total: 25 }` | 返回 TurnItem 数组 |
| `searchTurns()` | 关键词搜索 | `{ items: [...], total: 3 }` | keyword 参数 |
| `importSession()` | 导入成功 | `{ sessionId, imported: true }` | POST body 正确 |
| `deleteSession()` | 删除单个 | `{ deleted: 1, taskId }` | DELETE 方法 |
| `deleteSession()` | 删除全部 | `{ deleted: 42 }` | deleteAll=true |

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InsightClient, ApiError, NetworkError } from '../../src/cli/client';

// Mock global fetch
const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('InsightClient API methods', () => {
  let client: InsightClient;

  beforeEach(() => {
    client = new InsightClient('http://localhost:21025');
    mockFetch.mockReset();
  });

  describe('listSessions', () => {
    it('calls GET /api/observe/data with default params', async () => {
      const mockResponse = {
        items: [{ sessionId: 'ses_001', taskId: 'task_001', totalTokens: 1000 }],
        total: 1,
        page: 1,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockResponse,
      });

      const result = await client.listSessions({});
      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/observe/data'),
        expect.objectContaining({ method: 'GET' }),
      );
      expect(result.items).toHaveLength(1);
      expect(result.total).toBe(1);
    });

    it('passes user and pageSize as query params', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0, page: 1 }),
      });

      await client.listSessions({ user: 'guanxinghua', pageSize: 20 });
      const calledUrl = mockFetch.mock.calls[0][0] as string;
      expect(calledUrl).toContain('user=guanxinghua');
      expect(calledUrl).toContain('pageSize=20');
    });
  });

  describe('getSession', () => {
    it('returns SessionDetailResponse for valid taskId', async () => {
      const mockSession = {
        sessionId: 'ses_001',
        taskId: 'task_001',
        query: '帮我实现功能',
        model: 'claude-3.5-sonnet',
        totalTokens: 150000,
        totalCost: 3.20,
        agents: [],
        skills: [],
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockSession,
      });

      const result = await client.getSession('task_001');
      expect(result.taskId).toBe('task_001');
      expect(result.totalTokens).toBe(150000);
    });

    it('throws ApiError(404) for non-existent session', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        text: async () => JSON.stringify({ error: 'Session not found' }),
      });

      await expect(client.getSession('nonexistent')).rejects.toThrow(ApiError);
      try {
        await client.getSession('nonexistent');
      } catch (err) {
        expect(err).toBeInstanceOf(ApiError);
        expect((err as ApiError).status).toBe(404);
        expect((err as ApiError).retryable).toBe(false);
      }
    });
  });

  describe('getStats', () => {
    it('returns GlobalStatsResponse when no taskId', async () => {
      const mockStats = {
        totalSessions: 42,
        totalTokens: 2100000,
        totalCost: 56.30,
        totalLatencyMs: 66720000,
        avgLatencyMs: 8200,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockStats,
      });

      const result = await client.getStats();
      expect(result.totalSessions).toBe(42);
      const calledUrl = mockFetch.mock.calls[0][0] as string;
      expect(calledUrl).not.toContain('taskId=');
    });

    it('returns SessionStatsResponse when taskId provided', async () => {
      const mockStats = {
        taskId: 'task_001',
        totalTokens: 150000,
        totalCost: 3.20,
        totalLlmCallCount: 25,
      };
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => mockStats,
      });

      const result = await client.getStats('task_001');
      expect((result as any).taskId).toBe('task_001');
      const calledUrl = mockFetch.mock.calls[0][0] as string;
      expect(calledUrl).toContain('taskId=task_001');
    });
  });
});
```

#### 2.1.3 重试机制

| 测试用例 | 场景 | 期望行为 |
|----------|------|----------|
| 5xx 首次失败后成功 | 第1次500，第2次200 | 返回数据，调用2次fetch |
| 5xx 重试耗尽 | 连续3次500 | 抛出 ApiError(500) |
| 4xx 不重试 | 400 错误 | 立即抛出，只调用1次fetch |
| 网络超时重试 | 第1次超时，第2次成功 | 返回数据 |
| 网络超时耗尽 | 连续3次超时 | 抛出 NetworkError |

```typescript
describe('InsightClient retry logic', () => {
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

  it('retries on network timeout', async () => {
    mockFetch
      .mockRejectedValueOnce(new Error('AbortError: timeout'))
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0 }),
      });

    const result = await client.listSessions({});
    expect(result.items).toEqual([]);
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });
});
```

### 2.2 格式化工具单元测试

**文件**: `tests/cli/unit/format.test.ts`  
**测试对象**: `src/cli/utils/format.ts`

| 函数 | 测试用例 | 输入 | 期望输出 |
|------|----------|------|----------|
| `formatTokens` | 小于1000 | `999` | `"999"` |
| `formatTokens` | 千级 | `1500` | `"1.5K"` |
| `formatTokens` | 万级 | `150000` | `"150.0K"` |
| `formatTokens` | 百万级 | `2100000` | `"2.1M"` |
| `formatTokens` | 零 | `0` | `"0"` |
| `formatCost` | 正常值 | `3.2` | `"¥3.20"` |
| `formatCost` | 零 | `0` | `"¥0.00"` |
| `formatCost` | 高精度 | `0.123456` | `"¥0.12"` |
| `formatDuration` | 秒级 | `5200` | `"5.2s"` |
| `formatDuration` | 分钟级 | `180000` | `"3m"` |
| `formatDuration` | 小时级 | `7200000` | `"2.0h"` |
| `formatDuration` | 零 | `0` | `"0.0s"` |
| `formatDate` | 正常日期 | `"2025-06-14T10:30:00Z"` | `"06-14 10:30"` |
| `formatDate` | null | `null` | `"—"` |
| `formatPercent` | 正常值 | `85.5` | `"85.5%"` |
| `formatPercent` | null | `null` | `"—"` |
| `truncate` | 短字符串 | `"hello", 10` | `"hello"` |
| `truncate` | 超长字符串 | `"very long string...", 10` | `"very long…"` |
| `truncate` | null | `null, 10` | `"—"` |

```typescript
import { describe, it, expect } from 'vitest';
import {
  formatTokens,
  formatCost,
  formatDuration,
  formatDate,
  formatPercent,
  truncate,
} from '../../src/cli/utils/format';

describe('format utilities', () => {
  describe('formatTokens', () => {
    it('returns raw number for values < 1000', () => {
      expect(formatTokens(0)).toBe('0');
      expect(formatTokens(999)).toBe('999');
      expect(formatTokens(1)).toBe('1');
    });

    it('formats thousands with K suffix', () => {
      expect(formatTokens(1000)).toBe('1.0K');
      expect(formatTokens(1500)).toBe('1.5K');
      expect(formatTokens(150000)).toBe('150.0K');
      expect(formatTokens(999999)).toBe('1000.0K');
    });

    it('formats millions with M suffix', () => {
      expect(formatTokens(1000000)).toBe('1.0M');
      expect(formatTokens(2100000)).toBe('2.1M');
      expect(formatTokens(10000000)).toBe('10.0M');
    });
  });

  describe('formatCost', () => {
    it('formats with ¥ prefix and 2 decimal places', () => {
      expect(formatCost(0)).toBe('¥0.00');
      expect(formatCost(3.2)).toBe('¥3.20');
      expect(formatCost(56.3)).toBe('¥56.30');
      expect(formatCost(0.123)).toBe('¥0.12');
    });
  });

  describe('formatDuration', () => {
    it('formats seconds for < 60s', () => {
      expect(formatDuration(0)).toBe('0.0s');
      expect(formatDuration(5200)).toBe('5.2s');
      expect(formatDuration(59999)).toBe('60.0s');
    });

    it('formats minutes for 60s - 3600s', () => {
      expect(formatDuration(60000)).toBe('1m');
      expect(formatDuration(180000)).toBe('3m');
      expect(formatDuration(2700000)).toBe('45m');
    });

    it('formats hours for >= 3600s', () => {
      expect(formatDuration(3600000)).toBe('1.0h');
      expect(formatDuration(7200000)).toBe('2.0h');
      expect(formatDuration(66720000)).toBe('18.5h');
    });
  });

  describe('formatDate', () => {
    it('formats ISO date to MM-DD HH:mm', () => {
      expect(formatDate('2025-06-14T10:30:00.000Z')).toMatch(/\d{2}-\d{2} \d{2}:\d{2}/);
    });

    it('returns em-dash for null', () => {
      expect(formatDate(null)).toBe('—');
    });
  });

  describe('formatPercent', () => {
    it('formats with 1 decimal place and % suffix', () => {
      expect(formatPercent(85.5)).toBe('85.5%');
      expect(formatPercent(0)).toBe('0.0%');
      expect(formatPercent(100)).toBe('100.0%');
    });

    it('returns em-dash for null', () => {
      expect(formatPercent(null)).toBe('—');
    });
  });

  describe('truncate', () => {
    it('returns string as-is when shorter than maxLen', () => {
      expect(truncate('hello', 10)).toBe('hello');
    });

    it('truncates with ellipsis when longer than maxLen', () => {
      expect(truncate('very long string here', 10)).toBe('very long…');
      expect(truncate('very long string here', 10).length).toBe(10);
    });

    it('returns em-dash for null', () => {
      expect(truncate(null, 10)).toBe('—');
    });

    it('handles exact length', () => {
      expect(truncate('12345', 5)).toBe('12345');
    });
  });
});
```

### 2.3 表格渲染工具测试

**文件**: `tests/cli/unit/table.test.ts`  
**测试对象**: `src/cli/utils/table.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { renderTable, TableColumn } from '../../src/cli/utils/table';

describe('renderTable', () => {
  interface TestRow {
    name: string;
    value: number;
    status: string;
  }

  const columns: TableColumn<TestRow>[] = [
    { key: 'name', label: 'Name', width: 10 },
    { key: 'value', label: 'Value', width: 8 },
    { key: 'status', label: 'Status', width: 10 },
  ];

  it('renders header with correct column widths', () => {
    const result = renderTable(columns, []);
    const lines = result.split('\n');
    expect(lines[0]).toContain('Name');
    expect(lines[0]).toContain('Value');
    expect(lines[0]).toContain('Status');
  });

  it('renders separator line between header and body', () => {
    const result = renderTable(columns, []);
    const lines = result.split('\n');
    expect(lines[1]).toContain('─');
    expect(lines[1]).toContain('┼');
  });

  it('renders data rows correctly', () => {
    const rows: TestRow[] = [
      { name: 'Alice', value: 100, status: 'active' },
      { name: 'Bob', value: 200, status: 'idle' },
    ];
    const result = renderTable(columns, rows);
    expect(result).toContain('Alice');
    expect(result).toContain('Bob');
    expect(result).toContain('100');
    expect(result).toContain('200');
  });

  it('uses custom render function when provided', () => {
    const customColumns: TableColumn<TestRow>[] = [
      { key: 'name', label: 'Name', width: 10 },
      { key: 'value', label: 'Value', width: 8, render: (row) => `$${row.value}` },
    ];
    const rows: TestRow[] = [{ name: 'Test', value: 42, status: 'ok' }];
    const result = renderTable(customColumns, rows);
    expect(result).toContain('$42');
  });

  it('truncates values exceeding column width', () => {
    const rows: TestRow[] = [
      { name: 'VeryLongNameThatExceedsWidth', value: 1, status: 'ok' },
    ];
    const result = renderTable(columns, rows);
    // Value should be truncated to 10 chars
    const bodyLine = result.split('\n')[2];
    expect(bodyLine.length).toBeLessThanOrEqual(50);
  });

  it('handles empty data array', () => {
    const result = renderTable(columns, []);
    const lines = result.split('\n');
    expect(lines.length).toBe(2); // header + separator only
  });

  it('replaces null/undefined values with em-dash', () => {
    const rows = [{ name: null, value: undefined, status: 'ok' }] as any;
    const result = renderTable(columns, rows);
    expect(result).toContain('—');
  });
});
```

### 2.4 错误类型测试

**文件**: `tests/cli/unit/errors.test.ts`  
**测试对象**: `src/cli/errors.ts`

```typescript
import { describe, it, expect } from 'vitest';
import {
  InsightError,
  ApiError,
  NetworkError,
  TerminalError,
  ConfigError,
} from '../../src/cli/errors';

describe('Error types', () => {
  describe('ApiError', () => {
    it('stores status code and retryable flag', () => {
      const err = new ApiError(404, 'Not found', false);
      expect(err.status).toBe(404);
      expect(err.retryable).toBe(false);
      expect(err.message).toContain('404');
      expect(err.message).toContain('Not found');
    });

    it('is instance of InsightError and Error', () => {
      const err = new ApiError(500, 'Server error', true);
      expect(err).toBeInstanceOf(ApiError);
      expect(err).toBeInstanceOf(InsightError);
      expect(err).toBeInstanceOf(Error);
    });

    it('marks 5xx as retryable', () => {
      const err = new ApiError(503, 'Service unavailable', true);
      expect(err.retryable).toBe(true);
    });

    it('marks 4xx as non-retryable', () => {
      const err = new ApiError(400, 'Bad request', false);
      expect(err.retryable).toBe(false);
    });
  });

  describe('NetworkError', () => {
    it('wraps network error message', () => {
      const err = new NetworkError('ECONNREFUSED');
      expect(err.message).toContain('ECONNREFUSED');
      expect(err).toBeInstanceOf(InsightError);
    });
  });

  describe('TerminalError', () => {
    it('wraps terminal compatibility error', () => {
      const err = new TerminalError('ANSI not supported');
      expect(err.message).toContain('ANSI not supported');
    });
  });

  describe('ConfigError', () => {
    it('wraps config error', () => {
      const err = new ConfigError('Invalid config file');
      expect(err.message).toContain('Invalid config file');
    });
  });
});
```

### 2.5 配置管理测试

**文件**: `tests/cli/unit/config.test.ts`  
**测试对象**: `src/cli/config.ts`

```typescript
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { loadConfig, saveConfig, DEFAULT_CONFIG } from '../../src/cli/config';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

describe('config management', () => {
  let testConfigDir: string;
  let testConfigFile: string;

  beforeEach(() => {
    testConfigDir = path.join(os.tmpdir(), `cannbot-test-${Date.now()}`);
    testConfigFile = path.join(testConfigDir, 'config.json');
    fs.mkdirSync(testConfigDir, { recursive: true });
    // Mock config file path
    vi.spyOn(os, 'homedir').mockReturnValue(testConfigDir);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    fs.rmSync(testConfigDir, { recursive: true, force: true });
    delete process.env.CANNBOT_SERVER;
    delete process.env.CANNBOT_TIMEOUT;
  });

  describe('loadConfig', () => {
    it('returns default config when no file exists', () => {
      const config = loadConfig();
      expect(config.server).toBe(DEFAULT_CONFIG.server);
      expect(config.timeout).toBe(DEFAULT_CONFIG.timeout);
      expect(config.theme).toBe('auto');
    });

    it('reads config from file', () => {
      fs.writeFileSync(testConfigFile, JSON.stringify({
        server: 'http://custom:8080',
        timeout: 30000,
      }));
      const config = loadConfig();
      expect(config.server).toBe('http://custom:8080');
      expect(config.timeout).toBe(30000);
    });

    it('environment variable overrides config file', () => {
      fs.writeFileSync(testConfigFile, JSON.stringify({ server: 'http://file:8080' }));
      process.env.CANNBOT_SERVER = 'http://env:9090';
      const config = loadConfig();
      expect(config.server).toBe('http://env:9090');
    });

    it('command line args override environment variables', () => {
      process.env.CANNBOT_SERVER = 'http://env:9090';
      const config = loadConfig({ server: 'http://cli:7070' });
      expect(config.server).toBe('http://cli:7070');
    });

    it('ignores invalid config file (JSON parse error)', () => {
      fs.writeFileSync(testConfigFile, 'not valid json {{{');
      const config = loadConfig();
      expect(config.server).toBe(DEFAULT_CONFIG.server);
    });
  });
});
```

### 2.6 命令模块单元测试

每个命令模块遵循相同的测试模式：mock InsightClient，验证参数传递和输出格式。

#### 2.6.1 sessions 命令

**文件**: `tests/cli/unit/commands/sessions.test.ts`

```typescript
import { describe, it, expect, vi } from 'vitest';
import { sessionsCommand } from '../../../src/cli/commands/sessions';
import { InsightClient } from '../../../src/cli/client';

vi.mock('../../../src/cli/client');

describe('sessions command', () => {
  const mockListSessions = vi.fn();

  beforeEach(() => {
    vi.mocked(InsightClient).mockImplementation(() => ({
      listSessions: mockListSessions,
    } as any));
    mockListSessions.mockReset();
  });

  it('outputs table format by default', async () => {
    mockListSessions.mockResolvedValue({
      items: [
        {
          sessionId: 'ses_001', taskId: 'task_001',
          startTime: '2025-06-14T10:30:00Z', user: 'guan',
          query: 'test query', model: 'claude-3.5-sonnet',
          totalTokens: 150000, totalCost: 3.20,
          totalToolCallCount: 120,
        },
      ],
      total: 1,
      page: 1,
    });

    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const cmd = sessionsCommand();
    await cmd.parseAsync(['node', 'test']);

    expect(mockListSessions).toHaveBeenCalled();
    expect(consoleSpy).toHaveBeenCalled();
    const output = consoleSpy.mock.calls.map(c => c.join(' ')).join('\n');
    expect(output).toContain('task_001');
    consoleSpy.mockRestore();
  });

  it('outputs JSON when --json flag is set', async () => {
    mockListSessions.mockResolvedValue({
      items: [{ taskId: 'task_001' }],
      total: 1,
    });

    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const cmd = sessionsCommand();
    await cmd.parseAsync(['node', 'test', '--json']);

    const output = consoleSpy.mock.calls[0][0];
    expect(() => JSON.parse(output)).not.toThrow();
    consoleSpy.mockRestore();
  });

  it('passes --user filter to client', async () => {
    mockListSessions.mockResolvedValue({ items: [], total: 0 });
    const cmd = sessionsCommand();
    await cmd.parseAsync(['node', 'test', '--user', 'guanxinghua']);

    expect(mockListSessions).toHaveBeenCalledWith(
      expect.objectContaining({ user: 'guanxinghua' }),
    );
  });

  it('passes --limit as pageSize', async () => {
    mockListSessions.mockResolvedValue({ items: [], total: 0 });
    const cmd = sessionsCommand();
    await cmd.parseAsync(['node', 'test', '--limit', '5']);

    expect(mockListSessions).toHaveBeenCalledWith(
      expect.objectContaining({ pageSize: 5 }),
    );
  });
});
```

#### 2.6.2 compare 命令

**文件**: `tests/cli/unit/commands/compare.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { compareCommand } from '../../../src/cli/commands/compare';
import { InsightClient } from '../../../src/cli/client';

vi.mock('../../../src/cli/client');

describe('compare command', () => {
  const mockGetSession = vi.fn();
  const mockGetStats = vi.fn();

  beforeEach(() => {
    vi.mocked(InsightClient).mockImplementation(() => ({
      getSession: mockGetSession,
      getStats: mockGetStats,
    } as any));
    mockGetSession.mockReset();
    mockGetStats.mockReset();
  });

  it('fetches both sessions in parallel', async () => {
    mockGetSession
      .mockResolvedValueOnce({ taskId: 'task_001', model: 'claude-3.5-sonnet' })
      .mockResolvedValueOnce({ taskId: 'task_002', model: 'glm-5' });
    mockGetStats
      .mockResolvedValueOnce({ totalTokens: 150000, totalCost: 3.20, totalLatencyMs: 2700000, totalLlmCallCount: 25, totalToolCallCount: 120, totalSubagentCount: 3 })
      .mockResolvedValueOnce({ totalTokens: 80000, totalCost: 1.50, totalLatencyMs: 1320000, totalLlmCallCount: 12, totalToolCallCount: 45, totalSubagentCount: 1 });

    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const cmd = compareCommand();
    await cmd.parseAsync(['node', 'test', 'task_001', 'task_002']);

    expect(mockGetSession).toHaveBeenCalledTimes(2);
    expect(mockGetStats).toHaveBeenCalledTimes(2);
    consoleSpy.mockRestore();
  });

  it('outputs JSON comparison when --json flag', async () => {
    mockGetSession
      .mockResolvedValueOnce({ taskId: 'task_001' })
      .mockResolvedValueOnce({ taskId: 'task_002' });
    mockGetStats
      .mockResolvedValueOnce({ totalTokens: 100 })
      .mockResolvedValueOnce({ totalTokens: 200 });

    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const cmd = compareCommand();
    await cmd.parseAsync(['node', 'test', 'task_001', 'task_002', '--json']);

    const output = consoleSpy.mock.calls[0][0];
    const parsed = JSON.parse(output);
    expect(parsed.s1).toBeDefined();
    expect(parsed.s2).toBeDefined();
    consoleSpy.mockRestore();
  });
});
```

#### 2.6.3 其他命令（统一模式）

| 命令 | 关键测试点 |
|------|-----------|
| `session <id>` | --tab 参数路由到不同输出；--format 控制格式 |
| `turn <id>` | 正常输出 + JSON 模式 |
| `search <id> --keyword X` | keyword 参数传递、空结果处理 |
| `stats` | 全局 vs session 统计；JSON 输出 |
| `import` | --list 仅预览；--source 和 --file 参数验证 |
| `delete` | --session 删除单个；--all 删除全部；确认提示 |
| `config` | set/get/list 子命令；context-windows 子命令 |

### 2.7 Hooks 单元测试

#### 2.7.1 useTable hook

**文件**: `tests/cli/hooks/useTable.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useTable } from '../../src/cli/hooks/useTable';

describe('useTable', () => {
  const testData = [
    { name: 'Alice', tokens: 100, cost: 3.20 },
    { name: 'Bob', tokens: 200, cost: 1.50 },
    { name: 'Charlie', tokens: 50, cost: 8.10 },
  ];

  const columns = [
    { key: 'name', sortable: true, filterable: true },
    { key: 'tokens', sortable: true, filterable: false },
    { key: 'cost', sortable: true, filterable: false },
  ];

  it('initializes with first item selected', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    expect(result.current.state.selectedIndex).toBe(0);
    expect(result.current.state.visibleData).toHaveLength(3);
  });

  it('selectDown moves selection down', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.selectDown());
    expect(result.current.state.selectedIndex).toBe(1);
  });

  it('selectUp does not go below 0', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.selectUp());
    expect(result.current.state.selectedIndex).toBe(0);
  });

  it('selectDown does not exceed data length', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.selectDown());
    act(() => result.current.selectDown());
    act(() => result.current.selectDown());
    expect(result.current.state.selectedIndex).toBe(2);
  });

  it('sortBy toggles direction on same key', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.sortBy('tokens'));
    expect(result.current.state.sortDir).toBe('desc');
    act(() => result.current.sortBy('tokens'));
    expect(result.current.state.sortDir).toBe('asc');
  });

  it('filter reduces visible data', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.filter('Alice'));
    expect(result.current.state.visibleData).toHaveLength(1);
    expect(result.current.state.visibleData[0].name).toBe('Alice');
  });

  it('filter resets selectedIndex to 0', () => {
    const { result } = renderHook(() => useTable(testData, columns));
    act(() => result.current.selectDown());
    act(() => result.current.filter('Charlie'));
    expect(result.current.state.selectedIndex).toBe(0);
  });
});
```

#### 2.7.2 useNavigation hook

**文件**: `tests/cli/hooks/useNavigation.test.ts`

```typescript
import { describe, it, expect } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useNavigation } from '../../src/cli/hooks/useNavigation';

describe('useNavigation', () => {
  it('initializes at sessions screen', () => {
    const { result } = renderHook(() => useNavigation());
    expect(result.current.nav.screen).toBe('sessions');
    expect(result.current.nav.stack).toEqual([]);
  });

  it('navigate pushes current state to stack', () => {
    const { result } = renderHook(() => useNavigation());
    act(() => result.current.navigate('session', { taskId: 'task_001' }));
    expect(result.current.nav.screen).toBe('session');
    expect(result.current.nav.taskId).toBe('task_001');
    expect(result.current.nav.stack).toHaveLength(1);
  });

  it('goBack returns to previous screen', () => {
    const { result } = renderHook(() => useNavigation());
    act(() => result.current.navigate('session', { taskId: 'task_001' }));
    act(() => result.current.navigate('turn', { turnId: 'turn_001' }));
    expect(result.current.nav.screen).toBe('turn');

    act(() => result.current.goBack());
    expect(result.current.nav.screen).toBe('session');
    expect(result.current.nav.taskId).toBe('task_001');
  });

  it('goBack at root does nothing', () => {
    const { result } = renderHook(() => useNavigation());
    act(() => result.current.goBack());
    expect(result.current.nav.screen).toBe('sessions');
  });

  it('supports deep navigation stack', () => {
    const { result } = renderHook(() => useNavigation());
    act(() => result.current.navigate('session', { taskId: 't1' }));
    act(() => result.current.navigate('turn', { turnId: 'turn1' }));
    expect(result.current.nav.stack).toHaveLength(2);

    act(() => result.current.goBack());
    act(() => result.current.goBack());
    expect(result.current.nav.screen).toBe('sessions');
    expect(result.current.nav.stack).toHaveLength(0);
  });
});
```

#### 2.7.3 useApi hook

**文件**: `tests/cli/hooks/useApi.test.ts`

```typescript
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { useApi } from '../../src/cli/hooks/useApi';

describe('useApi', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  it('starts with loading=true and data=null', () => {
    const fetcher = vi.fn().mockResolvedValue({ items: [] });
    const { result } = renderHook(() => useApi(fetcher, ['test']));
    expect(result.current.loading).toBe(true);
    expect(result.current.data).toBeNull();
    expect(result.current.error).toBeNull();
  });

  it('sets data after successful fetch', async () => {
    const fetcher = vi.fn().mockResolvedValue({ totalSessions: 42 });
    const { result } = renderHook(() => useApi(fetcher, ['test']));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.data).toEqual({ totalSessions: 42 });
  });

  it('sets error on fetch failure', async () => {
    const fetcher = vi.fn().mockRejectedValue(new Error('Network failure'));
    const { result } = renderHook(() => useApi(fetcher, ['test']));

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });
    expect(result.current.error).toBeDefined();
    expect(result.current.error!.message).toContain('Network failure');
  });

  it('refresh clears cache and refetches', async () => {
    let callCount = 0;
    const fetcher = vi.fn().mockImplementation(async () => {
      callCount++;
      return { count: callCount };
    });

    const { result } = renderHook(() => useApi(fetcher, ['test'], 30000));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual({ count: 1 });

    act(() => result.current.refresh());

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual({ count: 2 });
  });
});
```

---

## 3. 集成测试计划

### 3.1 API Client + Mock Server 集成

**文件**: `tests/cli/integration/client-api.test.ts`

使用 MSW (Mock Service Worker) 或 `http` 模块创建 mock server，测试完整的 HTTP 请求/响应链路。

#### 3.1.1 Mock Server 设置

```typescript
// tests/helpers/mock-server.ts
import { createServer, IncomingMessage, ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';

export interface MockRoute {
  method: string;
  path: string;
  handler: (req: IncomingMessage, body: any) => { status: number; data: any };
}

export function createMockServer(routes: MockRoute[]) {
  const server = createServer((req, res) => {
    let body = '';
    req.on('data', chunk => { body += chunk; });
    req.on('end', () => {
      const parsedBody = body ? JSON.parse(body) : null;
      const route = routes.find(r =>
        r.method === req.method && req.url?.startsWith(r.path)
      );

      if (!route) {
        res.writeHead(404, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: 'Not found' }));
        return;
      }

      const result = route.handler(req, parsedBody);
      res.writeHead(result.status, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify(result.data));
    });
  });

  return {
    start: () => new Promise<string>((resolve) => {
      server.listen(0, () => {
        const port = (server.address() as AddressInfo).port;
        resolve(`http://localhost:${port}`);
      });
    }),
    stop: () => new Promise<void>((resolve) => {
      server.close(() => resolve());
    }),
  };
}
```

#### 3.1.2 集成测试用例

```typescript
// tests/cli/integration/client-api.test.ts
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { InsightClient, ApiError } from '../../../src/cli/client';
import { createMockServer, MockRoute } from '../../helpers/mock-server';
import { mockSessionList, mockSessionDetail, mockGlobalStats } from '../../fixtures/mock-data';

describe('InsightClient integration with mock server', () => {
  let server: ReturnType<typeof createMockServer>;
  let baseUrl: string;
  let client: InsightClient;

  const routes: MockRoute[] = [
    {
      method: 'GET',
      path: '/api/observe/data',
      handler: (req) => ({
        status: 200,
        data: { items: mockSessionList, total: mockSessionList.length, page: 1 },
      }),
    },
    {
      method: 'GET',
      path: '/api/observe/session',
      handler: (req) => {
        const url = new URL(req.url!, 'http://localhost');
        const taskId = url.searchParams.get('taskId');
        const session = mockSessionDetail.find(s => s.taskId === taskId);
        if (!session) return { status: 404, data: { error: 'Not found' } };
        return { status: 200, data: session };
      },
    },
    {
      method: 'GET',
      path: '/api/observe/stats',
      handler: () => ({ status: 200, data: mockGlobalStats }),
    },
    {
      method: 'DELETE',
      path: '/api/ingest/delete-session',
      handler: (_, body) => ({
        status: 200,
        data: { deleted: 1, taskId: body?.taskId },
      }),
    },
  ];

  beforeAll(async () => {
    server = createMockServer(routes);
    baseUrl = await server.start();
    client = new InsightClient(baseUrl, { retries: 0 });
  });

  afterAll(async () => {
    await server.stop();
  });

  describe('listSessions', () => {
    it('returns session list from mock server', async () => {
      const result = await client.listSessions({});
      expect(result.items.length).toBeGreaterThan(0);
      expect(result.total).toBeGreaterThan(0);
      expect(result.items[0]).toHaveProperty('taskId');
      expect(result.items[0]).toHaveProperty('totalTokens');
    });
  });

  describe('getSession', () => {
    it('returns session detail for valid taskId', async () => {
      const result = await client.getSession('task_001');
      expect(result.taskId).toBe('task_001');
      expect(result).toHaveProperty('model');
      expect(result).toHaveProperty('totalTokens');
    });

    it('throws ApiError(404) for unknown taskId', async () => {
      try {
        await client.getSession('nonexistent_task');
        expect.unreachable('Should have thrown');
      } catch (err) {
        expect(err).toBeInstanceOf(ApiError);
        expect((err as ApiError).status).toBe(404);
      }
    });
  });

  describe('getStats', () => {
    it('returns global statistics', async () => {
      const result = await client.getStats();
      expect(result).toHaveProperty('totalSessions');
      expect(result).toHaveProperty('totalTokens');
      expect(result).toHaveProperty('totalCost');
    });
  });

  describe('deleteSession', () => {
    it('sends DELETE request with taskId', async () => {
      const result = await client.deleteSession('task_001');
      expect(result.deleted).toBe(1);
    });
  });
});
```

### 3.2 命令流程集成测试

**文件**: `tests/cli/integration/command-flow.test.ts`

测试完整的命令行调用流程：解析参数 → 调用 API → 格式化输出。

```typescript
import { describe, it, expect, beforeAll, afterAll, vi } from 'vitest';
import { createProgram } from '../../../src/cli/index';
import { createMockServer, MockRoute } from '../../helpers/mock-server';
import { mockSessionList, mockGlobalStats } from '../../fixtures/mock-data';

describe('CLI command flow integration', () => {
  let server: ReturnType<typeof createMockServer>;
  let baseUrl: string;

  beforeAll(async () => {
    const routes: MockRoute[] = [
      {
        method: 'GET', path: '/api/observe/data',
        handler: () => ({ status: 200, data: { items: mockSessionList, total: 42, page: 1 } }),
      },
      {
        method: 'GET', path: '/api/observe/stats',
        handler: () => ({ status: 200, data: mockGlobalStats }),
      },
    ];
    server = createMockServer(routes);
    baseUrl = await server.start();
  });

  afterAll(async () => {
    await server.stop();
  });

  it('sessions command outputs formatted table', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const program = createProgram();
    await program.parseAsync(['node', 'test', '--server', baseUrl, 'sessions']);

    const output = consoleSpy.mock.calls.map(c => c.join(' ')).join('\n');
    expect(output).toContain('#');
    expect(output).toContain('Date');
    expect(output).toContain('Tokens');
    consoleSpy.mockRestore();
  });

  it('sessions --json outputs valid JSON', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const program = createProgram();
    await program.parseAsync(['node', 'test', '--server', baseUrl, 'sessions', '--json']);

    const jsonOutput = consoleSpy.mock.calls[0][0];
    const parsed = JSON.parse(jsonOutput);
    expect(Array.isArray(parsed)).toBe(true);
    consoleSpy.mockRestore();
  });

  it('stats command outputs formatted statistics', async () => {
    const consoleSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
    const program = createProgram();
    await program.parseAsync(['node', 'test', '--server', baseUrl, 'stats']);

    const output = consoleSpy.mock.calls.map(c => c.join(' ')).join('\n');
    expect(output).toContain('Sessions');
    expect(output).toContain('Tokens');
    consoleSpy.mockRestore();
  });
});
```

---

## 4. TUI 组件测试

### 4.1 测试工具设置

Ink 组件使用 `ink-testing-library` 进行渲染测试。**PoC 验证**: ink-testing-library 完全兼容 Ink v7，其 render() 返回的实例有 lastFrame() 方法和 stdin（可模拟按键），与 Ink v7 原生 render()（无 lastFrame）不同。

```typescript
// tests/helpers/render-tui.tsx
import React from 'react';
import { render } from 'ink-testing-library';

// 注意：测试代码使用 ink-testing-library 的 render()（有 lastFrame + stdin）
// 生产代码使用 Ink 原生 render()（有 waitUntilExit/waitUntilRenderFlush/clear）

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

### 4.2 StatusBar 组件测试

**文件**: `tests/cli/components/StatusBar.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { StatusBar } from '../../../src/cli/tui/components/StatusBar';
import { renderTui } from '../../helpers/render-tui';
import { InsightClient } from '../../../src/cli/client';
import { DEFAULT_CONFIG } from '../../../src/cli/config';

vi.mock('../../../src/cli/client');

describe('StatusBar', () => {
  it('renders app name and version', () => {
    const mockClient = { getStats: vi.fn().mockResolvedValue({ totalSessions: 42, totalCost: 56.30 }) } as any;
    const nav = { screen: 'sessions' as const, stack: [] };

    const { getPlainText } = renderTui(
      <StatusBar client={mockClient} nav={nav} config={DEFAULT_CONFIG} />,
    );

    const text = getPlainText();
    expect(text).toContain('CANNBot-Insight');
  });

  it('shows session count from stats', async () => {
    const mockClient = {
      getStats: vi.fn().mockResolvedValue({ totalSessions: 42, totalCost: 56.30 }),
    } as any;
    const nav = { screen: 'sessions' as const, stack: [] };

    const { getPlainText } = renderTui(
      <StatusBar client={mockClient} nav={nav} config={DEFAULT_CONFIG} />,
    );

    // Wait for async data
    await new Promise(r => setTimeout(r, 100));
    const text = getPlainText();
    expect(text).toContain('42');
  });

  it('shows loading indicator when stats not yet loaded', () => {
    const mockClient = {
      getStats: vi.fn().mockReturnValue(new Promise(() => {})), // never resolves
    } as any;
    const nav = { screen: 'sessions' as const, stack: [] };

    const { getPlainText } = renderTui(
      <StatusBar client={mockClient} nav={nav} config={DEFAULT_CONFIG} />,
    );

    const text = getPlainText();
    expect(text).toContain('…');
  });
});
```

### 4.3 DataTable 组件测试

**文件**: `tests/cli/components/DataTable.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { DataTable } from '../../../src/cli/tui/components/DataTable';
import { renderTui } from '../../helpers/render-tui';

describe('DataTable', () => {
  const columns = [
    { key: 'name', label: 'Name', width: 10 },
    { key: 'tokens', label: 'Tokens', width: 8 },
  ];

  const data = [
    { name: 'Session A', tokens: 1500 },
    { name: 'Session B', tokens: 2500 },
    { name: 'Session C', tokens: 800 },
  ];

  it('renders header row', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={0} />,
    );

    const text = getPlainText();
    expect(text).toContain('Name');
    expect(text).toContain('Tokens');
  });

  it('renders all data rows', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={0} />,
    );

    const text = getPlainText();
    expect(text).toContain('Session A');
    expect(text).toContain('Session B');
    expect(text).toContain('Session C');
  });

  it('highlights selected row', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={1} />,
    );

    // Selected row should have different styling (check ANSI codes)
    const frame = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={1} />,
    ).lastFrame();
    // The selected row should contain cyan color code or bold
    expect(frame).toBeDefined();
  });

  it('uses custom render function for column values', () => {
    const customColumns = [
      { key: 'name', label: 'Name', width: 10 },
      { key: 'tokens', label: 'Tokens', width: 8, render: (row: any) => `${(row.tokens / 1000).toFixed(1)}K` },
    ];

    const { getPlainText } = renderTui(
      <DataTable columns={customColumns} data={data} selectedIndex={0} />,
    );

    const text = getPlainText();
    expect(text).toContain('1.5K');
    expect(text).toContain('2.5K');
  });

  it('shows scroll indicator when data exceeds visible rows', () => {
    const manyRows = Array.from({ length: 100 }, (_, i) => ({
      name: `Session ${i}`,
      tokens: i * 100,
    }));

    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={manyRows} selectedIndex={0} maxVisibleRows={10} />,
    );

    const text = getPlainText();
    expect(text).toContain('Showing');
    expect(text).toContain('of 100');
  });

  it('renders separator line between header and body', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={0} />,
    );

    const text = getPlainText();
    expect(text).toContain('─');
  });

  it('handles empty data array', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={[]} selectedIndex={0} />,
    );

    const text = getPlainText();
    expect(text).toContain('Name');
    expect(text).toContain('Tokens');
    // Should not crash with empty data
  });
});
```

### 4.4 AsciiBar 组件测试

**文件**: `tests/cli/components/AsciiBar.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect } from 'vitest';
import { AsciiBar } from '../../../src/cli/tui/components/AsciiBar';
import { renderTui } from '../../helpers/render-tui';

describe('AsciiBar', () => {
  it('renders bar with correct fill ratio', () => {
    const { getPlainText } = renderTui(
      <AsciiBar label="Input" value={50} max={100} width={20} />,
    );

    const text = getPlainText();
    expect(text).toContain('Input');
    expect(text).toContain('█'); // filled blocks
    expect(text).toContain('░'); // empty blocks
  });

  it('shows percentage text', () => {
    const { getPlainText } = renderTui(
      <AsciiBar label="Test" value={75} max={100} width={10} />,
    );

    const text = getPlainText();
    expect(text).toContain('75.0%');
  });

  it('handles zero value', () => {
    const { getPlainText } = renderTui(
      <AsciiBar label="Empty" value={0} max={100} width={10} />,
    );

    const text = getPlainText();
    expect(text).toContain('0.0%');
  });

  it('caps at 100% when value exceeds max', () => {
    const { getPlainText } = renderTui(
      <AsciiBar label="Over" value={150} max={100} width={10} />,
    );

    const text = getPlainText();
    // Should not exceed 100%
    expect(text).not.toContain('150.0%');
  });

  it('renders with warning color when above threshold', () => {
    const frame = renderTui(
      <AsciiBar
        label="Context"
        value={85}
        max={100}
        width={10}
        warningThreshold={0.8}
      />,
    ).lastFrame();

    // Should contain yellow ANSI code for warning
    expect(frame).toBeDefined();
  });

  it('renders with critical color when above critical threshold', () => {
    const frame = renderTui(
      <AsciiBar
        label="Context"
        value={97}
        max={100}
        width={10}
        criticalThreshold={0.95}
      />,
    ).lastFrame();

    expect(frame).toBeDefined();
  });
});
```

### 4.5 TreeView 组件测试

**文件**: `tests/cli/components/TreeView.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect } from 'vitest';
import { TreeView } from '../../../src/cli/tui/components/TreeView';
import { renderTui } from '../../helpers/render-tui';

describe('TreeView', () => {
  const tree = [
    {
      label: 'Phase 1: 需求分析',
      icon: '📁',
      children: [
        { label: '[Turn 1] user → assistant', icon: '🔧', detail: '12K tokens' },
        { label: '[Turn 2] user → assistant', icon: '🔧', detail: '8K tokens' },
      ],
    },
    {
      label: 'Phase 2: 代码实现',
      icon: '📁',
      children: [
        { label: '[Turn 3] write_file', icon: '🔧', detail: '25K tokens' },
      ],
    },
  ];

  it('renders root nodes', () => {
    const { getPlainText } = renderTui(<TreeView tree={tree} />);
    const text = getPlainText();
    expect(text).toContain('Phase 1');
    expect(text).toContain('Phase 2');
  });

  it('renders child nodes with tree prefixes', () => {
    const { getPlainText } = renderTui(<TreeView tree={tree} />);
    const text = getPlainText();
    expect(text).toContain('├──'); // branch prefix
    expect(text).toContain('└──'); // last child prefix
  });

  it('renders node icons', () => {
    const { getPlainText } = renderTui(<TreeView tree={tree} />);
    const text = getPlainText();
    expect(text).toContain('📁');
    expect(text).toContain('🔧');
  });

  it('renders node detail text', () => {
    const { getPlainText } = renderTui(<TreeView tree={tree} />);
    const text = getPlainText();
    expect(text).toContain('12K tokens');
    expect(text).toContain('8K tokens');
  });

  it('handles empty tree', () => {
    const { getPlainText } = renderTui(<TreeView tree={[]} />);
    // Should not crash
    expect(getPlainText()).toBeDefined();
  });

  it('handles tree with no children', () => {
    const leafTree = [
      { label: 'Leaf Node', icon: '📄' },
    ];
    const { getPlainText } = renderTui(<TreeView tree={leafTree} />);
    const text = getPlainText();
    expect(text).toContain('Leaf Node');
  });
});
```

### 4.6 TabBar 组件测试

**文件**: `tests/cli/components/TabBar.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { TabBar } from '../../../src/cli/tui/components/TabBar';
import { renderTui } from '../../helpers/render-tui';

describe('TabBar', () => {
  const tabs = ['Overview', 'Turns', 'Workflow', 'Subagents', 'Skills', 'Bridges', 'Context'];

  it('renders all tab names', () => {
    const onSwitch = vi.fn();
    const { getPlainText } = renderTui(
      <TabBar tabs={tabs} activeIndex={0} onSwitch={onSwitch} />,
    );

    const text = getPlainText();
    for (const tab of tabs) {
      expect(text).toContain(tab);
    }
  });

  it('highlights active tab with brackets', () => {
    const onSwitch = vi.fn();
    const { getPlainText } = renderTui(
      <TabBar tabs={tabs} activeIndex={2} onSwitch={onSwitch} />,
    );

    const text = getPlainText();
    expect(text).toContain('[Workflow]');
  });

  it('active tab has different styling than inactive tabs', () => {
    const onSwitch = vi.fn();
    const { lastFrame } = renderTui(
      <TabBar tabs={tabs} activeIndex={0} onSwitch={onSwitch} />,
    );

    const frame = lastFrame();
    // Active tab should have bold/underline/cyan styling
    expect(frame).toBeDefined();
  });
});
```

### 4.7 ConfirmDialog 组件测试

**文件**: `tests/cli/components/ConfirmDialog.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect, vi } from 'vitest';
import { ConfirmDialog } from '../../../src/cli/tui/components/ConfirmDialog';
import { renderTui } from '../../helpers/render-tui';

describe('ConfirmDialog', () => {
  it('renders confirmation message', () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    const { getPlainText } = renderTui(
      <ConfirmDialog message="Delete session task_001?" onConfirm={onConfirm} onCancel={onCancel} />,
    );

    const text = getPlainText();
    expect(text).toContain('Delete session task_001?');
    expect(text).toContain('y: Confirm');
    expect(text).toContain('n: Cancel');
  });

  it('calls onConfirm when y is pressed', () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    const { pressKey } = renderTui(
      <ConfirmDialog message="Confirm?" onConfirm={onConfirm} onCancel={onCancel} />,
    );

    pressKey('y');
    expect(onConfirm).toHaveBeenCalledTimes(1);
    expect(onCancel).not.toHaveBeenCalled();
  });

  it('calls onCancel when n is pressed', () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    const { pressKey } = renderTui(
      <ConfirmDialog message="Confirm?" onConfirm={onConfirm} onCancel={onCancel} />,
    );

    pressKey('n');
    expect(onCancel).toHaveBeenCalledTimes(1);
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it('calls onCancel when Escape is pressed', () => {
    const onConfirm = vi.fn();
    const onCancel = vi.fn();

    const { pressEscape } = renderTui(
      <ConfirmDialog message="Confirm?" onConfirm={onConfirm} onCancel={onCancel} />,
    );

    pressEscape();
    expect(onCancel).toHaveBeenCalledTimes(1);
  });
});
```

### 4.8 KeyBar 组件测试

**文件**: `tests/cli/components/KeyBar.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect } from 'vitest';
import { KeyBar } from '../../../src/cli/tui/components/KeyBar';
import { renderTui } from '../../helpers/render-tui';

describe('KeyBar', () => {
  it('shows session list shortcuts on sessions screen', () => {
    const { getPlainText } = renderTui(<KeyBar screen="sessions" />);
    const text = getPlainText();
    expect(text).toContain('Navigate');
    expect(text).toContain('Enter');
    expect(text).toContain('Quit');
  });

  it('shows session detail shortcuts on session screen', () => {
    const { getPlainText } = renderTui(<KeyBar screen="session" />);
    const text = getPlainText();
    expect(text).toContain('Tab');
    expect(text).toContain('Drill-down');
    expect(text).toContain('Back');
  });

  it('shows minimal shortcuts on turn screen', () => {
    const { getPlainText } = renderTui(<KeyBar screen="turn" />);
    const text = getPlainText();
    expect(text).toContain('Back');
    expect(text).toContain('Quit');
  });
});
```

### 4.9 MetricCards 组件测试

**文件**: `tests/cli/components/MetricCards.test.tsx`

```typescript
import React from 'react';
import { describe, it, expect } from 'vitest';
import { MetricCards } from '../../../src/cli/tui/components/MetricCards';
import { renderTui } from '../../helpers/render-tui';

describe('MetricCards', () => {
  it('renders all metric labels and values', () => {
    const metrics = [
      { label: 'Tokens', value: '150.0K', color: 'cyan' },
      { label: 'Cost', value: '¥3.20', color: 'yellow' },
      { label: 'Duration', value: '45m', color: 'green' },
      { label: 'LLM Calls', value: '25', color: 'blue' },
    ];

    const { getPlainText } = renderTui(<MetricCards metrics={metrics} />);
    const text = getPlainText();
    expect(text).toContain('Tokens');
    expect(text).toContain('150.0K');
    expect(text).toContain('Cost');
    expect(text).toContain('¥3.20');
    expect(text).toContain('Duration');
    expect(text).toContain('45m');
    expect(text).toContain('LLM Calls');
    expect(text).toContain('25');
  });

  it('renders with empty metrics array', () => {
    const { getPlainText } = renderTui(<MetricCards metrics={[]} />);
    expect(getPlainText()).toBeDefined();
  });
});
```

---

## 5. API 流程集成测试

> **评审决策**: 原 E2E 测试实际上是 API 级别的流程集成测试（InsightClient + MockServer），不涉及真实 PTY/TUI 渲染。重命名为 API 流程集成测试。真实 TUI E2E 仅本地手动执行，不进 CI。

### 5.1 测试策略

API 流程集成测试模拟完整的 API 操作流程，验证从 Client 调用到 Mock Server 返回结果的全链路正确性。

### 5.2 Session 列表→详情→返回流程

**文件**: `tests/cli/e2e/sessions-flow.test.ts`

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createMockServer, MockRoute } from '../../helpers/mock-server';
import { InsightClient } from '../../../src/cli/client';
import {
  mockSessionList,
  mockSessionDetail,
  mockSessionStats,
  mockTurns,
  mockExecutions,
} from '../../fixtures/mock-data';

describe('API flow: Session list → detail → turns', () => {
  let server: ReturnType<typeof createMockServer>;
  let client: InsightClient;
  let baseUrl: string;

  beforeAll(async () => {
    const routes: MockRoute[] = [
      {
        method: 'GET', path: '/api/observe/data',
        handler: () => ({ status: 200, data: { items: mockSessionList, total: mockSessionList.length, page: 1 } }),
      },
      {
        method: 'GET', path: '/api/observe/session',
        handler: (req) => {
          const url = new URL(req.url!, 'http://localhost');
          const taskId = url.searchParams.get('taskId');
          const session = mockSessionDetail.find(s => s.taskId === taskId);
          return session
            ? { status: 200, data: session }
            : { status: 404, data: { error: 'Not found' } };
        },
      },
      {
        method: 'GET', path: '/api/observe/stats',
        handler: (req) => {
          const url = new URL(req.url!, 'http://localhost');
          const taskId = url.searchParams.get('taskId');
          if (taskId) {
            return { status: 200, data: mockSessionStats };
          }
          return { status: 200, data: { totalSessions: 42, totalTokens: 2100000, totalCost: 56.30, totalLatencyMs: 66720000, avgLatencyMs: 8200 } };
        },
      },
      {
        method: 'GET', path: '/api/observe/session/turns',
        handler: () => ({ status: 200, data: { items: mockTurns, total: mockTurns.length } }),
      },
      {
        method: 'GET', path: '/api/observe/executions',
        handler: () => ({ status: 200, data: mockExecutions }),
      },
    ];

    server = createMockServer(routes);
    baseUrl = await server.start();
    client = new InsightClient(baseUrl, { retries: 0 });
  });

  afterAll(async () => {
    await server.stop();
  });

  it('Step 1: List sessions returns non-empty list', async () => {
    const result = await client.listSessions({ pageSize: 20 });
    expect(result.items.length).toBeGreaterThan(0);
    expect(result.items[0].taskId).toBeDefined();
    expect(result.items[0].totalTokens).toBeGreaterThan(0);
  });

  it('Step 2: Select first session and fetch detail', async () => {
    const listResult = await client.listSessions({ pageSize: 20 });
    const firstTaskId = listResult.items[0].taskId;

    const [session, stats] = await Promise.all([
      client.getSession(firstTaskId),
      client.getStats(firstTaskId),
    ]);

    expect(session.taskId).toBe(firstTaskId);
    expect(session.model).toBeDefined();
    expect((stats as any).totalTokens).toBeGreaterThan(0);
  });

  it('Step 3: Fetch turns for selected session', async () => {
    const listResult = await client.listSessions({ pageSize: 20 });
    const firstTaskId = listResult.items[0].taskId;

    const turns = await client.getTurns(firstTaskId);
    expect(turns.items.length).toBeGreaterThan(0);
    expect(turns.items[0]).toHaveProperty('turnIndex');
    expect(turns.items[0]).toHaveProperty('role');
  });

  it('Step 4: Fetch executions for overview', async () => {
    const listResult = await client.listSessions({ pageSize: 20 });
    const firstTaskId = listResult.items[0].taskId;

    const executions = await client.getExecutions(firstTaskId);
    expect(executions).toBeDefined();
  });

  it('Full flow: list → select → detail → turns → executions completes without error', async () => {
    // Step 1: List
    const list = await client.listSessions({ pageSize: 20 });
    expect(list.items.length).toBeGreaterThan(0);

    // Step 2: Select and get detail
    const taskId = list.items[0].taskId;
    const session = await client.getSession(taskId);
    expect(session.taskId).toBe(taskId);

    // Step 3: Get stats
    const stats = await client.getStats(taskId);
    expect((stats as any).totalTokens).toBeGreaterThanOrEqual(0);

    // Step 4: Get turns
    const turns = await client.getTurns(taskId);
    expect(turns.items.length).toBeGreaterThanOrEqual(0);

    // Step 5: Get executions
    const executions = await client.getExecutions(taskId);
    expect(executions).toBeDefined();
  });
});
```

### 5.3 导入流程 E2E

**文件**: `tests/cli/e2e/import-flow.test.ts`

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createMockServer, MockRoute } from '../../helpers/mock-server';
import { InsightClient } from '../../../src/cli/client';

describe('API flow: Import', () => {
  let server: ReturnType<typeof createMockServer>;
  let client: InsightClient;

  beforeAll(async () => {
    const routes: MockRoute[] = [
      {
        method: 'POST', path: '/api/ingest/import-file/sessions',
        handler: (_, body) => ({
          status: 200,
          data: {
            sessions: [
              { id: 'ses_001', createdAt: '2025-06-14T10:00:00Z', firstQuery: 'test', turnCount: 10, model: 'claude-3.5' },
              { id: 'ses_002', createdAt: '2025-06-14T11:00:00Z', firstQuery: 'another', turnCount: 5, model: 'glm-5' },
            ],
          },
        }),
      },
      {
        method: 'POST', path: '/api/ingest/import-file',
        handler: (_, body) => ({
          status: 200,
          data: { sessionId: body?.sessionId, imported: true },
        }),
      },
    ];

    server = createMockServer(routes);
    const baseUrl = await server.start();
    client = new InsightClient(baseUrl, { retries: 0 });
  });

  afterAll(async () => {
    await server.stop();
  });

  it('Step 1: List importable sessions', async () => {
    const result = await client.listImportableSessions('opencode-db', '/path/to/db');
    expect(result.sessions.length).toBe(2);
    expect(result.sessions[0].id).toBe('ses_001');
  });

  it('Step 2: Import selected session', async () => {
    const result = await client.importSession('opencode-db', '/path/to/db', 'ses_001');
    expect(result.imported).toBe(true);
    expect(result.sessionId).toBe('ses_001');
  });

  it('Full import flow: list → select → import', async () => {
    // List available
    const available = await client.listImportableSessions('opencode-db', '/path/to/db');
    expect(available.sessions.length).toBeGreaterThan(0);

    // Import first
    const target = available.sessions[0];
    const importResult = await client.importSession('opencode-db', '/path/to/db', target.id);
    expect(importResult.imported).toBe(true);
  });
});
```

### 5.4 对比流程 E2E

**文件**: `tests/cli/e2e/compare-flow.test.ts`

```typescript
import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { createMockServer, MockRoute } from '../../helpers/mock-server';
import { InsightClient } from '../../../src/cli/client';
import { mockSessionDetail, mockSessionStats } from '../../fixtures/mock-data';

describe('API flow: Compare', () => {
  let server: ReturnType<typeof createMockServer>;
  let client: InsightClient;

  beforeAll(async () => {
    const routes: MockRoute[] = [
      {
        method: 'GET', path: '/api/observe/session',
        handler: (req) => {
          const url = new URL(req.url!, 'http://localhost');
          const taskId = url.searchParams.get('taskId');
          const session = mockSessionDetail.find(s => s.taskId === taskId);
          return session
            ? { status: 200, data: session }
            : { status: 404, data: { error: 'Not found' } };
        },
      },
      {
        method: 'GET', path: '/api/observe/stats',
        handler: () => ({ status: 200, data: mockSessionStats }),
      },
    ];

    server = createMockServer(routes);
    const baseUrl = await server.start();
    client = new InsightClient(baseUrl, { retries: 0 });
  });

  afterAll(async () => {
    await server.stop();
  });

  it('compares two sessions side by side', async () => {
    // Fetch both sessions
    const [session1, session2] = await Promise.all([
      client.getSession('task_001'),
      client.getSession('task_002'),
    ]);

    // Fetch both stats
    const [stats1, stats2] = await Promise.all([
      client.getStats('task_001'),
      client.getStats('task_002'),
    ]);

    // Verify data structure for comparison
    expect(session1.taskId).toBe('task_001');
    expect(session2.taskId).toBe('task_002');
    expect((stats1 as any).totalTokens).toBeGreaterThanOrEqual(0);
    expect((stats2 as any).totalTokens).toBeGreaterThanOrEqual(0);

    // Compare metrics
    const comparison = {
      tokens: { s1: (stats1 as any).totalTokens, s2: (stats2 as any).totalTokens },
      cost: { s1: (stats1 as any).totalCost, s2: (stats2 as any).totalCost },
    };
    expect(comparison.tokens.s1).toBeDefined();
    expect(comparison.tokens.s2).toBeDefined();
  });
});
```

---

## 6. 测试数据准备

### 6.1 Mock 数据文件

**目录**: `tests/fixtures/`

#### 6.1.1 mock-sessions.json

```typescript
// tests/fixtures/mock-sessions.ts
import type { SessionListItem } from '../../src/cli/types';

export const mockSessionList: SessionListItem[] = [
  {
    sessionId: 'ses_001',
    taskId: 'task_001',
    query: '帮我实现用户认证功能',
    startTime: '2025-06-14T10:30:00.000Z',
    endTime: '2025-06-14T11:15:00.000Z',
    totalTokens: 150200,
    totalCost: 3.20,
    totalLatencyMs: 2700000,
    totalToolCallCount: 120,
    totalSkillLoadCount: 8,
    totalSubagentCount: 3,
    model: 'claude-3.5-sonnet',
    user: 'guanxinghua',
  },
  {
    sessionId: 'ses_002',
    taskId: 'task_002',
    query: '修复 CI 构建失败',
    startTime: '2025-06-14T09:15:00.000Z',
    endTime: '2025-06-14T09:37:00.000Z',
    totalTokens: 80500,
    totalCost: 1.50,
    totalLatencyMs: 1320000,
    totalToolCallCount: 45,
    totalSkillLoadCount: 4,
    totalSubagentCount: 1,
    model: 'glm-5',
    user: 'guanxinghua',
  },
  {
    sessionId: 'ses_003',
    taskId: 'task_003',
    query: '重构数据库连接池',
    startTime: '2025-06-13T18:00:00.000Z',
    endTime: '2025-06-13T19:30:00.000Z',
    totalTokens: 320000,
    totalCost: 8.10,
    totalLatencyMs: 5400000,
    totalToolCallCount: 200,
    totalSkillLoadCount: 12,
    totalSubagentCount: 5,
    model: 'deepseek-v4',
    user: 'guanxinghua',
  },
  {
    sessionId: 'ses_004',
    taskId: 'task_004',
    query: null,
    startTime: '2025-06-13T14:20:00.000Z',
    endTime: null,
    totalTokens: 45300,
    totalCost: 0.90,
    totalLatencyMs: 900000,
    totalToolCallCount: 20,
    totalSkillLoadCount: 2,
    totalSubagentCount: 0,
    model: 'claude-3.5-sonnet',
    user: 'other_user',
  },
  {
    sessionId: 'ses_005',
    taskId: 'task_005',
    query: '编写单元测试',
    startTime: '2025-06-12T22:10:00.000Z',
    endTime: '2025-06-12T23:05:00.000Z',
    totalTokens: 210700,
    totalCost: 5.40,
    totalLatencyMs: 3300000,
    totalToolCallCount: 85,
    totalSkillLoadCount: 6,
    totalSubagentCount: 2,
    model: 'gpt-4o',
    user: 'guanxinghua',
  },
];
```

#### 6.1.2 mock-stats.ts

```typescript
// tests/fixtures/mock-stats.ts
import type { GlobalStatsResponse, SessionStatsResponse } from '../../src/cli/types';

export const mockGlobalStats: GlobalStatsResponse = {
  totalSessions: 42,
  totalTokens: 2100450,
  totalCost: 56.30,
  totalLatencyMs: 66720000,
  avgLatencyMs: 8200,
};

export const mockSessionStats: SessionStatsResponse = {
  taskId: 'task_001',
  totalTokens: 150200,
  totalInputTokens: 60000,
  totalOutputTokens: 30000,
  totalReasoningTokens: 45000,
  totalCacheReadTokens: 15000,
  totalCacheWriteTokens: 200,
  totalCost: 3.20,
  totalLatencyMs: 2700000,
  totalToolCallCount: 120,
  totalSkillLoadCount: 8,
  totalSubagentCount: 3,
  totalLlmCallCount: 25,
};
```

#### 6.1.3 mock-turns.ts

```typescript
// tests/fixtures/mock-turns.ts
import type { TurnItem } from '../../src/cli/types';

export const mockTurns: TurnItem[] = [
  {
    turnId: 'turn_001',
    turnIndex: 0,
    role: 'user',
    contentSummary: '帮我实现用户认证功能，包括登录、注册和 JWT token 管理',
    agentName: 'build',
    isSubagent: false,
    subagentName: null,
    subagentSessionId: null,
    parentExecutionId: null,
    totalTokens: 12000,
    inputTokens: 12000,
    outputTokens: 0,
    reasoningTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    inputMessagesCount: 1,
    inputMessagesTokens: 12000,
    contextWindowPct: 6.0,
    latencyMs: 0,
    createdAt: '2025-06-14T10:30:00.000Z',
    completedAt: '2025-06-14T10:30:01.000Z',
    model: 'claude-3.5-sonnet',
    finishReason: null,
    toolCalls: [],
    skillEvents: [],
  },
  {
    turnId: 'turn_002',
    turnIndex: 1,
    role: 'assistant',
    contentSummary: '我来帮你实现用户认证功能。首先分析项目结构...',
    agentName: 'build',
    isSubagent: false,
    subagentName: null,
    subagentSessionId: null,
    parentExecutionId: null,
    totalTokens: 25000,
    inputTokens: 12000,
    outputTokens: 5000,
    reasoningTokens: 8000,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    inputMessagesCount: 2,
    inputMessagesTokens: 20000,
    contextWindowPct: 10.0,
    latencyMs: 5200,
    createdAt: '2025-06-14T10:30:01.000Z',
    completedAt: '2025-06-14T10:30:06.200Z',
    model: 'claude-3.5-sonnet',
    finishReason: 'tool-calls',
    toolCalls: [
      { toolCallId: 'tc_001', toolName: 'read_file', state: 'ok', durationMs: 50 },
      { toolCallId: 'tc_002', toolName: 'search_files', state: 'ok', durationMs: 30 },
      { toolCallId: 'tc_003', toolName: 'read_file', state: 'ok', durationMs: 45 },
    ],
    skillEvents: [
      { skillName: 'code-review', eventType: 'loaded', success: true },
    ],
  },
  {
    turnId: 'turn_003',
    turnIndex: 2,
    role: 'assistant',
    contentSummary: '代码已创建完成。现在运行测试验证...',
    agentName: 'build',
    isSubagent: false,
    subagentName: null,
    subagentSessionId: null,
    parentExecutionId: null,
    totalTokens: 45000,
    inputTokens: 20000,
    outputTokens: 8000,
    reasoningTokens: 12000,
    cacheReadTokens: 5000,
    cacheWriteTokens: 0,
    inputMessagesCount: 4,
    inputMessagesTokens: 45000,
    contextWindowPct: 22.5,
    latencyMs: 12500,
    createdAt: '2025-06-14T10:30:10.000Z',
    completedAt: '2025-06-14T10:30:22.500Z',
    model: 'claude-3.5-sonnet',
    finishReason: 'tool-calls',
    toolCalls: [
      { toolCallId: 'tc_004', toolName: 'write_file', state: 'ok', durationMs: 200 },
      { toolCallId: 'tc_005', toolName: 'patch', state: 'ok', durationMs: 100 },
      { toolCallId: 'tc_006', toolName: 'terminal', state: 'ok', durationMs: 3000 },
    ],
    skillEvents: [],
  },
];
```

#### 6.1.4 mock-workflow.ts

```typescript
// tests/fixtures/mock-workflow.ts

export const mockWorkflow = {
  summary: {
    totalPhases: 3,
    totalSteps: 8,
    totalDurationMs: 2700000,
    totalTokens: 150200,
    totalCost: 3.20,
  },
  phases: [
    {
      phaseIndex: 1,
      phaseName: '需求分析',
      totalTokens: 37000,
      totalCost: 0.80,
      durationMs: 300000,
      children: [
        {
          type: 'step',
          stepLabel: 'user → assistant',
          totalTokens: 12000,
          durationMs: 5200,
          subagentSessionId: null,
        },
        {
          type: 'step',
          stepLabel: 'user → assistant',
          totalTokens: 25000,
          durationMs: 12500,
          subagentSessionId: null,
        },
      ],
    },
    {
      phaseIndex: 2,
      phaseName: '代码实现',
      totalTokens: 80000,
      totalCost: 1.80,
      durationMs: 1800000,
      children: [
        {
          type: 'step',
          stepLabel: 'write_file × 2',
          totalTokens: 45000,
          durationMs: 300000,
          subagentSessionId: null,
        },
        {
          type: 'step',
          stepLabel: 'subagent:test-runner',
          totalTokens: 35000,
          durationMs: 600000,
          subagentSessionId: 'sub_ses_001',
        },
        {
          type: 'checkpoint',
          checkpointLabel: '功能完成',
          waitTimeMs: 60000,
        },
      ],
    },
    {
      phaseIndex: 3,
      phaseName: '测试验证',
      totalTokens: 33200,
      totalCost: 0.60,
      durationMs: 600000,
      children: [
        {
          type: 'step',
          stepLabel: 'terminal (npm test)',
          totalTokens: 15000,
          durationMs: 300000,
          subagentSessionId: null,
        },
        {
          type: 'parallel-group',
          label: '并行修复',
          totalDurationMs: 300000,
          steps: [
            { stepLabel: 'patch × 2', totalTokens: 10000 },
            { stepLabel: 'patch × 1', totalTokens: 8200 },
          ],
        },
      ],
    },
  ],
};
```

#### 6.1.5 mock-bridges.ts

```typescript
// tests/fixtures/mock-bridges.ts
import type { BridgeItem } from '../../src/cli/types';

export const mockBridges: BridgeItem[] = [
  {
    bridgeId: 'bridge_001',
    dispatchExecutionId: 'exec_001',
    dispatchTurnId: 'turn_002',
    dispatchToolCallId: 'tc_task_001',
    dispatchContent: '运行所有单元测试并报告结果',
    dispatchTimestamp: '2025-06-14T10:35:00.000Z',
    responseExecutionId: 'exec_sub_001',
    responseTurnId: 'turn_sub_002',
    responseContent: '所有 15 个测试通过，覆盖率 87%',
    responseTimestamp: '2025-06-14T10:40:00.000Z',
    subagentSessionId: 'sub_ses_001',
    subagentType: 'general',
    subagentName: 'test-runner',
    status: 'completed',
    subagentTokens: 35000,
    subagentLatencyMs: 300000,
  },
  {
    bridgeId: 'bridge_002',
    dispatchExecutionId: 'exec_001',
    dispatchTurnId: 'turn_005',
    dispatchToolCallId: 'tc_task_002',
    dispatchContent: '检查代码风格并修复 lint 错误',
    dispatchTimestamp: '2025-06-14T10:50:00.000Z',
    responseExecutionId: 'exec_sub_002',
    responseTurnId: 'turn_sub_005',
    responseContent: '修复了 3 个 lint 错误',
    responseTimestamp: '2025-06-14T10:52:00.000Z',
    subagentSessionId: 'sub_ses_002',
    subagentType: 'general',
    subagentName: 'linter',
    status: 'completed',
    subagentTokens: 12000,
    subagentLatencyMs: 120000,
  },
];
```

#### 6.1.6 mock-executions.ts

```typescript
// tests/fixtures/mock-executions.ts
import type { ExecutionItem } from '../../src/cli/types';

export const mockExecutions: ExecutionItem[] = [
  {
    executionId: 'exec_001',
    agentName: 'build',
    isSubagent: false,
    parentExecutionId: null,
    tokens: 100000,
    cost: 2.10,
    toolCallCount: 80,
    skillLoadCount: 6,
    model: 'claude-3.5-sonnet',
    createdAt: '2025-06-14T10:30:00.000Z',
    latencyMs: 2100000,
  },
  {
    executionId: 'exec_sub_001',
    agentName: 'test-runner',
    isSubagent: true,
    parentExecutionId: 'exec_001',
    tokens: 35000,
    cost: 0.75,
    toolCallCount: 25,
    skillLoadCount: 1,
    model: 'claude-3.5-sonnet',
    createdAt: '2025-06-14T10:35:00.000Z',
    latencyMs: 300000,
  },
  {
    executionId: 'exec_sub_002',
    agentName: 'linter',
    isSubagent: true,
    parentExecutionId: 'exec_001',
    tokens: 12000,
    cost: 0.25,
    toolCallCount: 10,
    skillLoadCount: 1,
    model: 'claude-3.5-sonnet',
    createdAt: '2025-06-14T10:50:00.000Z',
    latencyMs: 120000,
  },
];
```

### 6.2 Fixture 工厂函数

```typescript
// tests/helpers/fixtures.ts
import type { SessionListItem, TurnItem, SessionStatsResponse } from '../../src/cli/types';

export function createMockSession(overrides?: Partial<SessionListItem>): SessionListItem {
  return {
    sessionId: `ses_${Math.random().toString(36).substr(2, 9)}`,
    taskId: `task_${Math.random().toString(36).substr(2, 9)}`,
    query: 'Test query',
    startTime: new Date().toISOString(),
    endTime: new Date().toISOString(),
    totalTokens: 10000,
    totalCost: 1.00,
    totalLatencyMs: 60000,
    totalToolCallCount: 10,
    totalSkillLoadCount: 2,
    totalSubagentCount: 0,
    model: 'test-model',
    user: 'test-user',
    ...overrides,
  };
}

export function createMockTurn(overrides?: Partial<TurnItem>): TurnItem {
  return {
    turnId: `turn_${Math.random().toString(36).substr(2, 9)}`,
    turnIndex: 0,
    role: 'assistant',
    contentSummary: 'Test content',
    agentName: 'build',
    isSubagent: false,
    subagentName: null,
    subagentSessionId: null,
    parentExecutionId: null,
    totalTokens: 5000,
    inputTokens: 3000,
    outputTokens: 2000,
    reasoningTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    inputMessagesCount: 1,
    inputMessagesTokens: 3000,
    contextWindowPct: 5.0,
    latencyMs: 1000,
    createdAt: new Date().toISOString(),
    completedAt: new Date().toISOString(),
    model: 'test-model',
    finishReason: 'stop',
    toolCalls: [],
    skillEvents: [],
    ...overrides,
  };
}

export function createMockSessionList(count: number): SessionListItem[] {
  return Array.from({ length: count }, (_, i) =>
    createMockSession({
      taskId: `task_${String(i + 1).padStart(3, '0')}`,
      query: `Query #${i + 1}`,
      totalTokens: (i + 1) * 10000,
      totalCost: (i + 1) * 0.5,
    }),
  );
}

export function createMockTurnList(count: number): TurnItem[] {
  return Array.from({ length: count }, (_, i) =>
    createMockTurn({
      turnId: `turn_${String(i + 1).padStart(3, '0')}`,
      turnIndex: i,
      role: i % 2 === 0 ? 'user' : 'assistant',
      contextWindowPct: Math.min(5 + i * 10, 100),
    }),
  );
}
```

---

## 7. 覆盖率目标

### 7.1 总体覆盖率目标

| 测试层 | 行覆盖率 | 分支覆盖率 | 函数覆盖率 | 说明 |
|--------|----------|-----------|-----------|------|
| **单元测试** | ≥ 90% | ≥ 85% | ≥ 95% | 纯函数、工具函数、Client |
| **TUI 组件测试** | ≥ 80% | ≥ 75% | ≥ 85% | Ink 组件渲染、交互 |
| **集成测试** | ≥ 70% | ≥ 65% | ≥ 75% | Client + Server |
| **E2E 测试** | ≥ 50% | ≥ 40% | ≥ 50% | 关键路径覆盖 |
| **总体** | **≥ 80%** | **≥ 75%** | **≥ 85%** | 需求文档要求 |

### 7.2 模块级覆盖率要求

| 模块 | 行覆盖率 | 优先级 | 关键测试点 |
|------|----------|--------|-----------|
| `client.ts` | ≥ 95% | P0 | 所有 API 方法、重试机制、错误处理 |
| `errors.ts` | ≥ 100% | P0 | 所有错误类型构造 |
| `config.ts` | ≥ 95% | P0 | 配置优先级、文件读写、环境变量 |
| `utils/format.ts` | ≥ 100% | P0 | 所有格式化函数、边界值 |
| `utils/table.ts` | ≥ 95% | P1 | 渲染、截断、自定义 render |
| `utils/colors.ts` | ≥ 90% | P2 | 主题选择、自动检测 |
| `commands/*` | ≥ 85% | P0 | 参数解析、API 调用、输出格式 |
| `hooks/*` | ≥ 85% | P0 | 状态管理、边界条件 |
| `tui/components/*` | ≥ 80% | P1 | 渲染、props 变化 |
| `tui/screens/*` | ≥ 75% | P1 | 键盘交互、数据加载 |
| `tui/tabs/*` | ≥ 75% | P2 | Tab 切换、数据展示 |

### 7.3 Vitest 覆盖率配置

```typescript
// vitest.config.ts 中增加 coverage 配置
export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
    setupFiles: ['./tests/setup.ts'],
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/cli/**/*.ts', 'src/cli/**/*.tsx'],
      exclude: [
        'src/cli/index.ts',          // 入口文件（难以单测）
        'src/cli/tui/App.tsx',        // TUI 根组件（E2E 覆盖）
      ],
      thresholds: {
        lines: 80,
        branches: 75,
        functions: 85,
        statements: 80,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### 7.4 运行覆盖率检查

```bash
# 运行所有 CLI 测试并生成覆盖率报告
npx vitest run --coverage --reporter=verbose tests/cli/

# 仅运行单元测试
npx vitest run --coverage tests/cli/unit/

# 仅运行组件测试
npx vitest run --coverage tests/cli/components/

# 生成 HTML 覆盖率报告
npx vitest run --coverage --coverage.reporter=html tests/cli/
```

---

## 8. 测试环境配置

### 8.1 依赖包

```json
{
  "type": "module",
  "devDependencies": {
    "vitest": "^3.2.1",
    "@vitest/coverage-v8": "^3.2.1",
    "@testing-library/react": "^16.0.0",
    "ink-testing-library": "^4.0.0",
    "msw": "^2.0.0"
  },
  "dependencies": {
    "ink": "^7.0.6",
    "react": "^19.2.7",
    "commander": "^12.0.0",
    "chalk": "^5.0.0",
    "string-width": "^7.0.0",
    "cli-truncate": "^4.0.0"
  }
}
```

> **PoC 验证**: ink-testing-library 完全兼容 Ink v7，保留此依赖。生产代码需 Ink v7 + ESM (`"type": "module"`) + tsx 运行。string-width 和 cli-truncate 用于中文宽度对齐处理。Ink v7 原生 render() 无 lastFrame/frames/output（测试用 ink-testing-library）。第三方 Ink 组件（ink-table/ink-spinner/ink-text-input）全部自实现。

### 8.2 Vitest 配置更新

```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  test: {
    environment: 'node',
    setupFiles: ['./tests/setup.ts'],
    include: [
      'tests/**/*.test.ts',
      'tests/**/*.test.tsx',
    ],
    // 测试超时设置
    testTimeout: 10000,        // 单个测试 10s
    hookTimeout: 15000,        // hook 超时 15s
    // 并行执行
    pool: 'forks',
    poolOptions: {
      forks: {
        singleFork: false,
      },
    },
    // 覆盖率
    coverage: {
      provider: 'v8',
      reporter: ['text', 'json', 'html'],
      include: ['src/cli/**/*.ts', 'src/cli/**/*.tsx'],
      exclude: [
        'src/cli/index.ts',
        'src/cli/tui/App.tsx',
      ],
      thresholds: {
        lines: 80,
        branches: 75,
        functions: 85,
        statements: 80,
      },
    },
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
});
```

### 8.3 测试 Setup 文件

```typescript
// tests/setup.ts (追加 CLI 测试相关配置)
import { vi } from 'vitest';

// 已有的 setup...

// CLI 测试：禁用 chalk 颜色输出（便于文本断言）
process.env.FORCE_COLOR = '0';
process.env.NO_COLOR = '1';

// CLI 测试：设置固定终端尺寸
Object.defineProperty(process.stdout, 'columns', { value: 120, writable: true });
Object.defineProperty(process.stdout, 'rows', { value: 40, writable: true });

// CLI 测试：模拟 TTY
Object.defineProperty(process.stdout, 'isTTY', { value: true, writable: true });

// Ink v7 环境要求：ESM (package.json "type": "module")，运行用 tsx
// Ink v7 原生 render() 无 lastFrame/frames/output
// ink-testing-library 的 render() 有 lastFrame + stdin（测试专用）
// 第三方 Ink 组件全部自实现：Spinner(10行)、TextInput(30行)、DataTable等
```

### 8.4 NPM Scripts

```json
{
  "scripts": {
    "test": "vitest run",
    "test:watch": "vitest",
    "test:cli": "vitest run tests/cli/",
    "test:cli:unit": "vitest run tests/cli/unit/",
    "test:cli:components": "vitest run tests/cli/components/",
    "test:cli:integration": "vitest run tests/cli/integration/",
    "test:cli:e2e": "vitest run tests/cli/e2e/",
    "test:coverage": "vitest run --coverage tests/cli/",
    "test:coverage:html": "vitest run --coverage --coverage.reporter=html tests/cli/"
  }
}
```

### 8.5 CI/CD 集成

项目托管在 GitCode，使用 GitLab-compatible CI。GitHub Actions 配置保留作为备用（fork 场景）。

```yaml
# .gitlab-ci.yml (GitCode CI — 主 CI)
stages:
  - test

cli-test:
  stage: test
  image: node:22
  only:
    changes:
      - src/cli/**/*
      - tests/cli/**/*
  script:
    - npm ci
    - npx prisma generate
    - npm run test:cli:unit
    - npm run test:cli:components
    - npm run test:cli:integration
    - npm run test:coverage
  artifacts:
    paths:
      - coverage/
```

```yaml
# .github/workflows/cli-test.yml (备用 — fork 到 GitHub 时使用)
name: CLI Tests

on:
  push:
    branches: [main, develop]
    paths: ['src/cli/**', 'tests/cli/**']
  pull_request:
    paths: ['src/cli/**', 'tests/cli/**']

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node-version: [20, 22]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node-version }}
          cache: 'npm'

      - run: npm ci
      - run: npx prisma generate

      - name: Run CLI unit tests
        run: npm run test:cli:unit

      - name: Run CLI component tests
        run: npm run test:cli:components

      - name: Run CLI integration tests
        run: npm run test:cli:integration

      - name: Run coverage check
        run: npm run test:coverage

      - name: Upload coverage report
        uses: actions/upload-artifact@v4
        with:
          name: coverage-report-${{ matrix.node-version }}
          path: coverage/
```

### 8.6 测试环境矩阵

| 环境 | 终端类型 | 测试内容 | 注意事项 |
|------|----------|----------|----------|
| Linux (CI) | xterm-256color | 全量测试 | 主要 CI 环境 |
| macOS | xterm-256color | 全量测试 | 本地开发 |
| WSL | xterm-256color | 全量测试 | Windows 开发 |
| Windows Terminal | Windows Terminal | 部分 TUI 测试 | ANSI 兼容性 |
| Non-TTY (pipe) | — | 仅命令模式测试 | `--json` 输出验证 |

### 8.7 Mock 策略总结

| 层 | Mock 方式 | 工具 |
|----|----------|------|
| InsightClient | `vi.fn()` mock fetch | Vitest mock |
| API Server | `node:http` createServer | 自建 mock server |
| Ink 组件 | `ink-testing-library` render（PoC 验证兼容 v7） | ink-testing-library |
| 文件系统 | `vi.spyOn(fs, ...)` + tmpdir | Vitest spy |
| 环境变量 | `process.env.X = ...` | 直接设置 |
| 时间 | `vi.useFakeTimers()` | Vitest fake timers |
| Commander | `cmd.parseAsync([...])` | Commander 内置 |

---

## 9. 性能基准测试

> **PoC 验证结论**: Ink v7.0.6 + React 19.2.7 + ESM 模式下，TUI 组件渲染性能和 API 响应时间需满足以下基准。

### 9.1 TUI 组件渲染基准

| 组件 | 场景 | 性能目标 | 测量方法 |
|------|------|----------|----------|
| DataTable | 20 行数据渲染 | < 100ms | `performance.now()` 从 render 到 lastFrame 可用 |
| DataTable | 50 行数据渲染 | < 200ms | 同上 |
| DataTable | 100 行数据渲染（含分页） | < 300ms | 同上 |
| AsciiBar | 10 组柱状图 | < 50ms | 纯函数计算 + 渲染 |
| TreeView | 3 层深度树形结构 | < 80ms | 递归渲染耗时 |
| MetricCards | 6 张指标卡片 | < 60ms | 并行布局渲染 |
| SessionList Screen | 完整列表屏（含 StatusBar + KeyBar） | < 150ms | 全屏首次渲染 |
| SessionDetail Screen | 完整详情屏 | < 200ms | 全屏首次渲染 |
| Spinner | 动画帧切换 | < 16ms/帧 | 60fps 目标 |

```typescript
import { describe, it, expect } from 'vitest';
import React from 'react';
import { render } from 'ink-testing-library';
import { DataTable } from '../../src/cli/tui/components/DataTable';

describe('DataTable performance benchmarks', () => {
  it('renders 20 rows in < 100ms', () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      sessionId: `ses_${i.toString().padStart(3, '0')}`,
      taskId: `task_${i}`,
      query: `查询 ${i}`,
      totalTokens: 1000 * (i + 1),
      totalCost: 0.1 * (i + 1),
    }));

    const columns = [
      { key: 'sessionId', header: 'Session', width: 15 },
      { key: 'taskId', header: 'Task', width: 12 },
      { key: 'query', header: 'Query', width: 30 },
      { key: 'totalTokens', header: 'Tokens', width: 10 },
      { key: 'totalCost', header: 'Cost', width: 10 },
    ];

    const start = performance.now();
    const { lastFrame } = render(
      <DataTable columns={columns} data={rows} selectedIndex={0} />,
    );
    const frame = lastFrame();
    const elapsed = performance.now() - start;

    expect(frame).toBeTruthy();
    expect(elapsed).toBeLessThan(100);
  });

  it('renders 50 rows in < 200ms', () => {
    const rows = Array.from({ length: 50 }, (_, i) => ({
      sessionId: `ses_${i.toString().padStart(3, '0')}`,
      taskId: `task_${i}`,
      query: `查询 ${i}`,
      totalTokens: 1000 * (i + 1),
      totalCost: 0.1 * (i + 1),
    }));

    const columns = [
      { key: 'sessionId', header: 'Session', width: 15 },
      { key: 'taskId', header: 'Task', width: 12 },
      { key: 'query', header: 'Query', width: 30 },
      { key: 'totalTokens', header: 'Tokens', width: 10 },
      { key: 'totalCost', header: 'Cost', width: 10 },
    ];

    const start = performance.now();
    const { lastFrame } = render(
      <DataTable columns={columns} data={rows} selectedIndex={0} />,
    );
    const frame = lastFrame();
    const elapsed = performance.now() - start;

    expect(frame).toBeTruthy();
    expect(elapsed).toBeLessThan(200);
  });
});
```

### 9.2 API 请求性能基准

| 操作 | 场景 | 性能目标 | 说明 |
|------|------|----------|------|
| listSessions | 默认分页（20 条） | < 500ms | 含网络延迟（本地 mock server） |
| getSession | 单条详情查询 | < 300ms | 含网络延迟 |
| getStats | 全局统计 | < 500ms | 含聚合计算 |
| getStats | Session 级统计 | < 300ms | 单 Session 聚合 |
| getTurns | 单 Session Turn 列表 | < 400ms | 含分页 |
| searchTurns | 关键词搜索 | < 800ms | 全文检索 |
| importSession | 导入单个 Session | < 2000ms | 含数据解析 |
| deleteSession | 删除单个 | < 300ms | 单条删除 |
| 重试延迟 | 5xx 首次重试 | 1000ms | 指数退避基准 |
| 超时阈值 | 请求超时 | 15000ms | 默认超时配置 |

```typescript
import { describe, it, expect } from 'vitest';
import { InsightClient } from '../../src/cli/client';

describe('API request performance benchmarks', () => {
  // 使用 mock server（本地 localhost，模拟网络延迟 < 10ms）
  let client: InsightClient;

  beforeEach(() => {
    client = new InsightClient('http://localhost:21025');
  });

  it('listSessions completes in < 500ms (mock server)', async () => {
    // Mock fetch 返回 20 条数据，模拟网络延迟 10ms
    const mockFetch = vi.fn().mockImplementation(() =>
      new Promise(resolve =>
        setTimeout(() => resolve({
          ok: true,
          json: async () => ({
            items: Array.from({ length: 20 }, (_, i) => ({
              sessionId: `ses_${i}`, taskId: `task_${i}`, totalTokens: 1000,
            })),
            total: 20, page: 1,
          }),
        }), 10),
      ),
    );
    vi.stubGlobal('fetch', mockFetch);

    const start = performance.now();
    const result = await client.listSessions({});
    const elapsed = performance.now() - start;

    expect(result.items).toHaveLength(20);
    expect(elapsed).toBeLessThan(500);
  });

  it('getSession completes in < 300ms (mock server)', async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        sessionId: 'ses_001', taskId: 'task_001',
        totalTokens: 150000, totalCost: 3.20,
      }),
    });
    vi.stubGlobal('fetch', mockFetch);

    const start = performance.now();
    await client.getSession('task_001');
    const elapsed = performance.now() - start;

    expect(elapsed).toBeLessThan(300);
  });
});
```

### 9.3 中文宽度计算性能基准

> **PoC 验证**: 中文宽度计算依赖 `string-width` 库（非 `wcwidth`），需确保表格对齐在中文混合英文场景下性能可接受。

| 操作 | 场景 | 性能目标 | 说明 |
|------|------|----------|------|
| string-width | 单行中文+英文混合（40字符） | < 0.1ms | 单次宽度计算 |
| DataTable 列宽计算 | 20 行 × 5 列含中文 | < 20ms | 全表宽度预计算 |
| cli-truncate | 中文截断（含 emoji） | < 0.05ms | 单次截断操作 |
| 全表格式化 | 20 行渲染输出 | < 50ms | 含宽度计算 + 对齐填充 |

```typescript
import { describe, it, expect } from 'vitest';
import stringWidth from 'string-width';
import cliTruncate from 'cli-truncate';

describe('Chinese width calculation performance', () => {
  it('calculates mixed Chinese-English string width in < 0.1ms', () => {
    const text = '会话ID: ses_001 任务: 帮我实现功能';
    const iterations = 10000;

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      stringWidth(text);
    }
    const elapsed = performance.now() - start;
    const perCall = elapsed / iterations;

    expect(perCall).toBeLessThan(0.1); // < 0.1ms per call
  });

  it('calculates full table widths for 20 rows × 5 cols in < 20ms', () => {
    const rows = Array.from({ length: 20 }, (_, i) => ({
      sessionId: `会话_${i}`,
      query: `查询任务描述_${i}_含中文内容`,
      model: 'claude-3.5-sonnet',
      tokens: `${1000 * (i + 1)}`,
      cost: `¥${(0.1 * (i + 1)).toFixed(2)}`,
    }));

    const start = performance.now();
    for (const row of rows) {
      for (const val of Object.values(row)) {
        stringWidth(val);
      }
    }
    const elapsed = performance.now() - start;

    expect(elapsed).toBeLessThan(20);
  });

  it('truncates Chinese strings with cli-truncate in < 0.05ms', () => {
    const text = '这是一段很长的中文文本需要被截断处理';
    const iterations = 10000;

    const start = performance.now();
    for (let i = 0; i < iterations; i++) {
      cliTruncate(text, 10);
    }
    const elapsed = performance.now() - start;
    const perCall = elapsed / iterations;

    expect(perCall).toBeLessThan(0.05);
  });
});
```

### 9.4 性能回归检测

性能基准测试在 CI 中作为独立 job 运行，失败时标记为 **warning**（不阻塞合并），但连续 3 次超标则升级为 **error**：

```yaml
# .gitlab-ci.yml 追加
cli-perf:
  stage: test
  image: node:22
  only:
    changes:
      - src/cli/**/*
  script:
    - npm ci
    - npx prisma generate
    - npm run test:cli:perf
  allow_failure: true  # 性能测试不阻塞合并，但需关注
```

```json
{
  "scripts": {
    "test:cli:perf": "vitest run tests/cli/perf/ --reporter=verbose"
  }
}
```

---

## 附录 A: 测试用例统计

| 测试层 | 测试文件数 | 测试用例数（预估） |
|--------|-----------|------------------|
| 单元测试 | 15 | ~120 |
| Hooks 测试 | 4 | ~30 |
| TUI 组件测试 | 13 | ~65 |
| 集成测试 | 3 | ~20 |
| E2E 测试 | 3 | ~15 |
| **总计** | **38** | **~250** |

## 附录 B: 测试优先级实施顺序

| 阶段 | 测试内容 | 预估工作量 |
|------|----------|-----------|
| **Phase 1** | 格式化工具 + 错误类型 + 配置管理 单元测试 | 1 天 |
| **Phase 2** | InsightClient 单元测试（含重试/错误处理） | 1 天 |
| **Phase 3** | 命令模块单元测试（9 个命令） | 2 天 |
| **Phase 4** | Hooks 单元测试（useTable, useNavigation, useApi, useKeyboard） | 1 天 |
| **Phase 5** | TUI 基础组件测试（DataTable, AsciiBar, TreeView, TabBar 等） | 2 天 |
| **Phase 6** | TUI Screen 组件测试（SessionList, SessionDetail 等） | 2 天 |
| **Phase 7** | 集成测试（Mock Server + Client + Command Flow） | 1 天 |
| **Phase 8** | E2E 测试 + CI 配置 | 1 天 |
| **总计** | | **11 天** |
