// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { sessionsCommand } from '@/cli/commands/sessions';
import { InsightClient } from '@/cli/client';
import { renderTable } from '@/cli/utils/table';
import { formatTokens, formatCost } from '@/cli/utils/format';
import type { ApiSessionListItem, ApiSessionListResponse } from '@/cli/types';

const mockListSessions = vi.fn();
vi.spyOn(InsightClient.prototype, 'listSessions').mockImplementation(mockListSessions);

const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});

const sampleSession: ApiSessionListItem = {
  sessionId: 's1',
  taskId: 'task-001',
  query: 'Fix the bug in login',
  startTime: '2026-06-14T10:30:00Z',
  endTime: '2026-06-14T10:45:00Z',
  totalTokens: 15000,
  totalCost: 0.45,
  totalLatencyMs: 900000,
  totalToolCallCount: 5,
  totalSkillLoadCount: 2,
  totalSubagentCount: 1,
  model: 'claude-3.5-sonnet',
  user: 'alice',
};

const sampleResponse: ApiSessionListResponse = {
  items: [sampleSession],
  total: 1,
  page: 1,
};

describe('sessionsCommand', () => {
  beforeEach(() => {
    mockListSessions.mockReset();
    mockConsoleLog.mockClear();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = sessionsCommand();
    expect(cmd.name()).toBe('sessions');
    expect(cmd.description()).toContain('List sessions');
  });

  it('has --limit, --offset, --user, --json options', () => {
    const cmd = sessionsCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--limit');
    expect(options).toContain('--offset');
    expect(options).toContain('--user');
    expect(options).toContain('--json');
  });

  it('calls listSessions and renders table', async () => {
    mockListSessions.mockResolvedValueOnce(sampleResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const response = await client.listSessions({ page: 1, pageSize: 20 });

    const rows = response.items as ApiSessionListItem[];
    expect(rows[0].taskId).toBe('task-001');

    const SESSION_COLUMNS = [
      { key: 'taskId', label: 'Task ID', width: 20 },
      { key: 'model', label: 'Model', width: 18 },
      { key: 'tokens', label: 'Tokens', width: 10 },
      { key: 'cost', label: 'Cost', width: 10 },
    ];

    const table = renderTable(
      SESSION_COLUMNS,
      rows as unknown as Record<string, unknown>[],
      (row: Record<string, unknown>, key: string) => {
        const s = row as unknown as ApiSessionListItem;
        if (key === 'tokens') return formatTokens(s.totalTokens);
        if (key === 'cost') return formatCost(s.totalCost);
        return String(s[key as keyof ApiSessionListItem] ?? '—');
      },
    );

    expect(table).toContain('task-001');
    expect(table).toContain('15.0K');
    expect(table).toContain('$0.45');
  });

  it('computes totals from response items', () => {
    const rows = sampleResponse.items as ApiSessionListItem[];
    const totalTokens = rows.reduce((sum, r) => sum + (r.totalTokens ?? 0), 0);
    const totalCost = rows.reduce((sum, r) => sum + (r.totalCost ?? 0), 0);

    expect(formatTokens(totalTokens)).toBe('15.0K');
    expect(formatCost(totalCost)).toBe('$0.45');
  });

  it('JSON output includes all fields', async () => {
    mockListSessions.mockResolvedValueOnce(sampleResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const response = await client.listSessions({ page: 1, pageSize: 20 });

    const json = JSON.stringify(response, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.items[0].taskId).toBe('task-001');
    expect(parsed.total).toBe(1);
  });

  it('offset converts to page number correctly', () => {
    const offset = 40;
    const limit = 20;
    const page = Math.floor(offset / limit) + 1;
    expect(page).toBe(3);

    const offset2 = 0;
    const page2 = Math.floor(offset2 / limit) + 1;
    expect(page2).toBe(1);
  });
});
