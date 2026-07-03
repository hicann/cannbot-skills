// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { compareCommand } from '@/cli/commands/compare';
import { InsightClient } from '@/cli/client';
import { renderTable } from '@/cli/utils/table';
import { formatTokens, formatCost, formatDuration } from '@/cli/utils/format';
import type { ApiSessionDetailResponse } from '@/cli/types';

const mockGetSession = vi.fn();
vi.spyOn(InsightClient.prototype, 'getSession').mockImplementation(mockGetSession);

const session1: ApiSessionDetailResponse = {
  sessionId: 's1',
  taskId: 'task-001',
  label: null,
  query: 'Fix bug',
  framework: 'opencode',
  startTime: '2026-06-14T10:00:00Z',
  endTime: '2026-06-14T10:15:00Z',
  totalTokens: 15000,
  totalInputTokens: 10000,
  totalOutputTokens: 5000,
  totalReasoningTokens: 2000,
  totalCacheReadTokens: 3000,
  totalCacheWriteTokens: 1000,
  totalCost: 0.45,
  totalLatencyMs: 900000,
  totalToolCallCount: 5,
  totalLlmCallCount: 10,
  totalSkillLoadCount: 2,
  totalSubagentCount: 1,
  model: 'claude-3.5-sonnet',
  user: 'alice',
  sourcePath: null,
  agents: [],
  skills: [],
};

const session2: ApiSessionDetailResponse = {
  sessionId: 's2',
  taskId: 'task-002',
  label: null,
  query: 'Refactor code',
  framework: 'opencode',
  startTime: '2026-06-14T11:00:00Z',
  endTime: '2026-06-14T11:20:00Z',
  totalTokens: 25000,
  totalInputTokens: 18000,
  totalOutputTokens: 7000,
  totalReasoningTokens: 3000,
  totalCacheReadTokens: 5000,
  totalCacheWriteTokens: 2000,
  totalCost: 0.75,
  totalLatencyMs: 1200000,
  totalToolCallCount: 8,
  totalLlmCallCount: 15,
  totalSkillLoadCount: 3,
  totalSubagentCount: 2,
  model: 'gpt-4o',
  user: 'bob',
  sourcePath: null,
  agents: [],
  skills: [],
};

describe('compareCommand', () => {
  beforeEach(() => {
    mockGetSession.mockReset();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = compareCommand();
    expect(cmd.name()).toBe('compare');
  });

  it('has --json option', () => {
    const cmd = compareCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--json');
  });

  it('calls getSession for both task IDs', async () => {
    mockGetSession.mockResolvedValueOnce(session1);
    mockGetSession.mockResolvedValueOnce(session2);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const [s1, s2] = await Promise.all([
      client.getSession('task-001'),
      client.getSession('task-002'),
    ]);

    expect(mockGetSession).toHaveBeenCalledWith('task-001');
    expect(mockGetSession).toHaveBeenCalledWith('task-002');
    expect(s1.taskId).toBe('task-001');
    expect(s2.taskId).toBe('task-002');
  });

  it('builds compare rows with formatted metrics', () => {
    const rows = [
      { metric: 'Tokens', session1: formatTokens(session1.totalTokens), session2: formatTokens(session2.totalTokens) },
      { metric: 'Cost', session1: formatCost(session1.totalCost), session2: formatCost(session2.totalCost) },
      { metric: 'Duration', session1: formatDuration(session1.totalLatencyMs), session2: formatDuration(session2.totalLatencyMs) },
      { metric: 'Model', session1: session1.model ?? '—', session2: session2.model ?? '—' },
      { metric: 'User', session1: session1.user ?? '—', session2: session2.user ?? '—' },
    ];

    expect(rows[0].session1).toBe('15.0K');
    expect(rows[0].session2).toBe('25.0K');
    expect(rows[1].session1).toBe('$0.45');
    expect(rows[1].session2).toBe('$0.75');
    expect(rows[2].session1).toBe('15m0s');
    expect(rows[2].session2).toBe('20m0s');
    expect(rows[3].session1).toBe('claude-3.5-sonnet');
    expect(rows[3].session2).toBe('gpt-4o');
  });

  it('renders comparison table', () => {
    const COMPARE_COLUMNS = [
      { key: 'metric', label: 'Metric', width: 20 },
      { key: 'session1', label: 'Session A', width: 25 },
      { key: 'session2', label: 'Session B', width: 25 },
    ];

    const rows = [
      { metric: 'Tokens', session1: formatTokens(session1.totalTokens), session2: formatTokens(session2.totalTokens) },
      { metric: 'Cost', session1: formatCost(session1.totalCost), session2: formatCost(session2.totalCost) },
    ];

    const table = renderTable(COMPARE_COLUMNS, rows as unknown as Record<string, unknown>[]);
    expect(table).toContain('Metric');
    expect(table).toContain('15.0K');
    expect(table).toContain('25.0K');
    expect(table).toContain('$0.45');
    expect(table).toContain('$0.75');
  });

  it('JSON output includes both sessions', async () => {
    mockGetSession.mockResolvedValueOnce(session1);
    mockGetSession.mockResolvedValueOnce(session2);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const [s1, s2] = await Promise.all([
      client.getSession('task-001'),
      client.getSession('task-002'),
    ]);

    const json = JSON.stringify({ session1: s1, session2: s2 }, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.session1.taskId).toBe('task-001');
    expect(parsed.session2.taskId).toBe('task-002');
  });
});
