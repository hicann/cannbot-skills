// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { sessionCommand } from '@/cli/commands/session';
import { InsightClient } from '@/cli/client';
import { formatTokens, formatCost, formatDuration, formatDate } from '@/cli/utils/format';
import { formatHeader, formatLabel } from '@/cli/utils/colors';
import type { ApiSessionDetailResponse, ApiSessionStatsResponse } from '@/cli/types';

const mockGetSession = vi.fn();
const mockGetStats = vi.fn();
vi.spyOn(InsightClient.prototype, 'getSession').mockImplementation(mockGetSession);
vi.spyOn(InsightClient.prototype, 'getStats').mockImplementation(mockGetStats);

const sampleSession: ApiSessionDetailResponse = {
  sessionId: 's1',
  taskId: 'task-001',
  label: null,
  query: 'Fix the bug',
  framework: 'opencode',
  startTime: '2026-06-14T10:30:00Z',
  endTime: '2026-06-14T10:45:00Z',
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
  sourcePath: '/path/to/db',
  agents: [],
  skills: [],
};

const sampleStats: ApiSessionStatsResponse = {
  taskId: 'task-001',
  totalTokens: 15000,
  totalInputTokens: 10000,
  totalOutputTokens: 5000,
  totalReasoningTokens: 2000,
  totalCacheReadTokens: 3000,
  totalCacheWriteTokens: 1000,
  totalCost: 0.45,
  totalLatencyMs: 900000,
  totalToolCallCount: 5,
  totalSkillLoadCount: 2,
  totalSubagentCount: 1,
  totalLlmCallCount: 10,
};

describe('sessionCommand', () => {
  beforeEach(() => {
    mockGetSession.mockReset();
    mockGetStats.mockReset();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = sessionCommand();
    expect(cmd.name()).toBe('session');
  });

  it('has --json option', () => {
    const cmd = sessionCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--json');
  });

  it('calls client.getSession and client.getStats', async () => {
    mockGetSession.mockResolvedValueOnce(sampleSession);
    mockGetStats.mockResolvedValueOnce(sampleStats);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    await client.getSession('task-001');
    await client.getStats('task-001');

    expect(mockGetSession).toHaveBeenCalledWith('task-001');
    expect(mockGetStats).toHaveBeenCalledWith('task-001');
  });

  it('formats session detail values correctly', () => {
    const s = sampleSession;

    expect(formatTokens(s.totalTokens)).toBe('15.0K');
    expect(formatCost(s.totalCost)).toBe('$0.45');
    expect(formatDuration(s.totalLatencyMs)).toBe('15m0s');
    expect(formatDate(s.startTime)).toBeTruthy();

    const header = formatHeader(`Session: ${s.taskId}`);
    expect(header).toContain('task-001');
  });

  it('formats label/value pairs', () => {
    const s = sampleSession;
    const label = formatLabel('Model', s.model ?? '—');
    expect(label).toContain('Model');
    expect(label).toContain('claude-3.5-sonnet');

    const tokenLabel = formatLabel('Total Tokens', formatTokens(s.totalTokens));
    expect(tokenLabel).toContain('15.0K');
  });

  it('JSON output includes both session and stats', async () => {
    mockGetSession.mockResolvedValueOnce(sampleSession);
    mockGetStats.mockResolvedValueOnce(sampleStats);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const session = await client.getSession('task-001');
    const stats = await client.getStats('task-001');

    const json = JSON.stringify({ session, stats }, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.session.taskId).toBe('task-001');
    expect(parsed.stats.totalTokens).toBe(15000);
  });

  it('formats agent data when agents exist', () => {
    const agents = [
      { executionId: 'e1', agentName: 'main', isSubagent: false, parentExecutionId: null, tokens: 10000, cost: 0.3, toolCallCount: 3, skillLoadCount: 1, model: 'claude-3.5-sonnet', createdAt: '2026-06-14T10:30:00Z', latencyMs: 600000 },
    ];

    expect(formatTokens(agents[0].tokens)).toBe('10.0K');
    expect(formatCost(agents[0].cost)).toBe('$0.30');
    expect(formatDuration(agents[0].latencyMs)).toBe('10m0s');
  });
});
