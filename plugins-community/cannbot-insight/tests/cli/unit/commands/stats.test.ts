// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { statsCommand } from '@/cli/commands/stats';
import { InsightClient } from '@/cli/client';
import { formatTokens, formatCost, formatDuration } from '@/cli/utils/format';
import type { ApiGlobalStatsResponse, ApiSessionStatsResponse } from '@/cli/types';

const mockGetStats = vi.fn();
vi.spyOn(InsightClient.prototype, 'getStats').mockImplementation(mockGetStats);

const globalStats: ApiGlobalStatsResponse = {
  totalSessions: 10,
  totalTokens: 150000,
  totalCost: 4.50,
  totalLatencyMs: 9_000_000,
  avgLatencyMs: 900_000,
};

const sessionStats: ApiSessionStatsResponse = {
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

describe('statsCommand', () => {
  beforeEach(() => {
    mockGetStats.mockReset();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = statsCommand();
    expect(cmd.name()).toBe('stats');
  });

  it('has --session and --json options', () => {
    const cmd = statsCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--session');
    expect(options).toContain('--json');
  });

  it('formats global stats correctly', () => {
    const s = globalStats;

    expect(String(s.totalSessions)).toBe('10');
    expect(formatTokens(s.totalTokens)).toBe('150.0K');
    expect(formatCost(s.totalCost)).toBe('$4.50');
    expect(formatDuration(s.totalLatencyMs)).toBe('150m0s');
    expect(formatDuration(s.avgLatencyMs)).toBe('15m0s');
  });

  it('computes avg tokens/cost per session', () => {
    const s = globalStats;
    const avgTokens = s.totalTokens / s.totalSessions;
    const avgCost = s.totalCost / s.totalSessions;

    expect(formatTokens(avgTokens)).toBe('15.0K');
    expect(formatCost(avgCost)).toBe('$0.45');
  });

  it('formats session stats correctly', () => {
    const s = sessionStats;

    expect(formatTokens(s.totalTokens)).toBe('15.0K');
    expect(formatTokens(s.totalInputTokens)).toBe('10.0K');
    expect(formatTokens(s.totalOutputTokens)).toBe('5.0K');
    expect(formatCost(s.totalCost)).toBe('$0.45');
    expect(formatDuration(s.totalLatencyMs)).toBe('15m0s');
  });

  it('JSON output for global stats', () => {
    const json = JSON.stringify(globalStats, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.totalSessions).toBe(10);
    expect(parsed.totalTokens).toBe(150000);
  });

  it('JSON output for session stats', () => {
    const json = JSON.stringify(sessionStats, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.taskId).toBe('task-001');
    expect(parsed.totalTokens).toBe(15000);
  });
});
