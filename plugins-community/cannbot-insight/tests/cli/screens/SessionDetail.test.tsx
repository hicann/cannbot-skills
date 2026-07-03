// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { SessionDetail } from '@/cli/tui/screens/SessionDetail';
import { InsightClient } from '@/cli/client';

let callCount = 0;
vi.mock('@/cli/hooks/useApi', () => ({
  useApi: () => {
    callCount++;
    if (callCount === 1) {
      return {
        data: {
          sessionId: 's1', taskId: 'task-001', label: 'Test', query: 'Fix bug', framework: null,
          startTime: '2026-06-14T10:00:00Z', endTime: '2026-06-14T10:05:00Z',
          totalTokens: 5000, totalInputTokens: 3000, totalOutputTokens: 1500, totalReasoningTokens: 500,
          totalCacheReadTokens: 200, totalCacheWriteTokens: 100, totalCost: 0.05, totalLatencyMs: 300000,
          totalToolCallCount: 10, totalLlmCallCount: 5, totalSkillLoadCount: 3, totalSubagentCount: 2,
          model: 'gpt-4o', user: 'alice', sourcePath: null,
          agents: [], skills: [],
        },
        loading: false, refresh: vi.fn(),
      };
    }
    if (callCount === 2) return { data: { items: [], total: 0 }, loading: false, refresh: vi.fn() };
    if (callCount === 3) return { data: { phases: [], summary: { totalPhases: 0, totalSteps: 0, totalCheckpoints: 0, totalActiveTimeMs: 0, totalWaitTimeMs: 0, activeTimePct: 0, iterations: 0 } }, loading: false, refresh: vi.fn() };
    if (callCount === 4) return { data: { items: [], root: [], subagents: [], totalExecutions: 0, subagentCount: 0 }, loading: false, refresh: vi.fn() };
    return { data: { items: [], total: 0 }, loading: false, refresh: vi.fn() };
  },
}));

vi.mock('@/cli/hooks/useKeyboard', () => ({
  useKeyboard: () => {},
}));

describe('SessionDetail', () => {
  beforeEach(() => {
    callCount = 0;
  });

  it('renders tab bar', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <SessionDetail client={client} taskId="task-001" onBack={vi.fn()} onSelectTurn={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Overview');
    expect(output).toContain('Turns');
    expect(output).toContain('Workflow');
  });

  it('renders overview tab by default', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <SessionDetail client={client} taskId="task-001" onBack={vi.fn()} onSelectTurn={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('task-001');
  });
});
