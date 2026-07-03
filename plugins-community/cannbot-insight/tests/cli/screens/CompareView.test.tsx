// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { CompareView } from '@/cli/tui/screens/CompareView';
import { InsightClient } from '@/cli/client';

vi.mock('@/cli/hooks/useApi', () => ({
  useApi: () => ({
    data: {
      sessionId: 's1', taskId: 'task-001', label: 'Test', query: 'Fix bug', framework: null,
      startTime: '2026-06-14T10:00:00Z', endTime: '2026-06-14T10:05:00Z',
      totalTokens: 5000, totalInputTokens: 3000, totalOutputTokens: 1500, totalReasoningTokens: 500,
      totalCacheReadTokens: 200, totalCacheWriteTokens: 100, totalCost: 0.05, totalLatencyMs: 300000,
      totalToolCallCount: 10, totalLlmCallCount: 5, totalSkillLoadCount: 3, totalSubagentCount: 2,
      model: 'gpt-4o', user: 'alice', sourcePath: null, agents: [], skills: [],
    },
    loading: false, refresh: vi.fn(),
  }),
}));

vi.mock('@/cli/hooks/useKeyboard', () => ({
  useKeyboard: () => {},
}));

describe('CompareView', () => {
  it('renders compare header', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <CompareView client={client} taskId1="task-001" taskId2="task-002" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Compare');
    expect(output).toContain('task-001');
    expect(output).toContain('task-002');
  });

  it('renders comparison table', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <CompareView client={client} taskId1="task-001" taskId2="task-002" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Metric');
    expect(output).toContain('Tokens');
    expect(output).toContain('Diff');
  });
});
