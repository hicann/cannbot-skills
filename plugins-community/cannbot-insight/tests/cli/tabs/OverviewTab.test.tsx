// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { OverviewTab } from '@/cli/tui/tabs/OverviewTab';
import type { ApiSessionDetailResponse, ApiTurnItem } from '@/cli/types';

const mockSession: ApiSessionDetailResponse = {
  sessionId: 's1',
  taskId: 'task-001',
  label: 'Test Session',
  query: 'Fix bug in auth module',
  framework: null,
  startTime: '2026-06-14T10:00:00Z',
  endTime: '2026-06-14T10:05:00Z',
  totalTokens: 5000,
  totalInputTokens: 3000,
  totalOutputTokens: 1500,
  totalReasoningTokens: 500,
  totalCacheReadTokens: 200,
  totalCacheWriteTokens: 100,
  totalCost: 0.05,
  totalLatencyMs: 300000,
  totalToolCallCount: 10,
  totalLlmCallCount: 5,
  totalSkillLoadCount: 3,
  totalSubagentCount: 2,
  model: 'gpt-4o',
  user: 'testuser',
  sourcePath: null,
  agents: [
    {
      executionId: 'e1',
      agentName: 'main',
      isSubagent: false,
      parentExecutionId: null,
      tokens: 5000,
      cost: 0.05,
      toolCallCount: 10,
      skillLoadCount: 3,
      model: 'gpt-4o',
      createdAt: '2026-06-14T10:00:00Z',
      latencyMs: 300000,
    },
    {
      executionId: 'e2',
      agentName: 'sub1',
      isSubagent: true,
      parentExecutionId: 'e1',
      tokens: 2000,
      cost: 0.02,
      toolCallCount: 5,
      skillLoadCount: 1,
      model: 'gpt-4o',
      createdAt: '2026-06-14T10:01:00Z',
      latencyMs: 120000,
    },
  ],
  skills: [
    { skillName: 'agent-debug', version: '0.4', invocationCount: 5 },
    { skillName: 'find-skills', version: '1.0', invocationCount: 2 },
  ],
};

const mockTurns: ApiTurnItem[] = [];

describe('OverviewTab', () => {
  it('renders session info', () => {
    const { getPlainText } = renderTui(<OverviewTab session={mockSession} turns={mockTurns} />);
    const output = getPlainText();
    expect(output).toContain('task-001');
    expect(output).toContain('Fix bug in auth module');
    expect(output).toContain('gpt-4o');
    expect(output).toContain('testuser');
  });

  it('renders metric cards', () => {
    const { getPlainText } = renderTui(<OverviewTab session={mockSession} turns={mockTurns} />);
    const output = getPlainText();
    expect(output).toContain('Tokens');
    expect(output).toContain('Cost');
    expect(output).toContain('Duration');
  });

  it('renders token breakdown', () => {
    const { getPlainText } = renderTui(<OverviewTab session={mockSession} turns={mockTurns} />);
    const output = getPlainText();
    expect(output).toContain('Token Breakdown');
    expect(output).toContain('Input');
    expect(output).toContain('Output');
  });

  it('renders agents list', () => {
    const { getPlainText } = renderTui(<OverviewTab session={mockSession} turns={mockTurns} />);
    const output = getPlainText();
    expect(output).toContain('Agents');
    expect(output).toContain('main');
    expect(output).toContain('sub1');
  });

  it('renders skills list', () => {
    const { getPlainText } = renderTui(<OverviewTab session={mockSession} turns={mockTurns} />);
    const output = getPlainText();
    expect(output).toContain('Skills');
    expect(output).toContain('agent-debug');
    expect(output).toContain('find-skills');
  });
});
