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
import { TurnsTab } from '@/cli/tui/tabs/TurnsTab';
import type { ApiTurnItem } from '@/cli/types';

const mockTurns: ApiTurnItem[] = [
  {
    turnId: 't1', turnIndex: 0, role: 'user', contentSummary: 'Hello',
    agentName: 'main', isSubagent: false, subagentName: null, subagentSessionId: null,
    parentExecutionId: null, totalTokens: 100, inputTokens: 50, outputTokens: 50,
    reasoningTokens: 0, cacheReadTokens: 0, cacheWriteTokens: 0,
    inputMessagesCount: 1, inputMessagesTokens: 50, contextWindowPct: 10,
    latencyMs: 500, createdAt: '2026-06-14T10:00:00Z', completedAt: '2026-06-14T10:00:01Z',
    model: 'gpt-4o', finishReason: 'stop', toolCalls: [], skillEvents: [],
  },
  {
    turnId: 't2', turnIndex: 1, role: 'assistant', contentSummary: 'Response',
    agentName: 'main', isSubagent: false, subagentName: null, subagentSessionId: null,
    parentExecutionId: null, totalTokens: 200, inputTokens: 100, outputTokens: 80,
    reasoningTokens: 20, cacheReadTokens: 10, cacheWriteTokens: 5,
    inputMessagesCount: 2, inputMessagesTokens: 100, contextWindowPct: 80,
    latencyMs: 1000, createdAt: '2026-06-14T10:01:00Z', completedAt: '2026-06-14T10:01:02Z',
    model: 'gpt-4o', finishReason: 'stop', toolCalls: [], skillEvents: [],
  },
];

describe('TurnsTab', () => {
  it('renders turns count header', () => {
    const { getPlainText } = renderTui(<TurnsTab turns={mockTurns} selectedIndex={0} turnDetail={null} />);
    const output = getPlainText();
    expect(output).toContain('Turns (2)');
  });

  it('renders turn data rows', () => {
    const { getPlainText } = renderTui(<TurnsTab turns={mockTurns} selectedIndex={0} turnDetail={null} />);
    const output = getPlainText();
    expect(output).toContain('user');
    expect(output).toContain('assistant');
  });

  it('renders empty turns', () => {
    const { getPlainText } = renderTui(<TurnsTab turns={[]} selectedIndex={0} turnDetail={null} />);
    const output = getPlainText();
    expect(output).toContain('Turns (0)');
  });

  it('shows turn detail when provided', () => {
    const { getPlainText } = renderTui(<TurnsTab turns={mockTurns} selectedIndex={0} turnDetail={{
      turnId: 't2', sessionId: 's1', turnIndex: 1, role: 'assistant',
      content: 'Full response text', contentJson: null, contentSummary: 'Response',
      inputMessagesJson: null, inputMessagesCount: 2, inputMessagesTokens: 100,
      contextWindowPct: 80, agentName: 'main', subagentName: null,
      subagentSessionId: null, isSubagent: false,
      totalTokens: 200, inputTokens: 100, outputTokens: 80, reasoningTokens: 20,
      cacheReadTokens: 10, cacheWriteTokens: 5,
      latencyMs: 1000, ttftMs: 200,
      createdAt: '2026-06-14T10:01:00Z', completedAt: '2026-06-14T10:01:02Z',
      model: 'gpt-4o', modelId: null, providerId: null, finishReason: 'stop',
      toolCalls: [], skillEvents: [],
    }} />);
    const output = getPlainText();
    expect(output).toContain('#1');
    expect(output).toContain('assistant');
  });
});
