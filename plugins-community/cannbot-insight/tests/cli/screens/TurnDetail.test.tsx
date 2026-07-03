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
import { TurnDetail } from '@/cli/tui/screens/TurnDetail';
import { InsightClient } from '@/cli/client';

vi.mock('@/cli/hooks/useApi', () => ({
  useApi: () => ({
    data: {
      turnId: 't1', sessionId: 's1', turnIndex: 1, role: 'assistant',
      content: 'Hello, I can help you fix that bug.', contentJson: null, contentSummary: 'Response',
      inputMessagesJson: null, inputMessagesCount: 3, inputMessagesTokens: 100,
      contextWindowPct: 50, agentName: 'main', subagentName: null, subagentSessionId: null,
      isSubagent: false, totalTokens: 200, inputTokens: 100, outputTokens: 80,
      reasoningTokens: 20, cacheReadTokens: 10, cacheWriteTokens: 5,
      latencyMs: 1000, ttftMs: 200, createdAt: '2026-06-14T10:00:00Z', completedAt: '2026-06-14T10:00:01Z',
      model: 'gpt-4o', modelId: null, providerId: null, finishReason: 'stop',
      toolCalls: [], skillEvents: [],
    },
    loading: false, error: null, refresh: vi.fn(),
  }),
}));

vi.mock('@/cli/hooks/useKeyboard', () => ({
  useKeyboard: () => {},
}));

describe('TurnDetail', () => {
  it('renders turn header', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <TurnDetail client={client} turnId="t1" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Turn #1');
    expect(output).toContain('assistant');
  });

  it('renders metric cards', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <TurnDetail client={client} turnId="t1" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Tokens');
    expect(output).toContain('Cost');
  });

  it('renders content section', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <TurnDetail client={client} turnId="t1" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Content');
    expect(output).toContain('Hello');
  });

  it('renders context bar', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <TurnDetail client={client} turnId="t1" onBack={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Context');
  });
});
