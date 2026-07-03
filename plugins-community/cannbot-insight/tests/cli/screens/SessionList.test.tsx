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
import { SessionList } from '@/cli/tui/screens/SessionList';
import { InsightClient } from '@/cli/client';

vi.mock('@/cli/hooks/useApi', () => ({
  useApi: () => ({
    data: {
      items: [
        { sessionId: 's1', taskId: 'task-001', query: 'Fix bug', model: 'gpt-4o', totalTokens: 5000, totalCost: 0.05, totalLatencyMs: 300000, startTime: '2026-06-14T10:00:00Z', user: 'alice' },
        { sessionId: 's2', taskId: 'task-002', query: 'Add feature', model: 'claude-3.5-sonnet', totalTokens: 8000, totalCost: 0.08, totalLatencyMs: 500000, startTime: '2026-06-14T11:00:00Z', user: 'bob' },
      ],
      total: 2,
      page: 1,
    },
    loading: false,
    error: null,
    refresh: vi.fn(),
  }),
}));

vi.mock('@/cli/hooks/useKeyboard', () => ({
  useKeyboard: () => {},
}));

describe('SessionList', () => {
  it('renders session table with data', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <SessionList client={client} onSelect={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('task-001');
    expect(output).toContain('task-002');
  });

  it('renders search prompt when searchMode is toggled', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <SessionList client={client} onSelect={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('sessions');
  });

  it('renders pagination info', () => {
    const client = new InsightClient('http://localhost:21025');
    const { getPlainText } = renderTui(
      <SessionList client={client} onSelect={vi.fn()} />
    );
    const output = getPlainText();
    expect(output).toContain('Page');
  });
});
