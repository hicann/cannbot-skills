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
import { SubagentsTab } from '@/cli/tui/tabs/SubagentsTab';
import type { ApiExecutionItem, ApiBridgeItem } from '@/cli/types';

const mockSubagents: ApiExecutionItem[] = [
  {
    executionId: 'e1', sessionId: 's1', agentName: 'sub1', agentSessionId: 'as1',
    isSubagent: true, subagentType: 'coding', subagentName: 'coder',
    parentExecutionId: 'e0', rootExecutionId: 'e0', depth: 1,
    tokens: 2000, inputTokens: 1000, outputTokens: 800, reasoningTokens: 200,
    cost: 0.02, latencyMs: 120000, toolCallCount: 5, toolCallErrorCount: 0,
    llmCallCount: 3, skillLoadCount: 1, skillInvokeCount: 2,
    model: 'gpt-4o', createdAt: '2026-06-14T10:01:00Z',
    skills: [],
  },
];

const mockBridges: ApiBridgeItem[] = [];

describe('SubagentsTab', () => {
  it('renders subagents count header', () => {
    const { getPlainText } = renderTui(<SubagentsTab subagents={mockSubagents} bridges={mockBridges} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('1 subagents');
  });

  it('renders subagent data rows', () => {
    const { getPlainText } = renderTui(<SubagentsTab subagents={mockSubagents} bridges={mockBridges} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('sub1');
  });

  it('renders empty subagents', () => {
    const { getPlainText } = renderTui(<SubagentsTab subagents={[]} bridges={mockBridges} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('No subagents');
  });
});
