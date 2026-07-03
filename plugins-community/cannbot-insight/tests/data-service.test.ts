// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { computeSessionAggregates } from '../src/lib/ingest/data-service.ts';
import type { TurnRow } from '../src/lib/ingest/turn-split.ts';
import type { ToolCallRow, SkillEventRow } from '../src/lib/ingest/turn-split.ts';

function makeTurn(overrides: Partial<TurnRow> & { role: string; turnIndex: number }): TurnRow {
  return {
    id: `turn-${overrides.turnIndex}`,
    sessionId: 'session-1',
    turnIndex: overrides.turnIndex,
    role: overrides.role,
    content: null,
    contentJson: null,
    contentSummary: null,
    inputMessagesJson: null,
    inputMessagesCount: 0,
    inputMessagesTokens: 0,
    contextWindowPct: null,
    agentName: null,
    subagentName: null,
    subagentSessionId: overrides.subagentSessionId ?? null,
    subagentType: overrides.subagentType ?? null,
    totalTokens: overrides.totalTokens ?? 0,
    inputTokens: overrides.inputTokens ?? 0,
    outputTokens: overrides.outputTokens ?? 0,
    reasoningTokens: overrides.reasoningTokens ?? 0,
    cacheReadTokens: overrides.cacheReadTokens ?? 0,
    cacheWriteTokens: overrides.cacheWriteTokens ?? 0,
    cost: overrides.cost ?? 0,
    createdAt_ts: overrides.createdAt_ts ?? '2026-01-01T00:00:00.000Z',
    completedAt: overrides.completedAt ?? null,
    latencyMs: overrides.latencyMs ?? 0,
    ttftMs: null,
    model: overrides.model ?? null,
    modelId: null,
    providerId: null,
    temperature: null,
    maxTokens: null,
    finishReason: null,
    isSubagent: overrides.isSubagent ?? false,
    parentExecutionId: null,
  };
}

const emptyToolCalls: ToolCallRow[] = [];
const emptySkillEvents: SkillEventRow[] = [];

describe('computeSessionAggregates', () => {
  it('accumulates cost from turn.cost directly for assistant turns', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 500, cost: 0.015 }),
      makeTurn({ turnIndex: 2, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 3, role: 'assistant', totalTokens: 800, cost: 0.025 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBeCloseTo(0.04, 10);
  });

  it('skips cost for user turns', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100, cost: 0.01 }),
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 500, cost: 0.02 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBeCloseTo(0.02, 10);
  });

  it('skips cost for assistant turns with totalTokens=0', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 0, cost: 0.01 }),
      makeTurn({ turnIndex: 2, role: 'assistant', totalTokens: 500, cost: 0.02 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBeCloseTo(0.02, 10);
  });

  it('includes subagent turn costs', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 500, cost: 0.01 }),
      makeTurn({ turnIndex: 2, role: 'user', totalTokens: 50 }),
      makeTurn({ turnIndex: 3, role: 'assistant', totalTokens: 300, cost: 0.008, isSubagent: true, subagentSessionId: 'sub-1' }),
      makeTurn({ turnIndex: 4, role: 'assistant', totalTokens: 200, cost: 0.005, isSubagent: true, subagentSessionId: 'sub-2' }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBeCloseTo(0.01 + 0.008 + 0.005, 10);
    expect(result.totalSubagentCount).toBe(2);
  });

  it('does not double-count cacheRead tokens in cost', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 10000, inputTokens: 8000, outputTokens: 2000, cacheReadTokens: 5000, cost: 0.023 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBeCloseTo(0.023, 10);
    expect(result.totalTokens).toBe(10000);
    expect(result.totalCacheReadTokens).toBe(5000);
  });

  it('returns zero cost when all turns are user turns', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 1, role: 'user', totalTokens: 200 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalCost).toBe(0);
    expect(result.totalLlmCallCount).toBe(0);
  });

  it('counts totalLlmCallCount correctly', () => {
    const turns: TurnRow[] = [
      makeTurn({ turnIndex: 0, role: 'user', totalTokens: 100 }),
      makeTurn({ turnIndex: 1, role: 'assistant', totalTokens: 500, cost: 0.01 }),
      makeTurn({ turnIndex: 2, role: 'assistant', totalTokens: 0, cost: 0.005 }),
      makeTurn({ turnIndex: 3, role: 'assistant', totalTokens: 300, cost: 0.008 }),
    ];
    const result = computeSessionAggregates(turns, emptyToolCalls, emptySkillEvents);
    expect(result.totalLlmCallCount).toBe(2);
  });
});
