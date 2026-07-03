// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { dedupSession, mergeTurns, mergeToolCalls, mergeSkillEvents } from '../src/lib/ingest/merge.ts';
import type { TurnRow, ToolCallRow, SkillEventRow } from '../src/lib/ingest/turn-split.ts';

function makeTurn(turnIndex: number, role: string, sessionId: string = 's1'): TurnRow {
  return {
    id: `turn-${turnIndex}-${role}`,
    sessionId,
    turnIndex,
    role,
    content: `${role} at ${turnIndex}`,
    contentJson: null,
    contentSummary: `${role} at ${turnIndex}`,
    inputMessagesJson: null,
    inputMessagesCount: 0,
    inputMessagesTokens: 0,
    contextWindowPct: null,
    agentName: null,
    subagentName: null,
    subagentSessionId: null,
    totalTokens: 0,
    inputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    createdAt_ts: '2026-01-01T00:00:00.000Z',
    completedAt: null,
    latencyMs: 0,
    ttftMs: null,
    model: null,
    modelId: null,
    providerId: null,
    temperature: null,
    maxTokens: null,
    finishReason: null,
    isSubagent: false,
    parentExecutionId: null,
  };
}

function makeToolCall(toolCallId: string, turnId: string, toolName: string): ToolCallRow {
  return {
    id: `tc-${toolCallId}`,
    turnId,
    toolCallId,
    toolName,
    argsJson: null,
    resultJson: null,
    state: 'ok',
    errorType: null,
    errorMessage: null,
    startedAt: null,
    completedAt: null,
    durationMs: 0,
    dispatchBridgeId: null,
    isSkillRelated: false,
  };
}

function makeSkillEvent(turnId: string, skillName: string, eventType: string): SkillEventRow {
  return {
    id: `se-${turnId}-${skillName}-${eventType}`,
    turnId,
    skillName,
    skillVersion: null,
    eventType,
    success: true,
    errorMessage: null,
    argsJson: null,
    startedAt: null,
    completedAt: null,
    durationMs: 0,
  };
}

describe('merge', () => {
  describe('dedupSession', () => {
    it('returns shouldImport=true when no existing session', () => {
      const result = dedupSession(null, 'task-001');
      expect(result.shouldImport).toBe(true);
      expect(result.existingSessionId).toBeNull();
    });

    it('returns shouldImport=false when existing session exists (same taskId)', () => {
      const result = dedupSession('existing-ses-id', 'task-001');
      expect(result.shouldImport).toBe(false);
      expect(result.existingSessionId).toBe('existing-ses-id');
    });

    it('skips second import for same taskId', () => {
      const first = dedupSession(null, 'task-001');
      expect(first.shouldImport).toBe(true);
      const second = dedupSession('session-created-id', 'task-001');
      expect(second.shouldImport).toBe(false);
      expect(second.existingSessionId).toBe('session-created-id');
    });
  });

  describe('mergeTurns', () => {
    it('empty existing data: all new data imported', () => {
      const newTurns = [makeTurn(0, 'user'), makeTurn(1, 'assistant'), makeTurn(2, 'user')];
      const result = mergeTurns([], newTurns);
      expect(result.length).toBe(3);
      expect(result.map(t => t.turnIndex)).toEqual([0, 1, 2]);
    });

    it('identical data: no new rows added', () => {
      const turns = [makeTurn(0, 'user'), makeTurn(1, 'assistant')];
      const result = mergeTurns(turns, turns);
      expect(result.length).toBe(2);
    });

    it('existing 5 turns, new adds 3 more: result has 8 turns, no duplicates', () => {
      const existing = [
        makeTurn(0, 'user'),
        makeTurn(1, 'assistant'),
        makeTurn(2, 'user'),
        makeTurn(3, 'assistant'),
        makeTurn(4, 'user'),
      ];
      const newTurns = [
        makeTurn(4, 'user'),
        makeTurn(5, 'assistant'),
        makeTurn(6, 'user'),
      ];
      const result = mergeTurns(existing, newTurns);
      expect(result.length).toBe(7);
      const indices = result.map(t => t.turnIndex);
      expect(indices).toContain(0);
      expect(indices).toContain(4);
      expect(indices).toContain(5);
      expect(indices).toContain(6);
    });

    it('deduplicates by turnIndex+role combination', () => {
      const existing = [makeTurn(0, 'user'), makeTurn(1, 'assistant')];
      const newTurns = [makeTurn(0, 'user'), makeTurn(1, 'assistant'), makeTurn(2, 'user')];
      const result = mergeTurns(existing, newTurns);
      expect(result.length).toBe(3);
      const same0User = result.filter(t => t.turnIndex === 0 && t.role === 'user');
      expect(same0User.length).toBe(1);
    });

    it('allows different roles at same turnIndex', () => {
      const existing = [makeTurn(0, 'user')];
      const newTurns = [makeTurn(0, 'system')];
      const result = mergeTurns(existing, newTurns);
      expect(result.length).toBe(2);
    });

    it('result is sorted by turnIndex then role', () => {
      const existing = [makeTurn(1, 'assistant')];
      const newTurns = [makeTurn(0, 'user'), makeTurn(2, 'user')];
      const result = mergeTurns(existing, newTurns);
      expect(result[0].turnIndex).toBe(0);
      expect(result[1].turnIndex).toBe(1);
      expect(result[2].turnIndex).toBe(2);
    });
  });

  describe('mergeToolCalls', () => {
    it('empty existing: all incoming tool calls imported', () => {
      const incoming = [
        makeToolCall('tc1', 't1', 'bash'),
        makeToolCall('tc2', 't1', 'read'),
      ];
      const result = mergeToolCalls([], incoming);
      expect(result.length).toBe(2);
    });

    it('identical data: no duplicates', () => {
      const calls = [makeToolCall('tc1', 't1', 'bash')];
      const result = mergeToolCalls(calls, calls);
      expect(result.length).toBe(1);
    });

    it('deduplicates by toolCallId', () => {
      const existing = [makeToolCall('tc1', 't1', 'bash')];
      const incoming = [makeToolCall('tc1', 't1', 'bash'), makeToolCall('tc2', 't1', 'read')];
      const result = mergeToolCalls(existing, incoming);
      expect(result.length).toBe(2);
      const ids = result.map(tc => tc.toolCallId);
      expect(ids).toContain('tc1');
      expect(ids).toContain('tc2');
      expect(ids.filter(id => id === 'tc1').length).toBe(1);
    });
  });

  describe('mergeSkillEvents', () => {
    it('empty existing: all incoming skill events imported', () => {
      const incoming = [
        makeSkillEvent('t1', 'my-skill', 'load'),
        makeSkillEvent('t1', 'my-skill', 'invoke'),
      ];
      const result = mergeSkillEvents([], incoming);
      expect(result.length).toBe(2);
    });

    it('identical data: no duplicates', () => {
      const events = [makeSkillEvent('t1', 'my-skill', 'load')];
      const result = mergeSkillEvents(events, events);
      expect(result.length).toBe(1);
    });

    it('deduplicates by turnId+skillName+eventType', () => {
      const existing = [makeSkillEvent('t1', 'my-skill', 'load')];
      const incoming = [
        makeSkillEvent('t1', 'my-skill', 'load'),
        makeSkillEvent('t2', 'my-skill', 'load'),
      ];
      const result = mergeSkillEvents(existing, incoming);
      expect(result.length).toBe(2);
    });
  });
});
