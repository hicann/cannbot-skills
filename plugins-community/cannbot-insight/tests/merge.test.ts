// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { dedupSession, mergeTurns, mergeToolCalls, mergeSkillEvents, diffTurns, diffToolCalls, rankedKeyMap, keyByRankedTurn } from '../src/lib/ingest/merge.ts';
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

    it('existing 5 turns, full re-parse of 7: result has 7 turns, no duplicates', () => {
      const existing = [
        makeTurn(0, 'user'),
        makeTurn(1, 'assistant'),
        makeTurn(2, 'user'),
        makeTurn(3, 'assistant'),
        makeTurn(4, 'user'),
      ];
      const newTurns = [
        makeTurn(0, 'user'),
        makeTurn(1, 'assistant'),
        makeTurn(2, 'user'),
        makeTurn(3, 'assistant'),
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

    it('deduplicates by context rank slot', () => {
      const existing = [makeTurn(0, 'user'), makeTurn(1, 'assistant')];
      const newTurns = [makeTurn(0, 'user'), makeTurn(1, 'assistant'), makeTurn(2, 'user')];
      const result = mergeTurns(existing, newTurns);
      expect(result.length).toBe(3);
      const same0User = result.filter(t => t.turnIndex === 0 && t.role === 'user');
      expect(same0User.length).toBe(1);
    });

    it('main 增长平移 subagent 全局下标时不产生重复（Bug #7）', () => {
      const sub = (i: number, role: string) => ({ ...makeTurn(i, role), subagentSessionId: 'sub-x', isSubagent: true });
      const existing = [
        makeTurn(0, 'user'),
        makeTurn(1, 'assistant'),
        makeTurn(2, 'user'),
        makeTurn(3, 'assistant'),
        sub(4, 'user'),
        sub(5, 'assistant'),
      ];
      const newTurns = [
        makeTurn(0, 'user'),
        makeTurn(1, 'assistant'),
        makeTurn(2, 'user'),
        makeTurn(3, 'assistant'),
        makeTurn(4, 'user'),
        sub(5, 'user'),
        sub(6, 'assistant'),
      ];
      const result = mergeTurns(existing, newTurns);
      expect(result.length).toBe(7);
      expect(result.filter(t => t.subagentSessionId === 'sub-x').length).toBe(2);
      expect(result.some(t => t.turnIndex === 4 && t.role === 'user' && !t.subagentSessionId)).toBe(true);
    });

    it('result is sorted by turnIndex then role', () => {
      const existing = [makeTurn(0, 'user'), makeTurn(1, 'assistant')];
      const newTurns = [makeTurn(0, 'user'), makeTurn(1, 'assistant'), makeTurn(2, 'user')];
      const result = mergeTurns(existing, newTurns);
      expect(result[0].turnIndex).toBe(0);
      expect(result[1].turnIndex).toBe(1);
      expect(result[2].turnIndex).toBe(2);
    });
  });

  describe('ranked keys (Bug #7)', () => {
    it('main 增长不改变 subagent 的键', () => {
      const sub = (i: number, role: string) => ({ ...makeTurn(i, role), subagentSessionId: 'sub-x' });
      const before = rankedKeyMap([
        makeTurn(0, 'user'), makeTurn(1, 'assistant'),
        sub(2, 'user'), sub(3, 'assistant'),
      ]);
      const after = rankedKeyMap([
        makeTurn(0, 'user'), makeTurn(1, 'assistant'), makeTurn(2, 'user'),
        sub(3, 'user'), sub(4, 'assistant'),
      ]);
      expect(after.get([...after.keys()].find(r => r.subagentSessionId === 'sub-x' && r.role === 'user')!)).toBe('sub-x#0');
      expect(after.get([...after.keys()].find(r => r.subagentSessionId === 'sub-x' && r.role === 'assistant')!)).toBe('sub-x#1');
      expect(before.get([...before.keys()].find(r => r.subagentSessionId === 'sub-x')!)).toBe('sub-x#0');
    });

    it('keyByRankedTurn 从 DB 全局下标还原上下文 rank', () => {
      const sub = (i: number, role: string) => ({ ...makeTurn(i, role), subagentSessionId: 'sub-x' });
      const dbRows = [
        makeTurn(0, 'user'), makeTurn(1, 'assistant'),
        sub(4, 'user'), sub(5, 'assistant'),
      ].slice().sort((a, b) => b.turnIndex - a.turnIndex);
      const byKey = keyByRankedTurn(dbRows);
      expect(byKey.get('main#0')!.turnIndex).toBe(0);
      expect(byKey.get('main#1')!.turnIndex).toBe(1);
      expect(byKey.get('sub-x#0')!.turnIndex).toBe(4);
      expect(byKey.get('sub-x#1')!.turnIndex).toBe(5);
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

  describe('diffTurns', () => {
    it('all new turns → all inserted, zero updated', () => {
      const incoming = [makeTurn(0, 'user'), makeTurn(1, 'assistant')];
      const { toInsert, toUpdate } = diffTurns([], incoming);
      expect(toInsert.length).toBe(2);
      expect(toUpdate.length).toBe(0);
    });

    it('identical turns → zero inserted, zero updated', () => {
      const t0 = makeTurn(0, 'user');
      const t1 = makeTurn(1, 'assistant');
      const { toInsert, toUpdate } = diffTurns([t0, t1], [t0, t1]);
      expect(toInsert.length).toBe(0);
      expect(toUpdate.length).toBe(0);
    });

    it('existing turn with changed content → 1 updated', () => {
      const old = makeTurn(1, 'assistant');
      const newTurn = { ...old, content: 'updated content', completedAt: '2026-01-02', latencyMs: 500 };
      const { toInsert, toUpdate } = diffTurns([old], [newTurn]);
      expect(toInsert.length).toBe(0);
      expect(toUpdate.length).toBe(1);
      expect(toUpdate[0].dbId).toBe(old.id);
      expect(toUpdate[0].data['content']).toBe('updated content');
      expect(toUpdate[0].data['completedAt']).toBe('2026-01-02');
      expect(toUpdate[0].data['latencyMs']).toBe(500);
    });

    it('mix of new + changed + unchanged', () => {
      const old0 = makeTurn(0, 'user');
      const old1 = makeTurn(1, 'assistant');
      const new1 = { ...old1, totalTokens: 999, completedAt: '2026-01-03' };
      const new2 = makeTurn(2, 'user');
      const { toInsert, toUpdate } = diffTurns([old0, old1], [old0, new1, new2]);
      expect(toInsert.length).toBe(1);
      expect(toInsert[0].turnIndex).toBe(2);
      expect(toUpdate.length).toBe(1);
      expect(toUpdate[0].data['totalTokens']).toBe(999);
    });

    it('null→null changes are not included in update', () => {
      const old = makeTurn(1, 'assistant');
      const newTurn = { ...old };
      const { toUpdate } = diffTurns([old], [newTurn]);
      expect(toUpdate.length).toBe(0);
    });

    it('main 增长平移 subagent 下标时：subagent 匹配旧行、新 main turn 插入（Bug #7）', () => {
      const sub = (i: number, role: string) => ({
        ...makeTurn(i, role),
        subagentSessionId: 'sub-x',
        isSubagent: true,
        content: `${role} sub`,
        contentSummary: `${role} sub`,
      });
      const existing = [
        makeTurn(0, 'user'), makeTurn(1, 'assistant'),
        makeTurn(2, 'user'), makeTurn(3, 'assistant'),
        sub(4, 'user'), sub(5, 'assistant'),
      ];
      const incoming = [
        makeTurn(0, 'user'), makeTurn(1, 'assistant'),
        makeTurn(2, 'user'), makeTurn(3, 'assistant'),
        makeTurn(4, 'user'),
        sub(5, 'user'), sub(6, 'assistant'),
      ];
      const { toInsert, toUpdate } = diffTurns(existing, incoming);
      expect(toInsert.length).toBe(1);
      expect(toInsert[0].turnIndex).toBe(4);
      expect(toInsert[0].subagentSessionId).toBeNull();
      expect(toUpdate.length).toBe(0);
    });
  });

  describe('diffToolCalls', () => {
    it('all new tool calls → all inserted, zero updated', () => {
      const existing = new Map<string, ToolCallRow>();
      const incoming = [makeToolCall('tc1', 't1', 'bash'), makeToolCall('tc2', 't1', 'read')];
      const { toInsert, toUpdate } = diffToolCalls(existing, incoming);
      expect(toInsert.length).toBe(2);
      expect(toUpdate.length).toBe(0);
    });

    it('identical tool calls → zero inserted, zero updated', () => {
      const tc = makeToolCall('tc1', 't1', 'bash');
      const existing = new Map<string, ToolCallRow>([['tc1', tc]]);
      const { toInsert, toUpdate } = diffToolCalls(existing, [tc]);
      expect(toInsert.length).toBe(0);
      expect(toUpdate.length).toBe(0);
    });

    it('tool call with completed result → 1 updated', () => {
      const old = makeToolCall('tc1', 't1', 'bash');
      const newTc = { ...old, resultJson: '{"stdout":"ok"}', state: 'completed', completedAt: '2026-01-02', durationMs: 1200 };
      const existing = new Map<string, ToolCallRow>([['tc1', old]]);
      const { toInsert, toUpdate } = diffToolCalls(existing, [newTc]);
      expect(toInsert.length).toBe(0);
      expect(toUpdate.length).toBe(1);
      expect(toUpdate[0].data['resultJson']).toBe('{"stdout":"ok"}');
      expect(toUpdate[0].data['state']).toBe('completed');
      expect(toUpdate[0].data['completedAt']).toBe('2026-01-02');
      expect(toUpdate[0].data['durationMs']).toBe(1200);
    });

    it('tool call with error → 1 updated', () => {
      const old = makeToolCall('tc2', 't1', 'read');
      const newTc = { ...old, state: 'error', errorType: 'NotFound', errorMessage: 'file not found' };
      const existing = new Map<string, ToolCallRow>([['tc2', old]]);
      const { toInsert, toUpdate } = diffToolCalls(existing, [newTc]);
      expect(toUpdate.length).toBe(1);
      expect(toUpdate[0].data['state']).toBe('error');
      expect(toUpdate[0].data['errorType']).toBe('NotFound');
    });
  });
});
