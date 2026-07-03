// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll } from 'vitest';
import { splitExecutions, resetIdCounter } from '../src/lib/ingest/execution-split.ts';
import { splitIntoTurns, resetIdCounter as resetTurnIdCounter } from '../src/lib/ingest/turn-split.ts';
import { normalize } from '../src/lib/ingest/normalize.ts';
import { readSession, listSessions } from '../src/lib/ingest/adapters/opencode-db.ts';
import { buildBridges, resetIdCounter as resetBridgeIdCounter } from '../src/lib/ingest/bridge-builder.ts';
import type { RawInteraction } from '../src/lib/shared/types.ts';
import type { TurnRow, ToolCallRow, SkillEventRow } from '../src/lib/ingest/turn-split.ts';
import type { ExecutionRow } from '../src/lib/ingest/execution-split.ts';
import type { InteractionBridgeRow } from '../src/lib/ingest/bridge-builder.ts';
import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';

const SYNTHETIC_DATA_PATH = path.resolve(__dirname, 'data/synthetic-opencode.json');
const REAL_DB_PATH = path.resolve(__dirname, 'data/opencode-sessions.db');
const hasRealDB = fs.existsSync(REAL_DB_PATH);

function loadSyntheticData(): RawInteraction[] {
  const raw = fs.readFileSync(SYNTHETIC_DATA_PATH, 'utf-8');
  return JSON.parse(raw) as RawInteraction[];
}

describe('execution-split', () => {
  describe('with synthetic data (has root + subagent)', () => {
    let turns: TurnRow[];
    let toolCalls: ToolCallRow[];
    let skillEvents: SkillEventRow[];
    let executions: ExecutionRow[];

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = loadSyntheticData();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-synthetic-001');
      turns = splitResult.turns;
      toolCalls = splitResult.toolCalls;
      skillEvents = splitResult.skillEvents;
      executions = splitExecutions(turns, toolCalls, skillEvents, 'session-synthetic-001');
    });

    it('produces 2 executions: 1 root + 1 subagent', () => {
      expect(executions.length).toBe(2);
      const rootExecs = executions.filter(e => !e.isSubagent);
      const subExecs = executions.filter(e => e.isSubagent);
      expect(rootExecs.length).toBe(1);
      expect(subExecs.length).toBe(1);
    });

    it('each execution has all required ExecutionRow fields', () => {
      const requiredKeys: Array<keyof ExecutionRow> = [
        'id', 'sessionId', 'agentName', 'agentSessionId', 'isSubagent',
        'subagentType', 'subagentName', 'parentExecutionId', 'rootExecutionId',
        'depth', 'tokens', 'inputTokens', 'outputTokens', 'reasoningTokens',
        'cacheReadInputTokens', 'cacheCreationInputTokens', 'maxSingleCallTokens',
        'cost', 'latencyMs', 'createdAt', 'toolCallCount', 'toolCallErrorCount',
        'llmCallCount', 'skillLoadCount', 'skillInvokeCount',
        'finalResult', 'model',
      ];
      for (const exec of executions) {
        for (const key of requiredKeys) {
          expect(exec).toHaveProperty(key);
        }
      }
    });

    it('root execution aggregates non-subagent turns', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      expect(rootExec.tokens).toBe(rootTurns.reduce((s, t) => s + t.totalTokens, 0));
      expect(rootExec.inputTokens).toBe(rootTurns.reduce((s, t) => s + t.inputTokens, 0));
      expect(rootExec.outputTokens).toBe(rootTurns.reduce((s, t) => s + t.outputTokens, 0));
      expect(rootExec.isSubagent).toBe(false);
      expect(rootExec.parentExecutionId).toBeNull();
      expect(rootExec.rootExecutionId).toBe(rootExec.id);
      expect(rootExec.depth).toBe(0);
      expect(rootExec.agentSessionId).toBe('session-synthetic-001');
    });

    it('subagent execution aggregates subagent turns', () => {
      const subExec = executions.find(e => e.isSubagent)!;
      const subTurns = turns.filter(t => t.isSubagent);
      expect(subExec.tokens).toBe(subTurns.reduce((s, t) => s + t.totalTokens, 0));
      expect(subExec.inputTokens).toBe(subTurns.reduce((s, t) => s + t.inputTokens, 0));
      expect(subExec.outputTokens).toBe(subTurns.reduce((s, t) => s + t.outputTokens, 0));
      expect(subExec.isSubagent).toBe(true);
      expect(subExec.parentExecutionId).toBe(executions.find(e => !e.isSubagent)!.id);
      expect(subExec.rootExecutionId).toBe(executions.find(e => !e.isSubagent)!.id);
      expect(subExec.depth).toBe(1);
      expect(subExec.agentSessionId).toBe('ses_synthetic_subagent_001');
    });

    it('root execution llmCallCount equals number of root assistant turns', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootAssistantTurns = turns.filter(t => !t.isSubagent && t.role === 'assistant');
      expect(rootExec.llmCallCount).toBe(rootAssistantTurns.length);
    });

    it('root execution toolCallCount equals root tool calls', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurnIds = new Set(turns.filter(t => !t.isSubagent).map(t => t.id));
      const rootToolCalls = toolCalls.filter(tc => rootTurnIds.has(tc.turnId));
      expect(rootExec.toolCallCount).toBe(rootToolCalls.length);
    });

    it('root execution skillLoadCount and skillInvokeCount', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurnIds = new Set(turns.filter(t => !t.isSubagent).map(t => t.id));
      const rootSkillEvents = skillEvents.filter(se => rootTurnIds.has(se.turnId));
      expect(rootExec.skillLoadCount).toBe(rootSkillEvents.filter(se => se.eventType === 'load').length);
      expect(rootExec.skillInvokeCount).toBe(rootSkillEvents.filter(se => se.eventType === 'invoke' || se.eventType === 'use').length);
    });

    it('subagent execution toolCallCount equals subagent tool calls', () => {
      const subExec = executions.find(e => e.isSubagent)!;
      const subTurnIds = new Set(turns.filter(t => t.isSubagent).map(t => t.id));
      const subToolCalls = toolCalls.filter(tc => subTurnIds.has(tc.turnId));
      expect(subExec.toolCallCount).toBe(subToolCalls.length);
    });

    it('maxSingleCallTokens is max of turn totalTokens', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      const expectedMax = Math.max(...rootTurns.map(t => t.totalTokens));
      expect(rootExec.maxSingleCallTokens).toBe(expectedMax);
    });

    it('cost is sum of turn costs', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      const expectedCost = rootTurns.reduce((s, t) => s + t.cost, 0);
      expect(rootExec.cost).toBeCloseTo(expectedCost, 4);
    });

    it('latencyMs is wall-clock duration from first turn start to last turn end', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      const timestamps = rootTurns.filter(t => t.createdAt_ts).map(t => new Date(t.createdAt_ts!).getTime());
      const endTimestamps = rootTurns.filter(t => t.completedAt).map(t => new Date(t.completedAt!).getTime());
      const earliestStart = Math.min(...timestamps);
      const latestEnd = endTimestamps.length > 0 ? Math.max(...endTimestamps) : Math.max(...timestamps) + rootTurns[rootTurns.length - 1].latencyMs;
      expect(rootExec.latencyMs).toBe(latestEnd - earliestStart);
    });

    it('cacheReadInputTokens from cacheReadTokens', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      const expected = rootTurns.reduce((s, t) => s + t.cacheReadTokens, 0);
      expect(rootExec.cacheReadInputTokens).toBe(expected);
    });

    it('cacheCreationInputTokens from cacheWriteTokens', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      const expected = rootTurns.reduce((s, t) => s + t.cacheWriteTokens, 0);
      expect(rootExec.cacheCreationInputTokens).toBe(expected);
    });

    it('finalResult is last root turn contentSummary', () => {
      const rootExec = executions.find(e => !e.isSubagent)!;
      const lastRootTurn = turns.filter(t => !t.isSubagent)[turns.filter(t => !t.isSubagent).length - 1];
      expect(rootExec.finalResult).toBe(lastRootTurn?.contentSummary ?? null);
    });

    it('subagentName extracted from subagent turns', () => {
      const subExec = executions.find(e => e.isSubagent)!;
      expect(subExec.subagentName).toBe('Kuafu');
    });
  });

  describe('session with no subagents produces just 1 root ExecutionRow', () => {
    it('only root execution when no subagent turns', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'user',
          content: 'Hello',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1735689600000 },
          agent: 'build', subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: null, model: 'test', modelID: null, providerID: null, latency: null, finish_reason: null,
        },
        {
          role: 'assistant',
          content: 'Response',
          timestamp: '2026-01-01T00:00:01.000Z',
          timeInfo: { created: 1735689601000, completed: 1735689605000 },
          agent: 'build', subagent_name: null, subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc1', toolName: 'bash', argsJson: '{"command":"ls"}',
            resultJson: 'output', state: 'ok',
          }],
          usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.005, inputMessagesTokens: 400 },
          model: 'test', modelID: null, providerID: null, latency: 4000, finish_reason: 'stop',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-no-sub');
      const executions = splitExecutions(splitResult.turns, splitResult.toolCalls, splitResult.skillEvents, 'session-no-sub');
      expect(executions.length).toBe(1);
      expect(executions[0].isSubagent).toBe(false);
      expect(executions[0].parentExecutionId).toBeNull();
      expect(executions[0].tokens).toBe(500);
      expect(executions[0].llmCallCount).toBe(1);
      expect(executions[0].toolCallCount).toBe(1);
    });
  });

  describe.skipIf(!hasRealDB)('with real DB data', () => {
    let executions: ExecutionRow[];
    let sessionId: string;

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const db = new Database(REAL_DB_PATH, { readonly: true });
      const subagentSessions = db.prepare('SELECT id, parent_id FROM session WHERE parent_id IS NOT NULL').all() as { id: string; parent_id: string }[];
      db.close();

      if (subagentSessions.length === 0) {
        sessionId = '';
        executions = [];
        return;
      }

      const rootSessionId = subagentSessions[0].parent_id;
      sessionId = rootSessionId;

      const rootInteractions = readSession(REAL_DB_PATH, rootSessionId);
      const allInteractions: RawInteraction[] = [...rootInteractions];

      for (const sub of subagentSessions.filter(s => s.parent_id === rootSessionId)) {
        const subInteractions = readSession(REAL_DB_PATH, sub.id);
        allInteractions.push(...subInteractions);
      }

      const normalized = normalize(allInteractions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, sessionId);
      executions = splitExecutions(splitResult.turns, splitResult.toolCalls, splitResult.skillEvents, sessionId);
    });

    it('produces root + subagent executions', () => {
      if (executions.length === 0) return;
      const rootExecs = executions.filter(e => !e.isSubagent);
      const subExecs = executions.filter(e => e.isSubagent);
      expect(rootExecs.length).toBe(1);
      expect(subExecs.length).toBeGreaterThan(0);
    });

    it('root execution has correct structure', () => {
      if (executions.length === 0) return;
      const rootExec = executions.find(e => !e.isSubagent)!;
      expect(rootExec.isSubagent).toBe(false);
      expect(rootExec.parentExecutionId).toBeNull();
      expect(rootExec.rootExecutionId).toBe(rootExec.id);
      expect(rootExec.depth).toBe(0);
      expect(rootExec.tokens).toBeGreaterThan(0);
    });

    it('subagent execution references root as parent', () => {
      if (executions.length === 0) return;
      const rootExec = executions.find(e => !e.isSubagent)!;
      const subExecs = executions.filter(e => e.isSubagent);
      for (const sub of subExecs) {
        expect(sub.parentExecutionId).toBe(rootExec.id);
        expect(sub.rootExecutionId).toBe(rootExec.id);
        expect(sub.depth).toBe(1);
      }
    });

    it('each execution has all required fields populated', () => {
      for (const exec of executions) {
        expect(typeof exec.id).toBe('string');
        expect(typeof exec.sessionId).toBe('string');
        expect(typeof exec.tokens).toBe('number');
        expect(typeof exec.toolCallCount).toBe('number');
        expect(typeof exec.llmCallCount).toBe('number');
      }
    });

    it('logs execution results for inspection', () => {
      console.log('\n=== Real DB Execution-Split Results ===');
      console.log(`Session: ${sessionId}`);
      console.log(`Total executions: ${executions.length}`);
      for (const exec of executions) {
        console.log(JSON.stringify({
          isSubagent: exec.isSubagent,
          agentName: exec.agentName,
          subagentName: exec.subagentName,
          agentSessionId: exec.agentSessionId,
          tokens: exec.tokens,
          inputTokens: exec.inputTokens,
          outputTokens: exec.outputTokens,
          cost: exec.cost,
          latencyMs: exec.latencyMs,
          toolCallCount: exec.toolCallCount,
          llmCallCount: exec.llmCallCount,
          skillLoadCount: exec.skillLoadCount,
          skillInvokeCount: exec.skillInvokeCount,
          finalResult: exec.finalResult?.substring(0, 80),
        }, null, 2));
      }
    });
  });

  describe('edge cases', () => {
    it('handles empty turns array', () => {
      resetIdCounter();
      const executions = splitExecutions([], [], [], 'session-empty');
      expect(executions.length).toBe(1);
      expect(executions[0].isSubagent).toBe(false);
      expect(executions[0].tokens).toBe(0);
      expect(executions[0].toolCallCount).toBe(0);
    });

    it('handles turns with only user messages', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'user',
          content: 'Hello',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1735689600000 },
          agent: 'build', subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: null, model: 'test', modelID: null, providerID: null, latency: null, finish_reason: null,
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-user-only');
      const executions = splitExecutions(splitResult.turns, splitResult.toolCalls, splitResult.skillEvents, 'session-user-only');
      expect(executions.length).toBe(1);
      expect(executions[0].llmCallCount).toBe(0);
      expect(executions[0].toolCallCount).toBe(0);
    });

    it('toolCallErrorCount counts non-ok states', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'Response',
          timestamp: '2026-01-01T00:00:01.000Z',
          timeInfo: { created: 1735689601000, completed: 1735689605000 },
          agent: 'build', subagent_name: null, subagent_session_id: null,
          tool_calls: [
            { toolCallId: 'tc1', toolName: 'bash', argsJson: '{"command":"ls"}', resultJson: 'ok', state: 'ok' },
            { toolCallId: 'tc2', toolName: 'bash', argsJson: '{"command":"rm"}', resultJson: 'error', state: 'error' },
          ],
          usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 400 },
          model: 'test', modelID: null, providerID: null, latency: 4000, finish_reason: 'stop',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-errors');
      const executions = splitExecutions(splitResult.turns, splitResult.toolCalls, splitResult.skillEvents, 'session-errors');
      expect(executions[0].toolCallCount).toBe(2);
      expect(executions[0].toolCallErrorCount).toBe(1);
    });
  });

  describe.skipIf(!hasRealDB)('S5-05: full pipeline validation with real DB', () => {
    let allInteractions: RawInteraction[];
    let turns: TurnRow[];
    let toolCalls: ToolCallRow[];
    let skillEvents: SkillEventRow[];
    let executions: ExecutionRow[];
    let bridges: InteractionBridgeRow[];
    let sessionId: string;
    let rootExecutionId: string;

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      resetBridgeIdCounter();

      const db = new Database(REAL_DB_PATH, { readonly: true });
      const subagentSessions = db.prepare('SELECT id, parent_id FROM session WHERE parent_id IS NOT NULL').all() as { id: string; parent_id: string }[];
      db.close();

      if (subagentSessions.length === 0) {
        sessionId = '';
        allInteractions = [];
        turns = [];
        toolCalls = [];
        skillEvents = [];
        executions = [];
        bridges = [];
        rootExecutionId = '';
        return;
      }

      const rootSessionId = subagentSessions[0].parent_id;
      sessionId = rootSessionId;

      const rootInteractions = readSession(REAL_DB_PATH, rootSessionId);
      const mergedInteractions: RawInteraction[] = [...rootInteractions];

      for (const sub of subagentSessions.filter(s => s.parent_id === rootSessionId)) {
        const subInteractions = readSession(REAL_DB_PATH, sub.id);
        mergedInteractions.push(...subInteractions);
      }

      allInteractions = mergedInteractions;

      const normalized = normalize(mergedInteractions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, sessionId);
      turns = splitResult.turns;
      toolCalls = splitResult.toolCalls;
      skillEvents = splitResult.skillEvents;

      executions = splitExecutions(turns, toolCalls, skillEvents, sessionId);
      rootExecutionId = executions.find(e => !e.isSubagent)?.id ?? 'root-exec';

      bridges = buildBridges(normalized, toolCalls, turns, sessionId, rootExecutionId);
    });

    it('pipeline produces consistent data across all stages', () => {
      if (allInteractions.length === 0) return;

      expect(turns.length).toBe(allInteractions.length);
      expect(executions.length).toBeGreaterThan(0);
      expect(bridges.length).toBeGreaterThanOrEqual(0);
    });

    it('turns → executions aggregation is consistent', () => {
      if (turns.length === 0) return;

      const rootExec = executions.find(e => !e.isSubagent)!;
      const rootTurns = turns.filter(t => !t.isSubagent);
      expect(rootExec.tokens).toBe(rootTurns.reduce((s, t) => s + t.totalTokens, 0));
      expect(rootExec.llmCallCount).toBe(rootTurns.filter(t => t.role === 'assistant').length);
    });

    it('bridges reference correct root execution', () => {
      for (const bridge of bridges) {
        expect(bridge.dispatchExecutionId).toBe(rootExecutionId);
        expect(bridge.sessionId).toBe(sessionId);
      }
    });

    it('bridge subagentTokens matches execution subagent tokens', () => {
      const matchedBridges = bridges.filter(b => b.status === 'completed');
      for (const bridge of matchedBridges) {
        const subExec = executions.find(e => e.agentSessionId === bridge.subagentSessionId);
        if (subExec) {
          expect(bridge.subagentTokens).toBe(subExec.tokens);
        }
      }
    });

    it('logs full pipeline results', () => {
      console.log('\n=== S5-05: Full Pipeline Validation ===');
      console.log(`Session: ${sessionId}`);
      console.log(`Interactions: ${allInteractions.length}`);
      console.log(`Turns: ${turns.length}, ToolCalls: ${toolCalls.length}, SkillEvents: ${skillEvents.length}`);
      console.log(`Executions: ${executions.length}`);
      console.log(`Bridges: ${bridges.length}`);
      for (const exec of executions) {
        console.log(`  Exec: isSub=${exec.isSubagent} agent=${exec.agentName} tokens=${exec.tokens} tools=${exec.toolCallCount} llm=${exec.llmCallCount}`);
      }
      for (const bridge of bridges) {
        console.log(`  Bridge: status=${bridge.status} subagent=${bridge.subagentName} tokens=${bridge.subagentTokens} latency=${bridge.subagentLatencyMs}ms`);
      }
    });
  });
});
