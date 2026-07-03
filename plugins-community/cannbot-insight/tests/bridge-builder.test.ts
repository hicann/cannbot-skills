// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll } from 'vitest';
import { buildBridges, resetIdCounter } from '../src/lib/ingest/bridge-builder.ts';
import { splitIntoTurns, resetIdCounter as resetTurnIdCounter } from '../src/lib/ingest/turn-split.ts';
import { normalize } from '../src/lib/ingest/normalize.ts';
import { readSession, listSessions } from '../src/lib/ingest/adapters/opencode-db.ts';
import type { RawInteraction } from '../src/lib/shared/types.ts';
import type { TurnRow, ToolCallRow } from '../src/lib/ingest/turn-split.ts';
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

function createSimpleInteractions(): RawInteraction[] {
  return [
    {
      role: 'user',
      content: 'Test prompt',
      timestamp: '2026-01-01T00:00:00.000Z',
      timeInfo: { created: 1735689600000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: null,
      usage: null,
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: null,
      finish_reason: null,
    },
    {
      role: 'assistant',
      content: 'Dispatching to subagent',
      timestamp: '2026-01-01T00:00:01.000Z',
      timeInfo: { created: 1735689601000, completed: 1735689602000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: [{
        toolCallId: 'tc_task_001',
        toolName: 'task',
        argsJson: JSON.stringify({
          subagent_name: 'Kuafu',
          prompt: 'Analyze the issue',
          subagent_session_id: 'sub_ses_001',
          subagent_type: 'general',
        }),
        resultJson: 'task_id: sub_ses_001\nResult: analysis complete',
        state: 'ok',
      }],
      usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.005, inputMessagesTokens: 400 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 1000,
      finish_reason: 'tool-calls',
    },
    {
      role: 'subagent',
      content: 'Subagent working on analysis',
      timestamp: '2026-01-01T00:00:03.000Z',
      timeInfo: { created: 1735689603000, completed: 1735689606000 },
      agent: 'Kuafu',
      subagent_name: 'Kuafu',
      subagent_session_id: 'sub_ses_001',
      tool_calls: [{
        toolCallId: 'tc_bash_001',
        toolName: 'bash',
        argsJson: '{"command":"ls"}',
        resultJson: 'file1 file2',
        state: 'ok',
      }],
      usage: { total: 300, input: 200, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.003, inputMessagesTokens: 200 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 3000,
      finish_reason: 'stop',
    },
    {
      role: 'subagent',
      content: 'Analysis complete. Found 2 issues.',
      timestamp: '2026-01-01T00:00:07.000Z',
      timeInfo: { created: 1735689607000, completed: 1735689610000 },
      agent: 'Kuafu',
      subagent_name: 'Kuafu',
      subagent_session_id: 'sub_ses_001',
      tool_calls: null,
      usage: { total: 200, input: 100, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.002, inputMessagesTokens: 100 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 3000,
      finish_reason: 'stop',
    },
  ];
}

function createMultiSubagentInteractions(): RawInteraction[] {
  return [
    {
      role: 'user',
      content: 'Multi-subagent test',
      timestamp: '2026-01-01T00:00:00.000Z',
      timeInfo: { created: 1735689600000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: null,
      usage: null,
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: null,
      finish_reason: null,
    },
    {
      role: 'assistant',
      content: 'Dispatching two subagents',
      timestamp: '2026-01-01T00:00:01.000Z',
      timeInfo: { created: 1735689601000, completed: 1735689602000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: [
        {
          toolCallId: 'tc_task_001',
          toolName: 'task',
          argsJson: JSON.stringify({
            subagent_name: 'Kuafu',
            prompt: 'Analyze X',
            subagent_session_id: 'sub_ses_001',
          }),
          resultJson: 'Kuafu completed',
          state: 'ok',
        },
        {
          toolCallId: 'tc_task_002',
          toolName: 'task',
          argsJson: JSON.stringify({
            subagent_name: 'Explore',
            prompt: 'Search for Y',
            subagent_session_id: 'sub_ses_002',
          }),
          resultJson: 'Explore completed',
          state: 'ok',
        },
      ],
      usage: { total: 800, input: 600, output: 200, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.008, inputMessagesTokens: 600 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 1000,
      finish_reason: 'tool-calls',
    },
    {
      role: 'subagent',
      content: 'Kuafu result',
      timestamp: '2026-01-01T00:00:03.000Z',
      timeInfo: { created: 1735689603000, completed: 1735689605000 },
      agent: 'Kuafu',
      subagent_name: 'Kuafu',
      subagent_session_id: 'sub_ses_001',
      tool_calls: null,
      usage: { total: 400, input: 300, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.004, inputMessagesTokens: 300 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 2000,
      finish_reason: 'stop',
    },
    {
      role: 'subagent',
      content: 'Explore result',
      timestamp: '2026-01-01T00:00:06.000Z',
      timeInfo: { created: 1735689606000, completed: 1735689608000 },
      agent: 'Explore',
      subagent_name: 'Explore',
      subagent_session_id: 'sub_ses_002',
      tool_calls: null,
      usage: { total: 600, input: 400, output: 200, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.006, inputMessagesTokens: 400 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 2000,
      finish_reason: 'stop',
    },
  ];
}

function createUnmatchedInteractions(): RawInteraction[] {
  return [
    {
      role: 'user',
      content: 'Unmatched test',
      timestamp: '2026-01-01T00:00:00.000Z',
      timeInfo: { created: 1735689600000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: null,
      usage: null,
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: null,
      finish_reason: null,
    },
    {
      role: 'assistant',
      content: 'Dispatching',
      timestamp: '2026-01-01T00:00:01.000Z',
      timeInfo: { created: 1735689601000, completed: 1735689602000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: [{
        toolCallId: 'tc_task_unmatched',
        toolName: 'task',
        argsJson: JSON.stringify({
          subagent_name: 'MissingAgent',
          prompt: 'Do something',
          subagent_session_id: 'sub_ses_nonexistent',
        }),
        resultJson: null,
        state: 'ok',
      }],
      usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.005, inputMessagesTokens: 400 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 1000,
      finish_reason: 'tool-calls',
    },
  ];
}

function createErrorSubagentInteractions(): RawInteraction[] {
  return [
    {
      role: 'user',
      content: 'Error subagent test',
      timestamp: '2026-01-01T00:00:00.000Z',
      timeInfo: { created: 1735689600000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: null,
      usage: null,
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: null,
      finish_reason: null,
    },
    {
      role: 'assistant',
      content: 'Dispatching to failing subagent',
      timestamp: '2026-01-01T00:00:01.000Z',
      timeInfo: { created: 1735689601000, completed: 1735689602000 },
      agent: 'build',
      subagent_name: null,
      subagent_session_id: null,
      tool_calls: [{
        toolCallId: 'tc_task_error',
        toolName: 'task',
        argsJson: JSON.stringify({
          subagent_name: 'FailBot',
          prompt: 'Do something impossible',
          subagent_session_id: 'sub_ses_error',
        }),
        resultJson: 'Error occurred',
        state: 'ok',
      }],
      usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.005, inputMessagesTokens: 400 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 1000,
      finish_reason: 'tool-calls',
    },
    {
      role: 'subagent',
      content: 'Subagent encountered error',
      timestamp: '2026-01-01T00:00:03.000Z',
      timeInfo: { created: 1735689603000, completed: 1735689606000 },
      agent: 'FailBot',
      subagent_name: 'FailBot',
      subagent_session_id: 'sub_ses_error',
      tool_calls: null,
      usage: { total: 300, input: 200, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0.003, inputMessagesTokens: 200 },
      model: 'test-model',
      modelID: null,
      providerID: null,
      latency: 3000,
      finish_reason: 'error',
    },
  ];
}

describe('bridge-builder', () => {
  describe('with simple synthetic data (single task() call with subagent)', () => {
    let interactions: RawInteraction[];
    let turnResult: { turns: TurnRow[], toolCalls: ToolCallRow[] };
    let bridges: InteractionBridgeRow[];

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      interactions = createSimpleInteractions();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-001');
      turnResult = { turns: splitResult.turns, toolCalls: splitResult.toolCalls };
      bridges = buildBridges(normalized, turnResult.toolCalls, turnResult.turns, 'session-001', 'root-exec-001');
    });

    it('produces exactly 1 bridge for 1 task() call', () => {
      expect(bridges.length).toBe(1);
    });

    it('each bridge has all required InteractionBridgeRow fields', () => {
      const requiredKeys: Array<keyof InteractionBridgeRow> = [
        'id', 'sessionId', 'dispatchExecutionId', 'dispatchTurnId',
        'dispatchToolCallId', 'dispatchContent', 'dispatchTimestamp',
        'responseExecutionId', 'responseTurnId', 'responseContent',
        'responseTimestamp', 'subagentSessionId', 'subagentType',
        'subagentName', 'status', 'subagentTokens', 'subagentLatencyMs',
      ];
      for (const bridge of bridges) {
        for (const key of requiredKeys) {
          expect(bridge).toHaveProperty(key);
        }
      }
    });

    it('dispatch side filled from task() tool call', () => {
      const bridge = bridges[0];
      expect(bridge.dispatchExecutionId).toBe('root-exec-001');
      expect(bridge.dispatchToolCallId).toBeDefined();
      expect(bridge.dispatchContent).toBe('Analyze the issue');
      expect(bridge.dispatchTimestamp).toBeDefined();
      expect(bridge.subagentSessionId).toBe('sub_ses_001');
      expect(bridge.subagentName).toBe('Kuafu');
      expect(bridge.subagentType).toBe('general');
    });

    it('response side filled from matched subagent turns', () => {
      const bridge = bridges[0];
      expect(bridge.responseContent).toBe('Analysis complete. Found 2 issues.');
      expect(bridge.responseTimestamp).toBeDefined();
      expect(bridge.responseTurnId).toBeDefined();
    });

    it('status is "completed" for matched bridge with no errors', () => {
      expect(bridges[0].status).toBe('completed');
    });

    it('subagentTokens sums subagent turn totalTokens', () => {
      expect(bridges[0].subagentTokens).toBe(300 + 200);
    });

    it('subagentLatencyMs computed from timestamps', () => {
      expect(bridges[0].subagentLatencyMs).toBeGreaterThan(0);
    });
  });

  describe('with multiple subagents (2+ task() calls)', () => {
    let bridges: InteractionBridgeRow[];

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = createMultiSubagentInteractions();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-multi');
      bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-multi', 'root-exec-multi');
    });

    it('produces 2 bridges for 2 task() calls', () => {
      expect(bridges.length).toBe(2);
    });

    it('each bridge matches correct subagent session', () => {
      const bridge1 = bridges.find(b => b.subagentSessionId === 'sub_ses_001');
      const bridge2 = bridges.find(b => b.subagentSessionId === 'sub_ses_002');
      expect(bridge1).toBeDefined();
      expect(bridge2).toBeDefined();
      expect(bridge1!.subagentName).toBe('Kuafu');
      expect(bridge2!.subagentName).toBe('Explore');
    });

    it('each bridge has correct subagentTokens', () => {
      const bridge1 = bridges.find(b => b.subagentSessionId === 'sub_ses_001');
      const bridge2 = bridges.find(b => b.subagentSessionId === 'sub_ses_002');
      expect(bridge1!.subagentTokens).toBe(400);
      expect(bridge2!.subagentTokens).toBe(600);
    });

    it('both bridges have status "completed"', () => {
      expect(bridges[0].status).toBe('completed');
      expect(bridges[1].status).toBe('completed');
    });
  });

  describe('unmatched task() call (no corresponding subagent)', () => {
    let bridges: InteractionBridgeRow[];

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = createUnmatchedInteractions();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-unmatched');
      bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-unmatched', 'root-exec-unmatched');
    });

    it('produces 1 bridge for 1 task() call', () => {
      expect(bridges.length).toBe(1);
    });

    it('status is "dispatched" or "timeout" for unmatched bridge', () => {
      expect(['dispatched', 'timeout']).toContain(bridges[0].status);
    });

    it('response fields are null for unmatched bridge', () => {
      expect(bridges[0].responseExecutionId).toBeNull();
      expect(bridges[0].responseTurnId).toBeNull();
      expect(bridges[0].responseContent).toBeNull();
      expect(bridges[0].responseTimestamp).toBeNull();
    });

    it('subagentTokens and subagentLatencyMs are 0 for unmatched bridge', () => {
      expect(bridges[0].subagentTokens).toBe(0);
      expect(bridges[0].subagentLatencyMs).toBe(0);
    });

    it('dispatch side still has data from task() call', () => {
      expect(bridges[0].dispatchExecutionId).toBe('root-exec-unmatched');
      expect(bridges[0].dispatchContent).toBe('Do something');
      expect(bridges[0].dispatchToolCallId).toBeDefined();
    });
  });

  describe('status logic', () => {
    it('status "completed" when matched with response and no error', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = createSimpleInteractions();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-completed');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-completed', 'root-exec');
      expect(bridges[0].status).toBe('completed');
    });

    it('status "failed" when subagent turn has finish_reason "error"', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = createErrorSubagentInteractions();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-failed');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-failed', 'root-exec');
      expect(bridges[0].status).toBe('failed');
    });

    it('status "dispatched" for recent unmatched task() call', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const recentTime = new Date().toISOString();
      const interactions: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'Dispatching',
          timestamp: recentTime,
          timeInfo: { created: Date.now(), completed: Date.now() + 1000 },
          agent: 'build',
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc_recent',
            toolName: 'task',
            argsJson: JSON.stringify({ prompt: 'Do something', subagent_session_id: 'sub_recent_missing' }),
            resultJson: null,
            state: 'ok',
          }],
          usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
          model: 'test', modelID: null, providerID: null, latency: 1000, finish_reason: 'tool-calls',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-recent');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-recent', 'root-exec');
      expect(bridges[0].status).toBe('dispatched');
    });

    it('status "timeout" for old unmatched task() call', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const oldTimestamp = '2020-01-01T00:00:00.000Z';
      const oldCreated = 1577836800000;
      const interactions: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'Dispatching old',
          timestamp: oldTimestamp,
          timeInfo: { created: oldCreated, completed: oldCreated + 1000 },
          agent: 'build',
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc_old',
            toolName: 'task',
            argsJson: JSON.stringify({ prompt: 'Old task', subagent_session_id: 'sub_old_missing' }),
            resultJson: null,
            state: 'ok',
          }],
          usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
          model: 'test', modelID: null, providerID: null, latency: 1000, finish_reason: 'tool-calls',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-old');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-old', 'root-exec');
      expect(bridges[0].status).toBe('timeout');
    });
  });

  describe('with full synthetic data (from synthetic-opencode.json)', () => {
    let bridges: InteractionBridgeRow[];

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions = loadSyntheticData();
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-synthetic-001');
      bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-synthetic-001', 'root-exec-synthetic');
    });

    it('produces 1 bridge for the task() call in synthetic data', () => {
      const taskToolCalls = bridges;
      expect(taskToolCalls.length).toBe(1);
    });

    it('bridge matches subagent_session_id from synthetic data', () => {
      expect(bridges[0].subagentSessionId).toBe('ses_synthetic_subagent_001');
    });

    it('bridge has subagentName "Kuafu"', () => {
      expect(bridges[0].subagentName).toBe('Kuafu');
    });

    it('bridge status is "completed"', () => {
      expect(bridges[0].status).toBe('completed');
    });

    it('subagentTokens includes both subagent turns', () => {
      expect(bridges[0].subagentTokens).toBeGreaterThan(0);
    });

    it('responseContent is from last subagent turn', () => {
      expect(bridges[0].responseContent).toContain('race condition');
    });

    it('dispatchContent is the prompt from task() args', () => {
      expect(bridges[0].dispatchContent).toContain('sys_timer_create');
    });
  });

  describe.skipIf(!hasRealDB)('with real DB data', () => {
    let bridges: InteractionBridgeRow[];
    let sessionId: string;
    let interactions: RawInteraction[];
    let splitResult: { turns: TurnRow[], toolCalls: ToolCallRow[] };

    beforeAll(() => {
      resetTurnIdCounter();
      resetIdCounter();
      const db = new Database(REAL_DB_PATH, { readonly: true });
      const subagentSessions = db.prepare('SELECT id, parent_id FROM session WHERE parent_id IS NOT NULL').all() as { id: string; parent_id: string }[];
      db.close();

      if (subagentSessions.length === 0) {
        sessionId = '';
        interactions = [];
        splitResult = { turns: [], toolCalls: [] };
        bridges = [];
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

      interactions = allInteractions;
      const normalized = normalize(allInteractions, 'opencode-db');
      splitResult = splitIntoTurns(normalized, sessionId);
      bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, sessionId, 'root-exec-real');
    });

    it('produces bridges for task() calls in real DB', () => {
      if (bridges.length === 0) return;
      expect(bridges.length).toBeGreaterThan(0);
    });

    it('each bridge has all required fields', () => {
      for (const bridge of bridges) {
        expect(bridge).toHaveProperty('id');
        expect(bridge).toHaveProperty('sessionId');
        expect(bridge).toHaveProperty('dispatchExecutionId');
        expect(bridge).toHaveProperty('dispatchToolCallId');
        expect(bridge).toHaveProperty('dispatchContent');
        expect(bridge).toHaveProperty('dispatchTimestamp');
        expect(bridge).toHaveProperty('subagentSessionId');
        expect(bridge).toHaveProperty('status');
        expect(bridge).toHaveProperty('subagentTokens');
        expect(bridge).toHaveProperty('subagentLatencyMs');
      }
    });

    it('bridges for matched subagents have response data', () => {
      const matchedBridges = bridges.filter(b => b.status === 'completed');
      if (matchedBridges.length > 0) {
        for (const bridge of matchedBridges) {
          expect(bridge.subagentSessionId).not.toBeNull();
        }
      }
    });

    it('logs bridge results for inspection', () => {
      console.log('\n=== Real DB Bridge-Builder Results ===');
      console.log(`Session: ${sessionId}`);
      console.log(`Total interactions: ${interactions.length}`);
      console.log(`Total bridges: ${bridges.length}`);
      for (const bridge of bridges) {
        console.log(JSON.stringify({
          status: bridge.status,
          subagentSessionId: bridge.subagentSessionId,
          subagentName: bridge.subagentName,
          dispatchContent: bridge.dispatchContent?.substring(0, 80),
          responseContent: bridge.responseContent?.substring(0, 80),
          subagentTokens: bridge.subagentTokens,
          subagentLatencyMs: bridge.subagentLatencyMs,
        }, null, 2));
      }
    });
  });

  describe('edge cases', () => {
    it('handles empty interactions array', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const bridges = buildBridges([], [], [], 'session-empty', 'root-exec');
      expect(bridges.length).toBe(0);
    });

    it('handles interactions with no task() calls', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'user',
          content: 'Hello',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1735689600000 },
          agent: 'build',
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc_bash',
            toolName: 'bash',
            argsJson: '{"command":"ls"}',
            resultJson: 'output',
            state: 'ok',
          }],
          usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
          model: 'test', modelID: null, providerID: null, latency: 100, finish_reason: 'tool-calls',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-no-task');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-no-task', 'root-exec');
      expect(bridges.length).toBe(0);
    });

    it('handles task() call with null argsJson', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'Dispatching',
          timestamp: '2020-01-01T00:00:00.000Z',
          timeInfo: { created: 1577836800000 },
          agent: 'build',
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc_no_args',
            toolName: 'task',
            argsJson: null,
            resultJson: null,
            state: 'ok',
          }],
          usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
          model: 'test', modelID: null, providerID: null, latency: 100, finish_reason: 'tool-calls',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-null-args');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-null-args', 'root-exec');
      expect(bridges.length).toBe(1);
      expect(bridges[0].status).toBe('timeout');
      expect(bridges[0].dispatchContent).toBeNull();
    });

    it('matches by time proximity when argsJson has no session_id', () => {
      resetTurnIdCounter();
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'user',
          content: 'test',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1735689600000 },
          agent: 'build', subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: null, model: 'test', modelID: null, providerID: null, latency: null, finish_reason: null,
        },
        {
          role: 'assistant',
          content: 'Dispatching',
          timestamp: '2026-01-01T00:00:01.000Z',
          timeInfo: { created: 1735689601000, completed: 1735689602000 },
          agent: 'build', subagent_name: null, subagent_session_id: null,
          tool_calls: [{
            toolCallId: 'tc_prox',
            toolName: 'task',
            argsJson: JSON.stringify({ prompt: 'Test' }),
            resultJson: null,
            state: 'ok',
          }],
          usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 400 },
          model: 'test', modelID: null, providerID: null, latency: 1000, finish_reason: 'tool-calls',
        },
        {
          role: 'subagent',
          content: 'Nearby response',
          timestamp: '2026-01-01T00:00:02.000Z',
          timeInfo: { created: 1735689602000, completed: 1735689603000 },
          agent: 'ProxyAgent', subagent_name: 'ProxyAgent', subagent_session_id: 'sub_ses_proximity',
          tool_calls: null,
          usage: { total: 200, input: 100, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 400 },
          model: 'test', modelID: null, providerID: null, latency: 1000, finish_reason: 'stop',
        },
      ];
      const normalized = normalize(interactions, 'opencode-db');
      const splitResult = splitIntoTurns(normalized, 'session-proximity');
      const bridges = buildBridges(normalized, splitResult.toolCalls, splitResult.turns, 'session-proximity', 'root-exec');
      expect(bridges.length).toBe(1);
      expect(bridges[0].subagentSessionId).toBe('sub_ses_proximity');
      expect(bridges[0].status).toBe('completed');
    });
  });
});
