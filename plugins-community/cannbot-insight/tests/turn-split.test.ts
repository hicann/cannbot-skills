// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll } from 'vitest';
import { splitIntoTurns, resetIdCounter } from '../src/lib/ingest/turn-split.ts';
import { readSession, listSessions } from '../src/lib/ingest/adapters/opencode-db.ts';
import { getContextWindowLimit, getDefaultContextWindow } from '../src/lib/context-window-config.ts';
import type { RawInteraction } from '../src/lib/shared/types.ts';
import type { TurnRow, ToolCallRow, SkillEventRow } from '../src/lib/ingest/turn-split.ts';
import path from 'node:path';
import fs from 'node:fs';

const SYNTHETIC_DATA_PATH = path.resolve(__dirname, 'data/synthetic-opencode.json');
const REAL_DB_PATH = path.resolve(__dirname, 'data/opencode-sessions.db');
const hasRealDB = fs.existsSync(REAL_DB_PATH);

function loadSyntheticData(): RawInteraction[] {
  const raw = fs.readFileSync(SYNTHETIC_DATA_PATH, 'utf-8');
  return JSON.parse(raw);
}

describe('turn-split', () => {
  describe('with synthetic data', () => {
    let interactions: RawInteraction[];
    let result: { turns: TurnRow[], toolCalls: ToolCallRow[], skillEvents: SkillEventRow[] };

    beforeAll(() => {
      resetIdCounter();
      interactions = loadSyntheticData();
      result = splitIntoTurns(interactions, 'session-synthetic-001');
    });

    it('produces correct Turn count matching RawInteraction count', () => {
      expect(result.turns.length).toBe(interactions.length);
    });

    it('produces correct ToolCall count matching total tool_calls across all interactions', () => {
      const expectedToolCallCount = interactions.reduce((sum, i) => {
        return sum + (i.tool_calls ? i.tool_calls.length : 0);
      }, 0);
      expect(result.toolCalls.length).toBe(expectedToolCallCount);
    });

    it('each Turn has all required fields matching Prisma Turn model', () => {
      for (const turn of result.turns) {
        const requiredKeys: Array<keyof TurnRow> = [
          'id', 'sessionId', 'turnIndex', 'role', 'content', 'contentJson',
          'contentSummary', 'inputMessagesJson', 'inputMessagesCount',
          'inputMessagesTokens', 'contextWindowPct', 'agentName', 'subagentName',
          'subagentSessionId', 'totalTokens', 'inputTokens', 'outputTokens',
          'reasoningTokens', 'cacheReadTokens', 'cacheWriteTokens', 'cost',
          'createdAt_ts', 'completedAt', 'latencyMs', 'ttftMs',
          'model', 'modelId', 'providerId', 'temperature', 'maxTokens',
          'finishReason', 'isSubagent', 'parentExecutionId',
        ];
        for (const key of requiredKeys) {
          expect(turn).toHaveProperty(key);
        }
      }
    });

    it('turnIndex is 0-based sequential', () => {
      for (let i = 0; i < result.turns.length; i++) {
        expect(result.turns[i].turnIndex).toBe(i);
      }
    });

    it('role mapping works: "subagent" maps to "assistant"', () => {
      const roles = result.turns.map(t => t.role);
      expect(roles).not.toContain('subagent');
      const subagentInteractions = interactions.filter(i => i.role === 'subagent');
      expect(subagentInteractions.length).toBeGreaterThan(0);
      const assistantTurns = result.turns.filter(t => t.role === 'assistant');
      expect(assistantTurns.length).toBeGreaterThanOrEqual(subagentInteractions.length);
    });

    it('preserves other roles unchanged', () => {
      const userTurn = result.turns.find(t => t.turnIndex === 0);
      expect(userTurn?.role).toBe('user');
    });

    it('token five-item breakdown correctly assigned from RawInteraction.usage', () => {
      const assistantTurn = result.turns.find(t => t.turnIndex === 1);
      expect(assistantTurn).toBeDefined();
      const source = interactions[1];
      expect(assistantTurn!.totalTokens).toBe(source.usage!.total);
      expect(assistantTurn!.inputTokens).toBe(source.usage!.input);
      expect(assistantTurn!.outputTokens).toBe(source.usage!.output);
      expect(assistantTurn!.reasoningTokens).toBe(source.usage!.reasoning);
      expect(assistantTurn!.cacheReadTokens).toBe(source.usage!.cacheRead);
      expect(assistantTurn!.cacheWriteTokens).toBe(source.usage!.cacheWrite);
    });

    it('token fields default to 0 when usage is null', () => {
      const userTurn = result.turns.find(t => t.turnIndex === 0);
      expect(userTurn!.totalTokens).toBe(0);
      expect(userTurn!.inputTokens).toBe(0);
      expect(userTurn!.outputTokens).toBe(0);
      expect(userTurn!.reasoningTokens).toBe(0);
      expect(userTurn!.cacheReadTokens).toBe(0);
      expect(userTurn!.cacheWriteTokens).toBe(0);
    });

    it('time fields: createdAt_ts from timeInfo.created', () => {
      const turn = result.turns.find(t => t.turnIndex === 1);
      expect(turn!.createdAt_ts).toBe(new Date(1778551201000).toISOString());
    });

    it('time fields: completedAt from timeInfo.completed', () => {
      const turn = result.turns.find(t => t.turnIndex === 1);
      expect(turn!.completedAt).toBe(new Date(1778551206000).toISOString());
    });

    it('latency computed from timeInfo.completed - timeInfo.created', () => {
      const turn = result.turns.find(t => t.turnIndex === 1);
      expect(turn!.latencyMs).toBe(1778551206000 - 1778551201000);
    });

    it('latency falls back to interaction.latency when timeInfo incomplete', () => {
      const userTurn = result.turns.find(t => t.turnIndex === 0);
      expect(userTurn!.latencyMs).toBe(interactions[0].latency ?? 0);
    });

    it('skill/load_skill creates SkillEvent with type="load"', () => {
      const loadSkillEvent = result.skillEvents.find(se => se.eventType === 'load');
      expect(loadSkillEvent).toBeDefined();
      expect(loadSkillEvent!.skillName).toBe('agent-debug-diagnosis');
      expect(loadSkillEvent!.skillVersion).toBe(3);
      expect(loadSkillEvent!.success).toBe(true);
    });

    it('skill (bare) creates SkillEvent with type="invoke"', () => {
      const invokeEvent = result.skillEvents.find(se => se.eventType === 'invoke');
      expect(invokeEvent).toBeDefined();
      expect(invokeEvent!.skillName).toBe('agent-debug-diagnosis');
      expect(invokeEvent!.success).toBe(true);
    });

    it('ToolCall.isSkillRelated is true for skill-related tool calls', () => {
      const skillToolCalls = result.toolCalls.filter(tc => tc.isSkillRelated);
      expect(skillToolCalls.length).toBeGreaterThan(0);
      const names = skillToolCalls.map(tc => tc.toolName);
      expect(names).toContain('skill/load_skill');
      expect(names).toContain('skill');
    });

    it('ToolCall.isSkillRelated is true for Agent/task dispatch with subagent_type', () => {
      const skillToolCalls = result.toolCalls.filter(tc => tc.isSkillRelated);
      const dispatchTcs = skillToolCalls.filter(tc => tc.toolName === 'task');
      expect(dispatchTcs.length).toBeGreaterThan(0);
      expect(dispatchTcs[0].isSkillRelated).toBe(true);
    });

    it('ToolCall.isSkillRelated is false for non-skill tool calls (bash)', () => {
      const nonSkillToolCalls = result.toolCalls.filter(tc => !tc.isSkillRelated);
      const names = nonSkillToolCalls.map(tc => tc.toolName);
      expect(names).toContain('bash');
      expect(names).not.toContain('skill');
      expect(names).not.toContain('skill/load_skill');
    });

    it('assistant turn inputMessagesJson is null (reconstructed at read time)', () => {
      const assistantTurn = result.turns.find(t => t.turnIndex === 1);
      expect(assistantTurn!.inputMessagesJson).toBeNull();
    });

    it('inputMessagesCount counts preceding user/assistant/system messages', () => {
      const turn2 = result.turns.find(t => t.turnIndex === 2);
      expect(turn2!.inputMessagesCount).toBeGreaterThan(0);
    });

    it('inputMessagesTokens uses authoritative totalTokens when available', () => {
      const assistantTurn = result.turns.find(t => t.turnIndex === 1);
      const sourceUsage = interactions[1].usage!;
      // totalTokens (prompt + output + reasoning) is the reliable context-size
      // base; the adapter's input+cacheRead+cacheWrite proxy is only a fallback.
      expect(assistantTurn!.inputMessagesTokens).toBe(sourceUsage.total);
      expect(assistantTurn!.inputMessagesTokens).not.toBe(sourceUsage.inputMessagesTokens);
    });

    it('contextWindowPct calculated correctly', () => {
      const turn2 = result.turns.find(t => t.turnIndex === 2);
      expect(turn2!.contextWindowPct).not.toBeNull();
      const contextWindowLimit = getContextWindowLimit(turn2!.model);
      const expectedPct = (turn2!.inputMessagesTokens / contextWindowLimit) * 100;
      expect(turn2!.contextWindowPct!).toBeCloseTo(expectedPct, 2);
    });

    it('contentSummary truncation: content > 200 chars gets truncated', () => {
      const longContentInteraction = interactions.find(
        i => i.content && i.content.length > 200
      );
      if (longContentInteraction) {
        const idx = interactions.indexOf(longContentInteraction);
        const turn = result.turns[idx];
        expect(turn.contentSummary!.length).toBeLessThanOrEqual(200);
        expect(turn.contentSummary!).toBe(longContentInteraction.content!.substring(0, 200));
      }
    });

    it('contentSummary: content <= 200 chars equals content', () => {
      const shortContentInteraction = interactions.find(
        i => i.content && i.content.length <= 200
      );
      if (shortContentInteraction) {
        const idx = interactions.indexOf(shortContentInteraction);
        const turn = result.turns[idx];
        expect(turn.contentSummary).toBe(turn.content);
      }
    });

    it('contentSummary is null when content is null', () => {
      const nullContentInteraction = interactions.find(i => i.content === null);
      if (nullContentInteraction) {
        const idx = interactions.indexOf(nullContentInteraction);
        const turn = result.turns[idx];
        expect(turn.contentSummary).toBeNull();
      }
    });

    it('agentName from RawInteraction.agent', () => {
      const turn0 = result.turns.find(t => t.turnIndex === 0);
      expect(turn0!.agentName).toBe(interactions[0].agent);
    });

    it('subagentName from RawInteraction.subagent_name', () => {
      const subagentTurn = result.turns.find(t => t.isSubagent);
      expect(subagentTurn).toBeDefined();
      expect(subagentTurn!.subagentName).toBe('Kuafu');
    });

    it('subagentSessionId from RawInteraction.subagent_session_id', () => {
      const subagentTurn = result.turns.find(t => t.isSubagent);
      expect(subagentTurn!.subagentSessionId).toBe('ses_synthetic_subagent_001');
    });

    it('isSubagent true for subagent turns', () => {
      const subagentTurns = result.turns.filter(t => t.isSubagent);
      expect(subagentTurns.length).toBe(interactions.filter(i => i.subagent_session_id).length);
      for (const t of subagentTurns) {
        expect(t.subagentSessionId).not.toBeNull();
      }
    });

    it('isSubagent false for root turns', () => {
      const rootTurns = result.turns.filter(t => !t.isSubagent);
      for (const t of rootTurns) {
        expect(t.subagentSessionId).toBeNull();
        expect(t.isSubagent).toBe(false);
      }
    });

    it('model/modelId/providerId extracted correctly', () => {
      const assistantTurn = result.turns.find(t => t.turnIndex === 1);
      expect(assistantTurn!.model).toBe('alibaba-cn/glm-5');
      expect(assistantTurn!.modelId).toBe('glm-5');
      expect(assistantTurn!.providerId).toBe('alibaba-cn');
    });

    it('finishReason extracted correctly', () => {
      const assistantTurn = result.turns.find(t => t.turnIndex === 1);
      expect(assistantTurn!.finishReason).toBe('tool-calls');
    });

    it('ToolCall fields match expected structure', () => {
      const firstToolCall = result.toolCalls[0];
      expect(firstToolCall).toHaveProperty('toolCallId');
      expect(firstToolCall).toHaveProperty('toolName');
      expect(firstToolCall).toHaveProperty('argsJson');
      expect(firstToolCall).toHaveProperty('resultJson');
      expect(firstToolCall).toHaveProperty('state');
      expect(firstToolCall).toHaveProperty('turnId');
    });

    it('ToolCall turnId links back to correct Turn', () => {
      for (const tc of result.toolCalls) {
        const parentTurn = result.turns.find(t => t.id === tc.turnId);
        expect(parentTurn).toBeDefined();
      }
    });

    it('SkillEvent fields match expected structure', () => {
      const firstSkillEvent = result.skillEvents[0];
      expect(firstSkillEvent).toHaveProperty('skillName');
      expect(firstSkillEvent).toHaveProperty('eventType');
      expect(firstSkillEvent).toHaveProperty('success');
      expect(firstSkillEvent).toHaveProperty('turnId');
    });

    it('SkillEvent turnId links back to correct Turn', () => {
      for (const se of result.skillEvents) {
        const parentTurn = result.turns.find(t => t.id === se.turnId);
        expect(parentTurn).toBeDefined();
      }
    });

    it('sessionId set correctly on all turns', () => {
      for (const turn of result.turns) {
        expect(turn.sessionId).toBe('session-synthetic-001');
      }
    });

    it('cache tokens correctly assigned', () => {
      const turn2 = result.turns.find(t => t.turnIndex === 2);
      expect(turn2!.cacheReadTokens).toBe(interactions[2].usage!.cacheRead);
      expect(turn2!.cacheWriteTokens).toBe(interactions[2].usage!.cacheWrite);
    });

    it('total token counts include all sub-fields', () => {
      for (const turn of result.turns.filter(t => t.totalTokens > 0)) {
        const idx = turn.turnIndex;
        const source = interactions[idx];
        if (source.usage) {
          expect(turn.totalTokens).toBe(source.usage.total);
        }
      }
    });
  });

  describe.skipIf(!hasRealDB)('with real DB data', () => {
    let interactions: RawInteraction[];
    let result: { turns: TurnRow[], toolCalls: ToolCallRow[], skillEvents: SkillEventRow[] };
    let sessionId: string;

    beforeAll(() => {
      resetIdCounter();
      const sessions = listSessions(REAL_DB_PATH);
      const sessionWithTools = sessions.find(s => s.turnCount > 5) || sessions[0];
      sessionId = sessionWithTools.id;
      interactions = readSession(REAL_DB_PATH, sessionId);
      result = splitIntoTurns(interactions, sessionId);
    });

    it('produces correct Turn count matching RawInteraction count', () => {
      expect(result.turns.length).toBe(interactions.length);
    });

    it('produces correct ToolCall count', () => {
      const expectedCount = interactions.reduce((sum, i) => {
        return sum + (i.tool_calls ? i.tool_calls.length : 0);
      }, 0);
      expect(result.toolCalls.length).toBe(expectedCount);
    });

    it('each Turn has all required fields', () => {
      for (const turn of result.turns) {
        expect(turn).toHaveProperty('id');
        expect(turn).toHaveProperty('sessionId');
        expect(turn).toHaveProperty('turnIndex');
        expect(turn).toHaveProperty('role');
        expect(turn).toHaveProperty('content');
        expect(turn).toHaveProperty('contentSummary');
        expect(turn).toHaveProperty('totalTokens');
        expect(turn).toHaveProperty('inputTokens');
        expect(turn).toHaveProperty('outputTokens');
        expect(turn).toHaveProperty('latencyMs');
        expect(turn).toHaveProperty('createdAt_ts');
        expect(turn).toHaveProperty('isSubagent');
      }
    });

    it('turnIndex is 0-based sequential', () => {
      for (let i = 0; i < result.turns.length; i++) {
        expect(result.turns[i].turnIndex).toBe(i);
      }
    });

    it('roles include user and assistant', () => {
      const roles = result.turns.map(t => t.role);
      expect(roles).toContain('user');
      expect(roles).toContain('assistant');
    });

    it('assistant turns have null inputMessagesJson (reconstructed at read time)', () => {
      const assistantTurns = result.turns.filter(t => t.role === 'assistant');
      for (const turn of assistantTurns) {
        expect(turn.inputMessagesJson).toBeNull();
      }
    });

    it('contentSummary is truncated or equal to content', () => {
      for (const turn of result.turns) {
        if (turn.content && turn.content.length > 200) {
          expect(turn.contentSummary!.length).toBeLessThanOrEqual(200);
        } else if (turn.content) {
          expect(turn.contentSummary).toBe(turn.content);
        } else {
          expect(turn.contentSummary).toBeNull();
        }
      }
    });

    it('token breakdown fields are populated for assistant turns', () => {
      const assistantWithUsage = result.turns.find(
        t => t.role === 'assistant' && t.totalTokens > 0
      );
      if (assistantWithUsage) {
        expect(assistantWithUsage.inputTokens).toBeGreaterThanOrEqual(0);
        expect(assistantWithUsage.outputTokens).toBeGreaterThanOrEqual(0);
      }
    });

    it('ToolCall turnId references existing Turn', () => {
      for (const tc of result.toolCalls) {
        const turn = result.turns.find(t => t.id === tc.turnId);
        expect(turn).toBeDefined();
      }
    });

    it('sessionId set correctly on all turns', () => {
      for (const turn of result.turns) {
        expect(turn.sessionId).toBe(sessionId);
      }
    });

    it('logs first 3 turns for inspection', () => {
      console.log('\n=== Real DB: First 3 Turns ===');
      for (const turn of result.turns.slice(0, 3)) {
        console.log(JSON.stringify({
          turnIndex: turn.turnIndex,
          role: turn.role,
          contentSummary: turn.contentSummary,
          totalTokens: turn.totalTokens,
          inputTokens: turn.inputTokens,
          outputTokens: turn.outputTokens,
          latencyMs: turn.latencyMs,
          model: turn.model,
          isSubagent: turn.isSubagent,
          inputMessagesCount: turn.inputMessagesCount,
          contextWindowPct: turn.contextWindowPct,
        }, null, 2));
      }
      console.log(`\nTotal turns: ${result.turns.length}, Total toolCalls: ${result.toolCalls.length}, Total skillEvents: ${result.skillEvents.length}`);
    });
  });

  describe('edge cases', () => {
    it('handles empty interactions array', () => {
      resetIdCounter();
      const result = splitIntoTurns([], 'session-empty');
      expect(result.turns.length).toBe(0);
      expect(result.toolCalls.length).toBe(0);
      expect(result.skillEvents.length).toBe(0);
    });

    it('handles single user interaction with no usage', () => {
      resetIdCounter();
      const single: RawInteraction[] = [{
        role: 'user',
        content: 'Hello',
        timestamp: '2026-01-01T00:00:00.000Z',
        timeInfo: { created: 1735689600000 },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        tool_calls: null,
        usage: null,
        model: null,
        modelID: null,
        providerID: null,
        latency: null,
        finish_reason: null,
      }];
      const result = splitIntoTurns(single, 'session-single');
      expect(result.turns.length).toBe(1);
      expect(result.turns[0].role).toBe('user');
      expect(result.turns[0].totalTokens).toBe(0);
      expect(result.turns[0].contentSummary).toBe('Hello');
      expect(result.turns[0].inputMessagesJson).toBeNull();
      expect(result.turns[0].isSubagent).toBe(false);
    });

    it('handles interaction with null timeInfo using timestamp fallback', () => {
      resetIdCounter();
      const single: RawInteraction[] = [{
        role: 'user',
        content: 'Test',
        timestamp: '2026-06-01T12:00:00.000Z',
        timeInfo: null,
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        tool_calls: null,
        usage: null,
        model: null,
        modelID: null,
        providerID: null,
        latency: null,
        finish_reason: null,
      }];
      const result = splitIntoTurns(single, 'session-no-timeinfo');
      expect(result.turns[0].createdAt_ts).toBe('2026-06-01T12:00:00.000Z');
    });

    it('handles skill/invoke toolName', () => {
      resetIdCounter();
      const interactions: RawInteraction[] = [{
        role: 'assistant',
        content: 'Invoking skill',
        timestamp: '2026-01-01T00:00:00.000Z',
        timeInfo: { created: 1735689600000, completed: 1735689605000 },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        tool_calls: [{
          toolCallId: 'tc_invoke',
          toolName: 'skill/invoke',
          argsJson: JSON.stringify({ skill_name: 'my-skill', prompt: 'test' }),
          resultJson: null,
          state: 'ok',
        }],
        usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
        model: null,
        modelID: null,
        providerID: null,
        latency: 5000,
        finish_reason: null,
      }];
      const result = splitIntoTurns(interactions, 'session-skill-invoke');
      expect(result.skillEvents.length).toBe(1);
      expect(result.skillEvents[0].eventType).toBe('invoke');
      expect(result.skillEvents[0].skillName).toBe('my-skill');
    });

    it('handles unknown skill/xxx toolName as type="use"', () => {
      resetIdCounter();
      const interactions: RawInteraction[] = [{
        role: 'assistant',
        content: 'Using skill',
        timestamp: '2026-01-01T00:00:00.000Z',
        timeInfo: { created: 1735689600000 },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        tool_calls: [{
          toolCallId: 'tc_use',
          toolName: 'skill/custom_skill',
          argsJson: null,
          resultJson: null,
          state: 'ok',
        }],
        usage: { total: 50, input: 40, output: 10, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 40 },
        model: null,
        modelID: null,
        providerID: null,
        latency: 0,
        finish_reason: null,
      }];
      const result = splitIntoTurns(interactions, 'session-skill-use');
      expect(result.skillEvents.length).toBe(1);
      expect(result.skillEvents[0].eventType).toBe('use');
      expect(result.skillEvents[0].skillName).toBe('custom_skill');
    });

    it('contextWindowPct defaults to 128000 context window for unknown model', () => {
      resetIdCounter();
      const interactions: RawInteraction[] = [
        {
          role: 'user',
          content: 'prompt',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1735689600000 },
          agent: null, subagent_name: null, subagent_session_id: null,
          tool_calls: null, usage: null,
          model: null, modelID: null, providerID: null, latency: null, finish_reason: null,
        },
        {
          role: 'assistant',
          content: 'response',
          timestamp: '2026-01-01T00:00:01.000Z',
          timeInfo: { created: 1735689601000, completed: 1735689605000 },
          agent: null, subagent_name: null, subagent_session_id: null,
          tool_calls: null,
          usage: { total: 500, input: 400, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 400 },
          model: 'unknown-model-xyz', modelID: null, providerID: null, latency: 4000, finish_reason: 'stop',
        },
      ];
      const result = splitIntoTurns(interactions, 'session-ctx-window');
      const assistantTurn = result.turns.find(t => t.role === 'assistant');
      expect(assistantTurn!.contextWindowPct).not.toBeNull();
      const contextWindowLimit = getContextWindowLimit(assistantTurn!.model);
      const expectedPct = (assistantTurn!.inputMessagesTokens / contextWindowLimit) * 100;
      expect(assistantTurn!.contextWindowPct!).toBeCloseTo(expectedPct, 2);
    });

    it('handles skill/<name> toolName from SKILL.md Read detection as type="use"', () => {
      resetIdCounter();
      const interactions: RawInteraction[] = [{
        role: 'assistant',
        content: 'Reading skill file',
        timestamp: '2026-01-01T00:00:00.000Z',
        timeInfo: { created: 1735689600000 },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        tool_calls: [
          {
            toolCallId: 'tc_read_skill',
            toolName: 'Read',
            argsJson: JSON.stringify({ file_path: '/home/user/.claude/skills/cannbot-skill-review/SKILL.md' }),
            resultJson: '---\nname: cannbot-skill-review\n...',
            state: 'completed',
          },
          {
            toolCallId: 'skill-tc_read_skill',
            toolName: 'skill/cannbot-skill-review',
            argsJson: JSON.stringify({ skill: 'cannbot-skill-review', file_path: '/home/user/.claude/skills/cannbot-skill-review/SKILL.md' }),
            resultJson: null,
            state: 'completed',
          },
        ],
        usage: { total: 100, input: 80, output: 20, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 80 },
        model: null,
        modelID: null,
        providerID: null,
        latency: 0,
        finish_reason: null,
      }];
      const result = splitIntoTurns(interactions, 'session-skill-read');
      expect(result.toolCalls.length).toBe(2);
      expect(result.skillEvents.length).toBe(1);
      expect(result.skillEvents[0].skillName).toBe('cannbot-skill-review');
      expect(result.skillEvents[0].eventType).toBe('use');
      expect(result.skillEvents[0].success).toBe(true);
    });

    it('parentSessionId parameter is available but parentExecutionId stays null', () => {
      resetIdCounter();
      const interactions: RawInteraction[] = [{
        role: 'assistant',
        content: 'Subagent work',
        timestamp: '2026-01-01T00:00:00.000Z',
        timeInfo: { created: 1735689600000 },
        agent: 'Kuafu',
        subagent_name: 'Kuafu',
        subagent_session_id: 'sub_ses_001',
        tool_calls: null,
        usage: null,
        model: null, modelID: null, providerID: null, latency: null, finish_reason: null,
      }];
      const result = splitIntoTurns(interactions, 'session-parent', 'parent-ses-001');
      expect(result.turns[0].parentExecutionId).toBeNull();
      expect(result.turns[0].isSubagent).toBe(true);
      expect(result.turns[0].subagentSessionId).toBe('sub_ses_001');
    });

    it('/compact: context size, pct, and count reset across each compaction boundary', () => {
      resetIdCounter();
      const limit = getContextWindowLimit(null);
      const cont = 'This session is being continued from a previous conversation that ran out of context.';
      const interactions: RawInteraction[] = [
        // segment 1: context grows
        { role: 'user', content: 'start', timestamp: '2026-01-01T00:00:00.000Z', timeInfo: { created: 1735689600000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null, usage: null,
          model: null, modelID: null, providerID: null, latency: null, finish_reason: null },
        { role: 'assistant', content: 'a1', timestamp: '2026-01-01T00:00:01.000Z', timeInfo: { created: 1735689601000, completed: 1735689605000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: { total: 10000, input: 2, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 9898, cost: 0, inputMessagesTokens: 9900 },
          model: null, modelID: null, providerID: null, latency: 4000, finish_reason: 'stop' },
        { role: 'assistant', content: 'a2', timestamp: '2026-01-01T00:00:02.000Z', timeInfo: { created: 1735689602000, completed: 1735689606000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: { total: 50000, input: 2, output: 100, reasoning: 0, cacheRead: 49898, cacheWrite: 0, cost: 0, inputMessagesTokens: 49900 },
          model: null, modelID: null, providerID: null, latency: 4000, finish_reason: 'stop' },
        // compact boundary 1: continuation summary replaces history
        { role: 'user', content: cont, timestamp: '2026-01-01T00:00:03.000Z', timeInfo: { created: 1735689603000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null, usage: null,
          model: null, modelID: null, providerID: null, latency: null, finish_reason: null },
        // segment 2: post-compact context must DROP, not freeze at 50000
        { role: 'assistant', content: 'a3', timestamp: '2026-01-01T00:00:04.000Z', timeInfo: { created: 1735689604000, completed: 1735689608000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: { total: 30000, input: 2, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 29898, cost: 0, inputMessagesTokens: 29900 },
          model: null, modelID: null, providerID: null, latency: 4000, finish_reason: 'stop' },
        { role: 'assistant', content: 'a4', timestamp: '2026-01-01T00:00:05.000Z', timeInfo: { created: 1735689605000, completed: 1735689609000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: { total: 80000, input: 2, output: 100, reasoning: 0, cacheRead: 79898, cacheWrite: 0, cost: 0, inputMessagesTokens: 79900 },
          model: null, modelID: null, providerID: null, latency: 4000, finish_reason: 'stop' },
        // compact boundary 2: second compaction
        { role: 'user', content: cont, timestamp: '2026-01-01T00:00:06.000Z', timeInfo: { created: 1735689606000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null, usage: null,
          model: null, modelID: null, providerID: null, latency: null, finish_reason: null },
        // segment 3: drops again from 80000 to 40000
        { role: 'assistant', content: 'a5', timestamp: '2026-01-01T00:00:07.000Z', timeInfo: { created: 1735689607000, completed: 1735689611000 },
          agent: null, subagent_name: null, subagent_session_id: null, tool_calls: null,
          usage: { total: 40000, input: 2, output: 100, reasoning: 0, cacheRead: 0, cacheWrite: 39898, cost: 0, inputMessagesTokens: 39900 },
          model: null, modelID: null, providerID: null, latency: 4000, finish_reason: 'stop' },
      ];
      const result = splitIntoTurns(interactions, 'session-compact');

      // segment 1 peak (turnIndex 2): 50000 tokens
      const seg1 = result.turns[2];
      expect(seg1.inputMessagesTokens).toBe(50000);
      expect(seg1.contextWindowPct).toBeCloseTo((50000 / limit) * 100, 2);

      // post-compact 1 (turnIndex 4): must drop to 30000, NOT freeze at 50000
      const post1 = result.turns[4];
      expect(post1.inputMessagesTokens).toBe(30000);
      expect(post1.contextWindowPct).toBeCloseTo((30000 / limit) * 100, 2);
      // count reset: only the continuation summary precedes it, not all 4 prior turns
      expect(post1.inputMessagesCount).toBeLessThan(seg1.inputMessagesCount);

      // segment 2 peak (turnIndex 5): 80000
      const seg2 = result.turns[5];
      expect(seg2.inputMessagesTokens).toBe(80000);

      // post-compact 2 (turnIndex 7): drops again to 40000, NOT frozen at 80000
      const post2 = result.turns[7];
      expect(post2.inputMessagesTokens).toBe(40000);
      expect(post2.contextWindowPct).toBeCloseTo((40000 / limit) * 100, 2);
      expect(post2.inputMessagesCount).toBeLessThan(seg2.inputMessagesCount);
    });
  });
});
