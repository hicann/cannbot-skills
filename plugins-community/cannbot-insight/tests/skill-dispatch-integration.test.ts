// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll } from 'vitest';
import { readSession } from '../src/lib/ingest/adapters/claude-jsonl.ts';
import { splitIntoTurns, resetIdCounter } from '../src/lib/ingest/turn-split.ts';
import { computeSessionAggregates } from '../src/lib/ingest/data-service.ts';
import path from 'node:path';

const FIXTURE_DIR = path.resolve(__dirname, 'data/claude-sessions');
const SKILL_DISPATCH_FILE = path.join(FIXTURE_DIR, 'skill-dispatch-session.jsonl');

describe('skill dispatch integration', () => {
  let interactions: ReturnType<typeof readSession>;
  let result: ReturnType<typeof splitIntoTurns>;
  let aggregates: ReturnType<typeof computeSessionAggregates>;

  beforeAll(() => {
    resetIdCounter();
    interactions = readSession(SKILL_DISPATCH_FILE, 'skill-dispatch-session');
    result = splitIntoTurns(interactions, 'session-skill-dispatch-test');
    aggregates = computeSessionAggregates(result.turns, result.toolCalls, result.skillEvents);
  });

  // ── Adapter: claude-jsonl → RawInteraction ──

  describe('adapter layer', () => {
    it('parses Skill tool call result from JSONL', () => {
      expect(interactions.length).toBeGreaterThan(0);
      const assistantWithSkill = interactions.find(
        i => i.tool_calls?.some(tc => tc.toolName === 'Skill')
      );
      expect(assistantWithSkill).toBeDefined();
      const skillTc = assistantWithSkill!.tool_calls!.find(tc => tc.toolName === 'Skill');
      expect(skillTc).toBeDefined();
      expect(skillTc!.argsJson).toContain('ops-registry-invoke-workflow');
      expect(skillTc!.resultJson).toContain('核心原则');
    });

    it('parses Agent tool calls with subagent_type field', () => {
      const agentTcs = interactions.flatMap(i => i.tool_calls ?? []).filter(tc => tc.toolName === 'Agent');
      expect(agentTcs.length).toBe(4);

      const architectTc = agentTcs.find(tc => tc.argsJson?.includes('ascendc-ops-architect'));
      expect(architectTc).toBeDefined();
      expect(architectTc!.argsJson).toContain('subagent_type');
      expect(architectTc!.argsJson).toContain('需求分析');
    });

    it('preserves parallel Agent dispatches in same interaction', () => {
      const parallelAssistant = interactions.find(
        i => i.tool_calls?.some(tc => tc.argsJson?.includes('ascendc-ops-designer')) &&
             i.tool_calls?.some(tc => tc.argsJson?.includes('ascendc-ops-tester'))
      );
      expect(parallelAssistant).toBeDefined();
      expect(parallelAssistant!.tool_calls!.length).toBe(2);
    });
  });

  // ── turn-split: RawInteraction → SkillEventRow ──

  describe('turn-split: Agent dispatch SkillEvent', () => {
    it('creates dispatch SkillEvent for Agent with subagent_type', () => {
      const dispatchEvents = result.skillEvents.filter(se => se.eventType === 'dispatch');
      expect(dispatchEvents.length).toBe(3);
    });

    it('dispatch SkillEvent skillName comes from subagent_type', () => {
      const architectDispatch = result.skillEvents.find(
        se => se.skillName === 'ascendc-ops-architect' && se.eventType === 'dispatch'
      );
      expect(architectDispatch).toBeDefined();
      expect(architectDispatch!.success).toBe(true);
      expect(architectDispatch!.argsJson).toContain('需求分析');
      expect(architectDispatch!.argsJson).toContain('ascendc-ops-architect');
    });

    it('excludes general-purpose and general from dispatch SkillEvents', () => {
      const generalDispatch = result.skillEvents.find(
        se => se.skillName === 'general-purpose' || se.skillName === 'general'
      );
      expect(generalDispatch).toBeUndefined();
    });

    it('dispatch SkillEvent has null skillVersion', () => {
      const dispatchEvent = result.skillEvents.find(se => se.eventType === 'dispatch');
      expect(dispatchEvent).toBeDefined();
      expect(dispatchEvent!.skillVersion).toBeNull();
    });

    it('Agent ToolCall isSkillRelated=true for skill-driven dispatches', () => {
      const skillAgentCalls = result.toolCalls.filter(
        tc => tc.isSkillRelated && (tc.toolName === 'Agent' || tc.toolName === 'Task')
      );
      expect(skillAgentCalls.length).toBe(3);

      const names = skillAgentCalls.map(tc => {
        try { return JSON.parse(tc.argsJson!).subagent_type; } catch { return null; }
      });
      expect(names).toContain('ascendc-ops-architect');
      expect(names).toContain('ascendc-ops-designer');
      expect(names).toContain('ascendc-ops-tester');
    });

    it('Agent ToolCall isSkillRelated=false for general-purpose', () => {
      const generalAgentCall = result.toolCalls.find(
        tc => tc.toolName === 'Agent' && tc.argsJson?.includes('general-purpose')
      );
      expect(generalAgentCall).toBeDefined();
      expect(generalAgentCall!.isSkillRelated).toBe(false);
    });

    it('Skill tool call creates invoke SkillEvent (not dispatch)', () => {
      const invokeEvent = result.skillEvents.find(
        se => se.skillName === 'ops-registry-invoke-workflow' && se.eventType === 'invoke'
      );
      expect(invokeEvent).toBeDefined();
      expect(invokeEvent!.argsJson).toContain('ops-registry-invoke-workflow');
    });

    it('Skill ToolCall isSkillRelated=true', () => {
      const skillCall = result.toolCalls.find(tc => tc.toolName === 'Skill');
      expect(skillCall).toBeDefined();
      expect(skillCall!.isSkillRelated).toBe(true);
    });

    it('SkillEvent count: 1 invoke + 3 dispatch = 4 total', () => {
      expect(result.skillEvents.length).toBe(4);
      const invokeEvents = result.skillEvents.filter(se => se.eventType === 'invoke');
      const dispatchEvents = result.skillEvents.filter(se => se.eventType === 'dispatch');
      expect(invokeEvents.length).toBe(1);
      expect(dispatchEvents.length).toBe(3);
    });
  });

  // ── data-service: aggregates ──

  describe('computeSessionAggregates: skill counts', () => {
    it('totalSkillLoadCount equals total SkillEvent count', () => {
      expect(aggregates.totalSkillLoadCount).toBe(result.skillEvents.length);
    });

    it('totalToolCallCount includes all tool calls', () => {
      expect(aggregates.totalToolCallCount).toBe(result.toolCalls.length);
    });

    it('unique skill names from skillEvents include dispatch types', () => {
      const uniqueNames = [...new Set(result.skillEvents.map(se => se.skillName))];
      expect(uniqueNames).toContain('ops-registry-invoke-workflow');
      expect(uniqueNames).toContain('ascendc-ops-architect');
      expect(uniqueNames).toContain('ascendc-ops-designer');
      expect(uniqueNames).toContain('ascendc-ops-tester');
      expect(uniqueNames).not.toContain('general-purpose');
    });

    it('invocationCount logic: invoke+use+dispatch counts per skill', () => {
      // Manually compute invocationCount for each skill
      const invocationCounts = new Map<string, number>();
      for (const se of result.skillEvents) {
        if (se.eventType === 'invoke' || se.eventType === 'use' || se.eventType === 'dispatch') {
          invocationCounts.set(se.skillName, (invocationCounts.get(se.skillName) ?? 0) + 1);
        }
      }
      expect(invocationCounts.get('ops-registry-invoke-workflow')).toBe(1);
      expect(invocationCounts.get('ascendc-ops-architect')).toBe(1);
      expect(invocationCounts.get('ascendc-ops-designer')).toBe(1);
      expect(invocationCounts.get('ascendc-ops-tester')).toBe(1);
      expect(invocationCounts.get('general-purpose')).toBeUndefined();
    });
  });
});
