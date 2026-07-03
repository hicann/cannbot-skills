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
import { extractSkillNameFromArgs, groupSkillEvents, type SkillEventItem, type SkillToolCallItem } from '../src/lib/skill-event-grouping.ts';
import path from 'node:path';

const FIXTURE_DIR = path.resolve(__dirname, 'data/claude-sessions');
const SKILL_DISPATCH_FILE = path.join(FIXTURE_DIR, 'skill-dispatch-session.jsonl');

describe('skill dispatch display integration', () => {
  let interactions: ReturnType<typeof readSession>;
  let result: ReturnType<typeof splitIntoTurns>;
  let aggregates: ReturnType<typeof computeSessionAggregates>;

  // Simulate what TurnDetail.tsx passes to SkillEventList
  let skillEvents: SkillEventItem[];
  let skillToolCalls: SkillToolCallItem[];
  let groups: ReturnType<typeof groupSkillEvents>;

  beforeAll(() => {
    resetIdCounter();
    interactions = readSession(SKILL_DISPATCH_FILE, 'skill-dispatch-session');
    result = splitIntoTurns(interactions, 'session-skill-dispatch-test');
    aggregates = computeSessionAggregates(result.turns, result.toolCalls, result.skillEvents);

    skillEvents = result.skillEvents.map(se => ({
      id: se.id,
      skillName: se.skillName,
      skillVersion: se.skillVersion,
      eventType: se.eventType,
      success: se.success,
      errorMessage: se.errorMessage,
      argsJson: se.argsJson,
      durationMs: se.durationMs,
    }));

    skillToolCalls = result.toolCalls.filter(tc => tc.isSkillRelated).map(tc => ({
      id: tc.id,
      toolCallId: tc.toolCallId,
      toolName: tc.toolName,
      argsJson: tc.argsJson,
      resultJson: tc.resultJson,
      state: tc.state,
      durationMs: tc.durationMs,
    }));

    groups = groupSkillEvents(skillEvents, skillToolCalls);
  });

  // ── extractSkillNameFromArgs ──

  describe('extractSkillNameFromArgs: skill name extraction', () => {
    it('extracts skill_name from Skill tool call args', () => {
      expect(extractSkillNameFromArgs(JSON.stringify({ skill: 'ops-registry-invoke-workflow' }))).toBe('ops-registry-invoke-workflow');
      expect(extractSkillNameFromArgs(JSON.stringify({ skill_name: 'my-skill' }))).toBe('my-skill');
    });

    it('extracts subagent_type from Agent tool call args', () => {
      expect(extractSkillNameFromArgs(JSON.stringify({ subagent_type: 'ascendc-ops-architect', prompt: '需求分析' }))).toBe('ascendc-ops-architect');
    });

    it('extracts subagent_name from Task tool call args', () => {
      expect(extractSkillNameFromArgs(JSON.stringify({ subagent_name: 'ascendc-ops-designer' }))).toBe('ascendc-ops-designer');
    });

    it('returns null for args without skill identifiers', () => {
      expect(extractSkillNameFromArgs(JSON.stringify({ command: 'ls' }))).toBeNull();
    });

    it('returns null for null args', () => {
      expect(extractSkillNameFromArgs(null)).toBeNull();
    });

    it('returns null for invalid JSON', () => {
      expect(extractSkillNameFromArgs('not-json')).toBeNull();
    });
  });

  // ── groupSkillEvents: dispatch result display ──

  describe('groupSkillEvents: dispatch groups have result data', () => {
    it('creates a group for each dispatch skillName', () => {
      const dispatchGroups = groups.filter(g => g.dispatchEvent !== null);
      expect(dispatchGroups.length).toBe(3);
      const names = dispatchGroups.map(g => g.skillName);
      expect(names).toContain('ascendc-ops-architect');
      expect(names).toContain('ascendc-ops-designer');
      expect(names).toContain('ascendc-ops-tester');
    });

    it('dispatch group has dispatchResultJson from Agent tool call', () => {
      const architectGroup = groups.find(g => g.skillName === 'ascendc-ops-architect');
      expect(architectGroup).toBeDefined();
      expect(architectGroup!.dispatchResultJson).not.toBeNull();
      expect(architectGroup!.dispatchResultJson).toContain('需求分析已完成');
    });

    it('dispatch group has dispatchArgsJson from Agent tool call', () => {
      const architectGroup = groups.find(g => g.skillName === 'ascendc-ops-architect');
      expect(architectGroup).toBeDefined();
      expect(architectGroup!.dispatchArgsJson).not.toBeNull();
      expect(architectGroup!.dispatchArgsJson).toContain('subagent_type');
      expect(architectGroup!.dispatchArgsJson).toContain('ascendc-ops-architect');
    });

    it('dispatch group dispatchEvent is correctly set', () => {
      const architectGroup = groups.find(g => g.skillName === 'ascendc-ops-architect');
      expect(architectGroup!.dispatchEvent).not.toBeNull();
      expect(architectGroup!.dispatchEvent!.eventType).toBe('dispatch');
      expect(architectGroup!.dispatchEvent!.skillName).toBe('ascendc-ops-architect');
    });

    it('invoke group has invokeResultJson from Skill tool call', () => {
      const invokeGroup = groups.find(g => g.skillName === 'ops-registry-invoke-workflow');
      expect(invokeGroup).toBeDefined();
      expect(invokeGroup!.invokeResultJson).not.toBeNull();
      expect(invokeGroup!.invokeResultJson).toContain('核心原则');
    });

    it('invoke group has no dispatch fields', () => {
      const invokeGroup = groups.find(g => g.skillName === 'ops-registry-invoke-workflow');
      expect(invokeGroup!.dispatchEvent).toBeNull();
      expect(invokeGroup!.dispatchArgsJson).toBeNull();
      expect(invokeGroup!.dispatchResultJson).toBeNull();
    });

    it('dispatch group has no invoke fields', () => {
      const architectGroup = groups.find(g => g.skillName === 'ascendc-ops-architect');
      expect(architectGroup!.invokeEvent).toBeNull();
      expect(architectGroup!.invokeResultJson).toBeNull();
    });

    it('parallel dispatches produce separate groups with their own results', () => {
      const designerGroup = groups.find(g => g.skillName === 'ascendc-ops-designer');
      const testerGroup = groups.find(g => g.skillName === 'ascendc-ops-tester');
      expect(designerGroup).toBeDefined();
      expect(testerGroup).toBeDefined();
      expect(designerGroup!.dispatchResultJson).toContain('设计准备任务完成');
      expect(testerGroup!.dispatchResultJson).toContain('测试设计任务完成');
    });

    it('general-purpose tool calls are excluded from skillToolCalls', () => {
      const generalTc = skillToolCalls.find(tc => tc.argsJson?.includes('general-purpose'));
      expect(generalTc).toBeUndefined();
    });
  });

  // ── groupSkillEvents: total group count ──

  describe('groupSkillEvents: total counts', () => {
    it('group count equals unique skillName count', () => {
      const uniqueNames = [...new Set(skillEvents.map(se => se.skillName))];
      expect(groups.length).toBe(uniqueNames.length);
      expect(groups.length).toBe(4);
    });

    it('all groups have allSuccess=true', () => {
      expect(groups.every(g => g.allSuccess)).toBe(true);
    });
  });
});
