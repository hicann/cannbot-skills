// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll } from 'vitest';
import { splitIntoTurns, resetIdCounter } from '../src/lib/ingest/turn-split.ts';
import { splitWorkflow } from '../src/lib/ingest/phase-split.ts';
import { resetSkillFamilyConfigCache } from '../src/lib/skill-family-config.ts';
import type { RawInteraction } from '../src/lib/shared/types.ts';

function makeTurn(role: string, turnIndex: number, content: string): RawInteraction {
  return {
    role,
    content,
    timestamp: new Date(2026, 0, 1, 0, 0, turnIndex).toISOString(),
    timeInfo: { created: Date.UTC(2026, 0, 1, 0, 0, turnIndex), completed: Date.UTC(2026, 0, 1, 0, 0, turnIndex + 1) },
    agent: null,
    subagent_name: null,
    subagent_session_id: null,
    subagent_type: null,
    tool_calls: null,
    usage: { total: 100, input: 50, output: 50, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: 0, inputMessagesTokens: 50 },
    model: 'test-model',
    modelID: null,
    providerID: null,
    latency: 1000,
    finish_reason: 'stop',
  };
}

function dispatchTurn(turnIndex: number, skills: string[]): RawInteraction {
  const tool_calls = skills.map((s, i) => ({
    toolCallId: `tc-${turnIndex}-${i}`,
    toolName: 'agent',
    argsJson: JSON.stringify({ subagent_type: s, description: s }),
    resultJson: '',
    state: 'ok',
  }));
  return { ...makeTurn('assistant', turnIndex, `dispatch ${skills.join(',')}`), tool_calls };
}

function invokeTurn(turnIndex: number, skill: string): RawInteraction {
  return {
    ...makeTurn('assistant', turnIndex, `invoke ${skill}`),
    tool_calls: [{
      toolCallId: `tc-${turnIndex}`,
      toolName: 'skill/invoke',
      argsJson: JSON.stringify({ skill }),
      resultJson: '',
      state: 'ok',
    }],
  };
}

const SKILL_TURNS: Array<[number, 'dispatch' | 'invoke', string | string[]]> = [
  [1, 'invoke', 'ops-registry-invoke-workflow'],
  [27, 'dispatch', 'ascendc-ops-architect'],
  [32, 'dispatch', 'ascendc-ops-design-reviewer'],
  [37, 'dispatch', 'ascendc-ops-designer'],
  [42, 'dispatch', ['ascendc-ops-design-overview', 'ascendc-ops-design-architecture', 'ascendc-ops-design-implementation', 'ascendc-ops-design-quality', 'ascendc-ops-design-plan']],
  [76, 'dispatch', ['ascendc-ops-design-implementation', 'ascendc-ops-design-plan']],
  [98, 'dispatch', 'ascendc-ops-designer'],
  [106, 'dispatch', 'ascendc-ops-tester'],
  [108, 'dispatch', 'ascendc-ops-design-reviewer'],
  [109, 'dispatch', 'ascendc-ops-tester'],
  [110, 'dispatch', 'ascendc-ops-design-reviewer'],
  [155, 'invoke', 'ops-spec-gen'],
  [187, 'invoke', 'ops-spec-gen'],
  [188, 'invoke', 'npu-arch'],
  [189, 'invoke', 'ascendc-env-check'],
  [190, 'invoke', 'ascendc-docs-gen'],
  [191, 'invoke', 'ascendc-docs-search'],
  [204, 'invoke', 'ascendc-st-design'],
  [205, 'invoke', 'ascendc-registry-invoke-template'],
  [299, 'invoke', 'npu-arch'],
  [301, 'invoke', 'ascendc-regbase-best-practice'],
  [418, 'invoke', 'ascendc-st-design'],
  [419, 'invoke', 'ascendc-registry-invoke-template'],
  [443, 'invoke', 'ops-registry-invoke-workflow'],
  [455, 'invoke', 'ascendc-env-check'],
];

const MAX_TURN = 460;

function buildInteractions(markers: Record<number, string> = {}): RawInteraction[] {
  const skillByIndex = new Map<number, RawInteraction>();
  for (const [idx, kind, skills] of SKILL_TURNS) {
    if (kind === 'dispatch') {
      const arr = Array.isArray(skills) ? skills : [skills as string];
      skillByIndex.set(idx, dispatchTurn(idx, arr));
    } else {
      skillByIndex.set(idx, invokeTurn(idx, skills as string));
    }
  }
  const out: RawInteraction[] = [];
  for (let i = 0; i <= MAX_TURN; i++) {
    if (markers[i]) out.push(makeTurn('assistant', i, markers[i]));
    else if (skillByIndex.has(i)) out.push(skillByIndex.get(i)!);
    else out.push(makeTurn(i % 2 === 0 ? 'user' : 'assistant', i, `turn ${i}`));
  }
  return out;
}

function stepNames(phase: { children: unknown[] }): string[] {
  const names: string[] = [];
  for (const c of phase.children as Array<Record<string, unknown>>) {
    if (c.type === 'step') names.push(c.stepName as string);
    if (c.type === 'parallel-group') for (const s of c.steps as Array<Record<string, unknown>>) names.push(s.stepName as string);
  }
  return names;
}

describe('splitWorkflow marker-driven phase division', () => {
  beforeAll(() => { resetSkillFamilyConfigCache(); });

  it('whole session is 阶段一 when only 阶段一 completion marker exists (no real 阶段二 start)', () => {
    resetIdCounter();
    const interactions = buildInteractions({ 114: '## 🏁 阶段一（需求与设计）完成 — Op\n完成，暂停于此。后续可继续执行阶段二（开发）。' });
    const { turns, skillEvents } = splitIntoTurns(interactions, 'session-one-phase');
    const wf = splitWorkflow(turns, [], skillEvents, 'session-one-phase');

    expect(wf.phases.length).toBe(1);
    expect(wf.phases[0].fullLabel).toBe('阶段一：需求与设计');
    const names = stepNames(wf.phases[0]);
    expect(names).toContain('ops-registry-invoke-workflow');
    expect(names).toContain('ascendc-ops-architect');
    expect(names).toContain('ascendc-docs-search');
    expect(names).toContain('ascendc-st-design');
    expect(names).toContain('ascendc-ops-tester');
  });

  it('does not split on future-tense 阶段二 mention ("后续可继续执行阶段二")', () => {
    resetIdCounter();
    const interactions = buildInteractions({ 114: '完成，暂停于此。后续可继续执行阶段二（开发）。' });
    const { turns, skillEvents } = splitIntoTurns(interactions, 'session-future');
    const wf = splitWorkflow(turns, [], skillEvents, 'session-future');
    expect(wf.phases.length).toBe(1);
    expect(wf.phases[0].fullLabel).toBe('阶段一：需求与设计');
  });

  it('splits into 阶段一 / 阶段二 on a real 阶段二 start marker', () => {
    resetIdCounter();
    const interactions = buildInteractions({ 120: '## 阶段二：开发\n开始执行阶段二' });
    const { turns, skillEvents } = splitIntoTurns(interactions, 'session-two-phase');
    const wf = splitWorkflow(turns, [], skillEvents, 'session-two-phase');

    expect(wf.phases.map(p => p.fullLabel)).toEqual(['阶段一：需求与设计', '阶段二：开发']);
    const p1 = wf.phases[0];
    const p2 = wf.phases[1];
    expect(stepNames(p1)).toContain('ascendc-ops-architect');
    expect(stepNames(p1)).toContain('ascendc-ops-designer');
    expect(stepNames(p2)).toContain('ascendc-docs-gen');
    expect(stepNames(p2)).toContain('ascendc-st-design');
    expect(p1.turnIndexEnd ?? 0).toBeLessThan(p2.turnIndexStart ?? 0);
  });

  it('groups same-turn parallel dispatches into a parallel-group (turn 42, 5 design skills)', () => {
    resetIdCounter();
    const interactions = buildInteractions({ 114: '## 阶段一（需求与设计）完成' });
    const { turns, skillEvents } = splitIntoTurns(interactions, 'session-pg');
    const wf = splitWorkflow(turns, [], skillEvents, 'session-pg');
    const groups = wf.phases[0].children.filter(c => c.type === 'parallel-group') as Array<{ steps: Array<{ stepName: string }> }>;
    const pg42 = groups.find(g => g.steps.length === 5);
    expect(pg42).toBeTruthy();
    expect(pg42!.steps.map(s => s.stepName).sort()).toEqual(
      ['ascendc-ops-design-architecture', 'ascendc-ops-design-implementation', 'ascendc-ops-design-overview', 'ascendc-ops-design-plan', 'ascendc-ops-design-quality'],
    );
  });

  it('falls back to a single Main phase when there are no skill events', () => {
    resetIdCounter();
    const interactions = [makeTurn('user', 0, 'hi'), makeTurn('assistant', 1, 'hello')];
    const { turns } = splitIntoTurns(interactions, 'session-no-skill');
    const wf = splitWorkflow(turns, [], [], 'session-no-skill');
    expect(wf.phases.length).toBe(1);
    expect(wf.phases[0].phaseName).toBe('Main');
  });

  it('falls back to family-gap heuristics when no phase markers exist', () => {
    resetIdCounter();
    const interactions: RawInteraction[] = [
      invokeTurn(0, 'foo-bar'),
      invokeTurn(1, 'foo-baz'),
      invokeTurn(2, 'qux-zap'),
    ];
    const { turns, skillEvents } = splitIntoTurns(interactions, 'session-unmapped');
    const wf = splitWorkflow(turns, [], skillEvents, 'session-unmapped');
    expect(wf.phases.length).toBeGreaterThanOrEqual(1);
    expect(wf.phases.every(p => p.phaseName !== 'Main')).toBe(true);
    expect(wf.phases.every(p => !p.fullLabel.startsWith('阶段'))).toBe(true);
  });
});
