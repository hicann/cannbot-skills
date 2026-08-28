// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prisma } from './setup';
import { readSession } from '../src/lib/ingest/adapters/claude-jsonl.ts';
import { splitIntoTurns, resetIdCounter } from '../src/lib/ingest/turn-split.ts';
import { readAvailableSkills, proxySourceOf } from '../src/lib/ingest/adapters/claude-jsonl-full-context.ts';
import {
  parseClaudeAvailableSkills,
  parseOpencodeAvailableSkills,
  buildCoverage,
  aggregateUsedSkills,
} from '../src/lib/skill-coverage.ts';
import { GET } from '@/app/api/observe/session/skill-coverage/route';

const FIXTURE = path.resolve(__dirname, 'data/claude-sessions/skill-coverage-session.jsonl');

const createdSessionIds: string[] = [];

afterEach(async () => {
  for (const id of createdSessionIds.splice(0)) {
    await prisma.session.delete({ where: { id } }).catch(() => {});
  }
});

describe('skill-coverage parsing', () => {
  it('parses claude available-skills list with origin', () => {
    const text =
      'The following skills are available for use with the Skill tool:\n\n' +
      '- alpha-workflow: 主工作流技能，负责流程编排。 (from plugins-official/alpha)\n' +
      '- beta-helper: 辅助工具技能。 (from skills/beta)\n' +
      '- no-origin-skill: 没有来源标注的技能。\n\n' +
      'Always use the Skill tool.';
    const list = parseClaudeAvailableSkills(text);
    expect(list.map(s => s.name)).toEqual(['alpha-workflow', 'beta-helper', 'no-origin-skill']);
    expect(list[0].description).toBe('主工作流技能，负责流程编排。');
    expect(list[0].origin).toBe('plugins-official/alpha');
    expect(list[2].origin).toBeNull();
  });

  it('parses opencode <available_skills> XML block', () => {
    const text =
      'Instructions from: AGENTS.md\n<available_skills>\n<skill>\n<name>cannbot-skill-review</name>\n' +
      '<description>检视 Skill 模块。</description>\n<location>.opencode/skills/cannbot-skill-review</location>\n</skill>\n' +
      '<skill>\n<name>plain</name>\n</skill>\n</available_skills>';
    const list = parseOpencodeAvailableSkills(text);
    expect(list).toHaveLength(2);
    expect(list[0]).toEqual({
      name: 'cannbot-skill-review',
      description: '检视 Skill 模块。',
      origin: '.opencode/skills/cannbot-skill-review',
    });
    expect(list[1]).toEqual({ name: 'plain', description: '', origin: null });
  });

  it('routes status by event type: invoke > load > dispatch > unused, and marks extras', () => {
    const available = [
      { name: 'a', description: '', origin: null },
      { name: 'b', description: '', origin: null },
      { name: 'c', description: '', origin: null },
      { name: 'd', description: '', origin: null },
    ];
    const used = aggregateUsedSkills([
      { skillName: 'a', eventType: 'invoke', success: true },
      { skillName: 'a', eventType: 'load', success: true },
      { skillName: 'b', eventType: 'load', success: true },
      { skillName: 'c', eventType: 'dispatch', success: true },
      { skillName: 'ghost', eventType: 'invoke', success: true },
    ]);
    const cov = buildCoverage(available, used);
    const byName = new Map(cov.items.map(i => [i.name, i.status]));
    expect(byName.get('a')).toBe('invoked');
    expect(byName.get('b')).toBe('loaded');
    expect(byName.get('c')).toBe('dispatched');
    expect(byName.get('d')).toBe('unused');
    expect(byName.get('ghost')).toBe('extra');
    expect(cov.stats).toEqual({
      availableTotal: 4, invoked: 1, loaded: 1, dispatched: 1, unused: 1, extra: 1,
    });
    // 未使用置顶：覆盖度看板核心是"哪些没用上"
    expect(cov.items[0].status).toBe('unused');
    expect(cov.items.map(i => i.status)).toEqual(['unused', 'invoked', 'loaded', 'dispatched', 'extra']);
  });

  it('degrades to used-only when available list is null', () => {
    const used = aggregateUsedSkills([{ skillName: 'a', eventType: 'invoke', success: true }]);
    const cov = buildCoverage(null, used);
    expect(cov.items).toHaveLength(1);
    expect(cov.items[0].status).toBe('invoked');
    expect(cov.stats.availableTotal).toBe(0);
  });
});

describe('skill-coverage integration (fixture → adapter → turn-split → coverage)', () => {
  let interactions: ReturnType<typeof readSession>;
  let result: ReturnType<typeof splitIntoTurns>;

  beforeAll(() => {
    resetIdCounter();
    interactions = readSession(FIXTURE, 'skill-coverage-session');
    result = splitIntoTurns(interactions, 'session-skill-coverage-test');
  });

  it('fixture produces invoke/load/dispatch skill events', () => {
    const types = result.skillEvents.map(se => `${se.skillName}:${se.eventType}`).sort();
    expect(types).toContain('alpha-workflow:invoke');
    expect(types).toContain('beta-helper:load');
    expect(types).toContain('delta-agent:dispatch');
  });

  it('readAvailableSkills extracts the claude-list full set from the fixture', () => {
    const parsed = readAvailableSkills(FIXTURE);
    expect(parsed).not.toBeNull();
    expect(parsed!.format).toBe('claude-list');
    expect(parsed!.skills.map(s => s.name)).toEqual([
      'alpha-workflow', 'beta-helper', 'gamma-unused', 'delta-agent',
    ]);
  });

  it('end-to-end coverage marks unused and routes statuses', () => {
    const parsed = readAvailableSkills(FIXTURE)!;
    const used = aggregateUsedSkills(
      result.skillEvents.map(se => ({ skillName: se.skillName, eventType: se.eventType, success: se.success }))
    );
    const cov = buildCoverage(parsed.skills, used);
    const byName = new Map(cov.items.map(i => [i.name, i.status]));
    expect(byName.get('alpha-workflow')).toBe('invoked');
    expect(byName.get('beta-helper')).toBe('loaded');
    expect(byName.get('delta-agent')).toBe('dispatched');
    expect(byName.get('gamma-unused')).toBe('unused');
    expect(cov.stats.availableTotal).toBe(4);
    expect(cov.stats.unused).toBe(1);
  });

  it('readAvailableSkills parses opencode-proxy capture format (system field XML)', () => {
    const tmp = path.join(os.tmpdir(), `cov-oc-proxy-${Date.now()}.jsonl`);
    const sysText =
      'Instructions from: AGENTS.md\n<available_skills>\n<skill>\n<name>oc-skill-a</name>\n' +
      '<description>描述A</description>\n</skill>\n<skill>\n<name>oc-skill-b</name>\n' +
      '<description>描述B</description>\n</skill>\n</available_skills>';
    fs.writeFileSync(
      tmp,
      JSON.stringify({ type: 'assistant', source: 'opencode-proxy', system: sysText, message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } }) + '\n'
    );
    const parsed = readAvailableSkills(tmp);
    expect(parsed).not.toBeNull();
    expect(parsed!.format).toBe('opencode-xml');
    expect(parsed!.skills.map(s => s.name)).toEqual(['oc-skill-a', 'oc-skill-b']);
    fs.unlinkSync(tmp);
  });
});

describe('proxySourceOf wire fingerprint fallback (early unmarked captures)', () => {
  function tmpFile(name: string, lines: object[]): string {
    const tmp = path.join(os.tmpdir(), `proxysrc-${name}-${Date.now()}.jsonl`);
    fs.writeFileSync(tmp, lines.map(l => JSON.stringify(l)).join('\n') + '\n');
    return tmp;
  }

  it('early opencode capture without source marker: detected via top-level system field, framework inferred from "You are opencode"', () => {
    const f = tmpFile('oc-unmarked', [
      { type: 'user', message: { role: 'user', content: 'go' } },
      { type: 'assistant', system: 'You are opencode, an interactive CLI tool that helps users with software engineering tasks.', tools: [{ name: 'bash' }], message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } },
    ]);
    expect(proxySourceOf(f)).toBe('opencode-proxy');
    fs.unlinkSync(f);
  });

  it('early claude capture without source marker: detected via top-level tools field', () => {
    const f = tmpFile('cc-unmarked', [
      { type: 'user', message: { role: 'user', content: 'go' } },
      { type: 'assistant', system: 'You are Claude Code, Anthropics official CLI for Claude.', tools: [{ name: 'Bash' }], message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } },
    ]);
    expect(proxySourceOf(f)).toBe('claude-proxy');
    fs.unlinkSync(f);
  });

  it('line-level source marker still wins and is returned verbatim', () => {
    const f = tmpFile('marked', [
      { type: 'user', message: { role: 'user', content: 'go' }, source: 'claude-proxy', tools: [{ name: 'Bash' }] },
    ]);
    expect(proxySourceOf(f)).toBe('claude-proxy');
    fs.unlinkSync(f);
  });

  it('native claude jsonl (no source marker, no system/tools fields) stays null', () => {
    const f = tmpFile('native', [
      { type: 'user', message: { role: 'user', content: 'go' }, parentUuid: null, uuid: 'u1', sessionId: 's1', cwd: '/x', version: '1.0.0', gitBranch: 'main' },
      { type: 'assistant', message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] }, uuid: 'a1' },
    ]);
    expect(proxySourceOf(f)).toBeNull();
    fs.unlinkSync(f);
  });
});

describe('skill-coverage API', () => {
  it('returns 400 when taskId missing', async () => {
    const res = await GET(new NextRequest('http://localhost/api/observe/session/skill-coverage'));
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown session', async () => {
    const res = await GET(
      new NextRequest('http://localhost/api/observe/session/skill-coverage?taskId=no-such')
    );
    expect(res.status).toBe(404);
  });

  it('hides for non-proxy sessions (opencode native: system prompt not persisted)', async () => {
    const taskId = `cov-oc-native-${Date.now()}`;
    const session = await prisma.session.create({
      data: { taskId, framework: 'opencode' },
    });
    createdSessionIds.push(session.id);
    const turn = await prisma.turn.create({ data: { sessionId: session.id, turnIndex: 0, role: 'assistant' } });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'oc-skill-a', eventType: 'invoke', success: true },
    });

    const res = await GET(
      new NextRequest(`http://localhost/api/observe/session/skill-coverage?taskId=${taskId}`)
    );
    expect(res.status).toBe(200);
    const d = await res.json();
    expect(d.hasAvailableList).toBe(false);
    expect(d.degradedReason).toContain('仅 proxy 捕获');
    expect(d.items).toHaveLength(1);
    expect(d.items[0].status).toBe('invoked');
  });

  it('hides for native claude-code sessions even when sourcePath has an injected list (proxy-only gate)', async () => {
    const taskId = `cov-claude-native-${Date.now()}`;
    const session = await prisma.session.create({
      data: { taskId, framework: 'claude-code', sourcePath: FIXTURE },
    });
    createdSessionIds.push(session.id);

    const res = await GET(
      new NextRequest(`http://localhost/api/observe/session/skill-coverage?taskId=${taskId}`)
    );
    expect(res.status).toBe(200);
    const d = await res.json();
    expect(d.hasAvailableList).toBe(false);
    expect(d.stats.availableTotal).toBe(0);
  });

  it('reports coverage against the claude-format capture for proxy sessions', async () => {
    const taskId = `cov-claude-${Date.now()}`;
    const session = await prisma.session.create({
      data: { taskId, framework: 'claude-code', version: 'claude-proxy', sourcePath: FIXTURE },
    });
    createdSessionIds.push(session.id);
    const turn = await prisma.turn.create({ data: { sessionId: session.id, turnIndex: 0, role: 'assistant' } });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'alpha-workflow', eventType: 'invoke', success: true },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'beta-helper', eventType: 'load', success: true },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'delta-agent', eventType: 'dispatch', success: true },
    });

    const res = await GET(
      new NextRequest(`http://localhost/api/observe/session/skill-coverage?taskId=${taskId}`)
    );
    expect(res.status).toBe(200);
    const d = await res.json();
    expect(d.hasAvailableList).toBe(true);
    expect(d.listFormat).toBe('claude-list');
    expect(d.stats.availableTotal).toBe(4);
    const byName = new Map<string, string>(d.items.map((i: { name: string; status: string }) => [i.name, i.status]));
    expect(byName.get('alpha-workflow')).toBe('invoked');
    expect(byName.get('beta-helper')).toBe('loaded');
    expect(byName.get('gamma-unused')).toBe('unused');
    expect(byName.get('delta-agent')).toBe('dispatched');
  });
});
