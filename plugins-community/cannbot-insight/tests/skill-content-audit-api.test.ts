// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import { prisma } from './setup';
import { GET } from '@/app/api/observe/session/skill-content-audit/route';

const createdSessionIds: string[] = [];

async function seedSession(taskId: string) {
  const session = await prisma.session.create({ data: { taskId, framework: 'unknown' } });
  createdSessionIds.push(session.id);
  const turn = await prisma.turn.create({
    data: { sessionId: session.id, turnIndex: 0, role: 'assistant' },
  });
  return { session, turn };
}

afterEach(async () => {
  for (const id of createdSessionIds.splice(0)) {
    await prisma.session.delete({ where: { id } }).catch(() => {});
  }
});

function makeRequest(taskId: string) {
  return new NextRequest(
    `http://localhost/api/observe/session/skill-content-audit?taskId=${encodeURIComponent(taskId)}`
  );
}

describe('skill-content-audit API', () => {
  it('returns 400 when taskId missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/skill-content-audit');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown session', async () => {
    const res = await GET(makeRequest('no-such-task'));
    expect(res.status).toBe(404);
  });

  it('marks a skill with Skill-tool result as hasContent, and a skill with no captured content as missing', async () => {
    const taskId = `audit-mix-${Date.now()}`;
    const { session, turn } = await seedSession(taskId);

    await prisma.sessionSkill.create({
      data: { sessionId: session.id, skillName: 'loaded', skillVersion: 1, invocationCount: 1 },
    });
    // 真实 ingest 里 Skill 工具调用会同时产生 invoke SkillEvent（SessionSkill 即由 SkillEvent 派生）
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'loaded', eventType: 'invoke', success: true },
    });
    await prisma.sessionSkill.create({
      data: { sessionId: session.id, skillName: 'bare', skillVersion: 1, invocationCount: 1 },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'bare', eventType: 'invoke', success: true },
    });

    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Skill',
        argsJson: JSON.stringify({ skill: 'loaded' }),
        resultJson: 'Base directory for this skill: /x/loaded\n\n# Loaded\n\nfull body',
        isSkillRelated: true,
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    const json = await res.json();
    const byName = new Map(json.items.map(i => [i.skillName, i]));

    const loaded = byName.get('loaded');
    expect(loaded.hasContent).toBe(true);
    expect(loaded.source).toBe('skill-tool');
    expect(loaded.length).toBeGreaterThan(0);
    expect(loaded.lines).toBe(3);
    expect(loaded.fullRead).toBe(true);
    expect(loaded.maxLine).toBeNull();

    const bare = byName.get('bare');
    expect(bare.hasContent).toBe(false);
    expect(bare.source).toBeNull();
    expect(bare.length).toBe(0);
    expect(bare.lines).toBe(0);
    expect(bare.fullRead).toBe(false);
  });

  it('includes skills that only appear in SkillEvent (not SessionSkill)', async () => {
    const taskId = `audit-event-${Date.now()}`;
    const { session, turn } = await seedSession(taskId);

    await prisma.skillEvent.create({
      data: {
        turnId: turn.id,
        skillName: 'event-only',
        skillVersion: 2,
        eventType: 'invoke',
        success: true,
      },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c2',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/a/b/event-only/SKILL.md' }),
        resultJson: '1\t# EventOnly\n2\t\n3\tbody',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    const json = await res.json();
    const item = json.items.find(i => i.skillName === 'event-only');
    expect(item).toBeTruthy();
    expect(item.hasContent).toBe(true);
    expect(item.source).toBe('read');
    expect(item.lines).toBe(3);
    expect(item.fullRead).toBe(true);
    expect(item.maxLine).toBe(3);
  });

  it('flags a Read with offset/limit as partial (not fullRead)', async () => {
    const taskId = `audit-partial-${Date.now()}`;
    const { session, turn } = await seedSession(taskId);

    await prisma.sessionSkill.create({
      data: { sessionId: session.id, skillName: 'partial', skillVersion: 1, invocationCount: 1 },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'partial', eventType: 'invoke', success: true },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c3',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/a/b/partial/SKILL.md', offset: 10, limit: 5 }),
        resultJson: '10\t# Partial\n11\t\n12\tbody',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    const json = await res.json();
    const item = json.items.find(i => i.skillName === 'partial');
    expect(item.hasContent).toBe(true);
    expect(item.fullRead).toBe(false);
    expect(item.maxLine).toBe(12);
  });

  it('excludes dispatch-only subagents (e.g. blackbox-designer) from the skill audit', async () => {
    const taskId = `audit-agent-${Date.now()}`;
    const { session, turn } = await seedSession(taskId);

    // blackbox-designer: SessionSkill 行存在，但只有 dispatch 事件 → 子代理，不应出现在 skill 审计
    await prisma.sessionSkill.create({
      data: { sessionId: session.id, skillName: 'blackbox-designer', skillVersion: null, invocationCount: 1 },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'blackbox-designer', eventType: 'dispatch', success: true },
    });
    // 真正的 skill：有 invoke + 全文
    await prisma.sessionSkill.create({
      data: { sessionId: session.id, skillName: 'real-skill', skillVersion: 1, invocationCount: 1 },
    });
    await prisma.skillEvent.create({
      data: { turnId: turn.id, skillName: 'real-skill', eventType: 'invoke', success: true },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c4',
        toolName: 'Skill',
        argsJson: JSON.stringify({ skill: 'real-skill' }),
        resultJson: 'Base directory for this skill: /x/real\n\n# Real\n\nbody',
        isSkillRelated: true,
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    const json = await res.json();
    const names = json.items.map(i => i.skillName);
    expect(names).not.toContain('blackbox-designer');
    expect(names).toContain('real-skill');
  });

  it('returns empty items for a session with no skills', async () => {
    const taskId = `audit-empty-${Date.now()}`;
    await seedSession(taskId);

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.items).toEqual([]);
  });
});
