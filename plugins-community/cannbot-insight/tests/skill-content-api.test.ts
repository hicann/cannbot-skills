// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import { prisma } from './setup';
import { GET } from '@/app/api/observe/session/skill-content/route';

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

function makeRequest(taskId: string, skillName: string) {
  const q = new URLSearchParams({ taskId, skillName });
  return new NextRequest(`http://localhost/api/observe/session/skill-content?${q.toString()}`);
}

describe('skill-content API', () => {
  it('returns 400 when taskId missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/skill-content?skillName=x');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 400 when skillName missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/skill-content?taskId=x');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown session', async () => {
    const res = await GET(makeRequest('no-such-task', 'foo'));
    expect(res.status).toBe(404);
  });

  it('returns SKILL.md content from native Skill tool (preamble stripped)', async () => {
    const taskId = `skill-content-tool-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Skill',
        argsJson: JSON.stringify({ skill: 'foo' }),
        resultJson: 'Base directory for this skill: /x/foo\n\n# Foo\n\nreal content',
        isSkillRelated: true,
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, 'foo'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.skillName).toBe('foo');
    expect(json.source).toBe('skill-tool');
    expect(json.content).toBe('# Foo\n\nreal content');
    expect(json.length).toBe(json.content.length);
  });

  it('falls back to Read of SKILL.md when no Skill tool result', async () => {
    const taskId = `skill-content-read-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/a/b/foo/SKILL.md' }),
        resultJson: '1\t# Foo\n2\t\n3\tbody',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, 'foo'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.source).toBe('read');
    expect(json.content).toBe('# Foo\n\nbody');
  });

  it('returns content null when skill not found', async () => {
    const taskId = `skill-content-none-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Skill',
        argsJson: JSON.stringify({ skill: 'other' }),
        resultJson: 'Base directory for this skill: /x\n\n# Other',
        isSkillRelated: true,
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, 'foo'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.content).toBeNull();
    expect(json.source).toBeNull();
  });
});
