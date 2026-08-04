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
import { GET } from '@/app/api/observe/session/file-restore/route';

const createdSessionIds: string[] = [];

async function seedSession(taskId: string) {
  const session = await prisma.session.create({
    data: { taskId, framework: 'unknown' },
  });
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

function makeRequest(taskId: string, filePath: string) {
  const q = new URLSearchParams({ taskId, filePath });
  return new NextRequest(`http://localhost/api/observe/session/file-restore?${q.toString()}`);
}

describe('file-restore API', () => {
  it('returns 400 when taskId missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/file-restore?filePath=/a');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 400 when filePath missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/file-restore?taskId=x');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown session', async () => {
    const res = await GET(makeRequest('no-such-task', '/a'));
    expect(res.status).toBe(404);
  });

  it('last write wins: read(t1) then write(t2) → write content', async () => {
    const taskId = `restore-wins-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    const t1 = new Date('2026-01-01T00:00:00Z');
    const t2 = new Date('2026-01-02T00:00:00Z');
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/docs/STATE.md', offset: 1 }),
        resultJson: '1\told1\n2\told2',
        startedAt: t1,
      },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c2',
        toolName: 'Write',
        argsJson: JSON.stringify({ file_path: '/docs/STATE.md', content: 'new1\nnew2' }),
        startedAt: t2,
      },
    });

    const res = await GET(makeRequest(taskId, '/docs/STATE.md'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.path).toBe('/docs/STATE.md');
    expect(json.maxLine).toBe(2);
    expect(json.lines[0]).toMatchObject({ n: 1, content: 'new1', source: 'write' });
    expect(json.lines[1]).toMatchObject({ n: 2, content: 'new2', source: 'write' });
  });

  it('marks uncovered lines as per-line gaps', async () => {
    const taskId = `restore-gap-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/gaps.txt' }),
        resultJson: '1\talpha\n5\tbeta',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, '/gaps.txt'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.maxLine).toBe(5);
    expect(json.lines[0]).toMatchObject({ n: 1, content: 'alpha' });
    expect(json.lines[1]).toMatchObject({ n: 2, content: null, source: 'gap' });
    expect(json.lines[2]).toMatchObject({ n: 3, content: null, source: 'gap' });
    expect(json.lines[3]).toMatchObject({ n: 4, content: null, source: 'gap' });
    expect(json.lines[4]).toMatchObject({ n: 5, content: 'beta' });
  });

  it('filters by filePath: only the requested path is restored', async () => {
    const taskId = `restore-filter-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/a.txt' }),
        resultJson: '1\tA',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c2',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/b.txt' }),
        resultJson: '1\tB',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, '/a.txt'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.path).toBe('/a.txt');
    expect(json.lines[0]).toMatchObject({ n: 1, content: 'A' });
  });

  it('skips directory reads', async () => {
    const taskId = `restore-dir-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/src/' }),
        resultJson: '<type>directory</type>',
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, '/src/'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.maxLine).toBe(0);
    expect(json.lines).toEqual([]);
  });

  it('parses opencode envelope format (<content> + N: colon) end-to-end', async () => {
    const taskId = `restore-opencode-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    const opencodeResult = [
      '<path>/docs/STATE.md</path>',
      '<type>file</type>',
      '<content>',
      '1: # Title',
      '2: ',
      '3: body line',
      '</content>',
    ].join('\n');
    await prisma.toolCall.create({
      data: {
        turnId: turn.id,
        toolCallId: 'c1',
        toolName: 'read',
        argsJson: JSON.stringify({ file_path: '/docs/STATE.md' }),
        resultJson: opencodeResult,
        startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId, '/docs/STATE.md'));
    expect(res.status).toBe(200);
    const json = await res.json();
    expect(json.maxLine).toBe(3);
    expect(json.lines[0]).toMatchObject({ n: 1, content: '# Title', source: 'read' });
    expect(json.lines[1]).toMatchObject({ n: 2, content: '', source: 'read' });
    expect(json.lines[2]).toMatchObject({ n: 3, content: 'body line', source: 'read' });
  });
});
