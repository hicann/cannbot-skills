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
import { GET } from '@/app/api/observe/session/dir-restore/route';

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
  return new NextRequest(`http://localhost/api/observe/session/dir-restore?taskId=${encodeURIComponent(taskId)}`);
}

function findBytes(hay: Uint8Array, needle: number[]): number {
  outer: for (let i = 0; i <= hay.length - needle.length; i++) {
    for (let j = 0; j < needle.length; j++) if (hay[i + j] !== needle[j]) continue outer;
    return i;
  }
  return -1;
}

const PK_LOCAL = [0x50, 0x4b, 0x03, 0x04];
const PK_END = [0x50, 0x4b, 0x05, 0x06];

describe('dir-restore API (zip download)', () => {
  it('returns 400 when taskId missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/dir-restore');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown session', async () => {
    const res = await GET(makeRequest('no-such-task'));
    expect(res.status).toBe(404);
  });

  it('returns a valid zip with correct headers and embedded file paths', async () => {
    const taskId = `dir-restore-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id, toolCallId: 'a1', toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/proj/docs/A.md' }),
        resultJson: '1\talpha\n2\tbeta', startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });
    await prisma.toolCall.create({
      data: {
        turnId: turn.id, toolCallId: 'b1', toolName: 'Write',
        argsJson: JSON.stringify({ file_path: '/proj/docs/B.md', content: 'new\nsecond' }),
        startedAt: new Date('2026-01-02T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    expect(res.headers.get('Content-Type')).toBe('application/zip');
    expect(res.headers.get('Content-Disposition')).toContain('session_');
    expect(res.headers.get('X-Restored-Count')).toBe('2');

    const arr = new Uint8Array(await res.arrayBuffer());
    // valid zip: local header at start, EOCD present
    expect(findBytes(arr, PK_LOCAL)).toBe(0);
    expect(findBytes(arr, PK_END)).toBeGreaterThan(0);
    // both file paths embedded (normalized: leading slash stripped)
    const enc = (s: string) => Array.from(new TextEncoder().encode(s));
    expect(findBytes(arr, enc('proj/docs/A.md'))).toBeGreaterThan(0);
    expect(findBytes(arr, enc('proj/docs/B.md'))).toBeGreaterThan(0);
    // last-write-wins content for B
    expect(findBytes(arr, enc('new'))).toBeGreaterThan(0);
  });

  it('skips directory reads', async () => {
    const taskId = `dir-restore-skip-${Date.now()}`;
    const { turn } = await seedSession(taskId);
    await prisma.toolCall.create({
      data: {
        turnId: turn.id, toolCallId: 'd1', toolName: 'Read',
        argsJson: JSON.stringify({ file_path: '/proj/src/' }),
        resultJson: '<type>directory</type>', startedAt: new Date('2026-01-01T00:00:00Z'),
      },
    });

    const res = await GET(makeRequest(taskId));
    expect(res.status).toBe(200);
    expect(res.headers.get('X-Restored-Count')).toBe('0');
    const arr = new Uint8Array(await res.arrayBuffer());
    expect(findBytes(arr, PK_END)).toBeGreaterThan(-1);
  });
});
