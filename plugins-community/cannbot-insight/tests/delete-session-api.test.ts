// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import { prisma } from './setup';
import { DELETE } from '@/app/api/ingest/delete-session/route';

const createdSessionIds: string[] = [];

async function seedSession(taskId: string, framework: string = 'unknown') {
  const session = await prisma.session.create({ data: { taskId, framework } });
  createdSessionIds.push(session.id);
  return session;
}

afterEach(async () => {
  for (const id of createdSessionIds.splice(0)) {
    await prisma.session.delete({ where: { id } }).catch(() => {});
  }
});

function makeDeleteRequest(body: unknown) {
  return new NextRequest('http://localhost/api/ingest/delete-session', {
    method: 'DELETE',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

describe('delete-session API (batch)', () => {
  it('deletes multiple sessions by explicit (taskId, framework) list', async () => {
    const a = await seedSession(`batch-del-${Date.now()}-a`, 'opencode');
    const b = await seedSession(`batch-del-${Date.now()}-b`, 'claude-code');
    const c = await seedSession(`batch-del-${Date.now()}-c`, 'unknown');

    const res = await DELETE(makeDeleteRequest({
      sessions: [
        { taskId: a.taskId, framework: 'opencode' },
        { taskId: b.taskId, framework: 'claude-code' },
        { taskId: c.taskId, framework: 'unknown' },
      ],
    }));
    const data = await res.json();

    expect(res.status).toBe(200);
    expect(data.deleted).toBe(3);
    expect(data.batch).toBe(3);

    const remaining = await prisma.session.findMany({
      where: { id: { in: [a.id, b.id, c.id] } },
    });
    expect(remaining).toHaveLength(0);
    for (const id of [a.id, b.id, c.id]) {
      createdSessionIds.splice(createdSessionIds.indexOf(id), 1);
    }
  });

  it('scopes deletion by framework when taskId collides across frameworks', async () => {
    const stamp = `${Date.now()}`;
    const keep = await seedSession(`collide-${stamp}`, 'claude-code');
    const drop = await seedSession(`collide-${stamp}`, 'opencode');

    const res = await DELETE(makeDeleteRequest({
      sessions: [{ taskId: `collide-${stamp}`, framework: 'opencode' }],
    }));
    const data = await res.json();

    expect(res.status).toBe(200);
    expect(data.deleted).toBe(1);

    const survivor = await prisma.session.findUnique({ where: { id: keep.id } });
    expect(survivor).not.toBeNull();
    const gone = await prisma.session.findUnique({ where: { id: drop.id } });
    expect(gone).toBeNull();
    createdSessionIds.splice(createdSessionIds.indexOf(drop.id), 1);
  });

  it('skips list entries missing taskId but still deletes valid ones', async () => {
    const a = await seedSession(`partial-${Date.now()}`, 'opencode');

    const res = await DELETE(makeDeleteRequest({
      sessions: [
        { taskId: a.taskId, framework: 'opencode' },
        { framework: 'opencode' },
        { taskId: '', framework: 'unknown' },
      ],
    }));
    const data = await res.json();

    expect(res.status).toBe(200);
    expect(data.deleted).toBe(1);
    expect(data.batch).toBe(3);

    const gone = await prisma.session.findUnique({ where: { id: a.id } });
    expect(gone).toBeNull();
    createdSessionIds.splice(createdSessionIds.indexOf(a.id), 1);
  });

  it('returns deleted:0 for an empty batch list without touching existing rows', async () => {
    const res = await DELETE(makeDeleteRequest({ sessions: [] }));
    const data = await res.json();
    expect(res.status).toBe(200);
    expect(data.deleted).toBe(0);
    expect(data.batch).toBe(0);
  });

  it('still rejects a missing taskId in the single-session path', async () => {
    const res = await DELETE(makeDeleteRequest({}));
    expect(res.status).toBe(400);
  });
});
