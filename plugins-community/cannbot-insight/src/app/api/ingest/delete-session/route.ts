// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

// Delete a session and ALL its child rows. SQLite foreign-key cascade is not
// reliably enforced (PRAGMA state varies), so child rows are removed explicitly
// in dependency order inside a transaction. Without this, turns/toolCalls/etc.
// orphan and accumulate across delete cycles.
async function cascadeDeleteSession(id: string): Promise<void> {
  await prisma.$transaction([
    prisma.executionSkill.deleteMany({ where: { execution: { sessionId: id } } }),
    prisma.toolCall.deleteMany({ where: { turn: { sessionId: id } } }),
    prisma.skillEvent.deleteMany({ where: { turn: { sessionId: id } } }),
    prisma.execution.deleteMany({ where: { sessionId: id } }),
    prisma.sessionSkill.deleteMany({ where: { sessionId: id } }),
    prisma.interactionBridge.deleteMany({ where: { sessionId: id } }),
    prisma.turn.deleteMany({ where: { sessionId: id } }),
    prisma.session.delete({ where: { id } }),
  ]);
}

export async function DELETE(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId, framework, deleteAll, sessions: batchSessions } = body;

    if (deleteAll) {
      const count = await prisma.session.count();
      // Explicit cascade: delete child rows in dependency order inside a
      // transaction. SQLite foreign-key cascade is not reliably enforced
      // (PRAGMA state varies), so without this child rows orphan and accumulate.
      await prisma.$transaction([
        prisma.executionSkill.deleteMany({}),
        prisma.toolCall.deleteMany({}),
        prisma.skillEvent.deleteMany({}),
        prisma.execution.deleteMany({}),
        prisma.sessionSkill.deleteMany({}),
        prisma.interactionBridge.deleteMany({}),
        prisma.turn.deleteMany({}),
        prisma.session.deleteMany({}),
      ]);
      return NextResponse.json({ deleted: count });
    }

    if (Array.isArray(batchSessions)) {
      let deleted = 0;
      for (const entry of batchSessions) {
        if (!entry || typeof entry.taskId !== 'string') continue;
        const where: Record<string, string> = { taskId: entry.taskId };
        if (typeof entry.framework === 'string') where.framework = entry.framework;
        const found = await prisma.session.findMany({ where });
        for (const s of found) {
          await cascadeDeleteSession(s.id);
          deleted++;
        }
      }
      return NextResponse.json({ deleted, batch: batchSessions.length });
    }

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    const where: Record<string, string> = { taskId };
    if (framework) where.framework = framework;

    const sessions = await prisma.session.findMany({ where });

    if (sessions.length === 0) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 });
    }

    for (const s of sessions) {
      await cascadeDeleteSession(s.id);
    }
    return NextResponse.json({ deleted: sessions.length, taskId });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
