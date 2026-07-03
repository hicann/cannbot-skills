// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { importSession } from '@/lib/ingest/data-service';
import { prisma } from '@/lib/db';

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId, framework } = body;

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    const where: Record<string, string> = { taskId };
    if (framework) where.framework = framework;

    const session = await prisma.session.findFirst({ where });

    if (!session) {
      return NextResponse.json({ error: `Session not found: "${taskId}"` }, { status: 404 });
    }

    if (!session.sourcePath) {
      return NextResponse.json({ error: 'No sourcePath stored for this session — cannot refresh' }, { status: 400 });
    }

    const sourceType = session.framework === 'opencode' ? 'opencode-db'
      : session.framework === 'claude-code' ? 'claude-jsonl'
      : session.framework;

    const existingTurnCount = await prisma.turn.count({ where: { sessionId: session.id } });

    const result = await importSession(session.sourcePath, taskId, prisma, session.sourcePath, sourceType);

    const newTurnCount = await prisma.turn.count({ where: { sessionId: session.id } });
    const addedTurns = newTurnCount - existingTurnCount;

    return NextResponse.json({
      success: true,
      sessionId: session.id,
      newTurnCount: addedTurns,
      totalTurnCount: newTurnCount,
      message: addedTurns > 0 ? `刷新完成，新增 ${addedTurns} 轮` : '刷新完成，无新增数据',
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
