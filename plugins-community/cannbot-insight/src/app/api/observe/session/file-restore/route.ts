// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { isDirectoryRead } from '@/lib/file-reads';
import {
  extractFilePath,
  restoreFile,
  type RestoreOp,
} from '@/lib/file-restore';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');
    const filePath = searchParams.get('filePath');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required query param: taskId' },
        { status: 400 }
      );
    }
    if (!filePath) {
      return NextResponse.json(
        { error: 'Missing required query param: filePath' },
        { status: 400 }
      );
    }

    const session = await prisma.session.findFirst({
      where: framework ? { taskId, framework } : { taskId },
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found for taskId: "${taskId}"` },
        { status: 404 }
      );
    }

    const toolCalls = await prisma.toolCall.findMany({
      where: {
        toolName: { in: ['read', 'Read', 'write', 'Write'] },
        turn: { sessionId: session.id },
      },
      select: {
        toolName: true,
        argsJson: true,
        resultJson: true,
        startedAt: true,
        completedAt: true,
        createdAt: true,
      },
    });

    const ops: RestoreOp[] = [];
    for (const tc of toolCalls) {
      const fp = extractFilePath(tc.argsJson);
      if (!fp || fp !== filePath) continue;

      const kind = tc.toolName === 'write' || tc.toolName === 'Write' ? 'write' : 'read';
      if (kind === 'read' && isDirectoryRead(fp, tc.resultJson)) continue;

      const ts = tc.startedAt ?? tc.completedAt ?? tc.createdAt ?? null;
      ops.push({
        kind,
        argsJson: tc.argsJson,
        resultJson: tc.resultJson,
        ts,
      });
    }

    const { lines, maxLine, opsUsed } = restoreFile(ops);

    return NextResponse.json({ path: filePath, lines, maxLine, opsUsed });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
