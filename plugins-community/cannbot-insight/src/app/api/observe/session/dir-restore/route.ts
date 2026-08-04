// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { isDirectoryRead } from '@/lib/file-reads';
import { extractFilePath, restoreFile, renderRestoredText, type RestoreOp } from '@/lib/file-restore';
import { buildZip } from '@/lib/zip-store';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required query param: taskId' },
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

    const byPath = new Map<string, RestoreOp[]>();

    for (const tc of toolCalls) {
      const fp = extractFilePath(tc.argsJson);
      if (!fp) continue;
      if (isDirectoryRead(fp, tc.resultJson)) continue;

      const kind = tc.toolName === 'write' || tc.toolName === 'Write' ? 'write' : 'read';
      const ts = tc.startedAt ?? tc.completedAt ?? tc.createdAt ?? null;
      const op: RestoreOp = {
        kind,
        argsJson: tc.argsJson,
        resultJson: tc.resultJson,
        ts,
      };
      const list = byPath.get(fp);
      if (list) list.push(op);
      else byPath.set(fp, [op]);
    }

    const encoder = new TextEncoder();
    const entries = [];
    let totalLines = 0;
    let gapLines = 0;
    for (const [path, ops] of byPath) {
      const { lines } = restoreFile(ops);
      totalLines += lines.length;
      gapLines += lines.filter(l => l.source === 'gap').length;
      entries.push({ path, data: encoder.encode(renderRestoredText(lines)) });
    }

    const blob = buildZip(entries);
    const buf = Buffer.from(await blob.arrayBuffer());

    return new NextResponse(buf, {
      status: 200,
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': `attachment; filename="session_${encodeURIComponent(taskId)}_restored.zip"`,
        'X-Restored-Count': String(entries.length),
        'X-Restored-Total-Lines': String(totalLines),
        'X-Restored-Gap-Lines': String(gapLines),
      },
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
