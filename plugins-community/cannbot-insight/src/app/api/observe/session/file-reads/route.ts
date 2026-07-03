// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { parseReadArgs, isDirectoryRead, computeRange, analyzeReads } from '@/lib/file-reads';
import type { ReadEntry } from '@/lib/file-reads';

interface FileAnalysis {
  path: string;
  displayPath: string;
  reads: ReadEntry[];
  totalReads: number;
  overlappingReads: number;
  totalLinesRead: number;
  uniqueLinesRead: number;
  redundancyRate: number;
}

interface Summary {
  totalFiles: number;
  totalReads: number;
  filesWithOverlap: number;
  redundancyRate: number;
}

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
        toolName: { in: ['read', 'Read'] },
        turn: { sessionId: session.id },
      },
      select: {
        argsJson: true,
        resultJson: true,
        turn: {
          select: {
            id: true,
            turnIndex: true,
            agentName: true,
            isSubagent: true,
            subagentName: true,
            subagentSessionId: true,
            contentSummary: true,
          },
        },
      },
    });

    const subSessionIds = [...new Set(
      toolCalls
        .filter(tc => tc.turn.isSubagent && tc.turn.subagentSessionId)
        .map(tc => tc.turn.subagentSessionId!)
    )];

    const promptMap = new Map<string, string | null>();
    if (subSessionIds.length > 0) {
      const bridges = await prisma.interactionBridge.findMany({
        where: { sessionId: session.id, subagentSessionId: { in: subSessionIds } },
        select: { subagentSessionId: true, dispatchContent: true },
      });
      for (const b of bridges) {
        if (b.subagentSessionId) promptMap.set(b.subagentSessionId, b.dispatchContent);
      }
    }

    const fileMap = new Map<string, { displayPath: string; reads: ReadEntry[] }>();

    for (const tc of toolCalls) {
      const args = parseReadArgs(tc.argsJson);
      if (!args || !args.filePath) continue;
      if (isDirectoryRead(args.filePath, tc.resultJson)) continue;

      const filePath = args.filePath;
      const displayPath = args.summary || filePath;
      const range = computeRange(args);

      const agent = tc.turn.isSubagent
        ? (tc.turn.agentName || tc.turn.subagentName || 'subagent')
        : (tc.turn.agentName || 'root');

      const prompt = tc.turn.isSubagent && tc.turn.subagentSessionId
        ? (promptMap.get(tc.turn.subagentSessionId) ?? null)
        : null;

      if (!fileMap.has(filePath)) {
        fileMap.set(filePath, { displayPath, reads: [] });
      }
      fileMap.get(filePath)!.reads.push({
        turnId: tc.turn.id,
        turnIndex: tc.turn.turnIndex,
        agent,
        prompt,
        subagentSessionId: tc.turn.subagentSessionId ?? null,
        llmOutput: tc.turn.contentSummary ?? null,
        range,
      });
    }

    const files: FileAnalysis[] = [];
    let globalTotalLines = 0;
    let globalUniqueLines = 0;

    for (const [path, data] of fileMap) {
      const metrics = analyzeReads(data.reads);
      globalTotalLines += metrics.totalLinesRead;
      globalUniqueLines += metrics.uniqueLinesRead;

      files.push({
        path,
        displayPath: data.displayPath,
        reads: data.reads,
        totalReads: data.reads.length,
        ...metrics,
      });
    }

    files.sort((a, b) => b.totalReads - a.totalReads);

    const summary: Summary = {
      totalFiles: files.length,
      totalReads: files.reduce((sum, f) => sum + f.totalReads, 0),
      filesWithOverlap: files.filter(f => f.overlappingReads > 0).length,
      redundancyRate: globalTotalLines > 0 ? 1 - globalUniqueLines / globalTotalLines : 0,
    };

    return NextResponse.json({ files, summary });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
