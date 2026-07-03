// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

function extractMatchContext(text: string, keyword: string, contextRadius: number = 60): string {
  const lowerText = text.toLowerCase();
  const lowerKeyword = keyword.toLowerCase();
  const idx = lowerText.indexOf(lowerKeyword);
  if (idx === -1) return text.substring(0, contextRadius * 2);
  const start = Math.max(0, idx - contextRadius);
  const end = Math.min(text.length, idx + keyword.length + contextRadius);
  let context = text.substring(start, end);
  if (start > 0) context = '...' + context;
  if (end < text.length) context = context + '...';
  return context;
}

interface SearchItem {
  turnId: string;
  turnIndex: number;
  role: string;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  contentSummary: string | null;
  matchContext: string;
  matchField: 'content' | 'contentSummary' | 'toolResult' | 'toolError';
  toolName?: string;
  createdAt: string;
  hasDispatchBridge: boolean;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const keyword = searchParams.get('keyword');
    const framework = searchParams.get('framework');

    if (!taskId) {
      return NextResponse.json({ error: 'Missing required param: taskId' }, { status: 400 });
    }
    if (!keyword || keyword.trim().length === 0) {
      return NextResponse.json({ error: 'Missing required param: keyword' }, { status: 400 });
    }

    const session = await prisma.session.findFirst({ where: framework ? { taskId, framework } : { taskId } });
    if (!session) {
      return NextResponse.json({ error: `Session not found: taskId=${taskId}` }, { status: 404 });
    }

    const turns = await prisma.turn.findMany({
      where: { sessionId: session.id },
      orderBy: [{ turnIndex: 'asc' }],
      include: {
        toolCalls: {
          select: { toolCallId: true, toolName: true, dispatchBridgeId: true, resultJson: true, errorMessage: true },
        },
      },
    });

    const normalizedKeyword = keyword.trim().toLowerCase();
    const seenTurnIds = new Set<string>();
    const items: SearchItem[] = [];

    for (const t of turns) {
      const contentLower = (t.content ?? '').toLowerCase();
      const summaryLower = (t.contentSummary ?? '').toLowerCase();
      const contentMatch = contentLower.includes(normalizedKeyword);
      const summaryMatch = summaryLower.includes(normalizedKeyword);

      if (contentMatch) {
        seenTurnIds.add(t.id);
        items.push({
          turnId: t.id,
          turnIndex: t.turnIndex,
          role: t.role,
          agentName: t.agentName,
          isSubagent: t.isSubagent,
          subagentName: t.subagentName,
          subagentSessionId: t.subagentSessionId,
          contentSummary: t.contentSummary ?? t.content?.substring(0, 200) ?? null,
          matchContext: extractMatchContext(t.content ?? '', keyword.trim()),
          matchField: 'content',
          createdAt: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
          hasDispatchBridge: t.toolCalls.some(tc => tc.dispatchBridgeId !== null && tc.dispatchBridgeId !== undefined),
        });
      } else if (summaryMatch) {
        seenTurnIds.add(t.id);
        items.push({
          turnId: t.id,
          turnIndex: t.turnIndex,
          role: t.role,
          agentName: t.agentName,
          isSubagent: t.isSubagent,
          subagentName: t.subagentName,
          subagentSessionId: t.subagentSessionId,
          contentSummary: t.contentSummary ?? t.content?.substring(0, 200) ?? null,
          matchContext: extractMatchContext(t.contentSummary ?? t.content?.substring(0, 200) ?? '', keyword.trim()),
          matchField: 'contentSummary',
          createdAt: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
          hasDispatchBridge: t.toolCalls.some(tc => tc.dispatchBridgeId !== null && tc.dispatchBridgeId !== undefined),
        });
      }

      for (const tc of t.toolCalls) {
        const resultLower = (tc.resultJson ?? '').toLowerCase();
        const errorLower = (tc.errorMessage ?? '').toLowerCase();
        const resultMatch = resultLower.includes(normalizedKeyword);
        const errorMatch = errorLower.includes(normalizedKeyword);

        if (resultMatch || errorMatch) {
          if (seenTurnIds.has(t.id) && items.some(it => it.turnId === t.id && (it.matchField === 'content' || it.matchField === 'contentSummary'))) continue;
          seenTurnIds.add(t.id);
          const matchText = resultMatch ? tc.resultJson ?? '' : tc.errorMessage ?? '';
          const field: 'toolResult' | 'toolError' = resultMatch ? 'toolResult' : 'toolError';
          items.push({
            turnId: t.id,
            turnIndex: t.turnIndex,
            role: t.role,
            agentName: t.agentName,
            isSubagent: t.isSubagent,
            subagentName: t.subagentName,
            subagentSessionId: t.subagentSessionId,
            contentSummary: t.contentSummary ?? t.content?.substring(0, 200) ?? null,
            matchContext: `🔧 ${tc.toolName}: ${extractMatchContext(matchText, keyword.trim())}`,
            matchField: field,
            toolName: tc.toolName,
            createdAt: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
            hasDispatchBridge: tc.dispatchBridgeId !== null && tc.dispatchBridgeId !== undefined,
          });
        }
      }
    }

    return NextResponse.json({ items, total: items.length });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
