// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { splitWorkflow } from '@/lib/ingest/phase-split';
import type { TurnRow } from '@/lib/ingest/turn-split';
import type { InteractionBridgeRow } from '@/lib/ingest/bridge-builder';

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

    const dbTurns = await prisma.turn.findMany({
      where: { sessionId: session.id },
      orderBy: [{ turnIndex: 'asc' }],
      include: {
        toolCalls: {
          select: {
            id: true,
            toolCallId: true,
            toolName: true,
            argsJson: true,
            state: true,
            durationMs: true,
            dispatchBridgeId: true,
          },
        },
      },
    });

    const dbBridges = await prisma.interactionBridge.findMany({
      where: { sessionId: session.id },
    });

    const turnIds = dbTurns.map(t => t.id);
    const dbSkillEvents = turnIds.length > 0
      ? await prisma.skillEvent.findMany({ where: { turnId: { in: turnIds } } })
      : [];

    const skillEvents = dbSkillEvents.map(se => ({
      id: se.id,
      turnId: se.turnId,
      skillName: se.skillName,
      skillVersion: se.skillVersion,
      eventType: se.eventType,
      success: se.success,
      errorMessage: se.errorMessage,
      argsJson: se.argsJson,
      startedAt: se.startedAt?.toISOString() ?? null,
      completedAt: se.completedAt?.toISOString() ?? null,
      durationMs: se.durationMs,
    }));

    const turns: TurnRow[] = dbTurns.map(t => ({
      id: t.id,
      sessionId: t.sessionId,
      turnIndex: t.turnIndex,
      role: t.role,
      content: t.content,
      contentJson: t.contentJson,
      contentSummary: t.contentSummary,
      inputMessagesJson: null,
      inputMessagesCount: t.inputMessagesCount,
      inputMessagesTokens: t.inputMessagesTokens,
      contextWindowPct: t.contextWindowPct,
      agentName: t.agentName,
      subagentName: t.subagentName,
      subagentSessionId: t.subagentSessionId,
      subagentType: null,
      totalTokens: t.totalTokens,
      inputTokens: t.inputTokens,
      outputTokens: t.outputTokens,
      reasoningTokens: t.reasoningTokens,
      cacheReadTokens: t.cacheReadTokens,
      cacheWriteTokens: t.cacheWriteTokens,
      cost: 0,
      createdAt_ts: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
      completedAt: t.completedAt?.toISOString() ?? null,
      latencyMs: t.latencyMs,
      ttftMs: t.ttftMs,
      model: t.model,
      modelId: t.modelId,
      providerId: t.providerId,
      temperature: t.temperature,
      maxTokens: t.maxTokens,
      finishReason: t.finishReason,
      isSubagent: t.isSubagent,
      parentExecutionId: t.parentExecutionId,
    }));

    const bridges: InteractionBridgeRow[] = dbBridges.map(b => ({
      id: b.id,
      sessionId: b.sessionId,
      dispatchExecutionId: b.dispatchExecutionId,
      dispatchTurnId: b.dispatchTurnId,
      dispatchToolCallId: b.dispatchToolCallId,
      dispatchContent: b.dispatchContent,
      dispatchTimestamp: b.dispatchTimestamp?.toISOString() ?? null,
      responseExecutionId: b.responseExecutionId,
      responseTurnId: b.responseTurnId,
      responseContent: b.responseContent,
      responseTimestamp: b.responseTimestamp?.toISOString() ?? null,
      subagentSessionId: b.subagentSessionId,
      subagentType: b.subagentType,
      subagentName: b.subagentName,
      status: b.status,
      subagentTokens: b.subagentTokens,
      subagentLatencyMs: b.subagentLatencyMs,
    }));

    const workflow = splitWorkflow(turns, bridges, skillEvents, session.id);

    return NextResponse.json(workflow);
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
