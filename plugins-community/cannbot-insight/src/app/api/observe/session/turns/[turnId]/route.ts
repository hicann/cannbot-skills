// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { getContextWindowLimit } from '@/lib/context-window-config';
import { selectInputContextTurns } from '@/lib/ingest/input-reconstruct';

// Compute stable system overhead from the first assistant turn
// For root turns: use root session; for subagent turns: use subagent's own turns
async function computeSystemOverhead(sessionId: string, subagentSessionId: string | null): Promise<number> {
  const where = subagentSessionId
    ? { sessionId, role: 'assistant', subagentSessionId }
    : { sessionId, role: 'assistant', isSubagent: false };
  const firstAssistant = await prisma.turn.findFirst({
    where,
    orderBy: [{ turnIndex: 'asc' }],
    select: { id: true, turnIndex: true, inputMessagesTokens: true },
  });
  if (!firstAssistant || firstAssistant.inputMessagesTokens === 0) return 0;

  const priorWhere = subagentSessionId
    ? { sessionId, turnIndex: { lt: firstAssistant.turnIndex }, role: { in: ['user', 'assistant', 'system', 'tool_result'] }, subagentSessionId }
    : { sessionId, turnIndex: { lt: firstAssistant.turnIndex }, role: { in: ['user', 'assistant', 'system', 'tool_result'] }, isSubagent: false };
  const priorMessages = await prisma.turn.findMany({
    where: priorWhere,
    select: { id: true, role: true, content: true },
  });

  // Include tool call args tokens for prior assistant turns
  const priorAssistantIds = priorMessages.filter(ct => ct.role === 'assistant').map(ct => ct.id);
  const priorToolCalls = priorAssistantIds.length > 0 ? await prisma.toolCall.findMany({
    where: { turnId: { in: priorAssistantIds } },
    select: { turnId: true, argsJson: true },
  }) : [];
  const toolArgsTokens = priorToolCalls.reduce((s, tc) => s + Math.round((tc.argsJson?.length ?? 0) / 3.5), 0);

  const visibleEstimated = priorMessages.reduce((s, ct) => s + Math.round((ct.content?.length ?? 0) / 3.5), 0) + toolArgsTokens;
  return Math.max(0, firstAssistant.inputMessagesTokens - visibleEstimated);
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ turnId: string }> }
) {
  try {
    const { turnId } = await params;

    const turn = await prisma.turn.findUnique({
      where: { id: turnId },
      include: {
        toolCalls: true,
        skillEvents: true,
      },
    });

    if (!turn) {
      return NextResponse.json(
        { error: `Turn not found: "${turnId}"` },
        { status: 404 }
      );
    }

    // Reconstruct inputMessagesJson if not stored (import optimization: not stored during import)
    let inputMessagesJson = turn.inputMessagesJson;
    if (!inputMessagesJson && turn.role === 'assistant') {
      // For subagent turns, only include prior turns in the same subagent session
      // For root turns, only include prior root turns (skip subagent turns)
      const prevWhere = turn.isSubagent && turn.subagentSessionId
        ? {
            sessionId: turn.sessionId,
            turnIndex: { lt: turn.turnIndex },
            role: { in: ['user', 'assistant', 'system', 'tool_result'] },
            subagentSessionId: turn.subagentSessionId,
          }
        : {
            sessionId: turn.sessionId,
            turnIndex: { lt: turn.turnIndex },
            role: { in: ['user', 'assistant', 'system', 'tool_result'] },
            isSubagent: false,
          };
      const previousTurns = await prisma.turn.findMany({
        where: prevWhere,
        orderBy: [{ turnIndex: 'asc' }],
        select: { id: true, turnIndex: true, role: true, content: true },
      });

      // Fetch tool calls (args + result) for prior assistant turns
      const assistantIds = previousTurns.filter(ct => ct.role === 'assistant').map(ct => ct.id);
      const priorToolCalls = assistantIds.length > 0 ? await prisma.toolCall.findMany({
        where: { turnId: { in: assistantIds } },
        select: { turnId: true, toolCallId: true, toolName: true, argsJson: true, resultJson: true, isSkillRelated: true },
        orderBy: [{ id: 'asc' }],
      }) : [];
      // Map assistant turnId → tool calls
      const toolCallsByTurnId = new Map<string, typeof priorToolCalls>();
      for (const tc of priorToolCalls) {
        const arr = toolCallsByTurnId.get(tc.turnId) ?? [];
        arr.push(tc);
        toolCallsByTurnId.set(tc.turnId, arr);
      }

      // Build ordered message list: assistant messages include tool_calls, tool_results keep their content.
      // Apply compact-aware windowing: start at the most recent /compact continuation
      // before this turn and skip local CLI command noise. See input-reconstruct.
      const filtered = selectInputContextTurns(previousTurns, turn.turnIndex);
      const messages: Array<{ role: string; content: string | null; tokenCount: number; tool_calls?: Array<{ name: string; args: string | null; result: string | null; isSkillRelated?: boolean }> }> = [];
      for (const ct of filtered) {
        const contentLen = ct.content?.length ?? 0;
        const baseTokens = Math.round(contentLen / 3.5);

        if (ct.role === 'assistant') {
          const tcs = toolCallsByTurnId.get(ct.id) ?? [];
          const argsTokens = tcs.reduce((s, tc) => s + Math.round((tc.argsJson?.length ?? 0) / 3.5), 0);
          const msg: typeof messages[0] = {
            role: ct.role,
            content: ct.content ?? null,
            tokenCount: baseTokens + argsTokens,
          };
          if (tcs.length > 0) {
            msg.tool_calls = tcs.map(tc => {
              const isSkill = tc.isSkillRelated;
              const argsMax = isSkill ? 2000 : 1500;
              const resultMax = isSkill ? 5000 : 3000;
              return {
                name: tc.toolName,
                args: tc.argsJson ? (tc.argsJson.length > argsMax ? tc.argsJson.substring(0, argsMax) + '...' : tc.argsJson) : null,
                result: tc.resultJson ? (tc.resultJson.length > resultMax ? tc.resultJson.substring(0, resultMax) + '...' : tc.resultJson) : null,
                isSkillRelated: isSkill ? true : undefined,
              };
            });
          }
          messages.push(msg);
        } else {
          messages.push({
            role: ct.role,
            content: ct.content ?? null,
            tokenCount: baseTokens,
          });
        }
      }

      inputMessagesJson = JSON.stringify(messages);
    }

    // Compute stable system overhead (fixed for entire session)
    const systemOverheadTokens = await computeSystemOverhead(turn.sessionId, turn.subagentSessionId);

    return NextResponse.json({
      turnId: turn.id,
      sessionId: turn.sessionId,
      turnIndex: turn.turnIndex,
      role: turn.role,
      content: turn.content,
      contentJson: turn.contentJson,
      contentSummary: turn.contentSummary ?? turn.content?.substring(0, 200) ?? null,
      inputMessagesJson,
      inputMessagesCount: turn.inputMessagesCount,
      inputMessagesTokens: turn.inputMessagesTokens,
      contextWindowPct: turn.contextWindowPct,
      systemOverheadTokens,
      agentName: turn.agentName,
      subagentName: turn.subagentName,
      subagentSessionId: turn.subagentSessionId,
      isSubagent: turn.isSubagent,
      totalTokens: turn.totalTokens,
      inputTokens: turn.inputTokens,
      outputTokens: turn.outputTokens,
      reasoningTokens: turn.reasoningTokens,
      cacheReadTokens: turn.cacheReadTokens,
      cacheWriteTokens: turn.cacheWriteTokens,
      latencyMs: turn.latencyMs,
      ttftMs: turn.ttftMs,
      createdAt: turn.createdAt_ts?.toISOString() ?? turn.createdAt.toISOString(),
      completedAt: turn.completedAt?.toISOString() ?? null,
      model: turn.model,
      modelId: turn.modelId,
      providerId: turn.providerId,
      contextWindowLimit: getContextWindowLimit(turn.model),
      finishReason: turn.finishReason,
      toolCalls: turn.toolCalls.map(tc => ({
        id: tc.id,
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        argsJson: tc.argsJson,
        resultJson: tc.resultJson,
        state: tc.state,
        errorType: tc.errorType,
        errorMessage: tc.errorMessage,
        startedAt: tc.startedAt?.toISOString() ?? null,
        completedAt: tc.completedAt?.toISOString() ?? null,
        durationMs: tc.durationMs,
        dispatchBridgeId: tc.dispatchBridgeId,
        isSkillRelated: tc.isSkillRelated,
      })),
      skillEvents: (() => {
        const skillToolCalls = turn.toolCalls.filter(tc => tc.isSkillRelated)
        return turn.skillEvents.map((se, idx) => {
          const matchedTc = skillToolCalls[idx]
          const resultError = matchedTc?.resultJson && (matchedTc.resultJson.includes('<tool_use_error>') || matchedTc.resultJson.includes('Exit code'))
          return {
            id: se.id,
            skillName: se.skillName,
            skillVersion: se.skillVersion,
            eventType: se.eventType,
            success: resultError ? false : se.success,
            errorMessage: resultError ? matchedTc!.resultJson!.substring(0, 200) : se.errorMessage,
            argsJson: se.argsJson,
            startedAt: se.startedAt?.toISOString() ?? null,
            completedAt: se.completedAt?.toISOString() ?? null,
            durationMs: se.durationMs,
          }
        })
      })(),
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
