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

type ToolCallDetail = {
  id: string;
  toolCallId: string;
  toolName: string;
  state: string;
  durationMs: number;
  argsJson?: string | null;
  resultJson?: string | null;
  errorType?: string | null;
  errorMessage?: string | null;
  isSkillRelated?: boolean;
};

type SkillEventDetail = {
  id: string;
  skillName: string;
  eventType: string;
  success: boolean;
  skillVersion?: number | null;
  errorMessage?: string | null;
  argsJson?: string | null;
  durationMs?: number;
};

// Compute stable system overhead from the first assistant turn
// For root: find first non-subagent assistant; for subagent: find first in that subagentSessionId
async function computeSystemOverhead(sessionId: string, subagentSessionId: string | null = null): Promise<number> {
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

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');
    const isSubagent = searchParams.get('isSubagent');
    const role = searchParams.get('role');
    const subagentSessionId = searchParams.get('subagentSessionId');
    const includeContent = searchParams.get('includeContent') === 'true';
    const includeDetail = searchParams.get('includeDetail') === 'true';
    const includeToolDetail = searchParams.get('includeToolDetail') === 'true';

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required query param: taskId' },
        { status: 400 }
      );
    }

    const sessionWhere: Record<string, string> = { taskId };
    if (framework) sessionWhere.framework = framework;

    const session = await prisma.session.findFirst({
      where: sessionWhere,
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found for taskId: "${taskId}"` },
        { status: 404 }
      );
    }

    // Compute stable system overhead per agent (root + each subagent)
    const rootOverhead = await computeSystemOverhead(session.id);
    const subagentIds = await prisma.turn.findMany({
      where: { sessionId: session.id, isSubagent: true, subagentSessionId: { not: null } },
      select: { subagentSessionId: true },
      distinct: ['subagentSessionId'],
    });
    const overheadMap = new Map<string, number>();
    overheadMap.set("", rootOverhead);
    for (const { subagentSessionId } of subagentIds) {
      if (subagentSessionId) {
        overheadMap.set(subagentSessionId, await computeSystemOverhead(session.id, subagentSessionId));
      }
    }

    const where: Record<string, unknown> = { sessionId: session.id };
    if (isSubagent !== null && isSubagent !== undefined) {
      where.isSubagent = isSubagent === 'true';
    }
    if (role) {
      where.role = role;
    }
    if (subagentSessionId) {
      where.subagentSessionId = subagentSessionId;
    }

    const turns = await prisma.turn.findMany({
      where,
      orderBy: [{ turnIndex: 'asc' }],
      include: {
        toolCalls: {
          select: includeDetail || includeToolDetail
            ? {
                id: true,
                toolCallId: true,
                toolName: true,
                argsJson: true,
                resultJson: true,
                state: true,
                errorType: includeDetail ? true : undefined,
                errorMessage: includeDetail ? true : undefined,
                durationMs: true,
                isSkillRelated: (includeDetail || includeToolDetail) ? true : undefined,
              }
            : {
                id: true,
                toolCallId: true,
                toolName: true,
                state: true,
                durationMs: true,
              },
        },
        skillEvents: {
          select: includeDetail || includeToolDetail
            ? {
                id: true,
                skillName: true,
                skillVersion: true,
                eventType: true,
                success: true,
                errorMessage: true,
                argsJson: true,
                durationMs: true,
              }
            : {
                id: true,
                skillName: true,
                eventType: true,
                success: true,
              },
        },
      },
    });

    const total = turns.length;

    // Reconstruct inputMessagesJson for assistant turns if not stored (import optimization)
    if (includeDetail) {
      const assistantTurnsNeedingReconstruction = turns.filter(
        t => t.role === 'assistant' && !t.inputMessagesJson
      );
      if (assistantTurnsNeedingReconstruction.length > 0) {
        const allSessionTurns = await prisma.turn.findMany({
          where: { sessionId: session.id },
          orderBy: [{ turnIndex: 'asc' }],
          select: { id: true, role: true, content: true, turnIndex: true, isSubagent: true, subagentSessionId: true },
        });

        const rootContextTurns = allSessionTurns.filter(t => !t.isSubagent);
        const subagentContextMap = new Map<string, typeof allSessionTurns>();
        for (const t of allSessionTurns.filter(t => t.isSubagent && t.subagentSessionId)) {
          const arr = subagentContextMap.get(t.subagentSessionId!) ?? [];
          arr.push(t);
          subagentContextMap.set(t.subagentSessionId!, arr);
        }

        // Fetch tool calls (args + result) for prior assistant turns
        const allAssistantIds = allSessionTurns.filter(t => t.role === 'assistant').map(t => t.id);
        const allToolCalls = allAssistantIds.length > 0 ? await prisma.toolCall.findMany({
          where: { turnId: { in: allAssistantIds } },
          select: { turnId: true, toolCallId: true, toolName: true, argsJson: true, resultJson: true, isSkillRelated: true },
          orderBy: [{ id: 'asc' }],
        }) : [];
        const toolCallsByTurnId = new Map<string, typeof allToolCalls>();
        for (const tc of allToolCalls) {
          const arr = toolCallsByTurnId.get(tc.turnId) ?? [];
          arr.push(tc);
          toolCallsByTurnId.set(tc.turnId, arr);
        }

        const inputMessagesMap = new Map<string, string>();
        for (const t of assistantTurnsNeedingReconstruction) {
          const contextTurns = t.isSubagent && t.subagentSessionId
            ? (subagentContextMap.get(t.subagentSessionId!) ?? [])
            : rootContextTurns;
          // Reconstruct the LLM input window: start at the most recent /compact
          // continuation before this turn (a compact replaces prior history with
          // a summary), and skip local CLI command noise. See input-reconstruct.
          const previous = selectInputContextTurns(contextTurns, t.turnIndex);
          const msgs: Array<{ role: string; content: string | null; tokenCount: number; tool_calls?: Array<{ name: string; args: string | null; result: string | null; isSkillRelated?: boolean }> }> = [];
          for (const ct of previous) {
            const contentLen = ct.content?.length ?? 0;
            const baseTokens = Math.round(contentLen / 3.5);
            if (ct.role === 'assistant') {
              const tcs = toolCallsByTurnId.get(ct.id) ?? [];
              const argsTokens = tcs.reduce((s, tc) => s + Math.round((tc.argsJson?.length ?? 0) / 3.5), 0);
              const msg: typeof msgs[0] = { role: ct.role, content: ct.content ?? null, tokenCount: baseTokens + argsTokens };
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
              msgs.push(msg);
            } else {
              msgs.push({ role: ct.role, content: ct.content ?? null, tokenCount: baseTokens });
            }
          }
          inputMessagesMap.set(t.id, JSON.stringify(msgs));
        }

        // Patch original turn data so it flows through the map below
        for (const t of turns) {
          if (inputMessagesMap.has(t.id)) {
            t.inputMessagesJson = inputMessagesMap.get(t.id) ?? null;
          }
        }
      }
    }

    const items = turns.map(t => ({
      turnId: t.id,
      turnIndex: t.turnIndex,
      role: t.role,
      content: includeContent ? t.content : undefined,
      contentJson: includeDetail ? t.contentJson : undefined,
      inputMessagesJson: includeDetail ? t.inputMessagesJson : undefined,
      ttftMs: includeDetail ? t.ttftMs : undefined,
      modelId: includeDetail ? t.modelId : undefined,
      providerId: includeDetail ? t.providerId : undefined,
      contentSummary: t.contentSummary ?? t.content?.substring(0, 200) ?? null,
      agentName: t.agentName,
      isSubagent: t.isSubagent,
      subagentName: t.subagentName,
      subagentSessionId: t.subagentSessionId,
      parentExecutionId: t.parentExecutionId,
      totalTokens: t.totalTokens,
      inputTokens: t.inputTokens,
      outputTokens: t.outputTokens,
      reasoningTokens: t.reasoningTokens,
      cacheReadTokens: t.cacheReadTokens,
      cacheWriteTokens: t.cacheWriteTokens,
      inputMessagesCount: t.inputMessagesCount,
      inputMessagesTokens: t.inputMessagesTokens,
      contextWindowPct: t.contextWindowPct,
      systemOverheadTokens: t.isSubagent && t.subagentSessionId
        ? overheadMap.get(t.subagentSessionId) ?? 0
        : overheadMap.get("") ?? 0,
      latencyMs: t.latencyMs,
      createdAt: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
      completedAt: t.completedAt?.toISOString() ?? null,
      model: t.model,
      contextWindowLimit: getContextWindowLimit(t.model),
      finishReason: t.finishReason,
      toolCalls: t.toolCalls.map(tc => {
        const detail: ToolCallDetail = tc;
        return {
          id: tc.id,
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          argsJson: (includeDetail || includeToolDetail) ? (detail.argsJson ?? null) : undefined,
          resultJson: (includeDetail || includeToolDetail) ? (detail.resultJson ?? null) : undefined,
          state: tc.state,
          errorType: includeDetail ? (detail.errorType ?? null) : undefined,
          errorMessage: includeDetail ? (detail.errorMessage ?? null) : undefined,
          durationMs: tc.durationMs,
          isSkillRelated: (includeDetail || includeToolDetail) ? (detail.isSkillRelated ?? false) : undefined,
        }
      }),
      skillEvents: (() => {
        const skillToolCalls = t.toolCalls.filter(tc => {
          const d: ToolCallDetail = tc
          return d.isSkillRelated
        })
        return t.skillEvents.map((se, idx) => {
          const detail: SkillEventDetail = se
          const matchedTc: ToolCallDetail | undefined = skillToolCalls[idx]
          const matchedResult = matchedTc?.resultJson
          const resultError = matchedResult && (matchedResult.includes('<tool_use_error>') || matchedResult.includes('Exit code'))
          return {
            id: se.id,
            skillName: se.skillName,
            skillVersion: (includeDetail || includeToolDetail) ? (detail.skillVersion ?? null) : undefined,
            eventType: se.eventType,
            success: resultError ? false : se.success,
            errorMessage: resultError ? (matchedResult!.substring(0, 200)) : ((includeDetail || includeToolDetail) ? (detail.errorMessage ?? null) : undefined),
            argsJson: (includeDetail || includeToolDetail) ? (detail.argsJson ?? null) : undefined,
            durationMs: (includeDetail || includeToolDetail) ? (detail.durationMs ?? 0) : undefined,
          }
        })
      })(),
    }));

    return NextResponse.json({ items, total });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
