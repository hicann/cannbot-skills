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
import { isContinuationTurn } from '@/lib/shared/command-parser';

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

  // firstAssistant is chosen by turnIndex asc, so no prior turn can be an
  // assistant — the prior-assistant toolCall.findMany branch was dead code
  // and has been removed. priorMessages only ever contains user/system turns.
  const visibleEstimated = priorMessages.reduce((s, ct) => s + Math.round((ct.content?.length ?? 0) / 3.5), 0);
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
    // skipOverhead: when the caller doesn't need systemOverheadTokens (e.g. the
    // compare page), skips the per-(root+subagent) computeSystemOverhead loop
    // which is the main N+1 source on sessions with many subagents.
    const skipOverhead = searchParams.get('skipOverhead') === 'true';
    // maxContentLen: truncates the `content` field to the first N characters.
    // The compare page only needs enough content for diff highlighting + the
    // initial 500-char preview — full content is rarely viewed and accounts
    // for ~50% of payload size on large sessions. Trimming to 1000 chars
    // preserves <thinking>...</thinking> blocks (usually < 500 chars) while
    // cutting transfer + fetch time on 4000-turn sessions.
    const maxContentLenRaw = searchParams.get('maxContentLen');
    const maxContentLen = maxContentLenRaw ? parseInt(maxContentLenRaw, 10) : null;

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

    // Compute stable system overhead per agent (root + each subagent).
    // Skipped entirely when skipOverhead=true — the compare page only needs
    // role/content/tokens/tools/skills and never reads systemOverheadTokens,
    // so the per-subagent findFirst+findMany loop is pure waste there.
    // On a 27-subagent session this drops 28 findFirst + 28 findMany calls.
    const overheadMap = new Map<string, number>();
    if (!skipOverhead) {
      const rootOverhead = await computeSystemOverhead(session.id);
      const subagentIds = await prisma.turn.findMany({
        where: { sessionId: session.id, isSubagent: true, subagentSessionId: { not: null } },
        select: { subagentSessionId: true },
        distinct: ['subagentSessionId'],
      });
      overheadMap.set("", rootOverhead);
      for (const { subagentSessionId } of subagentIds) {
        if (subagentSessionId) {
          overheadMap.set(subagentSessionId, await computeSystemOverhead(session.id, subagentSessionId));
        }
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
      // Phase 9: when maxContentLen=0, omit the content column entirely from
      // the SQL SELECT — the DB stops reading multi-MB of content text, and
      // the compare page falls back to contentSummary (200 chars) for diff.
      // When maxContentLen>0, keep reading content (API layer truncates after).
      omit: maxContentLen === 0 ? { content: true } : undefined,
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

    // Reconstruct inputMessagesJson for assistant turns if not stored (import optimization)
    if (includeDetail) {
      const assistantTurnsNeedingReconstruction = turns.filter(
        t => t.role === 'assistant' && !t.inputMessagesJson
      );
      if (assistantTurnsNeedingReconstruction.length > 0) {
        const allSessionTurns = await prisma.turn.findMany({
          where: { sessionId: session.id },
          orderBy: [{ turnIndex: 'asc' }],
          select: { id: true, role: true, content: true, turnIndex: true, isSubagent: true, subagentSessionId: true, agentName: true },
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

        const compactionTokenMap = new Map<string, number>();
        for (const ct of turns) {
          if (ct.agentName === 'compaction') {
            compactionTokenMap.set(ct.id, ct.totalTokens);
          }
        }

        const inputMessagesMap = new Map<string, string>();
        for (const t of assistantTurnsNeedingReconstruction) {
          const contextTurns = t.isSubagent && t.subagentSessionId
            ? (subagentContextMap.get(t.subagentSessionId!) ?? [])
            : rootContextTurns;
          // Reconstruct the LLM input window: start at the most recent /compact
          // continuation before this turn (a compact replaces prior history with
          // a summary), and skip local CLI command noise. See input-reconstruct.
          const isCompactionAgent = t.agentName === 'compaction';
          const previous = selectInputContextTurns(contextTurns, t.turnIndex, t.agentName);

          if (isCompactionAgent && previous.length > 0) {
            const compactionTotalTokens = compactionTokenMap.get(t.id) ?? 0;
            let totalPriorTokens = 0;
            const lines: string[] = [];
            for (const ct of previous) {
              const contentLen = ct.content?.length ?? 0;
              const baseTokens = Math.round(contentLen / 3.5);
              if (ct.role === 'assistant') {
                const tcs = toolCallsByTurnId.get(ct.id) ?? [];
                const argsTokens = tcs.reduce((s, tc) => s + Math.round((tc.argsJson?.length ?? 0) / 3.5), 0);
                totalPriorTokens += baseTokens + argsTokens;
              } else {
                totalPriorTokens += baseTokens;
              }
              const preview = (ct.content ?? '').substring(0, 120);
              lines.push(`[${ct.role}] ${preview}${preview.length < (ct.content?.length ?? 0) ? '...' : ''}`);
            }
            const effectiveTokens = compactionTotalTokens > 0 ? compactionTotalTokens : totalPriorTokens;
            const summaryContent = `Prior conversation context (${previous.length} turns before /compact, ≈${effectiveTokens} tokens):\n\n${lines.join('\n')}`;
            const msgs = [{ role: 'user', content: summaryContent, tokenCount: effectiveTokens }];
            inputMessagesMap.set(t.id, JSON.stringify(msgs));
          } else {
            const targetInputTokens = t.agentName === 'compaction'
              ? (t.inputMessagesTokens > 0 ? t.inputMessagesTokens : 0)
              : (t.inputMessagesTokens > 0 ? Math.max(0, t.inputMessagesTokens - t.outputTokens) : 0);
            const msgs: Array<{ role: string; content: string | null; tokenCount: number; agentName?: string; tool_calls?: Array<{ name: string; args: string | null; result: string | null; isSkillRelated?: boolean }> }> = [];
            for (const ct of previous) {
              const contentLen = ct.content?.length ?? 0;
              const baseTokens = Math.round(contentLen / 3.5);
              const isCompaction = ct.agentName === 'compaction';
              const isContinuation = isCompaction || (ct.role === 'user' && ct.content && isContinuationTurn(ct.content));
              const effectiveRole = isContinuation ? 'user' : ct.role;
              if (ct.role === 'assistant') {
                const tcs = toolCallsByTurnId.get(ct.id) ?? [];
                const argsTokens = tcs.reduce((s, tc) => s + Math.round((tc.argsJson?.length ?? 0) / 3.5), 0);
                const compactionTotal = compactionTokenMap.get(ct.id) ?? 0;
                const continuationTokens = isCompaction && compactionTotal > 0 ? compactionTotal : (isContinuation && targetInputTokens > 0 ? targetInputTokens : baseTokens + argsTokens);
                const msg: typeof msgs[0] = { role: effectiveRole, content: ct.content ?? null, tokenCount: continuationTokens };
                if (isContinuation) msg.agentName = 'continuation';
                if (tcs.length > 0 && !isCompaction) {
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
                const isContinuationUser = isContinuation && ct.role === 'user';
                const continuationTokens = isContinuationUser && targetInputTokens > 0 ? targetInputTokens : baseTokens;
                const msg: typeof msgs[0] = { role: effectiveRole, content: ct.content ?? null, tokenCount: continuationTokens };
                if (isContinuation) msg.agentName = 'continuation';
                msgs.push(msg);
              }
            }
            inputMessagesMap.set(t.id, JSON.stringify(msgs));
          }
        }

        // Patch original turn data so it flows through the map below
        for (const t of turns) {
          if (inputMessagesMap.has(t.id)) {
            t.inputMessagesJson = inputMessagesMap.get(t.id) ?? null;
          }
        }
      }
    }

    const items = turns.map(t => {
      // Prisma omit (maxContentLen===0) makes t.content type-optional at the
      // TS level even when present; normalize via a typed view so the rest of
      // the mapping reads content uniformly.
      const tContent = (t as { content?: string | null }).content ?? null
      return {
      turnId: t.id,
      turnIndex: t.turnIndex,
      role: t.role,
      content: includeContent
        ? (maxContentLen && tContent ? tContent.substring(0, maxContentLen) : tContent)
        : undefined,
      contentJson: includeDetail ? t.contentJson : undefined,
      inputMessagesJson: includeDetail ? t.inputMessagesJson : undefined,
      ttftMs: includeDetail ? t.ttftMs : undefined,
      modelId: includeDetail ? t.modelId : undefined,
      providerId: includeDetail ? t.providerId : undefined,
      contentSummary: t.contentSummary ?? tContent?.substring(0, 200) ?? null,
      contentLength: tContent?.length ?? 0,
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
      inputMessagesCount: t.agentName === 'compaction' ? (() => {
        const json = (t as any).inputMessagesJson; // eslint-disable-line @typescript-eslint/no-explicit-any
        return json ? JSON.parse(json).length : t.inputMessagesCount;
      })() : t.inputMessagesCount,
      inputMessagesTokens: t.agentName === 'compaction' && t.inputMessagesTokens === t.outputTokens ? t.inputTokens : t.inputMessagesTokens,
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
      }
    })

    // Insert virtual continuation user turns after each compaction assistant turn.
    // In claude-code, the continuation summary is a real user turn (e.g. #44).
    // In opencode, the summary is the compaction agent's content (#125), so we
    // synthesize a continuation turn from it to match claude-code's data model.
    const continuationItems: typeof items = [];
    for (const item of items) {
      continuationItems.push(item);
      if (item.agentName === 'compaction' && item.role === 'assistant') {
        const compTurn = turns.find(t => t.id === item.turnId)
        const compContent = includeContent ? ((compTurn as { content?: string | null } | undefined)?.content ?? null) : undefined;
        const compSummary = compTurn?.contentSummary ?? null;
        if (compContent || compSummary) {
          continuationItems.push({
            turnId: `${item.turnId}-continuation`,
            turnIndex: item.turnIndex,
            role: 'user',
            content: compContent ?? compSummary,
            contentJson: undefined,
            inputMessagesJson: null,
            ttftMs: undefined,
            modelId: undefined,
            providerId: undefined,
            contentSummary: compSummary,
            contentLength: compContent?.length ?? 0,
            agentName: 'continuation',
            isSubagent: item.isSubagent,
            subagentName: item.subagentName,
            subagentSessionId: item.subagentSessionId,
            parentExecutionId: item.parentExecutionId,
            totalTokens: 0,
            inputTokens: 0,
            outputTokens: 0,
            reasoningTokens: 0,
            cacheReadTokens: 0,
            cacheWriteTokens: 0,
            inputMessagesCount: 0,
            inputMessagesTokens: 0,
            contextWindowPct: null,
            systemOverheadTokens: 0,
            latencyMs: 0,
            createdAt: item.createdAt,
            completedAt: null,
            model: null,
            contextWindowLimit: 200000,
            finishReason: null,
            toolCalls: [],
            skillEvents: [],
          });
        }
      }
    }

    return NextResponse.json({ items: continuationItems, total: continuationItems.length });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
