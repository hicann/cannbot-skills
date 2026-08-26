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
import { selectInputContextTurns, toWireOrder } from '@/lib/ingest/input-reconstruct';
import { isContinuationTurn } from '@/lib/shared/command-parser';
import { listSubagentSessions } from '@/lib/ingest/adapters/claude-jsonl';
import { readFullContext, type FullContext } from '@/lib/ingest/adapters/claude-jsonl-full-context';
import { readWireEnrichments, wireEnrichmentKey } from '@/lib/ingest/adapters/claude-jsonl-wire';
import path from 'node:path';

// Read the verbatim captured context (system prompt, tools, memory files, skills)
// from the session's capture file via Session.sourcePath. The proxy writes
// extended claude-format, so this applies to claude-code framework sessions.
// Returns null for other sources — those have no verbatim capture.
//
// For a SUBAGENT turn, read the subagent's OWN jsonl (subagents/<subId>.jsonl)
// — subagents have a different system prompt (cc_is_subagent=true) and a
// restricted toolset. Reading the main session's file would mislabel the
// main agent's context as the subagent's. Falls back to null (honest empty)
// if the subagent file is missing.
async function readSessionFullContext(sessionId: string, subagentSessionId: string | null = null): Promise<FullContext | null> {
  const session = await prisma.session.findUnique({
    where: { id: sessionId },
    select: { taskId: true, framework: true, sourcePath: true },
  });
  if (!session || !session.sourcePath) return null;
  if (session.framework !== 'claude-code') return null;
  if (subagentSessionId) {
    try {
      // Subagent dirs are keyed by the session id (Session.taskId), NOT the
      // capture filename — proxy captures carry a cpx- filename prefix while
      // their <sid>/subagents/ dir uses the unprefixed id. For native imports
      // taskId === filename stem, so behavior is unchanged there.
      const fileSid = session.taskId || path.basename(session.sourcePath, '.jsonl');
      const sub = listSubagentSessions(session.sourcePath, fileSid)
        .find(s => s.id === subagentSessionId);
      if (sub) return readFullContext(sub.filePath);
    } catch { /* fall through to null */ }
    return null;
  }
  return readFullContext(session.sourcePath);
}

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

    // Virtual continuation turn: turnId ends with "-continuation"
    // The real turn in DB is the compaction turn whose id is the prefix.
    if (turnId.endsWith('-continuation')) {
      const compactionId = turnId.slice(0, -'-continuation'.length);
      const compaction = await prisma.turn.findUnique({
        where: { id: compactionId },
        select: { id: true, sessionId: true, turnIndex: true, role: true, content: true, contentSummary: true, contentJson: true, agentName: true, isSubagent: true, subagentName: true, subagentSessionId: true, createdAt: true, createdAt_ts: true },
      });
      if (!compaction || compaction.agentName !== 'compaction') {
        return NextResponse.json(
          { error: `Continuation turn not found: "${turnId}"` },
          { status: 404 }
        );
      }
      const systemOverheadTokens = await computeSystemOverhead(compaction.sessionId, compaction.subagentSessionId);
      const fc = await readSessionFullContext(compaction.sessionId, compaction.subagentSessionId);
      return NextResponse.json({
        turnId,
        sessionId: compaction.sessionId,
        turnIndex: compaction.turnIndex,
        role: 'user',
        content: compaction.content,
        contentJson: compaction.contentJson,
        contentSummary: compaction.contentSummary ?? compaction.content?.substring(0, 200) ?? null,
        inputMessagesJson: null,
        inputMessagesCount: 0,
        inputMessagesTokens: 0,
        contextWindowPct: null,
        systemOverheadTokens,
        systemPrompt: fc?.systemPrompt ?? null,
        fullContext: fc ? { tools: fc.tools, memoryFiles: fc.memoryFiles, skills: fc.skills } : null,
        agentName: 'continuation',
        subagentName: compaction.subagentName,
        subagentSessionId: compaction.subagentSessionId,
        isSubagent: compaction.isSubagent,
        totalTokens: 0,
        inputTokens: 0,
        outputTokens: 0,
        reasoningTokens: 0,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        latencyMs: 0,
        ttftMs: null,
        createdAt: compaction.createdAt_ts?.toISOString() ?? compaction.createdAt.toISOString(),
        completedAt: null,
        model: null,
        modelId: null,
        providerId: null,
        contextWindowLimit: 200000,
        finishReason: null,
        toolCalls: [],
        skillEvents: [],
      });
    }

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
        select: { id: true, turnIndex: true, role: true, content: true, agentName: true },
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
      // toWireOrder then folds reminder-split system turns back into their user
      // message so the list mirrors the original request exactly.
      const filtered = toWireOrder(selectInputContextTurns(previousTurns, turn.turnIndex, turn.agentName));

      // Fetch compaction turn totalTokens for accurate continuation tokenCount
      const compactionIds = filtered.filter(ct => ct.agentName === 'compaction').map(ct => ct.id);
      const compactionTokenRows = compactionIds.length > 0 ? await prisma.turn.findMany({
        where: { id: { in: compactionIds } },
        select: { id: true, totalTokens: true },
      }) : [];
      const compactionTokenMap = new Map<string, number>();
      for (const cr of compactionTokenRows) {
        compactionTokenMap.set(cr.id, cr.totalTokens);
      }

      const messages: Array<{ role: string; content: string | null; tokenCount: number; agentName?: string; tool_calls?: Array<{ name: string; args: string | null; result: string | null; isSkillRelated?: boolean }> }> = [];
      const targetInputTokens = turn.agentName === 'compaction'
        ? (turn.inputMessagesTokens > 0 ? turn.inputMessagesTokens : 0)
        : (turn.inputMessagesTokens > 0 ? Math.max(0, turn.inputMessagesTokens - turn.outputTokens) : 0);
      for (const ct of filtered) {
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
          const msg: typeof messages[0] = {
            role: effectiveRole,
            content: ct.content ?? null,
            tokenCount: continuationTokens,
          };
          if (isContinuation) msg.agentName = 'continuation';
          if (tcs.length > 0 && !isCompaction) {
            msg.tool_calls = tcs.map(tc => {
              const isSkill = tc.isSkillRelated;
              const argsMax = isSkill ? 2000 : 1500;
              return {
                name: tc.toolName,
                args: tc.argsJson ? (tc.argsJson.length > argsMax ? tc.argsJson.substring(0, argsMax) + '...' : tc.argsJson) : null,
                result: null,
                isSkillRelated: isSkill ? true : undefined,
              };
            });
          }
          messages.push(msg);
          // Wire fidelity (Anthropic protocol): tool_result is its own user
          // message AFTER the assistant tool_use — not folded into the
          // tool_calls entry. One result message per call, in call order.
          if (!isCompaction) {
            for (const tc of tcs) {
              if (tc.resultJson == null) continue;
              const isSkill = tc.isSkillRelated;
              const resultMax = isSkill ? 5000 : 3000;
              const r = tc.resultJson.length > resultMax ? tc.resultJson.substring(0, resultMax) + '...' : tc.resultJson;
              messages.push({
                role: 'tool_result',
                content: r,
                tokenCount: Math.round(r.length / 3.5),
                name: tc.toolName,
              });
            }
          }
        } else {
          const isContinuationUser = isContinuation && ct.role === 'user';
          const continuationTokens = isContinuationUser && targetInputTokens > 0 ? targetInputTokens : baseTokens;
          const msg: typeof messages[0] = {
            role: effectiveRole,
            content: ct.content ?? null,
            tokenCount: continuationTokens,
          };
          if (isContinuation) msg.agentName = 'continuation';
          messages.push(msg);
        }
      }

      inputMessagesJson = JSON.stringify(messages);
      // The stored count predates the wire-order merge (reminder turn was its
      // own message); use the reconstructed list length so header == list.
      if (turn.agentName !== 'compaction') {
        (turn as { inputMessagesCount?: number }).inputMessagesCount = messages.length;
      }
    }

    // Compute stable system overhead (fixed for entire session)
    const systemOverheadTokens = await computeSystemOverhead(turn.sessionId, turn.subagentSessionId);
    const fc = await readSessionFullContext(turn.sessionId, turn.subagentSessionId);

    // Proxy 扩展层（OCP）：proxy turn 的 contentJson/inputMessagesJson 不存 DB
    // （管线不感知），由扩展层 readWireEnrichments 从捕获文件按需读取 —— 与
    // readFullContext 同构。仅 proxy 捕获 session（version 带 -proxy 后缀）触发。
    // 按 (role, createdAt_ts) 稳定键查 —— 不按 turnIndex，因为管线在 compact
    // 边界折叠 interaction 会使数组下标 ≠ DB turnIndex（compact 错位）。
    // 限制：仅 root turn（subagent 的全局 createdAt_ts 与 wire 文件本地键不
    // 对应；subagent 的 verbatim 由 readFullContext 提供 system/tools，
    // inputMessagesJson 走标准重建）。
    let wireContentJson = turn.contentJson;
    let wireInputMessagesJson = inputMessagesJson;
    let wireInputMessagesCount = turn.inputMessagesCount;
    // ttftMs has no pipeline field (turn-split hardcodes null), so proxy output
    // turns override it from the extension layer. latencyMs / finishReason flow
    // through the standard pipeline (emitter duration_ms + stopReason → adapter
    // → turn-split → DB), so they're read straight from the turn — no override.
    let wireTtftMs = turn.ttftMs;
    const sessionRow = await prisma.session.findUnique({
      where: { id: turn.sessionId },
      select: { version: true, sourcePath: true },
    });
    if (sessionRow?.version?.endsWith('-proxy') && sessionRow.sourcePath && !turn.subagentSessionId) {
      const enrichments = readWireEnrichments(sessionRow.sourcePath);
      // createdAt_ts 经管线透传不变 = buildWireRounds 赋的 timeInfo.created
      // （已验证）；fallback createdAt（CLAUDE.md: createdAt_ts nullable）
      const tsMs = (turn.createdAt_ts ?? turn.createdAt).getTime();
      const enrich = enrichments.get(wireEnrichmentKey(turn.role, tsMs));
      if (enrich) {
        wireContentJson = enrich.contentJson;
        // proxy 输出 turn 的 inputMessagesJson 由扩展层提供 → 跳过重建
        if (enrich.inputMessagesJson) {
          wireInputMessagesJson = enrich.inputMessagesJson;
          try { wireInputMessagesCount = JSON.parse(enrich.inputMessagesJson).length; } catch { /* keep */ }
        }
        if (enrich.ttftMs != null) wireTtftMs = enrich.ttftMs;
      }
    }

    return NextResponse.json({
      turnId: turn.id,
      sessionId: turn.sessionId,
      turnIndex: turn.turnIndex,
      role: turn.role,
      content: turn.content,
      contentJson: wireContentJson,
      contentSummary: turn.contentSummary ?? turn.content?.substring(0, 200) ?? null,
      inputMessagesJson: wireInputMessagesJson,
      inputMessagesCount: turn.agentName === 'compaction' ? (wireInputMessagesJson ? JSON.parse(wireInputMessagesJson).length : wireInputMessagesCount) : wireInputMessagesCount,
      inputMessagesTokens: turn.agentName === 'compaction' && turn.inputMessagesTokens === turn.outputTokens ? turn.inputTokens : turn.inputMessagesTokens,
      contextWindowPct: turn.contextWindowPct,
      systemOverheadTokens,
      systemPrompt: fc?.systemPrompt ?? null,
      fullContext: fc ? { tools: fc.tools, memoryFiles: fc.memoryFiles, skills: fc.skills } : null,
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
      ttftMs: wireTtftMs,
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
