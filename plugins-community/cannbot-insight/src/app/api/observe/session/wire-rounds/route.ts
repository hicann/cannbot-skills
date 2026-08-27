// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { isClaudeFormatSession } from '@/lib/shared/session-format';
import { parseJsonlLines } from '@/lib/ingest/adapters/claude-jsonl';

// Wire-rounds view: rebuild each WIRE REQUEST directly from the capture
// jsonl, independent of the DB/import pipeline. The file stores each message
// once (delta), so a round's request = the messages accumulated before its
// assistant response line(s) — exactly what the agent sent on the wire.
// This is the ground-truth view to compare against the LLM Input tab (which
// reconstructs from imported DB turns).
// verbatim: no server-side truncation — the client collapses per message
// with an expand toggle (like the Turns tab)

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const taskId = searchParams.get('taskId');
    const full = searchParams.get('full') === '1';
    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    const session = await prisma.session.findFirst({
      where: { taskId },
      select: { id: true, sourcePath: true, framework: true, version: true },
      orderBy: { startTime: 'desc' },
    });
    if (!session || !session.sourcePath) {
      return NextResponse.json({ error: 'Session or sourcePath not found' }, { status: 404 });
    }
    if (!isClaudeFormatSession(session.framework, session.version)) {
      return NextResponse.json({ error: 'Only claude-format sessions (claude-code or proxy capture) have a wire capture' }, { status: 400 });
    }

    const lines = parseJsonlLines(session.sourcePath);
    const preview = (s: string): { json: string; truncated: boolean; chars: number } => ({
      json: s,
      truncated: false,
      chars: s.length,
    });

    interface WireMessage { role: string; content: ReturnType<typeof preview>; timestamp: string | null }
    const rounds: Array<{
      index: number;
      timestamp: string | null;
      model: string | null;
      usage: { input?: number; output?: number } | null;
      // requestMessages[0..newFrom) carried over from the previous round;
      // [newFrom..) are NEW in this round (append-only wire history)
      newFrom: number;
      requestMessages: WireMessage[];
      // 本轮请求的累积消息总数（含历史 + 本轮新增）——只发新增消息体，
      // totalMessages 给前端显示「共 N 条历史」而不传输全量
      totalMessages: number;
      // /compact 边界：本轮首条消息是 continuation 摘要（历史被替换），
      // 前端据此渲染 compact 分隔标记 + 历史骤降提示
      compactBoundary: boolean;
      // compact 前的累积消息数（骤降前 → 后，如 86 → 3）
      prevTotalMessages: number;
      response: { content: ReturnType<typeof preview>; blocks: string[]; text: string };
    }> = [];

    interface HistEntry { role: string; content: string; timestamp: string | null }
    const history: HistEntry[] = [];
    // 只记本轮新增的消息索引范围，不发全量累积历史（O(N²) → O(N)）
    let roundStartHistLen = 0;
    // 上一轮结束时的 totalMessages（检测 compact 骤降）
    let prevTotal = 0;
    // 本轮在 closeRound 前是否检测到了 compact 边界
    let compactThisRound = false;
    let pending: { blocks: unknown[]; model: string | null; usage: unknown; ts: string | null; text: string } | null = null;

    const newMessagesSince = (from: number): WireMessage[] =>
      history.slice(from).map(h => ({ role: h.role, content: preview(h.content), timestamp: h.timestamp }));

    const closeRound = () => {
      if (!pending) return;
      const newMsgs = newMessagesSince(roundStartHistLen);
      rounds.push({
        index: rounds.length + 1,
        timestamp: pending.ts,
        model: pending.model,
        usage: (pending.usage as { input_tokens?: number; output_tokens?: number } | null) ?? null,
        newFrom: 0,
        requestMessages: newMsgs,
        totalMessages: history.length,
        compactBoundary: compactThisRound,
        prevTotalMessages: prevTotal,
        response: {
          content: preview(JSON.stringify(pending.blocks)),
          blocks: pending.blocks.map((b: { type?: string }) => b?.type ?? '?'),
          text: pending.text.slice(0, 300),
        },
      });
      prevTotal = history.length;
      history.push({ role: 'assistant', content: JSON.stringify(pending.blocks), timestamp: pending.ts });
      roundStartHistLen = history.length;
      compactThisRound = false;
      pending = null;
    };

    for (const line of lines) {
      if (line.type === 'assistant' && line.message) {
        if (!pending) {
          pending = { blocks: [], model: line.message.model ?? null, usage: line.message.usage ?? null, ts: line.timestamp ?? null, text: '' };
        }
        const blocks = Array.isArray(line.message.content) ? line.message.content : [];
        pending.blocks.push(...blocks);
        for (const b of blocks as Array<{ type?: string; text?: string }>) {
          if (b?.type === 'text' && b.text) pending.text += b.text;
        }
        if (line.message.model) pending.model = line.message.model;
        if (line.message.usage) pending.usage = line.message.usage;
        continue;
      }
      closeRound();
      if ((line.type === 'user' || line.type === 'system') && line.message) {
        // /compact 边界：claude-code 用 continuation 摘要替换整个历史，
        // 此前的消息从 messages 数组中消失。检测到时重置 history —— 后续
        // 轮次的累积请求只含摘要 + 新消息，不再重复已被丢弃的旧消息。
        // continuation 文本可能在第二个 text block（第一个是 system-reminder），
        // 故逐个 text block 用 startsWith 判断（isContinuationTurn 用 startsWith
        // 防止误匹配引用 marker 的普通消息）
        const blocks = Array.isArray(line.message.content)
          ? (line.message.content as Array<{ type?: string; text?: string }>)
          : [];
        const hasContinuation = line.type === 'user' && blocks.some(
          b => b.type === 'text' && typeof b.text === 'string' && b.text.startsWith('This session is being continued from a previous conversation')
        );
        if (hasContinuation) {
          history.length = 0;
          roundStartHistLen = 0;
          compactThisRound = true;
        }
        history.push({
          role: line.message.role,
          content: JSON.stringify(line.message.content ?? ''),
          timestamp: line.timestamp ?? null,
        });
      }
    }
    closeRound();

    return NextResponse.json({
      taskId,
      sourcePath: session.sourcePath,
      full,
      totalRounds: rounds.length,
      rounds,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
