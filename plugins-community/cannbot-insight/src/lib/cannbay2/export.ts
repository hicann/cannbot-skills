// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// opencode 来源会话没有捕获 jsonl，上传前由 insight DB 导出 claude-jsonl。
// 输出 = cannbay2 仓库布局（sessions/<sid>/<sid>.jsonl + sessions/<sid>/subagents/），
// 重新导入经 claude-jsonl adapter 走标准管线。保真边界：turns/toolCalls/token
// 级数据齐全；system prompt / Full Context 不含（opencode 原生 db 没有）。
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

interface ExportTurn {
  id: string;
  turnIndex: number;
  role: string;
  content: string | null;
  model: string | null;
  modelId: string | null;
  providerId: string | null;
  temperature: number | null;
  maxTokens: number | null;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  isSubagent: boolean;
  subagentSessionId: string | null;
  latencyMs: number;
  ttftMs: number | null;
  finishReason: string | null;
  createdAt_ts: Date | null;
  createdAt: Date;
}

interface ExportToolCall {
  turnId: string;
  toolCallId: string;
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
  errorType: string | null;
  errorMessage: string | null;
  durationMs: number;
  startedAt: Date | null;
}

interface ExportBridge {
  dispatchToolCallId: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
}

type MinimalPrisma = {
  session: { findFirst(args: { where: Record<string, string> }): Promise<{ id: string; taskId: string; framework: string; version: string | null } | null> };
  turn: { findMany(args: { where: { sessionId: string }; orderBy: { turnIndex: 'asc' } }): Promise<ExportTurn[]> };
  toolCall: { findMany(args: { where: { turn: { sessionId: string } } }): Promise<ExportToolCall[]> };
  interactionBridge: { findMany(args: { where: { sessionId: string } }): Promise<ExportBridge[]> };
};

// x_cannbay 命名空间（docs/cannbay-schema-spec.md）：声明式权威源，双轨期
// 与 legacy 顶层字段（source/duration_ms/stopReason）并存，读方 x_cannbay 优先。
function xCannbay(schema: string, data: Record<string, unknown>) {
  return { schema, version: 1, data };
}

function toISO(d: Date | null): string {
  return (d ?? new Date(0)).toISOString();
}

function toolResultText(tc: ExportToolCall): string {
  if (tc.resultJson == null) return '';
  try {
    const parsed: unknown = JSON.parse(tc.resultJson);
    if (typeof parsed === 'string') return parsed;
    return JSON.stringify(parsed);
  } catch {
    return tc.resultJson;
  }
}

// 一组 turn → claude 行。assistant 行带 tool_use 块（后续紧跟一条 user 行收
// tool_result，claude-jsonl 的 collectAllToolResults 从 user 行读结果）。
function turnGroupToLines(turns: ExportTurn[], toolCallsByTurn: Map<string, ExportToolCall[]>): string[] {
  const lines: string[] = [];
  for (const t of turns) {
    const ts = toISO(t.createdAt_ts ?? t.createdAt);
    const tcs = toolCallsByTurn.get(t.id) ?? [];

    if (t.role === 'assistant') {
      const blocks: Array<Record<string, unknown>> = [];
      if (t.content) blocks.push({ type: 'text', text: t.content });
      for (const tc of tcs) {
        let input: Record<string, unknown> = {};
        try { input = tc.argsJson ? JSON.parse(tc.argsJson) : {}; } catch { input = { raw: tc.argsJson }; }
        blocks.push({ type: 'tool_use', id: tc.toolCallId, name: tc.toolName, input });
      }
      lines.push(JSON.stringify({
        type: 'assistant',
        message: {
          id: crypto.randomUUID(),
          role: 'assistant',
          content: blocks.length > 0 ? blocks : (t.content ?? ''),
          model: t.model ?? undefined,
          usage: t.totalTokens > 0 ? {
            input_tokens: t.inputTokens,
            output_tokens: t.outputTokens,
            cache_read_input_tokens: t.cacheReadTokens || undefined,
            cache_creation_input_tokens: t.cacheWriteTokens || undefined,
          } : undefined,
        },
        timestamp: ts,
        // source 标记声明"本行由 insight 导出生成、duration_ms 承载真实 latency"。
        // 专用值 'insight-export'（不以 -proxy 结尾）：adapter 侧同源识别让
        // duration_ms 生效，但 proxySourceOf 不认 → version/徽标不误标成代理捕获。
        // 无任何标记时 adapter 走 native 分支，时间戳差算 latency 覆盖 duration_ms。
        source: 'insight-export',
        duration_ms: t.latencyMs || undefined,
        stopReason: t.finishReason ?? undefined,
        x_cannbay: xCannbay('cc-db-turn', {
          latencyMs: t.latencyMs,
          ttftMs: t.ttftMs ?? undefined,
          stopReason: t.finishReason ?? undefined,
          reasoningTokens: t.reasoningTokens || undefined,
          modelId: t.modelId ?? undefined,
          providerId: t.providerId ?? undefined,
          temperature: t.temperature ?? undefined,
          maxTokens: t.maxTokens ?? undefined,
          toolCalls: tcs.map(tc => ({
            toolUseId: tc.toolCallId,
            state: tc.state,
            errorType: tc.errorType ?? undefined,
            errorMessage: tc.errorMessage ?? undefined,
            durationMs: tc.durationMs || undefined,
            startedAt: tc.startedAt ? tc.startedAt.toISOString() : undefined,
          })),
          turnKind: t.role === 'system' ? 'system' : 'normal',
        }),
      }));
      if (tcs.length > 0) {
        lines.push(JSON.stringify({
          type: 'user',
          message: {
            role: 'user',
            content: tcs.map(tc => ({
              type: 'tool_result',
              tool_use_id: tc.toolCallId,
              content: toolResultText(tc),
              is_error: tc.state !== 'ok',
            })),
          },
          timestamp: ts,
        }));
      }
    } else if (t.role === 'user' && t.content != null) {
      lines.push(JSON.stringify({
        type: 'user',
        message: { role: 'user', content: t.content },
        timestamp: ts,
      }));
    } else if (t.role === 'system' && t.content != null) {
      // skill 注入 / 命令消息被管线重分类为 system turn —— type:'system' 行
      // 在 adapter 的 system 分支还原（缺这条分支时 system turn 导出即蒸发）
      lines.push(JSON.stringify({
        type: 'system',
        message: { role: 'system', content: t.content },
        timestamp: ts,
      }));
    }
  }
  return lines;
}

// 导出为 cannbay2 仓库布局文件，返回写出的相对路径（相对 outDir）。
export async function exportSessionToClaudeJsonl(
  prisma: MinimalPrisma,
  taskId: string,
  outDir: string,
): Promise<string[]> {
  const session = await prisma.session.findFirst({ where: { taskId } });
  if (!session) throw new Error(`Session not found: "${taskId}"`);

  const turns = await prisma.turn.findMany({
    where: { sessionId: session.id },
    orderBy: { turnIndex: 'asc' },
  });
  const toolCalls = await prisma.toolCall.findMany({ where: { turn: { sessionId: session.id } } });
  const bridges = await prisma.interactionBridge.findMany({ where: { sessionId: session.id } });

  const toolCallsByTurn = new Map<string, ExportToolCall[]>();
  for (const tc of toolCalls) {
    const list = toolCallsByTurn.get(tc.turnId) ?? [];
    list.push(tc);
    toolCallsByTurn.set(tc.turnId, list);
  }

  const sid = session.taskId;
  const written: string[] = [];
  const sessionDir = path.join(outDir, 'sessions', sid);
  fs.mkdirSync(sessionDir, { recursive: true });

  const mainTurns = turns.filter(t => !t.isSubagent);
  const mainFile = path.join(sessionDir, `${sid}.jsonl`);
  fs.writeFileSync(mainFile, turnGroupToLines(mainTurns, toolCallsByTurn).join('\n') + '\n');
  written.push(path.posix.join('sessions', sid, `${sid}.jsonl`));

  // cc-session-meta：文件级 producer/framework/version 声明（修复重导入
  // framework 错变 claude-code、version 丢失）
  fs.writeFileSync(
    path.join(sessionDir, `${sid}.meta.json`),
    JSON.stringify({
      x_cannbay: xCannbay('cc-session-meta', {
        producer: 'insight-export',
        framework: session.framework,
        ccVersion: session.version ?? undefined,
        sid,
      }),
    }, null, 2) + '\n',
  );
  written.push(path.posix.join('sessions', sid, `${sid}.meta.json`));

  // 子会话分组 → subagents/<subId>.jsonl + meta.json（toolUseId 从桥接数据取）
  const subGroups = new Map<string, ExportTurn[]>();
  for (const t of turns) {
    if (!t.isSubagent || !t.subagentSessionId) continue;
    const list = subGroups.get(t.subagentSessionId) ?? [];
    list.push(t);
    subGroups.set(t.subagentSessionId, list);
  }
  for (const [subId, subTurns] of subGroups) {
    const subFile = path.join(sessionDir, 'subagents', `${subId}.jsonl`);
    fs.mkdirSync(path.dirname(subFile), { recursive: true });
    fs.writeFileSync(subFile, turnGroupToLines(subTurns, toolCallsByTurn).join('\n') + '\n');
    written.push(path.posix.join('sessions', sid, 'subagents', `${subId}.jsonl`));

    const bridge = bridges.find(b => b.subagentSessionId === subId) ?? null;
    const meta = {
      toolUseId: bridge?.dispatchToolCallId ?? null,
      name: bridge?.subagentName ?? null,
      agentType: bridge?.subagentType ?? null,
      x_cannbay: xCannbay('cc-subagent-meta', {
        toolUseId: bridge?.dispatchToolCallId ?? null,
        name: bridge?.subagentName ?? null,
        agentType: bridge?.subagentType ?? null,
        subagentSessionId: subId,
      }),
    };
    const metaFile = path.join(sessionDir, 'subagents', `${subId}.meta.json`);
    fs.writeFileSync(metaFile, JSON.stringify(meta, null, 2) + '\n');
    written.push(path.posix.join('sessions', sid, 'subagents', `${subId}.meta.json`));
  }

  return written;
}
