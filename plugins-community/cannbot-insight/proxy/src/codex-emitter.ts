// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// codex-emitter: convert a captured codex wire-record (ProxyRecord,
// protocol='responses') into extended claude-format jsonl lines, so
// cannbot-insight's existing claude-jsonl adapter can ingest it with zero
// insight changes.
//
// Independent from claude-emitter / opencode-emitter — the three share NO
// conversion logic because the Responses API wire format differs from both
// Anthropic Messages and OpenAI Chat Completions in every relevant dimension:
//   - request body: {input: [...items]} (not messages[], not content blocks)
//   - items: message / function_call / function_call_output (independent items,
//     not blocks inside a message)
//   - system prompt: `instructions` field (not `system` / messages[0])
//
// Output format is the SAME extended claude-format + x_cannbay as the other
// two emitters, so the single claude-jsonl adapter consumes all three.

import crypto from 'node:crypto';
import type { ProxyRecord, AnthropicContentBlock, AnthropicUsage, OpenAIUsage } from './types';
import { sessionFilePath, appendClaudeLine, writeMeta, sessionMetaPath } from './writer';
import { proxySourceMarker } from './claude-emitter';

interface ResponseInputItem {
  type: string;
  role?: string;
  content?: Array<{ type: string; text?: string }>;
  name?: string;
  arguments?: string;
  call_id?: string;
  output?: string;
}
interface ResponseRequestBody {
  input?: ResponseInputItem[];
  tools?: unknown[];
  model?: string;
  instructions?: string;
  stream?: boolean;
}

function xCannbay(schema: string, data: Record<string, unknown>) {
  return { schema, version: 1, data };
}

const prevInputCount = new Map<string, number>();
const roundCount = new Map<string, number>();
const sessionMetaWritten = new Set<string>();

function toISO(ts: number): string {
  return new Date(ts).toISOString();
}

function mapUsage(usage: AnthropicUsage | OpenAIUsage | null): AnthropicUsage | undefined {
  if (!usage) return undefined;
  return usage as AnthropicUsage;
}

function convertInputItem(item: ResponseInputItem): { type: string; message: { role: string; content: string | AnthropicContentBlock[] }; source: string; x_cannbay: { schema: string; version: number; data: { roundIndex: number; kind: string } } } | null {
  if (item.type === 'message') {
    const role = item.role ?? 'user';
    if (role === 'assistant' || role === 'system' || role === 'developer') return null;
    const content = item.content && item.content.length > 0
      ? item.content.map(c => ({ type: 'text', text: c.text ?? '' })).filter(c => c.text)
      : '';
    return {
      type: 'user',
      message: { role: 'user', content: content || '' },
      source: 'codex-proxy',
      x_cannbay: xCannbay('cc-wire-input', { roundIndex: 0, kind: 'user' }),
    };
  }
  if (item.type === 'function_call_output') {
    return {
      type: 'user',
      message: {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: item.call_id ?? '', content: item.output ?? '' }],
      },
      source: 'codex-proxy',
      x_cannbay: xCannbay('cc-wire-input', { roundIndex: 0, kind: 'tool-result' }),
    };
  }
  return null;
}

export function emit(rec: ProxyRecord): void {
  const body = rec.request.body as ResponseRequestBody | null;
  if (!body?.input) return;

  const sid = rec.sid;
  const contextKey = 'main';
  if (!sessionMetaWritten.has(sid)) {
    sessionMetaWritten.add(sid);
    writeMeta(sessionMetaPath(sid), {
      x_cannbay: xCannbay('cc-session-meta', {
        producer: 'cpx',
        framework: 'codex',
        protocol: 'responses',
        sid,
      }),
    });
  }

  const items = body.input;
  const prev = prevInputCount.get(contextKey) ?? 0;
  const roundIndex = roundCount.get(contextKey) ?? 0;
  for (let i = prev; i < items.length; i++) {
    const line = convertInputItem(items[i]);
    if (line) {
      (line.x_cannbay as { data: { roundIndex: number } }).data.roundIndex = roundIndex;
      appendClaudeLine(sessionFilePath(sid), { ...line, timestamp: toISO(rec.receivedAt) });
    }
  }
  prevInputCount.set(contextKey, items.length);

  const contentBlocks = Array.isArray(rec.response.content) ? rec.response.content as AnthropicContentBlock[] : [];
  roundCount.set(contextKey, roundIndex + 1);

  appendClaudeLine(sessionFilePath(sid), {
    type: 'assistant',
    message: {
      role: 'assistant',
      id: crypto.randomUUID(),
      content: contentBlocks,
      model: rec.response.model,
      usage: mapUsage(rec.response.usage),
    },
    timestamp: toISO(rec.completedAt),
    system: body.instructions ?? undefined,
    tools: Array.isArray(body.tools) ? body.tools : undefined,
    source: proxySourceMarker(rec.protocol),
    x_cannbay: xCannbay('cc-wire-round', {
      roundIndex,
      protocol: 'responses',
      system: body.instructions ?? undefined,
      tools: Array.isArray(body.tools) ? body.tools : undefined,
      requestParams: { model: body.model ?? rec.request.model },
      latencyMs: rec.latencyMs,
      ttftMs: rec.ttftMs ?? undefined,
      stopReason: rec.response.stop_reason ?? undefined,
      status: rec.response.status,
    }),
  });
}
