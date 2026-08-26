// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// opencode-emitter: convert a captured opencode wire-record (ProxyRecord,
// protocol='openai') into extended claude-format jsonl lines, so cannbot-insight's
// existing claude-jsonl adapter can ingest it with zero insight changes.
//
// Independent from claude-emitter.ts — the two share NO conversion logic because
// the OpenAI wire format differs from Anthropic in every relevant dimension:
//   - system: OpenAI puts it in messages[0] role:system (no top-level `system`)
//   - tools:  OpenAI uses {type:"function",function:{name,description,parameters}}
//   - tool results: OpenAI uses role:"tool" + tool_call_id (claude uses tool_result
//     content blocks inside user messages)
//   - subagents: OpenCode child sessions carry a DIFFERENT x-session-id header
//     (claude relies on cc_is_subagent system-prompt text + task-prompt hashing)
//
// Output format is the SAME extended claude-format as claude-emitter (user/assistant
// lines + extension `system`/`tools` fields + subagents/<subId>.jsonl + meta.json),
// so the single claude-jsonl adapter consumes both. The contract is the format;
// the conversion logic is independent per the "two independent frameworks" rule.

import crypto from 'node:crypto';
import type { ProxyRecord, AnthropicContentBlock, AnthropicUsage, OpenAIUsage } from './types';
import { sessionFilePath, subagentFilePath, subagentMetaPath, appendClaudeLine, writeMeta } from './writer';

interface OpenAIMessage {
  role: string;
  content?: string | OpenAIContentBlock[] | null;
  tool_calls?: OpenAIToolCall[];
  tool_call_id?: string;
}
interface OpenAIContentBlock {
  type: string;
  text?: string;
}
interface OpenAIToolCall {
  id: string;
  type: string;
  function: { name: string; arguments: string };
}
interface OpenAITool {
  type: string;
  function: { name: string; description?: string; parameters?: unknown };
}
interface OpenAIRequestBody {
  messages?: OpenAIMessage[];
  tools?: OpenAITool[];
  model?: string;
}

// --- per-proxy-process state (one cpx run = one proxy process) ---
const prevMsgCount = new Map<string, number>(); // contextKey → last request.messages.length (delta)
const mainXSessionId = new Map<string, string>(); // insight sid → first-seen opencode x-session-id
const subMetaWritten = new Set<string>(); // subIds that already got a meta.json
// task signature (Task tool input.prompt/description) → dispatch meta, collected
// from main agent responses so child-session meta.json can link back to the spawn.
const taskDispatch = new Map<string, { toolUseId: string | null; name: string | null; agentType: string | null }>();

function extractSystemFromMessages(messages: OpenAIMessage[] | undefined): string | null {
  if (!messages) return null;
  const parts: string[] = [];
  for (const m of messages) {
    if (m.role !== 'system') continue;
    if (typeof m.content === 'string') {
      if (m.content) parts.push(m.content);
    } else if (Array.isArray(m.content)) {
      for (const b of m.content) {
        if (b.type === 'text' && b.text) parts.push(b.text);
      }
    }
  }
  return parts.length > 0 ? parts.join('\n\n') : null;
}

function convertTools(tools: OpenAITool[] | undefined): { name: string; description: string }[] | undefined {
  if (!Array.isArray(tools) || tools.length === 0) return undefined;
  const out = tools.map(t => ({
    name: t.function?.name ?? '',
    description: t.function?.description ?? '',
  })).filter(t => t.name);
  return out.length > 0 ? out : undefined;
}

function isTitleGenRecord(rec: ProxyRecord): boolean {
  const body = rec.request.body as OpenAIRequestBody | null;
  if (!body?.messages) return false;
  for (const m of body.messages) {
    if (m.role !== 'system') continue;
    const text = typeof m.content === 'string' ? m.content : '';
    if (text.includes('title generator')) return true;
  }
  return false;
}

function toISO(ts: number): string {
  return new Date(ts).toISOString();
}

function mapUsage(usage: AnthropicUsage | OpenAIUsage | null): AnthropicUsage | undefined {
  if (!usage) return undefined;
  // The OpenAIReassembler / reassembleOpenAIJson ALREADY normalized OpenAI wire
  // usage (prompt_tokens/completion_tokens → input_tokens/output_tokens) before
  // it reaches the emitter (see stream-reassembler.test.ts). Re-mapping here
  // would read prompt_tokens (now absent) and zero everything — the original
  // "usage all 0 → no System(hidden)/Other context in LLM Input" bug. Pass the
  // already-Anthropic usage through verbatim.
  return usage as AnthropicUsage;
}

// OpenAI user content (string | content-block array) → claude user content
// (string | AnthropicContentBlock[]). Only text blocks are mapped; non-text
// (image_url etc.) are dropped to keep the format contract simple.
function convertUserContent(content: string | OpenAIContentBlock[] | null | undefined): string | AnthropicContentBlock[] {
  if (content == null) return '';
  if (typeof content === 'string') return content;
  const out: AnthropicContentBlock[] = [];
  for (const b of content) {
    if (b.type === 'text' && b.text) out.push({ type: 'text', text: b.text });
  }
  return out.length > 0 ? out : '';
}

export function emit(rec: ProxyRecord): void {
  if (isTitleGenRecord(rec)) return;
  const body = rec.request.body as OpenAIRequestBody | null;
  if (!body?.messages) return;

  const sid = rec.sid;
  const xid = rec.xSessionId ?? null;
  const parentId = rec.parentSessionId ?? null;

  // Routing (verified on opencode 1.17.9): child sessions declare themselves
  // via x-parent-session-id — deterministic, and it names the parent's REAL
  // session id, so the child files under the parent's tree no matter which
  // session the run started in (resume-safe). Legacy fallback: pre-header
  // opencode routed every record to one shared sid; there, first-seen
  // x-session-id is the main session and any other is a child.
  let isSub: boolean;
  let mainSid: string;
  if (parentId) {
    isSub = true;
    mainSid = parentId;
  } else if (xid && xid !== sid) {
    if (!mainXSessionId.has(sid)) mainXSessionId.set(sid, xid);
    isSub = xid !== mainXSessionId.get(sid);
    mainSid = sid;
  } else {
    isSub = false;
    mainSid = sid;
  }

  // Collect Task-tool dispatch meta from main agent responses so child-session
  // meta.json can link back to the spawning tool_use id.
  if (!isSub) {
    const blocks = Array.isArray(rec.response.content) ? rec.response.content as AnthropicContentBlock[] : [];
    for (const b of blocks) {
      if (b.type !== 'tool_use' || !b.name) continue;
      const name = b.name.toLowerCase();
      if (name !== 'task' && !name.includes('task')) continue;
      const inp = b.input as { description?: string; prompt?: string; subagent_type?: string; agent?: string } | undefined;
      const sig = inp?.prompt ?? inp?.description ?? b.id ?? '';
      if (sig && !taskDispatch.has(sig)) {
        taskDispatch.set(sig, {
          toolUseId: b.id ?? null,
          name: inp?.description ?? null,
          agentType: inp?.subagent_type ?? inp?.agent ?? null,
        });
      }
    }
  }

  // Resolve context key + target file. contextKey is per-session (main: its
  // own sid) so a mid-run resume/fork to another session starts fresh delta
  // state instead of inheriting the previous session's consumed count.
  let contextKey: string;
  let targetFile: string;
  let subId: string | null = null;
  if (!isSub) {
    contextKey = mainSid;
    targetFile = sessionFilePath(mainSid);
  } else {
    subId = xid!;
    contextKey = subId;
    targetFile = subagentFilePath(mainSid, subId);
  }

  // Delta-emit new non-assistant, non-system messages. Skip role:"assistant"
  // (prior responses are emitted as their own rec's assistant line; re-emitting
  // would duplicate). Skip role:"system" (carried verbatim as the `system`
  // extension field on the assistant line). role:"tool" → user line with a
  // tool_result block (the claude-jsonl adapter's collectAllToolResults reads
  // tool_result from user lines).
  const msgs = body.messages;
  const prev = prevMsgCount.get(contextKey) ?? 0;
  for (let i = prev; i < msgs.length; i++) {
    const m = msgs[i];
    if (m.role === 'assistant') continue;
    if (m.role === 'system') continue;
    if (m.role === 'tool') {
      const tcId = m.tool_call_id ?? '';
      let tcContent: string;
      if (typeof m.content === 'string') tcContent = m.content;
      else if (Array.isArray(m.content)) tcContent = JSON.stringify(m.content);
      else tcContent = '';
      appendClaudeLine(targetFile, {
        type: 'user',
        message: {
          role: 'user',
          content: [{ type: 'tool_result', tool_use_id: tcId, content: tcContent }],
        },
        timestamp: toISO(rec.receivedAt),
      });
    } else {
      appendClaudeLine(targetFile, {
        type: m.role === 'user' ? 'user' : m.role,
        message: { role: m.role, content: convertUserContent(m.content) },
        timestamp: toISO(rec.receivedAt),
      });
    }
  }
  prevMsgCount.set(contextKey, msgs.length);

  // Subagent: write meta.json once (link toolUseId best-effort by matching the
  // child session's first user message to a collected Task dispatch signature).
  if (isSub && subId && !subMetaWritten.has(subId)) {
    subMetaWritten.add(subId);
    let meta: { toolUseId: string | null; name: string | null; agentType: string | null } =
      { toolUseId: null, name: null, agentType: null };
    for (const m of msgs) {
      if (m.role !== 'user') continue;
      const t = typeof m.content === 'string' ? m.content : '';
      if (!t) continue;
      const d = taskDispatch.get(t);
      if (d) { meta = d; break; }
    }
    writeMeta(subagentMetaPath(mainSid, subId), meta);
  }

  // Emit the response as one assistant line. `system` = the opencode system
  // VERBATIM (instructions + `Instructions from:` memory + `<available_skills>`
  // skills) — the proxy is a pure capture layer; parsing the opencode system
  // structure into Memory/Skills panels is done by the standalone
  // opencode-context-parser, not here (jsonl-driven design: capture ≠ interpret).
  // `tools` are normalized OpenAI function → {name,description} for any consumer.
  const systemText = extractSystemFromMessages(body.messages);
  const tools = convertTools(body.tools);
  const contentBlocks = Array.isArray(rec.response.content)
    ? rec.response.content as AnthropicContentBlock[]
    : [];

  appendClaudeLine(targetFile, {
    type: 'assistant',
    message: {
      role: 'assistant',
      id: crypto.randomUUID(),
      content: contentBlocks,
      model: rec.response.model,
      usage: mapUsage(rec.response.usage),
    },
    timestamp: toISO(rec.completedAt),
    system: systemText ?? undefined,
    tools,
  });
}
