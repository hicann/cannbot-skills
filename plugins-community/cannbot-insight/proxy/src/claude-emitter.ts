// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// claude-emitter: convert a captured wire-record (ProxyRecord) into claude's
// session-jsonl line format, so cannbot-insight's existing claude-jsonl adapter
// + its native subagent pipeline (subagents/ + meta.json) can ingest the proxy's
// output with zero cannbot-insight changes.
//
// Each ProxyRecord → delta-emit new user messages (incl. tool_result) as
// {type:"user"} lines + the response as one {type:"assistant"} line. The
// assistant line carries extension fields `system` + `tools` (verbatim
// request.body.system / request.body.tools) for the Full Context reader; the
// claude-jsonl adapter ignores these extra fields.
//
// Subagent records (cc_is_subagent=true) are routed to
// <PROXY_DIR>/<sid>/subagents/<subId>.jsonl + a one-time <subId>.meta.json
// carrying the spawning Agent tool_use's id (toolUseId) + name + agentType.

import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import type { ProxyRecord, Protocol, AnthropicContentBlock, AnthropicUsage, OpenAIUsage } from './types';
import { sessionFilePath, subagentFilePath, subagentMetaPath, appendClaudeLine, writeMeta, PROXY_DIR } from './writer';

// Hot switch: `cpx config dedup on|off` takes effect on a LIVE session. The
// config file is re-read on every emit (emit runs once per wire record —
// seconds apart — so the tiny read is free). The env var is an explicit
// override used by tests: '1'=on, '0'=off, unset=hot-read the file.
function dedupEnabled(): boolean {
  const env = process.env.CANNBOT_PROXY_DEDUP_INJECTION;
  if (env === '1') return true;
  if (env === '0') return false;
  try {
    const dir = process.env.CANNBOT_PROXY_DIR ?? PROXY_DIR;
    return JSON.parse(fs.readFileSync(path.join(path.dirname(dir), 'cpx-config.json'), 'utf-8')).dedupInjection === true;
  } catch {
    return false;
  }
}

// Provenance marker written on EVERY emitted line: the capture file stays a
// valid claude session jsonl (extra fields are ignored by consumers), but the
// file itself declares where it came from — so provenance survives any
// rename/move, independent of directory conventions. The marker KEEPS the
// original agent name (claude-proxy / opencode-proxy, by wire protocol).
// cannbot-insight reads this at import time to label the session.
export function proxySourceMarker(protocol: Protocol): string {
  return protocol === 'openai' ? 'opencode-proxy' : 'claude-proxy';
}

interface RequestBody {
  messages?: { role: string; content?: string | AnthropicContentBlock[] }[];
  system?: string | AnthropicContentBlock[];
  tools?: unknown;
}

function extractSystemText(system: string | AnthropicContentBlock[] | undefined): string | null {
  if (!system) return null;
  if (typeof system === 'string') return system || null;
  const parts: string[] = [];
  for (const b of system) {
    if (b.type === 'text' && b.text) parts.push(b.text);
  }
  return parts.length > 0 ? parts.join('\n\n') : null;
}

// The reassembler (AnthropicReassembler / OpenAIReassembler) already normalizes
// wire usage to Anthropic shape ({input_tokens, output_tokens,
// cache_read_input_tokens, cache_creation_input_tokens}). Pass it through
// verbatim — re-mapping here would double-convert and zero OpenAI-protocol
// usage (see opencode-emitter). For claude (anthropic) this is a no-op.
function mapUsage(usage: AnthropicUsage | OpenAIUsage | null): AnthropicUsage | undefined {
  if (!usage) return undefined;
  return usage as AnthropicUsage;
}

function isTitleGenRecord(rec: ProxyRecord): boolean {
  const body = rec.request.body as RequestBody | null;
  const sysText = extractSystemText(body?.system);
  return !!sysText && sysText.includes('sentence-case title');
}

function isSubagentRecord(rec: ProxyRecord): boolean {
  const body = rec.request.body as RequestBody | null;
  const sysText = extractSystemText(body?.system);
  return !!sysText && sysText.includes('cc_is_subagent=true');
}

// The subagent's task prompt = first non-system-reminder text block of its
// first user message. Identical to the spawning Agent tool_use input.prompt,
// so it groups a subagent's records + correlates to the dispatch meta.
function subagentTask(rec: ProxyRecord): string | null {
  const body = rec.request.body as RequestBody | null;
  const msgs = body?.messages;
  if (!msgs || !msgs.length) return null;
  const m0 = msgs[0];
  if (m0.role !== 'user') return null;
  if (typeof m0.content === 'string') {
    const t = m0.content.trimStart();
    return (t && !t.startsWith('<system-reminder>') && !t.startsWith('The user stepped away')) ? m0.content : null;
  }
  if (Array.isArray(m0.content)) {
    for (const b of m0.content) {
      if (b.type !== 'text' || !b.text) continue;
      const tr = b.text.trimStart();
      if (tr.startsWith('<system-reminder>') || tr.startsWith('The user stepped away')) continue;
      return b.text;
    }
  }
  return null;
}

// --- per-proxy-process state (one cpx run = one proxy process) ---
// Delta-emit by OCCURRENCE COUNT, not by array length: claude-code rewrites
// the messages array on stepped-away recaps / edits / compaction, so a
// message-count cursor silently skips new messages that land inside the
// already-consumed range. A multiset (hash → emitted count) detects new
// messages regardless of rewrite, shrink, or same-length reordering.
const emittedCounts = new Map<string, Map<string, number>>(); // contextKey → hash → emitted count
// contextKey → md5 fingerprints of injection system messages emitted IN FULL
const emittedInjections = new Map<string, Set<string>>();
const agentDispatch = new Map<string, { toolUseId: string | null; name: string | null; type: string | null }>();
const subIdForTask = new Map<string, string>();
const metaWritten = new Set<string>();

function subIdFor(task: string): string {
  let id = subIdForTask.get(task);
  if (!id) {
    id = 'sub-' + crypto.createHash('md5').update(task).digest('hex').slice(0, 12);
    subIdForTask.set(task, id);
  }
  return id;
}

function toISO(ts: number): string {
  return new Date(ts).toISOString();
}

// Joined text of a message content (string or text blocks).
function textOf(content: string | AnthropicContentBlock[] | undefined): string | null {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return null;
  const parts: string[] = [];
  for (const b of content) {
    if (b.type === 'text' && b.text) parts.push(b.text);
  }
  return parts.join('\n');
}

// An "injection" is the agent-registry / skills-list system message claude-code
// re-appends on later rounds. Identified by its opening line so ordinary
// system messages are never deduped.
function isInjectionText(text: string): boolean {
  return text.startsWith('Available agent types for the Agent tool')
    || text.startsWith('The following skills are available');
}

export function emit(rec: ProxyRecord): void {
  if (isTitleGenRecord(rec)) return;
  const body = rec.request.body as RequestBody | null;
  if (!body) return;
  const sid = rec.sid;
  const isSub = isSubagentRecord(rec);

  // Collect Agent-tool dispatch meta (prompt → {toolUseId, name, type}) from
  // ALL records (main + subagent) — a subagent that spawns its OWN subagent
  // emits that Agent tool_use inside an isSub record; gating on !isSub would
  // miss nested dispatches, leaving meta.json with null toolUseId/name/type.
  // Keyed by prompt (unique enough across contexts — each Agent tool_use has
  // a distinct task prompt).
  {
    const blocks = Array.isArray(rec.response.content) ? rec.response.content as AnthropicContentBlock[] : [];
    for (const b of blocks) {
      if (b.type !== 'tool_use' || b.name !== 'Agent' || !b.input) continue;
      const inp = b.input as { description?: string; subagent_type?: string; prompt?: string };
      if (inp.prompt && !agentDispatch.has(inp.prompt)) {
        agentDispatch.set(inp.prompt, {
          toolUseId: b.id ?? null,
          name: inp.description ?? null,
          type: inp.subagent_type ?? null,
        });
      }
    }
  }

  // Resolve context + target file.
  let contextKey: string;
  let targetFile: string;
  if (!isSub) {
    contextKey = 'main';
    targetFile = sessionFilePath(sid);
  } else {
    const task = subagentTask(rec);
    const subId = task ? subIdFor(task) : 'sub-unknown';
    contextKey = subId;
    targetFile = subagentFilePath(sid, subId);
    if (task && !metaWritten.has(subId)) {
      metaWritten.add(subId);
      const meta = agentDispatch.get(task);
      writeMeta(subagentMetaPath(sid, subId), {
        toolUseId: meta?.toolUseId ?? null,
        name: meta?.name ?? null,
        agentType: meta?.type ?? null,
      });
    }
  }

  // Delta-emit new non-assistant messages (user incl. tool_result, system).
  // Skip role:"assistant" — those are PRIOR responses already emitted as their
  // own rec's assistant line; re-emitting would duplicate. collectAllToolResults
  // reads tool_result from these user lines.
  //
  // Injection dedup (cpx config dedup, DEFAULT OFF, HOT — reads the config
  // file on every emit so toggling affects a live session): claude-code
  // re-appends an IDENTICAL agent-registry / skills-list system message on
  // later rounds. The FIRST copy is emitted in full; each later byte-identical
  // occurrence is replaced by a compact `[已压缩]` marker line carrying
  // originalChars, so the capture stays small while still recording that an
  // injection happened. Changed content is NOT deduped — it emits in full and
  // updates fingerprints. Fingerprints are tracked PER CONTEXT (main / each
  // subagent), so identical injections in different contexts are each kept in
  // full.
  const dedup = dedupEnabled();
  const msgs = body.messages ?? [];
  const counts = emittedCounts.get(contextKey) ?? new Map<string, number>();
  const seen = new Map<string, number>();
  const injections = dedup ? (emittedInjections.get(contextKey) ?? new Set<string>()) : new Set<string>();
  const markedInRound = new Set<string>(); // 每轮每个 fingerprint 只发一个 [已压缩] 标记
  for (const m of msgs) {
    if (m.role === 'assistant') continue;
    const hash = m.role + ':' + JSON.stringify(m.content ?? '');
    const occ = seen.get(hash) ?? 0;
    seen.set(hash, occ + 1);
    // Injection dedup: check fingerprint BEFORE the multiset skip. Otherwise
    // the multiset silently drops byte-identical injection re-sends (occ < counts)
    // before the fingerprint block runs → [已压缩] marker never emits, dedup
    // stats stay 0. Fingerprints only apply to role:'system' injection messages
    // (registry/skills); non-injection messages still use the multiset below.
    const injectionText = dedup && m.role === 'system' ? textOf(m.content) : null;
    const fingerprint = injectionText && isInjectionText(injectionText)
      ? crypto.createHash('md5').update(injectionText).digest('hex')
      : null;
    if (fingerprint && injections.has(fingerprint)) {
      if (markedInRound.has(fingerprint)) {
        // 同轮已发过 [已压缩] 标记，后续副本静默丢弃（不计入 counts，
        // 避免下一轮误判为「已发」而漏标）
        continue;
      }
      markedInRound.add(fingerprint);
      appendClaudeLine(targetFile, {
        type: 'system',
        message: { role: 'system', content: [{ type: 'text', text: `[已压缩] 注入上下文与之前一份完全相同（${injectionText!.length} chars），原文未重复记录` }] },
        timestamp: toISO(rec.receivedAt),
        deduped: true,
        originalChars: injectionText!.length,
        source: proxySourceMarker(rec.protocol),
      });
      counts.set(hash, occ + 1);
      continue;
    }
    // Multiset skip: non-injection duplicates already emitted (silent drop)
    if (occ < (counts.get(hash) ?? 0)) continue;
    appendClaudeLine(targetFile, {
      type: m.role === 'system' ? 'system' : 'user',
      message: { role: m.role, content: m.content ?? '' },
      timestamp: toISO(rec.receivedAt),
      source: proxySourceMarker(rec.protocol),
    });
    counts.set(hash, occ + 1);
    if (fingerprint) injections.add(fingerprint);
  }
  emittedCounts.set(contextKey, counts);
  if (dedup) emittedInjections.set(contextKey, injections);

  // Emit the response as one assistant line. Extension fields `system` + `tools`
  // (verbatim request body) ride on this line; the claude-jsonl adapter ignores
  // them, the Full Context reader consumes them.
  // Wire timing: `duration_ms` (native claude field, adapter reads it →
  // interaction.latency → turn-split → Turn.latencyMs, so perf tab sees it
  // WITHOUT an OCP-bending post-write) + `stopReason` (adapter reads →
  // finish_reason → Turn.finishReason). `ttftMs` has no pipeline field, so it
  // stays a pure extension field → readWireEnrichments → turns API override.
  const contentBlocks = Array.isArray(rec.response.content) ? rec.response.content as AnthropicContentBlock[] : [];
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
    system: body.system ?? undefined,
    tools: Array.isArray(body.tools) ? body.tools : undefined,
    duration_ms: rec.latencyMs,
    stopReason: rec.response.stop_reason ?? undefined,
    ttftMs: rec.ttftMs ?? undefined,
    source: proxySourceMarker(rec.protocol),
  });
}
