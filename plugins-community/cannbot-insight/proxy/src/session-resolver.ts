// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { Protocol } from './types';

export interface ResolvedSession {
  sid: string;
  protocol: Protocol | null;
}

const SID_RE = /^\/s\/([^/]+)(?:\/(v1\/.*))?$/;

export function sidFromPath(urlPath: string): string | null {
  const m = urlPath.match(SID_RE);
  return m ? m[1] : null;
}

export function protocolFromPath(urlPath: string): Protocol | null {
  if (urlPath.includes('/v1/responses')) return 'responses';
  if (urlPath.includes('/v1/messages') || urlPath.includes('/messages')) return 'anthropic';
  if (urlPath.includes('/v1/chat/completions') || urlPath.includes('/chat/completions')) return 'openai';
  return null;
}

function fingerprintUserMessage(body: unknown): string | null {
  if (!body || typeof body !== 'object') return null;
  const messages = (body as { messages?: unknown }).messages;
  if (!Array.isArray(messages)) return null;
  for (const msg of messages) {
    if (!msg || typeof msg !== 'object') continue;
    const m = msg as { role?: string; content?: unknown };
    if (m.role !== 'user') continue;
    const text = extractUserText(m.content);
    if (text) return hash(text.slice(0, 500));
  }
  return null;
}

function extractUserText(content: unknown): string | null {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return null;
  for (const block of content) {
    if (block && typeof block === 'object') {
      const b = block as { type?: string; text?: string };
      if (b.type === 'text' && typeof b.text === 'string') return b.text;
    }
  }
  return null;
}

function hash(input: string): string {
  let h = 0x811c9dc5;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return 'fp-' + (h >>> 0).toString(16).padStart(8, '0');
}

export interface ResolveResult {
  sid: string;
  protocol: Protocol | null;
}

// claude-code declares its REAL session id on every request via this header
// (mirrored in body metadata.user_id — verified identical). Routing by it over
// the env-pinned default names capture files by claude's own session id, so
// proxy captures and claude's native jsonl join 1:1; /resume switching ids
// mid-run rolls to a new capture file naturally. Subagent requests carry the
// MAIN session id (verified), so subagent routing stays on the emitter's
// system-prompt check — orthogonal to sid resolution.
const CLAUDE_SID_HEADER = 'x-claude-code-session-id';
const SAFE_SID = /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/;

export function resolveSession(
  urlPath: string,
  headerGet: (name: string) => string | null,
  body: unknown,
  defaultSid?: string
): ResolveResult {
  const protocol = protocolFromPath(urlPath);

  const claudeSid = protocol === 'anthropic' ? headerGet(CLAUDE_SID_HEADER) : null;
  if (claudeSid && SAFE_SID.test(claudeSid)) {
    return { sid: claudeSid, protocol };
  }

  // opencode declares its session id via x-session-id on EVERY request, and
  // child sessions additionally carry x-parent-session-id (routed in the
  // emitter). Same pattern as claude's header: --continue/--session resume
  // keeps the id (capture appends), --fork/new mints one (new capture file).
  // Verified on opencode 1.17.9.
  const openaiSid = protocol === 'openai' ? headerGet('x-session-id') : null;
  if (openaiSid && SAFE_SID.test(openaiSid)) {
    return { sid: openaiSid, protocol };
  }

  // Per-session port mode: proxy started with a pinned session id.
  // No path/header/fingerprint needed — this port IS this session.
  if (defaultSid) {
    return { sid: defaultSid, protocol };
  }

  const headerSid = headerGet('x-session-id');
  const pathSid = sidFromPath(urlPath);

  let sid: string | null = headerSid ?? pathSid;
  if (!sid) {
    sid = fingerprintUserMessage(body);
  }
  if (!sid) {
    sid = 'anon-' + (Date.now() % 0xffffffff).toString(16);
  }

  return { sid, protocol };
}
