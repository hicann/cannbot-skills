// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import http from 'node:http';
import fs from 'node:fs';
import { resolveSession } from './session-resolver';
import { createReassembler, type ReassembledResponse } from './stream-reassembler';
import { ensureProxyDir, sessionFilePath, appendSid } from './writer';
import { emit as claudeEmit } from './claude-emitter';
import { emit as opencodeEmit } from './opencode-emitter';
import { redactRecord } from './redactor';
import type { AnthropicContentBlock, AnthropicUsage, Protocol, ProxyRecord } from './types';

// Route a captured record to the right emitter. The two emitters are
// independent (no shared conversion logic) per the "two independent frameworks"
// rule; both produce the same extended claude-format so the single claude-jsonl
// adapter consumes either. First emit per sid lands in the <pinned>.sids
// manifest so cpx knows at exit which capture files this run produced.
const emittedSids = new Set<string>();
function dispatchEmit(rec: ProxyRecord): void {
  // 落盘前清洗（唯一咽喉：所有 emitter 的数据都经这里进 jsonl）
  redactRecord(rec);
  // Child sessions file under their PARENT's tree (x-parent-session-id), so
  // only parent/main sids belong in the manifest cpx imports from.
  if (SESSION_ID && !rec.parentSessionId && !emittedSids.has(rec.sid)) {
    emittedSids.add(rec.sid);
    try { appendSid(SESSION_ID, rec.sid); } catch { /* best-effort */ }
  }
  if (rec.protocol === 'anthropic') claudeEmit(rec);
  else opencodeEmit(rec);
}

const PORT = parseInt(process.env.CANNBOT_PROXY_PORT ?? '0', 10) || 0;
const SESSION_ID = process.env.CANNBOT_PROXY_SESSION_ID ?? '';
const ANTHROPIC_UPSTREAM = process.env.CANNBOT_PROXY_ANTHROPIC_UPSTREAM ?? 'https://api.anthropic.com';
const OPENAI_UPSTREAM = process.env.CANNBOT_PROXY_OPENAI_UPSTREAM ?? 'https://api.openai.com';
// opencode per-provider upstream map (JSON {providerId: baseURL}), discovered
// from the opencode binary by cpx-cli. Lets the proxy forward /<providerId>/…
// to each provider's REAL upstream (dashscope, bigmodel, …) — fully
// transparent multi-provider routing, no single-upstream limitation.
const PROVIDER_UPSTREAMS: Record<string, string> = (() => {
  try { return JSON.parse(process.env.CANNBOT_PROXY_PROVIDER_UPSTREAMS ?? '{}') as Record<string, string>; }
  catch { return {}; }
})();

const STRIP_REQ_HEADERS = new Set([
  'host', 'content-length', 'connection', 'transfer-encoding',
]);

ensureProxyDir();

function upstreamBase(protocol: Protocol): string {
  return protocol === 'anthropic' ? ANTHROPIC_UPSTREAM : OPENAI_UPSTREAM;
}

// Resolve the real upstream URL for a request. opencode requests carry a
// path prefix injected by cpx (/<providerId>/chat/completions); route those to
// the provider's real upstream (from PROVIDER_UPSTREAMS) with the prefix
// stripped. Everything else (claude, openai/generic profiles, or opencode
// providers not in the map) falls back to the single upstream + sid strip.
function resolveUpstreamUrl(protocol: Protocol, urlPath: string): string {
  if (Object.keys(PROVIDER_UPSTREAMS).length > 0) {
    const m = urlPath.match(/^\/([^/]+)(\/.*)$/);
    if (m && PROVIDER_UPSTREAMS[m[1]]) {
      return PROVIDER_UPSTREAMS[m[1]].replace(/\/$/, '') + m[2];
    }
  }
  return upstreamBase(protocol) + stripSessionPrefix(urlPath);
}

function buildForwardHeaders(req: http.IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(req.headers)) {
    if (v == null) continue;
    if (STRIP_REQ_HEADERS.has(k.toLowerCase())) continue;
    out[k] = Array.isArray(v) ? v.join(', ') : v;
  }
  return out;
}

function readBody(req: http.IncomingMessage): Promise<Buffer> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    req.on('data', (c: Buffer) => chunks.push(c));
    req.on('end', () => resolve(Buffer.concat(chunks)));
    req.on('error', reject);
  });
}

async function handleProxy(req: http.IncomingMessage, res: http.ServerResponse, body: Buffer): Promise<void> {
  let parsedBody: unknown = null;
  try {
    parsedBody = body.length > 0 ? JSON.parse(body.toString('utf-8')) : null;
  } catch {
    parsedBody = null;
  }

  const { sid, protocol } = resolveSession(req.url ?? '/', (name) => {
    const v = req.headers[name.toLowerCase()];
    return Array.isArray(v) ? v[0] ?? null : v ?? null;
  }, parsedBody, SESSION_ID || undefined);
  if (!protocol) {
    res.writeHead(404, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'Unknown protocol path' }));
    return;
  }

  const upstreamUrl = resolveUpstreamUrl(protocol, req.url ?? '');
  const forwardHeaders = buildForwardHeaders(req);
  forwardHeaders['content-type'] = req.headers['content-type'] ?? 'application/json';

  const receivedAt = Date.now();
  const isSseRequested = parsedBody != null &&
    (parsedBody as { stream?: unknown }).stream === true;

  let upstreamRes: Response;
  try {
    upstreamRes = await fetch(upstreamUrl, {
      method: req.method ?? 'POST',
      headers: forwardHeaders,
      body: body.length > 0 ? new Uint8Array(body) : undefined,
    });
  } catch (err) {
    res.writeHead(502, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ error: 'upstream unreachable', detail: String(err) }));
    return;
  }

  const respContentType = upstreamRes.headers.get('content-type') ?? '';
  const isSseResponse = respContentType.includes('text/event-stream');

  for (const [k, v] of upstreamRes.headers) {
    if (k.toLowerCase() === 'content-encoding' || k.toLowerCase() === 'content-length') continue;
    res.setHeader(k, v);
  }

  if (isSseResponse || isSseRequested) {
    await handleStream(req, res, upstreamRes, sid, protocol, parsedBody, receivedAt);
  } else {
    await handleNonStream(res, upstreamRes, sid, protocol, parsedBody, receivedAt);
  }
}

function stripSessionPrefix(url: string): string {
  const match = url.match(/^\/s\/[^/]+(\/v1\/.*)$/);
  if (match) return match[1];
  if (url.startsWith('/v1/')) return url;
  return url;
}

async function handleStream(
  req: http.IncomingMessage,
  res: http.ServerResponse,
  upstreamRes: Response,
  sid: string,
  protocol: Protocol,
  requestBody: unknown,
  receivedAt: number
): Promise<void> {
  res.writeHead(upstreamRes.status, { 'content-type': res.getHeader('content-type') ?? 'text/event-stream' });

  const reassembler = createReassembler(protocol);
  const reader = upstreamRes.body?.getReader();
  if (!reader) {
    res.end();
    return;
  }

  try {
    const decoder = new TextDecoder();
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (value) {
        reassembler.feed(decoder.decode(value, { stream: true }));
        res.write(value);
      }
    }
    // flush trailing multibyte bytes
    reassembler.feed(decoder.decode());
  } catch {
    // client disconnect — still record what we captured
  }

  // Persist the record BEFORE res.end() so the client receiving the response
  // (and possibly exiting immediately, causing cpx-cli to SIGTERM this proxy)
  // cannot race the disk write. Order: reassemble → append → then close.
  // emit() 调 appendFileSync 可能抛（ENOSPC/EIO）—— 不能让它阻断 res.end，
  // 否则客户端连接悬挂；也不能让异常冒泡到外层 catch（headers 已发 →
  // writeHead(500) 二次抛 ERR_HTTP_HEADERS_SENT → 进程崩）
  const completedAt = Date.now();
  const reassembled = reassembler.result();
  const ttftMs = reassembled.firstTokenAt != null ? reassembled.firstTokenAt - receivedAt : null;
  try {
    dispatchEmit(buildRecord(sid, protocol, requestBody, upstreamRes.status, reassembled, receivedAt, completedAt, req, ttftMs));
  } catch (e) {
    console.error('[proxy] emit failed (capture may be incomplete):', e instanceof Error ? e.message : e);
  }
  res.end();
}

async function handleNonStream(
  res: http.ServerResponse,
  upstreamRes: Response,
  sid: string,
  protocol: Protocol,
  requestBody: unknown,
  receivedAt: number
): Promise<void> {
  const text = await upstreamRes.text();

  let reassembled: ReassembledResponse;
  try {
    const json = JSON.parse(text);
    reassembled = protocol === 'anthropic'
      ? reassembleAnthropicJson(json)
      : reassembleOpenAIJson(json);
  } catch {
    reassembled = { model: null, stop_reason: null, content: [], usage: null };
  }

  // Persist before sending the response to the client (same race fix as handleStream).
  const completedAt = Date.now();
  dispatchEmit(buildRecord(sid, protocol, requestBody, upstreamRes.status, reassembled, receivedAt, completedAt, null, null));

  res.writeHead(upstreamRes.status, { 'content-type': upstreamRes.headers.get('content-type') ?? 'application/json' });
  res.end(text);
}

function reassembleAnthropicJson(json: Record<string, unknown>): ReassembledResponse {
  return {
    model: (json.model as string) ?? null,
    stop_reason: (json.stop_reason as string) ?? null,
    content: Array.isArray(json.content) ? json.content as AnthropicContentBlock[] : [],
    usage: (json.usage as AnthropicUsage) ?? null,
  };
}

function reassembleOpenAIJson(json: Record<string, unknown>): ReassembledResponse {
  const choices = json.choices as Array<Record<string, unknown>> | undefined;
  const choice = choices?.[0];
  const msg = choice?.message as Record<string, unknown> | undefined;
  const content: AnthropicContentBlock[] = [];
  if (typeof msg?.content === 'string' && msg.content) {
    content.push({ type: 'text', text: msg.content });
  }
  const toolCalls = msg?.tool_calls as Array<Record<string, unknown>> | undefined;
  if (Array.isArray(toolCalls)) {
    for (const tc of toolCalls) {
      let input: Record<string, unknown> = {};
      const fn = tc.function as { arguments?: string; name?: string } | undefined;
      if (fn?.arguments) {
        try { input = JSON.parse(fn.arguments); } catch { input = { _raw: fn.arguments }; }
      }
      content.push({ type: 'tool_use', id: tc.id as string, name: fn?.name, input });
    }
  }
  const u = json.usage as Record<string, unknown> | undefined;
  const usage = u ? {
    input_tokens: u.prompt_tokens as number | undefined,
    output_tokens: u.completion_tokens as number | undefined,
    cache_read_input_tokens: (u.prompt_tokens_details as { cached_tokens?: number } | undefined)?.cached_tokens,
  } : null;
  return {
    model: (json.model as string) ?? null,
    stop_reason: (choice?.finish_reason as string) ?? null,
    content,
    usage,
  };
}

function buildRecord(
  sid: string,
  protocol: Protocol,
  requestBody: unknown,
  status: number,
  reassembled: ReassembledResponse,
  receivedAt: number,
  completedAt: number,
  req: http.IncomingMessage | null,
  ttftMs: number | null
): ProxyRecord {
  const model = (requestBody && typeof requestBody === 'object' && 'model' in requestBody)
    ? String((requestBody as { model: unknown }).model) : null;
  // opencode sends x-session-id (ses_xxx); child sessions (subagents) carry a
  // different value plus x-parent-session-id pointing at the main session —
  // the opencode-emitter uses both for routing.
  const xSessionId = req ? (req.headers['x-session-id'] as string | undefined) ?? null : null;
  const userAgent = req ? (req.headers['user-agent'] as string | undefined) ?? null : null;
  const parentSessionId = req ? (req.headers['x-parent-session-id'] as string | undefined) ?? null : null;
  return {
    sid,
    protocol,
    receivedAt,
    completedAt,
    latencyMs: completedAt - receivedAt,
    ttftMs,
    request: {
      path: req?.url ?? '',
      model,
      body: requestBody ?? {},
    },
    response: {
      status,
      model: reassembled.model,
      stop_reason: reassembled.stop_reason,
      content: reassembled.content,
      usage: reassembled.usage,
    },
    xSessionId,
    parentSessionId,
    userAgent,
  };
}

const server = http.createServer(async (req, res) => {
  const url = req.url ?? '/';
  if (req.method === 'GET' && (url === '/healthz' || url === '/')) {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, port: server.address() }));
    return;
  }
  if (req.method === 'POST' && (url.includes('/v1/') || url.includes('/chat/completions') || url.includes('/messages'))) {
    try {
      const body = await readBody(req);
      await handleProxy(req, res, body);
    } catch (err) {
      if (!res.headersSent) {
        res.writeHead(500, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ error: 'proxy error', detail: String(err) }));
      } else {
        // headers already sent (streaming started) — can't send 500, just end
        res.end();
      }
    }
    return;
  }
  res.writeHead(404, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ error: 'not found' }));
});

server.listen(PORT, '127.0.0.1', () => {
  const addr = server.address();
  const port = typeof addr === 'object' && addr ? addr.port : PORT;
  // Create the capture file eagerly: cpx prints its path (and a `tail -f`
  // hint) at startup, and the first wire record may only arrive minutes
  // later — before that the path would not exist, breaking tail -f and
  // confusing manual imports. The emitter's append keeps writing to it.
  if (SESSION_ID) {
    try { fs.closeSync(fs.openSync(sessionFilePath(SESSION_ID), 'a')); } catch { /* best-effort */ }
  }
  console.log(`[cannbot-proxy] listening on http://127.0.0.1:${port}`);
  if (process.send) process.send({ type: 'listening', port });
});
