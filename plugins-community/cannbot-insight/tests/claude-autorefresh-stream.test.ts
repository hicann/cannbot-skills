// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { importSession } from '../src/lib/ingest/data-service.ts';
import type { AutoRefreshProbe } from '../src/lib/ingest/auto-refresh-probe.ts';
import { PrismaClient } from '@prisma/client';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';
import { GET } from '../src/app/api/observe/auto-refresh-stream/route.ts';

const FIXTURE_DIR = path.resolve(__dirname, 'data/e2e');
const BASE_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-base.jsonl');
const IDLE_ASSISTANT_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-idle-assistant.jsonl');

const CLAUDE_SESSION_ID = 'claude-autorefresh-stream';
const prisma = new PrismaClient();

function tmpCopy(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-sse-'));
  const dest = path.join(dir, 'claude-autorefresh-stream-live.jsonl');
  fs.copyFileSync(BASE_FIXTURE, dest);
  return dest;
}

function appendFile(dest: string, src: string): void {
  fs.appendFileSync(dest, fs.readFileSync(src, 'utf-8'));
}

function bumpMtime(filePath: string, secondsAhead: number): void {
  const future = new Date(Date.now() + secondsAhead * 1000);
  fs.utimesSync(filePath, future, future);
}

// Parse an SSE chunk into { data?: string, comment?: string } events.
function parseSseChunk(chunk: string): Array<{ data?: string; comment?: string }> {
  const events: Array<{ data?: string; comment?: string }> = [];
  for (const raw of chunk.split('\n\n')) {
    if (!raw.trim()) continue;
    const event: { data?: string; comment?: string } = {};
    for (const line of raw.split('\n')) {
      if (line.startsWith(': ')) event.comment = line.slice(2);
      else if (line.startsWith('data: ')) event.data = line.slice(6);
    }
    events.push(event);
  }
  return events;
}

// Read the SSE stream, returning the next `data:` payload (skipping comments).
async function nextDataPayload(reader: ReadableStreamDefaultReader<Uint8Array>, timeoutMs: number): Promise<Partial<AutoRefreshProbe> & { event?: string; maxTimeUpdated?: number; settled?: boolean }> {
  const decoder = new TextDecoder();
  const deadline = Date.now() + timeoutMs;
  let buffer = '';
  while (Date.now() < deadline) {
    const { done, value } = await reader.read();
    if (done) throw new Error('stream closed');
    buffer += decoder.decode(value, { stream: true });
    const events = parseSseChunk(buffer);
    const dataEvent = events.find(e => e.data);
    if (dataEvent) return JSON.parse(dataEvent.data!);
    buffer = '';
  }
  throw new Error(`timed out waiting for SSE data payload (${timeoutMs}ms)`);
}

describe('E2E: claude-code SSE file-watch auto-refresh', () => {
  let liveFile = '';
  const abort = new AbortController();

  beforeAll(async () => {
    await prisma.$connect();
    await prisma.session.deleteMany({ where: { taskId: CLAUDE_SESSION_ID, framework: 'claude-code' } });
    liveFile = tmpCopy();
    await importSession(liveFile, CLAUDE_SESSION_ID, prisma, liveFile, 'claude-jsonl');
  });

  afterAll(async () => {
    abort.abort();
    try {
      await prisma.session.deleteMany({ where: { taskId: CLAUDE_SESSION_ID, framework: 'claude-code' } });
    } catch { /* ignore */ }
    await prisma.$disconnect();
    try {
      if (liveFile && fs.existsSync(liveFile)) fs.rmSync(path.dirname(liveFile), { recursive: true, force: true });
    } catch { /* ignore */ }
  });

  it('emits an initial probe on connect (catch-up) reporting settled=true', async () => {
    const request = new Request(`http://localhost/api/observe/auto-refresh-stream?taskId=${CLAUDE_SESSION_ID}`, {
      signal: abort.signal,
    });
    const response = await GET(request);
    expect(response.status).toBe(200);
    expect(response.headers.get('content-type')).toBe('text/event-stream');

    const reader = (response.body as ReadableStream<Uint8Array>).getReader();
    const probe = await nextDataPayload(reader, 3000);
    // Base fixture ends in a `result` line -> settled.
    expect(probe.settled).toBe(true);
    reader.releaseLock();
  }, 10000);

  it('pushes an updated probe when fs.watch detects a file change', async () => {
    const request = new Request(`http://localhost/api/observe/auto-refresh-stream?taskId=${CLAUDE_SESSION_ID}`, {
      signal: abort.signal,
    });
    const response = await GET(request);
    const reader = (response.body as ReadableStream<Uint8Array>).getReader();

    // consume initial probe
    const initial = await nextDataPayload(reader, 3000);
    const baselineMtime = initial.maxTimeUpdated;

    // mutate the source file — append a finalized assistant turn (settled)
    appendFile(liveFile, IDLE_ASSISTANT_FIXTURE);
    bumpMtime(liveFile, 60);

    // fs.watch + 400ms debounce -> server re-probes and pushes
    const updated = await nextDataPayload(reader, 5000);
    expect(updated.settled).toBe(true);
    expect(updated.maxTimeUpdated).not.toBe(baselineMtime);

    reader.releaseLock();
  }, 15000);
});
