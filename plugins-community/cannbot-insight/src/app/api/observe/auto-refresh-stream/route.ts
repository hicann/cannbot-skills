// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest } from 'next/server';
import fs from 'node:fs';
import { prisma } from '@/lib/db';
import { probeAutoRefresh, resolveClaudeSessionFile, type ProbeSession } from '@/lib/ingest/auto-refresh-probe';

export const dynamic = 'force-dynamic';
export const maxDuration = 300;

// Real-time auto-refresh for claude-code sessions: watches the source JSONL file
// with fs.watch and pushes probe results to the client over SSE. The client gates
// on `settled && changed` before triggering a delta refresh — identical semantics
// to the opencode 5s poll, but event-driven (sub-second) instead of timer-driven.
export async function GET(request: NextRequest | Request) {
  const taskId = new URL(request.url).searchParams.get('taskId');
  if (!taskId) {
    return new Response('Missing taskId', { status: 400 });
  }

  const session = await prisma.session.findFirst({
    where: { taskId },
    select: { id: true, taskId: true, sourcePath: true, framework: true },
  });
  if (!session || session.framework !== 'claude-code' || !session.sourcePath) {
    return new Response('Not a claude-code session with a sourcePath', { status: 400 });
  }

  const filePath = resolveClaudeSessionFile(session.sourcePath, session.taskId);
  if (!filePath || !fs.existsSync(filePath) || !fs.statSync(filePath).isFile()) {
    return new Response('Source file not found', { status: 404 });
  }

  const probeSession: ProbeSession = {
    id: session.id,
    taskId: session.taskId,
    sourcePath: session.sourcePath,
    framework: session.framework,
  };

  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      const encoder = new TextEncoder();
      let closed = false;
      const send = (obj: unknown) => {
        if (closed) return;
        try {
          controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
        } catch {
          closed = true;
        }
      };
      const ping = () => {
        if (closed) return;
        try { controller.enqueue(encoder.encode(`: keep-alive\n\n`)); } catch { closed = true; }
      };

      // Initial probe on connect — also serves as catch-up on EventSource reconnect.
      probeAutoRefresh(probeSession, prisma).then(send).catch(() => {});

      // Primary mechanism: watch the file and re-probe on change (debounced).
      let debounceTimer: ReturnType<typeof setTimeout> | null = null;
      const DEBOUNCE_MS = 400;
      const onFileChange = () => {
        if (debounceTimer) clearTimeout(debounceTimer);
        debounceTimer = setTimeout(() => {
          debounceTimer = null;
          probeAutoRefresh(probeSession, prisma).then(send).catch(() => {});
        }, DEBOUNCE_MS);
      };

      const keepAlive = setInterval(ping, 15_000);

      let watcher: fs.FSWatcher | null = null;
      const cleanup = () => {
        if (debounceTimer) clearTimeout(debounceTimer);
        if (watcher) { try { watcher.close(); } catch { /* ignore */ } watcher = null; }
        clearInterval(keepAlive);
        closed = true;
        try { controller.close(); } catch { /* ignore */ }
      };
      try {
        watcher = fs.watch(filePath, { persistent: false }, onFileChange);
        watcher.on('error', cleanup);
      } catch {
        // fs.watch unavailable — the EventSource will auto-reconnect and the
        // initial probe on reconnect acts as a catch-up.
      }

      request.signal.addEventListener('abort', cleanup, { once: true });
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection': 'keep-alive',
      'Access-Control-Allow-Origin': '*',
    },
  });
}
