// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { importSession, deltaRefreshSession } from '../src/lib/ingest/data-service.ts';
import { probeAutoRefresh } from '../src/lib/ingest/auto-refresh-probe.ts';
import { PrismaClient } from '@prisma/client';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

const FIXTURE_DIR = path.resolve(__dirname, 'data/e2e');
const BASE_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-base.jsonl');
const APPEND_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-append-turn.jsonl');
const STREAMING_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-streaming.jsonl');
const IDLE_ASSISTANT_FIXTURE = path.join(FIXTURE_DIR, 'claude-autorefresh-idle-assistant.jsonl');

const CLAUDE_SESSION_ID = 'claude-autorefresh-base';
const prisma = new PrismaClient();

function tmpCopy(src: string): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-autorefresh-'));
  return path.join(dir, path.basename(src).replace('.jsonl', '-live.jsonl'));
}

function appendFile(dest: string, src: string): void {
  fs.appendFileSync(dest, fs.readFileSync(src, 'utf-8'));
}

function appendRaw(dest: string, raw: string): void {
  fs.appendFileSync(dest, raw);
}

// Bump mtime deterministically so timeChanged is always true regardless of FS resolution.
function bumpMtime(filePath: string, secondsAhead: number): void {
  const future = new Date(Date.now() + secondsAhead * 1000);
  fs.utimesSync(filePath, future, future);
}

async function getSessionRecord(taskId: string) {
  const s = await prisma.session.findFirst({
    where: { taskId, framework: 'claude-code' },
    select: { id: true, taskId: true, sourcePath: true, framework: true },
  });
  if (!s) throw new Error(`session not found: ${taskId}`);
  return s;
}

describe('E2E: claude-code auto-refresh (probe + delta refresh)', () => {
  let liveFile = '';
  let imported = false;
  let baselineMtime = 0;
  let initialTurnCount = 0;

  beforeAll(async () => {
    await prisma.$connect();
    // clean any leftover from a previous run
    await prisma.session.deleteMany({ where: { taskId: CLAUDE_SESSION_ID, framework: 'claude-code' } });

    liveFile = tmpCopy(BASE_FIXTURE);
    fs.copyFileSync(BASE_FIXTURE, liveFile);
    const res = await importSession(liveFile, CLAUDE_SESSION_ID, prisma, liveFile, 'claude-jsonl');
    imported = res.imported;

    const s = await getSessionRecord(CLAUDE_SESSION_ID);
    initialTurnCount = await prisma.turn.count({ where: { sessionId: s.id } });
    const probe0 = await probeAutoRefresh(s, prisma);
    baselineMtime = probe0.maxTimeUpdated;
  });

  afterAll(async () => {
    try {
      await prisma.session.deleteMany({ where: { taskId: CLAUDE_SESSION_ID, framework: 'claude-code' } });
    } catch { /* ignore */ }
    await prisma.$disconnect();
    try {
      if (liveFile && fs.existsSync(liveFile)) fs.rmSync(path.dirname(liveFile), { recursive: true, force: true });
    } catch { /* ignore */ }
  });

  describe('settled session (ends with result line)', () => {
    it('session imported successfully with claude-code framework', () => {
      expect(imported).toBe(true);
    });

    it('probe reports settled=true for a session ending in a result line', async () => {
      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const probe = await probeAutoRefresh(s, prisma);
      expect(probe.streaming).toBe(false);
      expect(probe.pendingInput).toBe(false);
      expect(probe.settled).toBe(true);
      expect(probe.maxTimeUpdated).toBeGreaterThan(0);
    });
  });

  describe('append a complete new turn -> probe detects change -> delta refresh', () => {
    beforeAll(async () => {
      appendFile(liveFile, APPEND_FIXTURE);
      bumpMtime(liveFile, 60);
    });

    it('probe keeps settled=true and reports a changed maxTimeUpdated', async () => {
      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const probe = await probeAutoRefresh(s, prisma);
      expect(probe.settled).toBe(true);
      expect(probe.streaming).toBe(false);
      expect(probe.maxTimeUpdated).not.toBe(baselineMtime);
    });

    it('deltaRefreshSession adds the new turn', async () => {
      const result = await deltaRefreshSession(CLAUDE_SESSION_ID, prisma);
      expect(result.addedTurns).toBeGreaterThan(0);

      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const afterCount = await prisma.turn.count({ where: { sessionId: s.id } });
      expect(afterCount).toBeGreaterThan(initialTurnCount);
      expect(afterCount - initialTurnCount).toBe(result.addedTurns);
    });

    it('deltaRefreshSession is idempotent on an unchanged file', async () => {
      const result = await deltaRefreshSession(CLAUDE_SESSION_ID, prisma);
      expect(result.addedTurns).toBe(0);
    });
  });

  describe('idle assistant (finalized response with stop_reason) -> settled', () => {
    // This is the core live-session scenario: the user's claude-code finished an
    // assistant response and sits idle. The transcript ends in an `assistant` line
    // carrying stop_reason="end_turn" (no per-turn `result` line exists — that
    // only appears at session end). The probe must report settled=true so the
    // 5s poll triggers a delta refresh.
    beforeAll(async () => {
      appendFile(liveFile, IDLE_ASSISTANT_FIXTURE);
      bumpMtime(liveFile, 90);
    });

    it('probe reports streaming=false / settled=true (regression: was stuck streaming)', async () => {
      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const probe = await probeAutoRefresh(s, prisma);
      expect(probe.streaming).toBe(false);
      expect(probe.pendingInput).toBe(false);
      expect(probe.settled).toBe(true);
    });

    it('deltaRefreshSession ingests the finalized assistant turn', async () => {
      const beforeCount = await prisma.turn.count({
        where: { sessionId: (await getSessionRecord(CLAUDE_SESSION_ID)).id },
      });
      const result = await deltaRefreshSession(CLAUDE_SESSION_ID, prisma);
      expect(result.addedTurns).toBeGreaterThan(0);
      const afterCount = await prisma.turn.count({
        where: { sessionId: (await getSessionRecord(CLAUDE_SESSION_ID)).id },
      });
      expect(afterCount).toBeGreaterThan(beforeCount);
    });
  });

  describe('streaming detection (assistant line without stop_reason, response in progress)', () => {
    beforeAll(async () => {
      appendFile(liveFile, STREAMING_FIXTURE);
      bumpMtime(liveFile, 120);
    });

    it('probe reports streaming=true / settled=false', async () => {
      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const probe = await probeAutoRefresh(s, prisma);
      expect(probe.streaming).toBe(true);
      expect(probe.settled).toBe(false);
    });
  });

  describe('pendingInput detection (user text line waiting for assistant)', () => {
    beforeAll(async () => {
      const userLine = JSON.stringify({
        type: 'user',
        message: { role: 'user', content: 'One more question?' },
        timestamp: '2026-07-16T10:20:00.000Z',
        uuid: 'u-pending',
        parentUuid: 'r3',
        isSidechain: false,
        promptId: 'p-pending',
      }) + '\n';
      appendRaw(liveFile, userLine);
      bumpMtime(liveFile, 180);
    });

    it('probe reports pendingInput=true / settled=false', async () => {
      const s = await getSessionRecord(CLAUDE_SESSION_ID);
      const probe = await probeAutoRefresh(s, prisma);
      expect(probe.pendingInput).toBe(true);
      expect(probe.settled).toBe(false);
    });
  });
});

describe('E2E: proxy 捕获的 auto-refresh probe（分支归属 + stopReason 字段）', () => {
  const proxyDir = fs.mkdtempSync(path.join(os.tmpdir(), 'proxy-probe-'));
  const proxyFile = path.join(proxyDir, 'cpx-proxy-probe-it.jsonl');

  // proxy 捕获是扩展 claude 格式：assistant 行的完结标记在顶层 stopReason
  const userLine = JSON.stringify({ type: 'user', message: { role: 'user', content: '问' }, timestamp: '2026-08-18T10:00:00.000Z' });
  const assistantDone = JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'm1', content: [{ type: 'text', text: '答' }] }, timestamp: '2026-08-18T10:00:05.000Z', stopReason: 'end_turn' });
  const assistantOpen = JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'm2', content: [{ type: 'text', text: '…' }] }, timestamp: '2026-08-18T10:00:06.000Z' });

  function writeCapture(lines: string[]): void {
    fs.writeFileSync(proxyFile, lines.join('\n') + '\n');
    const future = new Date(Date.now() + 1000);
    fs.utimesSync(proxyFile, future, future);
  }
  const ses = (framework: string, version: string | null) => ({
    id: 'probe-only', taskId: 'proxy-probe-it', sourcePath: proxyFile, framework, version,
  });

  afterAll(() => {
    try { fs.rmSync(proxyDir, { recursive: true, force: true }); } catch { /* ignore */ }
  });

  it('opencode-proxy 捕获走 claude 探测：不再全零 NO_CHANGE（回归：曾按 sqlite 打开抛错退化为 NO_CHANGE）', async () => {
    writeCapture([userLine, assistantDone]);
    const p = await probeAutoRefresh(ses('opencode', '1.17.9-opencode-proxy'), prisma);
    expect(p.maxTimeUpdated).toBeGreaterThan(0);
    expect(p.streaming).toBe(false);
    expect(p.settled).toBe(true);
  });

  it('claude-proxy 捕获：顶层 stopReason 识别为已完结（回归：曾永远 streaming=true 不触发自动刷新）', async () => {
    writeCapture([userLine, assistantDone]);
    const p = await probeAutoRefresh(ses('claude-code', '2.1.234.467-claude-proxy'), prisma);
    expect(p.maxTimeUpdated).toBeGreaterThan(0);
    expect(p.streaming).toBe(false);
    expect(p.settled).toBe(true);
  });

  it('进行中的 proxy 响应（无 stopReason）仍判 streaming', async () => {
    writeCapture([userLine, assistantOpen]);
    const p = await probeAutoRefresh(ses('opencode', '1.17.9-opencode-proxy'), prisma);
    expect(p.streaming).toBe(true);
    expect(p.settled).toBe(false);
  });

  it('原生 opencode（version 无 -proxy）仍走 opencode-db 探测：jsonl 按 sqlite 打开退化为 NO_CHANGE', async () => {
    writeCapture([userLine, assistantDone]);
    const p = await probeAutoRefresh(ses('opencode', null), prisma);
    expect(p.maxTimeUpdated).toBe(0);
    expect(p.settled).toBe(false);
  });
});
