// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Regression: the UI labeled EVERY imported:false as "Already exists". A
// missing capture file also yields imported:false — the API must distinguish
// (404 file-not-found / reason:no-interactions / reason:already-exists) so
// the message reflects the real cause.

import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import path from 'node:path';
import { prisma } from './setup';
import { POST } from '@/app/api/ingest/import-file/route';

const createdSessionIds: string[] = [];

function req(body: unknown): NextRequest {
  return new NextRequest('http://localhost/api/ingest/import-file', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body),
  });
}

afterEach(async () => {
  for (const id of createdSessionIds.splice(0)) {
    try { await prisma.session.delete({ where: { id } }); } catch { /* ignore */ }
  }
});

describe('POST /api/ingest/import-file — failure reasons', () => {
  it('missing file → 404 with reason file-not-found (NOT "already exists")', async () => {
    const res = await POST(req({
      source: 'claude-jsonl',
      sessionId: 'no-such-file-sid',
      filePath: '/root/.cannbot-insight/proxy/definitely-missing-capture.jsonl',
    }));
    expect(res.status).toBe(404);
    const data = await res.json();
    expect(data.reason).toBe('file-not-found');
    expect(data.error).toContain('File not found');
  });

  it('existing file that parses to 0 interactions → reason no-interactions', async () => {
    const emptyFile = path.join(__dirname, 'data/claude-sessions/empty-session.jsonl');
    fs.writeFileSync(emptyFile, '');
    try {
      const res = await POST(req({ source: 'claude-jsonl', sessionId: 'empty-session-it', filePath: emptyFile }));
      expect(res.status).toBe(200);
      const data = await res.json();
      expect(data.imported).toBe(false);
      expect(data.reason).toBe('no-interactions');
    } finally {
      fs.unlinkSync(emptyFile);
    }
  });

  it('re-import of an existing session → reason already-exists', async () => {
    const FIXTURE = path.join(__dirname, 'data/claude-sessions/system-reminder-strip.jsonl');
    const sid = 'import-reason-it';
    const first = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: FIXTURE }));
    const d1 = await first.json();
    if (d1.imported) {
      const s = await prisma.session.findFirst({ where: { taskId: sid } });
      if (s) createdSessionIds.push(s.id);
    }
    const second = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: FIXTURE }));
    expect(second.status).toBe(200);
    const d2 = await second.json();
    expect(d2.imported).toBe(false);
    expect(d2.reason).toBe('already-exists');
    expect(d2.query).toBeTruthy();
  });
});

describe('POST /api/ingest/import-file — empty claude capture placeholder', () => {
  const tmpCapture = path.join(__dirname, 'data/claude-sessions/live-capture-it.jsonl');

  afterEach(async () => {
    try { fs.unlinkSync(tmpCapture); } catch { /* */ }
    const s = await prisma.session.findFirst({ where: { taskId: 'live-capture-it' } });
    if (s) { try { await prisma.session.delete({ where: { id: s.id } }); } catch { /* */ } }
  });

  it('empty capture file imports as a visible placeholder session', async () => {
    fs.writeFileSync(tmpCapture, '');
    const res = await POST(req({ source: 'claude-jsonl', sessionId: 'live-capture-it', filePath: tmpCapture }));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.imported).toBe(true);
    expect(data.reason).toBe('imported-empty');
    const s = await prisma.session.findFirst({ where: { taskId: 'live-capture-it' } });
    expect(s).not.toBeNull();
    expect(s!.framework).toBe('claude-code');
    expect(s!.label).toContain('捕获中');
    expect(s!.sourcePath).toBe(tmpCapture);
  });

  it('re-import after the file fills up MERGES turns into the same session (dynamic refresh flow)', async () => {
    fs.writeFileSync(tmpCapture, '');
    await POST(req({ source: 'claude-jsonl', sessionId: 'live-capture-it', filePath: tmpCapture }));
    const s1 = await prisma.session.findFirst({ where: { taskId: 'live-capture-it' } });
    expect(s1).not.toBeNull();
    expect(await prisma.turn.count({ where: { sessionId: s1!.id } })).toBe(0);

    // the proxy appends real content (copy the fixture conversation in)
    fs.appendFileSync(tmpCapture, fs.readFileSync(path.join(__dirname, 'data/claude-sessions/system-reminder-strip.jsonl')));
    const res2 = await POST(req({ source: 'claude-jsonl', sessionId: 'live-capture-it', filePath: tmpCapture }));
    expect(res2.status).toBe(200);
    const s2 = await prisma.session.findFirst({ where: { taskId: 'live-capture-it' } });
    expect(s2!.id).toBe(s1!.id); // same session, not a duplicate
    const turns = await prisma.turn.count({ where: { sessionId: s2!.id } });
    expect(turns).toBeGreaterThan(0);
    expect(s2!.query).toBeTruthy();
    // 占位 label 随内容填满被替换为真实首问，不再永久显示「（捕获中）」
    expect(s2!.label).not.toContain('捕获中');
  });
});

describe('GET /api/observe/session/wire-rounds — direct-from-file round view', () => {
  it('rebuilds each wire request from the capture file (ground truth for LLM Input comparison)', async () => {
    const FIXTURE = path.join(__dirname, 'data/claude-sessions/system-reminder-strip.jsonl');
    const sid = 'wire-rounds-it';
    const first = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: FIXTURE }));
    const d1 = await first.json();
    if (d1.imported) {
      const s = await prisma.session.findFirst({ where: { taskId: sid } });
      if (s) createdSessionIds.push(s.id);
    }
    const { GET } = await import('@/app/api/observe/session/wire-rounds/route');
    const res = await GET(new NextRequest(`http://localhost/api/observe/session/wire-rounds?taskId=${sid}`));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.totalRounds).toBe(2); // fixture: 2 assistant responses
    // round 1 request = the new messages before the first response (delta only)
    const r1 = data.rounds[0];
    expect(r1.requestMessages.length).toBe(1); // the first user line (reminder + prompt in ONE message)
    expect(r1.totalMessages).toBe(1);
    // round 2: only the NEW messages since round 1's response (delta, not accumulated)
    const r2 = data.rounds[1];
    expect(r2.requestMessages.length).toBe(1); // the mid-session reminder user line
    expect(r2.totalMessages).toBe(3); // user1 + assistant1 + user3 = accumulated total
    expect(r2.requestMessages[0].role).toBe('user');
    expect(r2.requestMessages[0].content.json).toContain('system-reminder');
  });

  it('404 when the session is unknown', async () => {
    const { GET } = await import('@/app/api/observe/session/wire-rounds/route');
    const res = await GET(new NextRequest('http://localhost/api/observe/session/wire-rounds?taskId=no-such-sid'));
    expect(res.status).toBe(404);
  });
});
