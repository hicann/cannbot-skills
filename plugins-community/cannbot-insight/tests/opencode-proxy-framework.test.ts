// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// opencode proxy 捕获的 agent 归属 IT：framework 必须是 opencode（捕获文件
// 是 claude 格式，但会话属于 opencode —— 由行内 source:'opencode-proxy' 标记
// 判定），claude 捕获保持 claude-code，重复导入不产生重复行。
import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prisma } from './setup';
import { POST } from '@/app/api/ingest/import-file/route';

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'oc-proxy-fw-'));

function captureFile(name: string, marker: string): string {
  const file = path.join(tmpDir, name);
  fs.writeFileSync(file, [
    JSON.stringify({ type: 'user', message: { role: 'user', content: '第一问' }, timestamp: '2026-08-18T10:00:00.000Z', source: marker }),
    JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: `msg_${name}`, content: [{ type: 'text', text: '答' }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } }, timestamp: '2026-08-18T10:00:05.000Z', duration_ms: 500, source: marker }),
  ].join('\n') + '\n');
  return file;
}

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

describe('proxy 捕获的 framework 归属', () => {
  it("opencode-proxy 捕获 → framework='opencode'，version 带 marker", async () => {
    const file = captureFile('oc-proxy.jsonl', 'opencode-proxy');
    const sid = 'oc-proxy-fw-it-1';
    const res = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: file }));
    expect(res.status).toBe(200);
    const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true, framework: true, version: true } });
    expect(row).not.toBeNull();
    createdSessionIds.push(row!.id);
    expect(row!.framework).toBe('opencode');
    expect(row!.version).toBe('opencode-proxy');
  });

  it("claude-proxy 捕获 → framework 保持 'claude-code'", async () => {
    const file = captureFile('cc-proxy.jsonl', 'claude-proxy');
    const sid = 'cc-proxy-fw-it-1';
    const res = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: file }));
    expect(res.status).toBe(200);
    const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true, framework: true, version: true } });
    expect(row).not.toBeNull();
    createdSessionIds.push(row!.id);
    expect(row!.framework).toBe('claude-code');
    expect(row!.version).toBe('claude-proxy');
  });


  it('行内带 agent 版本号时 version = "<版本>-<marker>"（版本与来源都保留）', async () => {
    const file = path.join(tmpDir, 'oc-proxy-ver.jsonl');
    fs.writeFileSync(file, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '第一问' }, timestamp: '2026-08-18T10:00:00.000Z', source: 'opencode-proxy' }),
      JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'msg_ver', content: [{ type: 'text', text: '答' }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } }, timestamp: '2026-08-18T10:00:05.000Z', version: '1.17.9', source: 'opencode-proxy' }),
    ].join('\n') + '\n');
    const sid = 'oc-proxy-fw-it-3';
    const res = await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: file }));
    expect(res.status).toBe(200);
    const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true, framework: true, version: true } });
    expect(row).not.toBeNull();
    createdSessionIds.push(row!.id);
    expect(row!.framework).toBe('opencode');
    expect(row!.version).toBe('1.17.9-opencode-proxy');
  });

  it('重复导入同一 opencode 捕获不产生重复 session 行', async () => {
    const file = captureFile('oc-proxy-2.jsonl', 'opencode-proxy');
    const sid = 'oc-proxy-fw-it-2';
    await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: file }));
    await POST(req({ source: 'claude-jsonl', sessionId: sid, filePath: file }));
    const rows = await prisma.session.findMany({ where: { taskId: sid }, select: { id: true, framework: true } });
    expect(rows.length).toBe(1);
    expect(rows[0].framework).toBe('opencode');
    createdSessionIds.push(rows[0].id);
  });
});
