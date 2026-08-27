// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Bug 复现 IT：老代码把 opencode 捕获导入成 framework=claude-code /
// version=opencode-proxy（列表错显 Claude 徽标）。新导入路径已按 source
// 标记矫正（frameworkForCapture），但 merge/refresh 路径的 updateData
// 从不回填 framework/version → 历史脏数据永远无法自愈。
// 数据流：jsonl（source:opencode-proxy）→ 首次导入（模拟旧脏数据）→
// 重导入（merge 路径）→ framework/version 被矫正。
import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prisma } from './setup';
import { POST } from '@/app/api/ingest/import-file/route';

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'fw-repair-'));
const createdSids: string[] = [];

function opencodeCapture(sid: string): string {
  const file = path.join(tmpDir, `${sid}.jsonl`);
  fs.writeFileSync(file, [
    JSON.stringify({
      type: 'user', source: 'opencode-proxy',
      message: { role: 'user', content: '你是 verifier 角色' },
      timestamp: '2026-08-18T10:00:00.000Z',
    }),
    JSON.stringify({
      type: 'assistant', source: 'opencode-proxy',
      message: { role: 'assistant', id: 'msg-1', content: [{ type: 'text', text: 'PASS' }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } },
      timestamp: '2026-08-18T10:00:05.000Z',
      system: 'You are opencode', tools: [{ name: 'bash', description: '' }],
    }),
  ].join('\n') + '\n');
  return file;
}

function req(sid: string, file: string): NextRequest {
  return new NextRequest('http://localhost/api/ingest/import-file', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ source: 'claude-jsonl', sessionId: sid, filePath: file }),
  });
}

afterEach(async () => {
  for (const sid of createdSids.splice(0)) {
    const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true } });
    if (row) await prisma.session.delete({ where: { id: row.id } }).catch(() => {});
  }
});

describe('opencode-proxy framework 矫正', () => {
  it('新导入即正确归属 opencode', async () => {
    const sid = 'fw-repair-new';
    const file = opencodeCapture(sid);
    await POST(req(sid, file));
    const s = await prisma.session.findFirst({ where: { taskId: sid } });
    expect(s?.framework).toBe('opencode');
    expect(s?.version).toBe('opencode-proxy');
    createdSids.push(sid);
  });

  it('merge 路径回填 framework/version（历史脏数据自愈）', async () => {
    const sid = 'fw-repair-heal';
    const file = opencodeCapture(sid);
    await POST(req(sid, file));
    createdSids.push(sid);

    // 模拟旧代码导入出的脏数据：framework=claude-code
    await prisma.session.updateMany({
      where: { taskId: sid },
      data: { framework: 'claude-code' },
    });

    // 重导入 → merge 路径（reason: already-exists）→ 矫正归属
    const res = await POST(req(sid, file));
    const body = await res.json();
    expect(body.reason).toBe('already-exists');
    const s = await prisma.session.findFirst({ where: { taskId: sid } });
    expect(s?.framework).toBe('opencode');
    expect(s?.version).toBe('opencode-proxy');
  });

  it('非 proxy 的 claude-jsonl merge 不改 framework', async () => {
    const sid = 'fw-repair-native';
    const file = path.join(tmpDir, `${sid}.jsonl`);
    fs.writeFileSync(file, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '原生会话' }, timestamp: '2026-08-18T10:00:00.000Z' }),
      JSON.stringify({
        type: 'assistant',
        message: { role: 'assistant', id: 'msg-n', content: [{ type: 'text', text: '答' }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } },
        timestamp: '2026-08-18T10:00:05.000Z',
      }),
    ].join('\n') + '\n');
    await POST(req(sid, file));
    createdSids.push(sid);
    await POST(req(sid, file));
    const s = await prisma.session.findFirst({ where: { taskId: sid } });
    expect(s?.framework).toBe('claude-code');
  });
});
