// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Bug #7 复现 IT：dedup 键 `${turnIndex}:${role}` 是拼接数组
// [main..., subagent...] 的全局下标。resume 后 main 文件增长，所有
// subagent turn 的下标整体平移 —— 旧 DB 行的键对不上新算出的键：
// - 平移后落在未占用下标的 subagent turn 被当作新 turn 重复插入；
// - 平移后撞上旧 subagent 行键的新 main turn 被误判为"已存在"而丢弃。
// 数据流：jsonl（main + subagents/）→ 两次 import-file → Turn 行数与内容。
import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prisma } from './setup';
import { POST } from '@/app/api/ingest/import-file/route';
import { POST as refreshSession } from '@/app/api/ingest/refresh-session/route';

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'dedup-drift-'));
const createdSids: string[] = [];

function setupFiles(sid: string): { mainFile: string } {
  const mainFile = path.join(tmpDir, `${sid}.jsonl`);
  const subDir = path.join(tmpDir, sid, 'subagents');
  fs.writeFileSync(mainFile, [
    userLine('第一问', '2026-08-18T10:00:00.000Z'),
    assistantLine('msg-a1', '答1', '2026-08-18T10:00:05.000Z'),
    userLine('第二问', '2026-08-18T10:01:00.000Z'),
    assistantLine('msg-a2', '答2', '2026-08-18T10:01:05.000Z'),
  ].join('\n') + '\n');
  fs.mkdirSync(subDir, { recursive: true });
  fs.writeFileSync(path.join(subDir, 'sub-drift-a.jsonl'), [
    userLine('子任务', '2026-08-18T10:00:30.000Z'),
    assistantLine('msg-a3', '子答', '2026-08-18T10:00:35.000Z'),
  ].join('\n') + '\n');
  return { mainFile };
}

function growMain(mainFile: string): void {
  fs.appendFileSync(mainFile, userLine('第三问', '2026-08-18T10:02:00.000Z') + '\n');
}

function userLine(text: string, ts: string): string {
  return JSON.stringify({ type: 'user', message: { role: 'user', content: text }, timestamp: ts });
}

function assistantLine(id: string, text: string, ts: string): string {
  return JSON.stringify({
    type: 'assistant',
    message: { role: 'assistant', id, content: [{ type: 'text', text }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } },
    timestamp: ts,
  });
}

function req(sid: string, mainFile: string): NextRequest {
  return new NextRequest('http://localhost/api/ingest/import-file', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ source: 'claude-jsonl', sessionId: sid, filePath: mainFile }),
  });
}

function refreshReq(sid: string): NextRequest {
  return new NextRequest('http://localhost/api/ingest/refresh-session', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ taskId: sid }),
  });
}

afterEach(async () => {
  for (const sid of createdSids.splice(0)) {
    const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true } });
    if (row) await prisma.session.delete({ where: { id: row.id } }).catch(() => {});
  }
});

async function assertNoDrift(sid: string): Promise<void> {
  const sessionRow = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true } });
  expect(sessionRow).not.toBeNull();
  createdSids.push(sid);
  const turns = await prisma.turn.findMany({
    where: { sessionId: sessionRow!.id },
    select: { role: true, content: true, subagentSessionId: true },
  });
  expect(turns.length).toBe(7);
  expect(turns.filter(t => t.subagentSessionId === 'sub-drift-a').length).toBe(2);
  expect(turns.some(t => t.role === 'user' && t.content === '第三问')).toBe(true);
}

describe('merge dedup 键漂移（Bug #7）', () => {
  it('重导路径（import-file merge）：subagent turn 不重复、新增 main turn 不丢失', async () => {
    const sid = 'dedup-drift-it-1';
    const { mainFile } = setupFiles(sid);

    const res1 = await POST(req(sid, mainFile));
    expect(res1.status).toBe(200);
    const sessionRow = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true } });
    expect(sessionRow).not.toBeNull();
    expect(await prisma.turn.count({ where: { sessionId: sessionRow!.id } })).toBe(6);

    growMain(mainFile);

    const res2 = await POST(req(sid, mainFile));
    expect(res2.status).toBe(200);

    await assertNoDrift(sid);
  });

  it('增量刷新路径（refresh-session / 详情页 auto-refresh）：同样不漂移', async () => {
    const sid = 'dedup-drift-it-2';
    const { mainFile } = setupFiles(sid);

    const res1 = await POST(req(sid, mainFile));
    expect(res1.status).toBe(200);

    growMain(mainFile);

    const res2 = await refreshSession(refreshReq(sid));
    expect(res2.status).toBe(200);

    await assertNoDrift(sid);
  });
});
