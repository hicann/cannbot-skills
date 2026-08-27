// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Bug #5/#6 复现 IT：importSession 的 merge 路径只 merge turns/toolCalls/
// skillEvents，不重建 Execution/InteractionBridge —— resume 后新增的 subagent
// 有 Turn 无 Execution，dispatch 在执行树上断头。修复后 merge 路径与
// deltaRefresh 一样 deleteMany + 全量重建。
// 数据流：jsonl（main + subagents/）→ 两次 import-file → Execution/Bridge 行。
import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { prisma } from './setup';
import { POST } from '@/app/api/ingest/import-file/route';

const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'exec-rebuild-'));
const sid = 'exec-rebuild-it-1';
const mainFile = path.join(tmpDir, `${sid}.jsonl`);
const subDir = path.join(tmpDir, sid, 'subagents');

function userLine(text: string, ts: string): string {
  return JSON.stringify({ type: 'user', message: { role: 'user', content: text }, timestamp: ts });
}

function dispatchLine(id: string, text: string, subSid: string, ts: string): string {
  return JSON.stringify({
    type: 'assistant',
    message: {
      role: 'assistant', id,
      content: [
        { type: 'text', text },
        { type: 'tool_use', id: `tu-${subSid}`, name: 'task', input: { subagent_session_id: subSid, description: `子任务 ${subSid}`, prompt: `去做 ${subSid}` } },
      ],
      model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 },
    },
    timestamp: ts,
  });
}

function toolResultLine(toolUseId: string, ts: string): string {
  return JSON.stringify({
    type: 'user',
    message: { role: 'user', content: [{ type: 'tool_result', tool_use_id: toolUseId, content: 'done' }] },
    timestamp: ts,
  });
}

function subFile(subSid: string, ts: string): void {
  fs.writeFileSync(path.join(subDir, `${subSid}.jsonl`), [
    userLine(`去做 ${subSid}`, ts),
    JSON.stringify({
      type: 'assistant',
      message: { role: 'assistant', id: `msg-${subSid}`, content: [{ type: 'text', text: `子答 ${subSid}` }], model: 'glm-5.2', usage: { input_tokens: 10, output_tokens: 5 } },
      timestamp: ts.replace('T10:', 'T10:').slice(0, -1) + '5Z',
    }),
  ].join('\n') + '\n');
}

function req(): NextRequest {
  return new NextRequest('http://localhost/api/ingest/import-file', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ source: 'claude-jsonl', sessionId: sid, filePath: mainFile }),
  });
}

afterEach(async () => {
  const row = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true } });
  if (row) await prisma.session.delete({ where: { id: row.id } }).catch(() => {});
});

describe('merge 路径重建 execution/bridge（Bug #5/#6）', () => {
  it('resume 新增 subagent 重导后：Execution 与 Bridge 同步补齐，不重复', async () => {
    fs.mkdirSync(subDir, { recursive: true });

    // v1：一个 dispatch → sub-a
    fs.writeFileSync(mainFile, [
      userLine('查一下版本', '2026-08-18T10:00:00Z'),
      dispatchLine('m1', '派子代理', 'sub-a', '2026-08-18T10:00:05Z'),
      toolResultLine('tu-sub-a', '2026-08-18T10:00:40Z'),
    ].join('\n') + '\n');
    subFile('sub-a', '2026-08-18T10:00:10Z');

    const res1 = await POST(req());
    expect(res1.status).toBe(200);
    const sessionRow = await prisma.session.findFirst({ where: { taskId: sid }, select: { id: true, rootExecutionId: true } });
    expect(sessionRow).not.toBeNull();
    const sessionId = sessionRow!.id;
    expect(await prisma.execution.count({ where: { sessionId, isSubagent: true } })).toBe(1);
    expect(await prisma.interactionBridge.count({ where: { sessionId } })).toBe(1);

    // v2（resume）：新增 dispatch → sub-b
    fs.appendFileSync(mainFile, [
      userLine('再查一个', '2026-08-18T10:02:00Z'),
      dispatchLine('m4', '再派一个', 'sub-b', '2026-08-18T10:02:05Z'),
      toolResultLine('tu-sub-b', '2026-08-18T10:02:40Z'),
    ].join('\n') + '\n');
    subFile('sub-b', '2026-08-18T10:02:10Z');

    const res2 = await POST(req());
    expect(res2.status).toBe(200);

    // sub-b turn 已入库
    expect(await prisma.turn.count({ where: { sessionId, subagentSessionId: 'sub-b' } })).toBe(2);
    // Execution 补齐：root + sub-a + sub-b，且不重复
    const subExecs = await prisma.execution.findMany({ where: { sessionId, isSubagent: true }, select: { agentSessionId: true } });
    expect(subExecs.map(e => e.agentSessionId).sort()).toEqual(['sub-a', 'sub-b']);
    expect(await prisma.execution.count({ where: { sessionId } })).toBe(3);
    // Bridge 补齐：两条 dispatch 各自挂到对应 subagent
    const bridges = await prisma.interactionBridge.findMany({ where: { sessionId }, select: { subagentSessionId: true, status: true } });
    expect(bridges.map(b => b.subagentSessionId).sort()).toEqual(['sub-a', 'sub-b']);
    expect(bridges.every(b => b.status === 'completed')).toBe(true);
    // session.rootExecutionId 保持指向 root
    const updated = await prisma.session.findUnique({ where: { id: sessionId }, select: { rootExecutionId: true } });
    const rootExec = await prisma.execution.findFirst({ where: { sessionId, isSubagent: false }, select: { id: true } });
    expect(updated?.rootExecutionId).toBe(rootExec?.id);
  });
});
