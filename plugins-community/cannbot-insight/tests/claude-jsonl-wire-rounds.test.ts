// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// IT: proxy 捕获文件（行带 source:"-proxy" 标记）的 round-pair 切分。
// 完整数据流：proxy jsonl → readSession 分流 → readWireRounds → turn-split
// → Prisma → /turns API。核心断言：
//   1. 每个 wire round → 输入 turn（本轮新增，verbatim 存 contentJson）
//      + 输出 turn（response，累积请求 verbatim 存 inputMessagesJson）
//   2. wire 顺序原样保留：assistant → system(额度播报) → tool_result，
//      不再把 tool_result 折叠进 assistant（此前 LLM Input 重建的顺序偏差）
//   3. tool_result 按 tool_use_id 挂回输出 turn 的 ToolCall.result
//   4. 旧形态 session（无 contentJson）重新导入时整体重建，不新旧混杂

import { describe, it, expect, afterAll } from 'vitest';
import { NextRequest } from 'next/server';
import path from 'node:path';
import { prisma } from './setup';
import { importSession, deltaRefreshSession } from '@/lib/ingest/data-service';
import { GET as listTurns } from '@/app/api/observe/session/turns/route';
import { GET as getTurn } from '@/app/api/observe/session/turns/[turnId]/route';

const FIXTURE = path.join(__dirname, 'data/claude-sessions/proxy-wire-rounds.jsonl');
const SID = 'proxy-wire-rounds-it';

let sessionId: string | null = null;

afterAll(async () => {
  if (sessionId) {
    try { await prisma.session.delete({ where: { id: sessionId } }); } catch { /* ignore */ }
  }
});

async function importedTurns() {
  return prisma.turn.findMany({
    where: { sessionId: sessionId! },
    orderBy: { turnIndex: 'asc' },
    include: { toolCalls: true },
  });
}

describe('proxy 捕获 → round-pair 切分 → DB', () => {
  it('imports: 3 rounds → 3 输入 + 3 输出，交替成对', async () => {
    const result = await importSession(FIXTURE, SID, prisma, FIXTURE, 'claude-jsonl');
    expect(result.imported).toBe(true);
    const s = await prisma.session.findFirst({ where: { taskId: SID } });
    sessionId = s?.id ?? null;
    expect(sessionId).not.toBeNull();

    const turns = await importedTurns();
    expect(turns.length).toBe(6);
    expect(turns.map(t => t.role)).toEqual([
      'user', 'assistant', 'user', 'assistant', 'user', 'assistant',
    ]);
  });

  it('输入 turn：DB 不存 contentJson（OCP），API 扩展层按需返回 verbatim 消息', async () => {
    const turns = await importedTurns();
    const [in1, , in2] = turns;

    // OCP：DB 里 contentJson 恒 null（管线不感知 proxy 数据）
    expect(in1.contentJson).toBeNull();
    expect(in2.contentJson).toBeNull();

    // content 为可读摘要（真实提问，label 干净）
    expect(in1.content).toBe('帮我查一下版本');
    expect(in2.content).toBe('总结一下');

    // 扩展层（API detail）按需读取 verbatim 消息 —— 像 readFullContext
    const in1Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${in1.id}`),
      { params: Promise.resolve({ turnId: in1.id }) }
    );
    const d1 = await in1Detail.json();
    const w1 = JSON.parse(d1.contentJson);
    expect(w1.wireInput).toBe(true);
    expect(w1.messages.length).toBe(1);
    expect(w1.messages[0].role).toBe('user');
    expect(JSON.stringify(w1.messages[0].content)).toContain('<system-reminder>');

    const in2Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${in2.id}`),
      { params: Promise.resolve({ turnId: in2.id }) }
    );
    const d2 = await in2Detail.json();
    const w2 = JSON.parse(d2.contentJson);
    expect(w2.messages.map((m: { role: string }) => m.role)).toEqual([
      'system', 'user', 'system', 'user',
    ]);
    expect(JSON.stringify(w2.messages[1].content)).toContain('tool_result');
    expect(JSON.stringify(w2.messages[2].content)).toContain('Available agent types');

    // round3 输入含 [已压缩] 标记行
    const in3 = turns[4];
    const in3Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${in3.id}`),
      { params: Promise.resolve({ turnId: in3.id }) }
    );
    const w3 = JSON.parse((await in3Detail.json()).contentJson);
    expect(JSON.stringify(w3.messages[0].content)).toContain('[已压缩]');
  });

  it('输出 turn：tool_result 挂回 ToolCall.result；API 扩展层返回 verbatim LLM Input', async () => {
    const turns = await importedTurns();
    const out1 = turns[1];

    // tool_use → ToolCall，结果按 tool_use_id 挂回
    expect(out1.content).toBe('我来查一下');
    expect(out1.toolCalls.length).toBe(1);
    expect(out1.toolCalls[0].toolName).toBe('Bash');
    expect(out1.toolCalls[0].resultJson).toBe('4.1.0');

    // OCP：DB 里 inputMessagesJson 恒 null —— 扩展层在 API 层按需读取
    expect(out1.inputMessagesJson).toBeNull();

    // API detail 返回扩展层的 verbatim 累积请求
    const out1Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${out1.id}`),
      { params: Promise.resolve({ turnId: out1.id }) }
    );
    const d1 = await out1Detail.json();
    const req1 = JSON.parse(d1.inputMessagesJson);
    expect(req1.length).toBe(1);
    expect(d1.inputMessagesCount).toBe(1);

    // round2 请求 = 6 条，wire 顺序
    const out2 = turns[3];
    const out2Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${out2.id}`),
      { params: Promise.resolve({ turnId: out2.id }) }
    );
    const d2 = await out2Detail.json();
    const req2 = JSON.parse(d2.inputMessagesJson);
    expect(req2.length).toBe(6);
    expect(d2.inputMessagesCount).toBe(6);
    expect(req2.map((m: { role: string }) => m.role)).toEqual([
      'user', 'assistant', 'system', 'tool_result', 'system', 'user',
    ]);
    expect(req2[1].tool_calls[0].name).toBe('Bash');
    expect(req2[1].tool_calls[0].result).toBeNull();
    expect(req2[3].name).toBe('Bash');
    expect(req2[3].content).toBe('4.1.0');

    // round3 请求 = 9 条
    const out3 = turns[5];
    const out3Detail = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${out3.id}`),
      { params: Promise.resolve({ turnId: out3.id }) }
    );
    expect(JSON.parse((await out3Detail.json()).inputMessagesJson).length).toBe(9);
  });

  it('session label 取真实提问（不含 system-reminder）', async () => {
    const s = await prisma.session.findUnique({ where: { id: sessionId! } });
    expect(s!.label).toBe('帮我查一下版本');
    expect(s!.version).toBe('claude-proxy');
  });
});

describe('proxy round-pair → /turns API', () => {
  it('列表返回 6 条交替 turn；详情按存储 verbatim 返回（不再重建）', async () => {
    const listRes = await listTurns(new NextRequest(
      `http://localhost/api/observe/session/turns?taskId=${SID}&framework=claude-code`
    ));
    expect(listRes.status).toBe(200);
    const list = await listRes.json();
    expect(list.items.length).toBe(6);
    expect(list.items.map((t: { role: string }) => t.role)).toEqual([
      'user', 'assistant', 'user', 'assistant', 'user', 'assistant',
    ]);

    const out2Id = list.items[3].turnId;
    const detailRes = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${out2Id}`),
      { params: Promise.resolve({ turnId: out2Id }) }
    );
    expect(detailRes.status).toBe(200);
    const detail = await detailRes.json();
    const msgs = JSON.parse(detail.inputMessagesJson);
    expect(msgs.length).toBe(6);
    expect(msgs.map((m: { role: string }) => m.role)).toEqual([
      'user', 'assistant', 'system', 'tool_result', 'system', 'user',
    ]);
    expect(detail.inputMessagesCount).toBe(6);

    // 输入 turn 详情：contentJson 携带 verbatim wire 消息（前端逐消息渲染）
    const in2Id = list.items[2].turnId;
    const inRes = await getTurn(
      new NextRequest(`http://localhost/api/observe/session/turns/${in2Id}`),
      { params: Promise.resolve({ turnId: in2Id }) }
    );
    const inDetail = await inRes.json();
    const w = JSON.parse(inDetail.contentJson);
    expect(w.wireInput).toBe(true);
    expect(w.messages.length).toBe(4);
  });
});

describe('旧形态 session 重新导入 → 整体重建（不混杂）', () => {
  it('merge path: 旧原生切分 turns（含 system role）被删除重建为 round-pair', async () => {
    // 模拟旧原生切分导入的存量数据：含 system role（原生切分特征）
    // wire 切分只产 user/assistant，rebuildIfWireReshape 检测到旧有 system
    // 而新无 system → 删旧重建
    await prisma.turn.deleteMany({ where: { sessionId: sessionId! } });
    await prisma.turn.createMany({
      data: [
        { sessionId: sessionId!, turnIndex: 0, role: 'system', content: 'old-shape-sys' },
        { sessionId: sessionId!, turnIndex: 1, role: 'user', content: 'old-shape-user' },
        { sessionId: sessionId!, turnIndex: 2, role: 'assistant', content: 'old-shape-assistant' },
      ],
    });
    const before = await prisma.turn.findMany({ where: { sessionId: sessionId! } });
    expect(before.length).toBe(3);
    expect(before.some(t => t.role === 'system')).toBe(true);

    // 重新导入（session 已存在 → merge 路径）
    await importSession(FIXTURE, SID, prisma, FIXTURE, 'claude-jsonl');
    const after = await importedTurns();
    expect(after.length).toBe(6); // 重建，不是 3 旧 + 6 新
    expect(after.some(t => t.content === 'old-shape-user')).toBe(false);
    expect(after.some(t => t.role === 'system')).toBe(false); // wire 切分无 system turn
  });

  it('deltaRefresh path: 同样整体重建', async () => {
    await prisma.turn.deleteMany({ where: { sessionId: sessionId! } });
    await prisma.turn.createMany({
      data: [{ sessionId: sessionId!, turnIndex: 0, role: 'user', content: 'old-shape-user' }],
    });
    await deltaRefreshSession(SID, prisma);
    const after = await importedTurns();
    expect(after.length).toBe(6);
    expect(after.some(t => t.content === 'old-shape-user')).toBe(false);
  });
});
