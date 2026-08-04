// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, afterEach } from 'vitest';
import { NextRequest } from 'next/server';
import { prisma } from './setup';
import { GET } from '@/app/api/observe/session/turns/search/route';

const createdSessionIds: string[] = [];

async function seedSessionWithToolCall(opts: {
  taskId: string;
  turnIndex: number;
  toolName: string;
  argsJson?: string | null;
  resultJson?: string | null;
  content?: string | null;
}) {
  const session = await prisma.session.create({ data: { taskId: opts.taskId, framework: 'claude-code' } });
  createdSessionIds.push(session.id);
  const turn = await prisma.turn.create({
    data: { sessionId: session.id, turnIndex: opts.turnIndex, role: 'assistant', content: opts.content ?? null },
  });
  const toolCall = await prisma.toolCall.create({
    data: {
      turnId: turn.id,
      toolCallId: `call-${opts.taskId}-${opts.turnIndex}`,
      toolName: opts.toolName,
      argsJson: opts.argsJson ?? null,
      resultJson: opts.resultJson ?? null,
    },
  });
  return { session, turn, toolCall };
}

afterEach(async () => {
  for (const id of createdSessionIds.splice(0)) {
    await prisma.session.delete({ where: { id } }).catch(() => {});
  }
});

function makeRequest(taskId: string, keyword: string) {
  return new NextRequest(
    `http://localhost/api/observe/session/turns/search?taskId=${encodeURIComponent(taskId)}&keyword=${encodeURIComponent(keyword)}&framework=claude-code`
  );
}

describe('turns search API — tool call args matching', () => {
  it('finds keyword present only in toolCall.argsJson (e.g. file_path)', async () => {
    const taskId = `trace-args-${Date.now()}`;
    await seedSessionWithToolCall({
      taskId,
      turnIndex: 34,
      toolName: 'Read',
      argsJson: JSON.stringify({ file_path: '/repo/DESIGN.md' }),
      resultJson: '# Design Doc\n...content without the filename...',
      content: 'Let me read the design doc.',
    });

    const res = await GET(makeRequest(taskId, 'DESIGN.md'));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.items.length).toBe(1);
    const item = data.items[0];
    expect(item.turnIndex).toBe(34);
    expect(item.matchField).toBe('toolArgs');
    expect(item.toolName).toBe('Read');
    expect(item.matchContext).toContain('DESIGN.md');
  });

  it('does not match when keyword absent from content, args, result, error', async () => {
    const taskId = `trace-none-${Date.now()}`;
    await seedSessionWithToolCall({
      taskId,
      turnIndex: 1,
      toolName: 'Bash',
      argsJson: JSON.stringify({ command: 'ls -la' }),
      resultJson: 'total 0',
      content: 'running ls',
    });

    const res = await GET(makeRequest(taskId, 'DESIGN.md'));
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.items.length).toBe(0);
  });

  it('still matches keyword in resultJson as toolResult', async () => {
    const taskId = `trace-result-${Date.now()}`;
    await seedSessionWithToolCall({
      taskId,
      turnIndex: 5,
      toolName: 'Grep',
      argsJson: JSON.stringify({ pattern: 'foo' }),
      resultJson: 'match found in DESIGN.md line 12',
      content: 'searching',
    });

    const res = await GET(makeRequest(taskId, 'DESIGN.md'));
    const data = await res.json();
    expect(data.items.length).toBe(1);
    expect(data.items[0].matchField).toBe('toolResult');
  });

  it('returns 400 when keyword missing', async () => {
    const req = new NextRequest('http://localhost/api/observe/session/turns/search?taskId=whatever');
    const res = await GET(req);
    expect(res.status).toBe(400);
  });
});
