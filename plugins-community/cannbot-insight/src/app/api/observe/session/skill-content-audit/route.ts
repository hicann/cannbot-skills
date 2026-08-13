// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { selectSkillContent, type SkillToolCall } from '@/lib/skill-content';

export interface SkillContentAuditItem {
  skillName: string;
  hasContent: boolean;
  source: 'skill-tool' | 'read' | null;
  length: number;
  lines: number;
  /** 是否判定为读取了完整 SKILL.md（skill-tool 注入恒 true；read 需无 offset/limit 且无截断标记）。 */
  fullRead: boolean;
  /** read 来源时实际读到的最大行号；skill-tool 为 null。 */
  maxLine: number | null;
}

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required query param: taskId' },
        { status: 400 }
      );
    }

    const session = await prisma.session.findFirst({
      where: framework ? { taskId, framework } : { taskId },
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found for taskId: "${taskId}"` },
        { status: 404 }
      );
    }

    const skillEventsRaw = await prisma.skillEvent.findMany({
      where: { turn: { sessionId: session.id } },
      select: { skillName: true, eventType: true },
    });
    // 仅 dispatch 的 skillName 是被分派的子代理（如 blackbox-designer），非 skill，不参与全文审计。
    const names = Array.from(new Set(
      skillEventsRaw
        .filter(se => se.eventType !== 'dispatch')
        .map(se => se.skillName)
    ));

    const toolCalls = await prisma.toolCall.findMany({
      where: {
        turn: { sessionId: session.id },
        OR: [
          { isSkillRelated: true },
          { AND: [{ toolName: 'Read' }, { argsJson: { contains: 'SKILL.md' } }] },
          { AND: [{ toolName: 'read' }, { argsJson: { contains: 'SKILL.md' } }] },
        ],
      },
      select: {
        toolName: true,
        argsJson: true,
        resultJson: true,
        startedAt: true,
      },
    });

    const ops: SkillToolCall[] = toolCalls.map(tc => ({
      toolName: tc.toolName,
      argsJson: tc.argsJson,
      resultJson: tc.resultJson,
      startedAt: tc.startedAt ?? null,
    }));

    const items: SkillContentAuditItem[] = names.map(name => {
      const r = selectSkillContent(ops, name);
      const content = r?.content ?? '';
      return {
        skillName: name,
        hasContent: r != null && r.length > 0,
        source: r?.source ?? null,
        length: r?.length ?? 0,
        lines: content.length > 0 ? content.split('\n').length : 0,
        fullRead: r?.fullRead ?? false,
        maxLine: r?.maxLine ?? null,
      };
    });

    return NextResponse.json({ items });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
