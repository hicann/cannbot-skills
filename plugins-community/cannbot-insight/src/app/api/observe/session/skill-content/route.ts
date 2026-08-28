// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { selectSkillContent, type SkillToolCall } from '@/lib/skill-content';
import { recoverWorkflowDeclaration, MAIN_AGENT_WORKFLOW_NAME } from '@/lib/sift-audit';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');
    const skillName = searchParams.get('skillName');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required query param: taskId' },
        { status: 400 }
      );
    }
    if (!skillName) {
      return NextResponse.json(
        { error: 'Missing required query param: skillName' },
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

    // 主 agent 编排：声明是 workflow skill 的编排规程（SKILL.md body / skill resources），
    // 从主 agent Skill invoke 定位 skillName → recoverWorkflowDeclaration 恢复。
    if (skillName === MAIN_AGENT_WORKFLOW_NAME) {
      const decl = await recoverWorkflowDeclaration(session.id, prisma);
      if (!decl) {
        return NextResponse.json(
          { skillName, content: null, source: null, length: 0 },
          { status: 200 },
        );
      }
      return NextResponse.json({
        skillName,
        content: decl.content,
        source: decl.source,
        length: decl.content.length,
      });
    }

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

    const result = selectSkillContent(ops, skillName);

    if (!result) {
      return NextResponse.json(
        { skillName, content: null, source: null, length: 0, fullRead: false, maxLine: null },
        { status: 200 }
      );
    }

    return NextResponse.json({ skillName, ...result });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
