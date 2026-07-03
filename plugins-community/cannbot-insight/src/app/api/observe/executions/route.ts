// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');

    if (!taskId) {
      return NextResponse.json(
        { error: 'Missing required parameter: taskId' },
        { status: 400 }
      );
    }

    const session = await prisma.session.findFirst({
      where: framework ? { taskId, framework } : { taskId },
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found: taskId=${taskId}` },
        { status: 404 }
      );
    }

    const executions = await prisma.execution.findMany({
      where: { sessionId: session.id },
      orderBy: [{ isSubagent: 'asc' }, { createdAt: 'asc' }],
      include: {
        executionSkills: {
          select: {
            skillName: true,
            skillVersion: true,
            isPrimary: true,
          },
        },
      },
    });

    const rootExecutions = executions.filter(e => !e.isSubagent);
    const subagentExecutions = executions.filter(e => e.isSubagent);

    const items = executions.map(e => ({
      executionId: e.id,
      sessionId: e.sessionId,
      agentName: e.agentName,
      agentSessionId: e.agentSessionId,
      isSubagent: e.isSubagent,
      subagentType: e.subagentType,
      subagentName: e.subagentName,
      parentExecutionId: e.parentExecutionId,
      rootExecutionId: e.rootExecutionId,
      depth: e.depth,
      tokens: e.tokens,
      inputTokens: e.inputTokens,
      outputTokens: e.outputTokens,
      reasoningTokens: e.reasoningTokens,
      cost: e.cost,
      latencyMs: e.latencyMs,
      toolCallCount: e.toolCallCount,
      toolCallErrorCount: e.toolCallErrorCount,
      llmCallCount: e.llmCallCount,
      skillLoadCount: e.skillLoadCount,
      skillInvokeCount: e.skillInvokeCount,
      model: e.model,
      createdAt: e.createdAt.toISOString(),
      skills: e.executionSkills.map(s => ({
        skillName: s.skillName,
        skillVersion: s.skillVersion,
        isPrimary: s.isPrimary,
      })),
    }));

    const root = rootExecutions.map(e => ({
      executionId: e.id,
      agentName: e.agentName,
      agentSessionId: e.agentSessionId,
      isSubagent: e.isSubagent,
      tokens: e.tokens,
      inputTokens: e.inputTokens,
      outputTokens: e.outputTokens,
      reasoningTokens: e.reasoningTokens,
      cost: e.cost,
      latencyMs: e.latencyMs,
      toolCallCount: e.toolCallCount,
      skillLoadCount: e.skillLoadCount,
      skillInvokeCount: e.skillInvokeCount,
      llmCallCount: e.llmCallCount,
      model: e.model,
      subagentCount: subagentExecutions.length,
      createdAt: e.createdAt.toISOString(),
    }));

    const subagents = subagentExecutions.map(e => ({
      executionId: e.id,
      agentName: e.agentName,
      agentSessionId: e.agentSessionId,
      isSubagent: e.isSubagent,
      subagentType: e.subagentType,
      subagentName: e.subagentName,
      parentExecutionId: e.parentExecutionId,
      rootExecutionId: e.rootExecutionId,
      depth: e.depth,
      tokens: e.tokens,
      inputTokens: e.inputTokens,
      outputTokens: e.outputTokens,
      cost: e.cost,
      latencyMs: e.latencyMs,
      toolCallCount: e.toolCallCount,
      skillLoadCount: e.skillLoadCount,
      skillInvokeCount: e.skillInvokeCount,
      llmCallCount: e.llmCallCount,
      model: e.model,
      createdAt: e.createdAt.toISOString(),
    }));

    return NextResponse.json({
      items,
      root,
      subagents,
      totalExecutions: executions.length,
      subagentCount: subagentExecutions.length,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
