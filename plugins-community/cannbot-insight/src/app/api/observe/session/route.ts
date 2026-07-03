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
    const sessionId = searchParams.get('sessionId');

    if (!taskId && !sessionId) {
      return NextResponse.json(
        { error: 'Missing required parameter: taskId or sessionId' },
        { status: 400 }
      );
    }

    let where: Record<string, string>;
    if (sessionId) {
      where = { id: sessionId };
    } else if (framework) {
      where = { taskId: taskId!, framework };
    } else {
      where = { taskId: taskId! };
    }

    const sessions = await prisma.session.findMany({
      where,
      include: {
        executions: {
          orderBy: { createdAt: 'asc' },
          select: {
            id: true,
            agentName: true,
            agentSessionId: true,
            isSubagent: true,
            parentExecutionId: true,
            tokens: true,
            maxSingleCallTokens: true,
            cost: true,
            toolCallCount: true,
            skillLoadCount: true,
            model: true,
            createdAt: true,
            latencyMs: true,
          },
        },
        skills: {
          select: {
            skillName: true,
            skillVersion: true,
            invocationCount: true,
          },
        },
      },
    });

    if (sessions.length === 0) {
      return NextResponse.json(
        { error: `Session not found: taskId=${taskId}` },
        { status: 404 }
      );
    }

    if (sessions.length > 1 && !framework && !sessionId) {
      return NextResponse.json({
        multiple: true,
        sessions: sessions.map(s => ({
          sessionId: s.id,
          taskId: s.taskId,
          framework: s.framework,
          label: s.label,
          query: s.query,
          model: s.model,
          startTime: s.startTime,
        })),
      });
    }

    const session = sessions[0];

    const userTurns = await prisma.turn.findMany({
      where: { sessionId: session.id, role: 'user' },
      orderBy: { turnIndex: 'asc' },
      select: { contentSummary: true, content: true, isSubagent: true, subagentSessionId: true },
    });

    let rootFirstPrompt: string | null = null;
    const subFirstPrompts = new Map<string, string>();
    for (const t of userTurns) {
      if (!t.isSubagent && rootFirstPrompt === null) {
        rootFirstPrompt = t.contentSummary ?? t.content?.substring(0, 80) ?? null;
      }
      if (t.isSubagent && t.subagentSessionId && !subFirstPrompts.has(t.subagentSessionId)) {
        subFirstPrompts.set(t.subagentSessionId, t.contentSummary ?? t.content?.substring(0, 80) ?? '');
      }
    }

    const agents = session.executions.map((e) => ({
      executionId: e.id,
      agentName: e.agentName,
      agentSessionId: e.agentSessionId,
      isSubagent: e.isSubagent,
      parentExecutionId: e.parentExecutionId,
      tokens: e.tokens,
      maxSingleCallTokens: e.maxSingleCallTokens,
      cost: e.cost,
      toolCallCount: e.toolCallCount,
      skillLoadCount: e.skillLoadCount,
      model: e.model,
      createdAt: e.createdAt.toISOString(),
      latencyMs: e.latencyMs,
      firstPrompt: e.isSubagent
        ? (subFirstPrompts.get(e.agentSessionId ?? '') ?? null)
        : rootFirstPrompt,
    }));

    const skills = session.skills.map((s) => ({
      skillName: s.skillName,
      version: s.skillVersion,
      invocationCount: s.invocationCount,
    }));

    return NextResponse.json({
      sessionId: session.id,
      taskId: session.taskId,
      label: session.label,
      query: session.query,
      framework: session.framework,
      frameworkVersion: session.version,
      parentId: session.parentId,
      directory: session.directory,
      summaryAdditions: session.summaryAdditions,
      summaryDeletions: session.summaryDeletions,
      summaryFiles: session.summaryFiles,
      startTime: session.startTime,
      endTime: session.endTime,
      totalTokens: session.totalTokens,
      totalInputTokens: session.totalInputTokens,
      totalOutputTokens: session.totalOutputTokens,
      totalReasoningTokens: session.totalReasoningTokens,
      totalCacheReadTokens: session.totalCacheReadTokens,
      totalCacheWriteTokens: session.totalCacheWriteTokens,
      totalCost: session.totalCost,
      totalLatencyMs: session.totalLatencyMs,
      totalToolCallCount: session.totalToolCallCount,
      totalLlmCallCount: session.totalLlmCallCount,
      totalSkillLoadCount: session.totalSkillLoadCount,
      totalSubagentCount: session.totalSubagentCount,
      model: session.model,
      user: session.user,
      sourcePath: session.sourcePath,
      agents,
      skills,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
