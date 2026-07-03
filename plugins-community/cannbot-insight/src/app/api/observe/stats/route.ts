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

    if (!taskId) {
      const totalSessions = await prisma.session.count();
      const aggregates = await prisma.session.aggregate({
        _sum: { totalTokens: true, totalCost: true, totalLatencyMs: true },
        _avg: { totalLatencyMs: true },
      });

      return NextResponse.json({
        totalSessions,
        totalTokens: aggregates._sum.totalTokens ?? 0,
        totalCost: aggregates._sum.totalCost ?? 0,
        totalLatencyMs: aggregates._sum.totalLatencyMs ?? 0,
        avgLatencyMs: aggregates._avg.totalLatencyMs ?? 0,
      });
    }

    const session = await prisma.session.findFirst({
      where: { taskId },
      select: {
        totalTokens: true,
        totalCost: true,
        totalLatencyMs: true,
        totalInputTokens: true,
        totalOutputTokens: true,
        totalReasoningTokens: true,
        totalCacheReadTokens: true,
        totalCacheWriteTokens: true,
        totalToolCallCount: true,
        totalSkillLoadCount: true,
        totalSubagentCount: true,
        totalLlmCallCount: true,
      },
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found: taskId=${taskId}` },
        { status: 404 }
      );
    }

    return NextResponse.json({
      taskId,
      totalTokens: session.totalTokens,
      totalInputTokens: session.totalInputTokens,
      totalOutputTokens: session.totalOutputTokens,
      totalReasoningTokens: session.totalReasoningTokens,
      totalCacheReadTokens: session.totalCacheReadTokens,
      totalCacheWriteTokens: session.totalCacheWriteTokens,
      totalCost: session.totalCost,
      totalLatencyMs: session.totalLatencyMs,
      totalToolCallCount: session.totalToolCallCount,
      totalSkillLoadCount: session.totalSkillLoadCount,
      totalSubagentCount: session.totalSubagentCount,
      totalLlmCallCount: session.totalLlmCallCount,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
