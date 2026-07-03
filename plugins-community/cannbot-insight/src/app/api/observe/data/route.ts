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
    const page = Math.max(1, Number(searchParams.get('page') ?? 1));
    const pageSize = Math.max(1, Math.min(100, Number(searchParams.get('pageSize') ?? 20)));
    const isSubagent = searchParams.get('isSubagent');
    const user = searchParams.get('user');

    const where: Record<string, unknown> = {};
    if (user) where.user = user;
    if (isSubagent !== null && isSubagent !== undefined && isSubagent !== '') {
      where.executions = { some: { isSubagent: isSubagent === 'true' } };
    }

    const total = await prisma.session.count({ where });

    const sessions = await prisma.session.findMany({
      where,
      orderBy: { startTime: 'desc' },
      skip: (page - 1) * pageSize,
      take: pageSize,
      select: {
        id: true,
        taskId: true,
        query: true,
        startTime: true,
        endTime: true,
        totalTokens: true,
        totalCost: true,
        totalLatencyMs: true,
        totalToolCallCount: true,
        totalSkillLoadCount: true,
        totalSubagentCount: true,
        model: true,
        user: true,
        framework: true,
      },
    });

    const items = sessions.map((s) => ({
      sessionId: s.id,
      taskId: s.taskId,
      query: s.query,
      startTime: s.startTime,
      endTime: s.endTime,
      totalTokens: s.totalTokens,
      totalCost: s.totalCost,
      totalLatencyMs: s.totalLatencyMs,
      totalToolCallCount: s.totalToolCallCount,
      totalSkillLoadCount: s.totalSkillLoadCount,
      totalSubagentCount: s.totalSubagentCount,
      model: s.model,
      user: s.user,
      framework: s.framework,
    }));

    return NextResponse.json({ items, total, page });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
