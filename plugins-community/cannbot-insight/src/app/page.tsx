// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { MetricCards } from '@/components/MetricCards';
import { SessionList } from '@/components/SessionList';
import { LocalFileImport } from '@/components/LocalFileImport';
import { ImportHistory } from '@/components/ImportHistory';
import { VERSION_DISPLAY } from '@/lib/version';
import { BRAND_NAME } from '@/lib/branding';
import { prisma } from '@/lib/db';
import { BarChart3Icon } from 'lucide-react';
import { countToolCallErrors } from '@/lib/tool-call-errors';

export default async function Home({
  searchParams,
}: {
  searchParams: Promise<{ page?: string; pageSize?: string; isSubagent?: string; user?: string }>;
}) {
  const params = await searchParams;
  const page = Math.max(1, Number(params.page ?? 1));
  const pageSize = Math.max(1, Math.min(100, Number(params.pageSize ?? 20)));

  const totalSessions = await prisma.session.count();
  const aggregates = await prisma.session.aggregate({
    _sum: { totalTokens: true, totalCost: true, totalLatencyMs: true },
    _avg: { totalLatencyMs: true },
  });

  const where: Record<string, unknown> = {};
  if (params.user) where.user = params.user;

  const sessions = await prisma.session.findMany({
    where,
    orderBy: { startTime: 'desc' },
    skip: (page - 1) * pageSize,
    take: pageSize,
    select: {
      id: true,
      taskId: true,
      query: true,
      framework: true,
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
      sourcePath: true,
    },
  });

  const total = await prisma.session.count({ where });

  // Compute per-session error counts from tool call resultJson
  const sessionIds = sessions.map(s => s.id);
  const errorToolCalls = await prisma.toolCall.findMany({
    where: {
      turn: { sessionId: { in: sessionIds } },
      OR: [
        { state: { in: ['error', 'failed'] } },
        { errorType: { not: null } },
        { resultJson: { contains: '<tool_use_error>' } },
        { resultJson: { contains: 'Exit code' } },
      ],
    },
    select: {
      id: true,
      toolName: true,
      resultJson: true,
      state: true,
      errorType: true,
      errorMessage: true,
      turn: { select: { sessionId: true, turnIndex: true } },
    },
  });

  const errorSkillEvents = await prisma.skillEvent.findMany({
    where: {
      turn: { sessionId: { in: sessionIds } },
      success: false,
      errorMessage: { not: null },
    },
    select: {
      skillName: true,
      eventType: true,
      success: true,
      errorMessage: true,
      turn: { select: { sessionId: true, turnIndex: true } },
    },
  });

  const errorCountsBySession = countToolCallErrors(errorToolCalls, errorSkillEvents);

  const totalErrors = sessionIds.reduce((sum, sid) => sum + (errorCountsBySession.get(sid)?.count ?? 0), 0);

  const items = sessions.map((s) => {
    const eInfo = errorCountsBySession.get(s.id)
    return {
      sessionId: s.id,
      taskId: s.taskId,
      query: s.query,
      startTime: s.startTime.toISOString(),
      endTime: s.endTime ? s.endTime.toISOString() : null,
      totalTokens: s.totalTokens,
      totalCost: s.totalCost,
      totalLatencyMs: s.totalLatencyMs,
      totalToolCallCount: s.totalToolCallCount,
      totalSkillLoadCount: s.totalSkillLoadCount,
      totalSubagentCount: s.totalSubagentCount,
      model: s.model,
      user: s.user,
      framework: s.framework,
      sourcePath: s.sourcePath,
      errorCount: eInfo?.count ?? 0,
      firstErrorTurnIndex: eInfo?.firstTurnIndex ?? null,
    }
  });

  return (
    <div className="flex min-h-screen flex-col bg-background">
      <main className="flex w-full max-w-7xl flex-col gap-6 px-6 py-8 mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              {BRAND_NAME}
            </h1>
            <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
          </div>
          <div className="flex items-center gap-3">
            <LocalFileImport />
            <a href="/monitor" className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md border text-sm font-medium hover:bg-accent/30 transition-colors">
              <BarChart3Icon className="size-4 text-pink-500" />
              上下文监控
            </a>
          </div>
        </div>
        <MetricCards
          totalSessions={totalSessions}
          totalTokens={aggregates._sum.totalTokens ?? 0}
          totalCost={aggregates._sum.totalCost ?? 0}
          avgLatencyMs={aggregates._avg.totalLatencyMs ?? 0}
          totalErrors={totalErrors}
        />
        <SessionList items={items} total={total} page={page} pageSize={pageSize} />
        <ImportHistory />
      </main>
    </div>
  );
}
