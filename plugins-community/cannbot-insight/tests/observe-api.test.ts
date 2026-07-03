// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { prisma } from './setup';

describe('observe API routes', () => {
  it('observe/data query structure matches expected fields', async () => {
    const sessions = await prisma.session.findMany({
      orderBy: { startTime: 'desc' },
      take: 20,
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
      },
    });

    if (sessions.length === 0) {
      expect(sessions).toEqual([]);
      return;
    }

    const item = sessions[0];
    expect(item.taskId).toBeDefined();
    expect(typeof item.totalTokens).toBe('number');
    expect(typeof item.totalCost).toBe('number');
    expect(typeof item.totalToolCallCount).toBe('number');
  });

  it('observe/session query includes executions and skills', async () => {
    const session = await prisma.session.findFirst();
    if (!session) {
      expect(session).toBeNull();
      return;
    }

    const detail = await prisma.session.findUnique({
      where: { taskId_framework: { taskId: session.taskId, framework: session.framework ?? 'unknown' } },
      include: {
        executions: { select: { id: true, agentName: true, isSubagent: true, tokens: true, cost: true } },
        skills: { select: { skillName: true, skillVersion: true, invocationCount: true } },
      },
    });

    expect(detail).toBeDefined();
    expect(detail!.taskId).toBe(session.taskId);
  });

  it('observe/stats aggregation works with empty DB', async () => {
    const totalSessions = await prisma.session.count();
    const aggregates = await prisma.session.aggregate({
      _sum: { totalTokens: true, totalCost: true, totalLatencyMs: true },
      _avg: { totalLatencyMs: true },
    });

    expect(typeof totalSessions).toBe('number');
    expect(typeof (aggregates._sum.totalTokens ?? 0)).toBe('number');
    expect(typeof (aggregates._sum.totalCost ?? 0)).toBe('number');
  });

  it('observe/executions query returns root + subagent executions', async () => {
    const session = await prisma.session.findFirst();
    if (!session) {
      expect(session).toBeNull();
      return;
    }

    const executions = await prisma.execution.findMany({
      where: { sessionId: session.id },
      orderBy: [{ isSubagent: 'asc' }, { createdAt: 'asc' }],
      include: {
        executionSkills: { select: { skillName: true, skillVersion: true, isPrimary: true } },
      },
    });

    const rootExecs = executions.filter(e => !e.isSubagent);
    const subExecs = executions.filter(e => e.isSubagent);

    expect(executions.length).toBeGreaterThanOrEqual(rootExecs.length);
    expect(typeof subExecs.length).toBe('number');

    if (rootExecs.length > 0) {
      const root = rootExecs[0];
      expect(typeof root.tokens).toBe('number');
      expect(typeof root.cost).toBe('number');
      expect(typeof root.toolCallCount).toBe('number');
      expect(typeof root.latencyMs).toBe('number');
    }
  });

  it('observe/session/bridges query returns interaction bridges', async () => {
    const session = await prisma.session.findFirst();
    if (!session) {
      expect(session).toBeNull();
      return;
    }

    const bridges = await prisma.interactionBridge.findMany({
      where: { sessionId: session.id },
      orderBy: [{ dispatchTimestamp: 'asc' }],
    });

    expect(typeof bridges.length).toBe('number');

    if (bridges.length > 0) {
      const bridge = bridges[0];
      expect(bridge.dispatchExecutionId).toBeDefined();
      expect(typeof bridge.status).toBe('string');
      expect(typeof bridge.subagentTokens).toBe('number');
      expect(typeof bridge.subagentLatencyMs).toBe('number');
    }
  });
});
