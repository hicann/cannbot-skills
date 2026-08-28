// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { readAvailableSkills } from '@/lib/ingest/adapters/claude-jsonl-full-context';
import { aggregateUsedSkills, buildCoverage } from '@/lib/skill-coverage';
import { isProxyVersion } from '@/lib/shared/session-format';

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
      select: { id: true, taskId: true, framework: true, version: true, sourcePath: true },
    });

    if (!session) {
      return NextResponse.json(
        { error: `Session not found for taskId: "${taskId}"` },
        { status: 404 }
      );
    }

    const events = await prisma.skillEvent.findMany({
      where: { turn: { sessionId: session.id } },
      select: { skillName: true, eventType: true, success: true },
    });
    const used = aggregateUsedSkills(events);

    let available: Array<{ name: string; description: string; origin: string | null }> | null = null;
    let listFormat: 'claude-list' | 'opencode-xml' | null = null;
    let degradedReason: string | null = null;

    // 仅 proxy 捕获会话展示覆盖度：wire verbatim 落盘系统提示，全集确定性恢复。
    // proxy 标记 = 导入时统一分类器写入 Session.version 的 "-proxy" 后缀。
    const isProxy = isProxyVersion(session.version);

    if (isProxy && session.sourcePath) {
      const parsed = readAvailableSkills(session.sourcePath);
      if (parsed) {
        available = parsed.skills;
        listFormat = parsed.format;
      } else {
        degradedReason = 'proxy 捕获中未找到可用 skills 全集（系统提示未含 skills 注入段）';
      }
    } else if (isProxy) {
      degradedReason = 'proxy 捕获源文件不可用（sourcePath 缺失或文件已移走）';
    } else {
      degradedReason = '仅 proxy 捕获会话展示 Skill 覆盖度（wire verbatim 含系统提示注入）；原生会话暂不支持';
    }

    const coverage = buildCoverage(available, used);

    return NextResponse.json({
      taskId: session.taskId,
      framework: session.framework,
      hasAvailableList: available != null,
      listFormat,
      degradedReason,
      usedSkillCount: used.length,
      ...coverage,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
