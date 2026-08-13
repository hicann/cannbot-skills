// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import fs from 'node:fs';
import { resolveScanRoot, scanAgentDirs, resolveAgentMd } from '@/lib/agent-md-scan';
import { prisma } from '@/lib/db';

/**
 * Skills tab 的 subagent 行"全文"按钮：取 agent .md 内容。
 * agent .md 不在 session（dispatch args 只带 prompt，运行时加载的 system prompt 不持久化），
 * 故从本地文件系统扫（AGENTS_SCAN_ROOT env 或自动探测 skills-dev 根）。
 *
 * 与 skill-content 路由平行：skill-content 取 SKILL.md（从 session invoke 恢复），
 * agent-content 取 agent .md（从磁盘扫）。Skills tab 按行类型（skill/agent）分流。
 */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = request.nextUrl;
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');
    const agentName = searchParams.get('agentName');

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }
    if (!agentName) {
      return NextResponse.json({ error: 'Missing agentName' }, { status: 400 });
    }

    const scanRoot = resolveScanRoot();
    if (!scanRoot) {
      return NextResponse.json(
        { agentName, content: null, source: null, length: 0 },
        { status: 200 },
      );
    }

    // 收集本 session 被派发的 agent 名（用于多插件消歧）
    const session = await prisma.session.findFirst({
      where: framework ? { taskId, framework } : { taskId },
      select: { id: true },
    });
    const sessionAgentNames = new Set<string>();
    if (session) {
      const dispatchEvents = await prisma.skillEvent.findMany({
        where: { turn: { sessionId: session.id }, eventType: 'dispatch' },
        select: { skillName: true },
        distinct: ['skillName'],
      });
      for (const se of dispatchEvents) sessionAgentNames.add(se.skillName);
    }

    const dirs = scanAgentDirs(scanRoot);
    const resolved = resolveAgentMd(agentName, dirs, sessionAgentNames);
    if (!resolved) {
      return NextResponse.json(
        { agentName, content: null, source: null, length: 0 },
        { status: 200 },
      );
    }

    const content = fs.readFileSync(resolved.mdPath, 'utf8');
    return NextResponse.json({
      agentName,
      content,
      source: resolved.dir,
      length: content.length,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
