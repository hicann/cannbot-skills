// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { recoverMainAgentWorkflowBody, MAIN_AGENT_WORKFLOW_NAME } from '@/lib/skill-eval-audit';
import { resolveWorkflowSkillNameAuto } from '@/lib/skill-md-scan';

/**
 * 主 agent workflow skill 的真名（identifier）：扫盘按 turn0 body 前缀匹配 disk SKILL.md
 * → frontmatter name（如 ops-registry-invoke-glacier）。供 Skills/Audit 行显示真名用
 * （替代合成名「主 agent workflow」）。主 agent 不 invoke skill，其 workflow 声明在
 * session 首条 user turn（无 frontmatter/name），故真名只能扫盘按 body 匹配反查。
 *
 * 返回：{ name }（匹配到真名）；{ name: MAIN_AGENT_WORKFLOW_NAME }（有 turn0 但扫盘没对到，
 * 回退合成名）；null（无 turn0/过短 → 该 session 无 workflow 声明）。
 */
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const taskId = searchParams.get('taskId');
    const framework = searchParams.get('framework');

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    const session = await prisma.session.findFirst({
      where: framework ? { taskId, framework } : { taskId },
    });
    if (!session) {
      return NextResponse.json({ error: 'Session not found' }, { status: 404 });
    }

    const body = await recoverMainAgentWorkflowBody(session.id, prisma);
    if (!body) {
      return NextResponse.json({ name: null });
    }

    const name = resolveWorkflowSkillNameAuto(body) ?? MAIN_AGENT_WORKFLOW_NAME;
    return NextResponse.json({ name });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
