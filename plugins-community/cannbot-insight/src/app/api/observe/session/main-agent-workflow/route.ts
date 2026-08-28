// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This file is licensed under the CANN Open Software License Agreement Version 2.0.

import { NextResponse } from 'next/server';
import { prisma } from '@/lib/db';
import { recoverWorkflowDeclaration, resolveWorkflowSkillName, MAIN_AGENT_WORKFLOW_NAME } from '@/lib/sift-audit';

/**
 * 主 agent 编排 对账目标的可用性 + 真名：
 *
 * - available：session 有 workflow 编排规程（主 agent Skill invoke → SKILL.md body /
 *   skill resources 恢复）→ 可 root 对账。无 workflow skill invoke → available=false。
 *
 * - name：workflow skill 真名（主 agent Skill invoke 的 skillName，如
 *   ops-registry-invoke-glacier）。无 invoke → 回退合成名 MAIN_AGENT_WORKFLOW_NAME。
 *
 * - source：声明来源标识（如 "task-prompts.md" / "ops-registry-invoke-workflow (SKILL.md)"）。
 *
 * 返回：{ available, name, source }。available=true 时 name 非 null、source 非 null；
 * available=false 时 name=null、source=null。
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

    // available + source：workflow 编排规程能否恢复——root 对账送它
    const decl = await recoverWorkflowDeclaration(session.id, prisma);
    if (!decl) {
      return NextResponse.json({ available: false, name: null, source: null });
    }

    // name：主 agent Skill invoke 的 skillName（workflow skill 真名）
    const name = (await resolveWorkflowSkillName(session.id, prisma)) ?? MAIN_AGENT_WORKFLOW_NAME;
    return NextResponse.json({ available: true, name, source: decl.source });
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
