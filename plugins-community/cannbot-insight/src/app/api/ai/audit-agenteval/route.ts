// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { prisma } from "@/lib/db"
import { buildStructuredRecords, buildTranscriptArgs } from "@/lib/skill-eval-audit"
import { resolveScanRoot, scanAgentDirs, resolveAgentMd } from "@/lib/agent-md-scan"
import { makeStreamingAuditResponse } from "@/lib/skill-eval-runner"

/**
 * 对当前 session 的某个被 dispatch 的 agent 跑 skill-eval 对账（审"真实执行 vs agent .md 声明"）。
 *
 * agent .md 声明不在 session 里（dispatch args 只带任务 prompt，opencode 运行时加载 agent .md
 * 当 system prompt 不持久化），故从本地文件系统扫：AGENTS_SCAN_ROOT env（或自动探测 skills-dev 根）
 * 下 plugins-official/<plugin>/agents/ + <domain>/agents/ + .opencode/agents/。多插件歧义（developer.md 在多个插件）
 * 按本 session 被派发 agent 名集合的覆盖率消歧认对插件。
 *
 * session 侧走结构化 records（--transcript），buildStructuredRecords 已给子 agent turn 带 agent 字段，
 * skill-eval 切片器按 agent 归属切段（--kind agent）。skill-eval 自带 LLM（claude -p），不透传 provider。
 */
const AGENT_NAME_RE = /^[A-Za-z0-9._-]+$/

export async function POST(req: Request) {
  const { taskId, agentName, framework } = await req.json()
  if (!taskId || !agentName) {
    return NextResponse.json({ error: "Missing taskId or agentName" }, { status: 400 })
  }
  if (!AGENT_NAME_RE.test(String(agentName))) {
    return NextResponse.json(
      { error: `非法 agentName "${agentName}"（仅允许字母/数字/._-）` },
      { status: 400 },
    )
  }

  const scanRoot = resolveScanRoot()
  if (!scanRoot) {
    return NextResponse.json(
      { error: "未配 AGENTS_SCAN_ROOT 且自动探测 skills-dev 根失败：agent-audit 需要一个含 plugins-official/*/agents/ 的扫描根。设 AGENTS_SCAN_ROOT env 指向 skills-dev 仓库根即可。" },
      { status: 503 },
    )
  }

  const session = await prisma.session.findFirst({
    where: { taskId, ...(framework ? { framework } : {}) },
  })
  if (!session) {
    return NextResponse.json({ error: `找不到 taskId=${taskId} 的 session` }, { status: 404 })
  }

  // 本 session 被派发的 agent 名集合（用于多插件覆盖率消歧）
  const bridges = await prisma.interactionBridge.findMany({
    where: { sessionId: session.id },
    select: { subagentName: true, subagentType: true },
  })
  const sessionAgentNames = new Set<string>()
  for (const b of bridges) {
    const n = b.subagentName ?? b.subagentType
    if (n) sessionAgentNames.add(n)
  }

  const dirs = scanAgentDirs(scanRoot)
  const resolved = resolveAgentMd(String(agentName), dirs, sessionAgentNames)
  if (!resolved) {
    return NextResponse.json(
      { error: `在扫描根下找不到 agent "${agentName}" 的 .md（扫描了 ${dirs.length} 个 agents/ 目录）。检查 AGENTS_SCAN_ROOT 是否指向含该 agent 定义的仓库根。` },
      { status: 404 },
    )
  }

  let agentBody: string
  try {
    agentBody = fs.readFileSync(resolved.mdPath, "utf8")
  } catch {
    return NextResponse.json(
      { error: `读 agent 声明失败：${resolved.mdPath}` },
      { status: 404 },
    )
  }

  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "skill-eval-agent-audit-"))
  try {
    const safeName = String(agentName).replace(/[^a-zA-Z0-9._-]/g, "_")
    const skillDir = path.join(tmp, safeName)
    fs.mkdirSync(skillDir, { recursive: true })
    // agent .md 自带 YAML frontmatter（name: <agentName>），与 session 里 agentName 一致——
    // skill-eval 的 --kind agent 靠这个 name 切段（匹配 records 的 agent 字段）。原样物化即可。
    fs.writeFileSync(path.join(skillDir, "SKILL.md"), agentBody)

    const transcriptPath = path.join(tmp, "session.json")
    fs.writeFileSync(transcriptPath, JSON.stringify(await buildStructuredRecords(session.id, prisma)))

    const outputDir = path.join(tmp, "out")
    fs.mkdirSync(outputDir, { recursive: true })
    const args = buildTranscriptArgs({ skillDir, transcriptPath, outputDir, kind: "agent" })

    // 流式 NDJSON：spawn skill-eval，stdout 解析进度回传 progress 事件，结束回传 result/error。
    // tmp 清理放进流的 finally（等 skill-eval 跑完），不能在路由 return 时删。
    return makeStreamingAuditResponse(args, outputDir, () =>
      fs.rmSync(tmp, { recursive: true, force: true }),
    )
  } catch (e) {
    fs.rmSync(tmp, { recursive: true, force: true })
    return NextResponse.json(
      { error: `准备 agent 对账失败：${e instanceof Error ? e.message : String(e)}` },
      { status: 500 },
    )
  }
}
