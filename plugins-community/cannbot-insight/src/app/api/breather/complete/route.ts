// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import { OpenCodeClient } from "@/lib/breather/opencode-api"
import { readHandoffRegistry, writeHandoffRegistry, extractOperatorNameFromPath, computeSessionIdShort, computeContinuationTitle, computeHandoffDocName, extractHandoffNumFromBirthDoc, findSessionRecord, addOrUpdateSessionRecord, addChildToSession } from "@/lib/breather/handoff-registry"
import fs from "node:fs"
import path from "node:path"

function buildResumePrompt(operatorName: string, projectPath: string, sessionIdShort: string, handoffNum: number): string {
  const handoffDocName = `SESSION-HANDOFF-${sessionIdShort}-${handoffNum}.md`
  return `这是一个从上一个 session 交接过来的任务。请按以下顺序获取上下文，严格按顺序执行，不要跳步：

1. 用 Read 工具读取 operators/${operatorName}/docs/${handoffDocName} 的完整内容——这是交接文件，包含阻塞项、下一步指令和上一 session 的关键信息。

2. **如果第 1 节阻塞项中有标注"⚠️ 需用户确认"的问题，必须先用 question 工具逐一向用户提出这些问题，获得回答后再继续。不要跳过这些问题直接执行下一步。**

3. 用 Read 工具读取 operators/${operatorName}/docs/LOG.md 的完整内容——开发日志，包含进度状态表、交付件路径和开发记录。确认 LOG.md 中的进度与 ${handoffDocName} 中描述的一致。

4. 根据 ${handoffDocName} 中「如何接续」一节指定的下一步，读取该步骤所需的输入文件（如 REQUIREMENTS.md、spec.yaml、DESIGN_PREP.md 等）——输入文件清单已在 handoff 文档中明确列出。

5. 按 ${handoffDocName} 中「如何接续」一节继续执行下一步。**如果「如何接续」指定了触发 Subagent（如 ascendc-ops-architect），必须使用 task 工具以该 subagent 类型启动子任务来执行，不要用 general agent 代替。触发语句已在 handoff 文档的触发指令表格中给出，直接使用即可。**

**重要：每完成一个步骤后，必须用 Write 工具更新 operators/${operatorName}/docs/LOG.md 中的进度状态表**——将对应步骤的状态从 ⬜/🔄 改为 ✅，确保 LOG.md 始终反映最新进度。这是为了让监控工具（上下文窗口监控）能正确追踪当前进展。后续每个步骤完成时都要更新 LOG.md，不要遗漏。`
}

function cleanupBatchFiles(projectPath: string, operatorName: string, sessionId: string): void {
  const batchDir = operatorName
    ? path.join(projectPath, "operators", operatorName, "docs", ".handoff-batches")
    : path.join(projectPath, ".handoff-batches")
  if (!fs.existsSync(batchDir)) return
  const prefix = `batch-${sessionId.slice(0, 12)}`
  for (const file of fs.readdirSync(batchDir)) {
    if (file.startsWith(prefix)) {
      try { fs.unlinkSync(path.join(batchDir, file)) } catch {}
    }
  }
  try {
    if (fs.readdirSync(batchDir).length === 0) fs.rmdirSync(batchDir)
  } catch {}
}

export async function POST(request: Request) {
  try {
  const body = await request.json()
  const { handoffSessionId, originalSessionId, host = "localhost", port = 15031 } = body

  if (!handoffSessionId || !originalSessionId) {
    return NextResponse.json({ error: "缺少 handoffSessionId 或 originalSessionId" }, { status: 400 })
  }

  const baseUrl = `http://${host}:${port}`
  const client = new OpenCodeClient(baseUrl)

  const healthy = await client.health()
  if (!healthy) {
    return NextResponse.json({ error: `无法连接到 OpenCode (${baseUrl})` }, { status: 503 })
  }

  const originalSession = await client.getSession(originalSessionId)
  const projectPath = originalSession?.directory
  if (!projectPath) {
    return NextResponse.json({ error: `无法获取 session ${originalSessionId} 的 projectPath` }, { status: 500 })
  }
  const operatorName = extractOperatorNameFromPath(projectPath)
  const originalTitle = originalSession?.title ?? "session"

  const registry = readHandoffRegistry(projectPath)

  const sessionIdShort = computeSessionIdShort(originalSessionId)
  const handoffSessionRecord = findSessionRecord(registry, handoffSessionId)
  const handoffNum = handoffSessionRecord?.handoffDoc
    ? extractHandoffNumFromBirthDoc(handoffSessionRecord.handoffDoc)
    : 1

  if ((handoffSessionRecord?.to?.length ?? 0) > 0) {
    const existingContId = handoffSessionRecord!.to[0]
    const existingContRecord = findSessionRecord(registry, existingContId)
    return NextResponse.json({ success: true, phase: "already-completed", continuationSession: { id: existingContId, title: computeContinuationTitle(sessionIdShort, handoffNum) }, handoffSessionId, operatorName })
  }

  const handoffDocName = handoffSessionRecord?.handoffDoc ?? computeHandoffDocName(sessionIdShort, handoffNum)

  const continuationTitle = computeContinuationTitle(sessionIdShort, handoffNum)

  const createRes = await fetch(`${baseUrl}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: continuationTitle }),
  })
  if (!createRes.ok) {
    const err = await createRes.text()
    return NextResponse.json({ success: false, error: `创建 continuation session 失败: ${err}` }, { status: 500 })
  }
  const continuationSessionData = await createRes.json()
  const continuationSessionId = continuationSessionData.id

  if (operatorName) {
    const resumePrompt = buildResumePrompt(operatorName, projectPath, sessionIdShort, handoffNum)
    const promptRes = await fetch(`${baseUrl}/session/${encodeURIComponent(continuationSessionId)}/prompt_async`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parts: [{ type: "text", text: resumePrompt }] }),
    })
    if (!promptRes.ok && promptRes.status !== 204) {
      const errBody = await promptRes.text()
      return NextResponse.json({ success: false, error: `发送 resume prompt 失败: ${errBody}` }, { status: 500 })
    }
  }

  let updatedRegistry = addOrUpdateSessionRecord(registry, {
    sessionId: continuationSessionId,
    from: handoffSessionId,
    to: [],
    birthHandoffDoc: handoffDocName,
    handoffDoc: null,
  })
  const handoffRec = findSessionRecord(updatedRegistry, handoffSessionId)
  if (handoffRec) {
    updatedRegistry = addOrUpdateSessionRecord(updatedRegistry, {
      ...handoffRec,
      to: [...handoffRec.to, continuationSessionId],
    })
  }
  writeHandoffRegistry(projectPath, updatedRegistry)

  try {
    await fetch(`${baseUrl}/session/${encodeURIComponent(handoffSessionId)}/abort`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
  } catch {}

  cleanupBatchFiles(projectPath, operatorName ?? "unknown", originalSessionId)

  try {
    await fetch(`${baseUrl}/tui/execute-command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: "/sessions" }) })
    await new Promise(r => setTimeout(r, 500))
    await fetch(`${baseUrl}/tui/execute-command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: `/sessions ${continuationSessionId}` }) })
  } catch {}

  return NextResponse.json({
    success: true,
    phase: "finished",
    continuationSession: { id: continuationSessionId, title: continuationTitle },
    handoffSessionId,
    operatorName,
   })
  } catch (err: any) {
    console.error("[complete] UNHANDLED ERROR:", err)
    return NextResponse.json({ success: false, error: `complete 内部错误: ${err.message}`, stack: err.stack?.split("\n").slice(0,5).join("\n") }, { status: 500 })
  }
}
