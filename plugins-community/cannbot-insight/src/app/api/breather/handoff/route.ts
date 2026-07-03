// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import { OpenCodeClient, V1Message } from "@/lib/breather/opencode-api"
import { HandoffSessionRecord, readHandoffRegistry, writeHandoffRegistry, extractOperatorNameFromPath, computeSessionIdShort, computeHandoffNum, computeHandoffDocName, computeHandoffSessionTitle, findBirthHandoffDoc, findSessionRecord, addOrUpdateSessionRecord, addChildToSession } from "@/lib/breather/handoff-registry"
import fs from "node:fs"
import path from "node:path"

function formatMessageText(msg: V1Message): string {
  const role = msg.info.role
  const parts: string[] = []
  for (const part of msg.parts) {
    if (part.type === "text" && part.text) {
      parts.push(part.text as string)
    }
  }
  if (parts.length === 0) return ""
  return `[${role}]: ${parts.join("\n")}`
}

function splitMessagesIntoBatches(messages: V1Message[], maxCharsPerBatch: number): string[] {
  const formatted: string[] = []
  for (const msg of messages) {
    const text = formatMessageText(msg)
    if (text) formatted.push(text)
  }

  const batches: string[] = []
  let currentBatch = ""
  for (const text of formatted) {
    if (currentBatch.length + text.length > maxCharsPerBatch && currentBatch.length > 0) {
      batches.push(currentBatch)
      currentBatch = text
    } else {
      currentBatch += (currentBatch ? "\n\n---\n\n" : "") + text
    }
  }
  if (currentBatch) batches.push(currentBatch)
  return batches
}

function writeBatchFiles(batches: string[], sessionId: string, projectPath: string, operatorName: string): string[] {
  const batchDir = operatorName
    ? path.join(projectPath, "operators", operatorName, "docs", ".handoff-batches")
    : path.join(projectPath, ".handoff-batches")
  if (!fs.existsSync(batchDir)) fs.mkdirSync(batchDir, { recursive: true })

  const batchFiles: string[] = []
  for (let i = 0; i < batches.length; i++) {
    const filePath = path.join(batchDir, `batch-${sessionId.slice(0, 12)}-${i + 1}.txt`)
    fs.writeFileSync(filePath, batches[i], "utf-8")
    batchFiles.push(filePath)
  }
  return batchFiles
}

function buildHandoffSessionPrompt(
  oldSessionId: string,
  sessionIdShort: string,
  operatorName: string,
  projectPath: string,
  contextPct: number,
  batchFiles: string[],
  handoffNum: number,
  birthHandoffDoc: string | null
): string {
  const batchList = batchFiles.map((f, i) => `  - 批次 ${i + 1}: ${f}`).join("\n")
  const logPath = path.join(projectPath, "operators", operatorName, "docs", "LOG.md")
  const handoffToolsDir = path.join(projectPath, "scripts", "handoff-tools")
  const opDir = path.join(projectPath, "operators", operatorName)
  const opDocsDir = path.join(projectPath, "operators", operatorName, "docs")
  const handoffMdPath = path.join(opDocsDir, `SESSION-HANDOFF-${sessionIdShort}-${handoffNum}.md`)
  const handoffToolsParseLog = path.join(handoffToolsDir, "parse_log.py")
  const handoffToolsCheckDeliverables = path.join(handoffToolsDir, "check_deliverables.py")

  const dataFlowPaths = [
    path.join(projectPath, "operators", operatorName, "docs", "data-flow.md"),
    path.join(projectPath, "workflow", "resources", "data-flow.md"),
    path.join(projectPath, "plugins-official", "ops-registry-invoke", "workflow", "resources", "data-flow.md"),
  ]
  const dataFlowPath = dataFlowPaths.find(p => fs.existsSync(p)) ?? dataFlowPaths[0]

  return `你是一个 SESSION-HANDOFF 文档组装器。请严格按以下步骤工作：

## Step 0 — 确定性内容（用 Bash 运行脚本 + Read/Glob 工具）

1. 运行脚本获取 LOG.md 解析结果:
   Bash: python3 ${handoffToolsParseLog} ${logPath}

2. 运行脚本获取交付件检查结果:
   Bash: python3 ${handoffToolsCheckDeliverables} ${opDir} ${operatorName}

 3. Read ${dataFlowPath} 认下一步的输入文件路径（如果找不到 data-flow.md，从脚本输出的 status_rows 和 SUBAGENT_MAP 推导）

4. 根据脚本输出，写出以下确定性内容的文档文字:
   - "## 1. 阻塞项": 从脚本输出的 blocking_items 和交付件检查结果判断
   - "## 2. 如何接续": 从脚本输出的 next_step, next_subagent, next_scene 写出下一步触发指令 + 输入文件
   - "## 3. 交付件浅层检查": 直接使用 check_deliverables.py 的输出组装表格

  5. 检查最后一个批次文件中旧 session 最后一条 assistant 消息，提取其中等待用户回答的问题或需要用户确认的决策。把这些待确认问题写入第 1 节阻塞项（标注"⚠️ 需用户确认"），并在第 2 节如何接续中注明"接续 session 应先向用户提出这些问题并获得回答后再继续执行"。

  6. 增量继承: ${birthHandoffDoc ? `你正在为其写交接文档的前序 session 是从 handoff 文档 ${birthHandoffDoc} 创建的 continuation session。先用 Read 读取 operators/${operatorName}/docs/${birthHandoffDoc} 的第 4 节（核心发现/关键决策/踩过的坑），把所有已有条目**继承到本次文档的第 4 节**（只增不删）。新提取的内容追加到已有条目后面。其余节（1、2、3、5）用本次最新数据。` : "这是第一次 handoff，无需继承历史文档。"}

## Step 1 — 非确定性内容（起并行 subagent 提取）

前序 session (ID: ${oldSessionId}) 的 user + assistant 消息已分批写入以下文件:
${batchList}

起 ${batchFiles.length} 个 subagent 并行提取，每个 subagent 只读一个批次文件:
每个 subagent 的 prompt 应为: "请从以下对话片段中提取三类信息及上下文警告: 1.核心发现 2.关键决策 3.踩过的坑 4.影响下一步执行的具体警告或争议。每条用一句话概括，如果某类不存在输出'无'。对话内容: [Read 对应的批次文件]"

收到所有 subagent 返回结果后，合并去重:
- 同一发现/决策/踩坑被多个 subagent 提取时只保留一条
- 上下文警告合并后写入第 2 节「如何接续」的注意事项部分
- 如果某个 subagent 失败，只合并成功的结果; 全部失败则第 4 节和注意事项留空

## Step 2 — 写完自检

把自己当成那个**零记忆的新 session**，通读刚写完的文档:

1. 对照磁盘交付件验证: 阻塞项中说的"XX 文件不存在"是否真的不存在? 交付件表格中的存在性/大小/stub 是否与磁盘实际一致?
2. 对照脚本输出验证: 当前阶段和下一步是否与 parse_log.py 结果一致?
3. "如何接续"能否无歧义照做? 触发哪个 skill、说什么话、输入文件路径，写清楚了吗? 新 session 照这节一步就能开始干活吗?
4. 已完成的工作会不会被误重跑? 哪些步骤已 ✅、哪些交付件已存在可用——"不必重跑"列清楚了吗?
5. 自包含检查: 有没有依赖"我刚才看到的对话"才能看懂的句子? 任何脱离上下文就含义不清的句子必须补写背景。

任一处含糊就补。

## 最终输出

自检确认无误后，Write 最终文档到: ${handoffMdPath}

5 节结构:
1. 阻塞项
2. 如何接续（下一步 subagent/scene/输入文件 + 注意事项）
3. 交付件浅层检查（表格: 文件|路径|存在|大小|stub信号）
4. 核心发现 / 关键决策 / 踩过的坑（可选，来自 subagent 提取）
5. 补充信息（session ID、handoff 序号、上下文占用、交接时间）

## 补充信息节数据

- OpenCode session ID: ${oldSessionId}
- Handoff 序号: h${handoffNum}
- 上下文占用: ${contextPct.toFixed(1)}%
- 交接时间: ${new Date().toISOString()}
- 前序 session ID: ${oldSessionId}

## 完成信号

自检确认无误、文档已 Write 成功后，最后一步必须 Write 一个完成信号文件到: ${handoffMdPath.replace(/\.md$/, '.done')}，内容为 "done"。这个信号文件告诉监控系统你的整个工作流程（文档组装 + 自检 + 写入）已全部完成。没有这个信号文件，系统会认为你只是暂时中断而不会创建 continuation session。
`
}

export async function POST(request: Request) {
  try {
  const body = await request.json()
  const { sessionId, host = "localhost", port = 15031, contextPct = 0 } = body

  if (!sessionId) {
    return NextResponse.json({ error: "缺少 sessionId" }, { status: 400 })
  }

  const baseUrl = `http://${host}:${port}`
  const client = new OpenCodeClient(baseUrl)

  const healthy = await client.health()
  if (!healthy) {
    return NextResponse.json({ error: `无法连接到 OpenCode (${baseUrl})` }, { status: 503 })
  }

  const sessions = await client.listSessions()
  const targetSession = sessions.find(s => s.id === sessionId)
  const projectPath = targetSession?.directory
  if (!projectPath) {
    return NextResponse.json({ error: `无法获取 session ${sessionId} 的 projectPath` }, { status: 500 })
  }
  const originalSession = await client.getSession(sessionId)
  const originalTitle = originalSession?.title ?? "session"

  const registry = readHandoffRegistry(projectPath)

  const currentRecord = findSessionRecord(registry, sessionId)
  if (currentRecord) {
    for (const childId of currentRecord.to) {
      const childRecord = findSessionRecord(registry, childId)
      if (childRecord && !childRecord.birthHandoffDoc && childRecord.status === "running") {
        return NextResponse.json({ success: false, error: `session ${sessionId} 已有活跃的 handoff session (${childId.slice(0, 12)})`, existingHandoffId: childId }, { status: 409 })
      }
    }
  }

  const operatorName = extractOperatorNameFromPath(projectPath)
  const sessionIdShort = computeSessionIdShort(sessionId)
  const handoffNum = computeHandoffNum(registry, sessionId)
  const handoffDocName = computeHandoffDocName(sessionIdShort, handoffNum)
  const birthHandoffDoc = findBirthHandoffDoc(registry, sessionId)

  const scriptsDir = path.join(projectPath, "scripts", "handoff-tools")
  if (!fs.existsSync(scriptsDir)) {
    fs.mkdirSync(scriptsDir, { recursive: true })
  }
  const srcDir = path.join(process.cwd(), "scripts", "handoff-tools")
  for (const script of ["parse_log.py", "check_deliverables.py"]) {
    const src = path.join(srcDir, script)
    const dst = path.join(scriptsDir, script)
    if (fs.existsSync(src) && !fs.existsSync(dst)) {
      fs.copyFileSync(src, dst)
    }
  }

  let abortAttempted = false

  const messages = await client.getMessages(sessionId)
  const maxCharsPerBatch = 30000
  const batches = splitMessagesIntoBatches(messages, maxCharsPerBatch)
  const batchFiles = writeBatchFiles(batches, sessionId, projectPath, operatorName ?? "unknown")

  const handoffSessionTitle = computeHandoffSessionTitle(sessionIdShort, handoffNum)

  const createRes = await fetch(`${baseUrl}/session`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title: handoffSessionTitle }),
  })
  if (!createRes.ok) {
    const err = await createRes.text()
    return NextResponse.json({ success: false, error: `创建 handoff session 失败: ${err}` }, { status: 500 })
  }
  const handoffSessionData = await createRes.json()
  const handoffSessionId = handoffSessionData.id
  if (operatorName) {
    const handoffPrompt = buildHandoffSessionPrompt(
      sessionId,
      sessionIdShort,
      operatorName,
      projectPath,
      contextPct,
      batchFiles,
      handoffNum,
      birthHandoffDoc
    )
    const promptRes = await fetch(`${baseUrl}/session/${encodeURIComponent(handoffSessionId)}/prompt_async`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ parts: [{ type: "text", text: handoffPrompt }] }),
    })
    if (!promptRes.ok && promptRes.status !== 204) {
      const promptBody = await promptRes.text()
      return NextResponse.json({ success: false, error: `发送 handoff prompt 失败 (status=${promptRes.status})` }, { status: 500 })
    }
  } else {
  }

  try {
    const abortRes = await fetch(`${baseUrl}/session/${encodeURIComponent(sessionId)}/abort`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
    abortAttempted = abortRes.ok || abortRes.status === 200
  } catch {}

  let updatedRegistry = addOrUpdateSessionRecord(registry, {
    sessionId: handoffSessionId,
    from: sessionId,
    to: [],
    birthHandoffDoc: null,
    handoffDoc: handoffDocName,
    status: "running",
  })
  updatedRegistry = addOrUpdateSessionRecord(updatedRegistry, {
    sessionId,
    from: findSessionRecord(updatedRegistry, sessionId)?.from ?? null,
    to: [...(findSessionRecord(updatedRegistry, sessionId)?.to ?? []), handoffSessionId],
    birthHandoffDoc: findSessionRecord(updatedRegistry, sessionId)?.birthHandoffDoc ?? null,
    handoffDoc: findSessionRecord(updatedRegistry, sessionId)?.handoffDoc ?? null,
  })
  if (!findSessionRecord(updatedRegistry, sessionId)) {
    updatedRegistry = addOrUpdateSessionRecord(updatedRegistry, {
      sessionId,
      from: null,
      to: [handoffSessionId],
      birthHandoffDoc: null,
      handoffDoc: null,
    })
  }
  writeHandoffRegistry(projectPath, updatedRegistry)

  try {
    await fetch(`${baseUrl}/tui/execute-command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: "/sessions" }) })
    await new Promise(r => setTimeout(r, 500))
    await fetch(`${baseUrl}/tui/execute-command`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ command: `/sessions ${handoffSessionId}` }) })
  } catch {}

  return NextResponse.json({
    success: true,
    phase: "executing",
    abortAttempted,
    handoffSession: { id: handoffSessionId, title: handoffSessionTitle },
    operatorName,
    batchCount: batches.length,
    batchFiles,
    handoffNum,
    projectPath,
    originalSessionId: sessionId,
    originalTitle,
   })
  } catch (err: any) {
    console.error("[handoff] UNHANDLED ERROR:", err)
    return NextResponse.json({ success: false, error: `handoff 内部错误: ${err.message}`, stack: err.stack?.split("\n").slice(0,5).join("\n") }, { status: 500 })
  }
}
