// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import fs from "node:fs"
import path from "node:path"
import { prisma } from "@/lib/db"
import { exportSessionToMarkdown } from "@/lib/export/markdown-exporter"
import { buildAgentIO } from "@/lib/export/agent-io-export"

const PYTHON_AGENT_URL = process.env.CANNBOT_AGENT_URL ?? "http://localhost:21026"

export async function POST(req: Request) {
  const { taskId, framework, provider, mode } = await req.json()
  if (!taskId) return NextResponse.json({ error: "Missing taskId" }, { status: 400 })
  if (!provider?.apiKey || !provider?.baseUrl || !provider?.model) {
    return NextResponse.json({ error: "Missing provider config" }, { status: 400 })
  }

  // v2/v3 用 session-trajectory-analyse.md；v4 的 prompt 由 Python 自载 prompts/audit-v4-*.md
  const promptPath = path.resolve(process.cwd(), "prompts/session-trajectory-analyse.md")
  const promptMd = fs.existsSync(promptPath) ? fs.readFileSync(promptPath, "utf-8") : ""
  const outputDir = path.resolve(process.cwd(), "tmp")

  let trajectoryText: string
  try {
    trajectoryText = await exportSessionToMarkdown(taskId, prisma, framework)
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : "MD 组装失败" }, { status: 500 })
  }

  // v4 Step 1: deterministic agent-IO extraction (full fidelity, from Prisma)
  let agentIo: unknown = null
  if (mode === "v4") {
    try {
      agentIo = await buildAgentIO(taskId, prisma, framework)
    } catch (e) {
      return NextResponse.json({ error: e instanceof Error ? e.message : "agent-IO 提取失败" }, { status: 500 })
    }
  }

  let upstream: Response
  try {
    upstream = await fetch(`${PYTHON_AGENT_URL}/compress-and-analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        trajectoryText, provider, promptMd, outputDir,
        outputBasename: `session-${taskId}`,
        mode: mode ?? "agent",
        agentIo,
      }),
      signal: AbortSignal.timeout(1800000),
    })
  } catch (e) {
    const msg = e instanceof Error ? e.message : "agent v2 unreachable（Python 服务未启动？）"
    return NextResponse.json({ error: msg }, { status: 500 })
  }

  if (!upstream.ok || !upstream.body) {
    const err = await upstream.json().catch(() => ({ error: "agent v2 error" }))
    return NextResponse.json({ error: err.error ?? "agent v2 error" }, { status: 500 })
  }

  // 透传 Python 的 NDJSON 流
  return new Response(upstream.body, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache",
    },
  })
}
