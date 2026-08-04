// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from "next/server"
import { prisma } from "@/lib/db"
import { analyzeTrajectoryByTask, type ProgressEvent } from "@/lib/ai/trajectory-analyzer"
import type { AIProviderConfig } from "@/lib/ai/analyzer"

export async function POST(request: NextRequest) {
  let body: { taskId?: string; framework?: string; provider?: AIProviderConfig }
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 })
  }

  const { taskId, framework, provider } = body
  if (!taskId) {
    return NextResponse.json({ error: "Missing taskId" }, { status: 400 })
  }
  if (!provider?.baseUrl || !provider?.apiKey || !provider?.model) {
    return NextResponse.json(
      { error: "Missing provider config (baseUrl, apiKey, model)" },
      { status: 400 },
    )
  }

  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: unknown) => {
        controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"))
      }
      try {
        const result = await analyzeTrajectoryByTask({
          taskId,
          framework,
          prisma,
          provider,
          onProgress: (e: ProgressEvent) => send(e),
        })
        send({ stage: "result", outputPath: result.outputPath, rounds: result.rounds, analysis: result.analysis })
      } catch (error) {
        const message = error instanceof Error ? error.message : "Unknown error"
        send({ stage: "error", msg: message })
      } finally {
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache",
      "Connection": "keep-alive",
    },
  })
}
