// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from "next/server"
import path from "node:path"
import { analyzeTrajectory } from "@/lib/ai/trajectory-analyzer"
import type { AIProviderConfig } from "@/lib/ai/analyzer"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { trajectoryPath, provider } = body as {
      trajectoryPath: string
      provider: AIProviderConfig
    }

    if (!trajectoryPath) {
      return NextResponse.json({ error: "Missing trajectoryPath" }, { status: 400 })
    }
    if (path.isAbsolute(trajectoryPath)) {
      return NextResponse.json({ error: "trajectoryPath must be relative" }, { status: 400 })
    }
    if (/\.\.\//.test(trajectoryPath)) {
      return NextResponse.json({ error: "trajectoryPath must not traverse parent" }, { status: 400 })
    }
    if (!provider?.baseUrl || !provider?.apiKey || !provider?.model) {
      return NextResponse.json(
        { error: "Missing provider config (baseUrl, apiKey, model)" },
        { status: 400 },
      )
    }

    const resolvedPath = path.resolve(process.cwd(), trajectoryPath)
    const result = await analyzeTrajectory({ trajectoryPath: resolvedPath, provider })

    return NextResponse.json({ outputPath: result.outputPath, rounds: result.rounds })
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
