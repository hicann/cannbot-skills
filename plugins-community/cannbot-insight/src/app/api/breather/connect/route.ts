// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import { OpenCodeClient } from "@/lib/breather/opencode-api"

export async function POST(request: Request) {
  const body = await request.json()
  const { host = "localhost", port = 15031 } = body

  const baseUrl = `http://${host}:${port}`
  const client = new OpenCodeClient(baseUrl)

  const healthy = await client.health()

  if (!healthy) {
    return NextResponse.json(
      {
        connected: false,
        error: `无法连接到 ${baseUrl}。请确认 OpenCode 已使用 --port ${port} 启动。`,
        hint: `opencode --port ${port}`,
      },
      { status: 200 }
    )
  }

  const allSessions = await client.listSessions()
  const rootSessions = allSessions
    .filter(s => !s.parentID)
    .sort((a, b) => (b.time?.updated ?? 0) - (a.time?.updated ?? 0))

  return NextResponse.json({
    connected: true,
    baseUrl,
    sessions: rootSessions.map((s: any) => ({
      id: s.id,
      title: s.title,
      createdAt: new Date(s.time?.created ?? 0).toISOString(),
      directory: s.location?.directory ?? null,
      model: s.model?.id ?? null,
      agent: s.agent ?? null,
      tokensInput: s.tokens?.input ?? 0,
    })),
  })
}
