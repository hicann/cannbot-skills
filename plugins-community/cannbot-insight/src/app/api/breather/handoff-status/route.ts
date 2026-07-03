// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import { OpenCodeClient } from "@/lib/breather/opencode-api"
import { readHandoffRegistry, writeHandoffRegistry, updateSessionStatus } from "@/lib/breather/handoff-registry"

export async function PATCH(request: Request) {
  try {
    const body = await request.json()
    const { sessionId, status, host = "localhost", port = 15031 } = body

    if (!sessionId || !status) {
      return NextResponse.json({ error: "缺少 sessionId 或 status" }, { status: 400 })
    }

    const baseUrl = `http://${host}:${port}`
    const client = new OpenCodeClient(baseUrl)

    const session = await client.getSession(sessionId)
    const projectPath = session?.directory
    if (!projectPath) {
      return NextResponse.json({ error: `无法获取 session ${sessionId} 的 projectPath` }, { status: 500 })
    }

    const registry = readHandoffRegistry(projectPath)
    const updatedRegistry = updateSessionStatus(registry, sessionId, status)
    writeHandoffRegistry(projectPath, updatedRegistry)

    return NextResponse.json({ success: true, sessionId, status })
  } catch (err: any) {
    console.error("[handoff-status] ERROR:", err)
    return NextResponse.json({ error: err.message }, { status: 500 })
  }
}
