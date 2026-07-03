// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"
import { OpenCodeClient } from "@/lib/breather/opencode-api"
import { getSessionContextHistory, getLatestSessionContext } from "@/lib/breather/context-monitor"
import { HandoffSessionRecord, HandoffLinks, readHandoffRegistry, buildHandoffLinks } from "@/lib/breather/handoff-registry"

export const dynamic = "force-dynamic"

export async function GET(request: Request) {
  try {
  const url = new URL(request.url)
  const host = url.searchParams.get("host") ?? "localhost"
  const port = parseInt(url.searchParams.get("port") ?? "15031")
  const sessionId = url.searchParams.get("sessionId")
  const baseUrl = `http://${host}:${port}`
  const client = new OpenCodeClient(baseUrl)

  const healthy = await client.health()
  if (!healthy) {
    return NextResponse.json({ error: `无法连接到 ${baseUrl}` }, { status: 503 })
  }

  if (!sessionId) {
    return NextResponse.json({ error: "缺少 sessionId" }, { status: 400 })
  }

  const sessions = await client.listSessions()
  const targetSession = sessions.find(s => s.id === sessionId)
  if (!targetSession) {
    return NextResponse.json({ error: "Session 不存在" }, { status: 404 })
  }

  const currentStatus = await getLatestSessionContext(client, sessionId)

  const history = currentStatus
    ? await getSessionContextHistory(client, sessionId)
    : []

  const projectPath = targetSession?.directory
  if (!projectPath) {
    return NextResponse.json({ error: `无法获取 session ${sessionId} 的 projectPath` }, { status: 500 })
  }
  const registry = readHandoffRegistry(projectPath)
  const titleMap = new Map<string, string | null>()
  for (const s of sessions) titleMap.set(s.id, s.title)
  const handoffLinks = buildHandoffLinks(registry, sessionId, titleMap)

  const childSessions = sessions.filter(s => s.parentID === sessionId).map(s => ({ id: s.id, title: s.title }))

  return NextResponse.json({
    status: currentStatus ? {
      ...currentStatus,
      sessionTitle: targetSession.title,
    } : null,
    history,
    handoffLinks,
    childSessionIds: childSessions.map(s => s.id),
    childSessions,
    projectPath,
  })
  } catch (err: any) {
    console.error("[status] UNHANDLED ERROR:", err)
    return NextResponse.json({ error: `status 内部错误: ${err.message}` }, { status: 500 })
  }
}
