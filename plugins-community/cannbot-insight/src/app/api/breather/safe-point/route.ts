// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextResponse } from "next/server"

export async function GET(request: Request) {
  const url = new URL(request.url)
  const host = url.searchParams.get("host") ?? "localhost"
  const port = parseInt(url.searchParams.get("port") ?? "15031")
  const sessionId = url.searchParams.get("sessionId")
  const baseUrl = `http://${host}:${port}`

  if (!sessionId) {
    return NextResponse.json({ error: "缺少 sessionId" }, { status: 400 })
  }

  // We already got session.idle from SSE — the monitored session is idle.
  // Only need to check if child sessions (subagent turns) are still running.
  const childrenRes = await fetch(`${baseUrl}/session/${encodeURIComponent(sessionId)}/children`)
  if (!childrenRes.ok) {
    return NextResponse.json({ canHandoff: true, reason: "无法获取子 session，假定可以 handoff" })
  }
  const children = await childrenRes.json()

  // Check each child: if child has no assistant messages or messages are incomplete, it may be running
  const runningChildren: string[] = []
  for (const child of children) {
    const childRes = await fetch(`${baseUrl}/session/${encodeURIComponent(child.id)}`)
    if (!childRes.ok) continue
    const childData = await childRes.json()
    // If child session has cost > 0 but no completed assistant message, it might be running
    const msgsRes = await fetch(`${baseUrl}/session/${encodeURIComponent(child.id)}/message`)
    if (!msgsRes.ok) continue
    const msgs = await msgsRes.json()
    const lastAssistantMsg = msgs.filter((m: any) => m.info?.role === "assistant").pop()
    // Child is running if last assistant message has no "finish" field (still in progress)
    if (lastAssistantMsg && !lastAssistantMsg.info?.finish && lastAssistantMsg.info?.cost > 0) {
      runningChildren.push(child.id)
    }
  }

  const canHandoff = runningChildren.length === 0

  return NextResponse.json({
    canHandoff,
    reason: canHandoff
      ? "所有 subagent turn 已完成，可以 handoff"
      : `仍有 ${runningChildren.length} 个子 session 在运行`,
    runningChildren: runningChildren.length,
    totalChildren: children.length,
  })
}
