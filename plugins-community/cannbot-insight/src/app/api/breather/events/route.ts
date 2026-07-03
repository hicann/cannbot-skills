// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export async function GET(request: Request) {
  const url = new URL(request.url)
  const host = url.searchParams.get("host") ?? "localhost"
  const port = parseInt(url.searchParams.get("port") ?? "15031")
  const opencodeUrl = `http://${host}:${port}/global/event`

  const upstream = await fetch(opencodeUrl, {
    headers: { Accept: "text/event-stream" },
  })

  if (!upstream.ok) {
    return new Response(JSON.stringify({ error: "无法连接到 OpenCode SSE" }), {
      status: upstream.status,
      headers: { "Content-Type": "application/json" },
    })
  }

  const encoder = new TextEncoder()
  const stream = new ReadableStream({
    async start(controller) {
      const reader = upstream.body?.getReader()
      if (!reader) {
        controller.close()
        return
      }

      try {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          controller.enqueue(value ?? new Uint8Array())
        }
      } catch {
      } finally {
        reader.releaseLock()
        controller.close()
      }
    },
  })

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
    },
  })
}
