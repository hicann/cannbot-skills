// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import {
  startAuditJob,
  subscribeAuditJob,
  getAuditJobSnapshot,
  clearAuditJob,
  __resetAuditJobsForTests,
} from "@/lib/audit-job"

function ndjsonStream(chunks: string[]): ReadableStream<Uint8Array> {
  return new ReadableStream<Uint8Array>({
    start(controller) {
      for (const c of chunks) controller.enqueue(new TextEncoder().encode(c))
      controller.close()
    },
  })
}

function mockFetchStream(chunks: string[]): ReturnType<typeof vi.fn> {
  return vi.fn(async () =>
    new Response(ndjsonStream(chunks), {
      status: 200,
      headers: { "Content-Type": "application/x-ndjson" },
    }),
  ) as unknown as ReturnType<typeof vi.fn>
}

const provider = { apiKey: "k", baseUrl: "http://x", model: "m" }
const analysis = { flow: [], sessionSummary: "s", sessionMeta: {}, workflowLevelIssues: [], optimizationPriorities: [] }

describe("audit-job (survives tab unmount/remount)", () => {
  let fetchMock: ReturnType<typeof vi.fn>
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    __resetAuditJobsForTests()
    fetchMock = mockFetchStream([
      JSON.stringify({ stage: "ping", msg: "thinking", round: 1 }) + "\n",
      JSON.stringify({ stage: "step", msg: "parsing turns" }) + "\n",
      JSON.stringify({ stage: "result", analysis }) + "\n",
    ])
    globalThis.fetch = fetchMock as unknown as typeof fetch
  })

  afterEach(() => {
    __resetAuditJobsForTests()
    globalThis.fetch = originalFetch
  })

  it("does not call fetch when provider is missing; surfaces genError in job state", () => {
    startAuditJob({
      taskId: "t-noconf",
      agentVersion: "v1",
      provider: { apiKey: "", baseUrl: "", model: "" },
    })
    expect(fetchMock).not.toHaveBeenCalled()
    const st = getAuditJobSnapshot("t-noconf")
    expect(st.generating).toBe(false)
    expect(st.genError).toMatch(/请先配置/)
  })

  it("keeps generating state across unmount/remount (simulate subscribe drop + re-subscribe)", async () => {
    const onResult = vi.fn()
    const taskId = "t-survive"

    // First mount: start job + subscribe.
    startAuditJob({ taskId, agentVersion: "v1", provider, onResult })
    let seen = getAuditJobSnapshot(taskId)
    expect(seen.generating).toBe(true)

    // Let the stream produce at least the first progress events but NOT finish.
    await Promise.resolve()
    await Promise.resolve()
    seen = getAuditJobSnapshot(taskId)
    expect(seen.generating).toBe(true)
    expect(seen.progress.length).toBeGreaterThan(0)

    // Simulate navigating away: unsubscribe (component unmount) but module state must persist.
    const unsub = subscribeAuditJob(() => {})
    unsub()

    const afterUnmount = getAuditJobSnapshot(taskId)
    expect(afterUnmount.generating).toBe(true)
    expect(afterUnmount.progress.length).toBeGreaterThan(0)

    // Simulate coming back: a brand-new subscriber reads live module state.
    let remounted: typeof seen = getAuditJobSnapshot(taskId)
    const sub2 = subscribeAuditJob(() => {
      remounted = getAuditJobSnapshot(taskId)
    })

    // Drain the stream to completion.
    await vi.waitFor(() => {
      expect(getAuditJobSnapshot(taskId).generating).toBe(false)
    })

    const final = getAuditJobSnapshot(taskId)
    expect(final.generating).toBe(false)
    expect(final.result).not.toBeNull()
    expect(onResult).toHaveBeenCalledTimes(1)
    expect(remounted.result).not.toBeNull()
    sub2()
  })

  it("is idempotent: a second start while generating does not fire a second fetch", () => {
    startAuditJob({ taskId: "t-idem", agentVersion: "v1", provider })
    startAuditJob({ taskId: "t-idem", agentVersion: "v1", provider })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    clearAuditJob("t-idem")
  })
})
