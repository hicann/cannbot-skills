// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import {
  startSkillAudit,
  subscribeSkillAudit,
  getSkillAuditSnapshot,
  skillAuditKey,
  hydrateSkillAuditFromStorage,
  loadSkillAuditFromStorage,
  __resetSkillAuditForTests,
} from "@/lib/skill-audit-job"

const REPORT = { skill_name: "x", summary: { total: 2, pass: 1, fail: 1, na: 0 }, warnings: [] }

/** NDJSON 流：progress → result 事件（模拟 sift-runner 回传）。 */
function ndjsonResponse(events: Array<Record<string, unknown>>): Response {
  const body = events.map((e) => JSON.stringify(e)).join("\n") + "\n"
  return new Response(body, { status: 200, headers: { "Content-Type": "application/x-ndjson" } })
}

// vitest 是 node 环境（无 DOM）；store 的 sessionStorage 调用有 try/catch 安全，但测试要断言持久化，故 mock。
function makeSessionStorage() {
  const m = new Map<string, string>()
  return {
    getItem: (k: string) => (m.has(k) ? m.get(k)! : null),
    setItem: (k: string, v: string) => { m.set(k, String(v)) },
    removeItem: (k: string) => { m.delete(k) },
    clear: () => { m.clear() },
  }
}

describe("skill-audit-job (cross-tab resume)", () => {
  let fetchMock: ReturnType<typeof vi.fn>
  const originalFetch = globalThis.fetch

  beforeEach(() => {
    __resetSkillAuditForTests()
    vi.stubGlobal("sessionStorage", makeSessionStorage())
    fetchMock = vi.fn(async () =>
      ndjsonResponse([
        { stage: "progress", percent: 40, msg: "  [条件性] 批 1/2 判定 10 条…" },
        { stage: "result", percent: 100, report: REPORT, _html: "<html>STUB</html>" },
      ]),
    ) as unknown as ReturnType<typeof vi.fn>
    globalThis.fetch = fetchMock as unknown as typeof fetch
  })

  afterEach(() => {
    __resetSkillAuditForTests()
    vi.unstubAllGlobals()
    globalThis.fetch = originalFetch
  })

  it("start → running=true；resolve 后 result 就位、running=false", async () => {
    const key = skillAuditKey("ses_1", "skill", "foo")
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    expect(getSkillAuditSnapshot(key).running).toBe(true)
    expect(fetchMock).toHaveBeenCalledTimes(1)

    await vi.waitFor(() => {
      expect(getSkillAuditSnapshot(key).running).toBe(false)
    })
    const st = getSkillAuditSnapshot(key)
    expect(st.result).not.toBeNull()
    expect(st.result?.summary?.fail).toBe(1)
    expect(st.error).toBeNull()
  })

  it("agent kind → POST 到 audit-agentsift", () => {
    startSkillAudit({ taskId: "ses_1", kind: "agent", name: "developer" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const body = JSON.parse((fetchMock.mock.calls[0][1] as RequestInit).body as string)
    expect(body.agentName).toBe("developer")
    expect(body.skillName).toBeUndefined()
  })

  it("root kind → POST 到 audit-skillsift 且带 kind:root（合成主 agent 编排 目标，body 从 turn0 恢复）", () => {
    startSkillAudit({ taskId: "ses_1", kind: "root", name: "主 agent 编排" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit]
    expect(url).toBe("/api/ai/audit-skillsift")
    const body = JSON.parse(init.body as string)
    expect(body.skillName).toBe("主 agent 编排")
    expect(body.kind).toBe("root")
    expect(body.agentName).toBeUndefined()
  })

  it("去重：running 中再 start 不二次 fetch", () => {
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  it("finished 后可重跑（running=false 时 start 重发）", async () => {
    const key = skillAuditKey("ses_1", "skill", "foo")
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    await vi.waitFor(() => expect(getSkillAuditSnapshot(key).running).toBe(false))
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    expect(fetchMock).toHaveBeenCalledTimes(2)
  })

  it("跨 tab resume：start → unsubscribe（模拟 unmount）→ 状态保留 → 重订阅读到 live → resolve", async () => {
    const key = skillAuditKey("ses_1", "skill", "foo")
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    expect(getSkillAuditSnapshot(key).running).toBe(true)

    // 模拟切走：取消订阅（unmount），模块级 store 必须保留 running=true
    const unsub = subscribeSkillAudit(() => {})
    unsub()
    expect(getSkillAuditSnapshot(key).running).toBe(true)

    // 模拟切回：全新订阅者读 live 状态
    let remounted = getSkillAuditSnapshot(key)
    const sub2 = subscribeSkillAudit(() => {
      remounted = getSkillAuditSnapshot(key)
    })

    await vi.waitFor(() => expect(getSkillAuditSnapshot(key).running).toBe(false))
    expect(getSkillAuditSnapshot(key).result).not.toBeNull()
    expect(remounted.result).not.toBeNull()
    sub2()
  })

  it("resolve 后写 sessionStorage（reload 可恢复）", async () => {
    const key = skillAuditKey("ses_1", "skill", "foo")
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    await vi.waitFor(() => expect(getSkillAuditSnapshot(key).running).toBe(false))
    expect(loadSkillAuditFromStorage("ses_1", "skill", "foo")).not.toBeNull()
  })

  it("hydrate：sessionStorage 有结果 → 灌进 store（已有 live state 跳过）", async () => {
    // 预置一份 sessionStorage 结果
    sessionStorage.setItem(
      "skill-audit-ses_1-skill-foo",
      JSON.stringify({ summary: { total: 5, pass: 5, fail: 0, na: 0 } }),
    )
    const entries = [{ name: "foo", kind: "skill" as const }, { name: "bar", kind: "skill" as const }]
    hydrateSkillAuditFromStorage("ses_1", entries)
    const foo = getSkillAuditSnapshot(skillAuditKey("ses_1", "skill", "foo"))
    const bar = getSkillAuditSnapshot(skillAuditKey("ses_1", "skill", "bar"))
    expect(foo.result?.summary?.total).toBe(5)
    expect(bar.result).toBeNull() // sessionStorage 无 bar → 不动
  })

  it("error：fetch 非 2xx → error 设置、running=false", async () => {
    globalThis.fetch = vi.fn(async () =>
      new Response(JSON.stringify({ error: "boom" }), { status: 500, headers: { "Content-Type": "application/json" } }),
    ) as unknown as typeof fetch
    const key = skillAuditKey("ses_1", "skill", "foo")
    startSkillAudit({ taskId: "ses_1", kind: "skill", name: "foo" })
    await vi.waitFor(() => expect(getSkillAuditSnapshot(key).running).toBe(false))
    expect(getSkillAuditSnapshot(key).error).toMatch(/boom/)
    expect(getSkillAuditSnapshot(key).result).toBeNull()
  })
})
