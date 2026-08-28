// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest"
import { EventEmitter } from "node:events"
import fs from "node:fs"
import os from "node:os"
import path from "node:path"

/**
 * sift-runner：mock spawn，验证 stdout 进度解析（批 k/total + 完成 → 百分比）+ result/error 事件。
 */

const REPORT = { skill_name: "x", summary: { total: 2, pass: 1, fail: 1, na: 0 }, findings: [] }

// vi.mock 工厂会被提升到文件顶部，引用不到 it 内的局部变量；用 vi.hoisted 拿稳定引用。
const { spawnMock } = vi.hoisted(() => ({ spawnMock: vi.fn() }))
vi.mock("node:child_process", () => ({ spawn: spawnMock }))

function happyImpl(progressLines: string[]) {
  return (_cmd: string, args: string[]) => {
    const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
    proc.stdout = new EventEmitter()
    proc.stderr = new EventEmitter()
    proc.kill = vi.fn()
    setImmediate(() => {
      for (const line of progressLines) proc.stdout.emit("data", Buffer.from(line + "\n"))
      const i = args.indexOf("-o")
      const outDir = i >= 0 ? args[i + 1] : undefined
      if (outDir) {
        fs.mkdirSync(outDir, { recursive: true })
        fs.writeFileSync(path.join(outDir, "audit-report.json"), JSON.stringify(REPORT))
        fs.writeFileSync(path.join(outDir, "audit-report.html"), "<!doctype html><body>STUB</body>")
      }
      proc.emit("close", 0)
    })
    return proc
  }
}

let tmpOut = ""
beforeEach(() => {
  tmpOut = fs.mkdtempSync(path.join(os.tmpdir(), "runner-out-"))
  spawnMock.mockReset()
})
afterEach(() => {
  fs.rmSync(tmpOut, { recursive: true, force: true })
  vi.restoreAllMocks()
})

describe("runSiftAuditStreaming (via makeStreamingAuditResponse)", () => {
  it("progress 解析 + result 事件", async () => {
    spawnMock.mockImplementation(happyImpl([
      "对账 transcript 1/1: session.json",
      "  [精筛] 判定 5 条…",
      "  [精筛] 完成 ⏱ 10s",
      "  [条件性] 批 1/2 判定 10 条…",
      "  [条件性] 批 1/2 完成 ⏱ 60s",
      "  [禁令] 判定 8 条…",
      "  [禁令] 完成 ⏱ 50s",
      "  [步骤] 判定 12 条…",
      "  [步骤] 完成 ⏱ 40s",
      "  [条件性] 批 2/2 判定 5 条…",
      "  [条件性] 批 2/2 完成 ⏱ 30s",
    ]))
    const { makeStreamingAuditResponse } = await import("@/lib/sift-runner")

    const res = makeStreamingAuditResponse(["audit", "/tmp/skill", "-o", tmpOut], tmpOut)
    const text = await res.text()
    const events = text.trim().split("\n").map((l) => JSON.parse(l))
    const stages = events.map((e: { stage: string }) => e.stage)
    expect(stages).toContain("progress")
    expect(stages.at(-1)).toBe("result")
    const result = events.at(-1) as { stage: string; report: unknown; _html: string; percent: number }
    expect(result.report).toMatchObject({ skill_name: "x" })
    expect(result._html).toContain("STUB")
    expect(result.percent).toBe(100)

    // 百分比单调非减
    const percents = events.filter((e: { stage: string }) => e.stage === "progress").map((e: { percent: number }) => e.percent)
    for (let i = 1; i < percents.length; i++) {
      expect(percents[i]).toBeGreaterThanOrEqual(percents[i - 1])
    }
    expect(percents.at(-1)).toBeGreaterThan(0)
  })

  it("ENOENT → error 事件（sift 未找到）", async () => {
    spawnMock.mockImplementation(() => {
      const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
      proc.stdout = new EventEmitter()
      proc.stderr = new EventEmitter()
      proc.kill = vi.fn()
      setImmediate(() => proc.emit("error", Object.assign(new Error("enoent"), { code: "ENOENT" })))
      return proc
    })
    const { makeStreamingAuditResponse } = await import("@/lib/sift-runner")

    const res = makeStreamingAuditResponse(["audit", "/tmp/skill", "-o", tmpOut], tmpOut)
    const text = await res.text()
    const events = text.trim().split("\n").map((l) => JSON.parse(l))
    expect(events.at(-1)).toMatchObject({ stage: "error" })
    expect((events.at(-1) as { msg: string }).msg).toMatch(/sift 未找到/)
  })

  it("非 0 退出 → error 事件带 exit code", async () => {
    spawnMock.mockImplementation(() => {
      const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
      proc.stdout = new EventEmitter()
      proc.stderr = new EventEmitter()
      proc.kill = vi.fn()
      setImmediate(() => {
        proc.stderr.emit("data", Buffer.from("切段为空"))
        proc.emit("close", 2)
      })
      return proc
    })
    const { makeStreamingAuditResponse } = await import("@/lib/sift-runner")

    const res = makeStreamingAuditResponse(["audit", "/tmp/skill", "-o", tmpOut], tmpOut)
    const text = await res.text()
    const events = text.trim().split("\n").map((l) => JSON.parse(l))
    expect(events.at(-1)).toMatchObject({ stage: "error" })
    expect((events.at(-1) as { msg: string }).msg).toMatch(/exit 2/)
  })

  it("退出 0 但无 audit-report.json → error", async () => {
    spawnMock.mockImplementation(() => {
      const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
      proc.stdout = new EventEmitter()
      proc.stderr = new EventEmitter()
      proc.kill = vi.fn()
      setImmediate(() => proc.emit("close", 0))
      return proc
    })
    const { makeStreamingAuditResponse } = await import("@/lib/sift-runner")

    const res = makeStreamingAuditResponse(["audit", "/tmp/skill", "-o", tmpOut], tmpOut)
    const text = await res.text()
    const events = text.trim().split("\n").map((l) => JSON.parse(l))
    expect((events.at(-1) as { stage: string }).stage).toBe("error")
  })

  it("cleanup 在流结束后调用", async () => {
    spawnMock.mockImplementation(happyImpl([]))
    const { makeStreamingAuditResponse } = await import("@/lib/sift-runner")

    const cleanup = vi.fn()
    const res = makeStreamingAuditResponse(["audit", "/tmp/skill", "-o", tmpOut], tmpOut, cleanup)
    await res.text()
    expect(cleanup).toHaveBeenCalledTimes(1)
  })
})
