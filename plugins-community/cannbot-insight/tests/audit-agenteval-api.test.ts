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
 * audit-agenteval 路由测试：agent .md 从 AGENTS_SCAN_ROOT 扫描恢复（不在 session），
 * session 侧走结构化 records（--transcript）。mock spawn + prisma + 真实 tmp 扫描根。
 */

const FAKE_REPORT = {
  skill_name: "developer",
  transcript_sources: ["structured"],
  summary: { total: 2, pass: 1, fail: 1, na: 0 },
  findings: [
    { instruction_text: "do x", source: "STEP", verdict: "PASS", method: "LLM", category: "step", evidence: "ok", seq: 0 },
  ],
}

let scanRoot = ""
let origScanRoot: string | undefined

const { spawnMock } = vi.hoisted(() => ({ spawnMock: vi.fn() }))
vi.mock("node:child_process", () => ({ spawn: spawnMock }))

// resolveScanRoot 默认委托真实实现（测试 env 下自动探测到 skills-dev 根）；
// 503 用例单独 mockReturnValueOnce(null) 强制走"无扫描根"路径。
vi.mock("@/lib/agent-md-scan", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/agent-md-scan")>()
  return { ...actual, resolveScanRoot: vi.fn(actual.resolveScanRoot) }
})

const fakePrisma = {
  session: { findFirst: vi.fn() },
  interactionBridge: { findMany: vi.fn() },
  toolCall: { findMany: vi.fn() },
  turn: { findMany: vi.fn() },
}
vi.mock("@/lib/db", () => ({ prisma: fakePrisma }))

const { POST } = await import("@/app/api/ai/audit-agenteval/route")
const { resolveScanRoot } = await import("@/lib/agent-md-scan")

function makeRequest(body: unknown): Request {
  return new Request("http://localhost/api/ai/audit-agenteval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

/** spawn mock：往 -o 目录写假 audit-report.json/.html + emit close 0。 */
function happySpawn() {
  return vi.fn((_cmd: string, args: string[]) => {
    const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
    proc.stdout = new EventEmitter()
    proc.stderr = new EventEmitter()
    proc.kill = vi.fn()
    setImmediate(() => {
      const i = args.indexOf("-o")
      const outDir = i >= 0 ? args[i + 1] : undefined
      if (outDir) {
        fs.mkdirSync(outDir, { recursive: true })
        fs.writeFileSync(path.join(outDir, "audit-report.json"), JSON.stringify(FAKE_REPORT))
        fs.writeFileSync(path.join(outDir, "audit-report.html"), "<!doctype html><body>STUB</body>")
      }
      proc.emit("close", 0)
    })
    return proc
  })
}

async function readNdjson(res: Response): Promise<unknown[]> {
  const text = await res.text()
  return text.trim().split("\n").filter(Boolean).map((l) => JSON.parse(l))
}

beforeEach(() => {
  spawnMock.mockReset()
  spawnMock.mockImplementation(happySpawn())
  fakePrisma.session.findFirst.mockReset()
  fakePrisma.interactionBridge.findMany.mockReset()
  fakePrisma.toolCall.findMany.mockReset()
  fakePrisma.turn.findMany.mockReset()

  scanRoot = fs.mkdtempSync(path.join(os.tmpdir(), "agents-scanroot-"))
  fs.mkdirSync(path.join(scanRoot, "plugins-official", "glacier", "agents"), { recursive: true })
  fs.writeFileSync(
    path.join(scanRoot, "plugins-official", "glacier", "agents", "developer.md"),
    "---\nname: developer\ndescription: dev agent\n---\n\n# developer\n\n你是开发者。\n",
  )
  origScanRoot = process.env.AGENTS_SCAN_ROOT
  process.env.AGENTS_SCAN_ROOT = scanRoot
})

afterEach(() => {
  if (origScanRoot === undefined) delete process.env.AGENTS_SCAN_ROOT
  else process.env.AGENTS_SCAN_ROOT = origScanRoot
  fs.rmSync(scanRoot, { recursive: true, force: true })
})

describe("audit-agenteval route", () => {
  it("happy：扫 .md → 结构化 records → spawn skill-eval --kind agent → NDJSON result", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "cannbot-insight" })
    fakePrisma.interactionBridge.findMany.mockResolvedValue([
      { subagentName: "developer", subagentType: null },
      { subagentName: "verifier", subagentType: null },
    ])
    fakePrisma.turn.findMany.mockResolvedValue([
      { id: "t1", turnIndex: 0, role: "assistant", content: "干活", agentName: "developer", isSubagent: true, subagentSessionId: "sub-1", parentExecutionId: null },
    ])
    fakePrisma.toolCall.findMany.mockResolvedValue([])

    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "developer" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; report: { summary: { fail: number } }; _html: string }
    expect(last.stage).toBe("result")
    expect(last.report.summary.fail).toBe(1)
    expect(last._html).toContain("STUB")

    const [, args] = spawnMock.mock.calls[0]
    expect(args[0]).toBe("audit")
    expect(args).toContain("--transcript")
    expect(args).not.toContain("--db")
    expect(args).toContain("--kind")
    expect(args).toContain("agent")
  })

  it("缺 taskId/agentName → 400", async () => {
    const r1 = await POST(makeRequest({ agentName: "developer" }))
    expect(r1.status).toBe(400)
    const r2 = await POST(makeRequest({ taskId: "ses_1" }))
    expect(r2.status).toBe(400)
  })

  it("agentName 含路径穿越 → 400（不扫盘）", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", framework: "cannbot-insight" })
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "../etc/passwd" }))
    expect(res.status).toBe(400)
    expect(spawnMock).not.toHaveBeenCalled()
  })

  it("AGENTS_SCAN_ROOT 未配且自动探测失败 → 503", async () => {
    vi.mocked(resolveScanRoot).mockReturnValueOnce(null)
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", framework: "cannbot-insight" })
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "developer" }))
    expect(res.status).toBe(503)
    const json = await res.json()
    expect(String(json.error)).toMatch(/AGENTS_SCAN_ROOT/i)
  })

  it("agent .md 不在扫描根 → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", framework: "cannbot-insight" })
    fakePrisma.interactionBridge.findMany.mockResolvedValue([])
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "ghost-agent" }))
    expect(res.status).toBe(404)
  })

  it("session 不存在 → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue(null)
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "developer" }))
    expect(res.status).toBe(404)
  })

  it("skill-eval 不在 PATH（spawn ENOENT）→ NDJSON error 事件", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", framework: "cannbot-insight" })
    fakePrisma.interactionBridge.findMany.mockResolvedValue([{ subagentName: "developer", subagentType: null }])
    fakePrisma.turn.findMany.mockResolvedValue([])
    fakePrisma.toolCall.findMany.mockResolvedValue([])
    spawnMock.mockImplementation(() => {
      const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
      proc.stdout = new EventEmitter()
      proc.stderr = new EventEmitter()
      proc.kill = vi.fn()
      setImmediate(() => proc.emit("error", Object.assign(new Error("enoent"), { code: "ENOENT" })))
      return proc
    })
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "developer" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; msg: string }
    expect(last.stage).toBe("error")
    expect(last.msg).toMatch(/skill-eval 未找到/)
  })

  it("skill-eval 非零退出（exit 2）→ NDJSON error 事件", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", framework: "cannbot-insight" })
    fakePrisma.interactionBridge.findMany.mockResolvedValue([{ subagentName: "developer", subagentType: null }])
    fakePrisma.turn.findMany.mockResolvedValue([])
    fakePrisma.toolCall.findMany.mockResolvedValue([])
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
    const res = await POST(makeRequest({ taskId: "ses_1", agentName: "developer" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; msg: string }
    expect(last.stage).toBe("error")
    expect(last.msg).toMatch(/exit 2/)
  })
})
