// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from "vitest"
import { EventEmitter } from "node:events"
import fs from "node:fs"
import path from "node:path"

const FAKE_REPORT = {
  skill_name: "ascendc-crash-debug",
  transcript_sources: ["opencode"],
  summary: { total: 2, pass: 1, fail: 1, na: 0 },
  findings: [
    { instruction_text: "do x", source: "STEP", verdict: "PASS", method: "LLM", category: "step", evidence: "ok", seq: 0 },
  ],
}

let lastWrittenSkillMd: string | null = null

const { spawnMock } = vi.hoisted(() => ({ spawnMock: vi.fn() }))
vi.mock("node:child_process", () => ({ spawn: spawnMock }))

const fakePrisma = {
  session: { findFirst: vi.fn() },
  skillEvent: { findFirst: vi.fn() },
  toolCall: { findMany: vi.fn() },
  turn: { findMany: vi.fn(), findFirst: vi.fn() },
}
vi.mock("@/lib/db", () => ({ prisma: fakePrisma }))

const { POST } = await import("@/app/api/ai/audit-skilleval/route")

function makeRequest(body: unknown): Request {
  return new Request("http://localhost/api/ai/audit-skilleval", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  })
}

/** spawn mock：写 SKILL.md 抓取 + 往 -o 目录写假 audit-report.json/.html + emit close 0。 */
function happySpawn() {
  return vi.fn((_cmd: string, args: string[]) => {
    const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
    proc.stdout = new EventEmitter()
    proc.stderr = new EventEmitter()
    proc.kill = vi.fn()
    setImmediate(() => {
      const skillPath = args[1]
      try {
        lastWrittenSkillMd = fs.readFileSync(path.join(skillPath, "SKILL.md"), "utf8")
      } catch {
        lastWrittenSkillMd = null
      }
      const i = args.indexOf("-o")
      const outDir = i >= 0 ? args[i + 1] : undefined
      if (outDir) {
        fs.mkdirSync(outDir, { recursive: true })
        fs.writeFileSync(path.join(outDir, "audit-report.json"), JSON.stringify(FAKE_REPORT))
        fs.writeFileSync(path.join(outDir, "audit-report.html"), "<!doctype html><body>STUB AUDIT HTML</body>")
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

function seedSkillBody(): void {
  fakePrisma.skillEvent.findFirst.mockResolvedValue({ turnId: "t1" })
  fakePrisma.toolCall.findMany.mockResolvedValue([
    {
      toolName: "skill",
      argsJson: '{"name":"ascendc-crash-debug"}',
      resultJson: '<skill_content name="ascendc-crash-debug">BODY</skill_content>',
    },
  ])
}

describe("audit-skilleval route", () => {
  beforeEach(() => {
    lastWrittenSkillMd = null
    spawnMock.mockReset()
    fakePrisma.session.findFirst.mockReset()
    fakePrisma.skillEvent.findFirst.mockReset()
    fakePrisma.toolCall.findMany.mockReset()
    fakePrisma.turn.findMany.mockReset()
    fakePrisma.turn.findFirst.mockReset()
  })

  it("happy：恢复正文 → spawn skill-eval → NDJSON result 事件", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    seedSkillBody()
    spawnMock.mockImplementation(happySpawn())

    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "ascendc-crash-debug" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; report: { skill_name: string; summary: { fail: number } }; _html: string }
    expect(last.stage).toBe("result")
    expect(last.report.skill_name).toBe("ascendc-crash-debug")
    expect(last.report.summary.fail).toBe(1)
    expect(last._html).toContain("STUB AUDIT HTML")

    const [, args] = spawnMock.mock.calls[0]
    expect(args[0]).toBe("audit")
    expect(args).toContain("--db")
    expect(args).toContain("/tmp/fake.db")
    expect(args).toContain("--session")
    expect(args).toContain("ses_1")
    expect(lastWrittenSkillMd?.startsWith("---")).toBe(true)
    expect(lastWrittenSkillMd).toContain("name: ascendc-crash-debug")
    expect(lastWrittenSkillMd).toContain("BODY")
  })

  it("kind=root → skill resources 文件（task-prompts.md）恢复 → --kind root", async () => {
    const resourceContent = "# Task 调用参数\n\n## 1.1 开发准备\n\n- [ ] 开发日志已创建\n- [ ] 问题目录已创建\n- [ ] 环境检查已执行\n" + "x".repeat(120)
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    fakePrisma.skillEvent.findFirst.mockResolvedValue({ skillName: "ops-registry-invoke-workflow" })
    fakePrisma.toolCall.findMany.mockResolvedValue([
      { toolName: "read", argsJson: '{"filePath":"/x/ops-registry-invoke-workflow/resources/task-prompts.md"}', resultJson: `<content>${resourceContent}</content>` },
    ])
    spawnMock.mockImplementation(happySpawn())

    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "主 agent 编排", kind: "root" }))
    await readNdjson(res)
    expect(res.status).toBe(200)
    const [, args] = spawnMock.mock.calls[0]
    expect(args).toContain("--kind")
    expect(args[args.indexOf("--kind") + 1]).toBe("root")
    expect(lastWrittenSkillMd).toContain("开发准备")
    expect(lastWrittenSkillMd).toContain("name: main-agent-workflow")
  })

  it("kind=root → 无 resources 文件 → 退 SKILL.md body（编排规程）", async () => {
    const skillBody = "# Skill: ops-registry-invoke-workflow\n\n## 核心原则\n\n- 测试驱动\n- 阶段递进\n- 阶段门控\n" + "x".repeat(120)
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    // resolveWorkflowSkillName → skillName; recoverSkillBody → turnId
    fakePrisma.skillEvent.findFirst
      .mockResolvedValueOnce({ skillName: "ops-registry-invoke-workflow" })
      .mockResolvedValueOnce({ turnId: "t1" })
    // first findMany: resource file (empty); second: skill invoke ToolCall
    fakePrisma.toolCall.findMany
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([
        { toolName: "skill", argsJson: '{"name":"ops-registry-invoke-workflow"}', resultJson: `<skill_content name="ops-registry-invoke-workflow">${skillBody}</skill_content>` },
      ])
    spawnMock.mockImplementation(happySpawn())

    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "主 agent 编排", kind: "root" }))
    await readNdjson(res)
    expect(res.status).toBe(200)
    expect(lastWrittenSkillMd).toContain("核心原则")
    expect(lastWrittenSkillMd).toContain("name: main-agent-workflow")
  })

  it("kind=root → 无主 agent Skill invoke → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    fakePrisma.skillEvent.findFirst.mockResolvedValue(null)
    const res = await POST(makeRequest({ taskId: "ses_1", kind: "root" }))
    expect(res.status).toBe(404)
  })

  it("缺 taskId → 400", async () => {
    const res = await POST(makeRequest({ skillName: "x" }))
    expect(res.status).toBe(400)
  })

  it("session 无 sourcePath → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: null })
    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "x" }))
    expect(res.status).toBe(404)
  })

  it("session 不存在 → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue(null)
    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "x" }))
    expect(res.status).toBe(404)
  })

  it("skill 正文恢复不到 → 404", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    fakePrisma.skillEvent.findFirst.mockResolvedValue(null)
    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "ghost" }))
    expect(res.status).toBe(404)
  })

  it("skill-eval 不在 PATH（spawn ENOENT）→ NDJSON error 事件", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    seedSkillBody()
    spawnMock.mockImplementation(() => {
      const proc = new EventEmitter() as EventEmitter & { stdout: EventEmitter; stderr: EventEmitter; kill: () => void }
      proc.stdout = new EventEmitter()
      proc.stderr = new EventEmitter()
      proc.kill = vi.fn()
      setImmediate(() => proc.emit("error", Object.assign(new Error("enoent"), { code: "ENOENT" })))
      return proc
    })
    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "ascendc-crash-debug" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; msg: string }
    expect(last.stage).toBe("error")
    expect(last.msg).toMatch(/skill-eval 未找到/)
  })

  it("skill-eval 非零退出 → NDJSON error 事件带 exit code", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "opencode" })
    seedSkillBody()
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
    const res = await POST(makeRequest({ taskId: "ses_1", skillName: "ascendc-crash-debug" }))
    expect(res.status).toBe(200)
    const events = await readNdjson(res)
    const last = events.at(-1) as { stage: string; msg: string }
    expect(last.stage).toBe("error")
    expect(last.msg).toMatch(/exit 2/)
  })

  it("framework=cannbot-insight → --transcript 结构化 JSON(不走 --db)", async () => {
    fakePrisma.session.findFirst.mockResolvedValue({ id: "ses_cuid_1", sourcePath: "/tmp/fake.db", framework: "cannbot-insight" })
    seedSkillBody()
    fakePrisma.turn.findMany.mockResolvedValue([
      { id: "t1", turnIndex: 0, role: "assistant", content: "调用 skill", agentName: "kernel-developer" },
    ])
    spawnMock.mockImplementation(happySpawn())
    const res = await POST(
      makeRequest({ taskId: "ses_1", skillName: "ascendc-crash-debug", framework: "cannbot-insight" }),
    )
    expect(res.status).toBe(200)
    const [, args] = spawnMock.mock.calls[0]
    expect(args).toContain("--transcript")
    expect(args).not.toContain("--db")
    expect(args.some((a: unknown) => typeof a === "string" && (a as string).endsWith("session.json"))).toBe(true)
  })
})
