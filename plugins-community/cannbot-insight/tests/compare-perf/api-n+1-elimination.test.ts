// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from "vitest"
import { NextRequest } from "next/server"

// vi.mock is hoisted to the top of the file by vitest's transformer. Any
// identifier referenced inside the factory must also be hoisted, otherwise
// it points to a TDZ-bound undefined. vi.hoisted lifts the mock + counts
// into a context the factory can safely reach.
const mocked = vi.hoisted(() => {
  const counts = {
    sessionFindFirst: 0,
    sessionFindMany: 0,
    turnFindFirst: 0,
    turnFindMany: 0,
    toolCallFindMany: 0,
    totalCalls: 0,
    reset() {
      this.sessionFindFirst = 0
      this.sessionFindMany = 0
      this.turnFindFirst = 0
      this.turnFindMany = 0
      this.toolCallFindMany = 0
      this.totalCalls = 0
    },
  }

  const dummySession = {
    id: "s-1",
    taskId: "test-task",
    framework: "opencode",
    version: "1.0.0",
    parentId: null,
    directory: "/tmp",
    summaryAdditions: 0,
    summaryDeletions: 0,
    summaryFiles: 0,
    startTime: new Date("2026-01-01"),
    endTime: new Date("2026-01-02"),
    totalTokens: 1000000,
    totalInputTokens: 800000,
    totalOutputTokens: 200000,
    totalReasoningTokens: 50000,
    totalCacheReadTokens: 700000,
    totalCacheWriteTokens: 100000,
    totalCost: 0.5,
    totalLatencyMs: 60000,
    totalToolCallCount: 200,
    totalLlmCallCount: 100,
    totalSkillLoadCount: 5,
    totalSubagentCount: 27,
    model: "claude-sonnet-4-5",
    user: "dev",
    sourcePath: "/tmp/x",
    executions: Array.from({ length: 28 }, (_, i) => ({
      id: `e-${i}`,
      agentName: i === 0 ? "coder" : `subagent-${i}`,
      agentSessionId: i === 0 ? "root" : `sub-${i}`,
      isSubagent: i > 0,
      parentExecutionId: i > 0 ? "e-0" : null,
      tokens: 50000,
      maxSingleCallTokens: 8000,
      cost: 0.01,
      toolCallCount: 10,
      skillLoadCount: 1,
      model: "claude-sonnet-4-5",
      createdAt: new Date("2026-01-01"),
      latencyMs: 2000,
    })),
    skills: [
      { skillName: "cannbot-skill-review", skillVersion: 1, invocationCount: 3 },
      { skillName: "gitcode-pr-handler", skillVersion: 2, invocationCount: 1 },
    ],
  }

  function makeTurn(idx: number, role: string, isSubagent = false, subagentSessionId: string | null = null) {
    return {
      id: `t-${idx}`,
      turnIndex: idx,
      role,
      content: `content ${idx}`,
      contentJson: null,
      contentSummary: `summary ${idx}`,
      agentName: "coder",
      isSubagent,
      subagentName: null,
      subagentSessionId,
      parentExecutionId: null,
      totalTokens: 1000,
      inputTokens: 800,
      outputTokens: 200,
      reasoningTokens: 0,
      cacheReadTokens: 600,
      cacheWriteTokens: 100,
      inputMessagesCount: 1,
      inputMessagesTokens: 800,
      inputMessagesJson: null,
      contextWindowPct: 0.5,
      ttftMs: 100,
      modelId: "model-1",
      providerId: "anthropic",
      model: "claude-sonnet-4-5",
      finishReason: "stop",
      latencyMs: 1000,
      createdAt_ts: new Date(`2026-01-01T00:00:${String(idx % 60).padStart(2, "0")}Z`),
      createdAt: new Date(`2026-01-01T00:00:${String(idx % 60).padStart(2, "0")}Z`),
      completedAt: null,
      toolCalls: [],
      skillEvents: [],
    }
  }

  const turns = Array.from({ length: 600 }, (_, i) => {
    // Each 22-turn block starts with a (user, assistant) pair belonging to one
    // subagent, so firstAssistant lookups actually find a row and trigger the
    // full computeSystemOverhead findMany path (otherwise it short-circuits
    // and the N+1 findMany count stays hidden).
    const subBlock = Math.floor(i / 22)
    const posInBlock = i % 22
    const inSub = subBlock > 0 && posInBlock < 2
    const subId = inSub ? `sub-${subBlock}` : null
    const role = inSub
      ? (posInBlock === 0 ? "user" : "assistant")
      : (i === 0 ? "assistant" : i === 1 ? "user" : i === 2 ? "assistant" : i % 2 === 1 ? "assistant" : i % 4 === 0 ? "tool_result" : "user")
    return makeTurn(i, role, inSub, subId)
  })

  const prismaMock = {
    session: {
      findFirst: async () => {
        counts.sessionFindFirst++
        counts.totalCalls++
        return dummySession
      },
      findMany: async () => {
        counts.sessionFindMany++
        counts.totalCalls++
        return [dummySession]
      },
    },
    turn: {
      findMany: async (args: unknown) => {
        counts.turnFindMany++
        counts.totalCalls++
        const a = args as {
          where?: {
            subagentSessionId?: string | { not: null } | null
            role?: string
            turnIndex?: { lt?: number }
            isSubagent?: boolean
            sessionId?: string
          }
          distinct?: string[]
        }
        const where = a?.where ?? {}
        if (a?.distinct?.includes("subagentSessionId")) {
          const seen = new Set<string>()
          for (const t of turns) {
            if (t.subagentSessionId && !seen.has(t.subagentSessionId)) seen.add(t.subagentSessionId)
          }
          return Array.from(seen).map(id => ({ subagentSessionId: id }))
        }
        if (typeof where.subagentSessionId === "string") {
          return turns.filter(t => t.subagentSessionId === where.subagentSessionId)
        }
        let result = turns
        if (where.role) result = result.filter(t => t.role === where.role)
        if (where.isSubagent !== undefined) result = result.filter(t => t.isSubagent === where.isSubagent)
        if (where.turnIndex?.lt !== undefined) result = result.filter(t => t.turnIndex < (where.turnIndex!.lt as number))
        if (where.subagentSessionId && typeof where.subagentSessionId === "object") {
          result = result.filter(t => t.subagentSessionId !== null)
        }
        return result
      },
      findFirst: async (args: unknown) => {
        counts.turnFindFirst++
        counts.totalCalls++
        const a = args as { where?: { subagentSessionId?: string; isSubagent?: boolean; role?: string } }
        const where = a?.where ?? {}
        let result = turns
        if (where.subagentSessionId) result = result.filter(t => t.subagentSessionId === where.subagentSessionId)
        if (where.isSubagent !== undefined) result = result.filter(t => t.isSubagent === where.isSubagent)
        if (where.role) result = result.filter(t => t.role === where.role)
        return result[0] ?? null
      },
    },
    toolCall: {
      findMany: async () => {
        counts.toolCallFindMany++
        counts.totalCalls++
        return []
      },
    },
  }

  return { counts, prismaMock }
})

vi.mock("@/lib/db", () => ({ prisma: mocked.prismaMock }))

import { GET as turnsGet } from "@/app/api/observe/session/turns/route"
import { GET as sessionGet } from "@/app/api/observe/session/route"

const counts = mocked.counts

function makeReq(url: string): NextRequest {
  return new NextRequest(new Request(url))
}

// Fixture: 600 turns with subagentSessionId on every 22nd index (>0) → 27
// subagent ids, mirroring the real-world case in AGENTS.md.
const SUBAGENT_COUNT = 27
const EXPECTED_OVERHEAD_CALLS = SUBAGENT_COUNT + 1 // root + each subagent

describe("compare-perf — turns API N+1 elimination via skipOverhead", () => {
  beforeEach(() => counts.reset())

  it("without skipOverhead: computeSystemOverhead runs 28 findFirst + 28 findMany (N+1 baseline)", async () => {
    const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true")
    const res = await turnsGet(req)
    expect(res.ok).toBe(true)

    // 28 = root + 27 subagents, each issuing findFirst + findMany = 56 overhead calls.
    expect(counts.turnFindFirst).toBe(EXPECTED_OVERHEAD_CALLS)
    expect(counts.turnFindMany).toBeGreaterThanOrEqual(EXPECTED_OVERHEAD_CALLS)
    expect(counts.totalCalls).toBeGreaterThanOrEqual(EXPECTED_OVERHEAD_CALLS * 2)
  })

  it("with skipOverhead=true: computeSystemOverhead skipped entirely, 0 overhead findFirst calls", async () => {
    const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true&skipOverhead=true")
    const res = await turnsGet(req)
    expect(res.ok).toBe(true)

    const body = await res.json()
    expect(body.items[0]).toHaveProperty("systemOverheadTokens")
    expect(body.items[0].systemOverheadTokens).toBe(0)

    expect(counts.turnFindFirst).toBe(0)
    expect(counts.turnFindMany).toBeLessThan(EXPECTED_OVERHEAD_CALLS)
  })

  it("compare flow (2x sessions + 2x turns with skipOverhead): N+1 gone vs unskipped", async () => {
    // Run the compare flow with skipOverhead; the only turn.findFirst calls
    // should be the 2 from sessionGet (one per session for rootFirstPrompt),
    // not the 2*28 from the N+1 overhead loop.
    counts.reset()
    const [s1, s2, t1, t2] = await Promise.all([
      sessionGet(makeReq("http://localhost/api/observe/session?taskId=A")),
      sessionGet(makeReq("http://localhost/api/observe/session?taskId=B")),
      turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=A&includeContent=true&skipOverhead=true")),
      turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=B&includeContent=true&skipOverhead=true")),
    ])
    expect([s1.ok, s2.ok, t1.ok, t2.ok]).toEqual([true, true, true, true])
    const skippedFindFirst = counts.turnFindFirst

    counts.reset()
    await Promise.all([
      sessionGet(makeReq("http://localhost/api/observe/session?taskId=A")),
      sessionGet(makeReq("http://localhost/api/observe/session?taskId=B")),
      turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=A&includeContent=true")),
      turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=B&includeContent=true")),
    ])
    const unskippedFindFirst = counts.turnFindFirst

    // Skipped: 2 (one per session API call). Unskipped: 2 + 2*28 (N+1 overhead).
    expect(skippedFindFirst).toBe(2)
    expect(unskippedFindFirst).toBe(2 + 2 * EXPECTED_OVERHEAD_CALLS)
    // The N+1 elimination is the headline: 56 fewer findFirst calls.
    expect(unskippedFindFirst - skippedFindFirst).toBe(2 * EXPECTED_OVERHEAD_CALLS)
  })
})

describe("compare-perf — session API firstPrompt query optimization", () => {
  beforeEach(() => counts.reset())

  it("session API does not load all user turns — uses findFirst for root + findMany for subagents", async () => {
    const req = makeReq("http://localhost/api/observe/session?taskId=test")
    const res = await sessionGet(req)
    expect(res.ok).toBe(true)

    const body = await res.json()

    const rootAgent = body.agents.find((a: { isSubagent: boolean }) => !a.isSubagent)
    expect(rootAgent.firstPrompt).not.toBeNull()

    const subAgents = body.agents.filter((a: { isSubagent: boolean }) => a.isSubagent)
    expect(subAgents.length).toBeGreaterThan(0)
    expect(subAgents[0].firstPrompt).not.toBeNull()

    // Headline: 1 root findFirst + 1 sub findMany (not N per-subagent serial).
    expect(counts.turnFindFirst).toBe(1)
    expect(counts.turnFindMany).toBe(1)
  })

  it("session API response shape is preserved (agents[].firstPrompt, skills, totals)", async () => {
    const req = makeReq("http://localhost/api/observe/session?taskId=test")
    const res = await sessionGet(req)
    expect(res.ok).toBe(true)

    const body = await res.json()
    expect(body).toHaveProperty("taskId")
    expect(body).toHaveProperty("totalTokens")
    expect(body).toHaveProperty("agents")
    expect(body.agents[0]).toHaveProperty("firstPrompt")
    expect(body.agents[0]).toHaveProperty("agentName")
    expect(body.agents[0]).toHaveProperty("tokens")
    expect(body).toHaveProperty("skills")
    expect(body.skills[0]).toHaveProperty("skillName")
  })
})

describe("compare-perf — turns API maxContentLen payload trimming", () => {
  beforeEach(() => counts.reset())

  it("without maxContentLen: content returned in full", async () => {
    const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true&skipOverhead=true")
    const res = await turnsGet(req)
    expect(res.ok).toBe(true)
    const body = await res.json()
    // Mock content is "content N" (~10 chars) — no truncation.
    expect(body.items[0].content).toBe("content 0")
  })

  it("with maxContentLen=5: content truncated to 5 chars", async () => {
    const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true&skipOverhead=true&maxContentLen=5")
    const res = await turnsGet(req)
    expect(res.ok).toBe(true)
    const body = await res.json()
    expect(body.items[0].content).toBe("conte")
  })

  it("maxContentLen doesn't affect contentSummary (always 200 chars from DB)", async () => {
    const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true&skipOverhead=true&maxContentLen=5")
    const res = await turnsGet(req)
    const body = await res.json()
    // contentSummary is a separate DB field, unaffected by maxContentLen.
    expect(body.items[0].contentSummary).toBe("summary 0")
  })
})
