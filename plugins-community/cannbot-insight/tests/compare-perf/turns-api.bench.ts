// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { bench, describe, vi } from "vitest"
import { NextRequest } from "next/server"

vi.mock("@/lib/db", () => {
  const counts = {
    sessionFindFirst: 0,
    turnFindMany: 0,
    turnFindFirst: 0,
    toolCallFindMany: 0,
    totalCalls: 0,
    reset() {
      this.sessionFindFirst = 0
      this.turnFindMany = 0
      this.turnFindFirst = 0
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
    // Relations consumed by /api/observe/session/route.ts
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
      outputTokens_duplicate: undefined,
    }
  }

  // Weave assistant turns at the start so computeSystemOverhead's priorMessages
  // contains at least one assistant → triggers the toolCall.findMany branch
  // (otherwise the N+1 tool-call query is silently skipped, masking the issue).
  const turns = Array.from({ length: 600 }, (_, i) => {
    const isSub = i % 22 === 0 && i > 0
    const subId = isSub ? `sub-${i}` : null
    const role = i === 0 ? "assistant" : i === 1 ? "user" : i === 2 ? "assistant" : i % 2 === 1 ? "assistant" : i % 4 === 0 ? "tool_result" : "user"
    return makeTurn(i, role, isSub, subId)
  })

  const prismaMock = {
    session: {
      findFirst: async () => {
        counts.sessionFindFirst++
        counts.totalCalls++
        return dummySession
      },
      findMany: async () => {
        counts.sessionFindFirst++
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
            subagentSessionId?: string | { not: null | string } | null
            role?: string
            turnIndex?: { lt?: number }
            isSubagent?: boolean
            sessionId?: string
          }
          distinct?: string[]
          select?: Record<string, boolean>
        }
        const where = a?.where ?? {}
        // Match the distinct subagent-id query used by turns/route.ts:
        //   where: { sessionId, isSubagent: true, subagentSessionId: { not: null } },
        //   distinct: ['subagentSessionId'],
        //   select: { subagentSessionId: true }
        if (a?.distinct?.includes("subagentSessionId")) {
          const seen = new Set<string>()
          for (const t of turns) {
            if (t.subagentSessionId && !seen.has(t.subagentSessionId)) seen.add(t.subagentSessionId)
          }
          return Array.from(seen).map(id => ({ subagentSessionId: id }))
        }
        // Match a per-subagent turn query (subagentSessionId === string).
        if (typeof where.subagentSessionId === "string") {
          return turns.filter(t => t.subagentSessionId === where.subagentSessionId)
        }
        // Generic match: respect role + subagentSessionId === { not: null }.
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
        const a = args as { where?: { subagentSessionId?: string; isSubagent?: boolean } }
        const where = a?.where ?? {}
        let result = turns
        if (where.subagentSessionId) result = result.filter(t => t.subagentSessionId === where.subagentSessionId)
        if (where.isSubagent !== undefined) result = result.filter(t => t.isSubagent === where.isSubagent)
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

  return { prisma: prismaMock, __counts: counts }
})

import { __counts } from "@/lib/db"
import { GET as turnsGet } from "@/app/api/observe/session/turns/route"
import { GET as sessionGet } from "@/app/api/observe/session/route"

function makeReq(url: string): NextRequest {
  return new NextRequest(new Request(url))
}

describe("turns API — N+1 query count vs subagent count", () => {
  // Fixture turns have 600 entries with subagentSessionId on every 22nd index (>0)
  // → 27 subagent ids (600 / 22 ≈ 27). This mirrors the real-world case described
  // in AGENTS.md (27 independent subagent sessions).
  const subagentCount = 27
  const expectedOverheadCalls = subagentCount + 1 // root + each subagent

  bench(
    `GET /api/observe/session/turns includeContent=true (27 subagents)`,
    async () => {
      __counts.reset()
      const req = makeReq("http://localhost/api/observe/session/turns?taskId=test&includeContent=true")
      const res = await turnsGet(req)
      if (!res.ok) throw new Error(`unexpected status: ${res.status}`)
      // The headline N+1: computeSystemOverhead is called once per (root + subagent),
      // and each call issues at least one prisma.turn.findFirst + one prisma.turn.findMany.
      // 28 subagents → 28 findFirst + 28 findMany just for overhead calculation,
      // even though the result is only used for the contextWindowPct display field.
      console.log(`  [N+1 exposed] turnFindFirst=${__counts.turnFindFirst} (expected ${expectedOverheadCalls}) turnFindMany=${__counts.turnFindMany} total=${__counts.totalCalls}`)
      if (__counts.turnFindFirst !== expectedOverheadCalls) {
        throw new Error(`expected ${expectedOverheadCalls} findFirst calls, got ${__counts.turnFindFirst}`)
      }
    },
    { iterations: 5 },
  )
})

describe("session API — single session overhead", () => {
  bench(
    `GET /api/observe/session (single session, 600 user turns)`,
    async () => {
      __counts.reset()
      const req = makeReq("http://localhost/api/observe/session?taskId=test")
      const res = await sessionGet(req)
      if (!res.ok) throw new Error(`unexpected status: ${res.status}`)
      console.log(`  [session API] sessionFindFirst=${__counts.sessionFindFirst} turnFindMany=${__counts.turnFindMany} total=${__counts.totalCalls}`)
    },
    { iterations: 10 },
  )
})

describe("compare scenario — 2x sessions + 2x turns APIs", () => {
  bench(
    `compare flow: 2x session GET + 2x turns GET (BEFORE skipOverhead)`,
    async () => {
      __counts.reset()
      const [s1, s2, t1, t2] = await Promise.all([
        sessionGet(makeReq("http://localhost/api/observe/session?taskId=A")),
        sessionGet(makeReq("http://localhost/api/observe/session?taskId=B")),
        turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=A&includeContent=true")),
        turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=B&includeContent=true")),
      ])
      console.log(`  [BEFORE] totalCalls=${__counts.totalCalls} turnFindFirst=${__counts.turnFindFirst} turnFindMany=${__counts.turnFindMany}`)
      if (!(s1.ok && s2.ok && t1.ok && t2.ok)) throw new Error("one of the requests failed")
    },
    { iterations: 3 },
  )

  bench(
    `compare flow: 2x session GET + 2x turns GET (AFTER skipOverhead=true)`,
    async () => {
      __counts.reset()
      const [s1, s2, t1, t2] = await Promise.all([
        sessionGet(makeReq("http://localhost/api/observe/session?taskId=A")),
        sessionGet(makeReq("http://localhost/api/observe/session?taskId=B")),
        turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=A&includeContent=true&skipOverhead=true")),
        turnsGet(makeReq("http://localhost/api/observe/session/turns?taskId=B&includeContent=true&skipOverhead=true")),
      ])
      console.log(`  [AFTER]  totalCalls=${__counts.totalCalls} turnFindFirst=${__counts.turnFindFirst} turnFindMany=${__counts.turnFindMany}`)
      if (!(s1.ok && s2.ok && t1.ok && t2.ok)) throw new Error("one of the requests failed")
    },
    { iterations: 3 },
  )
})
