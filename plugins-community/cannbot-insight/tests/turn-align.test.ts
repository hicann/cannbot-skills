// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { alignTurns, computeAlignStats, computeContentDiff, similarityLabel, type TurnData } from "@/lib/compare/turn-align"

function makeTurn(overrides: Partial<TurnData> & { turnIndex: number; role: string }): TurnData {
  return {
    turnId: `t-${overrides.turnIndex}`,
    turnIndex: overrides.turnIndex,
    role: overrides.role,
    content: overrides.content ?? null,
    contentSummary: overrides.contentSummary ?? null,
    totalTokens: overrides.totalTokens ?? 0,
    inputTokens: overrides.inputTokens ?? 0,
    outputTokens: overrides.outputTokens ?? 0,
    reasoningTokens: overrides.reasoningTokens ?? 0,
    latencyMs: overrides.latencyMs ?? 0,
    model: overrides.model ?? null,
    toolCalls: overrides.toolCalls ?? [],
    skillEvents: overrides.skillEvents ?? [],
  }
}

describe("alignTurns", () => {
  it("aligns identical turn sequences", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the authentication bug in login module" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I will fix the authentication bug by updating the login module" }),
      makeTurn({ turnIndex: 2, role: "user", content: "how are the tests running" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the authentication bug in login module" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I will fix the authentication bug by updating the login module" }),
      makeTurn({ turnIndex: 2, role: "user", content: "how are the tests running" }),
    ]

    const pairs = alignTurns(a, b)
    const matches = pairs.filter(p => p.type === "match")
    expect(matches.length).toBe(3)
    expect(matches.every(p => p.similarity >= 0.4)).toBe(true)
  })

  it("aligns similar content via cosine similarity even with different wording", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "please fix the authentication bug in the login module" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I will fix the authentication bug by updating the login module validation logic" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the authentication bug in the login module please" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "the authentication bug in login module requires fixing the validation pipeline" }),
    ]

    const pairs = alignTurns(a, b)
    const matches = pairs.filter(p => p.type === "match")
    expect(matches.length).toBeGreaterThanOrEqual(2)

    const userMatch = matches.find(p => p.a?.role === "user" && p.b?.role === "user")
    expect(userMatch).toBeDefined()
    expect(userMatch!.similarity).toBeGreaterThanOrEqual(0.5)
  })

  it("distinguishes unrelated content from related content via cosine", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "implement database migration script" }),
      makeTurn({ turnIndex: 1, role: "user", content: "deploy the application to production server" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "implement database migration script" }),
      makeTurn({ turnIndex: 1, role: "user", content: "configure network firewall settings" }),
    ]

    const pairs = alignTurns(a, b)
    const matches = pairs.filter(p => p.type === "match")

    const sameTopicMatch = matches.find(p => p.a?.content?.includes("database") && p.b?.content?.includes("database"))
    expect(sameTopicMatch).toBeDefined()
    expect(sameTopicMatch!.similarity).toBeGreaterThan(
      matches.find(p => p.a?.content?.includes("deploy") || p.b?.content?.includes("network"))?.similarity ?? 0
    )
  })

  it("aligns similar content turns across different indices", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the bug in module X" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I will help fix the bug" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "system", content: "system prompt" }),
      makeTurn({ turnIndex: 1, role: "user", content: "fix the bug in module X" }),
      makeTurn({ turnIndex: 2, role: "assistant", content: "I will help fix the bug" }),
    ]

    const pairs = alignTurns(a, b)
    const matches = pairs.filter(p => p.type === "match")
    expect(matches.length).toBeGreaterThanOrEqual(2)

    const userMatch = matches.find(p => p.a?.role === "user" && p.b?.role === "user")
    expect(userMatch).toBeDefined()

    const bOnly = pairs.filter(p => p.type === "bOnly")
    expect(bOnly.length).toBeGreaterThanOrEqual(1)
    expect(bOnly.some(p => p.b?.role === "system")).toBe(true)
  })

  it("handles empty arrays", () => {
    const pairs1 = alignTurns([], [])
    expect(pairs1).toEqual([])

    const pairs2 = alignTurns([makeTurn({ turnIndex: 0, role: "user", content: "hello" })], [])
    expect(pairs2.length).toBe(1)
    expect(pairs2[0].type).toBe("aOnly")

    const pairs3 = alignTurns([], [makeTurn({ turnIndex: 0, role: "user", content: "hello" })])
    expect(pairs3.length).toBe(1)
    expect(pairs3[0].type).toBe("bOnly")
  })

  it("aligns by tool call overlap", () => {
    const a = [
      makeTurn({
        turnIndex: 0, role: "assistant", content: "reading file",
        toolCalls: [{ toolCallId: "tc1", toolName: "Read", state: "ok", durationMs: 100 }],
      }),
      makeTurn({
        turnIndex: 1, role: "assistant", content: "writing file",
        toolCalls: [{ toolCallId: "tc2", toolName: "Write", state: "ok", durationMs: 200 }],
      }),
    ]
    const b = [
      makeTurn({
        turnIndex: 0, role: "assistant", content: "checking file",
        toolCalls: [{ toolCallId: "tc3", toolName: "Read", state: "ok", durationMs: 150 }],
      }),
      makeTurn({
        turnIndex: 1, role: "assistant", content: "modifying file",
        toolCalls: [{ toolCallId: "tc4", toolName: "Write", state: "ok", durationMs: 250 }],
      }),
    ]

    const pairs = alignTurns(a, b)
    const matches = pairs.filter(p => p.type === "match")
    expect(matches.length).toBe(2)

    const readMatch = matches.find(p =>
      p.a?.toolCalls.some(tc => tc.toolName === "Read") &&
      p.b?.toolCalls.some(tc => tc.toolName === "Read")
    )
    expect(readMatch).toBeDefined()
  })

  it("aligns by skill event overlap", () => {
    const a = [
      makeTurn({
        turnIndex: 0, role: "assistant", content: "loading skill",
        skillEvents: [{ skillName: "cannbot-skill-review", eventType: "load", success: true }],
      }),
    ]
    const b = [
      makeTurn({
        turnIndex: 0, role: "assistant", content: "skill loaded",
        skillEvents: [{ skillName: "cannbot-skill-review", eventType: "load", success: true }],
      }),
    ]

    const pairs = alignTurns(a, b)
    expect(pairs[0].type).toBe("match")
    expect(pairs[0].similarity).toBeGreaterThan(0.3)
  })

  it("penalizes role mismatch", () => {
    const a = [makeTurn({ turnIndex: 0, role: "user", content: "hello" })]
    const b = [makeTurn({ turnIndex: 0, role: "assistant", content: "hello" })]

    const pairs = alignTurns(a, b)
    expect(pairs[0].similarity).toBeLessThan(0.5)
  })
})

describe("computeAlignStats", () => {
  it("computes statistics from aligned pairs", () => {
    const pairs = [
      { indexA: 0, indexB: 0, a: makeTurn({ turnIndex: 0, role: "user", content: "x" }), b: makeTurn({ turnIndex: 0, role: "user", content: "x" }), similarity: 0.85, type: "match" as const },
      { indexA: 1, indexB: 1, a: makeTurn({ turnIndex: 1, role: "user", content: "y" }), b: makeTurn({ turnIndex: 1, role: "user", content: "z" }), similarity: 0.55, type: "match" as const },
      { indexA: 2, indexB: null, a: makeTurn({ turnIndex: 2, role: "user", content: "w" }), b: null, similarity: 0, type: "aOnly" as const },
      { indexA: null, indexB: 2, a: null, b: makeTurn({ turnIndex: 2, role: "user", content: "v" }), similarity: 0, type: "bOnly" as const },
    ]

    const stats = computeAlignStats(pairs)
    expect(stats.matched).toBe(2)
    expect(stats.aOnly).toBe(1)
    expect(stats.bOnly).toBe(1)
    expect(stats.highSimilarity).toBe(1)
    expect(stats.mediumSimilarity).toBe(1)
    expect(stats.lowSimilarity).toBe(0)
  })
})

describe("computeContentDiff", () => {
  it("returns equal range for identical content", () => {
    const ranges = computeContentDiff("hello\nworld", "hello\nworld")
    expect(ranges).toEqual([{ type: "equal", text: "hello\nworld" }])
  })

  it("returns added for B-only content", () => {
    const ranges = computeContentDiff(null, "new content")
    expect(ranges).toEqual([{ type: "added", text: "new content" }])
  })

  it("returns removed for A-only content", () => {
    const ranges = computeContentDiff("old content", null)
    expect(ranges).toEqual([{ type: "removed", text: "old content" }])
  })

  it("detects added and removed lines", () => {
    const ranges = computeContentDiff("line1\nline2\nline3", "line1\nline2b\nline3")
    expect(ranges.filter(r => r.type === "added").length + ranges.filter(r => r.type === "removed").length).toBeGreaterThanOrEqual(1)
    expect(ranges.filter(r => r.type === "equal").length).toBeGreaterThanOrEqual(1)
  })

  it("returns empty for both null", () => {
    const ranges = computeContentDiff(null, null)
    expect(ranges).toEqual([])
  })
})

describe("similarityLabel", () => {
  it("returns correct labels for similarity thresholds", () => {
    expect(similarityLabel(0.9).label).toBe("相似")
    expect(similarityLabel(0.5).label).toBe("相似")
    expect(similarityLabel(0.3).label).toBe("不相似")
    expect(similarityLabel(0.1).label).toBe("不相似")
  })
})
