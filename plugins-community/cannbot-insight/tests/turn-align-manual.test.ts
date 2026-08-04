// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { alignTurns, alignTurnsWithManual, type TurnData } from "@/lib/compare/turn-align"

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

describe("alignTurnsWithManual", () => {
  it("returns pure auto alignment when no manual alignments given", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the login bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I fixed the login bug" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the login bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I fixed the login bug" }),
    ]

    const auto = alignTurns(a, b)
    const manual = alignTurnsWithManual(a, b, [])

    expect(manual.length).toBe(auto.length)
    expect(manual.every(p => !p.isManual)).toBe(true)
    for (let i = 0; i < auto.length; i++) {
      expect(manual[i].type).toBe(auto[i].type)
      expect(manual[i].indexA).toBe(auto[i].indexA)
      expect(manual[i].indexB).toBe(auto[i].indexB)
    }
  })

  it("forces a manual anchor pair as match with isManual=true", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "write a python script" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "here is the python script" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "write a python script" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "here is the python script" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 1 }])
    const manualPair = pairs.find(p => p.isManual)
    expect(manualPair).toBeDefined()
    expect(manualPair!.indexA).toBe(0)
    expect(manualPair!.indexB).toBe(1)
    expect(manualPair!.type).toBe("match")
  })

  it("realigns segments around anchor — anchor splits sequences", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "create file A" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "created file A" }),
      makeTurn({ turnIndex: 2, role: "user", content: "create file B" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "created file B" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "create file X" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "created file X" }),
      makeTurn({ turnIndex: 2, role: "user", content: "create file B" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "created file B" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 2, indexB: 2 }])
    const manualPair = pairs.find(p => p.isManual)
    expect(manualPair!.indexA).toBe(2)
    expect(manualPair!.indexB).toBe(2)

    const preSegment = pairs.filter(p => p.indexA !== null && p.indexA < 2)
    expect(preSegment.length).toBeGreaterThan(0)

    const postSegment = pairs.filter(p => p.indexA !== null && p.indexA > 2)
    expect(postSegment.length).toBeGreaterThan(0)
  })

  it("handles multiple anchors in correct order", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "task alpha" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done alpha" }),
      makeTurn({ turnIndex: 2, role: "user", content: "task beta" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "done beta" }),
      makeTurn({ turnIndex: 4, role: "user", content: "task gamma" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "task alpha" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done alpha" }),
      makeTurn({ turnIndex: 2, role: "user", content: "task beta" }),
      makeTurn({ turnIndex: 3, role: "assistant", content: "done beta" }),
      makeTurn({ turnIndex: 4, role: "user", content: "task gamma" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [
      { indexA: 0, indexB: 0 },
      { indexA: 4, indexB: 4 },
    ])
    const manualPairs = pairs.filter(p => p.isManual)
    expect(manualPairs.length).toBe(2)
    expect(manualPairs[0].indexA).toBe(0)
    expect(manualPairs[1].indexA).toBe(4)

    const midPairs = pairs.filter(p => !p.isManual && p.indexA !== null && p.indexA > 0 && p.indexA < 4)
    expect(midPairs.length).toBeGreaterThan(0)
  })

  it("anchor pair has computed similarity score", () => {
    const a = [makeTurn({ turnIndex: 0, role: "user", content: "hello world" })]
    const b = [makeTurn({ turnIndex: 0, role: "user", content: "hello world" })]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 0 }])
    const manualPair = pairs.find(p => p.isManual)
    expect(manualPair!.similarity).toBeGreaterThan(0)
  })

  it("anchor with dissimilar content still forced as match", () => {
    const a = [makeTurn({ turnIndex: 0, role: "user", content: "python web server" })]
    const b = [makeTurn({ turnIndex: 0, role: "assistant", content: "rust database optimization" })]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 0 }])
    const manualPair = pairs.find(p => p.isManual)
    expect(manualPair).toBeDefined()
    expect(manualPair!.type).toBe("match")
    expect(manualPair!.isManual).toBe(true)
  })

  it("anchor consumes turns — they appear in no other pair", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "hello" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "world" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "hello" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "world" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 1 }])
    const aIndices = pairs.filter(p => p.indexA === 0)
    expect(aIndices.length).toBe(1)
    const bIndices = pairs.filter(p => p.indexB === 1)
    expect(bIndices.length).toBe(1)
  })

  it("empty sequences with manual alignment", () => {
    const a: TurnData[] = []
    const b: TurnData[] = []

    const pairs = alignTurnsWithManual(a, b, [])
    expect(pairs.length).toBe(0)
  })

  it("anchor at tail end — segment after anchor is empty", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "task one" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done one" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "task one" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done one" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 1, indexB: 1 }])
    const manualPair = pairs.find(p => p.isManual)
    expect(manualPair!.indexA).toBe(1)
    expect(manualPair!.indexB).toBe(1)

    const prePairs = pairs.filter(p => !p.isManual)
    expect(prePairs.length).toBeGreaterThan(0)
  })

  it("result is sorted by sequence position", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "alpha" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "beta" }),
      makeTurn({ turnIndex: 2, role: "user", content: "gamma" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "alpha" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "beta" }),
      makeTurn({ turnIndex: 2, role: "user", content: "gamma" }),
    ]

    const pairs = alignTurnsWithManual(a, b, [{ indexA: 1, indexB: 1 }])
    const idxSequence = pairs.map(p => p.indexA ?? p.indexB ?? -1)
    for (let i = 1; i < idxSequence.length; i++) {
      expect(idxSequence[i]).toBeGreaterThanOrEqual(idxSequence[i - 1])
    }
  })

  it("anchor forces cross-role alignment that auto-alignment would not make", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "fix the auth bug" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "I fixed the auth bug" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "assistant", content: "I fixed the auth bug" }),
      makeTurn({ turnIndex: 1, role: "user", content: "verify the auth fix" }),
    ]

    const autoPairs = alignTurns(a, b)
    const autoScore = autoPairs.filter(p => p.type === "match").reduce((s, p) => s + p.similarity, 0)

    const manualPairs = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 1 }])
    const forcedPair = manualPairs.find(p => p.isManual)
    expect(forcedPair!.a!.role).toBe("user")
    expect(forcedPair!.b!.role).toBe("user")

    const manualMatches = manualPairs.filter(p => p.type === "match" && !p.isManual)
    const manualScore = manualMatches.reduce((s, p) => s + p.similarity, 0) + (forcedPair?.similarity ?? 0)
    expect(manualScore).toBeGreaterThan(autoScore * 0.5)
  })

  it("removing a manual alignment restores auto alignment", () => {
    const a = [
      makeTurn({ turnIndex: 0, role: "user", content: "task one" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done one" }),
    ]
    const b = [
      makeTurn({ turnIndex: 0, role: "user", content: "task one" }),
      makeTurn({ turnIndex: 1, role: "assistant", content: "done one" }),
    ]

    const withManual = alignTurnsWithManual(a, b, [{ indexA: 0, indexB: 1 }])
    expect(withManual.some(p => p.isManual)).toBe(true)

    const restored = alignTurnsWithManual(a, b, [])
    expect(restored.every(p => !p.isManual)).toBe(true)
  })
})
