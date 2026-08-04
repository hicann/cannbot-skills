// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// @vitest-environment happy-dom

// Phase 5/6 render smoke tests. We don't have @testing-library/react, so we
// use react-dom/server renderToStaticMarkup. useEffect doesn't run under
// SSR, so we mock useTurnAlignWorker to inject a non-loading state and
// exercise the PairCard render path (which the loading-branch test below
// doesn't reach). The virtualization window logic itself runs inside useEffect
// (scroll listener), so it's not covered here — it's exercised manually and
// guarded by typecheck + the existing alignment tests.

import { describe, it, expect, vi } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { buildScenarioPair, buildSession } from "./fixtures/generate-turns"
import { alignTurnsWithManual, computeAlignStats } from "@/lib/compare/turn-align"
import type { AlignedPair } from "@/lib/compare/turn-align"

// vi.mock is hoisted above all imports by vitest's transformer. Anything the
// factory closes over must also be hoisted; vi.hoisted lifts a mutable holder
// the factory can safely return, which we populate per-test.
const mockWorker = vi.hoisted(() => ({
  pairs: [] as AlignedPair[],
  stats: {
    matched: 0, aOnly: 0, bOnly: 0, avgSimilarity: 0,
    highSimilarity: 0, mediumSimilarity: 0, lowSimilarity: 0,
  },
  loading: true,
  progress: 0,
}))

vi.mock("@/lib/compare/use-turn-align-worker", () => ({
  useTurnAlignWorker: () => mockWorker,
}))

import { CompareTurns } from "@/components/compare/CompareTurns"

describe("CompareTurns — Phase 5/6 render smoke test", () => {
  it("renders loading state without throwing (validates PairCard extraction imports)", () => {
    const pair = buildScenarioPair("similar", 50, 50, 999, 998)
    mockWorker.loading = true
    mockWorker.progress = 0
    mockWorker.pairs = []

    const html = renderToStaticMarkup(
      createElement(CompareTurns, { turnsA: pair.turnsA, turnsB: pair.turnsB })
    )
    expect(html).toContain("正在对齐 turn")
    expect(html).toContain("2,500")
    expect(html).toMatch(/传输|对齐|渲染/)
  })

  it("renders gracefully when both turn arrays are empty (no loading spinner crash)", () => {
    mockWorker.loading = false
    mockWorker.progress = 1
    mockWorker.pairs = []

    const html = renderToStaticMarkup(
      createElement(CompareTurns, { turnsA: [], turnsB: [] })
    )
    expect(typeof html).toBe("string")
    expect(html.length).toBeGreaterThan(0)
  })

  it("renders PairCard path when worker returns pairs (validates Phase 6 virtualization JSX)", () => {
    const a = buildSession({ seed: 1, count: 20, scenario: "similar", language: "cn" })
    const b = buildSession({ seed: 1, count: 20, scenario: "similar", language: "cn" })
    const pairs = alignTurnsWithManual(a, b, [])
    mockWorker.loading = false
    mockWorker.progress = 1
    mockWorker.pairs = pairs
    mockWorker.stats = computeAlignStats(pairs)

    // Under SSR, useEffect doesn't run, so visibleStartIdx/End stay at their
    // initial values (0 / filteredPairs.length-1). virtualRange = full range,
    // so all 20 PairCards render. This still validates the PairCard +
    // placeholder JSX without exercising the scroll-driven window logic.
    const html = renderToStaticMarkup(
      createElement(CompareTurns, { turnsA: a, turnsB: b })
    )
    // Must not contain the loading indicator — pairs path was taken.
    expect(html).not.toContain("正在对齐 turn")
    // PairCard renders TurnPanels with the assistant role badge.
    expect(html).toContain("assistant")
    expect(html.length).toBeGreaterThan(0)
  })
})
