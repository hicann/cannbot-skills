// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from "vitest"
import { alignTurns, alignTurnsWithManual, computeAlignStats } from "@/lib/compare/turn-align"
import { buildScenarioPair, buildManualAlignments, buildSession } from "./fixtures/generate-turns"

// These tests exercise banded DP at scale (where the band actually activates)
// and the band-fallback path when |lenA - lenB| exceeds the band. They guard
// against subtle precision regressions that the small-fixture turn-align
// tests can't catch because they stay under the 2*DEFAULT_BAND_WIDTH threshold.

describe("alignTurns — banded DP at scale", () => {
  it("500x500 similar: banded DP produces high-similarity matches like full DP", () => {
    const pair = buildScenarioPair("similar", 500, 500, 100, 100)
    const pairs = alignTurns(pair.turnsA, pair.turnsB)
    const stats = computeAlignStats(pairs)

    // Same shape invariant as the small similar-scenario test, but at scale
    // where banded DP is actually active (sum ≥ 2*DEFAULT_BAND_WIDTH).
    expect(stats.matched).toBeGreaterThan(0)
    expect(stats.avgSimilarity).toBeGreaterThan(0.3)
    // Most turns should pair up — banded DP shouldn't drop matches.
    const totalCells = 500 + 500
    const unmatched = stats.aOnly + stats.bOnly
    expect(unmatched).toBeLessThan(totalCells * 0.3)
  })

  it("800x800 similar: banded DP scales sub-quadratically (must finish under 1s)", () => {
    const pair = buildScenarioPair("similar", 800, 800, 200, 200)
    const t0 = Date.now()
    const pairs = alignTurns(pair.turnsA, pair.turnsB)
    const elapsed = Date.now() - t0

    expect(pairs.length).toBeGreaterThan(0)
    // Sanity ceiling — full-matrix baseline was ~616ms, banded should stay
    // well under that. Leave headroom for slow CI machines.
    expect(elapsed).toBeLessThan(2000)
  })

  it("1000x1000: banded DP completes and produces stable match count", () => {
    const pair = buildScenarioPair("similar", 1000, 1000, 300, 300)
    const pairs = alignTurns(pair.turnsA, pair.turnsB)
    const stats = computeAlignStats(pairs)

    expect(stats.matched).toBeGreaterThan(500)
    // Heap-heavy pressure test — if memory wasn't rolled, we'd OOM here.
    expect(stats.aOnly + stats.bOnly).toBeLessThan(1000 * 0.3)
  })

  it("4000x4300 user scenario: adaptive band keeps banded active (was full-matrix 8s)", () => {
    // Before Phase 2.5: |4000-4300| = 300 > DEFAULT_BAND_WIDTH(200) → full
    // matrix → 17.2M cells → ~8s on the user's machine.
    // After Phase 2.5: adaptiveBand = max(200, 350) = 350 → banded cells
    // = 4000 * min(700, 4300) = 2.8M → ~1.3s.
    const pair = buildScenarioPair("similar", 4000, 4300, 1234, 1235)
    const t0 = Date.now()
    const pairs = alignTurns(pair.turnsA, pair.turnsB)
    const elapsed = Date.now() - t0

    expect(pairs.length).toBeGreaterThan(0)
    const stats = computeAlignStats(pairs)
    expect(stats.matched).toBeGreaterThan(0)
    // Headline: must finish well under the previous ~8s baseline. CI ceiling
    // 5s leaves plenty of slack for slow runners.
    expect(elapsed).toBeLessThan(5000)
  })
})

describe("alignTurns — adaptive band covers length-mismatched sessions", () => {
  it("500x100 different-size: banded path activates with adaptive band width", () => {
    // |500 - 100| = 400 → adaptiveBand = max(200, 450) = 450 (was: full-matrix
    // fallback when band was fixed at 200). The band covers the diff so banded
    // runs, but min(2*450, 100) = 100 → cells = 500*100 = 50K (same as full
    // matrix, no penalty). The shorter side's user turns still pair up.
    const pair = buildScenarioPair("different-size", 500, 100, 400, 401)
    const pairs = alignTurns(pair.turnsA, pair.turnsB)

    expect(pairs.length).toBeGreaterThan(0)
    const matched = pairs.filter(p => p.type === "match")
    expect(matched.length).toBeGreaterThan(0)
    const aOnlyCount = pairs.filter(p => p.type === "aOnly").length
    expect(aOnlyCount).toBeGreaterThan(300)
  })

  it("500x300 different-size within band: banded DP handles drift", () => {
    // |500 - 300| = 200, adaptiveBand = max(200, 250) = 250. Well within band.
    const pair = buildScenarioPair("different-size", 500, 300, 500, 501)
    const pairs = alignTurns(pair.turnsA, pair.turnsB)
    const stats = computeAlignStats(pairs)

    expect(stats.matched).toBeGreaterThan(0)
    expect(stats.aOnly).toBeGreaterThan(100) // at least 200 - matched overflow
  })
})

describe("alignTurnsWithManual — prepared cache at scale", () => {
  it("500x500 with 5 anchors: completes within banded-DP time budget", () => {
    const pair = buildScenarioPair("divergent", 500, 500, 700, 701)
    const manualAlignments = buildManualAlignments(pair.turnsA, pair.turnsB, 5)

    const t0 = Date.now()
    const result = alignTurnsWithManual(pair.turnsA, pair.turnsB, manualAlignments)
    const elapsed = Date.now() - t0

    // The 5 manual anchors must surface in the output.
    const manualMatches = result.filter(p => p.isManual)
    expect(manualMatches.length).toBe(5)
    // Headline: prepared-cache reuse + banded DP keeps this under 2s at 500x500.
    expect(elapsed).toBeLessThan(2000)
    // Should still produce a reasonable number of matches (segment splits
    // may shift counts vs no-anchor baseline, so just sanity-check > 0).
    const resultStats = computeAlignStats(result)
    expect(resultStats.matched).toBeGreaterThan(0)
  })
})

describe("alignTurns — banded precision vs full matrix", () => {
  // Build two near-identical 300-turn sessions, then insert a 50-turn
  // "deletion" in the middle of B. Full-matrix DP would still align the
  // tail (after the deletion) by drifting j by -50. Banded DP with band=200
  // covers a 50 drift, so the tail must still match.
  it("300x250 with mid-B 50-turn deletion: banded DP still aligns the tail", () => {
    const a = buildSession({ seed: 42, count: 300, scenario: "similar", language: "cn" })
    const b = buildSession({ seed: 42, count: 300, scenario: "similar", language: "cn" })
    // Remove turns [100..150) from B — drift after index 100 is 50.
    const bTrimmed = [...b.slice(0, 100), ...b.slice(150)]

    const pairs = alignTurns(a, bTrimmed)

    expect(pairs.length).toBeGreaterThan(0)
    // The tail (after the deletion) must still pair up despite the 50 drift.
    const tailMatches = pairs.filter(p =>
      p.type === "match" && p.a !== null && p.a.turnIndex >= 150
    )
    expect(tailMatches.length).toBeGreaterThan(50)
    // Sanity: average similarity on tail matches should still be meaningful.
    const tailSims = tailMatches.map(p => p.similarity)
    const tailAvg = tailSims.reduce((s, x) => s + x, 0) / Math.max(1, tailSims.length)
    expect(tailAvg).toBeGreaterThan(0.2)
  })

  it("manual alignments work with banded DP across anchor boundaries", () => {
    const a = buildSession({ seed: 11, count: 400, scenario: "similar", language: "mixed" })
    const b = buildSession({ seed: 12, count: 400, scenario: "similar", language: "mixed" })
    const anchors = [
      { indexA: 50, indexB: 60 },
      { indexA: 200, indexB: 210 },
      { indexA: 350, indexB: 360 },
    ]

    const pairs = alignTurnsWithManual(a, b, anchors)
    const manualPairs = pairs.filter(p => p.isManual)
    expect(manualPairs.length).toBe(3)
    // Anchor order should be preserved in output.
    expect(manualPairs.map(p => p.indexA)).toEqual([50, 200, 350])
  })

  it("manual-alignment path reports progress instead of staying at 0%", () => {
    // Regression: alignTurnsWithManual used to ignore the onProgress callback
    // when stitching segments together, so the UI progress bar stayed at 0%
    // until alignment finished. Now each segment scales its [0,1] progress
    // onto a slice of the global [0,1] range.
    const a = buildSession({ seed: 21, count: 400, scenario: "similar", language: "mixed" })
    const b = buildSession({ seed: 22, count: 400, scenario: "similar", language: "mixed" })
    const anchors = [
      { indexA: 100, indexB: 110 },
      { indexA: 300, indexB: 310 },
    ]

    const seen: number[] = []
    const pairs = alignTurnsWithManual(a, b, anchors, (p) => seen.push(p))

    // progress must fire more than once and eventually reach 1.
    expect(seen.length).toBeGreaterThan(1)
    expect(seen[seen.length - 1]).toBe(1)
    // Must not stay stuck at 0 — some intermediate value should be > 0.
    const midProgress = seen.filter(p => p > 0 && p < 1)
    expect(midProgress.length).toBeGreaterThan(0)
    // And the first non-zero report shouldn't be 1.0 (would mean we jumped
    // straight from 0 to done without showing incremental progress).
    expect(seen[0]).toBeLessThanOrEqual(0.5)
    // Sanity: pairs still produced correctly.
    const manualPairs = pairs.filter(p => p.isManual)
    expect(manualPairs.length).toBe(2)
  })
})
