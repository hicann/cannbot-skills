// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { bench, describe } from "vitest"
import {
  alignTurns,
  alignTurnsWithManual,
  computeContentDiff,
  computeAlignStats,
} from "@/lib/compare/turn-align"
import {
  buildScenarioPair,
  buildManualAlignments,
  buildSession,
} from "./fixtures/generate-turns"

// Capture memory delta alongside wall time so regressions are visible even
// when timing noise dominates.
function heapDeltaMB(): () => number {
  const baseline = process.memoryUsage().heapUsed
  return () => (process.memoryUsage().heapUsed - baseline) / 1024 / 1024
}

describe("alignTurns — scaling with session size", () => {
  const sizes: Array<[number, number, string]> = [
    [50, 50, "small"],
    [200, 200, "medium"],
    [500, 500, "large"],
    [800, 800, "xlarge"],
  ]

  for (const [n, m, label] of sizes) {
    const pair = buildScenarioPair("similar", n, m, 100, 100)

    bench(
      `alignTurns ${label} ${n}x${m} (similar)`,
      () => {
        alignTurns(pair.turnsA, pair.turnsB)
      },
      { iterations: n >= 500 ? 3 : 10 },
    )
  }
})

describe("alignTurns — scenario cost at fixed 500x500", () => {
  const scenarios = ["similar", "divergent", "different-size", "many-subagents"] as const
  for (const scenario of scenarios) {
    const countB = scenario === "different-size" ? 200 : 500
    const pair = buildScenarioPair(scenario, 500, countB, 200, 201)

    bench(
      `alignTurns scenario=${scenario} 500x${countB}`,
      () => {
        alignTurns(pair.turnsA, pair.turnsB)
      },
      { iterations: 3 },
    )
  }
})

describe("alignTurnsWithManual — repeat prepareTurn cost", () => {
  const pair = buildScenarioPair("divergent", 500, 500, 300, 301)
  const noManual = alignTurns(pair.turnsA, pair.turnsB)
  const matchedPairs = noManual.filter(p => p.type === "match").slice(0, 5)
  const manualAlignments = matchedPairs.map(p => ({
    indexA: p.indexA!,
    indexB: p.indexB!,
  }))

  bench(
    `alignTurnsWithManual 500x500 with ${manualAlignments.length} anchors`,
    () => {
      alignTurnsWithManual(pair.turnsA, pair.turnsB, manualAlignments)
    },
    { iterations: 3 },
  )

  bench(
    `alignTurnsWithManual 500x500 with 0 anchors (baseline path)`,
    () => {
      alignTurnsWithManual(pair.turnsA, pair.turnsB, [])
    },
    { iterations: 3 },
  )
})

describe("alignTurns — memory pressure", () => {
  const pair = buildScenarioPair("similar", 1000, 1000, 400, 400)

  bench(
    `alignTurns 1000x1000 heap delta (MB)`,
    () => {
      const delta = heapDeltaMB()
      alignTurns(pair.turnsA, pair.turnsB)
      return delta()
    },
    { iterations: 1 },
  )
})

describe("computeContentDiff — per-pair LCS cost", () => {
  const pair = buildScenarioPair("similar", 200, 200, 500, 500)

  bench(
    `computeContentDiff × 200 pairs (each side recomputes LCS)`,
    () => {
      for (let i = 0; i < 200; i++) {
        const a = pair.turnsA[i].content
        const b = pair.turnsB[i].content
        computeContentDiff(a, b)
      }
    },
    { iterations: 5 },
  )
})

describe("computeAlignStats — overhead over pairs array", () => {
  const pair = buildScenarioPair("similar", 500, 500, 600, 600)
  const precomputed = alignTurns(pair.turnsA, pair.turnsB)

  bench(
    `computeAlignStats on ${precomputed.length} pairs`,
    () => {
      computeAlignStats(precomputed)
    },
    { iterations: 50 },
  )
})

describe("prepareTurn cost via buildSession", () => {
  bench(
    `buildSession 1000 turns fixture generation`,
    () => {
      buildSession({ seed: 999, count: 1000, scenario: "many-subagents", language: "mixed" })
    },
    { iterations: 5 },
  )
})

describe("manual alignment builder", () => {
  const pair = buildScenarioPair("similar", 500, 500, 700, 701)

  bench(
    `buildManualAlignments 500x500 count=5`,
    () => {
      buildManualAlignments(pair.turnsA, pair.turnsB, 5)
    },
    { iterations: 1000 },
  )
})
