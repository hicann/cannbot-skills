// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { bench, describe } from "vitest"
import zlib from "node:zlib"
import { promisify } from "node:util"
import type { TurnData } from "@/lib/compare/turn-align"
import { buildScenarioPair } from "./fixtures/generate-turns"

const gzip = promisify(zlib.gzip)

function bytesToKB(n: number): number {
  return Math.round(n / 1024 * 10) / 10
}

async function gzipBytes(s: string): Promise<number> {
  const buf = await gzip(s)
  return buf.length
}

interface FieldSubset {
  name: string
  pick: (t: TurnData) => Record<string, unknown>
}

const FULL: FieldSubset = {
  name: "full (current includeContent=true)",
  pick: t => ({ ...t }),
}

const ALIGNMENT_ONLY: FieldSubset = {
  name: "alignment-only (role+content+contentSummary+toolCalls+skillEvents)",
  pick: t => ({
    turnId: t.turnId,
    turnIndex: t.turnIndex,
    role: t.role,
    content: t.content,
    contentSummary: t.contentSummary,
    toolCalls: t.toolCalls.map(tc => ({ toolName: tc.toolName, state: tc.state })),
    skillEvents: t.skillEvents.map(se => ({ skillName: se.skillName, success: se.success })),
  }),
}

const SUMMARY_ONLY: FieldSubset = {
  name: "summary-only (no content, just contentSummary)",
  pick: t => ({
    turnId: t.turnId,
    turnIndex: t.turnIndex,
    role: t.role,
    contentSummary: t.contentSummary,
    toolCalls: t.toolCalls.map(tc => ({ toolName: tc.toolName, state: tc.state })),
    skillEvents: t.skillEvents.map(se => ({ skillName: se.skillName, success: se.success })),
  }),
}

const METRICS_ONLY: FieldSubset = {
  name: "metrics-only (no content/summary, just numbers)",
  pick: t => ({
    turnId: t.turnId,
    turnIndex: t.turnIndex,
    role: t.role,
    totalTokens: t.totalTokens,
    inputTokens: t.inputTokens,
    outputTokens: t.outputTokens,
    reasoningTokens: t.reasoningTokens,
    latencyMs: t.latencyMs,
    model: t.model,
  }),
}

// Phase 7: compare page now fetches with maxContentLen=1000, trimming each
// turn's content to the first 1000 chars. Simulate the same trim here so the
// payload bench reflects the real compare-mode wire size.
const COMPARE_PHASE7: FieldSubset = {
  name: "compare Phase 7 (content trimmed to 1000 chars)",
  pick: t => ({
    turnId: t.turnId,
    turnIndex: t.turnIndex,
    role: t.role,
    content: t.content ? t.content.substring(0, 1000) : null,
    contentSummary: t.contentSummary,
    toolCalls: t.toolCalls.map(tc => ({ toolName: tc.toolName, state: tc.state })),
    skillEvents: t.skillEvents.map(se => ({ skillName: se.skillName, success: se.success })),
  }),
}

const SUBSETS = [FULL, ALIGNMENT_ONLY, SUMMARY_ONLY, COMPARE_PHASE7, METRICS_ONLY]

describe("turns payload — JSON size by field subset", () => {
  const pair = buildScenarioPair("similar", 500, 500, 800, 800)
  const pair2 = buildScenarioPair("many-subagents", 500, 500, 900, 901)

  for (const subset of SUBSETS) {
    bench(
      `serialize 500 turns × 2 sessions — ${subset.name}`,
      () => {
        const a = JSON.stringify(pair.turnsA.map(subset.pick))
        const b = JSON.stringify(pair.turnsB.map(subset.pick))
        return a.length + b.length
      },
      { iterations: 20 },
    )
  }

  bench(
    `serialize 500 turns × 2 (many-subagents) — ${FULL.name}`,
    () => {
      const a = JSON.stringify(pair2.turnsA.map(FULL.pick))
      const b = JSON.stringify(pair2.turnsB.map(FULL.pick))
      return a.length + b.length
    },
    { iterations: 20 },
  )
})

describe("turns payload — gzip compression ratio (full content)", () => {
  const pair = buildScenarioPair("similar", 500, 500, 800, 800)
  const fullA = JSON.stringify(pair.turnsA.map(FULL.pick))
  const fullB = JSON.stringify(pair.turnsB.map(FULL.pick))

  bench(
    `gzip full payload (500 turns × 2 sessions)`,
    async () => {
      const [ga, gb] = await Promise.all([gzipBytes(fullA), gzipBytes(fullB)])
      return ga + gb
    },
    { iterations: 10 },
  )

  const alignmentA = JSON.stringify(pair.turnsA.map(ALIGNMENT_ONLY.pick))
  const alignmentB = JSON.stringify(pair.turnsB.map(ALIGNMENT_ONLY.pick))

  bench(
    `gzip alignment-only payload (500 turns × 2 sessions)`,
    async () => {
      const [ga, gb] = await Promise.all([gzipBytes(alignmentA), gzipBytes(alignmentB)])
      return ga + gb
    },
    { iterations: 10 },
  )
})

describe("turns payload — summary numbers for baseline report", () => {
  const pair = buildScenarioPair("similar", 500, 500, 800, 800)

  bench(
    `report payload sizes (raw + gzip) for ${SUBSETS.length} subsets`,
    async () => {
      const lines: string[] = []
      for (const subset of SUBSETS) {
        const a = JSON.stringify(pair.turnsA.map(subset.pick))
        const b = JSON.stringify(pair.turnsB.map(subset.pick))
        const rawTotal = a.length + b.length
        const gz = (await gzipBytes(a)) + (await gzipBytes(b))
        lines.push(
          `  ${subset.name.padEnd(60)} raw=${bytesToKB(rawTotal)}KB gzip=${bytesToKB(gz)}KB ratio=${Math.round(gz * 100 / rawTotal)}%`
        )
      }
      console.log(lines.join("\n"))
    },
    { iterations: 1 },
  )
})
