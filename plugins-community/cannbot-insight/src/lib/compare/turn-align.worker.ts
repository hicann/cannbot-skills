// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { alignTurnsWithManual, computeAlignStats } from "./turn-align"
import type { LightAlignedPair } from "./turn-align"

self.onmessage = (e: MessageEvent) => {
  const { turnsA, turnsB, manualAlignments } = e.data as {
    turnsA: any[]; turnsB: any[]; manualAlignments: any[]
  }
  const onProgress = (progress: number) => {
    ;(self as any).postMessage({ type: "progress", progress })
  }
  const pairs = alignTurnsWithManual(turnsA, turnsB, manualAlignments, onProgress)
  const stats = computeAlignStats(pairs)
  // Phase 8: return light pairs (index/sim/type only) instead of full pairs.
  // The main thread rebuilds `a`/`b` from turnsA[indexA]/turnsB[indexB] which
  // it already holds — avoids structured-cloning ~8000 × full TurnData objects
  // (content + toolCalls + skillEvents) on the way out of the worker.
  const lightPairs: LightAlignedPair[] = pairs.map(p => ({
    indexA: p.indexA,
    indexB: p.indexB,
    similarity: p.similarity,
    type: p.type,
    isManual: p.isManual,
  }))
  ;(self as any).postMessage({ type: "done", pairs: lightPairs, stats })
}
