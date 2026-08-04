// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useRef, useState } from "react"
import { alignTurnsWithManual, computeAlignStats } from "./turn-align"
import type { TurnData, ManualAlignment, AlignedPair, LightAlignedPair } from "./turn-align"

const emptyStats = {
  matched: 0, aOnly: 0, bOnly: 0, avgSimilarity: 0,
  highSimilarity: 0, mediumSimilarity: 0, lowSimilarity: 0,
}

export function useTurnAlignWorker(
  turnsA: TurnData[],
  turnsB: TurnData[],
  manualAlignments: ManualAlignment[],
): { pairs: AlignedPair[]; stats: ReturnType<typeof computeAlignStats>; loading: boolean; progress: number } {
  const [pairs, setPairs] = useState<AlignedPair[]>([])
  const [stats, setStats] = useState<ReturnType<typeof computeAlignStats>>(emptyStats)
  const [loading, setLoading] = useState(true)
  const [progress, setProgress] = useState(0)
  const workerRef = useRef<Worker | null>(null)

  useEffect(() => {
    if (turnsA.length === 0 || turnsB.length === 0) {
      setPairs([])
      setStats(emptyStats)
      setLoading(false)
      setProgress(0)
      return
    }

    setLoading(true)
    setProgress(0)

    const runFallback = () => {
      const onProgress = (p: number) => setProgress(p)
      const result = alignTurnsWithManual(turnsA, turnsB, manualAlignments, onProgress)
      setPairs(result)
      setStats(computeAlignStats(result))
      setLoading(false)
      setProgress(1)
    }

    if (typeof Worker === "undefined") {
      runFallback()
      return
    }

    let worker: Worker
    try {
      worker = new Worker(new URL("./turn-align.worker.ts", import.meta.url))
    } catch {
      runFallback()
      return
    }
    workerRef.current = worker

    worker.onmessage = (e: MessageEvent) => {
      const data = e.data
      if (data.type === "progress") {
        setProgress(data.progress)
      } else if (data.type === "done") {
        // Phase 8: worker returns light pairs (no TurnData). Rebuild full
        // pairs from turnsA/B which are already in main-thread memory —
        // no need to structured-clone TurnData back across the worker boundary.
        const lightPairs = data.pairs as LightAlignedPair[]
        const fullPairs: AlignedPair[] = lightPairs.map(p => ({
          ...p,
          a: p.indexA !== null ? turnsA[p.indexA] ?? null : null,
          b: p.indexB !== null ? turnsB[p.indexB] ?? null : null,
        }))
        setPairs(fullPairs)
        setStats(data.stats)
        setLoading(false)
        setProgress(1)
      }
    }

    worker.onerror = () => {
      runFallback()
    }

    worker.postMessage({ turnsA, turnsB, manualAlignments })

    return () => {
      worker.terminate()
      workerRef.current = null
    }
  }, [turnsA, turnsB, manualAlignments])

  return { pairs, stats, loading, progress }
}
