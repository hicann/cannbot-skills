// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect } from "react"
import type { CoverageItem, CoverageStats } from "@/lib/skill-coverage"

export interface SkillCoverageResponse {
  taskId: string
  framework: string | null
  hasAvailableList: boolean
  listFormat: "claude-list" | "opencode-xml" | null
  degradedReason: string | null
  usedSkillCount: number
  items: CoverageItem[]
  stats: CoverageStats
}

const cache = new Map<string, SkillCoverageResponse>()

export function useSkillCoverage(taskId: string, framework?: string) {
  const cached = cache.get(taskId)
  const [data, setData] = useState<SkillCoverageResponse | null>(cached ?? null)
  const [loading, setLoading] = useState(!cached)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const q = new URLSearchParams({ taskId })
    if (framework) q.set("framework", framework)
    fetch(`/api/observe/session/skill-coverage?${q.toString()}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => {
        if (cancelled) return
        cache.set(taskId, d)
        setData(d)
        setError(null)
        setLoading(false)
      })
      .catch(e => {
        if (cancelled) return
        setError(e instanceof Error ? e.message : String(e))
        setLoading(false)
      })
    return () => { cancelled = true }
  }, [taskId, framework])

  return { data, loading, error }
}
