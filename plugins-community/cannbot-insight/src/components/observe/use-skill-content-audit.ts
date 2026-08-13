// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.

import { useState, useEffect, useCallback } from "react"

export interface SkillContentAuditItem {
  skillName: string
  hasContent: boolean
  source: "skill-tool" | "read" | null
  length: number
  lines: number
  fullRead: boolean
  maxLine: number | null
}

export function useSkillContentAudit(taskId: string, framework?: string) {
  const [items, setItems] = useState<SkillContentAuditItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    const q = new URLSearchParams({ taskId })
    if (framework) q.set("framework", framework)
    fetch(`/api/observe/session/skill-content-audit?${q.toString()}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => {
        if (!cancelled) {
          setItems(d.items ?? [])
          setError(null)
          setLoading(false)
        }
      })
      .catch(e => {
        if (!cancelled) {
          setError(e instanceof Error ? e.message : String(e))
          setItems([])
          setLoading(false)
        }
      })
    return () => { cancelled = true }
  }, [taskId, framework])

  const refetch = useCallback(() => {
    setLoading(true)
    const q = new URLSearchParams({ taskId })
    if (framework) q.set("framework", framework)
    fetch(`/api/observe/session/skill-content-audit?${q.toString()}`)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      })
      .then(d => {
        setItems(d.items ?? [])
        setError(null)
        setLoading(false)
      })
      .catch(e => {
        setError(e instanceof Error ? e.message : String(e))
        setItems([])
        setLoading(false)
      })
  }, [taskId, framework])

  return { items, loading, error, refetch }
}
