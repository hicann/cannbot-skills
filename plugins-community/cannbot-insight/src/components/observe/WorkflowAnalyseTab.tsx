"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState, useSyncExternalStore } from "react"
import { Button } from "@/components/ui/button"
import { WorkflowFlowChart, type Analysis } from "./WorkflowFlowChart"
import demoAnalysis from "@/lib/workflow-demo-analysis.json"

const STORAGE_KEY = (taskId: string) => `wf-analysis-${taskId}`

function isAnalysis(obj: unknown): obj is Analysis {
  return !!obj && typeof obj === "object" && Array.isArray((obj as { flow?: unknown }).flow)
}

// Same-tab localStorage writes don't fire the `storage` event, so writes call emit()
// to notify useSyncExternalStore subscribers to re-read.
const listeners = new Set<() => void>()
function subscribe(cb: () => void) {
  listeners.add(cb)
  return () => { listeners.delete(cb) }
}
function emit() {
  for (const l of listeners) l()
}

interface Props {
  taskId: string
}

export function WorkflowAnalyseTab({ taskId }: Props) {
  const [text, setText] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)

  const raw = useSyncExternalStore(
    subscribe,
    () => localStorage.getItem(STORAGE_KEY(taskId)),
    () => null,
  )
  const analysis = useMemo<Analysis | null>(() => {
    if (!raw) return null
    try {
      const parsed = JSON.parse(raw)
      return isAnalysis(parsed) ? parsed : null
    } catch {
      return null
    }
  }, [raw])

  function persist(json: string) {
    localStorage.setItem(STORAGE_KEY(taskId), json)
    emit()
    setEditing(false)
  }

  function render() {
    setError(null)
    try {
      const trimmed = text.trim()
      const parsed = JSON.parse(trimmed)
      if (!isAnalysis(parsed)) {
        setError("JSON 缺少 flow 数组，请确认是分析输出格式")
        return
      }
      persist(trimmed)
    } catch (e) {
      setError(`JSON 解析失败: ${e instanceof Error ? e.message : String(e)}`)
    }
  }

  function loadDemo() {
    const demo = JSON.stringify(demoAnalysis)
    setText(demo)
    persist(demo)
    setError(null)
  }

  function reset() {
    setEditing(true)
    setError(null)
  }

  function clearSaved() {
    localStorage.removeItem(STORAGE_KEY(taskId))
    emit()
    setEditing(true)
    setText("")
    setError(null)
  }

  if (analysis && !editing) {
    return (
      <div className="h-full overflow-auto">
        <div className="flex items-center justify-between px-4 py-2 border-b sticky top-0 bg-background z-10">
          <p className="text-xs text-muted-foreground">
            分析数据存于 localStorage（按 session 隔离）。重新分析：导出 MD → Claude Code → 粘贴 JSON。
          </p>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={reset}>重新粘贴</Button>
            <Button variant="outline" size="sm" onClick={clearSaved}>清除</Button>
          </div>
        </div>
        <WorkflowFlowChart analysis={analysis} />
      </div>
    )
  }

  return (
    <div className="h-full overflow-auto p-4">
      <div className="max-w-3xl mx-auto space-y-4">
        <div className="rounded-lg border bg-card p-4 space-y-2">
          <h3 className="font-semibold text-sm">Audit（流程框图 + 问题审计）</h3>
          <p className="text-xs text-muted-foreground leading-relaxed">
            把该 session 的实际流程轨迹 MD 喂给 Claude Code（用约定的提示词），将返回的 JSON 粘贴到下方，
            点「渲染分析」即画出实际流程框图 + 每个节点/阶段的问题与优化建议。结果按 session 存于 localStorage。
          </p>
          <ol className="text-xs text-muted-foreground list-decimal pl-5 space-y-1">
            <li>导出该 session 的 workflow 轨迹 MD（精简格式）</li>
            <li>用 Audit 提示词喂给 Claude Code，拿到 JSON</li>
            <li>把 JSON 粘贴到下方文本框 → 渲染</li>
          </ol>
        </div>

        <div className="space-y-2">
          <textarea
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder='在此粘贴分析 JSON（含 sessionSummary / flow / workflowLevelIssues / optimizationPriorities）'
            className="w-full h-72 rounded-md border bg-background p-3 font-mono text-xs resize-y focus:outline-none focus:ring-2 focus:ring-ring"
          />
          {error && <p className="text-xs text-red-600">{error}</p>}
          <div className="flex gap-2">
            <Button size="sm" onClick={render} disabled={!text.trim()}>渲染分析</Button>
            <Button size="sm" variant="outline" onClick={loadDemo}>载入示例 (SoftplusV2Grad)</Button>
          </div>
        </div>
      </div>
    </div>
  )
}
