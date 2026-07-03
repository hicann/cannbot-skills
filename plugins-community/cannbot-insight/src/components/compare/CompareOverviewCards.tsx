"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

interface SkillData {
  skillName: string
  version: number | null
  invocationCount: number
}

interface SessionCompareData {
  taskId: string
  query: string | null
  model: string | null
  totalTokens: number
  totalCost: number
  totalLatencyMs: number
  totalLlmCallCount: number
  totalToolCallCount: number
  totalSubagentCount: number
  totalSkillLoadCount: number
  skills: SkillData[]
}

interface MetricRow {
  label: string
  valueA: number | string
  valueB: number | string
  lowerIsBetter: boolean
  format: (v: number | string) => string
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

function formatCost(n: number): string {
  return `$${n.toFixed(4)}`
}

function formatDuration(ms: number): string {
  if (ms <= 0) return "—"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}min`
  return `${(ms / 3600000).toFixed(1)}h`
}

function numberOrStr(v: number | string): number {
  return typeof v === "number" ? v : 0
}

function compareHighlight(a: number, b: number, lowerIsBetter: boolean): "a" | "b" | "none" {
  if (a === b) return "none"
  if (lowerIsBetter) return a < b ? "a" : "b"
  return a > b ? "a" : "b"
}

export function CompareOverviewCards({ sessionA, sessionB }: { sessionA: SessionCompareData; sessionB: SessionCompareData }) {
  const metrics: MetricRow[] = [
    { label: "Model", valueA: sessionA.model ?? "—", valueB: sessionB.model ?? "—", lowerIsBetter: false, format: v => String(v) },
    { label: "Total Tokens", valueA: sessionA.totalTokens, valueB: sessionB.totalTokens, lowerIsBetter: true, format: v => formatTokens(numberOrStr(v)) },
    { label: "Total Cost", valueA: sessionA.totalCost, valueB: sessionB.totalCost, lowerIsBetter: true, format: v => formatCost(numberOrStr(v)) },
    { label: "Latency", valueA: sessionA.totalLatencyMs, valueB: sessionB.totalLatencyMs, lowerIsBetter: true, format: v => formatDuration(numberOrStr(v)) },
    { label: "LLM Calls", valueA: sessionA.totalLlmCallCount, valueB: sessionB.totalLlmCallCount, lowerIsBetter: true, format: v => String(numberOrStr(v)) },
    { label: "Tool Calls", valueA: sessionA.totalToolCallCount, valueB: sessionB.totalToolCallCount, lowerIsBetter: false, format: v => String(numberOrStr(v)) },
    { label: "Subagents", valueA: sessionA.totalSubagentCount, valueB: sessionB.totalSubagentCount, lowerIsBetter: false, format: v => String(numberOrStr(v)) },
    { label: "Skills", valueA: sessionA.totalSkillLoadCount, valueB: sessionB.totalSkillLoadCount, lowerIsBetter: false, format: v => String(numberOrStr(v)) },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {[sessionA, sessionB].map((session, idx) => (
        <Card key={session.taskId} size="sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Badge variant={idx === 0 ? "blue" : "orange"}>
                Session {idx === 0 ? "A" : "B"}
              </Badge>
              <span className="truncate text-sm">{session.query ?? session.taskId}</span>
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {metrics.map(m => {
                const val = idx === 0 ? m.valueA : m.valueB
                const highlight = typeof m.valueA === "number" && typeof m.valueB === "number"
                  ? compareHighlight(numberOrStr(m.valueA), numberOrStr(m.valueB), m.lowerIsBetter)
                  : "none"
                const isGreen = (highlight === "a" && idx === 0) || (highlight === "b" && idx === 1)

                return (
                  <div key={m.label} className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">{m.label}</span>
                    <span className={cn("tabular-nums font-medium", isGreen && "text-emerald-600 dark:text-emerald-400")}>
                      {m.format(val)}
                    </span>
                  </div>
                )
              })}
            </div>
            {session.skills.length > 0 && (
              <div className="mt-3 pt-2 border-t">
                <div className="text-xs text-muted-foreground mb-1">Skills ({session.skills.length})</div>
                <div className="flex flex-wrap gap-1">
                  {session.skills.map(s => (
                    <Badge key={s.skillName} variant="yellow" className="text-xs">
                      {s.skillName}{s.version ? ` v${s.version}` : ""}
                      {s.invocationCount > 0 ? ` (${s.invocationCount}x)` : ""}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  )
}
