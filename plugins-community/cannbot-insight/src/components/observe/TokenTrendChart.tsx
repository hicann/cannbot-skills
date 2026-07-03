"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"
import { cn } from "@/lib/utils"

interface TokenTrendChartProps {
  turns: Array<{
    turnIndex: number
    role: string
    totalTokens: number
    inputTokens: number
    outputTokens: number
  }>
  defaultShow?: number
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

const ROLE_COLORS: Record<string, { bar: string; label: string }> = {
  user: { bar: "bg-blue-400 dark:bg-blue-500", label: "text-blue-600 dark:text-blue-400" },
  assistant: { bar: "bg-emerald-400 dark:bg-emerald-500", label: "text-emerald-600 dark:text-emerald-400" },
  system: { bar: "bg-gray-400 dark:bg-gray-500", label: "text-gray-600 dark:text-gray-400" },
  tool_result: { bar: "bg-teal-400 dark:bg-teal-500", label: "text-teal-600 dark:text-teal-400" },
}

export function TokenTrendChart({ turns, defaultShow = 10 }: TokenTrendChartProps) {
  const turnsWithTokens = turns.filter(t => t.totalTokens > 0)
  if (turnsWithTokens.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground text-xs">
        No token data
      </div>
    )
  }

  const maxTokens = Math.max(...turnsWithTokens.map(t => t.totalTokens))
  const [showCount, setShowCount] = useState(defaultShow)
  const visibleTurns = turnsWithTokens.slice(0, showCount)
  const hasMore = showCount < turnsWithTokens.length

  return (
    <div className="space-y-0.5">
      {visibleTurns.map(t => {
        const pct = (t.totalTokens / maxTokens) * 100
        const colors = ROLE_COLORS[t.role] ?? ROLE_COLORS.system
        return (
          <div key={t.turnIndex} className="flex items-center gap-2 group">
            <span className="w-8 text-xs text-muted-foreground tabular-nums text-right">#{t.turnIndex}</span>
            <div className="flex-1 h-4 bg-muted/30 rounded relative overflow-hidden">
              <div
                className={cn("h-full rounded transition-all", colors.bar)}
                style={{ width: `${Math.max(pct, 2)}%` }}
              />
            </div>
            <span className={cn("w-20 text-xs tabular-nums text-right", colors.label)}>
              {formatTokenCount(t.totalTokens)}
            </span>
          </div>
        )
      })}
      {hasMore && (
        <button
          className="text-xs text-muted-foreground hover:text-primary transition-colors cursor-pointer w-full py-1"
          onClick={() => setShowCount(turnsWithTokens.length)}
        >
          ▼ 展示全部 {turnsWithTokens.length} 个 turns
        </button>
      )}
      <div className="flex items-center justify-between pt-2 border-t mt-1 text-xs text-muted-foreground">
        <span>Total turns: {turnsWithTokens.length}</span>
        <span>Total: {formatTokenCount(turnsWithTokens.reduce((sum, t) => sum + t.totalTokens, 0))}</span>
      </div>
    </div>
  )
}
