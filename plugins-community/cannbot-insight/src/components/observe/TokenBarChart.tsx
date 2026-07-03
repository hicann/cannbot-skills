"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { cn } from "@/lib/utils"

interface TokenBarChartProps {
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  toolOverheadTokens?: number
  contextWindowLimit?: number
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

export function TokenBarChart({
  totalTokens,
  inputTokens,
  outputTokens,
  reasoningTokens,
  cacheReadTokens,
  cacheWriteTokens,
  toolOverheadTokens,
  contextWindowLimit = 200000,
}: TokenBarChartProps) {
  if (totalTokens === 0) {
    return <div className="text-sm text-muted-foreground">No token data</div>
  }

  const totalInput = inputTokens + cacheReadTokens + cacheWriteTokens
  const totalOutput = outputTokens + reasoningTokens
  const toolTokens = toolOverheadTokens ?? 0

  const segments = [
    { label: "Input", value: totalInput, color: "bg-blue-500", pct: ((totalInput / contextWindowLimit) * 100).toFixed(1) },
    { label: "Output", value: totalOutput, color: "bg-emerald-500", pct: ((totalOutput / contextWindowLimit) * 100).toFixed(1) },
    ...(toolTokens > 0 ? [{ label: "Tool Calls", value: toolTokens, color: "bg-orange-500", pct: ((toolTokens / contextWindowLimit) * 100).toFixed(1) }] : []),
  ].filter(s => s.value > 0)

  // Stacked bar: Input (blue) + Output (green) + Tool Calls (orange)
  const barSegments = [
    { label: "Input", value: totalInput, color: "bg-blue-500" },
    { label: "Output", value: totalOutput, color: "bg-emerald-500" },
    ...(toolTokens > 0 ? [{ label: "Tool Calls", value: toolTokens, color: "bg-orange-500" }] : []),
  ].filter(s => s.value > 0)

  return (
    <div className="space-y-3">
      <div className="flex gap-0.5 h-3 rounded-full overflow-hidden bg-muted">
        {/* Full bar = contextWindowLimit, segments fill proportionally */}
        {barSegments.map(segment => (
          <div
            key={segment.label}
            className={cn("h-full transition-all", segment.color)}
            style={{
              width: `${(segment.value / contextWindowLimit) * 100}%`,
              minWidth: segment.value > 0 ? "1px" : "0",
            }}
          />
        ))}
      </div>

      <div className="space-y-1.5">
        {segments.map(segment => {
          const pctWidth = (segment.value / contextWindowLimit) * 100

          return (
            <div key={segment.label} className="flex items-center gap-3 text-sm">
              <span className="w-24 text-muted-foreground text-xs">{segment.label}</span>
              <div className="flex-1 h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn("h-full rounded-full transition-all", segment.color)}
                  style={{ width: `${Math.min(pctWidth * 5, 100)}%` }}
                />
              </div>
              <span className="w-24 text-xs text-right tabular-nums">
                {formatTokenCount(segment.value)} <span className="text-muted-foreground">{segment.pct}%</span>
              </span>
            </div>
          )
        })}
      </div>

      <div className="flex items-center gap-3 text-sm pt-1 border-t">
        <span className="w-24 text-xs font-medium">Total</span>
        <span className="text-xs tabular-nums">{formatTokenCount(totalTokens)} <span className="text-muted-foreground">{((totalTokens / contextWindowLimit) * 100).toFixed(1)}%</span></span>
      </div>
    </div>
  )
}
