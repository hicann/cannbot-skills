"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface TokenData {
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

const TOKEN_SEGMENTS: Array<{ key: keyof TokenData; label: string; colorA: string; colorB: string }> = [
  { key: "inputTokens", label: "Input", colorA: "#3b82f6", colorB: "#f97316" },
  { key: "outputTokens", label: "Output", colorA: "#10b981", colorB: "#fb923c" },
  { key: "reasoningTokens", label: "Reasoning", colorA: "#8b5cf6", colorB: "#c084fc" },
  { key: "cacheReadTokens", label: "Cache Read", colorA: "#eab308", colorB: "#fbbf24" },
  { key: "cacheWriteTokens", label: "Cache Write", colorA: "#6366f1", colorB: "#a855f7" },
]

export function CompareTokenChart({ tokenA, tokenB }: { tokenA: TokenData; tokenB: TokenData }) {
  const SVG_W = 600
  const SVG_H = 300
  const PAD_L = 80
  const PAD_R = 60
  const PAD_T = 20
  const PAD_B = 40
  const GROUP_GAP = 24

  const plotW = SVG_W - PAD_L - PAD_R
  const plotH = SVG_H - PAD_T - PAD_B
  const groupH = (plotH - GROUP_GAP * (TOKEN_SEGMENTS.length - 1)) / TOKEN_SEGMENTS.length

  const maxVal = Math.max(
    ...TOKEN_SEGMENTS.map(s => Math.max(tokenA[s.key], tokenB[s.key])),
    1
  )

  function barHeight(value: number): number {
    return (value / maxVal) * groupH
  }

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Token 对比</CardTitle>
      </CardHeader>
      <CardContent>
        <svg
          width="100%"
          height={SVG_H}
          viewBox={`0 0 ${SVG_W} ${SVG_H}`}
          preserveAspectRatio="xMidYMid meet"
          className="min-w-[400px]"
        >
          {TOKEN_SEGMENTS.map((seg, i) => {
            const groupY = PAD_T + i * (groupH + GROUP_GAP)
            const groupBottom = groupY + groupH
            const valA = tokenA[seg.key]
            const valB = tokenB[seg.key]
            const hA = barHeight(valA)
            const hB = barHeight(valB)
            const barW = plotW / 2 - 4

            return (
              <g key={seg.key}>
                <text x={4} y={groupY + groupH / 2 + 3} fontSize={10} fill="#9ca3af">{seg.label}</text>
                <line x1={PAD_L} y1={groupBottom} x2={SVG_W - PAD_R} y2={groupBottom} stroke="#374151" strokeWidth={0.5} />

                <rect x={PAD_L} y={groupBottom - hA} width={barW} height={Math.max(hA, 1)} fill={seg.colorA} rx={2} opacity={0.85} />
                <text x={PAD_L + barW - 4} y={groupBottom - hA + hA / 2 + 3} fontSize={9} fill={seg.colorA} textAnchor="end">{formatTokenCount(valA)}</text>

                <rect x={PAD_L + plotW / 2 + 4} y={groupBottom - hB} width={barW} height={Math.max(hB, 1)} fill={seg.colorB} rx={2} opacity={0.85} />
                <text x={PAD_L + plotW / 2 + 4 + barW - 4} y={groupBottom - hB + hB / 2 + 3} fontSize={9} fill={seg.colorB} textAnchor="end">{formatTokenCount(valB)}</text>
              </g>
            )
          })}
        </svg>

        <div className="flex items-center gap-4 mt-2 text-xs text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-blue-500" /> Session A
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-3 h-3 rounded bg-orange-500" /> Session B
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
