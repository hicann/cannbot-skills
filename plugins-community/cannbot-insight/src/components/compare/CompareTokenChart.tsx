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
  if (n === 0) return "-"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

const COLOR_A = "#3b82f6"
const COLOR_B = "#f97316"

const TOKEN_SEGMENTS: Array<{ key: keyof TokenData; label: string }> = [
  { key: "inputTokens", label: "Input" },
  { key: "outputTokens", label: "Output" },
  { key: "reasoningTokens", label: "Reasoning" },
  { key: "cacheReadTokens", label: "Cache Read" },
  { key: "cacheWriteTokens", label: "Cache Write" },
]

export function CompareTokenChart({ tokenA, tokenB }: { tokenA: TokenData; tokenB: TokenData }) {
  const globalMax = Math.max(
    ...TOKEN_SEGMENTS.map(s => Math.max(tokenA[s.key], tokenB[s.key])),
    1
  )

  function blendedPct(value: number, rowMax: number): number {
    const globalPct = (value / (globalMax * 1.25)) * 100
    const rowPct = (value / (rowMax * 1.25)) * 100
    return Math.min(globalPct * 0.2 + rowPct * 0.8, 82)
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Token 对比</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {TOKEN_SEGMENTS.map((seg) => {
            const valA = tokenA[seg.key]
            const valB = tokenB[seg.key]
            const rowMax = Math.max(valA, valB, 1)
            const pctA = valA === 0 ? 0 : blendedPct(valA, rowMax)
            const pctB = valB === 0 ? 0 : blendedPct(valB, rowMax)
            const zeroA = valA === 0
            const zeroB = valB === 0
            const bigA = pctA > 25
            const bigB = pctB > 25

            return (
              <div key={seg.key} className="flex items-center">
                <div className="w-[90px] shrink-0 text-sm font-medium text-muted-foreground text-right pr-3">
                  {seg.label}
                </div>

                <div className="flex-1 min-w-0 flex">
                  <div className="flex-1 flex justify-end items-center">
                    {zeroA ? (
                      <div
                        className="h-6 rounded-l-md flex items-center justify-center text-xs font-semibold"
                        style={{ width: "24px", backgroundColor: COLOR_A, opacity: 0.18, color: "#9ca3af" }}
                      >
                        -
                      </div>
                    ) : bigA ? (
                      <div
                        className="h-6 rounded-l-md flex items-center justify-end pr-2 text-xs font-semibold"
                        style={{ width: `${pctA}%`, backgroundColor: COLOR_A, opacity: 0.85, color: "#fff", minWidth: "40px" }}
                      >
                        <span className="truncate">{formatTokenCount(valA)}</span>
                      </div>
                    ) : (
                      <>
                        <span className="text-xs font-semibold tabular-nums mr-2" style={{ color: "#374151" }}>
                          {formatTokenCount(valA)}
                        </span>
                        <div
                          className="h-6 rounded-l-md shrink-0"
                          style={{ width: `${pctA}%`, backgroundColor: COLOR_A, opacity: 0.85, minWidth: "4px" }}
                        />
                      </>
                    )}
                  </div>

                  <div className="shrink-0 w-[2px] bg-border self-stretch" />

                  <div className="flex-1 flex justify-start items-center">
                    {zeroB ? (
                      <div
                        className="h-6 rounded-r-md flex items-center justify-center text-xs font-semibold"
                        style={{ width: "24px", backgroundColor: COLOR_B, opacity: 0.18, color: "#9ca3af" }}
                      >
                        -
                      </div>
                    ) : bigB ? (
                      <div
                        className="h-6 rounded-r-md flex items-center justify-start pl-2 text-xs font-semibold"
                        style={{ width: `${pctB}%`, backgroundColor: COLOR_B, opacity: 0.85, color: "#fff", minWidth: "40px" }}
                      >
                        <span className="truncate">{formatTokenCount(valB)}</span>
                      </div>
                    ) : (
                      <>
                        <div
                          className="h-6 rounded-r-md shrink-0"
                          style={{ width: `${pctB}%`, backgroundColor: COLOR_B, opacity: 0.85, minWidth: "4px" }}
                        />
                        <span className="text-xs font-semibold tabular-nums ml-2" style={{ color: "#374151" }}>
                          {formatTokenCount(valB)}
                        </span>
                      </>
                    )}
                  </div>
                </div>
              </div>
            )
          })}
        </div>

        <div className="flex items-center gap-6 mt-3 text-xs text-muted-foreground">
          <span className="flex items-center gap-2">
            <span className="inline-block w-4 h-2.5 rounded" style={{ backgroundColor: COLOR_A, opacity: 0.85 }} /> Session A
          </span>
          <span className="flex items-center gap-2">
            <span className="inline-block w-4 h-2.5 rounded" style={{ backgroundColor: COLOR_B, opacity: 0.85 }} /> Session B
          </span>
        </div>
      </CardContent>
    </Card>
  )
}
