"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useMemo } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { cn } from "@/lib/utils"

interface BridgeItem {
  bridgeId: string
  dispatchExecutionId: string
  dispatchContent: string | null
  dispatchTimestamp: string | null
  responseExecutionId: string | null
  responseContent: string | null
  responseTimestamp: string | null
  subagentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface InteractionGraphProps {
  bridges: BridgeItem[]
  rootAgentName: string | null
  sessionStartTime: string
  sessionLatencyMs: number
}

const STATUS_BADGE: Record<string, "green" | "red" | "orange" | "gray"> = {
  completed: "green",
  failed: "red",
  running: "orange",
  dispatched: "gray",
  timeout: "orange",
}

const STATUS_ICON: Record<string, string> = {
  completed: "✅",
  failed: "❌",
  running: "⏳",
  dispatched: "📤",
  timeout: "⚠️",
}

const TYPE_PALETTE: Record<string, { bg: string; border: string; text: string; bar: string }> = {
  ascendc_kernel_architect: { bg: "bg-emerald-50 dark:bg-emerald-500/10", border: "border-emerald-300 dark:border-emerald-600", text: "text-emerald-700 dark:text-emerald-400", bar: "bg-emerald-500" },
  ascendc_kernel_developer: { bg: "bg-violet-50 dark:bg-violet-500/10", border: "border-violet-300 dark:border-violet-600", text: "text-violet-700 dark:text-violet-400", bar: "bg-violet-500" },
  ascendc_ops_tester: { bg: "bg-amber-50 dark:bg-amber-500/10", border: "border-amber-300 dark:border-amber-600", text: "text-amber-700 dark:text-amber-400", bar: "bg-amber-500" },
  general: { bg: "bg-gray-50 dark:bg-gray-500/10", border: "border-gray-300 dark:border-gray-600", text: "text-gray-700 dark:text-gray-400", bar: "bg-gray-400" },
}

function getTypeColor(type: string | null) {
  const key = type?.toLowerCase().replace(/-/g, "_") ?? "general"
  return TYPE_PALETTE[key] ?? TYPE_PALETTE.general
}

function formatLatency(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 6000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return ""
  try {
    const d = new Date(ts)
    const month = String(d.getMonth() + 1).padStart(2, "0")
    const day = String(d.getDate()).padStart(2, "0")
    const hour = String(d.getHours()).padStart(2, "0")
    const minute = String(d.getMinutes()).padStart(2, "0")
    return `${month}/${day} ${hour}:${minute}`
  } catch {
    return ts
  }
}

function formatTimeShort(ts: string | null): string {
  if (!ts) return ""
  try {
    const d = new Date(ts)
    return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`
  } catch {
    return ""
  }
}

function ContentBlock({ label, content, timestamp }: { label: string; content: string | null; timestamp: string | null }) {
  const [expanded, setExpanded] = useState(false)
  const text = content ?? "No content"

  return (
    <div>
      <div className="text-xs font-medium mb-1 text-muted-foreground">{label}</div>
      <div
        className={cn(
          "text-xs bg-muted/30 p-2 rounded whitespace-pre-wrap break-words",
          expanded ? "max-h-[300px] overflow-y-auto" : "line-clamp-3"
        )}
      >
        {text}
      </div>
      {(text.length > 150 || text.includes("\n")) && (
        <button
          className="text-xs text-primary hover:underline mt-1 cursor-pointer"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Collapse" : "Expand"}
        </button>
      )}
      {timestamp && (
        <div className="text-xs text-muted-foreground mt-1">{formatTimestamp(timestamp)}</div>
      )}
    </div>
  )
}

export function InteractionGraph({ bridges, rootAgentName, sessionStartTime, sessionLatencyMs }: InteractionGraphProps) {
  const [selectedBridge, setSelectedBridge] = useState<string | null>(null)

  if (bridges.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No interactions found
      </div>
    )
  }

  const sessionStartMs = new Date(sessionStartTime).getTime()
  const totalDuration = sessionLatencyMs || 1
  const parentName = rootAgentName === "build" ? "root agent" : (rootAgentName ?? "root agent")
  const totalTokens = bridges.reduce((s, b) => s + b.subagentTokens, 0)

  const typeCounts = useMemo(() => {
    const counts = new Map<string, number>()
    for (const b of bridges) {
      const key = b.subagentType ?? "general"
      counts.set(key, (counts.get(key) ?? 0) + 1)
    }
    return counts
  }, [bridges])

  const sortedBridges = useMemo(() =>
    [...bridges].sort((a, b) => {
      const aTime = a.dispatchTimestamp ? new Date(a.dispatchTimestamp).getTime() : 0
      const bTime = b.dispatchTimestamp ? new Date(b.dispatchTimestamp).getTime() : 0
      return aTime - bTime
    }),
  [bridges])

  function getStartPct(bridge: BridgeItem): number {
    if (!bridge.dispatchTimestamp) return 0
    const ms = new Date(bridge.dispatchTimestamp).getTime() - sessionStartMs
    return Math.max(0, Math.min(ms / totalDuration * 100, 100))
  }

  function getWidthPct(bridge: BridgeItem): number {
    return Math.max(2, bridge.subagentLatencyMs / totalDuration * 100)
  }

  const tickPcts = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center gap-3 mb-2">
        <Badge variant="outline">Interactions</Badge>
        <span className="text-xs text-muted-foreground">
          {bridges.length} dispatches | {formatLatency(totalDuration)} total | {formatTokenCount(totalTokens)} tok
        </span>
        <div className="flex items-center gap-1.5 ml-auto">
          {Object.entries(TYPE_PALETTE).map(([key, color]) => {
            const count = typeCounts.get(key) ?? 0
            if (count === 0) return null
            return (
              <div key={key} className="flex items-center gap-1 text-xs">
                <div className={cn("w-2.5 h-2.5 rounded", color.bar)} />
                <span className={color.text}>{key.replace(/_/g, "-")} ({count})</span>
              </div>
            )
          })}
        </div>
      </div>

      <div className="relative h-1.5 bg-muted rounded-full mb-0.5">
        {tickPcts.filter(p => p > 0 && p < 1).map(pct => (
          <div key={pct} className="absolute top-0 h-full w-px bg-border" style={{ left: `${pct * 100}%` }} />
        ))}
      </div>
      <div className="flex justify-between text-xs text-muted-foreground tabular-nums">
        {tickPcts.map(pct => {
          const ms = totalDuration * pct
          const label = formatTimeShort(new Date(sessionStartMs + ms).toISOString())
          return <span key={pct}>{label}</span>
        })}
      </div>

      <div className="space-y-2">
        {sortedBridges.map((bridge, idx) => {
          const colors = getTypeColor(bridge.subagentType)
          const isSelected = selectedBridge === bridge.bridgeId
          const childName = bridge.subagentName ?? bridge.subagentType ?? "subagent"
          const dispatch = bridge.dispatchContent ?? "—"
          const timePct = Math.round(bridge.subagentLatencyMs / totalDuration * 100)
          const startPct = getStartPct(bridge)
          const widthPct = getWidthPct(bridge)

          return (
            <div key={bridge.bridgeId}>
              <Card
                className={cn(
                  "cursor-pointer transition-all",
                  colors.border,
                  isSelected ? "ring-2 ring-primary/50" : "hover:ring-1 hover:ring-primary/30"
                )}
                onClick={() => setSelectedBridge(isSelected ? null : bridge.bridgeId)}
              >
                <CardContent className="py-2.5 px-3">
                  <div className="flex items-start gap-2 text-xs">
                    <div className="shrink-0">
                      <div className="text-muted-foreground">父 agent：</div>
                      <div className="text-muted-foreground mt-0.5">子 agent：</div>
                    </div>
                    <div>
                      <div><Badge variant="blue">{parentName}</Badge></div>
                      <div className="mt-0.5"><Badge className={cn(colors.bg, colors.border, colors.text)}>{childName}</Badge>
                      {bridge.subagentType && bridge.subagentType.replace(/_/g, "-") !== childName && <Badge variant="outline">{bridge.subagentType.replace(/_/g, "-")}</Badge>}</div>
                    </div>
                    <div className="ml-auto flex items-center gap-2">
                      <Badge variant={STATUS_BADGE[bridge.status] ?? "gray"}>{STATUS_ICON[bridge.status] ?? "?"} {bridge.status}</Badge>
                    </div>
                  </div>
                  <div className="text-xs font-medium text-foreground/90 mt-1">
                    指令：{dispatch}
                  </div>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                    <span>#{idx + 1}</span>
                    <span>{formatLatency(bridge.subagentLatencyMs)}</span>
                    <span className="font-medium">{timePct}%</span>
                    <span>{formatTokenCount(bridge.subagentTokens)} tok</span>
                    {bridge.dispatchTimestamp && <span>{formatTimeShort(bridge.dispatchTimestamp)}</span>}
                    {bridge.responseTimestamp && <span>→ {formatTimeShort(bridge.responseTimestamp)}</span>}
                  </div>

                  <div className="relative h-2 bg-muted/50 rounded-full mt-2 overflow-hidden">
                    {tickPcts.filter(p => p > 0 && p < 1).map(pct => (
                      <div key={pct} className="absolute top-0 h-full w-px bg-border/50" style={{ left: `${pct * 100}%` }} />
                    ))}
                    <div
                      className={cn("absolute top-0 h-full rounded-full", colors.bar)}
                      style={{ left: `${startPct}%`, width: `${widthPct}%` }}
                    />
                  </div>
                </CardContent>
              </Card>

              {isSelected && (
                <Card className="mt-2">
                  <CardContent className="p-4 space-y-3">
                    <ContentBlock
                      label="Dispatch (任务描述)"
                      content={bridge.dispatchContent}
                      timestamp={bridge.dispatchTimestamp}
                    />
                    <ContentBlock
                      label="Response (返回结果)"
                      content={bridge.responseContent}
                      timestamp={bridge.responseTimestamp}
                    />
                    <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                      <span className="text-muted-foreground">Tokens</span>
                      <span className="tabular-nums">{formatTokenCount(bridge.subagentTokens)}</span>
                      <span className="text-muted-foreground">Duration</span>
                      <span className="tabular-nums">{formatLatency(bridge.subagentLatencyMs)} ({timePct}%)</span>
                      <span className="text-muted-foreground">Session ID</span>
                      <span className="truncate">{bridge.subagentSessionId ?? "N/A"}</span>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>
          )
        })}
      </div>
    </div>
  )
}
