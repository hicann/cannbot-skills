"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"

interface AgentItem {
  executionId: string
  agentName: string | null
  agentSessionId: string | null
  isSubagent: boolean
  parentExecutionId: string | null
  tokens: number
  cost: number
  toolCallCount: number
  skillLoadCount: number
  model: string | null
  createdAt: string
  latencyMs: number
  firstPrompt: string | null
}

interface AgentTimelineProps {
  agents: AgentItem[]
  onViewTurns?: (agentSessionId: string | null) => void
}

const ROW_HEIGHT_PX = 44
const BAR_HEIGHT_PX = 12
const MAX_VISIBLE_ROWS = 12
const LABEL_WIDTH_PX = 140
const COLOR_LOW = [96, 165, 250]
const COLOR_HIGH = [59, 130, 246]
const MAX_TICKS = 6

function formatLatency(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatRelativeTime(ms: number): string {
  if (ms === 0) return "0s"
  if (ms < 60000) return `+${Math.round(ms / 1000)}s`
  if (ms < 3600000) return `+${Math.round(ms / 60000)}min`
  return `+${(ms / 3600000).toFixed(1)}hr`
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

function formatCost(cost: number): string {
  if (cost === 0) return "$0.00"
  if (cost < 0.01) return `$${cost.toFixed(4)}`
  return `$${cost.toFixed(2)}`
}

function formatTimeAbsolute(iso: string): string {
  const d = new Date(iso)
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`
}

function computeTickInterval(totalMs: number): number {
  const candidates = [1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000, 600000, 1200000, 1800000, 3600000]
  for (const c of candidates) {
    if (totalMs / c <= MAX_TICKS) return c
  }
  return Math.ceil(totalMs / MAX_TICKS / 3600000) * 3600000
}

interface TooltipData {
  x: number
  y: number
  label: string
  tokens: number
  cost: number
  latencyMs: number
  startTime: string
  endTime: string
  toolCallCount: number
  model: string | null
}

export function AgentTimeline({ agents, onViewTurns }: AgentTimelineProps) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)

  const sorted = useMemo(() =>
    [...agents].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()),
    [agents]
  )

  const globalStartMs = useMemo(() => {
    if (agents.length === 0) return 0
    return Math.min(...agents.map(a => new Date(a.createdAt).getTime()))
  }, [agents])

  const globalEndMs = useMemo(() => {
    if (agents.length === 0) return 1
    return Math.max(...agents.map(a => new Date(a.createdAt).getTime() + a.latencyMs))
  }, [agents])

  const totalDuration = globalEndMs - globalStartMs || 1

  const maxTokens = useMemo(() =>
    Math.max(1, ...agents.map(a => a.tokens)),
    [agents]
  )

  const minTokens = useMemo(() =>
    Math.min(...agents.map(a => a.tokens)),
    [agents]
  )

  const tickInterval = computeTickInterval(totalDuration)
  const ticks = useMemo(() => {
    const result: Array<{ ms: number; pct: number; relative: string; absolute: string }> = []
    for (let ms = 0; ms <= totalDuration; ms += tickInterval) {
      const pct = (ms / totalDuration) * 100
      const absTime = new Date(globalStartMs + ms).toISOString()
      result.push({ ms, pct, relative: formatRelativeTime(ms), absolute: formatTimeAbsolute(absTime) })
    }
    return result
  }, [totalDuration, tickInterval, globalStartMs])

  function getBarColor(tokens: number): string {
    const t = Math.log(tokens + 1) / Math.log(maxTokens + 1)
    const r = Math.round(COLOR_LOW[0] + (COLOR_HIGH[0] - COLOR_LOW[0]) * t)
    const g = Math.round(COLOR_LOW[1] + (COLOR_HIGH[1] - COLOR_LOW[1]) * t)
    const b = Math.round(COLOR_LOW[2] + (COLOR_HIGH[2] - COLOR_LOW[2]) * t)
    return `rgb(${r},${g},${b})`
  }

  function getLeftPct(createdAt: string): number {
    const startMs = new Date(createdAt).getTime()
    return Math.max(0, Math.min(((startMs - globalStartMs) / totalDuration) * 100, 98))
  }

  function getWidthPct(latencyMs: number): number {
    if (totalDuration === 0) return 5
    return Math.max(2, (latencyMs / totalDuration) * 100)
  }

  const handleMouseEnter = useCallback((e: React.MouseEvent, agent: AgentItem) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    setTooltip({
      x: rect.left + rect.width / 2,
      y: rect.top,
      label: agent.agentName ?? (agent.isSubagent ? "subagent" : "root"),
      tokens: agent.tokens,
      cost: agent.cost,
      latencyMs: agent.latencyMs,
      startTime: formatTimeAbsolute(agent.createdAt),
      endTime: formatTimeAbsolute(new Date(new Date(agent.createdAt).getTime() + agent.latencyMs).toISOString()),
      toolCallCount: agent.toolCallCount,
      model: agent.model,
    })
  }, [])

  const handleMouseLeave = useCallback(() => {
    setTooltip(null)
  }, [])

  if (agents.length === 0) {
    return (
      <Card size="sm">
        <CardHeader>
          <CardTitle>Agents (0)</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-sm text-muted-foreground">No agents found</div>
        </CardContent>
      </Card>
    )
  }

  const bodyMaxHeight = MAX_VISIBLE_ROWS * ROW_HEIGHT_PX
  const needsScroll = sorted.length > MAX_VISIBLE_ROWS

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Agents ({agents.length})</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="relative" style={{ marginLeft: LABEL_WIDTH_PX }}>
          <div className="relative" style={{ height: 36 }}>
            {ticks.map((t, i) => (
              <div
                key={i}
                className="absolute text-[10px] text-muted-foreground tabular-nums whitespace-nowrap"
                style={{ left: `${t.pct}%`, transform: "translateX(-50%)", top: 0 }}
              >
                {t.relative}
              </div>
            ))}
            {ticks.map((t, i) => (
              <div
                key={`abs-${i}`}
                className="absolute text-[10px] text-muted-foreground/60 tabular-nums whitespace-nowrap"
                style={{ left: `${t.pct}%`, transform: "translateX(-50%)", top: "16px" }}
              >
                {t.absolute}
              </div>
            ))}
          </div>
          <div className="relative h-px bg-border" />
        </div>

        <div
          className={needsScroll ? "overflow-y-auto" : ""}
          style={needsScroll ? { maxHeight: bodyMaxHeight } : undefined}
        >
          {sorted.map((agent, idx) => {
            const leftPct = getLeftPct(agent.createdAt)
            const widthPct = getWidthPct(agent.latencyMs)
            const barColor = getBarColor(agent.tokens)
            const label = agent.agentName ?? (agent.isSubagent ? "subagent" : "root")
            const isHovered = tooltip?.label === label && tooltip?.startTime === formatTimeAbsolute(agent.createdAt)

            return (
              <div
                key={agent.executionId}
                className="flex items-center"
                style={{
                  height: ROW_HEIGHT_PX,
                  backgroundColor: idx % 2 === 0 ? "transparent" : "rgba(0,0,0,0.06)",
                }}
              >
                <div
                  className="shrink-0 pr-2 text-left cursor-pointer hover:opacity-70"
                  style={{ width: LABEL_WIDTH_PX }}
                  title={`${label}${agent.firstPrompt ? `\n${agent.firstPrompt}` : ""}`}
                  onClick={() => onViewTurns?.(agent.isSubagent ? agent.agentSessionId : null)}
                >
                  <div className="font-semibold text-foreground truncate" style={{ fontSize: "14px", lineHeight: "18px" }}>{label}</div>
                  {agent.firstPrompt && (
                    <div className="text-muted-foreground truncate" style={{ fontSize: "11px", lineHeight: "14px", fontWeight: "normal" }}>{agent.firstPrompt}</div>
                  )}
                </div>
                <div className="flex-1 relative" style={{ height: ROW_HEIGHT_PX }}>
                  {ticks.filter(t => t.pct > 0 && t.pct < 100).map((t, i) => (
                    <div
                      key={i}
                      className="absolute top-0 h-full w-px bg-border/20"
                      style={{ left: `${t.pct}%` }}
                    />
                  ))}
                  <div
                    className="absolute rounded-full"
                    style={{
                      left: `${leftPct}%`,
                      width: `${widthPct}%`,
                      height: BAR_HEIGHT_PX,
                      top: (ROW_HEIGHT_PX - BAR_HEIGHT_PX) / 2,
                      backgroundColor: barColor,
                      filter: isHovered ? "brightness(1.3)" : "none",
                      transition: "filter 0.15s",
                    }}
                    onMouseEnter={(e) => handleMouseEnter(e, agent)}
                    onMouseLeave={handleMouseLeave}
                  />
                </div>
              </div>
            )
          })}
        </div>

        <div className="pt-2" style={{ marginLeft: LABEL_WIDTH_PX }}>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-muted-foreground tabular-nums">{formatTokenCount(minTokens)}</span>
            <div
              className="flex-1 h-3 rounded-full"
              style={{ background: `linear-gradient(to right, rgb(${COLOR_LOW.join(",")}), rgb(${COLOR_HIGH.join(",")}))` }}
            />
            <span className="text-[10px] text-muted-foreground tabular-nums">{formatTokenCount(maxTokens)}</span>
          </div>
          <div className="text-center text-[10px] text-muted-foreground mt-0.5">tokens (log scale)</div>
        </div>
      </CardContent>

      {tooltip && (
        <div
          className="fixed z-[9999] bg-popover border rounded-md shadow-lg px-3 py-2 text-xs whitespace-nowrap pointer-events-none"
          style={{
            left: tooltip.x,
            top: tooltip.y - 8,
            transform: "translate(-50%, -100%)",
          }}
        >
          <div className="font-medium mb-1">{tooltip.label}</div>
          <div className="text-muted-foreground space-y-0.5">
            <div>{formatTokenCount(tooltip.tokens)} tokens · {formatCost(tooltip.cost)}</div>
            <div>{formatLatency(tooltip.latencyMs)} · {tooltip.startTime} → {tooltip.endTime}</div>
            <div>{tooltip.toolCallCount} tools{tooltip.model ? ` · ${tooltip.model}` : ""}</div>
          </div>
        </div>
      )}
    </Card>
  )
}
