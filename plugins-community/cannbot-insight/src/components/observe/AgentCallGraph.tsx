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
  maxSingleCallTokens: number
  cost: number
  toolCallCount: number
  skillLoadCount: number
  model: string | null
  createdAt: string
  latencyMs: number
  firstPrompt: string | null
}

interface BridgeItem {
  bridgeId: string
  dispatchExecutionId: string
  dispatchTurnId: string | null
  dispatchToolCallId: string | null
  dispatchContent: string | null
  dispatchTimestamp: string | null
  responseExecutionId: string | null
  responseTurnId: string | null
  responseContent: string | null
  responseTimestamp: string | null
  subagentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  agentName: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface AgentCallGraphProps {
  agents: AgentItem[]
  bridges: BridgeItem[]
  onViewTurns?: (agentSessionId: string | null) => void
}

const LANE_LABEL_WIDTH = 109
const SUB_BAR_H = 12
const SUB_GAP = 3
const MAX_VISIBLE_HEIGHT = 300

const AGENT_COLORS = [
  "#3b82f6", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6",
  "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
  "#14b8a6", "#e11d48", "#a855f7", "#22c55e", "#d946ef",
]

const agentColorCache = new Map<string, string>()
let agentColorIdx = 0

function getAgentColor(agentName: string | null): string {
  const key = agentName ?? "root"
  if (agentColorCache.has(key)) return agentColorCache.get(key)!
  agentColorCache.set(key, AGENT_COLORS[agentColorIdx % AGENT_COLORS.length])
  agentColorIdx++
  return agentColorCache.get(key)!
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatTokenCount(n: number): string {
  if (n === 0) return ""
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

function formatTimeAbsolute(iso: string): string {
  const d = new Date(iso)
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}:${String(d.getSeconds()).padStart(2, "0")}`
}

function computeTickInterval(totalMs: number): number {
  const candidates = [1000, 2000, 5000, 10000, 15000, 30000, 60000, 120000, 300000, 600000]
  for (const c of candidates) {
    if (totalMs / c <= 6) return c
  }
  return Math.ceil(totalMs / 6 / 3600000) * 3600000
}

interface LaneDef {
  key: string
  label: string
  color: string
  bg: string
}

const LANE_PATTERNS: Array<{ pattern: RegExp; label: string; color: string; bg: string }> = [
  { pattern: /architect/i, label: "架构", color: "#10b981", bg: "#10b98120" },
  { pattern: /design/i, label: "设计", color: "#f59e0b", bg: "#f59e0b20" },
  { pattern: /review/i, label: "评审", color: "#8b5cf6", bg: "#8b5cf620" },
  { pattern: /test/i, label: "测试", color: "#ef4444", bg: "#ef444420" },
  { pattern: /general/i, label: "通用", color: "#06b6d4", bg: "#06b6d420" },
]

function assignToLane(agent: AgentItem): LaneDef {
  if (!agent.isSubagent) {
    return { key: "command", label: "总控", color: "#3b82f6", bg: "#3b82f620" }
  }
  const name = (agent.agentName ?? "").toLowerCase()
  for (const p of LANE_PATTERNS) {
    if (p.pattern.test(name)) {
      return { key: p.label, label: p.label, color: p.color, bg: p.bg }
    }
  }
  const shortName = agent.agentName ?? "unknown"
  const color = getAgentColor(agent.agentName)
  return { key: shortName, label: shortName, color, bg: `${color}20` }
}

interface TooltipData {
  x: number
  y: number
  label: string
  tokens: number
  maxSingleCallTokens: number
  latencyMs: number
  startTime: string
  endTime: string
  toolCallCount: number
  model: string | null
  lane: string
  status: string
}

export function AgentCallGraph({ agents, bridges, onViewTurns }: AgentCallGraphProps) {
  const [tooltip, setTooltip] = useState<TooltipData | null>(null)

  const sorted = useMemo(() =>
    [...agents].sort((a, b) => new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()),
    [agents]
  )

  const globalStartMs = useMemo(() =>
    agents.length === 0 ? 0 : Math.min(...agents.map(a => new Date(a.createdAt).getTime())),
    [agents]
  )

  const globalEndMs = useMemo(() =>
    agents.length === 0 ? 1 : Math.max(...agents.map(a => new Date(a.createdAt).getTime() + a.latencyMs)),
    [agents]
  )

  const totalDuration = globalEndMs - globalStartMs || 1
  const tickInterval = computeTickInterval(totalDuration)

  const ticks = useMemo(() => {
    const result: Array<{ ms: number; pct: number; label: string }> = []
    for (let ms = 0; ms <= totalDuration; ms += tickInterval) {
      result.push({ ms, pct: (ms / totalDuration) * 100, label: formatTimeAbsolute(new Date(globalStartMs + ms).toISOString()) })
    }
    return result
  }, [totalDuration, tickInterval, globalStartMs])

  const lanes = useMemo(() => {
    const laneMap = new Map<string, { def: LaneDef; agents: AgentItem[] }>()
    for (const agent of sorted) {
      const def = assignToLane(agent)
      const existing = laneMap.get(def.key)
      if (existing) existing.agents.push(agent)
      else laneMap.set(def.key, { def, agents: [agent] })
    }
    const entries = [...laneMap.entries()]
    entries.sort((a, b) => {
      if (a[0] === "command") return -1
      if (b[0] === "command") return 1
      return new Date(a[1].agents[0].createdAt).getTime() - new Date(b[1].agents[0].createdAt).getTime()
    })
    return entries.map(([key, { def, agents }]) => {
      const trackEnds: number[] = []
      const packed = agents.map(agent => {
        const startMs = new Date(agent.createdAt).getTime()
        const endMs = startMs + agent.latencyMs
        let ti = trackEnds.findIndex(e => e <= startMs)
        if (ti === -1) { ti = trackEnds.length; trackEnds.push(endMs) }
        else trackEnds[ti] = endMs
        return { agent, trackIdx: ti }
      })
      return { key, def, packed, trackCount: trackEnds.length }
    })
  }, [sorted])

  const bridgeBySessionId = useMemo(() => {
    const map = new Map<string, BridgeItem>()
    for (const b of bridges) {
      if (b.subagentSessionId) map.set(b.subagentSessionId, b)
    }
    return map
  }, [bridges])

  function getStartPct(createdAt: string): number {
    return Math.max(0, Math.min(((new Date(createdAt).getTime() - globalStartMs) / totalDuration) * 100, 98))
  }

  function getWidthPct(latencyMs: number): number {
    return totalDuration === 0 ? 5 : Math.max(2, (latencyMs / totalDuration) * 100)
  }

  const handleMouseEnter = useCallback((e: React.MouseEvent, agent: AgentItem, laneLabel: string) => {
    const rect = (e.currentTarget as HTMLElement).getBoundingClientRect()
    const bridge = agent.isSubagent && agent.agentSessionId ? bridgeBySessionId.get(agent.agentSessionId) : undefined
    const endMs = new Date(agent.createdAt).getTime() + agent.latencyMs
    setTooltip({
      x: rect.left + rect.width / 2,
      y: rect.top,
      label: agent.agentName ?? (agent.isSubagent ? "subagent" : "root"),
      tokens: agent.tokens,
      maxSingleCallTokens: agent.maxSingleCallTokens,
      latencyMs: agent.latencyMs,
      startTime: formatTimeAbsolute(agent.createdAt),
      endTime: formatTimeAbsolute(new Date(endMs).toISOString()),
      toolCallCount: agent.toolCallCount,
      model: agent.model,
      lane: laneLabel,
      status: bridge?.status ?? "completed",
    })
  }, [bridgeBySessionId])

  const handleMouseLeave = useCallback(() => setTooltip(null), [])

  if (agents.length === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Agent Swimlane (0)</CardTitle></CardHeader>
        <CardContent><div className="text-sm text-muted-foreground">No agents found</div></CardContent>
      </Card>
    )
  }

  return (
    <Card size="sm">
      <CardHeader><CardTitle>Agent Swimlane ({agents.length})</CardTitle></CardHeader>
      <CardContent className="space-y-2">
        <div className="flex">
          <div className="shrink-0" style={{ width: LANE_LABEL_WIDTH }} />
          <div className="flex-1 relative h-[20px]">
            {ticks.map((t, i) => (
              <div key={i} className="absolute text-[10px] text-muted-foreground tabular-nums" style={{ left: `${t.pct}%`, transform: "translateX(-50%)" }}>
                {t.label}
              </div>
            ))}
          </div>
        </div>
        <div className="h-px bg-border" />

        <div className="overflow-y-auto" style={{ maxHeight: MAX_VISIBLE_HEIGHT }}>
          {lanes.map(({ key, def, packed, trackCount }) => {
            const laneHeight = Math.max(22, trackCount * SUB_BAR_H + (trackCount + 1) * SUB_GAP)
            return (
              <div key={key} className="flex border-b last:border-b-0">
                <div
                  className="shrink-0 flex flex-col items-center justify-center text-[10px] font-bold border-r-2 leading-tight px-1.5"
                  style={{ width: LANE_LABEL_WIDTH, height: laneHeight, backgroundColor: def.bg, borderColor: def.color, color: def.color }}
                >
                  <span className="break-all text-center">{def.label}</span>
                </div>

                <div className="flex-1 relative min-w-0" style={{ height: laneHeight }}>
                  {ticks.filter(t => t.pct > 0 && t.pct < 100).map((t, i) => (
                    <div key={i} className="absolute top-0 h-full w-px bg-border/20" style={{ left: `${t.pct}%` }} />
                  ))}
                  {packed.map(({ agent, trackIdx }) => {
                    const startPct = getStartPct(agent.createdAt)
                    const widthPct = getWidthPct(agent.latencyMs)
                    const agentColor = getAgentColor(agent.agentName)
                    const bridge = agent.isSubagent && agent.agentSessionId ? bridgeBySessionId.get(agent.agentSessionId) : undefined
                    const isError = bridge?.status === "error" || bridge?.status === "failed"
                    return (
                      <div
                        key={agent.executionId}
                        className="absolute rounded-sm cursor-pointer"
                        style={{
                          left: `${startPct}%`,
                          width: `${widthPct}%`,
                          minWidth: 4,
                          height: SUB_BAR_H,
                          top: SUB_GAP + trackIdx * (SUB_BAR_H + SUB_GAP),
                          backgroundColor: agentColor,
                          opacity: isError ? 0.4 : 0.8,
                          border: isError ? "2px solid #ef4444" : `1px solid ${agentColor}`,
                        }}
                        onClick={() => onViewTurns?.(agent.isSubagent ? agent.agentSessionId : null)}
                        onMouseEnter={(e) => handleMouseEnter(e, agent, def.label)}
                        onMouseLeave={handleMouseLeave}
                      />
                    )
                  })}
                </div>
              </div>
            )
          })}
        </div>

        <div className="flex flex-wrap gap-2 pt-1">
          {[...new Set(agents.map(a => a.agentName ?? "root"))].map(name => (
            <div key={name} className="flex items-center gap-1 text-xs">
              <div className="w-3 h-3 rounded-sm" style={{ backgroundColor: getAgentColor(name) }} />
              <span className="text-muted-foreground truncate max-w-[120px]">{name}</span>
            </div>
          ))}
        </div>
      </CardContent>

      {tooltip && (
        <div
          className="fixed z-[9999] bg-popover border rounded-md shadow-lg px-3 py-2 text-xs whitespace-nowrap pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y - 8, transform: "translate(-50%, -100%)" }}
        >
          <div className="font-medium mb-1">{tooltip.label}</div>
          <div className="text-muted-foreground space-y-0.5">
            <div>{formatTokenCount(tooltip.maxSingleCallTokens)} token max · {formatTokenCount(tooltip.tokens)} total · {formatLatency(tooltip.latencyMs)}</div>
            <div>{tooltip.startTime} → {tooltip.endTime}</div>
            <div>{tooltip.toolCallCount} tools{tooltip.model ? ` · ${tooltip.model}` : ""}</div>
          </div>
        </div>
      )}
    </Card>
  )
}
