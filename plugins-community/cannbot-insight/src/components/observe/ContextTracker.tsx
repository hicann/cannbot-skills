"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useMemo, useCallback, useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ZoomInIcon, ZoomOutIcon, RotateCcwIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { ContextReplayDialog } from "@/components/observe/ContextReplayDialog"
import { isCommandTurn, isContinuationTurn, parseCommandTurns, formatCommandDisplay } from "@/lib/shared/command-parser"

interface TurnRowItem {
  turnId: string
  turnIndex: number
  role: string
  contentSummary: string | null
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  parentExecutionId: string | null
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  contextWindowPct: number | null
  latencyMs: number
  createdAt: string | null
  completedAt: string | null
  model: string | null
  toolCalls: Array<{ toolCallId: string; toolName: string; state: string; durationMs: number }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
}

const AGENT_COLORS: Record<string, string> = {
  build: "#3b82f6",
  general: "#10b981",
  "ascendc-kernel-architect": "#f97316",
  "ascendc-kernel-developer": "#8b5cf6",
}
const DEFAULT_AGENT_COLOR = "#6b7280"

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return ""
  try {
    const d = new Date(ts)
    const m = String(d.getMonth() + 1)
    const day = String(d.getDate())
    const h = String(d.getHours()).padStart(2, "0")
    const min = String(d.getMinutes()).padStart(2, "0")
    return `${m}/${day} ${h}:${min}`
  } catch {
    return ts
  }
}

interface AgentStats {
  sessionId: string
  agentName: string
  isRoot: boolean
  turns: TurnRowItem[]
  peak: number
  average: number
  min: number
  totalTurns: number
  avgGrowthPerTurn: number
  cacheHitRate: number
  maxContextWindowPct: number
  events: ContextEvent[]
}

interface ContextEvent {
  type: "start" | "growth" | "peak" | "warning" | "end" | "compact"
  turnIndex: number
  timestamp: string | null
  contextSize: number
  note?: string
  growth?: number
  contTurnIndex?: number
  contTimestamp?: string | null
  preCompactTokens?: number
  postCompactTokens?: number
}

function computeAgentStats(turns: TurnRowItem[], contextWindowLimit: number): AgentStats[] {
  const sessionMap = new Map<string, TurnRowItem[]>()

  for (const t of turns) {
    if (t.totalTokens <= 0) continue
    const sid = t.isSubagent ? (t.subagentSessionId ?? `anon-${t.turnIndex}`) : "root"
    if (!sessionMap.has(sid)) sessionMap.set(sid, [])
    sessionMap.get(sid)!.push(t)
  }

  // Detect /compact + continuation events and merge into single events per group
  const compactMergedEvents: ContextEvent[] = (() => {
    const compactList: Array<{ turnIndex: number; timestamp: string | null; note: string }> = []
    const contList: Array<{ turnIndex: number; timestamp: string | null }> = []
    for (const t of turns) {
      if (t.role !== "user" || t.isSubagent) continue
      const text = t.contentSummary ?? ""
      if (isCommandTurn(text)) {
        const texts = [text]
        const idx = turns.indexOf(t)
        for (let j = idx + 1; j < turns.length; j++) {
          const next = turns[j]
          const nextText = next.contentSummary ?? ""
          if (nextText.includes("<local-command-caveat>") || nextText.includes("<local-command-stdout>")) { texts.push(nextText) } else break
        }
        const info = parseCommandTurns(texts)
        if (info.name === "/compact" || info.name === "compact") {
          compactList.push({ turnIndex: t.turnIndex, timestamp: t.createdAt, note: formatCommandDisplay(info) })
        }
      }
      if (isContinuationTurn(text)) {
        contList.push({ turnIndex: t.turnIndex, timestamp: t.createdAt })
      }
    }
    compactList.sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0
      return ta - tb
    })
    contList.sort((a, b) => {
      const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0
      const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0
      return ta - tb
    })
    // Find context drop data for each compact event pair
    const rootDataTurns = turns.filter(t => !t.isSubagent && t.totalTokens > 0)
    const merged: ContextEvent[] = []
    for (let i = 0; i < compactList.length; i++) {
      const c = compactList[i]
      const cont = i < contList.length ? contList[i] : null
      const compactTsMs = c.timestamp ? new Date(c.timestamp).getTime() : 0
      const contTsMs = cont?.timestamp ? new Date(cont.timestamp).getTime() : 0
      const preTurn = [...rootDataTurns].reverse().find(t => {
        const tTs = t.createdAt ? new Date(t.createdAt).getTime() : 0
        return tTs <= compactTsMs
      })
      const postTurn = contTsMs > 0
        ? rootDataTurns.find(t => {
            const tTs = t.createdAt ? new Date(t.createdAt).getTime() : 0
            return tTs >= contTsMs
          })
        : rootDataTurns.find(t => {
            const tTs = t.createdAt ? new Date(t.createdAt).getTime() : 0
            return tTs >= compactTsMs
          })
      const preTokens = preTurn?.totalTokens ?? 0
      const postTokens = postTurn?.totalTokens ?? 0
      const dropPct = preTokens > 0 && postTokens > 0 ? Math.round((1 - postTokens / preTokens) * 100) : 0
      merged.push({
        type: "compact",
        turnIndex: c.turnIndex,
        timestamp: c.timestamp,
        contextSize: 0,
        contTurnIndex: cont?.turnIndex ?? undefined,
        contTimestamp: cont?.timestamp ?? undefined,
        preCompactTokens: preTokens > 0 ? preTokens : undefined,
        postCompactTokens: postTokens > 0 ? postTokens : undefined,
        note: cont
          ? `⚡ /compact ↓${dropPct > 0 ? `-${dropPct}%` : ""}`
          : `⚡ /compact (#${c.turnIndex})`,
      })
    }
    return merged
  })()

  const stats: AgentStats[] = []

  for (const [sid, agentTurns] of sessionMap) {
    const isRoot = sid === "root"
    const agentName = agentTurns[0].agentName ?? (isRoot ? "Main Agent" : "?")
    const realInput = agentTurns.map(t => t.totalTokens)
    const cacheRead = agentTurns.map(t => t.cacheReadTokens ?? 0)
    const ctxPcts = agentTurns.map(t => t.contextWindowPct ?? 0).filter(p => p > 0)

    const peak = Math.max(...realInput)
    const average = Math.round(realInput.reduce((s, v) => s + v, 0) / realInput.length)
    const min = Math.min(...realInput)
    const avgGrowthPerTurn = agentTurns.length > 1
      ? Math.round((realInput[realInput.length - 1] - realInput[0]) / (agentTurns.length - 1))
      : 0

    let totalCacheHit = 0
    let totalTokensForCache = 0
    for (let i = 0; i < realInput.length; i++) {
      const it = realInput[i]
      const cr = cacheRead[i]
      if (it + cr > 0) {
        totalCacheHit += cr
        totalTokensForCache += it + cr
      }
    }
    const cacheHitRate = totalTokensForCache > 0 ? totalCacheHit / totalTokensForCache : 0
    const maxContextWindowPct = ctxPcts.length > 0 ? Math.max(...ctxPcts) : 0

    const events: ContextEvent[] = []

    if (agentTurns.length > 0) {
      events.push({ type: "start", turnIndex: agentTurns[0].turnIndex, timestamp: agentTurns[0].createdAt, contextSize: agentTurns[0].totalTokens, note: isRoot ? "Session 启动" : "Subagent 启动" })
    }

    for (let i = 1; i < agentTurns.length; i++) {
      const growth = agentTurns[i].totalTokens - agentTurns[i - 1].totalTokens
      if (growth > 5000) {
        events.push({ type: "growth", turnIndex: agentTurns[i].turnIndex, timestamp: agentTurns[i].createdAt, contextSize: agentTurns[i].totalTokens, growth, note: `增长 ${formatTokenCount(growth)}` })
      }
    }

    const peakTurn = agentTurns.reduce((max, t) => t.totalTokens > max.totalTokens ? t : max, agentTurns[0])
    if (peakTurn.totalTokens > 0) {
      const pct = peakTurn.contextWindowPct ?? (peakTurn.totalTokens / contextWindowLimit * 100)
      const note = pct > 80 ? `⚠️ 接近模型限制 (${contextWindowLimit / 1000}K)` : "峰值"
      events.push({ type: "peak", turnIndex: peakTurn.turnIndex, timestamp: peakTurn.createdAt, contextSize: peakTurn.totalTokens, note })
    }

    const warningTurns = agentTurns.filter(t => {
      const pct = t.contextWindowPct ?? (t.totalTokens / contextWindowLimit * 100)
      return pct > 80
    })
    for (const t of warningTurns) {
      if (!events.some(e => e.turnIndex === t.turnIndex && e.type === "peak")) {
        events.push({ type: "warning", turnIndex: t.turnIndex, timestamp: t.createdAt, contextSize: t.totalTokens, note: `窗口占比 ${(t.contextWindowPct ?? 0).toFixed(1)}%` })
      }
    }

    if (agentTurns.length > 0) {
      events.push({ type: "end", turnIndex: agentTurns[agentTurns.length - 1].turnIndex, timestamp: agentTurns[agentTurns.length - 1].createdAt, contextSize: agentTurns[agentTurns.length - 1].totalTokens, note: isRoot ? "Session 结束" : "Subagent 结束" })
    }

    events.sort((a, b) => a.turnIndex - b.turnIndex)

    // Merge compact events into root agent's timeline
    if (isRoot && compactMergedEvents.length > 0) {
      events.push(...compactMergedEvents)
      events.sort((a, b) => {
        const ta = a.timestamp ? new Date(a.timestamp).getTime() : 0
        const tb = b.timestamp ? new Date(b.timestamp).getTime() : 0
        return ta - tb
      })
    }

    stats.push({ sessionId: sid, agentName, isRoot, turns: agentTurns, peak, average, min, totalTurns: agentTurns.length, avgGrowthPerTurn, cacheHitRate, maxContextWindowPct, events })
  }

  stats.sort((a, b) => {
    if (a.isRoot !== b.isRoot) return a.isRoot ? -1 : 1
    return b.peak - a.peak
  })

  return stats
}

interface ContextTrackerProps {
  turns: TurnRowItem[]
  sessionModel: string | null
  onNavigateToTurn?: (turnId: string) => void
}

export function ContextTracker({ turns, sessionModel, onNavigateToTurn }: ContextTrackerProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)
  const [showAll, setShowAll] = useState(false)
  const [contextWindowLimit, setContextWindowLimit] = useState(200000)
  const [showReplay, setShowReplay] = useState(false)

  useEffect(() => {
    fetch(`/api/config/context-windows?model=${encodeURIComponent(sessionModel ?? "")}`)
      .then(r => r.json())
      .then(data => setContextWindowLimit(data.limit ?? 200000))
      .catch(() => setContextWindowLimit(200000))
  }, [sessionModel])
  const agentStats = useMemo(() => computeAgentStats(turns, contextWindowLimit), [turns, contextWindowLimit])

  const displayLabel = useCallback((s: AgentStats) => {
    if (s.isRoot) return s.agentName
    const aname = s.agentName === "ascendc-kernel-architect" ? "ascendc-arch" : s.agentName === "ascendc-kernel-developer" ? "ascendc-dev" : s.agentName
    const idx = agentStats.filter(a => a.agentName === s.agentName && !a.isRoot).indexOf(s) + 1
    return `${aname} #${idx}`
  }, [agentStats])

  const visibleStats = showAll ? agentStats : agentStats.slice(0, 8)

  if (agentStats.length === 0) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-muted-foreground text-sm">No context data available</div>
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium">上下文追踪</span>
        <Button variant="outline" size="sm" className="text-xs" onClick={() => setShowReplay(true)}>
          ⏯ Context Replay
        </Button>
      </div>
      <ContextReplayDialog
        open={showReplay}
        onOpenChange={setShowReplay}
        turns={turns}
        sessionModel={sessionModel}
      />

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-5 gap-2">
        {visibleStats.map(s => {
          const isExpanded = expandedAgent === s.sessionId
          const hasWarning = s.maxContextWindowPct > 80
          const label = displayLabel(s)

          return (
            <Card
              key={s.sessionId}
              size="sm"
              className={cn("cursor-pointer transition-colors hover:bg-accent/30", isExpanded && "ring-2 ring-primary/50", hasWarning && "border-yellow-500/50")}
              onClick={() => setExpandedAgent(isExpanded ? null : s.sessionId)}
            >
               <CardHeader className="pb-1">
                 <div className="flex items-center gap-1.5">
                   <Badge variant={s.isRoot ? "blue" : "orange"} className="text-xs">{label}</Badge>
                   <Badge variant="outline" className="text-xs">{s.isRoot ? "主agent" : `${s.totalTurns}轮`}</Badge>
                   {hasWarning && <Badge variant="yellow" className="text-xs">⚠️</Badge>}
                   {onNavigateToTurn && (
                     <Button
                       variant="ghost"
                       size="sm"
                       className="h-4 text-xs px-1 ml-auto opacity-60 hover:opacity-100"
                       onClick={(e) => {
                         e.stopPropagation()
                         onNavigateToTurn(s.turns[0].turnId)
                       }}
                     >
                       → Turns
                     </Button>
                   )}
                 </div>
               </CardHeader>
              <CardContent className="space-y-1">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">峰值</span>
                  <span className="font-medium tabular-nums">{formatTokenCount(s.peak)}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">平均</span>
                  <span className="tabular-nums">{formatTokenCount(s.average)}</span>
                </div>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">缓存率</span>
                  <span className="tabular-nums">{(s.cacheHitRate * 100).toFixed(0)}%</span>
                </div>
                {s.maxContextWindowPct > 0 && (
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">窗口占比</span>
                    <span className={cn("tabular-nums", s.maxContextWindowPct > 80 ? "text-yellow-600 dark:text-yellow-400 font-medium" : "")}>
                      {s.maxContextWindowPct.toFixed(1)}%
                    </span>
                  </div>
                )}
              </CardContent>
            </Card>
          )
        })}
      </div>

      {agentStats.length > 8 && (
        <Button variant="outline" size="sm" className="text-xs" onClick={() => setShowAll(prev => !prev)}>
          {showAll ? `收起 (只显示前8个)` : `展开全部 ${agentStats.length} 个执行`}
        </Button>
      )}

      <GrowthChart agentStats={agentStats} contextWindowLimit={contextWindowLimit} displayLabel={displayLabel} onNavigateToTurn={onNavigateToTurn} />

      {expandedAgent && (() => {
        const s = agentStats.find(a => a.sessionId === expandedAgent)
        if (!s) return null
        return <AgentDetail stats={s} label={displayLabel(s)} onClose={() => setExpandedAgent(null)} onNavigateToTurn={onNavigateToTurn} />
      })()}

    </div>
  )
}

function GrowthChart({ agentStats, contextWindowLimit, displayLabel, onNavigateToTurn }: { agentStats: AgentStats[]; contextWindowLimit: number; displayLabel: (s: AgentStats) => string; onNavigateToTurn?: (turnId: string) => void }) {
  const SVG_HEIGHT = 460
  const PADDING_TOP = 40
  const PADDING_BOTTOM = 40
  const PADDING_LEFT = 60
  const PADDING_RIGHT = 40

  const containerRef = useRef<HTMLDivElement>(null)
  const svgRef = useRef<SVGSVGElement>(null)
  const [visibleSessions, setVisibleSessions] = useState<Set<string>>(new Set(agentStats.map(s => s.sessionId)))
  const [hoveredPoint, setHoveredPoint] = useState<{
    sessionId: string; turnIndex: number; x: number; y: number
    tokens: number; cacheRead: number; newInput: number; cacheWrite: number
    contextWindowPct: number | null; time: string; label: string; color: string; turn: TurnRowItem
  } | null>(null)
  const [highlightedSessionId, setHighlightedSessionId] = useState<string | null>(null)
  const [viewMode, setViewMode] = useState<"area" | "line">("area")
  const [zoom, setZoom] = useState(1)
  const ZOOM_STEP = 0.15
  const ZOOM_MIN = 0.3
  const ZOOM_MAX = 3

  const activeHighlight = hoveredPoint?.sessionId ?? highlightedSessionId

  function clientToSvgCoords(clientX: number, clientY: number): { x: number; y: number } {
    const svg = svgRef.current
    if (!svg) return { x: 0, y: 0 }
    const ctm = svg.getScreenCTM()
    if (!ctm) return { x: 0, y: 0 }
    const inv = ctm.inverse()
    return { x: inv.a * clientX + inv.c * clientY + inv.e, y: inv.b * clientX + inv.d * clientY + inv.f }
  }

  function findNearestPoint(svgX: number, svgY: number) {
    let bestDist = Infinity
    let bestPoint: typeof hoveredPoint | null = null
    const HIT_RADIUS_SQ = 2500
    for (const line of agentLines) {
      for (const p of line.points) {
        const dx = p.x - svgX
        const dy = p.y - svgY
        const dist = dx * dx + dy * dy
        if (dist < bestDist && dist < HIT_RADIUS_SQ) {
          bestDist = dist
          bestPoint = {
            sessionId: line.sessionId, turnIndex: p.turnIndex, x: p.x, y: p.y,
            tokens: p.tokens, cacheRead: p.cacheRead, newInput: p.newInput, cacheWrite: p.cacheWrite,
            contextWindowPct: p.contextWindowPct, time: p.time, label: line.label, color: line.color, turn: p.turn,
          }
        }
      }
    }
    return bestPoint
  }
  const filteredStats = agentStats.filter(s => visibleSessions.has(s.sessionId))

  const allPoints = useMemo(() => {
    const pts: Array<{ agentName: string; turnIndex: number; timestamp: number; inputTokens: number; color: string }> = []
    for (const s of filteredStats) {
      const color = AGENT_COLORS[s.agentName] ?? DEFAULT_AGENT_COLOR
      for (const t of s.turns) {
        pts.push({ agentName: s.agentName, turnIndex: t.turnIndex, timestamp: t.createdAt ? new Date(t.createdAt).getTime() : 0, inputTokens: t.totalTokens, color })
      }
    }
    return pts
  }, [filteredStats])

  const minTime = allPoints.length > 0 ? Math.min(...allPoints.map(p => p.timestamp)) : 0
  const maxTime = allPoints.length > 0 ? Math.max(...allPoints.map(p => p.timestamp)) : 0
  const timeRange = maxTime - minTime || 1

  const maxInput = Math.max(...allPoints.map(p => p.inputTokens), 1)
  const yMax = Math.min(Math.ceil(maxInput * 1.15 / 10000) * 10000, contextWindowLimit * 1.1)

  const svgW = 1200
  const svgH = SVG_HEIGHT
  const chartWidth = svgW
  const plotWidth = chartWidth - PADDING_LEFT - PADDING_RIGHT
  const plotHeight = SVG_HEIGHT - PADDING_TOP - PADDING_BOTTOM
  const baselineY = SVG_HEIGHT - PADDING_BOTTOM

  function toXFn(ts: number): number { return PADDING_LEFT + ((ts - minTime) / timeRange) * plotWidth }
  function toYFn(tokens: number): number { return PADDING_TOP + plotHeight - (tokens / yMax) * plotHeight }

  const yTicks: Array<{ value: number; y: number; label: string }> = []
  const yStep = yMax <= 20000 ? 5000 : yMax <= 100000 ? 20000 : 40000
  for (let v = 0; v <= yMax; v += yStep) { yTicks.push({ value: v, y: toYFn(v), label: formatTokenCount(v) }) }

  const timeTicks: Array<{ x: number; label: string }> = []
  const timeStepMs = timeRange <= 600000 ? 120000 : timeRange <= 3600000 ? 600000 : 7200000
  for (let ts = minTime; ts <= maxTime; ts += timeStepMs) {
    const d = new Date(ts)
    timeTicks.push({ x: toXFn(ts), label: `${String(d.getMonth() + 1)}/${String(d.getDate())} ${String(d.getHours()).padStart(2, "0")}:00` })
  }

  const safeY = toYFn(contextWindowLimit * 0.5)
  const cautionY = toYFn(contextWindowLimit * 0.8)
  const limitY = toYFn(contextWindowLimit)

  const pctMarkers = useMemo(() => {
    const markers: Array<{ pct: number; y: number; label: string; color: string }> = []
    if (safeY >= PADDING_TOP && safeY <= baselineY) markers.push({ pct: 50, y: safeY, label: "50%", color: "#22c55e" })
    if (cautionY >= PADDING_TOP && cautionY <= baselineY) markers.push({ pct: 80, y: cautionY, label: "80%", color: "#eab308" })
    if (limitY >= PADDING_TOP && limitY <= baselineY) markers.push({ pct: 100, y: limitY, label: "100%", color: "#ef4444" })
    return markers
  }, [safeY, cautionY, limitY, PADDING_TOP, baselineY])

  function curvePath(pts: Array<{ x: number; y: number }>): string {
    if (pts.length < 2) return pts.length === 1 ? `M ${pts[0].x},${pts[0].y}` : ""
    if (pts.length === 2) return `M ${pts[0].x},${pts[0].y} L ${pts[1].x},${pts[1].y}`
    let d = `M ${pts[0].x},${pts[0].y}`
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(0, i - 1)]
      const p1 = pts[i]
      const p2 = pts[i + 1]
      const p3 = pts[Math.min(pts.length - 1, i + 2)]
      const cp1x = p1.x + (p2.x - p0.x) / 6
      const cp2x = p2.x - (p3.x - p1.x) / 6
      const minY = Math.min(p1.y, p2.y)
      const maxY = Math.max(p1.y, p2.y)
      const cp1y = Math.max(minY, Math.min(maxY, p1.y + (p2.y - p0.y) / 6))
      const cp2y = Math.max(minY, Math.min(maxY, p2.y - (p3.y - p1.y) / 6))
      d += ` C ${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`
    }
    return d
  }

  function areaPathStr(pts: Array<{ x: number; y: number }>): string {
    if (pts.length < 2) return ""
    return `${curvePath(pts)} L ${pts[pts.length - 1].x},${baselineY} L ${pts[0].x},${baselineY} Z`
  }

  const agentLines = filteredStats.map(s => {
    const color = AGENT_COLORS[s.agentName] ?? DEFAULT_AGENT_COLOR
    const label = displayLabel(s)
    const isRoot = s.isRoot
    const points = s.turns.map(t => ({
      x: toXFn(t.createdAt ? new Date(t.createdAt).getTime() : 0),
      y: toYFn(t.totalTokens),
      tokens: t.totalTokens,
      cacheRead: t.cacheReadTokens,
      newInput: t.inputTokens,
      cacheWrite: t.cacheWriteTokens,
      contextWindowPct: t.contextWindowPct,
      turnIndex: t.turnIndex,
      time: formatTimestamp(t.createdAt),
      turn: t,
    }))
    const chartEvents = s.events
      .filter(e => e.type !== "start" && e.type !== "end" && e.type !== "compact")
      .map(e => {
        const eventTurn = s.turns.find(t => t.turnIndex === e.turnIndex)
        if (!eventTurn) return null
        return {
          type: e.type, x: toXFn(eventTurn.createdAt ? new Date(eventTurn.createdAt).getTime() : 0),
          y: toYFn(eventTurn.totalTokens),
          note: e.note, growth: e.growth,
        }
      })
      .filter(e => e !== null) as Array<{ type: string; x: number; y: number; note: string | undefined; growth: number | undefined }>
    return { sessionId: s.sessionId, color, label, isRoot, points, totalTurns: s.totalTurns, events: chartEvents }
  })

  const lineLabels = (() => {
    const items = agentLines
      .filter(line => line.points.length >= 2)
      .map(line => {
        const lastPt = line.points[line.points.length - 1]
        const estWidth = line.label.length * 6.5 + 14
        const rightSpace = chartWidth - PADDING_RIGHT - lastPt.x
        return {
          sessionId: line.sessionId, label: line.label, color: line.color, isRoot: line.isRoot,
          endX: lastPt.x, endY: lastPt.y, targetY: lastPt.y,
          textAnchor: rightSpace > estWidth + 10 ? "start" as const : "end" as const,
          offsetX: rightSpace > estWidth + 10 ? 6 : -6,
          estWidth,
        }
      })
    const MIN_GAP = 17
    items.sort((a, b) => a.targetY - b.targetY)
    for (let i = 1; i < items.length; i++) {
      if (items[i].targetY - items[i - 1].targetY < MIN_GAP) {
        items[i].targetY = items[i - 1].targetY + MIN_GAP
      }
    }
    for (const item of items) {
      item.targetY = Math.max(PADDING_TOP + 10, Math.min(baselineY - 10, item.targetY))
    }
    return items
  })()

  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle>上下文增长趋势</CardTitle>
          <div className="flex items-center gap-2">
            <Button variant={viewMode === "area" ? "secondary" : "outline"} size="sm" className="text-xs h-6 px-2" onClick={() => setViewMode("area")}>面积</Button>
            <Button variant={viewMode === "line" ? "secondary" : "outline"} size="sm" className="text-xs h-6 px-2" onClick={() => setViewMode("line")}>折线</Button>
            <span className="text-xs text-muted-foreground mx-0.5">|</span>
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(z => Math.max(ZOOM_MIN, z - ZOOM_STEP))} title="缩小">
              <ZoomOutIcon className="size-3.5" />
            </Button>
            <span className="text-xs text-muted-foreground tabular-nums w-[36px] text-center">{Math.round(zoom * 100)}%</span>
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(z => Math.min(ZOOM_MAX, z + ZOOM_STEP))} title="放大">
              <ZoomInIcon className="size-3.5" />
            </Button>
            <Button variant="ghost" size="sm" className="h-6 w-6 p-0" onClick={() => setZoom(1)} title="重置">
              <RotateCcwIcon className="size-3.5" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div
          ref={containerRef}
          className="overflow-auto border rounded-md"
          style={{ maxWidth: "100%", maxHeight: 500 }}
          onWheel={(e) => {
            if (e.ctrlKey || e.metaKey) {
              e.preventDefault()
              const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP
              setZoom(z => Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, z + delta)))
            }
          }}
        >
        <svg ref={svgRef} width={svgW * zoom} height={svgH * zoom} viewBox={`0 0 ${svgW} ${svgH}`} className="block mx-auto">
          <defs>
            {filteredStats.map(s => {
              const color = AGENT_COLORS[s.agentName] ?? DEFAULT_AGENT_COLOR
              return (
                <linearGradient key={s.sessionId} id={`areaGrad-${s.sessionId}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={color} stopOpacity={0.28} />
                  <stop offset="100%" stopColor={color} stopOpacity={0.03} />
                </linearGradient>
              )
            })}
            <clipPath id="plotClip">
              <rect x={PADDING_LEFT} y={PADDING_TOP} width={plotWidth} height={plotHeight} />
            </clipPath>
          </defs>

          {/* Zone bands */}
          <rect x={PADDING_LEFT} y={safeY} width={plotWidth} height={baselineY - safeY} fill="#22c55e" opacity={0.04} clipPath="url(#plotClip)" />
          <rect x={PADDING_LEFT} y={cautionY} width={plotWidth} height={safeY - cautionY} fill="#eab308" opacity={0.05} clipPath="url(#plotClip)" />
          <rect x={PADDING_LEFT} y={limitY} width={plotWidth} height={cautionY - limitY} fill="#ef4444" opacity={0.06} clipPath="url(#plotClip)" />
          {yMax > contextWindowLimit && (
            <rect x={PADDING_LEFT} y={PADDING_TOP} width={plotWidth} height={limitY - PADDING_TOP} fill="#ef4444" opacity={0.12} clipPath="url(#plotClip)" />
          )}

          {/* Zone boundary lines */}
          {safeY > PADDING_TOP && safeY < baselineY && (
            <line x1={PADDING_LEFT} y1={safeY} x2={chartWidth - PADDING_RIGHT} y2={safeY} stroke="#22c55e" strokeWidth={1} strokeDasharray="6 4" opacity={0.35} />
          )}
          {cautionY > PADDING_TOP && cautionY < baselineY && (
            <line x1={PADDING_LEFT} y1={cautionY} x2={chartWidth - PADDING_RIGHT} y2={cautionY} stroke="#eab308" strokeWidth={1} strokeDasharray="6 4" opacity={0.4} />
          )}

          {/* Context limit line + label */}
          {contextWindowLimit <= yMax && limitY >= PADDING_TOP && (
            <>
              <line x1={PADDING_LEFT} y1={limitY} x2={chartWidth - PADDING_RIGHT} y2={limitY} stroke="#ef4444" strokeWidth={1.5} strokeDasharray="8 4" />
              <text x={PADDING_LEFT + 6} y={limitY - 5} fontSize={9} fill="#ef4444" fontFamily="system-ui, sans-serif">模型限制 {formatTokenCount(contextWindowLimit)}</text>
            </>
          )}

          {/* Grid lines */}
          {yTicks.map(tick => (
            <g key={tick.value}>
              <line x1={PADDING_LEFT} y1={tick.y} x2={chartWidth - PADDING_RIGHT} y2={tick.y} stroke="#e5e7eb" strokeWidth={tick.value === 0 ? 1 : 0.5} opacity={tick.value === 0 ? 0.8 : 0.4} />
              <text x={PADDING_LEFT - 5} y={tick.y + 4} textAnchor="end" fontSize={10} fill="#6b7280">{tick.label}</text>
            </g>
          ))}

          {/* Right Y-axis: context % markers */}
          {pctMarkers.map(m => (
            <text key={m.pct} x={chartWidth - PADDING_RIGHT + 5} y={m.y + 4} fontSize={9} fill={m.color} fontFamily="system-ui, sans-serif" fontWeight="bold">{m.label}</text>
          ))}

          {/* X-axis ticks */}
          {timeTicks.map(tick => (
            <g key={tick.x}>
              <line x1={tick.x} y1={baselineY} x2={tick.x} y2={baselineY + 6} stroke="#d1d5db" strokeWidth={0.5} />
              <text x={tick.x} y={baselineY + 18} textAnchor="middle" fontSize={9} fill="#6b7280">{tick.label}</text>
            </g>
          ))}

          {/* Plot border */}
          <rect x={PADDING_LEFT} y={PADDING_TOP} width={plotWidth} height={plotHeight} fill="none" stroke="#d1d5db" strokeWidth={0.5} rx={2} />

          {/* Area fills */}
          {viewMode === "area" && agentLines.map(line => {
            const isActive = activeHighlight === line.sessionId
            const isWeakened = activeHighlight != null && !isActive
            if (line.points.length < 2) return null
            return (
              <g key={`area-${line.sessionId}`} opacity={isWeakened ? 0.08 : isActive ? 1 : 0.6} clipPath="url(#plotClip)">
                <path d={areaPathStr(line.points)} fill={`url(#areaGrad-${line.sessionId})`} />
              </g>
            )
          })}

          {/* Agent lines (smooth curves) */}
          {agentLines.map(line => {
            const isActive = activeHighlight === line.sessionId
            const isWeakened = activeHighlight != null && !isActive
            const d = line.points.length >= 2 ? curvePath(line.points) : ""
            return (
              <g key={line.sessionId} opacity={isWeakened ? 0.2 : 1}>
                {d && (
                  <path
                    d={d}
                    fill="none"
                    stroke={line.color}
                    strokeWidth={isActive ? 2.5 : line.isRoot ? 2 : 1.5}
                    strokeLinecap="round"
                  />
                )}
                {line.points.map(p => (
                  <circle
                    key={`${line.sessionId}-${p.turnIndex}`}
                    cx={p.x}
                    cy={p.y}
                    r={isActive ? 4 : line.isRoot ? 3 : 2.5}
                    fill={line.color}
                    stroke="white"
                    strokeWidth={isActive ? 2 : 1}
                  />
                ))}
                {line.events.map((evt, i) => {
                  const showLabel = isActive || highlightedSessionId === line.sessionId
                  let markerColor = line.color
                  let markerShape: React.ReactNode = null
                  if (evt.type === "growth") {
                    markerColor = line.color
                    markerShape = <polygon points={`0,-8 4.5,0 -4.5,0`} fill={markerColor} opacity={0.85} />
                  } else if (evt.type === "peak") {
                    markerColor = "#f59e0b"
                    markerShape = <polygon points="0,-8 3,-3 8,0 3,3 0,8 -3,3 -8,0 -3,-3" fill={markerColor} opacity={0.9} />
                  } else if (evt.type === "warning") {
                    markerColor = "#ef4444"
                    markerShape = (
                      <g>
                        <polygon points="0,-8 7,5 -7,5" fill={markerColor} opacity={0.9} />
                        <line x1="0" y1="-4" x2="0" y2="1" stroke="white" strokeWidth={1.5} />
                        <circle cx="0" cy="3" r="1" fill="white" />
                      </g>
                    )
                  }
                  if (!markerShape) return null
                  return (
                    <g key={`evt-${line.sessionId}-${i}`} transform={`translate(${evt.x},${evt.y - 2})`} opacity={isWeakened ? 0.15 : 0.9}>
                      {markerShape}
                      {showLabel && evt.note && (
                        <text x={8} y={-4} fontSize={8} fill={evt.type === "warning" ? "#ef4444" : evt.type === "peak" ? "#f59e0b" : line.color} fontFamily="system-ui, sans-serif">{evt.note}</text>
                      )}
                    </g>
                  )
                })}
              </g>
            )
          })}

          {/* End-of-line labels */}
          {lineLabels.map(l => {
            const isActive = activeHighlight === l.sessionId
            const isWeakened = activeHighlight != null && !isActive
            const bgX = l.textAnchor === "start" ? l.endX + l.offsetX - 4 : l.endX + l.offsetX - l.estWidth + 4
            const pushed = Math.abs(l.targetY - l.endY) > 5
            return (
              <g key={`label-${l.sessionId}`} opacity={isWeakened ? 0.25 : isActive ? 1 : 0.75}>
                <rect x={bgX} y={l.targetY - 7} width={l.estWidth} height={14} rx={4} fill="white" fillOpacity={isActive ? 0.92 : 0.8} stroke={l.color} strokeWidth={isActive ? 1 : 0.5} strokeOpacity={0.35} />
                <text x={l.endX + l.offsetX} y={l.targetY + 3.5} fontSize={9} textAnchor={l.textAnchor} fill={l.color} fontWeight={l.isRoot ? "600" : "400"} fontFamily="system-ui, sans-serif"
                  onMouseOver={() => setHighlightedSessionId(l.sessionId)} onMouseLeave={() => setHighlightedSessionId(null)}
                  style={{ cursor: "pointer" }}
                >{l.label}</text>
                {pushed && (
                  <line x1={l.endX} y1={l.endY} x2={l.endX + (l.textAnchor === "start" ? 3 : -3)} y2={l.targetY} stroke={l.color} strokeWidth={0.5} strokeDasharray="2 2" opacity={0.3} />
                )}
              </g>
            )
          })}

          {/* Compact event markers with context drop annotation */}
          {(() => {
            const rootStats = agentStats.find(s => s.isRoot)
            if (!rootStats) return null
            const compactEvts = rootStats.events.filter(e => e.type === "compact")
            if (compactEvts.length === 0) return null
            const ICON_Y = PADDING_TOP + 5
            const LABEL_Y = ICON_Y + 16
            return compactEvts.map((evt, i) => {
              const compactTs = evt.timestamp ? new Date(evt.timestamp).getTime() : 0
              const contTs = evt.contTimestamp ? new Date(evt.contTimestamp).getTime() : 0
              if (!compactTs) return null
              const xCompact = toXFn(compactTs)
              const xCont = contTs > 0 ? toXFn(contTs) : xCompact
              const labelOffsetY = i * 14

              // Drop annotation: bracket from pre-compact to post-compact context level
              const dropAnnotation = (() => {
                const pre = evt.preCompactTokens ?? 0
                const post = evt.postCompactTokens ?? 0
                if (pre <= 0 || post <= 0) return null
                const yPre = toYFn(pre)
                const yPost = toYFn(post)
                const dropPct = Math.round((1 - post / pre) * 100)
                const bracketX = Math.max(xCompact, xCont) + 4
                return (
                  <g>
                    {/* Vertical bracket showing the drop */}
                    <line x1={bracketX} y1={yPre} x2={bracketX} y2={yPost} stroke="#8b5cf6" strokeWidth={1.5} opacity={0.7} />
                    {/* Horizontal ticks at top and bottom */}
                    <line x1={bracketX - 3} y1={yPre} x2={bracketX + 3} y2={yPre} stroke="#8b5cf6" strokeWidth={1} opacity={0.7} />
                    <line x1={bracketX - 3} y1={yPost} x2={bracketX + 3} y2={yPost} stroke="#8b5cf6" strokeWidth={1} opacity={0.7} />
                    {/* Drop arrow */}
                    <polygon points={`${bracketX},${yPost} ${bracketX - 2},${yPost + 6} ${bracketX + 2},${yPost + 6}`} fill="#8b5cf6" opacity={0.7} transform={`rotate(180, ${bracketX}, ${yPost})`} />
                    {/* Drop label */}
                    <text x={bracketX + 6} y={yPre + 2} fontSize={8} fill="#8b5cf6" textAnchor="start" fontFamily="system-ui, sans-serif">{formatTokenCount(pre)}</text>
                    <text x={bracketX + 6} y={yPost + 2} fontSize={8} fill="#8b5cf6" textAnchor="start" fontFamily="system-ui, sans-serif" fontWeight="bold">↓ {formatTokenCount(post)} (-{dropPct}%)</text>
                  </g>
                )
              })()

              return (
                <g key={`compact-${i}`} clipPath="url(#plotClip)">
                  {/* Shaded zone between /compact and continuation */}
                  {contTs > 0 && contTs !== compactTs && (
                    <rect x={xCompact} y={PADDING_TOP} width={xCont - xCompact} height={plotHeight} fill="#8b5cf6" opacity={0.06} />
                  )}
                  {/* Dashed line at /compact command */}
                  <line x1={xCompact} y1={PADDING_TOP} x2={xCompact} y2={baselineY} stroke="#8b5cf6" strokeWidth={1.5} strokeDasharray="6 3" opacity={0.6} />
                  {/* Thin line at continuation point */}
                  {contTs > 0 && contTs !== compactTs && (
                    <line x1={xCont} y1={PADDING_TOP} x2={xCont} y2={baselineY} stroke="#8b5cf6" strokeWidth={1} strokeDasharray="3 3" opacity={0.4} />
                  )}
                  {/* Icon badge */}
                  <rect x={xCompact - 5} y={ICON_Y} width={10} height={12} rx={2} fill="#8b5cf6" opacity={0.9} />
                  <text x={xCompact} y={ICON_Y + 9} fontSize={7} fill="white" textAnchor="middle" fontWeight="bold" fontFamily="system-ui, sans-serif">⚡</text>
                  {/* Label */}
                  <text x={xCompact + 8} y={LABEL_Y + labelOffsetY} fontSize={8} fill="#8b5cf6" textAnchor="start" fontFamily="system-ui, sans-serif">{evt.note}</text>
                  {/* Drop annotation */}
                  {dropAnnotation}
                </g>
              )
            }).filter(Boolean)
          })()}

          {/* Crosshair */}
          {hoveredPoint && (
            <g style={{ pointerEvents: "none" }}>
              <line x1={hoveredPoint.x} y1={PADDING_TOP} x2={hoveredPoint.x} y2={baselineY} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} clipPath="url(#plotClip)" />
              <line x1={PADDING_LEFT} y1={hoveredPoint.y} x2={chartWidth - PADDING_RIGHT} y2={hoveredPoint.y} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 3" opacity={0.5} clipPath="url(#plotClip)" />
            </g>
          )}

          {/* Tooltip */}
          {hoveredPoint && (() => {
            const tt = hoveredPoint
            const t = tt.turn
            const inputTotal = t.inputTokens + t.cacheReadTokens + t.cacheWriteTokens
            const cachePct = inputTotal > 0 ? tt.cacheRead / inputTotal * 100 : 0
            const newInputPct = inputTotal > 0 ? tt.newInput / inputTotal * 100 : 0
            const cacheWritePct = inputTotal > 0 ? tt.cacheWrite / inputTotal * 100 : 0
            const ctxPct = tt.contextWindowPct ?? (inputTotal / contextWindowLimit * 100)
            const ctxPctDisplay = tt.contextWindowPct != null ? `${tt.contextWindowPct.toFixed(1)}%` : `≈${ctxPct.toFixed(1)}%`

            const TOOLTIP_W = 250
            const HEADER_H = 18
            const BAR_H = 8
            const BAR_LABEL_H = 12
            const LINE_H = 14
            const TOOLTIP_PAD = 10
            const metricsLines = [
              `总量: ${formatTokenCount(tt.tokens)} | 输入: ${formatTokenCount(inputTotal)}`,
              `新输入: ${formatTokenCount(tt.newInput)} | 输出: ${formatTokenCount(t.outputTokens)}`,
              `窗口占比: ${ctxPctDisplay} | 模型: ${t.model ?? "—"}`,
            ]
            const TOOLTIP_H = TOOLTIP_PAD + HEADER_H + 4 + BAR_H + BAR_LABEL_H + metricsLines.length * LINE_H + TOOLTIP_PAD

            let tx = tt.x + 16
            let ty = tt.y - TOOLTIP_H / 2
            if (tx + TOOLTIP_W > chartWidth - PADDING_RIGHT) tx = tt.x - TOOLTIP_W - 16
            if (ty < PADDING_TOP) ty = PADDING_TOP
            if (ty + TOOLTIP_H > baselineY) ty = baselineY - TOOLTIP_H

            const barStartY = ty + TOOLTIP_PAD + HEADER_H + 4
            const barWidth = TOOLTIP_W - 2 * TOOLTIP_PAD
            const metricsStartY = barStartY + BAR_H + BAR_LABEL_H

            const ctxFillPct = Math.min(ctxPct / 100, 1)
            const ctxBarColor = ctxPct < 50 ? "#22c55e" : ctxPct < 80 ? "#eab308" : "#ef4444"

            return (
              <g style={{ pointerEvents: "none" }}>
                <rect x={tx} y={ty} width={TOOLTIP_W} height={TOOLTIP_H} rx={8} fill="white" stroke="#d1d5db" strokeWidth={1} style={{ filter: "drop-shadow(0 2px 6px rgba(0,0,0,0.12))" }} />
                <circle cx={tt.x} cy={tt.y} r={7} fill="none" stroke={tt.color} strokeWidth={2.5} opacity={0.8} />
                <circle cx={tt.x} cy={tt.y} r={3} fill={tt.color} opacity={0.6} />

                {/* Header */}
                <text x={tx + TOOLTIP_PAD} y={ty + TOOLTIP_PAD + 13} fontSize={12} fontWeight="bold" fill={tt.color} fontFamily="system-ui, sans-serif">{tt.label} #{tt.turnIndex}</text>
                <text x={tx + TOOLTIP_W - TOOLTIP_PAD} y={ty + TOOLTIP_PAD + 13} fontSize={9} fill="#6b7280" textAnchor="end" fontFamily="system-ui, sans-serif">{tt.time} · {t.role}</text>

                {/* Composition bar */}
                <rect x={tx + TOOLTIP_PAD} y={barStartY} width={Math.max(barWidth * cachePct / 100, 0)} height={BAR_H} rx={1} fill="#f59e0b" />
                <rect x={tx + TOOLTIP_PAD + barWidth * cachePct / 100} y={barStartY} width={Math.max(barWidth * newInputPct / 100, 0)} height={BAR_H} fill="#3b82f6" />
                {cacheWritePct > 0 && (
                  <rect x={tx + TOOLTIP_PAD + barWidth * (cachePct + newInputPct) / 100} y={barStartY} width={Math.max(barWidth * cacheWritePct / 100, 1)} height={BAR_H} fill="#f97316" />
                )}
                <rect x={tx + TOOLTIP_PAD} y={barStartY} width={barWidth} height={BAR_H} rx={2} fill="none" stroke="#d1d5db" strokeWidth={0.5} />
                <text x={tx + TOOLTIP_PAD} y={barStartY + BAR_H + 9} fontSize={8} fill="#f59e0b" fontFamily="system-ui, sans-serif">Cache {cachePct.toFixed(0)}%</text>
                <text x={tx + TOOLTIP_PAD + barWidth * 0.35} y={barStartY + BAR_H + 9} fontSize={8} fill="#3b82f6" fontFamily="system-ui, sans-serif">Input {newInputPct.toFixed(0)}%</text>
                {cacheWritePct > 0 && <text x={tx + TOOLTIP_PAD + barWidth * 0.7} y={barStartY + BAR_H + 9} fontSize={8} fill="#f97316" fontFamily="system-ui, sans-serif">CW {cacheWritePct.toFixed(0)}%</text>}

                {/* Context window fill bar */}
                <rect x={tx + TOOLTIP_PAD} y={metricsStartY - 6} width={barWidth} height={4} rx={1} fill="#e5e7eb" />
                <rect x={tx + TOOLTIP_PAD} y={metricsStartY - 6} width={barWidth * ctxFillPct} height={4} rx={1} fill={ctxBarColor} />

                {/* Metrics text */}
                {metricsLines.map((line, i) => (
                  <text key={i} x={tx + TOOLTIP_PAD} y={metricsStartY + 4 + i * LINE_H} fontSize={11} fill="#1f2937" fontFamily="system-ui, sans-serif">{line}</text>
                ))}
              </g>
            )
          })()}

          {/* Hover overlay */}
          <rect
            x={PADDING_LEFT} y={PADDING_TOP} width={plotWidth} height={plotHeight}
            fill="transparent" cursor="crosshair"
            onMouseMove={(e) => {
              const coords = clientToSvgCoords(e.clientX, e.clientY)
              const nearest = findNearestPoint(coords.x, coords.y)
              if (nearest) {
                setHoveredPoint(nearest)
                setHighlightedSessionId(nearest.sessionId)
              } else {
                setHoveredPoint(null)
                setHighlightedSessionId(null)
              }
            }}
            onMouseLeave={() => { setHoveredPoint(null); setHighlightedSessionId(null) }}
            onClick={() => { if (hoveredPoint) onNavigateToTurn?.(hoveredPoint.turn.turnId) }}
          />

        </svg>
        </div>

        <div className="flex flex-wrap gap-2 mt-3 items-center">
          <span className="text-xs text-muted-foreground shrink-0">显示:</span>
          {agentStats.map(s => {
            const color = AGENT_COLORS[s.agentName] ?? DEFAULT_AGENT_COLOR
            const label = displayLabel(s)
            const checked = visibleSessions.has(s.sessionId)
            const isActive = activeHighlight === s.sessionId
            const isWeakened = activeHighlight != null && !isActive
            return (
              <label
                key={s.sessionId}
                className={cn("flex items-center gap-1.5 text-xs cursor-pointer select-none transition-opacity px-1.5 py-0.5 rounded-md", isWeakened && "opacity-40", isActive && "bg-accent/40")}
                onMouseOver={() => setHighlightedSessionId(s.sessionId)}
                onMouseLeave={() => setHighlightedSessionId(null)}
              >
                <input
                  type="checkbox" checked={checked}
                  onChange={() => {
                    setVisibleSessions(prev => {
                      const next = new Set(prev)
                      if (checked) next.delete(s.sessionId)
                      else next.add(s.sessionId)
                      return next
                    })
                  }}
                  className="size-3 accent-current" style={{ accentColor: color }}
                />
                <span className={cn("inline-block w-2.5 h-2.5 rounded-full transition-all", isActive && "ring-2 ring-offset-1 ring-primary/40")} style={{ backgroundColor: checked ? color : "#d1d5db" }} />
                <span className={cn(checked ? "font-medium" : "text-muted-foreground", s.isRoot && "text-blue-600 dark:text-blue-400", isActive && "font-semibold")}>{label}</span>
                {!s.isRoot && checked && <span className="text-muted-foreground">{s.totalTurns}轮</span>}
              </label>
            )
          })}
          <span className="text-xs text-muted-foreground ml-1">|</span>
          <label className="flex items-center gap-1.5 text-xs cursor-pointer select-none">
            <input type="checkbox" checked={visibleSessions.size === agentStats.length}
              onChange={() => { if (visibleSessions.size === agentStats.length) setVisibleSessions(new Set()); else setVisibleSessions(new Set(agentStats.map(s => s.sessionId))) }}
              className="size-3 accent-red-500"
            />
            <span className="text-xs">全部</span>
          </label>
          <span className="text-xs text-muted-foreground ml-1">|</span>
          <div className="flex items-center gap-1.5 text-xs">
            <span className="inline-block w-4 h-0 border-t-2 border-dashed border-red-500" />
            <span className="text-red-500">模型限制 ({formatTokenCount(contextWindowLimit)})</span>
          </div>
          <span className="text-xs text-muted-foreground ml-1">|</span>
          <div className="flex items-center gap-1 text-xs">
            <span className="inline-block w-2 h-2 rounded-sm bg-green-500/30" />
            安全区 <span className="inline-block w-2 h-2 rounded-sm bg-yellow-500/40 ml-1" />
            警戒区 <span className="inline-block w-2 h-2 rounded-sm bg-red-500/50 ml-1" />
            危险区
          </div>
          {agentStats.find(s => s.isRoot)?.events.some(e => e.type === "compact") && (
            <>
              <span className="text-xs text-muted-foreground ml-1">|</span>
              <div className="flex items-center gap-1 text-xs">
                <span className="inline-block w-4 h-0 border-t-2 border-dashed border-purple-500" />
                <span className="text-purple-600 dark:text-purple-400">⚡ /compact 压缩区</span>
              </div>
            </>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

const EVENT_CONFIG: Record<string, { icon: string; color: string }> = {
  start: { icon: "🟢", color: "text-green-600 dark:text-green-400" },
  growth: { icon: "📈", color: "text-blue-600 dark:text-blue-400" },
  peak: { icon: "⭐", color: "text-yellow-600 dark:text-yellow-400" },
  warning: { icon: "⚠️", color: "text-red-600 dark:text-red-400" },
  compact: { icon: "⚡", color: "text-purple-600 dark:text-purple-400" },
  end: { icon: "🔴", color: "text-gray-600 dark:text-gray-400" },
}

function AgentDetail({ stats, label, onClose, onNavigateToTurn }: { stats: AgentStats; label: string; onClose: () => void; onNavigateToTurn?: (turnId: string) => void }) {
  const peakTurn = stats.turns.reduce((max, t) => t.totalTokens > max.totalTokens ? t : max, stats.turns[0])
  const peakCacheRead = peakTurn.cacheReadTokens ?? 0
  const peakRealInput = peakTurn.inputTokens + peakTurn.cacheReadTokens + peakTurn.cacheWriteTokens
  const peakNewInput = peakTurn.inputTokens
  const peakCachePct = peakRealInput > 0 ? (peakCacheRead / peakRealInput * 100) : 0

  const totalSaved = stats.turns.reduce((s, t) => s + (t.cacheReadTokens ?? 0), 0)
  const totalInput = stats.turns.reduce((s, t) => s + t.totalTokens, 0)

  return (
    <Card size="sm">
      <CardHeader>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <Badge variant={stats.isRoot ? "blue" : "orange"} className="text-xs">{label}</Badge>
            <Badge variant="outline" className="text-xs">{stats.isRoot ? "主agent" : "subagent"}</Badge>
            <span className="text-xs text-muted-foreground">上下文详情</span>
            {onNavigateToTurn && (
              <Button
                variant="outline"
                size="sm"
                className="h-5 text-xs ml-auto"
                onClick={() => onNavigateToTurn(stats.turns[0].turnId)}
              >
                → Turns
              </Button>
            )}
          </div>
          <Button variant="ghost" size="sm" className="h-5 text-xs" onClick={onClose}>关闭</Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="space-y-1.5">
          <span className="text-xs font-medium">上下文组成 (第 #{peakTurn.turnIndex} 轮，峰值)</span>
          <div className="flex gap-0.5 h-3 rounded-full overflow-hidden bg-muted">
            <div className="h-full bg-yellow-500 rounded-l-full" style={{ width: `${peakCachePct}%` }} />
            <div className="h-full bg-blue-500 rounded-r-full" style={{ width: `${100 - peakCachePct}%`, minWidth: peakCachePct < 100 ? "2px" : "0" }} />
          </div>
          <div className="flex items-center gap-4 text-xs">
            <span className="text-yellow-600 dark:text-yellow-400">Cache Read: {formatTokenCount(peakCacheRead)} ({peakCachePct.toFixed(0)}%)</span>
            <span className="text-blue-600 dark:text-blue-400">New Input: {formatTokenCount(Math.max(peakNewInput, 0))} ({(100 - peakCachePct).toFixed(0)}%)</span>
            <span className="text-muted-foreground">Total: {formatTokenCount(peakRealInput)}</span>
          </div>
        </div>

        <div className="space-y-1">
          <span className="text-xs font-medium">关键事件</span>
          <div className="space-y-1">
            {stats.events.map((evt, i) => {
              const cfg = EVENT_CONFIG[evt.type] ?? EVENT_CONFIG.start
              return (
                <div key={i} className="flex items-center gap-2 text-xs px-2 py-1 rounded border">
                  <span className={cfg.color}>{cfg.icon}</span>
                  <span className="font-mono text-muted-foreground">#{evt.turnIndex}</span>
                  <span className="text-muted-foreground">{formatTimestamp(evt.timestamp)}</span>
                  <span className="tabular-nums">{formatTokenCount(evt.contextSize)}</span>
                  <span className="flex-1 truncate">{evt.note ?? evt.type}</span>
                </div>
              )
            })}
          </div>
        </div>

        <div className="space-y-1">
          <span className="text-xs font-medium">缓存效率</span>
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div className="border rounded p-2 text-center">
              <span className="text-muted-foreground">命中率</span>
              <span className="font-medium ml-1">{(stats.cacheHitRate * 100).toFixed(0)}%</span>
            </div>
            <div className="border rounded p-2 text-center">
              <span className="text-muted-foreground">节省</span>
              <span className="font-medium ml-1">{formatTokenCount(totalSaved)}</span>
            </div>
            <div className="border rounded p-2 text-center">
              <span className="text-muted-foreground">总消耗</span>
              <span className="font-medium ml-1">{formatTokenCount(totalInput)}</span>
            </div>
          </div>
          <p className="text-xs text-muted-foreground">
            如无缓存，总消耗将增加约 {totalInput > 0 ? ((totalSaved + totalInput) / totalInput).toFixed(1) : "1"} 倍
          </p>
        </div>
      </CardContent>
    </Card>
  )
}
