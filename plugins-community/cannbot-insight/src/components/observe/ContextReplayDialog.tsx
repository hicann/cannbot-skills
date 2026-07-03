"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useMemo, useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"

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
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

interface AgentTimeline {
  sessionId: string
  agentName: string
  isRoot: boolean
  color: string
  spawnTurn: number
  deathTurn: number
  points: Array<{
    turnIndex: number
    inputTokens: number
    contextWindowPct: number | null
    timestamp: string | null
  }>
}

function computeTimelines(turns: TurnRowItem[]): AgentTimeline[] {
  const sessionMap = new Map<string, TurnRowItem[]>()
  for (const t of turns) {
    if ((t.inputTokens + t.cacheReadTokens) <= 0) continue
    const sid = t.isSubagent ? (t.subagentSessionId ?? `anon-${t.turnIndex}`) : "root"
    if (!sessionMap.has(sid)) sessionMap.set(sid, [])
    sessionMap.get(sid)!.push(t)
  }

  const timelines: AgentTimeline[] = []

  for (const [sid, agentTurns] of sessionMap) {
    const agentName = agentTurns[0].agentName ?? "?"
    const isRoot = sid === "root"
    const color = AGENT_COLORS[agentName] ?? DEFAULT_AGENT_COLOR
    const spawnTurn = agentTurns[0].turnIndex
    const deathTurn = agentTurns[agentTurns.length - 1].turnIndex
    const points = agentTurns.map(t => ({
      turnIndex: t.turnIndex,
      inputTokens: t.inputTokens + t.cacheReadTokens,
      contextWindowPct: t.contextWindowPct,
      timestamp: t.createdAt,
    }))
    timelines.push({ sessionId: sid, agentName, isRoot, color, spawnTurn, deathTurn, points })
  }

  timelines.sort((a, b) => {
    if (a.isRoot !== b.isRoot) return a.isRoot ? -1 : 1
    return a.spawnTurn - b.spawnTurn
  })

  return timelines
}

function displayLabel(s: AgentTimeline, timelines: AgentTimeline[]): string {
  if (s.isRoot) return s.agentName
  const aname = s.agentName === "ascendc-kernel-architect" ? "ascendc-arch" : s.agentName === "ascendc-kernel-developer" ? "ascendc-dev" : s.agentName
  const idx = timelines.filter(a => a.agentName === s.agentName && !a.isRoot).indexOf(s) + 1
  return `${aname} #${idx}`
}

interface ContextReplayDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  turns: TurnRowItem[]
  sessionModel: string | null
}

export function ContextReplayDialog({ open, onOpenChange, turns, sessionModel }: ContextReplayDialogProps) {
  const [contextWindowLimit, setContextWindowLimit] = useState(200000)

  useEffect(() => {
    if (!open) return
    fetch(`/api/config/context-windows?model=${encodeURIComponent(sessionModel ?? "")}`)
      .then(r => r.json())
      .then(data => setContextWindowLimit(data.limit ?? 200000))
      .catch(() => setContextWindowLimit(200000))
  }, [sessionModel, open])

  const timelines = useMemo(() => computeTimelines(turns), [turns])

  if (!open || timelines.length === 0) return null

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="sm:max-w-[900px] max-w-[calc(100%-1rem)] max-h-[90vh] overflow-y-auto p-4 gap-2"
        showCloseButton={true}
      >
        <DialogTitle className="text-base">⏯ Context Replay — 上下文演变动画</DialogTitle>
        <DialogDescription className="text-xs text-muted-foreground">
          播放整个会话的上下文窗口变化过程，观察 root agent 和 subagent 的 context 生长与消亡
        </DialogDescription>
        <ContextReplayPlayer
          timelines={timelines}
          turns={turns}
          contextWindowLimit={contextWindowLimit}
        />
      </DialogContent>
    </Dialog>
  )
}

const SPEED_OPTIONS: Array<{ value: number; label: string }> = [
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" },
]

const BASE_DELAY_MS = 120

function ContextReplayPlayer({
  timelines,
  turns,
  contextWindowLimit,
}: {
  timelines: AgentTimeline[]
  turns: TurnRowItem[]
  contextWindowLimit: number
}) {
  const maxTurnIndex = Math.max(...turns.map(t => t.turnIndex), 0)
  const [currentTurn, setCurrentTurn] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [hoverPoint, setHoverPoint] = useState<{ sessionId: string; turnIndex: number; inputTokens: number } | null>(null)
  const animRef = useRef<number | null>(null)
  const lastTickRef = useRef<number>(0)

  useEffect(() => {
    if (!isPlaying) {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current)
        animRef.current = null
      }
      return
    }

    const delay = BASE_DELAY_MS / speed

    function tick(now: number) {
      if (now - lastTickRef.current >= delay) {
        lastTickRef.current = now
        setCurrentTurn(prev => prev + 1)
      }
      animRef.current = requestAnimationFrame(tick)
    }

    lastTickRef.current = performance.now()
    animRef.current = requestAnimationFrame(tick)

    return () => {
      if (animRef.current) {
        cancelAnimationFrame(animRef.current)
        animRef.current = null
      }
    }
  }, [isPlaying, speed])

  useEffect(() => {
    if (currentTurn >= maxTurnIndex && isPlaying) {
      setIsPlaying(false)
    }
  }, [currentTurn, maxTurnIndex, isPlaying])

  function handlePlay() {
    if (currentTurn >= maxTurnIndex) {
      setCurrentTurn(0)
    }
    setIsPlaying(true)
  }

  function handlePause() {
    setIsPlaying(false)
  }

  function handleReset() {
    setIsPlaying(false)
    setCurrentTurn(0)
  }

  const SVG_HEIGHT = 420
  const PADDING_TOP = 20
  const PADDING_BOTTOM = 30
  const PADDING_LEFT = 55
  const PADDING_RIGHT = 20
  const chartWidth = 1100
  const plotWidth = chartWidth - PADDING_LEFT - PADDING_RIGHT
  const plotHeight = SVG_HEIGHT - PADDING_TOP - PADDING_BOTTOM

  const currentTurnPoints = useMemo(() => {
    return timelines.map(tl => {
      const visible = tl.points.filter(p => p.turnIndex <= currentTurn)
      const isAlive = currentTurn <= tl.deathTurn || tl.isRoot
      return { ...tl, visiblePoints: visible, isAlive }
    })
  }, [timelines, currentTurn])

  const allVisibleTokens = currentTurnPoints.flatMap(tl => tl.visiblePoints.map(p => p.inputTokens))
  const yMax = Math.min(
    Math.ceil((Math.max(...allVisibleTokens, 1) * 1.15) / 10000) * 10000,
    contextWindowLimit * 1.1
  )

  function toX(turnIdx: number): number {
    if (maxTurnIndex === 0) return PADDING_LEFT + plotWidth / 2
    return PADDING_LEFT + (turnIdx / maxTurnIndex) * plotWidth
  }
  function toY(tokens: number): number {
    return PADDING_TOP + plotHeight - (tokens / yMax) * plotHeight
  }

  const yTicks: Array<{ value: number; y: number; label: string }> = []
  const yStep = yMax <= 20000 ? 5000 : yMax <= 100000 ? 20000 : 40000
  for (let v = 0; v <= yMax; v += yStep) {
    yTicks.push({ value: v, y: toY(v), label: formatTokenCount(v) })
  }

  const xStep = maxTurnIndex <= 20 ? 1 : maxTurnIndex <= 50 ? 5 : maxTurnIndex <= 100 ? 10 : maxTurnIndex <= 200 ? 20 : 50
  const xTicks: Array<{ value: number; x: number; label: string }> = []
  for (let v = 0; v <= maxTurnIndex; v += xStep) {
    xTicks.push({ value: v, x: toX(v), label: `${v}` })
  }

  const limitY = toY(contextWindowLimit)
  const scanX = toX(currentTurn)

  const currentTurnData = useMemo(() => {
    const turnRows = turns.filter(t => t.turnIndex === currentTurn)
    return turnRows
  }, [turns, currentTurn])

  return (
    <div className="space-y-2">
      <div className="relative">
        <svg
          width="100%"
          height={SVG_HEIGHT}
          viewBox={`0 0 ${chartWidth} ${SVG_HEIGHT}`}
          preserveAspectRatio="xMidYMid meet"
          className="min-w-[500px]"
        >
          {yTicks.map(tick => (
            <g key={tick.value}>
              <line x1={PADDING_LEFT} y1={tick.y} x2={chartWidth - PADDING_RIGHT} y2={tick.y} stroke="#e5e7eb" strokeWidth={0.5} />
              <text x={PADDING_LEFT - 5} y={tick.y + 4} textAnchor="end" fontSize={9} fill="#6b7280">{tick.label}</text>
            </g>
          ))}

          {xTicks.map(tick => (
            <g key={tick.value}>
              <line x1={tick.x} y1={PADDING_TOP} x2={tick.x} y2={SVG_HEIGHT - PADDING_BOTTOM} stroke="#e5e7eb" strokeWidth={0.5} />
              <text x={tick.x} y={SVG_HEIGHT - 8} textAnchor="middle" fontSize={9} fill="#6b7280">{tick.label}</text>
            </g>
          ))}

          {contextWindowLimit < yMax && (
            <line
              x1={PADDING_LEFT}
              y1={limitY}
              x2={chartWidth - PADDING_RIGHT}
              y2={limitY}
              stroke="#ef4444"
              strokeWidth={1}
              strokeDasharray="6 3"
            />
          )}

          {currentTurnPoints.map(tl => {
            const pts = tl.visiblePoints
            if (pts.length === 0) return null

            const label = displayLabel(tl, timelines)
            const isDead = !tl.isAlive && currentTurn > tl.deathTurn
            const activeColor = isDead ? "#9ca3af" : tl.color
            const strokeW = tl.isRoot ? 2.5 : 1.5
            const opacity = isDead ? 0.4 : 1
            const dashArray = isDead ? "4 2" : undefined

            const polyline = pts.map(p => `${toX(p.turnIndex)},${toY(p.inputTokens)}`).join(" ")

            return (
              <g key={tl.sessionId} opacity={opacity}>
                <polyline
                  points={polyline}
                  fill="none"
                  stroke={activeColor}
                  strokeWidth={strokeW}
                  strokeLinejoin="round"
                  strokeDasharray={dashArray}
                />
                {pts.map(p => (
                  <circle
                    key={`${tl.sessionId}-${p.turnIndex}`}
                    cx={toX(p.turnIndex)}
                    cy={toY(p.inputTokens)}
                    r={tl.isRoot ? 3 : 2}
                    fill={activeColor}
                    stroke="white"
                    strokeWidth={1}
                    onMouseEnter={() => setHoverPoint({ sessionId: tl.sessionId, turnIndex: p.turnIndex, inputTokens: p.inputTokens })}
                    onMouseLeave={() => setHoverPoint(null)}
                  >
                    <title>{label} #{p.turnIndex} | {formatTokenCount(p.inputTokens)} | {(p.contextWindowPct ?? 0).toFixed(1)}%</title>
                  </circle>
                ))}

                {!tl.isRoot && pts.length > 0 && (
                  <g>
                    <circle
                      cx={toX(tl.spawnTurn)}
                      cy={toY(pts[0].inputTokens)}
                      r={5}
                      fill="#10b981"
                      stroke="white"
                      strokeWidth={1.5}
                    />
                    <text
                      x={toX(tl.spawnTurn) + 8}
                      y={toY(pts[0].inputTokens) - 2}
                      fontSize={8}
                      fill="#10b981"
                      fontWeight="bold"
                    >
                      🟢 {label}
                    </text>
                  </g>
                )}

                {!tl.isRoot && isDead && tl.points.length > 0 && (
                  <g>
                    <circle
                      cx={toX(tl.deathTurn)}
                      cy={toY(tl.points[tl.points.length - 1].inputTokens)}
                      r={5}
                      fill="#ef4444"
                      stroke="white"
                      strokeWidth={1.5}
                    />
                    <text
                      x={toX(tl.deathTurn) + 8}
                      y={toY(tl.points[tl.points.length - 1].inputTokens) - 2}
                      fontSize={8}
                      fill="#ef4444"
                      fontWeight="bold"
                    >
                      ✕
                    </text>
                  </g>
                )}
              </g>
            )
          })}

          <line
            x1={scanX}
            y1={PADDING_TOP}
            x2={scanX}
            y2={SVG_HEIGHT - PADDING_BOTTOM}
            stroke="#3b82f6"
            strokeWidth={1.5}
            opacity={0.3}
            strokeDasharray="3 3"
          />

          {hoverPoint && (
            <g>
              <circle
                cx={toX(hoverPoint.turnIndex)}
                cy={toY(hoverPoint.inputTokens)}
                r={6}
                fill="none"
                stroke="#3b82f6"
                strokeWidth={2}
              />
              <rect
                x={toX(hoverPoint.turnIndex) + 10}
                y={toY(hoverPoint.inputTokens) - 14}
                width={120}
                height={22}
                rx={4}
                fill="white"
                stroke="#e5e7eb"
                strokeWidth={1}
              />
              <text
                x={toX(hoverPoint.turnIndex) + 16}
                y={toY(hoverPoint.inputTokens)}
                fontSize={9}
                fill="#1f2937"
              >
                #{hoverPoint.turnIndex} {formatTokenCount(hoverPoint.inputTokens)}
              </text>
            </g>
          )}
        </svg>
      </div>

      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          className="h-6 text-xs"
          onClick={isPlaying ? handlePause : handlePlay}
        >
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </Button>
        <Button
          size="sm"
          variant="outline"
          className="h-6 text-xs"
          onClick={handleReset}
        >
          ↺ Reset
        </Button>

        <div className="flex items-center gap-1">
          <span className="text-xs text-muted-foreground">Speed:</span>
          {SPEED_OPTIONS.map(opt => (
            <Button
              key={opt.value}
              size="sm"
              variant={speed === opt.value ? "default" : "outline"}
              className="h-6 text-xs px-2"
              onClick={() => setSpeed(opt.value)}
            >
              {opt.label}
            </Button>
          ))}
        </div>

        <div className="flex-1 flex items-center gap-2">
          <span className="text-xs text-muted-foreground tabular-nums shrink-0">
            Turn {currentTurn}/{maxTurnIndex}
          </span>
          <input
            type="range"
            min={0}
            max={maxTurnIndex}
            value={currentTurn}
            onChange={(e) => {
              setIsPlaying(false)
              setCurrentTurn(Number(e.target.value))
            }}
            className="flex-1 h-1.5 accent-blue-500"
          />
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5 items-center border-t pt-2 max-h-[80px] overflow-y-auto">
        <span className="text-xs text-muted-foreground shrink-0">Agents ({timelines.length}):</span>
        {currentTurnPoints.map(tl => {
          const label = displayLabel(tl, timelines)
          const isDead = !tl.isAlive && currentTurn > tl.deathTurn
          const hasTokens = tl.visiblePoints.length > 0
          return (
            <div key={tl.sessionId} className="flex items-center gap-1 text-xs">
              <span className="inline-block w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: isDead ? "#9ca3af" : tl.color, opacity: isDead ? 0.4 : 1 }} />
              <span className={cn(hasTokens ? "font-medium" : "text-muted-foreground", tl.isRoot && hasTokens && "text-blue-600 dark:text-blue-400")}>
                {label}
              </span>
              {hasTokens && (
                <span className="text-muted-foreground tabular-nums">
                  {formatTokenCount(tl.visiblePoints[tl.visiblePoints.length - 1].inputTokens)}
                </span>
              )}
              {!tl.isRoot && isDead && (
                <Badge variant="red" className="text-xs">✕ dead</Badge>
              )}
              {!tl.isRoot && !isDead && tl.visiblePoints.length > 0 && (
                <Badge variant="green" className="text-xs">🟢 active</Badge>
              )}
              {!tl.isRoot && !isDead && tl.visiblePoints.length === 0 && currentTurn < tl.spawnTurn && (
                <Badge variant="outline" className="text-xs">waiting</Badge>
              )}
            </div>
          )
        })}
        {contextWindowLimit < yMax && (
          <div className="flex items-center gap-1.5 text-xs ml-2">
            <span className="inline-block w-4 h-0 border-t-2 border-dashed border-red-500" />
            <span className="text-red-500">Limit ({formatTokenCount(contextWindowLimit)})</span>
          </div>
        )}
      </div>

      {currentTurnData.length > 0 && (
        <div className="border rounded-lg px-3 py-2 space-y-1">
          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium">Turn #{currentTurn}</span>
            {currentTurnData.map(t => (
              <Badge key={t.turnId} variant={t.isSubagent ? "orange" : "blue"} className="text-xs">
                {t.isSubagent ? (t.subagentName ?? t.agentName ?? "sub") : (t.agentName ?? "root")}
              </Badge>
            ))}
          </div>
          <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
            {currentTurnData.map(t => (
              <span key={t.turnId}>
                {t.agentName ?? "?"}: {formatTokenCount(t.inputTokens)} input
                {t.contextWindowPct != null && t.contextWindowPct > 0 && ` (${t.contextWindowPct.toFixed(1)}%)`}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
