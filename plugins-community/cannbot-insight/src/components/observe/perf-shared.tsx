"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"

export interface TurnPerf {
  turnId: string
  turnIndex: number
  role: string
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  contentSummary: string | null
  createdAt: string | null
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  contextWindowPct: number | null
  latencyMs: number
  model: string | null
  toolCalls: Array<{
    toolName: string
    durationMs: number
    state: string
  }>
}

export interface BridgePerf {
  bridgeId: string
  dispatchTurnId: string | null
  dispatchContent: string | null
  subagentSessionId: string | null
  subagentName: string | null
  subagentLatencyMs: number
  subagentTokens: number
}

export interface SubagentStat {
  sessionId: string
  name: string
  taskDescription: string | null
  turns: TurnPerf[]
  totalMs: number
  totalTokens: number
  totalGenTokens: number
  tps: number
  cacheRead: number
  inputTokens: number
  cacheHitRate: number
}

function fmtMs(ms: number): string {
  if (ms <= 0) return "0s"
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(1)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  if (m < 60) return `${m}m${rem}s`
  const h = Math.floor(m / 60)
  return `${h}h${m % 60}m`
}

function fmtTok(n: number): string {
  if (n === 0) return "0"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return `${n}`
}

function categorizeTool(name: string): string {
  const n = name.toLowerCase()
  if (["read", "glob", "grep", "webfetch", "search", "list"].some(t => n.includes(t))) return "分析"
  if (["write", "edit", "create", "str_replace"].some(t => n.includes(t))) return "编写"
  if (["bash", "terminal", "shell", "test", "run", "python", "pytest"].some(t => n.includes(t))) return "执行"
  return "其他"
}

const ACTIVITY_COLOR: Record<string, string> = {
  "LLM思考": "#8b5cf6",
  "分析": "#2563eb",
  "编写": "#16a34a",
  "执行": "#f59e0b",
  "其他": "#6b7280",
}

function computeTurnSegments(t: TurnPerf): { label: string; ms: number; color: string }[] {
  const tools = t.toolCalls ?? []
  const toolMs = tools.reduce((sum, tc) => sum + Math.max(0, tc.durationMs), 0)

  if (toolMs > 0) {
    const llmMs = Math.max(0, t.latencyMs - toolMs)
    return [
      { label: "LLM思考", ms: llmMs, color: ACTIVITY_COLOR["LLM思考"] },
      { label: "工具调用", ms: toolMs, color: ACTIVITY_COLOR["分析"] },
    ].filter(s => s.ms > 0)
  }

  if (tools.length === 0) {
    return [{ label: "LLM思考", ms: t.latencyMs, color: ACTIVITY_COLOR["LLM思考"] }]
  }

  const tokens = (t.outputTokens ?? 0) + (t.reasoningTokens ?? 0)
  const estLlmMs = Math.min(t.latencyMs * 0.6, Math.max(500, tokens * 10))
  const estToolMs = Math.max(0, t.latencyMs - estLlmMs)

  const cats = tools.map(tc => categorizeTool(tc.toolName))
  const cat = cats.sort((a, b) =>
    cats.filter(c => c === b).length - cats.filter(c => c === a).length
  )[0]

  return [
    { label: "LLM思考", ms: estLlmMs, color: ACTIVITY_COLOR["LLM思考"] },
    { label: cat, ms: estToolMs, color: ACTIVITY_COLOR[cat] ?? ACTIVITY_COLOR["其他"] },
  ].filter(s => s.ms > 0)
}

const COLORS = {
  root: "#2563eb",
  sub: "#16a34a",
  input: "#2563eb",
  output: "#16a34a",
  reasoning: "#8b5cf6",
  cacheRead: "#f59e0b",
  cacheWrite: "#6b7280",
  danger: "#dc2626",
  axis: "#94a3b8",
  grid: "#e2e8f0",
}

export function PerfBenchmarkChart({ points, onJumpToTurn }: {
  points: { turnIndex: number; createdAt: string | null; tpot: number; tps: number }[]
  onJumpToTurn?: (turn: number) => void
}) {
  if (points.length === 0) {
    return <div className="flex items-center justify-center h-32 text-xs text-muted-foreground">暂无性能数据</div>
  }

  const tsPoints = points.map(p => ({ ...p, ts: p.createdAt ? new Date(p.createdAt).getTime() : 0 }))

  const fmtTime = (ts: number) => {
    const d = new Date(ts)
    return `${d.getHours().toString().padStart(2, "0")}:${d.getMinutes().toString().padStart(2, "0")}:${d.getSeconds().toString().padStart(2, "0")}`
  }

  const metrics = [
    { key: "tps" as const, label: "TPS", desc: "Tokens per Second", color: "#16a34a", unit: "tok/s" },
    { key: "tpot" as const, label: "TPOT", desc: "Time per Output Token", color: "#8b5cf6", unit: "ms" },
  ]

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
      {metrics.map(m => {
        const W = 480, H = 200
        const padL = 40, padR = 14, padT = 20, padB = 32
        const chartW = W - padL - padR
        const chartH = H - padT - padB
        const n = tsPoints.length
        const vals = tsPoints.map(p => p[m.key])
        const maxVal = Math.max(...vals, 1)
        const minVal = Math.min(...vals)
        const avgVal = vals.reduce((s, v) => s + v, 0) / vals.length
        // Cap the y-axis at the 95th percentile so a single outlier doesn't
        // squash every other bar to near-zero. Bars above the cap are clamped
        // to the top; true min/avg/max are still shown in the header.
        const sortedVals = [...vals].sort((a, b) => a - b)
        const p95 = sortedVals[Math.floor(sortedVals.length * 0.95)] ?? maxVal
        const yMaxRaw = Math.min(maxVal, Math.max(p95, minVal))
        const yMax = (m.unit === "ms" ? Math.ceil(yMaxRaw / 100) * 100 : Math.ceil(yMaxRaw / 50) * 50) || 1
        const slot = chartW / n
        const barW = Math.min(slot * 0.7, 14)
        const xOf = (i: number) => padL + slot * i + slot / 2
        const toY = (v: number) => padT + chartH - (Math.min(v, yMax) / yMax) * chartH
        const avgY = toY(avgVal)
        const yTicks = Array.from({ length: 4 }, (_, i) => Math.round(yMax * i / 3))
        const fmtVal = (v: number) => v >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(0)

        return (
          <div key={m.key} className="rounded border bg-muted/5 overflow-hidden">
            <div className="flex items-center justify-between px-3 py-1.5 border-b bg-muted/10">
              <div className="flex items-center gap-2">
                <span className="inline-block w-2 h-2" style={{ background: m.color }} />
                <span className="text-xs font-mono font-semibold">{m.label}</span>
                <span className="text-[10px] text-muted-foreground font-mono">{m.desc}</span>
              </div>
              <div className="flex items-center gap-3 text-[10px] font-mono text-muted-foreground">
                <span>min <span className="text-foreground">{fmtVal(minVal)}</span></span>
                <span>avg <span className="text-foreground">{fmtVal(avgVal)}</span></span>
                <span>max <span className="text-foreground">{fmtVal(maxVal)}</span></span>
              </div>
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} width="100%" height={H} className="block">
              {yTicks.map((v, i) => {
                const y = padT + chartH - (v / yMax) * chartH
                return (
                  <g key={i}>
                    <line x1={padL} y1={y} x2={W - padR} y2={y} stroke={COLORS.grid} strokeWidth={0.5} strokeDasharray="2 3" />
                    <text x={padL - 5} y={y + 3} textAnchor="end" fontSize="8" className="fill-muted-foreground" fontFamily="monospace">{fmtVal(v)}</text>
                  </g>
                )
              })}
              <line x1={padL} y1={avgY} x2={W - padR} y2={avgY} stroke={m.color} strokeWidth={0.8} strokeDasharray="4 2" opacity={0.6} />
              <text x={W - padR} y={avgY - 2} textAnchor="end" fontSize="8" className="fill-muted-foreground" fontFamily="monospace">avg {fmtVal(avgVal)}</text>
              <line x1={padL} y1={padT} x2={padL} y2={padT + chartH} stroke={COLORS.axis} strokeWidth={0.8} />
              <line x1={padL} y1={padT + chartH} x2={W - padR} y2={padT + chartH} stroke={COLORS.axis} strokeWidth={0.8} />
              {tsPoints.map((p, i) => {
                const y = toY(p[m.key])
                const h = (padT + chartH) - y
                return (
                  <rect key={i} x={xOf(i) - barW / 2} y={y} width={barW} height={Math.max(h, 0.5)}
                    fill={m.color} opacity={0.85} rx={1}
                    className="cursor-pointer" onClick={() => onJumpToTurn?.(p.turnIndex)}>
                    <title>{`${m.label} turn #${p.turnIndex} (${fmtTime(p.ts)}): ${fmtVal(p[m.key])} ${m.unit}`}</title>
                  </rect>
                )
              })}
              {tsPoints.length > 0 && (
                <text key="first" x={xOf(0)} y={H - 6} textAnchor="middle" fontSize="8" className="fill-muted-foreground" fontFamily="monospace">
                  turn {tsPoints[0].turnIndex}
                </text>
              )}
              {tsPoints.length > 1 && (
                <text key="last" x={xOf(n - 1)} y={H - 6} textAnchor="middle" fontSize="8" className="fill-muted-foreground" fontFamily="monospace">
                  turn {tsPoints[n - 1].turnIndex}
                </text>
              )}
              <text x={padL + chartW / 2} y={H - 0} textAnchor="middle" fontSize="8" className="fill-muted-foreground" fontFamily="monospace">turn</text>
              <text x={12} y={padT + chartH / 2} textAnchor="middle" fontSize="8" className="fill-muted-foreground" fontFamily="monospace"
                transform={`rotate(-90 12 ${padT + chartH / 2})`}>{m.unit}</text>
            </svg>
          </div>
        )
      })}
    </div>
  )
}

type SortKey = "ms" | "tokens" | "order"

const COLS: { key: SortKey; label: string }[] = [
  { key: "ms", label: "TIME" },
  { key: "tokens", label: "TOKENS" },
  { key: "order", label: "SEQUENCE" },
]

export function PerfTopTable({ subagents, onJumpToTurn }: {
  subagents: SubagentStat[]
  onJumpToTurn?: (turn: number) => void
}) {
  const [sortKey, setSortKey] = useState<SortKey>("ms")
  const [expanded, setExpanded] = useState<string | null>(null)
  const [showAllTurns, setShowAllTurns] = useState<Set<string>>(new Set())
  const [showBreakdown, setShowBreakdown] = useState<Set<string>>(new Set())
  const [showAllSubagents, setShowAllSubagents] = useState(false)

  const getVal = (s: SubagentStat, key: SortKey): number =>
    key === "ms" ? s.totalMs : key === "tokens" ? s.totalTokens : s.turns[0]?.turnIndex ?? 0

  const sorted = [...subagents].sort((a, b) => {
    if (sortKey === "order") return getVal(a, sortKey) - getVal(b, sortKey)
    return getVal(b, sortKey) - getVal(a, sortKey)
  })
  const totalVal = sorted.reduce((s, sub) => s + getVal(sub, sortKey), 0)
  const barColor = sortKey === "ms" ? COLORS.reasoning : sortKey === "tokens" ? COLORS.input : COLORS.output

  const sortTurns = (turns: TurnPerf[]): TurnPerf[] => {
    if (sortKey === "order") return [...turns].sort((a, b) => a.turnIndex - b.turnIndex)
    const val = (t: TurnPerf): number =>
      sortKey === "tokens" ? t.totalTokens : t.latencyMs
    return [...turns].sort((a, b) => val(b) - val(a))
  }

  return (
    <div className="rounded-lg border overflow-x-auto">
      <div className="min-w-[560px]">
        {/* Header */}
        <div className="grid grid-cols-[1fr_52px_80px_80px_80px_56px] bg-muted/30 px-3 py-2 text-xs font-semibold text-muted-foreground border-b">
          <span>SUBAGENT</span>
          <span></span>
          {COLS.map(col => (
            <span
              key={col.key}
              className="flex justify-center items-center gap-0.5 cursor-pointer hover:text-foreground select-none transition-colors whitespace-nowrap"
              onClick={() => setSortKey(col.key)}
            >
              {col.label}
              <span className={sortKey === col.key ? "opacity-100" : "opacity-0"}>{col.key === "order" ? "↑" : "↓"}</span>
            </span>
          ))}
          <span className="flex justify-center">TPS</span>
        </div>

        {/* Rows */}
        {(showAllSubagents ? sorted : sorted.slice(0, 10)).map(s => {
          const isOpen = expanded === s.sessionId
          const pct = totalVal > 0 ? (getVal(s, sortKey) / totalVal) * 100 : 0
          return (
            <div key={s.sessionId} className="border-b last:border-0">
              <div
                className="grid grid-cols-[1fr_52px_80px_80px_80px_56px] px-3 py-2 cursor-pointer hover:bg-muted/20 text-xs tabular-nums transition-colors items-center"
                onClick={() => setExpanded(isOpen ? null : s.sessionId)}
              >
                <span className="flex flex-col gap-1 pr-2">
                  <span className="flex items-center gap-1.5 truncate">
                    <span className="text-[10px] text-muted-foreground inline-block transition-transform duration-200"
                          style={{ transform: isOpen ? "rotate(90deg)" : "rotate(0deg)" }}>▶</span>
                    <span className="truncate">{s.name}</span>
                  </span>
                  {s.taskDescription && (
                    <span className="text-[10px] text-muted-foreground truncate ml-4">{s.taskDescription}</span>
                  )}
                  {sortKey !== "order" && (
                    <div className="h-2.5 bg-muted/30 rounded overflow-hidden ml-4">
                      <div className="h-full rounded transition-all" style={{ width: `${Math.max(pct, 1)}%`, background: barColor }} />
                    </div>
                  )}
                </span>
                <span className="text-center text-muted-foreground">{sortKey === "order" ? "—" : `${pct.toFixed(1)}%`}</span>
                <span className="text-center font-semibold" style={{ color: COLORS.reasoning }}>{fmtMs(s.totalMs)}</span>
                <span className="text-center font-semibold" style={{ color: COLORS.input }}>{fmtTok(s.totalTokens)}</span>
                <span className="text-center text-muted-foreground">#{s.turns[0]?.turnIndex ?? '?'}</span>
                <span className="text-center" style={{ color: COLORS.reasoning }}>{s.tps.toFixed(1)}</span>
              </div>

              {/* Drill-down: turns of this subagent */}
              {isOpen && (() => {
                const allTurns = sortTurns(s.turns)
                const isShowAll = showAllTurns.has(s.sessionId)
                const visibleTurns = isShowAll ? allTurns : allTurns.slice(0, 10)
                const hasMore = allTurns.length > 10
                return (
                  <div className="bg-muted/5 border-l-2" style={{ borderColor: COLORS.grid }}>
                    {visibleTurns.map(t => {
                      const segments = computeTurnSegments(t)
                      const segTotal = segments.reduce((sum, seg) => sum + seg.ms, 0)
                      return (
                        <div
                          key={t.turnId}
                          className="flex flex-col gap-1 px-3 py-1.5 cursor-pointer hover:bg-muted/30 text-[11px] tabular-nums border-b last:border-0 border-muted/20"
                          onClick={() => onJumpToTurn?.(t.turnIndex)}
                        >
                          <div className="flex items-center gap-3">
                            <span className="font-mono text-muted-foreground w-10 text-center">#{t.turnIndex}</span>
                            <span className="font-semibold w-20 text-center" style={{ color: COLORS.reasoning }}>
                              {fmtMs(t.latencyMs)} <span className="text-[10px] text-muted-foreground font-normal">({s.totalMs > 0 ? `${(t.latencyMs / s.totalMs * 100).toFixed(0)}%` : "—"})</span>
                            </span>
                            <span className="w-16 text-center" style={{ color: COLORS.input }}>{fmtTok(t.totalTokens)}</span>
                            <span className="w-24 text-center whitespace-nowrap" style={{
                              color: t.cacheReadTokens + t.inputTokens > 0 && (t.cacheReadTokens / (t.cacheReadTokens + t.inputTokens)) * 100 < 50
                                ? COLORS.danger : COLORS.output
                            }}
                              title={`Cache read: ${fmtTok(t.cacheReadTokens)} / Non-cached input (miss): ${fmtTok(t.inputTokens)}`}
                            >
                              {t.cacheReadTokens + t.inputTokens > 0
                                ? `Cache ${(t.cacheReadTokens / (t.cacheReadTokens + t.inputTokens) * 100).toFixed(2)}%`
                                : "—"}
                            </span>
                            <span className="w-20 text-center whitespace-nowrap" style={{ color: COLORS.reasoning }}>
                              {t.latencyMs > 0 ? `TPS ${(((t.outputTokens ?? 0) + (t.reasoningTokens ?? 0)) / (t.latencyMs / 1000)).toFixed(1)}` : "—"}
                            </span>
                            <span className="text-muted-foreground truncate flex-1">
                              {(t.toolCalls?.length ?? 0)} tools · {t.model ?? "?"}
                            </span>
                            {segTotal > 0 && (
                              <span
                                className="text-[10px] text-muted-foreground cursor-pointer hover:text-foreground select-none whitespace-nowrap"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setShowBreakdown(prev => {
                                    const next = new Set(prev)
                                    if (next.has(t.turnId)) next.delete(t.turnId)
                                    else next.add(t.turnId)
                                    return next
                                  })
                                }}
                              >
                                {showBreakdown.has(t.turnId) ? "▼" : "▶"} 活动
                              </span>
                            )}
                          </div>
                          {segTotal > 0 && showBreakdown.has(t.turnId) && (
                            <div className="flex items-center gap-4 pl-10 py-1">
                              <svg viewBox="0 0 80 80" width="80" height="80" className="flex-shrink-0">
                                <g transform="rotate(-90 40 40)">
                                  {(() => {
                                    const radius = 30
                                    const strokeWidth = 10
                                    const circumference = 2 * Math.PI * radius
                                    let offset = 0
                                    return segments.map((seg, i) => {
                                      const frac = seg.ms / segTotal
                                      const length = frac * circumference
                                      const circle = (
                                        <circle key={i} cx="40" cy="40" r={radius} fill="none"
                                          stroke={seg.color} strokeWidth={strokeWidth}
                                          strokeDasharray={`${length} ${circumference - length}`}
                                          strokeDashoffset={-offset} />
                                      )
                                      offset += length
                                      return circle
                                    })
                                  })()}
                                </g>
                                <text x="40" y="44" textAnchor="middle" fontSize="12" fontWeight="700" className="fill-foreground">
                                  {fmtMs(segTotal)}
                                </text>
                              </svg>
                              <div className="flex flex-col gap-1.5">
                                {segments.map(seg => {
                                  const desc = seg.label === "分析" ? "read/grep/glob 读取搜索"
                                    : seg.label === "编写" ? "write/edit 文件修改"
                                    : seg.label === "执行" ? "bash/test 命令执行"
                                    : seg.label === "LLM思考" ? "无工具纯推理"
                                    : ""
                                  return (
                                    <span key={seg.label} className="flex items-center gap-1.5 text-[10px]">
                                      <span className="inline-block w-2.5 h-2.5 rounded-sm flex-shrink-0" style={{ background: seg.color }} />
                                      <span className="text-muted-foreground w-14">{seg.label}</span>
                                      <span className="font-semibold tabular-nums w-12">{fmtMs(seg.ms)}</span>
                                      <span className="text-muted-foreground tabular-nums w-10">({(seg.ms / segTotal * 100).toFixed(0)}%)</span>
                                      <span className="text-muted-foreground/60 text-[9px]">{desc}</span>
                                    </span>
                                  )
                                })}
                              </div>
                            </div>
                          )}
                        </div>
                      )
                    })}
                    {hasMore && (
                      <div
                        className="px-4 py-1.5 text-[11px] text-blue-600 cursor-pointer hover:underline select-none"
                        onClick={(e) => {
                          e.stopPropagation()
                          setShowAllTurns(prev => {
                            const next = new Set(prev)
                            if (next.has(s.sessionId)) next.delete(s.sessionId)
                            else next.add(s.sessionId)
                            return next
                          })
                        }}
                      >
                        {isShowAll ? "收起" : `展开全部 (${allTurns.length} turns)`}
                      </div>
                    )}
                  </div>
                )
              })()}
            </div>
          )
        })}

        {sorted.length > 10 && (
          <div
            className="px-3 py-1.5 text-xs text-blue-600 cursor-pointer hover:underline select-none text-center"
            onClick={() => setShowAllSubagents(!showAllSubagents)}
          >
            {showAllSubagents ? "收起" : `展开全部 (${sorted.length} subagents)`}
          </div>
        )}

        {sorted.length === 0 && (
          <div className="px-3 py-6 text-center text-xs text-muted-foreground">暂无子代理数据</div>
        )}
      </div>
    </div>
  )
}
