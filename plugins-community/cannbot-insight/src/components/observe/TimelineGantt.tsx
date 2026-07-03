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
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface ExecutionItem {
  executionId: string
  agentName: string | null
  agentSessionId: string | null
  isSubagent: boolean
  subagentType: string | null
  subagentName: string | null
  parentExecutionId: string | null
  tokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cost: number
  latencyMs: number
  toolCallCount: number
  toolCallErrorCount: number
  skillLoadCount: number
  skillInvokeCount: number
  llmCallCount: number
  model: string | null
  createdAt: string
  skills: Array<{ skillName: string; skillVersion: number | null; isPrimary: boolean }>
}

interface PhaseOverlay {
  phaseIndex: number
  phaseName: string
  startTime: string | null
  endTime: string | null
}

interface CheckpointOverlay {
  checkpointLabel: string
  checkpointType: "block" | "info"
  requestedAt: string | null
  approvedAt: string | null
  waitTimeMs: number
}

interface TimelineGanttProps {
  executions: ExecutionItem[]
  sessionStartTime: string
  sessionEndTime: string | null
  sessionLatencyMs: number
  workflowPhases: PhaseOverlay[] | null
  workflowCheckpoints: CheckpointOverlay[] | null
}

const TYPE_PALETTE: Record<string, { bar: string; hover: string; bg: string; text: string }> = {
  build: { bar: "bg-blue-500", hover: "bg-blue-400", bg: "bg-blue-50 dark:bg-blue-500/5", text: "text-blue-700 dark:text-blue-400" },
  ascendc_kernel_architect: { bar: "bg-emerald-500", hover: "bg-emerald-400", bg: "bg-emerald-50 dark:bg-emerald-500/5", text: "text-emerald-700 dark:text-emerald-400" },
  ascendc_kernel_developer: { bar: "bg-violet-500", hover: "bg-violet-400", bg: "bg-violet-50 dark:bg-violet-500/5", text: "text-violet-700 dark:text-violet-400" },
  general: { bar: "bg-gray-500", hover: "bg-gray-400", bg: "bg-gray-50 dark:bg-gray-500/5", text: "text-gray-700 dark:text-gray-400" },
  root: { bar: "bg-sky-600", hover: "bg-sky-500", bg: "bg-sky-50 dark:bg-sky-500/5", text: "text-sky-700 dark:text-sky-400" },
}

const PHASE_COLORS: Record<number, string> = {
  1: "bg-blue-200/40 dark:bg-blue-500/10",
  2: "bg-emerald-200/40 dark:bg-emerald-500/10",
  3: "bg-orange-200/40 dark:bg-orange-500/10",
  4: "bg-violet-200/40 dark:bg-violet-500/10",
  5: "bg-pink-200/40 dark:bg-pink-500/10",
}

function getTypeColor(type: string | null, isSubagent: boolean) {
  if (!isSubagent) return TYPE_PALETTE.root
  const key = type?.toLowerCase().replace(/-/g, "_") ?? "general"
  return TYPE_PALETTE[key] ?? TYPE_PALETTE.general
}

function formatLatency(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
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
  if (totalMs <= 10000) return 1000
  if (totalMs <= 60000) return 5000
  if (totalMs <= 300000) return 30000
  if (totalMs <= 1800000) return 60000
  if (totalMs <= 7200000) return 300000
  return 600000
}

export function TimelineGantt({
  executions,
  sessionStartTime,
  sessionLatencyMs,
  workflowPhases,
  workflowCheckpoints,
}: TimelineGanttProps) {
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [hoveredId, setHoveredId] = useState<string | null>(null)
  const [showAbsolute, setShowAbsolute] = useState(false)
  const [filterTypes, setFilterTypes] = useState<Set<string>>(new Set())

  const rootExec = executions.find(e => !e.isSubagent)
  const subExecs = executions.filter(e => e.isSubagent)
  const totalDuration = sessionLatencyMs || (rootExec?.latencyMs ?? 1)
  const sessionStartMs = new Date(sessionStartTime).getTime()
  const rootStartMs = rootExec ? new Date(rootExec.createdAt).getTime() : sessionStartMs

  const typeGroups = useMemo(() => {
    const groups = new Map<string, ExecutionItem[]>()
    groups.set("root", rootExec ? [rootExec] : [])
    for (const s of subExecs) {
      const key = s.subagentType ?? "general"
      const arr = groups.get(key) ?? []
      arr.push(s)
      groups.set(key, arr)
    }
    return groups
  }, [rootExec, subExecs])

  const allTypes = useMemo(() => [...typeGroups.keys()].filter(k => k !== "root"), [typeGroups])
  const activeFilter = filterTypes.size > 0

  const filteredExecs = useMemo(() => {
    if (!activeFilter) return subExecs
    return subExecs.filter(e => filterTypes.has(e.subagentType ?? "general"))
  }, [subExecs, filterTypes, activeFilter])

  const filteredGroups = useMemo(() => {
    const groups = new Map<string, ExecutionItem[]>()
    for (const e of filteredExecs) {
      const key = e.subagentType ?? "general"
      const arr = groups.get(key) ?? []
      arr.push(e)
      groups.set(key, arr)
    }
    return groups
  }, [filteredExecs])

  const tickInterval = computeTickInterval(totalDuration)
  const ticks = useMemo(() => {
    const result: Array<{ ms: number; pct: number; label: string }> = []
    for (let ms = 0; ms <= totalDuration; ms += tickInterval) {
      const pct = ms / totalDuration * 100
      const label = showAbsolute
        ? formatTimeAbsolute(new Date(sessionStartMs + ms).toISOString())
        : formatLatency(ms)
      result.push({ ms, pct, label })
    }
    return result
  }, [totalDuration, tickInterval, showAbsolute, sessionStartMs])

  const phaseOverlays = useMemo(() => {
    if (!workflowPhases) return []
    return workflowPhases.map(p => {
      const startMs = p.startTime ? new Date(p.startTime).getTime() - rootStartMs : 0
      const endMs = p.endTime ? new Date(p.endTime).getTime() - rootStartMs : totalDuration
      const startPct = Math.max(0, Math.min(startMs / totalDuration * 100, 100))
      const endPct = Math.max(0, Math.min(endMs / totalDuration * 100, 100))
      return { ...p, startPct, endPct }
    })
  }, [workflowPhases, rootStartMs, totalDuration])

  const checkpointOverlays = useMemo(() => {
    if (!workflowCheckpoints) return []
    return workflowCheckpoints.map(cp => {
      const requestMs = cp.requestedAt ? new Date(cp.requestedAt).getTime() - rootStartMs : null
      const approveMs = cp.approvedAt ? new Date(cp.approvedAt).getTime() - rootStartMs : null
      const requestPct = requestMs != null ? Math.max(0, Math.min(requestMs / totalDuration * 100, 100)) : null
      const approvePct = approveMs != null ? Math.max(0, Math.min(approveMs / totalDuration * 100, 100)) : null
      return { ...cp, requestPct, approvePct }
    })
  }, [workflowCheckpoints, rootStartMs, totalDuration])

  const selectedExec = executions.find(e => e.executionId === selectedId)

  function getWidthPercent(latencyMs: number): number {
    if (totalDuration === 0) return 5
    return Math.max(2, (latencyMs / totalDuration) * 100)
  }

  function getOffsetPercent(e: ExecutionItem): number {
    if (!e.isSubagent) return 0
    const subStart = new Date(e.createdAt).getTime()
    const offset = (subStart - rootStartMs) / totalDuration * 100
    return Math.max(0, Math.min(offset, 98))
  }

  if (executions.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No executions found
      </div>
    )
  }

  function renderDetailPanel(exec: ExecutionItem | undefined) {
    if (!exec) return null
    const color = getTypeColor(exec.subagentType, exec.isSubagent)
    return (
      <Card>
        <CardContent className="py-3 space-y-2">
          <div className="flex items-center gap-2">
            <Badge className={cn(color.bar, "text-white")}>{exec.isSubagent ? (exec.subagentName ?? exec.agentName ?? "subagent") : (exec.agentName ?? "root")}</Badge>
            {exec.subagentType && <Badge variant="outline">{exec.subagentType}</Badge>}
            <Badge variant={exec.isSubagent ? "orange" : "blue"}>{exec.isSubagent ? "subagent" : "root"}</Badge>
          </div>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <span className="text-muted-foreground">Tokens</span>
              <div className="font-medium tabular-nums">{formatTokenCount(exec.tokens)} (in:{formatTokenCount(exec.inputTokens)} out:{formatTokenCount(exec.outputTokens)} re:{formatTokenCount(exec.reasoningTokens)})</div>
            </div>
            <div>
              <span className="text-muted-foreground">Duration</span>
              <div className="font-medium tabular-nums">{formatLatency(exec.latencyMs)}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Cost</span>
              <div className="font-medium tabular-nums">{formatCost(exec.cost)}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Tool Calls</span>
              <div className="font-medium tabular-nums">{exec.toolCallCount} ({exec.toolCallErrorCount} errors)</div>
            </div>
            <div>
              <span className="text-muted-foreground">LLM Calls</span>
              <div className="font-medium tabular-nums">{exec.llmCallCount}</div>
            </div>
            <div>
              <span className="text-muted-foreground">Skills</span>
              <div className="font-medium tabular-nums">{exec.skillLoadCount} + {exec.skillInvokeCount} invokes</div>
            </div>
          </div>
          {exec.skills.length > 0 && (
            <div className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">Skill Names:</span>
              {exec.skills.map((s, i) => (
                <Badge key={i} variant="yellow" className="text-xs">{s.skillName}{s.skillVersion ? ` v${s.skillVersion}` : ""}</Badge>
              ))}
            </div>
          )}
          <div className="text-xs text-muted-foreground">Model: {exec.model ?? "N/A"} | Session ID: {exec.agentSessionId ?? "N/A"}</div>
        </CardContent>
      </Card>
    )
  }

  function renderBar(exec: ExecutionItem, height: string = "h-10") {
    const color = getTypeColor(exec.subagentType, exec.isSubagent)
    const barWidth = getWidthPercent(exec.latencyMs)
    const barOffset = getOffsetPercent(exec)
    const isSelected = selectedId === exec.executionId
    const isHovered = hoveredId === exec.executionId
    const showLabel = barWidth > 8
    const label = exec.isSubagent ? (exec.subagentName ?? exec.agentName ?? "sub") : (exec.agentName ?? "root")

    return (
      <div
        key={exec.executionId}
        className={cn("relative cursor-pointer transition-all", height, isSelected && "ring-2 ring-primary ring-offset-1 rounded-lg")}
        onClick={() => setSelectedId(isSelected ? null : exec.executionId)}
        onMouseEnter={() => setHoveredId(exec.executionId)}
        onMouseLeave={() => setHoveredId(null)}
      >
        <div
          className={cn("absolute top-0 h-full rounded-lg transition-colors", color.bar, isHovered && color.hover)}
          style={{ left: `${barOffset}%`, width: `${barWidth}%` }}
        />
        {showLabel && (
          <div
            className="absolute top-0 h-full flex items-center justify-center text-xs font-medium text-white truncate px-1 pointer-events-none"
            style={{ left: `${barOffset}%`, width: `${barWidth}%` }}
          >
            {label}
          </div>
        )}
        {isHovered && !isSelected && (
          <div
            className="absolute bottom-full mb-2 bg-popover border rounded-lg shadow-lg p-2 z-20 text-xs min-w-[180px]"
            style={{ left: `${Math.min(barOffset, 60)}%` }}
          >
            <div className="font-medium">{label}</div>
            <div className="text-muted-foreground mt-1">
              {formatLatency(exec.latencyMs)} | {formatTokenCount(exec.tokens)} tok | {formatCost(exec.cost)}
            </div>
            {exec.subagentType && <div className="text-muted-foreground">{exec.subagentType}</div>}
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="p-4 space-y-3">
      <div className="flex items-center gap-3 flex-wrap">
        <Badge variant="outline">Timeline</Badge>
        <span className="text-sm text-muted-foreground">
          {formatLatency(totalDuration)} | {executions.length} executions ({subExecs.length} subagents)
        </span>
        <Button size="sm" variant="ghost" className="text-xs" onClick={() => setShowAbsolute(!showAbsolute)}>
          {showAbsolute ? "相对时间" : "绝对时间"}
        </Button>

        <div className="flex items-center gap-1.5 ml-auto">
          <span className="text-xs text-muted-foreground">过滤:</span>
          {allTypes.map(type => {
            const color = getTypeColor(type, true)
            const active = !filterTypes.has(type)
            return (
              <button
                key={type}
                className={cn(
                  "text-xs px-2 py-0.5 rounded-md border cursor-pointer transition-colors",
                  active ? cn(color.bar, "text-white border-transparent") : "bg-muted text-muted-foreground border-border"
                )}
                onClick={() => {
                  const next = new Set(filterTypes)
                  if (next.has(type)) { next.delete(type) } else { next.add(type) }
                  setFilterTypes(next)
                }}
              >
                {type}
              </button>
            )
          })}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="text-xs text-muted-foreground">图例:</span>
        {Object.entries(TYPE_PALETTE).map(([key, color]) => (
          <div key={key} className="flex items-center gap-1 text-xs">
            <div className={cn("w-3 h-3 rounded", color.bar)} />
            <span className={color.text}>{key === "root" ? "Root Agent" : key.replace(/_/g, "-")}</span>
          </div>
        ))}
      </div>

      <div className="flex gap-4">
        <div className="flex-1 min-w-0">
          <div className="relative mb-6">
            <div className="relative h-4">
              {phaseOverlays.map(po => (
                <div
                  key={po.phaseIndex}
                  className={cn("absolute top-0 h-full rounded", PHASE_COLORS[po.phaseIndex] ?? PHASE_COLORS[1])}
                  style={{ left: `${po.startPct}%`, width: `${po.endPct - po.startPct}%` }}
                >
                  <span className="absolute top-0 left-1 text-xs font-medium text-muted-foreground whitespace-nowrap truncate" style={{ maxWidth: `${po.endPct - po.startPct - 1}%` }}>
                    {po.phaseName}
                  </span>
                </div>
              ))}
            </div>

            <div className="relative mt-2">
              {ticks.map((t, i) => (
                <div
                  key={i}
                  className="absolute text-xs text-muted-foreground tabular-nums"
                  style={{ left: `${t.pct}%`, transform: "translateX(-50%)", top: 0 }}
                >
                  {t.label}
                </div>
              ))}
              <div className="relative h-1.5 bg-muted rounded-full mt-5">
                {ticks.filter(t => t.pct > 0 && t.pct < 100).map((t, i) => (
                  <div
                    key={i}
                    className="absolute top-0 h-full w-px bg-border"
                    style={{ left: `${t.pct}%` }}
                  />
                ))}
              </div>
            </div>

            {checkpointOverlays.map((cp, i) => (
              <div key={`cp-${i}`} className="relative" style={{ height: "24px" }}>
                {cp.requestPct != null && (
                  <div
                    className="absolute top-1 z-10 cursor-pointer"
                    style={{ left: `${cp.requestPct}%`, transform: "translateX(-50%)" }}
                    title={`${cp.checkpointLabel} | ${cp.checkpointType === "block" ? "⛔" : "⚪"} | ${formatLatency(cp.waitTimeMs)} wait`}
                  >
                    <span className="text-sm">{cp.checkpointType === "block" ? "⛔" : "⚪"}</span>
                  </div>
                )}
                {cp.requestPct != null && cp.approvePct != null && (
                  <div
                    className="absolute top-3 h-px border-t-2 border-dashed border-gray-400 dark:border-gray-500 z-5"
                    style={{ left: `${cp.requestPct}%`, width: `${cp.approvePct - cp.requestPct}%` }}
                    title={`等待 ${formatLatency(cp.waitTimeMs)}`}
                  />
                )}
              </div>
            ))}
          </div>

          {rootExec && (
            <div className="mb-3">
              <div className="flex items-center gap-2 mb-1">
                <Badge variant="blue">{rootExec.agentName ?? "root"}</Badge>
                <span className="text-xs text-muted-foreground">
                  {formatLatency(rootExec.latencyMs)} | {formatTokenCount(rootExec.tokens)} tok | {formatCost(rootExec.cost)}
                </span>
              </div>
              {renderBar(rootExec, "h-10")}
            </div>
          )}

          {[...filteredGroups.entries()].map(([type, execs]) => {
            const color = getTypeColor(type, true)
            return (
              <div key={type} className={cn("mb-3 rounded-lg border p-2", color.bg)}>
                <div className="flex items-center gap-2 mb-1.5">
                  <Badge className={cn(color.bar, "text-white")}>{type.replace(/_/g, "-")}</Badge>
                  <span className="text-xs text-muted-foreground">{execs.length} executions</span>
                </div>
                <div className="space-y-1.5">
                  {execs.map(exec => (
                    <div key={exec.executionId}>
                      {renderBar(exec, "h-7")}
                    </div>
                  ))}
                </div>
              </div>
            )
          })}
        </div>

        <div className="w-[320px] shrink-0">
          {selectedExec ? renderDetailPanel(selectedExec) : (
            <Card>
              <CardContent className="py-8 text-center text-sm text-muted-foreground">
                点击时间线条查看详情
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
