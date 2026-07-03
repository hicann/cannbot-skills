"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useState, type ReactNode } from "react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { LlmOutputView } from "./LlmOutputView"
import { ToolCallList } from "./ToolCallList"
import { SkillEventList } from "./SkillEventList"
import type {
  WorkflowTree,
  WorkflowPhaseNode,
  WorkflowStepNode,
  WorkflowCheckpointNode,
  WorkflowParallelGroupNode,
  WorkflowTurnNode,
} from "@/lib/ingest/phase-split"

interface BridgeItem {
  bridgeId: string
  dispatchContent: string | null
  dispatchTimestamp: string | null
  responseContent: string | null
  responseTimestamp: string | null
  subagentName: string | null
  subagentType: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface TurnRangeItem {
  turnIndex: number
  subagentSessionId: string | null
  isSubagent: boolean
  role: string
}

interface SubagentTurnItem {
  turnId: string
  turnIndex: number
  role: string
  content: string | null
  contentJson: string | null
  contentSummary: string | null
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  latencyMs: number
  createdAt: string | null
  completedAt: string | null
  model: string | null
  toolCalls: Array<{
    id: string
    toolCallId: string
    toolName: string
    argsJson: string | null
    resultJson: string | null
    state: string
    errorType: string | null
    errorMessage: string | null
    durationMs: number
    isSkillRelated: boolean
  }>
  skillEvents: Array<{
    id: string
    skillName: string
    skillVersion: number | null
    eventType: string
    success: boolean
    errorMessage: string | null
    argsJson: string | null
    durationMs: number
  }>
}

interface WorkflowTreeViewProps {
  workflow: WorkflowTree
  bridges: BridgeItem[]
  turns: TurnRangeItem[]
  taskId: string
  onViewTurnsInteraction?: (interactionIndex: number) => void
  onSelectTurn?: (turnId: string) => void
}

// --- Timeline item types for hierarchical rendering ---
interface RootTurnTimelineItem {
  type: "root-turn"
  turnNode: WorkflowTurnNode | null
  triggerTurnId: string | null
  subagentLanes: SubagentLaneData[]
  attachedCheckpoints: WorkflowCheckpointNode[]
}

interface StandaloneCheckpointItem {
  type: "standalone-checkpoint"
  checkpoint: WorkflowCheckpointNode
}

interface OrphanStepItem {
  type: "orphan-step"
  step: WorkflowStepNode
}

type TimelineItem = RootTurnTimelineItem | StandaloneCheckpointItem | OrphanStepItem

interface SubagentLaneData {
  steps: WorkflowStepNode[]
  label: string | null
  isParallel: boolean
  parallelGroupId: string | null
}

function formatLatency(ms: number): string {
  if (ms <= 0) return "0ms"
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

function formatTime(ts: string | null, refTs?: string | null, markup?: boolean): ReactNode {
  if (!ts) return ""
  try {
    const d = new Date(ts)
    const time = `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
    if (!refTs) return time
    const ref = new Date(refTs)
    const dayDiff = Math.floor((d.getTime() - ref.getTime()) / 86400000)
    if (dayDiff > 0) {
      if (markup) return <>{time} <span className="text-orange-500">(+{dayDiff})</span></>
      return `${time} (+${dayDiff})`
    }
    return time
  } catch {
    return ts ?? ""
  }
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "N/A"
  try {
    const d = new Date(ts)
    const month = String(d.getMonth() + 1)
    const day = String(d.getDate())
    const hour = String(d.getHours()).padStart(2, '0')
    const minute = String(d.getMinutes()).padStart(2, '0')
    return `${month}/${day} ${hour}:${minute}`
  } catch {
    return ts
  }
}

const STATUS_BADGE: Record<string, "green" | "red" | "orange" | "gray"> = {
  completed: "green",
  failed: "red",
  running: "orange",
  dispatched: "gray",
  timeout: "orange",
}

const ROLE_COLORS: Record<string, string> = {
  user: "border-l-blue-500 bg-blue-50/30 dark:bg-blue-500/5",
  assistant: "border-l-emerald-500 bg-emerald-50/30 dark:bg-emerald-500/5",
  system: "border-l-gray-400 bg-gray-50/30 dark:bg-gray-500/5",
}

const ROLE_BADGE_VARIANTS: Record<string, "blue" | "green" | "gray"> = {
  user: "blue",
  assistant: "green",
  system: "gray",
}

const ROLE_ICONS: Record<string, string> = {
  user: "👤",
  assistant: "🤖",
  system: "⚙️",
}

// --- buildPhaseTimeline: transform flat children into hierarchical timeline ---
function buildPhaseTimeline(phase: WorkflowPhaseNode): TimelineItem[] {
  const children = phase.children
  const turnNodes: WorkflowTurnNode[] = []
  const stepNodes: WorkflowStepNode[] = []
  const checkpointNodes: WorkflowCheckpointNode[] = []
  const parallelGroups: WorkflowParallelGroupNode[] = []

  for (const child of children) {
    switch (child.type) {
      case "turn":
        turnNodes.push(child)
        break
      case "step":
        stepNodes.push(child)
        break
      case "checkpoint":
        checkpointNodes.push(child)
        break
      case "parallel-group":
        parallelGroups.push(child)
        break
    }
  }

  // Flatten parallel group steps into stepNodes, marking their parallelGroupId
  for (const pg of parallelGroups) {
    for (const step of pg.steps) {
      stepNodes.push({ ...step, parallelGroupId: pg.groupId })
    }
  }

  // Group steps by triggerTurnId
  const stepsByTrigger = new Map<string, WorkflowStepNode[]>()
  for (const step of stepNodes) {
    const key = step.triggerTurnId ?? "__orphan__"
    const arr = stepsByTrigger.get(key) ?? []
    arr.push(step)
    stepsByTrigger.set(key, arr)
  }

  // Group checkpoints by triggerTurnId
  const checkpointsByTrigger = new Map<string, WorkflowCheckpointNode[]>()
  const standaloneCheckpoints: WorkflowCheckpointNode[] = []
  for (const cp of checkpointNodes) {
    if (cp.triggerTurnId) {
      const arr = checkpointsByTrigger.get(cp.triggerTurnId) ?? []
      arr.push(cp)
      checkpointsByTrigger.set(cp.triggerTurnId, arr)
    } else {
      standaloneCheckpoints.push(cp)
    }
  }

  // Build subagent lanes for a group of steps sharing the same trigger
  function buildLanes(steps: WorkflowStepNode[]): SubagentLaneData[] {
    const byParallelGroup = new Map<string | null, WorkflowStepNode[]>()
    for (const s of steps) {
      const key = s.parallelGroupId ?? null
      const arr = byParallelGroup.get(key) ?? []
      arr.push(s)
      byParallelGroup.set(key, arr)
    }

    const lanes: SubagentLaneData[] = []
    for (const [pgId, pgSteps] of byParallelGroup) {
      if (pgId) {
        const pg = parallelGroups.find(p => p.groupId === pgId)
        lanes.push({
          steps: pgSteps,
          label: pg?.label ?? null,
          isParallel: true,
          parallelGroupId: pgId,
        })
      } else {
        // Each non-parallel step is its own lane
        for (const s of pgSteps) {
          lanes.push({
            steps: [s],
            label: null,
            isParallel: false,
            parallelGroupId: null,
          })
        }
      }
    }
    return lanes
  }

  // Build turnId → turnNode map
  const turnById = new Map<string, WorkflowTurnNode>()
  for (const t of turnNodes) {
    turnById.set(t.turnId, t)
  }

  // Determine all trigger turn IDs that have steps (these are "dispatch turns")
  const dispatchTurnIds = new Set<string>()
  for (const [turnId] of stepsByTrigger) {
    if (turnId !== "__orphan__") dispatchTurnIds.add(turnId)
  }

  // Build timeline: merge turnNodes + dispatch turns + checkpoints into ordered items
  const timeline: TimelineItem[] = []
  const seenTurnIds = new Set<string>()

  // Walk turnNodes in order, interleaving dispatch turns and checkpoints
  for (const turnNode of turnNodes) {
    seenTurnIds.add(turnNode.turnId)

    const lanes = stepsByTrigger.has(turnNode.turnId)
      ? buildLanes(stepsByTrigger.get(turnNode.turnId)!)
      : []
    const attachedCPs = checkpointsByTrigger.get(turnNode.turnId) ?? []

    timeline.push({
      type: "root-turn",
      turnNode,
      triggerTurnId: turnNode.turnId,
      subagentLanes: lanes,
      attachedCheckpoints: attachedCPs,
    })
  }

  // Add dispatch turns that are NOT in turnNodes (triggerTurnId from steps but no matching turn)
  for (const turnId of dispatchTurnIds) {
    if (seenTurnIds.has(turnId)) continue
    seenTurnIds.add(turnId)

    const lanes = buildLanes(stepsByTrigger.get(turnId) ?? [])
    const attachedCPs = checkpointsByTrigger.get(turnId) ?? []

    // Create a synthetic turn-like entry using step metadata
    const steps = stepsByTrigger.get(turnId) ?? []
    const firstStep = steps[0]
    timeline.push({
      type: "root-turn",
      turnNode: null,
      triggerTurnId: turnId,
      subagentLanes: lanes,
      attachedCheckpoints: attachedCPs,
    })
  }

  // Insert standalone checkpoints
  for (const cp of standaloneCheckpoints) {
    timeline.push({
      type: "standalone-checkpoint",
      checkpoint: cp,
    })
  }

  // Insert orphan steps (no triggerTurnId)
  const orphanSteps = stepsByTrigger.get("__orphan__") ?? []
  for (const step of orphanSteps) {
    timeline.push({
      type: "orphan-step",
      step,
    })
  }

  return timeline
}

function computeTurnRanges(turns: TurnRangeItem[]): Map<string, { start: number; end: number }> {
  const ranges = new Map<string, { start: number; end: number }>()
  for (const t of turns) {
    if (t.subagentSessionId) {
      const existing = ranges.get(t.subagentSessionId)
      if (!existing) {
        ranges.set(t.subagentSessionId, { start: t.turnIndex, end: t.turnIndex })
      } else {
        if (t.turnIndex < existing.start) existing.start = t.turnIndex
        if (t.turnIndex > existing.end) existing.end = t.turnIndex
      }
    }
  }
  return ranges
}

function SummaryBar({ summary }: { summary: WorkflowTree["summary"] }) {
  return (
    <div className="flex flex-wrap items-center gap-3 text-sm">
      <Badge variant="blue">{summary.totalPhases} Phases</Badge>
      <Badge variant="green">{summary.totalSteps} Steps</Badge>
      <Badge variant="gray">{summary.totalCheckpoints} Checkpoints</Badge>
      <span className="text-muted-foreground">Active: {formatLatency(summary.totalActiveTimeMs)}</span>
      <span className="text-muted-foreground">Wait: {formatLatency(summary.totalWaitTimeMs)}</span>
      <span className="text-xs text-muted-foreground">Active {summary.activeTimePct}%</span>
    </div>
  )
}

function GlobalTimeline({ phases }: { phases: WorkflowPhaseNode[] }) {
  if (phases.length === 0) return null

  const startTimes = phases.map(p => p.startTime ? new Date(p.startTime).getTime() : 0).filter(t => t > 0)
  const endTimes = phases.map(p => p.endTime ? new Date(p.endTime).getTime() : 0).filter(t => t > 0)
  if (startTimes.length === 0) return null

  const globalStart = Math.min(...startTimes)
  const globalEnd = Math.max(...endTimes)
  const totalDuration = globalEnd - globalStart
  if (totalDuration <= 0) return null

  const allCheckpoints: Array<{ time: number; label: string; phaseIndex: number }> = []
  for (const phase of phases) {
    for (const child of phase.children) {
      if (child.type === "checkpoint" && child.requestedAt) {
        allCheckpoints.push({
          time: new Date(child.requestedAt).getTime(),
          label: child.checkpointLabel,
          phaseIndex: phase.phaseIndex,
        })
      }
    }
  }

  const phaseColors = [
    "bg-blue-400",
    "bg-emerald-400",
    "bg-violet-400",
    "bg-orange-400",
    "bg-rose-400",
    "bg-amber-400",
    "bg-teal-400",
  ]

  return (
    <div className="mt-2 mb-3">
      <div className="flex items-center gap-1 text-xs mb-1">
        <span className="text-muted-foreground">{formatTimestamp(new Date(globalStart).toISOString())}</span>
        <div className="flex-1 h-4 relative bg-muted rounded">
          {phases.map((phase, i) => {
            const phaseStart = phase.startTime ? new Date(phase.startTime).getTime() : globalStart
            const phaseEnd = phase.endTime ? new Date(phase.endTime).getTime() : globalEnd
            const leftPct = ((phaseStart - globalStart) / totalDuration) * 100
            const widthPct = ((phaseEnd - phaseStart) / totalDuration) * 100
            return (
              <div
                key={`phase-bar-${phase.phaseSequence ?? i}`}
                className={cn("absolute h-full rounded-sm", phaseColors[i % phaseColors.length])}
                style={{ left: `${leftPct}%`, width: `${Math.max(widthPct, 1)}%` }}
                title={`${phase.fullLabel}: ${formatTimestamp(phase.startTime)} → ${formatTimestamp(phase.endTime)}`}
              />
            )
          })}
          {allCheckpoints.map((cp, i) => {
            const pct = ((cp.time - globalStart) / totalDuration) * 100
            return (
              <div
                key={i}
                className="absolute h-full w-0.5 bg-red-500"
                style={{ left: `${pct}%` }}
                title={cp.label}
              />
            )
          })}
        </div>
        <span className="text-muted-foreground">{formatTimestamp(new Date(globalEnd).toISOString())}</span>
      </div>
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        {phases.map((phase, i) => (
          <span key={`phase-label-${phase.phaseSequence ?? i}`} className={cn("inline-flex items-center gap-1", phaseColors[i % phaseColors.length].replace("bg-", "text-"))}>
            <span className={cn("w-2 h-2 rounded-sm inline-block", phaseColors[i % phaseColors.length])} />
            P{phase.phaseSequence ?? phase.phaseIndex}
          </span>
        ))}
        <span className="inline-flex items-center gap-1 text-red-500">
          <span className="w-2 h-0.5 bg-red-500 inline-block" /> CP
        </span>
      </div>
    </div>
  )
}

// --- SubagentLaneInPhase: renders a subagent lane under a root turn ---
function SubagentLaneInPhase({
  steps,
  label,
  isParallel,
  bridges,
  turnRanges,
  taskId,
  onViewTurnsInteraction,
  onSelectTurn,
}: {
  steps: WorkflowStepNode[]
  label: string | null
  isParallel: boolean
  bridges: BridgeItem[]
  turnRanges: Map<string, { start: number; end: number }>
  taskId: string
  onViewTurnsInteraction?: (interactionIndex: number) => void
  onSelectTurn?: (turnId: string) => void
}) {
  // For parallel groups, render each step as a separate lane in a grid
  if (isParallel && steps.length > 1) {
    return (
      <div className={cn(
        "ml-4 mt-1",
        steps.length === 2 ? "grid gap-1.5 grid-cols-2" : steps.length === 3 ? "grid gap-1.5 grid-cols-3" : "grid gap-1.5 grid-cols-2"
      )}>
        {steps.map(step => (
          <SingleLane
            key={step.stepIndex}
            step={step}
            bridges={bridges}
            turnRanges={turnRanges}
            taskId={taskId}
            onViewTurnsInteraction={onViewTurnsInteraction}
            onSelectTurn={onSelectTurn}
          />
        ))}
      </div>
    )
  }

  // Single lane (one step, or parallel group label)
  const firstStep = steps[0]
  if (!firstStep) return null

  if (steps.length === 1) {
    return (
      <div className="ml-4 mt-1">
        <SingleLane
          step={firstStep}
          bridges={bridges}
          turnRanges={turnRanges}
          taskId={taskId}
          onViewTurnsInteraction={onViewTurnsInteraction}
          onSelectTurn={onSelectTurn}
        />
      </div>
    )
  }

  // Multi-step lane (sequential steps in one lane)
  return (
    <div className="ml-4 mt-1 space-y-1">
      {steps.map(step => (
        <SingleLane
          key={step.stepIndex}
          step={step}
          bridges={bridges}
          turnRanges={turnRanges}
          taskId={taskId}
          onViewTurnsInteraction={onViewTurnsInteraction}
          onSelectTurn={onSelectTurn}
        />
      ))}
    </div>
  )
}

function SingleLane({
  step,
  bridges,
  turnRanges,
  taskId,
  onViewTurnsInteraction,
  onSelectTurn,
}: {
  step: WorkflowStepNode
  bridges: BridgeItem[]
  turnRanges: Map<string, { start: number; end: number }>
  taskId: string
  onViewTurnsInteraction?: (interactionIndex: number) => void
  onSelectTurn?: (turnId: string) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const bridge = step.bridgeId ? bridges.find(b => b.bridgeId === step.bridgeId) : null
  const turnRange = step.subagentSessionId ? turnRanges.get(step.subagentSessionId) : null
  const isError = step.status === "error" || step.status === "failed"
  const statusBadgeVariant = STATUS_BADGE[step.status] ?? "gray"

  return (
    <div className={cn(
      "border rounded-lg",
      isError ? "border-red-300 bg-red-50/30 dark:bg-red-500/5" : "border-orange-200 bg-orange-50/20 dark:bg-orange-500/5"
    )}>
      <button
        className="w-full flex items-center gap-1.5 px-2 py-1.5 text-left cursor-pointer hover:bg-accent/30 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-xs select-none">{expanded ? "▼" : "▶"}</span>
        <Badge variant="orange" className="text-xs">{step.subagentName ?? step.subagentType ?? "subagent"}</Badge>
        {step.stepName && (
          <span className="text-xs text-foreground/80 truncate max-w-[300px]">{step.stepName}</span>
        )}
        {isError && <Badge variant="red" className="text-xs">error</Badge>}
        {step.status !== "completed" && !isError && (
          <Badge variant={statusBadgeVariant} className="text-xs">{step.status}</Badge>
        )}
      </button>

      <div className="flex items-center gap-2 px-2 pb-1 text-xs text-muted-foreground">
        <Badge variant={step.status === "completed" ? "green" : statusBadgeVariant} className="text-xs">
          {step.status === "completed" ? "✅" : step.status === "failed" ? "❌" : "⏳"} {step.status}
        </Badge>
        <span>{formatLatency(step.durationMs)}</span>
        <span>{formatTokenCount(step.totalTokens)} tok</span>
        {step.startTime && <span>{formatTime(step.startTime)}</span>}
      </div>

      {expanded && (
        <div className="border-t bg-background/50">
          {bridge && (
            <div className="px-3 py-2 space-y-2 border-b">
              <div>
                <span className="text-xs font-medium text-muted-foreground">Dispatch:</span>
                <div className="text-xs mt-0.5 max-h-[120px] overflow-y-auto whitespace-pre-wrap break-words">
                  {bridge.dispatchContent ?? "N/A"}
                </div>
              </div>
              <div>
                <span className="text-xs font-medium text-muted-foreground">Response:</span>
                <div className="text-xs mt-0.5 max-h-[120px] overflow-y-auto whitespace-pre-wrap break-words">
                  {bridge.responseContent ?? "N/A"}
                </div>
              </div>
              <div className="flex items-center gap-3 text-xs">
                <Badge variant="outline">{formatTokenCount(bridge.subagentTokens)} tokens</Badge>
                <Badge variant="outline">{formatLatency(bridge.subagentLatencyMs)}</Badge>
              </div>
            </div>
          )}
          {step.subagentSessionId ? (
            <SubagentTurnsView
              subagentSessionId={step.subagentSessionId}
              taskId={taskId}
              onSelectTurn={onSelectTurn}
            />
          ) : (
            <div className="px-3 py-2 text-xs text-muted-foreground">
              No subagent session data available
            </div>
          )}
          {onViewTurnsInteraction && step.subagentSessionId && (
            <div className="px-3 py-1.5 border-t">
              <button
                className="text-xs text-primary cursor-pointer hover:underline"
                onClick={() => onViewTurnsInteraction(-1)}
              >
                View all turns #{turnRange?.start ?? "?"} → #{turnRange?.end ?? "?"} →
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// --- PhaseTurnRow: root turn as timeline row + nested subagent lanes ---
function PhaseTurnRow({
  item,
  bridges,
  turnRanges,
  taskId,
  onViewTurnsInteraction,
  onSelectTurn,
}: {
  item: RootTurnTimelineItem
  bridges: BridgeItem[]
  turnRanges: Map<string, { start: number; end: number }>
  taskId: string
  onViewTurnsInteraction?: (interactionIndex: number) => void
  onSelectTurn?: (turnId: string) => void
}) {
  const turnNode = item.turnNode
  const hasLanes = item.subagentLanes.length > 0
  const borderClass = turnNode
    ? (ROLE_COLORS[turnNode.role] ?? "border-l-muted")
    : "border-l-emerald-500 bg-emerald-50/30 dark:bg-emerald-500/5"

  return (
    <div>
      {/* Root turn row */}
      {turnNode ? (
        <button
          className={cn(
            "w-full text-left rounded-lg border-l-3 p-2.5 transition-colors cursor-pointer",
            borderClass,
            "hover:bg-accent/50"
          )}
          onClick={() => onSelectTurn?.(turnNode.turnId)}
        >
          <div className="flex items-center gap-1.5 mb-1">
            <span className="text-xs font-mono text-muted-foreground">#{turnNode.turnIndex}</span>
            <Badge variant={ROLE_BADGE_VARIANTS[turnNode.role] ?? "gray"}>
              {ROLE_ICONS[turnNode.role]} {turnNode.role}
            </Badge>
            {hasLanes && (
              <Badge variant="orange" className="text-xs">🔗 {item.subagentLanes.length} subagent</Badge>
            )}
            {turnNode.totalTokens > 0 && (
              <span className="text-muted-foreground">{formatTokenCount(turnNode.totalTokens)} tok</span>
            )}
            {turnNode.durationMs > 0 && (
              <span className="text-muted-foreground">{formatLatency(turnNode.durationMs)}</span>
            )}
            {turnNode.startTime && (
              <span className="text-muted-foreground">{formatTime(turnNode.startTime)}</span>
            )}
          </div>
          {turnNode.contentSummary && (
            <p className="text-xs text-foreground/80 line-clamp-2">
              {turnNode.contentSummary}
            </p>
          )}
        </button>
      ) : (
        /* Synthetic dispatch turn (no matching turnNode, but has subagent lanes) */
        <div className={cn("rounded-lg border-l-3 p-2.5", borderClass)}>
          <div className="flex items-center gap-1.5 mb-1">
            <Badge variant="green">🤖 dispatch</Badge>
            {hasLanes && (
              <Badge variant="orange" className="text-xs">🔗 {item.subagentLanes.length} subagent</Badge>
            )}
          </div>
        </div>
      )}

      {/* Subagent lanes nested under this turn */}
      {hasLanes && (
        <div className="space-y-1 mt-0.5">
          {item.subagentLanes.map((lane, li) => (
            <SubagentLaneInPhase
              key={`lane-${li}`}
              steps={lane.steps}
              label={lane.label}
              isParallel={lane.isParallel}
              bridges={bridges}
              turnRanges={turnRanges}
              taskId={taskId}
              onViewTurnsInteraction={onViewTurnsInteraction}
              onSelectTurn={onSelectTurn}
            />
          ))}
        </div>
      )}

      {/* Attached checkpoints */}
      {item.attachedCheckpoints.length > 0 && (
        <div className="ml-4 mt-1 space-y-1">
          {item.attachedCheckpoints.map((cp, ci) => (
            <CheckpointNode key={`att-cp-${ci}`} checkpoint={cp} />
          ))}
        </div>
      )}
    </div>
  )
}

function CheckpointNode({ checkpoint }: { checkpoint: WorkflowCheckpointNode }) {
  const [expanded, setExpanded] = useState(false)
  const typeIcon = checkpoint.checkpointType === "block" ? "⛔" : "⚪"

  return (
    <div className="border-l-3 border-l-yellow-500 bg-yellow-50/30 dark:bg-yellow-500/5 rounded-md">
      <button
        className="w-full px-2 py-1.5 cursor-pointer hover:bg-accent/50 transition-colors text-xs"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2">
          <Badge variant="yellow">{typeIcon} {checkpoint.checkpointLabel}</Badge>
          {checkpoint.checkpointType === "block" && <Badge variant="red">Block</Badge>}
          {checkpoint.waitTimeMs > 0 && (
            <span className="text-muted-foreground">⏳ {formatLatency(checkpoint.waitTimeMs)} wait</span>
          )}
        </div>
        <div className="flex items-center gap-2 mt-0.5 text-muted-foreground">
          <span>{formatTime(checkpoint.requestedAt)}</span>
          {checkpoint.approvedAt && <span>→ {formatTime(checkpoint.approvedAt)}</span>}
        </div>
      </button>

      {expanded && (
        <div className="px-3 py-2 border-t bg-background/50 text-xs space-y-1">
          <div>
            <span className="font-medium text-muted-foreground">Requested:</span> {formatTimestamp(checkpoint.requestedAt)}
          </div>
          {checkpoint.approvedAt && (
            <div>
              <span className="font-medium text-muted-foreground">Approved:</span> {formatTimestamp(checkpoint.approvedAt)}
            </div>
          )}
          <div>
            <span className="font-medium text-muted-foreground">Wait time:</span> {formatLatency(checkpoint.waitTimeMs)}
          </div>
        </div>
      )}
    </div>
  )
}

function CompactTurnView({ turn, onSelectTurn }: { turn: SubagentTurnItem; onSelectTurn?: (turnId: string) => void }) {
  const borderClass = ROLE_COLORS[turn.role] ?? "border-l-muted"

  return (
    <div className={cn("border-l-3 rounded-md mb-2", borderClass)}>
      <div className={cn("px-2 py-1.5 text-xs", onSelectTurn && "cursor-pointer hover:bg-accent/30 transition-colors")} onClick={() => onSelectTurn?.(turn.turnId)}>
        <div className="flex items-center gap-2">
          <Badge variant={ROLE_BADGE_VARIANTS[turn.role] ?? "gray"}>
            {ROLE_ICONS[turn.role] ?? "?"} #{turn.turnIndex} {turn.role}
          </Badge>
          {turn.totalTokens > 0 && (
            <span className="text-muted-foreground">{formatTokenCount(turn.totalTokens)} tok</span>
          )}
          {turn.latencyMs > 0 && (
            <span className="text-muted-foreground">{formatLatency(turn.latencyMs)}</span>
          )}
          {turn.model && (
            <Badge variant="outline">{turn.model}</Badge>
          )}
        </div>
      </div>

      <div className="px-2 pb-1.5 text-xs" onClick={(e) => e.stopPropagation()}>
        <LlmOutputView
          content={turn.content}
          contentJson={turn.contentJson}
          contentSummary={turn.contentSummary ?? (turn.content ? (turn.content.length > 200 ? turn.content.substring(0, 200) + "..." : turn.content) : null)}
          outputTokens={turn.outputTokens}
          reasoningTokens={turn.reasoningTokens}
          role={turn.role}
        />

        {turn.toolCalls.length > 0 && (
          <div className="mt-2">
            <ToolCallList toolCalls={turn.toolCalls} />
          </div>
        )}

        {turn.skillEvents.length > 0 && (
          <div className="mt-2">
            <SkillEventList skillEvents={turn.skillEvents} />
          </div>
        )}
      </div>
    </div>
  )
}

function SubagentTurnsView({
  subagentSessionId,
  taskId,
  onSelectTurn,
}: {
  subagentSessionId: string
  taskId: string
  onSelectTurn?: (turnId: string) => void
}) {
  const [turns, setTurns] = useState<SubagentTurnItem[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    async function fetchTurns() {
      setLoading(true)
      setError(null)
      try {
        const params = new URLSearchParams({
          taskId,
          subagentSessionId,
          includeContent: "true",
          includeDetail: "true",
        })
        const res = await fetch(`/api/observe/session/turns?${params}`)
        if (!res.ok) {
          setError("Failed to load subagent turns")
          return
        }
        const data = await res.json()
        setTurns(data.items ?? [])
      } catch {
        setError("Network error")
      } finally {
        setLoading(false)
      }
    }
    fetchTurns()
  }, [taskId, subagentSessionId])

  if (loading) {
    return (
      <div className="px-3 py-2 border-t bg-background/50 text-xs text-muted-foreground">
        Loading subagent turns...
      </div>
    )
  }

  if (error) {
    return (
      <div className="px-3 py-2 border-t bg-background/50 text-xs text-red-600 dark:text-red-400">
        {error}
      </div>
    )
  }

  if (turns.length === 0) {
    return (
      <div className="px-3 py-2 border-t bg-background/50 text-xs text-muted-foreground">
        No turns found for this subagent
      </div>
    )
  }

  return (
    <div className="px-3 py-2 border-t bg-background/50">
      <div className="flex items-center gap-2 text-xs mb-2">
        <span className="font-medium">Subagent Turns ({turns.length})</span>
      </div>
      {turns.map(turn => (
        <CompactTurnView key={turn.turnId} turn={turn} onSelectTurn={onSelectTurn} />
      ))}
    </div>
  )
}

// --- PhaseNode: refactored with hierarchical timeline ---
function PhaseNode({
  phase,
  bridges,
  turnRanges,
  taskId,
  onViewTurnsInteraction,
  onSelectTurn,
}: {
  phase: WorkflowPhaseNode
  bridges: BridgeItem[]
  turnRanges: Map<string, { start: number; end: number }>
  taskId: string
  onViewTurnsInteraction?: (interactionIndex: number) => void
  onSelectTurn?: (turnId: string) => void
}) {
  const [expanded, setExpanded] = useState(true)

  const timeline = buildPhaseTimeline(phase)

  return (
    <div className="rounded-lg border mb-3">
      <button
        className="w-full px-3 py-2.5 cursor-pointer hover:bg-accent/50 transition-colors"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs">{expanded ? "▼" : "▶"}</span>
          <span className="font-medium">{phase.fullLabel}</span>
        </div>
        <div className="flex items-center gap-3 text-xs">
          {phase.turnIndexStart != null && phase.turnIndexEnd != null && (
            <Badge variant="outline">Turn #{phase.turnIndexStart} → #{phase.turnIndexEnd}</Badge>
          )}
          <span className="text-muted-foreground">
            {formatTime(phase.startTime)} → {formatTime(phase.endTime, phase.startTime, true)}
          </span>
          <span className="text-muted-foreground">
            {formatLatency(phase.durationMs)}
          </span>
          <span className="text-muted-foreground">Active {formatLatency(phase.activeTimeMs)}</span>
          <span className="text-muted-foreground">Wait {formatLatency(phase.waitTimeMs)}</span>
          <span className="text-muted-foreground">
            {formatTokenCount(phase.totalTokens)} tok
          </span>
          <span className="text-muted-foreground">{phase.subagentCount} subagents</span>
        </div>
      </button>

      {expanded && (
        <div className="px-4 pb-3 space-y-1.5">
          {timeline.map((item, i) => {
            if (item.type === "root-turn") {
              return (
                <PhaseTurnRow
                  key={`rt-${i}`}
                  item={item}
                  bridges={bridges}
                  turnRanges={turnRanges}
                  taskId={taskId}
                  onViewTurnsInteraction={onViewTurnsInteraction}
                  onSelectTurn={onSelectTurn}
                />
              )
            }
            if (item.type === "standalone-checkpoint") {
              return (
                <CheckpointNode key={`scp-${i}`} checkpoint={item.checkpoint} />
              )
            }
            if (item.type === "orphan-step") {
              // Orphan step without a trigger turn — render as a simple lane
              return (
                <div className="ml-4 mt-1" key={`os-${i}`}>
                  <SingleLane
                    step={item.step}
                    bridges={bridges}
                    turnRanges={turnRanges}
                    taskId={taskId}
                    onViewTurnsInteraction={onViewTurnsInteraction}
                    onSelectTurn={onSelectTurn}
                  />
                </div>
              )
            }
            return null
          })}
        </div>
      )}
    </div>
  )
}

export function WorkflowTreeView({ workflow, bridges, turns, taskId, onViewTurnsInteraction, onSelectTurn }: WorkflowTreeViewProps) {
  const { phases, summary } = workflow
  const turnRanges = computeTurnRanges(turns)

  return (
    <div className="p-4 space-y-3">
      <SummaryBar summary={summary} />
      <div className="flex items-center gap-3 text-xs mb-1">
        <span className="text-muted-foreground">✦ Hierarchical view: root turns → subagent lanes</span>
      </div>
      <GlobalTimeline phases={phases} />
      {phases.map((phase, i) => (
        <PhaseNode
          key={`phase-${phase.phaseSequence ?? i}`}
          phase={phase}
          bridges={bridges}
          turnRanges={turnRanges}
          taskId={taskId}
          onViewTurnsInteraction={onViewTurnsInteraction}
          onSelectTurn={onSelectTurn}
        />
      ))}
    </div>
  )
}
