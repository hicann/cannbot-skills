"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useMemo, useState } from "react"
import type {
  WorkflowTree,
  WorkflowStepNode,
  WorkflowParallelGroupNode,
} from "@/lib/ingest/phase-split"

interface TurnItem {
  turnId: string
  turnIndex: number
  role: string
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  contentSummary: string | null
  content?: string | null
  totalTokens: number
  latencyMs: number
  toolCalls: Array<{ toolName: string; durationMs: number; state: string; resultJson?: string | null; argsJson?: string | null }>
}

interface BridgeItem {
  bridgeId: string
  dispatchTurnId: string | null
  dispatchContent: string | null
  subagentSessionId: string | null
  subagentName: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface Props {
  taskId: string
  framework?: string
  onSelectTurnId?: (turnId: string) => void
}

interface StatePhase {
  label: string
  turnIndexStart: number
  turnIndexEnd: number
}

function detectStatePhases(rootTurns: TurnItem[]): StatePhase[] {
  const phases: StatePhase[] = []
  let phaseStart = rootTurns[0]?.turnIndex ?? 0
  const completedTasks = new Set<string>()
  const phasedTasks = new Set<number>()
  let stateMdFound = false
  let lastTaskNum = 0

  for (const turn of rootTurns) {
    if (turn.role !== "assistant") continue
    const content = turn.content ?? turn.contentSummary ?? ""

    // Signal 1: - [x] in tool calls (resultJson/argsJson)
    let xLines: string[] = []
    let hasCheckbox = false
    for (const tc of turn.toolCalls ?? []) {
      const combined = (tc.resultJson ?? "") + (tc.argsJson ?? "")
      if (!combined.includes("- [")) continue
      hasCheckbox = true
      xLines = combined.match(/- \[x\]\s*\d+\.[^\n]*/g) || []
      break
    }

    // Signal 2: "验收通过" in content
    const hasAcceptance = /验收通过|已勾选/.test(content)

    // First time seeing STATE.md → close "前置任务" phase
    if (!stateMdFound && hasCheckbox) {
      stateMdFound = true
      if (turn.turnIndex > phaseStart) {
        phases.push({ label: "前置任务", turnIndexStart: phaseStart, turnIndexEnd: turn.turnIndex - 1 })
      }
      phaseStart = turn.turnIndex
      for (const line of xLines) completedTasks.add(line.trim())
      continue
    }

    // Signal 1: new - [x] completions (only after STATE.md found)
    let xHandled = false
    if (stateMdFound) {
      const newTasks = xLines.filter(l => !completedTasks.has(l.trim()))
      if (newTasks.length > 0) {
        for (const l of newTasks) {
          completedTasks.add(l.trim())
          const m = l.match(/- \[x\]\s*(\d+)\./)
          if (m) lastTaskNum = parseInt(m[1])
        }
        if (!phasedTasks.has(lastTaskNum)) {
          phasedTasks.add(lastTaskNum)
          phases.push({ label: `Task ${lastTaskNum}`, turnIndexStart: phaseStart, turnIndexEnd: turn.turnIndex })
          phaseStart = turn.turnIndex + 1
        }
        xHandled = true
      }
    }

    // Signal 2: "验收通过" (works with or without STATE.md)
    if (!xHandled && hasAcceptance) {
      const m = content.match(/Task\s*(\d+)(?!\.\d)/)
      const taskNum = m ? parseInt(m[1]) : 0
      if (taskNum > 0 && !phasedTasks.has(taskNum)) {
        phasedTasks.add(taskNum)
        lastTaskNum = taskNum
        phases.push({ label: `Task ${taskNum}`, turnIndexStart: phaseStart, turnIndexEnd: turn.turnIndex - 1 })
        phaseStart = turn.turnIndex
      }
    }
  }

  const lastIdx = rootTurns[rootTurns.length - 1]?.turnIndex ?? 0
  if (phaseStart <= lastIdx) {
    phases.push({ label: stateMdFound ? "进行中" : "全部任务", turnIndexStart: phaseStart, turnIndexEnd: lastIdx })
  }
  return phases.length > 0 ? phases : []
}

function fmtMs(ms: number): string {
  if (ms <= 0) return "—"
  const s = ms / 1000
  if (s < 60) return `${s.toFixed(0)}s`
  const m = Math.floor(s / 60)
  const rem = Math.round(s % 60)
  if (m < 60) return `${m}m${rem}s`
  return `${Math.floor(m / 60)}h${m % 60}m`
}

function fmtTok(n: number): string {
  if (n === 0) return "0"
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`
  return `${n}`
}

function truncate(s: string | null, len: number): string {
  if (!s) return ""
  return s.length > len ? s.slice(0, len - 1) + "…" : s
}

function shortSkill(s: string): string {
  return s.replace(/^ascendc-ops-/, "").replace(/^ascendc-/, "").replace(/^ops-registry-invoke-/, "wf-")
}

function categorizeTool(name: string): string {
  const n = name.toLowerCase()
  if (["read", "glob", "grep", "webfetch", "search", "list"].some(t => n.includes(t))) return "分析"
  if (["write", "edit", "create", "str_replace"].some(t => n.includes(t))) return "编写"
  if (["bash", "terminal", "shell", "test", "run", "python", "pytest"].some(t => n.includes(t))) return "执行"
  return "其他"
}

interface ToolBreakdownItem { label: string; ms: number; color: string }

const PHASE_PALETTE = ["#2563eb", "#16a34a", "#dc2626", "#8b5cf6", "#f59e0b", "#06b6d4", "#ec4899"]
const STOP_WORDS = new Set(["的", "与", "或", "及", "和", "按", "其", "不", "只", "到", "下", "上", "后", "前", "为", "以", "等", "可", "能", "所", "有", "无", "全", "量", "已", "未", "并", "再", "已", "通过", "需", "需要"])

function parsePromptSteps(content: string | null): { label: string; keywords: string[] }[] {
  if (!content) return []
  const steps: { label: string; keywords: string[] }[] = []
  const regex = /(?:^|\n)\s*(\d+)[.、)]\s+(.+)/g
  let match
  while ((match = regex.exec(content)) !== null) {
    const text = match[2].trim()
    const parts = text.split(/[\s,，、/／（）()【】\[\]{}:：;；"'"'"`]+/).filter(p => p.length > 1 && !STOP_WORDS.has(p))
    steps.push({ label: text.length > 40 ? text.slice(0, 40) + "…" : text, keywords: parts })
  }
  return steps
}

function computePhaseBreakdown(subTurns: TurnItem[], dispatchContent: string | null): ToolBreakdownItem[] {
  const steps = parsePromptSteps(dispatchContent)

  if (steps.length > 0) {
    // Prompt-driven: match turns to steps by keyword
    const stepMs: number[] = new Array(steps.length).fill(0)
    const stepFirst: number[] = new Array(steps.length).fill(Infinity)
    const stepLast: number[] = new Array(steps.length).fill(0)
    let currentStep = 0

    for (const t of subTurns) {
      if (t.latencyMs <= 0) continue
      const text = ((t.contentSummary ?? "") + " " + (t.toolCalls ?? []).map(tc => tc.toolName).join(" ")).toLowerCase()
      let bestStep = currentStep
      let bestScore = 0.5
      for (let i = 0; i < steps.length; i++) {
        const score = steps[i].keywords.filter(k => text.includes(k.toLowerCase())).length
        if (score > bestScore) { bestScore = score; bestStep = i }
      }
      currentStep = bestStep
      stepMs[currentStep] += t.latencyMs
      stepFirst[currentStep] = Math.min(stepFirst[currentStep], t.turnIndex)
      stepLast[currentStep] = Math.max(stepLast[currentStep], t.turnIndex)
    }

    return steps.map((s, i) => ({
      label: `${s.label} §${stepFirst[i] === Infinity ? "?" : stepFirst[i]}–${stepLast[i]}`,
      ms: stepMs[i],
      color: PHASE_PALETTE[i % PHASE_PALETTE.length],
    })).filter(item => item.ms > 0)
  }

  // Fallback: activity-based detection with short-segment merge
  const raw: { activity: string; ms: number; firstIdx: number; lastIdx: number }[] = []
  for (const t of subTurns) {
    if (t.latencyMs <= 0) continue
    const tools = t.toolCalls ?? []
    let activity: string
    if (tools.length === 0) {
      activity = "LLM思考"
    } else {
      const cats = tools.map(tc => categorizeTool(tc.toolName))
      if (cats.includes("编写")) activity = "编写"
      else if (cats.includes("执行")) activity = "执行"
      else if (cats.includes("分析")) activity = "分析"
      else activity = "其他"
    }
    const last = raw[raw.length - 1]
    if (last && last.activity === activity) {
      last.ms += t.latencyMs
      last.lastIdx = t.turnIndex
    } else {
      raw.push({ activity, ms: t.latencyMs, firstIdx: t.turnIndex, lastIdx: t.turnIndex })
    }
  }
  const merged: typeof raw = []
  for (const seg of raw) {
    if (merged.length > 0 && seg.ms < 30000) {
      merged[merged.length - 1].ms += seg.ms
      merged[merged.length - 1].lastIdx = seg.lastIdx
    } else {
      merged.push({ ...seg })
    }
  }
  const finalSegs: typeof raw = []
  for (const seg of merged) {
    const prev = finalSegs[finalSegs.length - 1]
    if (prev && prev.activity === seg.activity) {
      prev.ms += seg.ms
      prev.lastIdx = seg.lastIdx
    } else {
      finalSegs.push({ ...seg })
    }
  }
  return finalSegs.map((s, i) => ({
    label: `${s.activity} §${s.firstIdx}–${s.lastIdx}`,
    ms: s.ms,
    color: PHASE_PALETTE[i % PHASE_PALETTE.length],
  }))
}

const ROLE_COLOR: Record<string, string> = {
  user: "#2563eb", assistant: "#16a34a", system: "#6b7280", tool_result: "#06b6d4",
}
const STATUS_COLOR: Record<string, string> = {
  completed: "#16a34a", ok: "#16a34a", failed: "#dc2626", error: "#dc2626",
  running: "#f59e0b", timeout: "#f59e0b", dispatched: "#6b7280",
}
const PHASE_COLORS = ["#2563eb", "#16a34a", "#8b5cf6", "#f59e0b", "#06b6d4", "#ec4899", "#6366f1"]

type MainTurnCategory = "规划/推理" | "工具调用" | "验收" | "compaction" | "真冗余"

const CATEGORY_ORDER: MainTurnCategory[] = ["规划/推理", "工具调用", "验收", "compaction", "真冗余"]

const CATEGORY_COLOR: Record<MainTurnCategory, string> = {
  "规划/推理": "#8b5cf6",
  "工具调用": "#2563eb",
  "验收": "#16a34a",
  "compaction": "#6b7280",
  "真冗余": "#dc2626",
}

function classifyMainTurn(turn: TurnItem): MainTurnCategory {
  if (turn.agentName === "compaction" || turn.agentName === "continuation") return "compaction"
  const content = turn.content ?? turn.contentSummary ?? ""
  for (const tc of turn.toolCalls ?? []) {
    const combined = (tc.resultJson ?? "") + (tc.argsJson ?? "")
    if (combined.includes("- [x]") || combined.includes("- [ ]")) return "验收"
  }
  if (/验收通过|已勾选/.test(content)) return "验收"
  if ((turn.toolCalls ?? []).length > 0) return "工具调用"
  if (turn.latencyMs > 1000) return "规划/推理"
  return "真冗余"
}

interface MainTurnGroup {
  category: MainTurnCategory
  turns: TurnItem[]
  totalMs: number
}

export function ComparePanorama({ taskId, framework, onSelectTurnId }: Props) {
  const [workflow, setWorkflow] = useState<WorkflowTree | null>(null)
  const [turns, setTurns] = useState<TurnItem[]>([])
  const [bridges, setBridges] = useState<BridgeItem[]>([])
  const [loading, setLoading] = useState(true)
  const [wfError, setWfError] = useState<string | null>(null)
  const [collapsed, setCollapsed] = useState(false)

  useEffect(() => {
    let cancelled = false
    const params = new URLSearchParams({ taskId })
    if (framework) params.set("framework", framework)
    Promise.all([
      fetch(`/api/observe/session/workflow?${params}`).then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`)
        return r.json()
      }).catch(e => { setWfError(e instanceof Error ? e.message : "load failed"); return null }),
      fetch(`/api/observe/session/turns?${params}&includeContent=true&includeToolDetail=true`).then(r => r.json()),
      fetch(`/api/observe/session/bridges?${params}`).then(r => r.json()),
    ])
      .then(([wfData, turnsData, bridgesData]) => {
        if (cancelled) return
        setWorkflow(wfData as WorkflowTree | null)
        setTurns(turnsData.items ?? [])
        setBridges(bridgesData.items ?? [])
        setLoading(false)
      })
      .catch(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [taskId, framework])

  const phaseData = useMemo(() => {
    const turnById = new Map<string, TurnItem>()
    for (const t of turns) turnById.set(t.turnId, t)

    const subTurnsBySession = new Map<string, TurnItem[]>()
    for (const t of turns.filter(t => t.isSubagent)) {
      const sid = t.subagentSessionId ?? "unknown"
      const arr = subTurnsBySession.get(sid) ?? []
      arr.push(t)
      subTurnsBySession.set(sid, arr)
    }

    const bridgesByDispatch = new Map<string, BridgeItem[]>()
    for (const b of bridges) {
      if (!b.dispatchTurnId) continue
      const arr = bridgesByDispatch.get(b.dispatchTurnId) ?? []
      arr.push(b)
      bridgesByDispatch.set(b.dispatchTurnId, arr)
    }

    const rootTurns = turns.filter(t => !t.isSubagent && t.role !== "system" && t.agentName !== "continuation")

    // 1. Detect STATE.md phases from - [x] markers in root turn content
    const statePhases = detectStatePhases(rootTurns)

    // 2. If no STATE.md phases, create fallback from workflow phases or single phase
    let phases: StatePhase[]
    if (statePhases.length > 0) {
      phases = statePhases
    } else if (workflow && workflow.phases.length > 0) {
      phases = workflow.phases.map(p => ({
        label: p.fullLabel,
        turnIndexStart: p.turnIndexStart ?? 0,
        turnIndexEnd: p.turnIndexEnd ?? 999999,
      }))
    } else {
      phases = [{ label: "Main Workflow", turnIndexStart: 0, turnIndexEnd: 999999 }]
    }

    // 3. Flatten all designed steps (global, not per-phase)
    const allSteps: WorkflowStepNode[] = []
    if (workflow) {
      for (const phase of workflow.phases) {
        for (const child of phase.children) {
          if (child.type === "step") {
            allSteps.push(child as WorkflowStepNode)
          } else if (child.type === "parallel-group") {
            for (const s of (child as WorkflowParallelGroupNode).steps) {
              allSteps.push(s)
            }
          }
        }
      }
    }

    const matchedBridgeIds = new Set<string>()
    const allBridgesWithTurn: { bridge: BridgeItem; dispatchTurn: TurnItem }[] = []
    for (const turn of rootTurns) {
      const dispatches = bridgesByDispatch.get(turn.turnId) ?? []
      for (const b of dispatches) {
        allBridgesWithTurn.push({ bridge: b, dispatchTurn: turn })
      }
    }

    interface PhaseGroup {
      label: string
      turnIndexStart: number | null
      turnIndexEnd: number | null
      subagentGroups: SubagentGroup[]
      mainAgentTurns: MainTurnGroup[]
    }

    const phaseGroups: PhaseGroup[] = []

    for (const sp of phases) {
      const phaseRootTurns = rootTurns.filter(t => t.turnIndex >= sp.turnIndexStart && t.turnIndex <= sp.turnIndexEnd)

      // Bridges in this phase, grouped by subagent
      const bridgesBySub = new Map<string, { bridge: BridgeItem; dispatchTurn: TurnItem }[]>()
      for (const turn of phaseRootTurns) {
        const dispatches = bridgesByDispatch.get(turn.turnId) ?? []
        for (const b of dispatches) {
          const name = b.subagentName ?? "Other"
          const arr = bridgesBySub.get(name) ?? []
          arr.push({ bridge: b, dispatchTurn: turn })
          bridgesBySub.set(name, arr)
        }
      }

      // Steps matched to bridges in this phase
      const phaseBridgeIds = new Set<string>()
      for (const { bridge } of bridgesBySub.size > 0 ? [...bridgesBySub.values()].flat() : []) {
        phaseBridgeIds.add(bridge.bridgeId)
      }

      // Steps grouped by subagent (global matching, then filter to this phase's bridges)
      const stepsBySub = new Map<string, WorkflowStepNode[]>()
      for (const step of allSteps) {
        const match = allBridgesWithTurn.find(({ bridge }) => bridge.bridgeId === step.bridgeId)
        if (match && step.bridgeId) {
          if (phaseBridgeIds.has(step.bridgeId)) {
            matchedBridgeIds.add(step.bridgeId)
            const name = step.subagentName ?? "Other"
            const arr = stepsBySub.get(name) ?? []
            arr.push(step)
            stepsBySub.set(name, arr)
          }
        } else {
          // Missing step — assign to phase by triggerTurnId if in range
          const triggerTurn = step.triggerTurnId ? turnById.get(step.triggerTurnId) : null
          if (triggerTurn && triggerTurn.turnIndex >= sp.turnIndexStart && triggerTurn.turnIndex <= sp.turnIndexEnd) {
            const name = step.subagentName ?? "Other"
            const arr = stepsBySub.get(name) ?? []
            arr.push(step)
            stepsBySub.set(name, arr)
          } else if (!triggerTurn && sp === phases[phases.length - 1]) {
            // No trigger turn — put in last phase
            const name = step.subagentName ?? "Other"
            const arr = stepsBySub.get(name) ?? []
            arr.push(step)
            stepsBySub.set(name, arr)
          }
        }
      }

      // Build subagent groups
      const allNames = new Set([...stepsBySub.keys(), ...bridgesBySub.keys()])
      const subGroups: SubagentGroup[] = []

      for (const name of allNames) {
        const designedSteps = stepsBySub.get(name) ?? []
        const actualBridges = bridgesBySub.get(name) ?? []
        const tasks: UnifiedRow[] = []

        for (const step of designedSteps) {
          const match = actualBridges.find(({ bridge }) => bridge.bridgeId === step.bridgeId)
          if (match && step.bridgeId) {
            matchedBridgeIds.add(step.bridgeId)
            tasks.push({
              diffType: "matched", steps: [step],
              dispatchTurn: match.dispatchTurn, bridges: [match.bridge],
              subTurnsBySession, sortKey: match.dispatchTurn.turnIndex,
            })
          } else {
            tasks.push({ diffType: "missing", steps: [step], sortKey: step.stepIndex })
          }
        }

        for (const { bridge, dispatchTurn } of actualBridges) {
          if (!matchedBridgeIds.has(bridge.bridgeId)) {
            tasks.push({
              diffType: "extra-dispatch", steps: [],
              dispatchTurn, bridges: [bridge],
              subTurnsBySession, sortKey: dispatchTurn.turnIndex,
            })
          }
        }

        const typeOrder: Record<string, number> = { matched: 0, missing: 1, "extra-dispatch": 2 }
        tasks.sort((a, b) => {
          const d = typeOrder[a.diffType] - typeOrder[b.diffType]
          return d !== 0 ? d : a.sortKey - b.sortKey
        })

        if (tasks.length > 0) {
          subGroups.push({
            subagentName: name, tasks,
            matchedCount: tasks.filter(t => t.diffType === "matched").length,
            missingCount: tasks.filter(t => t.diffType === "missing").length,
            extraCount: tasks.filter(t => t.diffType === "extra-dispatch").length,
          })
        }
      }

      const phaseMainTurns = phaseRootTurns.filter(t =>
        t.role === "assistant" &&
        (bridgesByDispatch.get(t.turnId) ?? []).length === 0
      )
      const categoryMap = new Map<MainTurnCategory, TurnItem[]>()
      for (const t of phaseMainTurns) {
        const cat = classifyMainTurn(t)
        const arr = categoryMap.get(cat) ?? []
        arr.push(t)
        categoryMap.set(cat, arr)
      }
      const mainTurnGroups: MainTurnGroup[] = CATEGORY_ORDER
        .map(category => {
          const catTurns = categoryMap.get(category) ?? []
          return { category, turns: catTurns, totalMs: catTurns.reduce((s, t) => s + t.latencyMs, 0) }
        })
        .filter(g => g.turns.length > 0)

      if (subGroups.length > 0 || mainTurnGroups.length > 0) {
        phaseGroups.push({
          label: sp.label,
          turnIndexStart: sp.turnIndexStart,
          turnIndexEnd: sp.turnIndexEnd,
          subagentGroups: subGroups,
          mainAgentTurns: mainTurnGroups,
        })
      }
    }

    const totalDesigned = new Set<string>()
    for (const s of allSteps) {
      if (s.bridgeId) totalDesigned.add(s.bridgeId)
    }
    const totalMatched = matchedBridgeIds.size
    const totalMissing = totalDesigned.size - totalMatched
    const totalExtra = bridges.filter(b => !matchedBridgeIds.has(b.bridgeId)).length

    return { phaseGroups, totalMatched, totalMissing, totalExtra, totalDesigned: totalDesigned.size }
  }, [workflow, turns, bridges])

  if (loading) {
    return (
      <div className="rounded-lg border bg-card p-4 flex items-center justify-center h-32">
        <span className="text-sm text-muted-foreground">加载设计 vs 实际对比…</span>
      </div>
    )
  }

  const hasDesigned = workflow && workflow.phases.length > 0
  const sd = phaseData

  return (
    <div className="rounded-lg border bg-card overflow-hidden">
      <div
        className="flex items-center justify-between flex-wrap gap-2 px-4 py-2.5 cursor-pointer hover:bg-muted/30 select-none"
        onClick={() => setCollapsed(!collapsed)}
      >
        <div className="flex items-center gap-2">
          <span className="text-[10px] text-muted-foreground inline-block transition-transform duration-150"
                style={{ transform: collapsed ? "rotate(0deg)" : "rotate(90deg)" }}>▶</span>
          <h3 className="font-semibold text-sm flex items-center gap-2">
            <span className="text-base">⚖️</span> 设计 vs 实际 对比全景
          </h3>
        </div>
        <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-0.5"><span className="text-green-600 font-bold">✓</span> matched {sd.totalMatched}</span>
          <span className="flex items-center gap-0.5"><span className="text-red-600 font-bold">✗</span> missing {sd.totalMissing}</span>
          <span className="flex items-center gap-0.5"><span className="text-blue-600 font-bold">+</span> extra {sd.totalExtra}</span>
          {hasDesigned && sd.totalDesigned > 0 && (
            <span className="text-muted-foreground">· {sd.totalMatched}/{sd.totalDesigned} designed executed</span>
          )}
        </div>
      </div>

      <div className={collapsed ? "hidden" : "p-4 pt-0 space-y-4"}>

      {!hasDesigned && (
        <div className="text-xs text-amber-600 bg-amber-50 dark:bg-amber-500/10 rounded p-2">
          设计工作流数据不可用{wfError ? `（${wfError}）` : "（无 skill events 或 bridges）"}，仅显示实际轨迹
        </div>
      )}

      <div className="space-y-4">
        {sd.phaseGroups.map((pg, pi) => {
          const color = PHASE_COLORS[pi % PHASE_COLORS.length]
          return (
            <div key={pi} className="space-y-2">
              {/* Phase header — lightweight */}
              <div className="flex items-center gap-2 flex-wrap px-1">
                <span className="inline-block w-1 h-4 rounded-sm" style={{ background: color }} />
                <span className="font-semibold text-sm">{pg.label}</span>
                {pg.turnIndexStart != null && pg.turnIndexEnd != null && (
                  <span className="text-[10px] text-muted-foreground">§{pg.turnIndexStart}–{pg.turnIndexEnd}</span>
                )}
              </div>

              {/* Subagent groups within this phase */}
              {pg.subagentGroups.map((group, gi) => (
                <SubagentGroupBlock
                  key={gi}
                  group={group}
                  color={PHASE_COLORS[(pi + gi) % PHASE_COLORS.length]}
                  onSelectTurnId={onSelectTurnId}
                />
              ))}

              {/* Main Agent turns — parallel to subagent groups */}
              {pg.mainAgentTurns.length > 0 && (
                <MainAgentBlock groups={pg.mainAgentTurns} onSelectTurnId={onSelectTurnId} />
              )}
            </div>
          )
        })}

        {sd.phaseGroups.length === 0 && (
          <div className="text-xs text-muted-foreground text-center py-4">暂无轨迹数据</div>
        )}
      </div>
      </div>
    </div>
  )
}

interface SubagentGroup {
  subagentName: string
  tasks: UnifiedRow[]
  matchedCount: number
  missingCount: number
  extraCount: number
}

interface UnifiedRow {
  diffType: "matched" | "missing" | "extra-dispatch"
  steps: WorkflowStepNode[]
  parallelLabel?: string
  dispatchTurn?: TurnItem
  bridges?: BridgeItem[]
  subTurnsBySession?: Map<string, TurnItem[]>
  sortKey: number
}

function SubagentGroupBlock({ group, color, onSelectTurnId }: {
  group: SubagentGroup
  color: string
  onSelectTurnId?: (turnId: string) => void
}) {
  return (
    <div className="rounded-lg border overflow-hidden" style={{ borderColor: color + "44" }}>
      <div className="px-3 py-2 flex items-center gap-2 flex-wrap" style={{ background: color + "0d" }}>
        <span className="inline-block w-1 h-4 rounded-sm" style={{ background: color }} />
        <span className="font-semibold text-sm">{group.subagentName}</span>
        <span className="text-[10px] text-muted-foreground">{group.tasks.length} tasks</span>
        <span className="ml-auto text-[10px] flex items-center gap-2">
          <span className="text-green-600">✓{group.matchedCount}</span>
          {group.missingCount > 0 && <span className="text-red-600">✗{group.missingCount}</span>}
          {group.extraCount > 0 && <span className="text-blue-600">+{group.extraCount}</span>}
        </span>
      </div>

      <div className="grid grid-cols-[24px_1fr_1fr] px-3 py-1 border-b bg-muted/20 text-[10px] font-semibold text-muted-foreground">
        <span></span>
        <span className="pr-3">设计任务</span>
        <span className="pl-3 border-l">实际执行</span>
      </div>

      <div className="divide-y divide-border">
        {group.tasks.map((row, i) => (
          <UnifiedRow key={i} row={row} onSelectTurnId={onSelectTurnId} />
        ))}
      </div>
    </div>
  )
}

function UnifiedRow({ row, onSelectTurnId }: {
  row: UnifiedRow
  onSelectTurnId?: (turnId: string) => void
}) {
  const isMatched = row.diffType === "matched"
  const isMissing = row.diffType === "missing"

  const bgColor = isMatched ? "bg-green-50 dark:bg-green-500/5"
    : isMissing ? "bg-red-50 dark:bg-red-500/5"
    : "bg-blue-50 dark:bg-blue-500/5"
  const icon = isMatched ? "✓" : isMissing ? "✗" : "+"
  const iconColor = isMatched ? "#16a34a" : isMissing ? "#dc2626" : "#2563eb"

  return (
    <div className={`grid grid-cols-[24px_1fr_1fr] items-start px-3 py-2 ${bgColor}`}>
      <div className="pt-0.5">
        <span className="text-xs font-bold" style={{ color: iconColor }}>{icon}</span>
      </div>

      <div className="pr-3">
        {row.steps.length === 0 ? (
          <span className="text-[10px] text-muted-foreground italic">无设计对应</span>
        ) : (
          row.steps.map((step, i) => {
            const taskContent = row.bridges?.[i]?.dispatchContent
            return (
              <div key={i} className="cursor-pointer hover:text-primary" onClick={() => step.triggerTurnId && onSelectTurnId?.(step.triggerTurnId)}>
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span className="font-mono text-xs font-semibold">{shortSkill(step.stepName)}</span>
                  {step.subagentName && <span className="text-[10px] text-muted-foreground ml-1">↳ {step.subagentName}</span>}
                </div>
                {taskContent ? (
                  <div className="text-[10px] text-foreground/70 truncate mt-0.5">{truncate(taskContent, 50)}</div>
                ) : !isMatched ? (
                  <div className="text-[10px] text-muted-foreground/50 mt-0.5">（未执行，无任务内容）</div>
                ) : null}
              </div>
            )
          })
        )}
      </div>

      <div className="pl-3 border-l">
        {isMissing ? (
          <span className="text-xs text-red-600 font-semibold">— 没走 —</span>
        ) : (
          <>
            {row.dispatchTurn && (
              <div className="flex items-center gap-1.5 mb-0.5">
                <span className="text-[10px] font-mono font-semibold" style={{ color: ROLE_COLOR[row.dispatchTurn.role] ?? "#6b7280" }}>
                  §{row.dispatchTurn.turnIndex}
                </span>
                <span className="text-[10px] text-muted-foreground">{row.dispatchTurn.role}</span>
                {row.dispatchTurn.latencyMs > 0 && <span className="text-[10px] text-red-600">{fmtMs(row.dispatchTurn.latencyMs)}</span>}
                {row.dispatchTurn.totalTokens > 0 && <span className="text-[10px] text-blue-600">{fmtTok(row.dispatchTurn.totalTokens)}</span>}
              </div>
            )}
            {row.bridges?.map((bridge, i) => {
              const statusColor = STATUS_COLOR[bridge.status] ?? "#6b7280"
              const subTurns = row.subTurnsBySession?.get(bridge.subagentSessionId ?? "") ?? []
              const totalMs = bridge.subagentLatencyMs || subTurns.reduce((s, t) => s + t.latencyMs, 0)
              const totalTok = bridge.subagentTokens || subTurns.reduce((s, t) => s + t.totalTokens, 0)
              return (
                <div key={i} className="rounded border p-1.5 bg-card/50" style={{ borderColor: statusColor + "33" }}>
                  <div className="flex items-center gap-1.5 flex-wrap">
                    <span className="text-[10px] text-muted-foreground">§{subTurns[0]?.turnIndex ?? '?'}–{subTurns[subTurns.length - 1]?.turnIndex ?? '?'} ({subTurns.length} turns)</span>
                    <span className="text-[10px] text-red-600 font-semibold">{fmtMs(totalMs)}</span>
                    <span className="text-[10px] text-blue-600">{fmtTok(totalTok)} tokens</span>
                    <span className="text-[10px] font-semibold" style={{ color: statusColor }}>{bridge.status}</span>
                  </div>
                  {(() => {
                    const items = computePhaseBreakdown(subTurns, bridge.dispatchContent)
                    const total = items.reduce((s, i) => s + i.ms, 0)
                    if (total === 0) return null
                    return (
                      <div className="mt-1">
                        <div className="flex h-2 rounded overflow-hidden">
                          {items.map(i => (
                            <div key={i.label} style={{ width: `${(i.ms / total) * 100}%`, background: i.color }} title={`${i.label} ${fmtMs(i.ms)}`} />
                          ))}
                        </div>
                        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
                          {items.map(i => (
                            <span key={i.label} className="text-[9px] text-muted-foreground flex items-center gap-0.5">
                              <span className="inline-block w-2 h-2 rounded-sm" style={{ background: i.color }} />
                              {i.label} <span className="font-semibold">{fmtMs(i.ms)}</span>
                            </span>
                          ))}
                        </div>
                      </div>
                    )
                  })()}
                </div>
              )
            })}
          </>
        )}
      </div>
    </div>
  )
}

function MainAgentBlock({ groups, onSelectTurnId }: {
  groups: MainTurnGroup[]
  onSelectTurnId?: (turnId: string) => void
}) {
  const totalMs = groups.reduce((s, g) => s + g.totalMs, 0)
  const totalTurns = groups.reduce((s, g) => s + g.turns.length, 0)
  const nonZeroMs = groups.filter(g => g.totalMs > 0)

  return (
    <div className="rounded-lg border overflow-hidden" style={{ borderColor: "#6b728044" }}>
      <div className="px-3 py-2 flex items-center gap-2" style={{ background: "#6b72800d" }}>
        <span className="inline-block w-1 h-4 rounded-sm" style={{ background: "#6b7280" }} />
        <span className="font-semibold text-sm">主 Agent</span>
        <span className="text-[10px] text-muted-foreground">{totalTurns} turns</span>
        {totalMs > 0 && <span className="text-[10px] text-red-600 ml-auto">{fmtMs(totalMs)}</span>}
      </div>

      {nonZeroMs.length > 0 && (
        <div className="flex h-2 border-b">
          {nonZeroMs.map(g => (
            <div key={g.category}
                 style={{ width: `${(g.totalMs / totalMs) * 100}%`, background: CATEGORY_COLOR[g.category] }}
                 title={`${g.category} ${fmtMs(g.totalMs)}`} />
          ))}
        </div>
      )}

      <div className="divide-y divide-border">
        {groups.map(g => (
          <div key={g.category} className="px-3 py-1.5">
            <div className="flex items-center gap-2 mb-1">
              <span className="inline-block w-2 h-2 rounded-sm" style={{ background: CATEGORY_COLOR[g.category] }} />
              <span className="text-[10px] font-semibold">{g.category}</span>
              <span className="text-[10px] text-muted-foreground">{g.turns.length} turns</span>
              <span className="text-[10px] text-red-600 ml-auto">{fmtMs(g.totalMs)}</span>
            </div>
            <div className="space-y-0.5">
              {g.turns.map(t => (
                <div key={t.turnId} className="flex items-center gap-2 px-2 py-0.5 cursor-pointer hover:bg-muted/20 rounded"
                     onClick={() => onSelectTurnId?.(t.turnId)}>
                  <span className="text-[10px] font-mono font-semibold" style={{ color: ROLE_COLOR[t.role] ?? "#6b7280" }}>§{t.turnIndex}</span>
                  {t.latencyMs > 0 && <span className="text-[10px] text-red-600">{fmtMs(t.latencyMs)}</span>}
                  {t.totalTokens > 0 && <span className="text-[10px] text-blue-600">{fmtTok(t.totalTokens)}</span>}
                  {t.contentSummary && <span className="text-[10px] text-muted-foreground truncate flex-1">{truncate(t.contentSummary, 50)}</span>}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
