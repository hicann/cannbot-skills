"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useRef, useState } from "react"
import { use } from "react"
import { useSearchParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { ArrowLeftIcon, UploadIcon, LayoutDashboardIcon, MessageSquareIcon, GitBranchIcon, SearchIcon, UsersIcon, SparklesIcon, BarChart3Icon, FileSearchIcon, BrainIcon, FileTextIcon, PlayIcon, CheckCircleIcon, RefreshCwIcon, ShieldCheckIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import { VERSION_DISPLAY } from "@/lib/version"
import { BRAND_NAME } from "@/lib/branding"
import { TurnTimeline } from "@/components/observe/TurnTimeline"
import { TurnDetail } from "@/components/observe/TurnDetail"
import { TurnContextPanel } from "@/components/observe/TurnContextPanel"
import { SubagentCards } from "@/components/observe/SubagentCards"
import { InteractionGraph } from "@/components/observe/InteractionGraph"
import { SkillDetail } from "@/components/observe/SkillDetail"
import { WorkflowTreeView } from "@/components/observe/WorkflowTreeView"
import { WorkflowAIView } from "@/components/observe/WorkflowAIView"
import { WorkflowAnalyseTab } from "@/components/observe/WorkflowAnalyseTab"
import { TraceView } from "@/components/observe/TraceView"
import { ContextTracker } from "@/components/observe/ContextTracker"
import { FileReadAnalysis } from "@/components/observe/FileReadAnalysis"
import { AgentCallGraph } from "@/components/observe/AgentCallGraph"
import { AgentRelationGraph } from "@/components/observe/AgentRelationGraph"
import { ChatReplayView } from "@/components/observe/ChatReplayView"
import { summarizeToolCallErrors } from "@/lib/tool-call-errors"
import type { WorkflowTree } from "@/lib/ingest/phase-split"
import type { AIProviderConfig } from "@/lib/ai/analyzer"

type TabKey = "overview" | "turns" | "workflow" | "trace" | "subagents" | "skills" | "interactions" | "workflowAI" | "workflowAnalyse" | "context" | "fileReads" | "replay"

interface SessionData {
  sessionId: string
  taskId: string
  label: string | null
  query: string | null
  framework: string | null
  frameworkVersion: string | null
  parentId: string | null
  directory: string | null
  summaryAdditions: number
  summaryDeletions: number
  summaryFiles: number
  model: string | null
  sourcePath: string | null
  startTime: string
  endTime: string | null
  totalTokens: number
  totalInputTokens: number
  totalOutputTokens: number
  totalReasoningTokens: number
  totalCacheReadTokens: number
  totalCacheWriteTokens: number
  totalCost: number
  totalLatencyMs: number
  totalToolCallCount: number
  totalLlmCallCount: number
  totalSkillLoadCount: number
  totalSubagentCount: number
  agents: Array<{
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
  }>
  skills: Array<{
    skillName: string
    version: number | null
    invocationCount: number
  }>
}

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
  inputMessagesCount: number
  inputMessagesTokens: number
  contextWindowPct: number | null
  contextWindowLimit?: number
  latencyMs: number
  createdAt: string | null
  completedAt: string | null
  model: string | null
  toolCalls: Array<{ toolCallId: string; toolName: string; argsJson?: string | null; resultJson?: string | null; state: string; durationMs: number }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
}

interface TurnDetailData {
  turnId: string
  turnIndex: number
  role: string
  content: string | null
  contentJson: string | null
  contentSummary: string | null
  inputMessagesJson: string | null
  inputMessagesCount: number
  inputMessagesTokens: number
  contextWindowPct: number | null
  agentName: string | null
  subagentName: string | null
  isSubagent: boolean
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  contextWindowLimit?: number
  systemOverheadTokens?: number
  latencyMs: number
  ttftMs: number | null
  createdAt: string | null
  completedAt: string | null
  model: string | null
  modelId: string | null
  providerId: string | null
  finishReason: string | null
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

interface SubagentItem {
  executionId: string
  agentName: string | null
  agentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  parentExecutionId: string | null
  rootExecutionId: string | null
  depth: number
  tokens: number
  inputTokens: number
  outputTokens: number
  cost: number
  latencyMs: number
  toolCallCount: number
  skillLoadCount: number
  skillInvokeCount: number
  llmCallCount: number
  model: string | null
  createdAt: string
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

interface SkillEventForDetail {
  id: string
  skillName: string
  skillVersion: number | null
  eventType: string
  success: boolean
  errorMessage: string | null
  durationMs: number
  turnIndex: number
  agentName: string | null
  isSubagent: boolean
  subagentSessionId: string | null
  turnTokens: {
    totalTokens: number
    inputTokens: number
    outputTokens: number
    reasoningTokens: number
    cacheReadTokens: number
    cacheWriteTokens: number
  }
}

const HIDDEN_TABS_DEFAULT = ["subagents", "interactions", "workflowAI", "workflow", "replay"]

const ALL_TABS: Array<{ key: TabKey; label: string; icon: React.ReactNode; highlight?: boolean }> = [
  { key: "overview", label: "Overview", icon: <LayoutDashboardIcon className="size-3.5 text-blue-500" /> },
  { key: "turns", label: "Turns", icon: <MessageSquareIcon className="size-3.5 text-emerald-500" /> },
  { key: "trace", label: "Trace", icon: <SearchIcon className="size-3.5 text-yellow-500" /> },
  { key: "context", label: "Context", icon: <BarChart3Icon className="size-3.5 text-pink-500" /> },
  { key: "workflowAnalyse", label: "Audit", icon: <ShieldCheckIcon className="size-3.5 text-emerald-500" />, highlight: true },
  { key: "skills", label: "Skills", icon: <SparklesIcon className="size-3.5 text-orange-500" /> },
  { key: "fileReads", label: "File Reads", icon: <FileSearchIcon className="size-3.5 text-teal-500" /> },
  { key: "workflow", label: "Workflow", icon: <GitBranchIcon className="size-3.5 text-violet-500" />, highlight: true },
  { key: "subagents", label: "Subagents", icon: <UsersIcon className="size-3.5 text-cyan-500" /> },
  { key: "interactions", label: "Interactions", icon: <MessageSquareIcon className="size-3.5 text-emerald-500" /> },
  { key: "workflowAI", label: "AI Workflow", icon: <BrainIcon className="size-3.5 text-violet-500" />, highlight: true },
  { key: "replay", label: "Replay", icon: <PlayIcon className="size-3.5 text-pink-500" /> },
]

const showAdvanced = process.env.NEXT_PUBLIC_SHOW_ADVANCED_TABS === "true"
const TABS = showAdvanced ? ALL_TABS : ALL_TABS.filter(t => !HIDDEN_TABS_DEFAULT.includes(t.key))

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

function formatLatency(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "N/A"
  try {
    const d = new Date(ts)
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1)
    const day = String(d.getDate())
    const hour = String(d.getHours()).padStart(2, '0')
    const minute = String(d.getMinutes()).padStart(2, '0')
    return `${year}/${month}/${day} ${hour}:${minute}`
  } catch {
    return ts
  }
}

export default function SessionDetailPage({
  params,
}: {
  params: Promise<{ taskId: string }>
}) {
  const { taskId } = use(params)
  const searchParams = useSearchParams()
  const framework = searchParams.get("framework") ?? undefined
  const errorTurnParam = searchParams.get("errorTurn")
  const [activeTab, setActiveTab] = useState<TabKey>("overview")
  const [session, setSession] = useState<SessionData | null>(null)
  const [turns, setTurns] = useState<TurnRowItem[]>([])
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null)
  const [highlightSubagentTurnId, setHighlightSubagentTurnId] = useState<string | null>(null)
  const [scrollToTurnId, setScrollToTurnId] = useState<string | null>(null)
  const [selectedBridgeId, setSelectedBridgeId] = useState<string | null>(null)
  const [highlightSubagentSessionId, setHighlightSubagentSessionId] = useState<string | null>(null)
  const [highlightAgent, setHighlightAgent] = useState<string | null>(null)
  const [selectedTurnDetail, setSelectedTurnDetail] = useState<TurnDetailData | null>(null)
  const [executions, setExecutions] = useState<ExecutionItem[]>([])
  const [subagents, setSubagents] = useState<SubagentItem[]>([])
  const [bridges, setBridges] = useState<BridgeItem[]>([])
  const [workflowData, setWorkflowData] = useState<WorkflowTree | null>(null)
  const [workflowAIResult, setWorkflowAIResult] = useState<WorkflowTree | null>(null)
  const [workflowAIAnalyzing, setWorkflowAIAnalyzing] = useState(false)
  const [workflowAIError, setWorkflowAIError] = useState<string | null>(null)
  const [allSkillEvents, setAllSkillEvents] = useState<SkillEventForDetail[]>([])
  const [workflowSelectedTurnId, setWorkflowSelectedTurnId] = useState<string | null>(null)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [exporting, setExporting] = useState(false)
  const [exportingMd, setExportingMd] = useState(false)
  const [refreshing, setRefreshing] = useState(false)

  async function handleExport() {
    if (exporting) return
    setExporting(true)
    try {
      const res = await fetch("/api/ingest/export-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taskId }),
      })
      if (!res.ok) {
        const err = await res.json()
        toast.error("Export failed", { description: err.error ?? "Unknown error" })
        return
      }
      const blob = await res.blob()
      const defaultName = `cannbot_session_${taskId}.db`
      if (typeof window.showSaveFilePicker === "function") {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: defaultName,
            types: [{ description: "SQLite Database", accept: { "application/x-sqlite3": [".db"] } }],
          })
          const writable = await handle.createWritable()
          await writable.write(blob)
          await writable.close()
          toast.success("Database exported", {
            description: `Saved to ${handle.name}.`,
            icon: <CheckCircleIcon className="size-4" />,
            duration: 5000,
          })
          return
        } catch (e: unknown) {
          if (e instanceof DOMException && e.name === "AbortError") return
        }
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = defaultName
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => URL.revokeObjectURL(url), 10000)
      toast.success("Database exported", {
        description: `${defaultName} has been downloaded.`,
        icon: <CheckCircleIcon className="size-4" />,
        duration: 5000,
      })
    } catch {
      toast.error("Export failed", { description: "Network error" })
    } finally {
      setExporting(false)
    }
  }

  async function handleRefresh() {
    if (refreshing) return
    setRefreshing(true)
    try {
      const fwParam = framework ? `&framework=${encodeURIComponent(framework)}` : ""
      const res = await fetch("/api/ingest/refresh-session", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taskId, framework }),
      })
      const data = await res.json()
      if (!res.ok) {
        toast.error("刷新失败", { description: data.error ?? "Unknown error" })
        return
      }
      toast.success(data.message ?? "刷新完成")
      // Re-trigger all data fetches by re-running the useEffect
      setLoading(true)
      loadAllData()
    } catch {
      toast.error("刷新失败", { description: "网络错误" })
    } finally {
      setRefreshing(false)
    }
  }

  async function handleExportMd() {
    if (exportingMd) return
    setExportingMd(true)
    try {
      const params = framework ? `&framework=${encodeURIComponent(framework)}` : ""
      const res = await fetch(`/api/observe/session/export-md?taskId=${encodeURIComponent(taskId)}${params}`)
      if (!res.ok) {
        const err = await res.json()
        toast.error("Export Markdown failed", { description: err.error ?? "Unknown error" })
        return
      }
      const text = await res.text()
      const blob = new Blob([text], { type: "text/markdown" })
      const defaultName = `session_${taskId}.md`
      if (typeof window.showSaveFilePicker === "function") {
        try {
          const handle = await window.showSaveFilePicker({
            suggestedName: defaultName,
            types: [{ description: "Markdown", accept: { "text/markdown": [".md"] } }],
          })
          const writable = await handle.createWritable()
          await writable.write(blob)
          await writable.close()
          toast.success("Markdown exported", {
            description: `Saved to ${handle.name}.`,
            icon: <CheckCircleIcon className="size-4" />,
            duration: 5000,
            action: {
              label: "View",
              onClick: () => window.open(`/api/observe/session/export-md?taskId=${encodeURIComponent(taskId)}${params}`, "_blank"),
            },
          })
          return
        } catch (e: unknown) {
          if (e instanceof DOMException && e.name === "AbortError") return
        }
      }
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = defaultName
      document.body.appendChild(a)
      a.click()
      document.body.removeChild(a)
      setTimeout(() => URL.revokeObjectURL(url), 10000)
      toast.success("Markdown exported", {
        description: `${defaultName} has been downloaded.`,
        icon: <CheckCircleIcon className="size-4" />,
        duration: 5000,
        action: {
          label: "View",
          onClick: () => window.open(url, "_blank"),
        },
      })
    } catch {
      toast.error("Export Markdown failed", { description: "Network error" })
    } finally {
      setExportingMd(false)
    }
  }

  const turnDetailRef = useRef<HTMLDivElement>(null)

  const frameworkParam = framework ? `&framework=${encodeURIComponent(framework)}` : ""

  async function loadAllData() {
    async function fetchSessionData() {
      try {
        const sessionRes = await fetch(`/api/observe/session?taskId=${encodeURIComponent(taskId)}${frameworkParam}`)
        if (!sessionRes.ok) {
          const err = await sessionRes.json()
          setError(err.error ?? "Failed to load session")
          return
        }
        const sessionData = await sessionRes.json()
        setSession(sessionData)
      } catch {
        setError("Failed to load session data")
      }
    }

    async function fetchTurns() {
      try {
        const turnsRes = await fetch(`/api/observe/session/turns?taskId=${encodeURIComponent(taskId)}${frameworkParam}&includeToolDetail=true`)
        if (turnsRes.ok) {
          const turnsData = await turnsRes.json()
          setTurns(turnsData.items ?? [])
        }
      } catch {
        setError("Failed to load turns")
      }
    }

    async function fetchExecutions() {
      try {
        const res = await fetch(`/api/observe/executions?taskId=${encodeURIComponent(taskId)}${frameworkParam}`)
        if (res.ok) {
          const data = await res.json()
          setExecutions(data.items ?? [])
          setSubagents(data.subagents ?? [])
        }
      } catch {
        setError("Failed to load executions")
      }
    }

    async function fetchBridges() {
      try {
        const res = await fetch(`/api/observe/session/bridges?taskId=${encodeURIComponent(taskId)}${frameworkParam}`)
        if (res.ok) {
          const data = await res.json()
          setBridges(data.items ?? [])
        }
      } catch {
        setError("Failed to load bridges")
      }
    }

    async function fetchSkillEvents() {
      try {
        const res = await fetch(`/api/observe/session/turns?taskId=${encodeURIComponent(taskId)}`)
        if (res.ok) {
          const data = await res.json()
          const events: SkillEventForDetail[] = []
          for (const turn of data.items ?? []) {
            for (const se of turn.skillEvents ?? []) {
              events.push({
                id: `${turn.turnId}-${se.skillName}-${se.eventType}`,
                skillName: se.skillName,
                skillVersion: null,
                eventType: se.eventType,
                success: se.success,
                errorMessage: null,
                durationMs: 0,
                turnIndex: turn.turnIndex ?? 0,
                agentName: turn.agentName ?? null,
                isSubagent: turn.isSubagent ?? false,
                subagentSessionId: turn.subagentSessionId ?? null,
                turnTokens: {
                  totalTokens: turn.totalTokens ?? 0,
                  inputTokens: turn.inputTokens ?? 0,
                  outputTokens: turn.outputTokens ?? 0,
                  reasoningTokens: turn.reasoningTokens ?? 0,
                  cacheReadTokens: turn.cacheReadTokens ?? 0,
                  cacheWriteTokens: turn.cacheWriteTokens ?? 0,
                },
              })
            }
          }
          setAllSkillEvents(events)
        }
      } catch {
        setError("Failed to load skill events")
      }
    }

    async function fetchWorkflow() {
      try {
        const res = await fetch(`/api/observe/session/workflow?taskId=${encodeURIComponent(taskId)}${frameworkParam}`)
        if (res.ok) {
          const data = await res.json()
          setWorkflowData(data)
        }
      } catch {
      }
    }

    setLoading(true)
    Promise.all([fetchSessionData(), fetchTurns(), fetchExecutions(), fetchBridges(), fetchSkillEvents(), fetchWorkflow()])
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    loadAllData()
  }, [taskId])

  useEffect(() => {
    if (!selectedTurnId) {
      setSelectedTurnDetail(null)
      return
    }

    async function fetchTurnDetail() {
      try {
        const res = await fetch(`/api/observe/session/turns/${encodeURIComponent(selectedTurnId!)}`)
        if (res.ok) {
          const data = await res.json()
          setSelectedTurnDetail(data)
        }
      } catch {
        setSelectedTurnDetail(null)
      }
    }

    fetchTurnDetail()
  }, [selectedTurnId])

  // No scroll on turn selection — keep current scroll position

  // Auto-select error turn from URL param
  useEffect(() => {
    if (errorTurnParam && turns.length > 0 && !selectedTurnId) {
      const turnIndex = Number(errorTurnParam)
      const errorTurn = turns.find(t => t.turnIndex === turnIndex)
      if (errorTurn) {
        setSelectedTurnId(errorTurn.turnId)
        if (errorTurn.isSubagent) setHighlightSubagentTurnId(errorTurn.turnId)
        setScrollToTurnId(errorTurn.turnId)
      }
    }
  }, [errorTurnParam, turns])


  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-muted-foreground">Loading session...</div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-red-600 dark:text-red-400">{error}</div>
      </div>
    )
  }

  if (!session) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-muted-foreground">Session not found</div>
      </div>
    )
  }

  const s = session

  // Compute endContextWindowPct: next same-scope assistant turn's contextWindowPct
  // (that IS the context % at the end of this turn)
  function computeEndPct(turn: TurnRowItem | null): number | null {
    if (!turn) return null
    const sortedTurns = [...turns].sort((a, b) => a.turnIndex - b.turnIndex)
    const nextTurn = sortedTurns.find(t =>
      t.turnIndex > turn.turnIndex &&
      t.role === 'assistant' &&
      t.isSubagent === turn.isSubagent &&
      (turn.isSubagent ? t.subagentSessionId === turn.subagentSessionId : true)
    )
    if (nextTurn?.contextWindowPct != null) return nextTurn.contextWindowPct
    if (turn.contextWindowPct != null && turn.outputTokens > 0) {
      const deltaPct = turn.outputTokens / 200000 * 100
      return turn.contextWindowPct + deltaPct
    }
    return null
  }

  function renderOverview() {
    return (
      <div className="p-4 space-y-4 overflow-y-auto h-full min-h-0">
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Tokens</span>
              <span className="text-sm font-medium tabular-nums">{formatTokenCount(s.totalTokens)}</span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Cost</span>
              <span className="text-sm font-medium tabular-nums">{formatCost(s.totalCost)}</span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Wall Clock</span>
              <span className="text-sm font-medium tabular-nums">
                {(() => {
                  const start = s.startTime ? new Date(s.startTime).getTime() : 0
                  const end = s.endTime ? new Date(s.endTime).getTime() : start
                  return formatLatency(end - start)
                })()}
              </span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">LLM Calls</span>
              <span className="text-sm font-medium tabular-nums">{s.totalLlmCallCount}</span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Tool Calls</span>
              <span className="text-sm font-medium tabular-nums">{s.totalToolCallCount}</span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Skills</span>
              <span className="text-sm font-medium tabular-nums">{s.totalSkillLoadCount}</span>
            </CardContent>
          </Card>
          <Card size="sm" className="flex-1">
            <CardContent className="flex items-center gap-2 py-2">
              <span className="text-xs text-muted-foreground">Subagents</span>
              <span className="text-sm font-medium tabular-nums">{s.totalSubagentCount}</span>
            </CardContent>
          </Card>
          {(() => {
            const allErrors = turns
              .map(t => ({ turn: t, errors: summarizeToolCallErrors(t.toolCalls, t.skillEvents) }))
              .filter(({ errors }) => errors.total > 0)
            const totalErrorCount = allErrors.reduce((s, { errors }) => s + errors.total, 0)
            if (totalErrorCount === 0) return null
            return (
              <Card size="sm" className="flex-1 border-red-200 dark:border-red-500/30 cursor-pointer hover:bg-red-100/30 dark:hover:bg-red-500/10 transition-colors"
                onClick={() => {
                  if (allErrors.length > 0) {
                    const first = allErrors[0].turn
                    setSelectedTurnId(first.turnId)
                    if (first.isSubagent) setHighlightSubagentTurnId(first.turnId)
                    setScrollToTurnId(first.turnId)
                    setActiveTab("turns")
                  }
                }}
              >
                <CardContent className="flex items-center gap-2 py-2">
                  <span className="text-xs text-red-600 dark:text-red-400">⚠ Errors</span>
                  <span className="text-sm font-medium tabular-nums text-red-600 dark:text-red-400">{totalErrorCount}</span>
                </CardContent>
              </Card>
            )
          })()}
        </div>

        <div className="space-y-4">
          <AgentCallGraph agents={s.agents} bridges={bridges} onViewTurns={(agentSessionId) => {
            if (agentSessionId) {
              setHighlightSubagentTurnId(agentSessionId)
              const firstSubTurn = turns.find(t => t.isSubagent && t.subagentSessionId === agentSessionId)
              if (firstSubTurn) {
                setSelectedTurnId(firstSubTurn.turnId)
                setScrollToTurnId(firstSubTurn.turnId)
              }
            }
            setActiveTab("turns")
          }} />

          {(() => {
            const errorTurns = turns
              .map(t => ({ turn: t, errors: summarizeToolCallErrors(t.toolCalls, t.skillEvents) }))
              .filter(({ errors }) => errors.total > 0)

            if (errorTurns.length === 0) return <div />

            return (
              <Card size="sm" className="border-red-200 dark:border-red-500/30">
                <CardHeader>
                  <CardTitle className="text-red-600 dark:text-red-400">⚠ Error Turns ({errorTurns.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-2">
                    {errorTurns.map(({ turn: t, errors }) => (
                      <button
                        key={t.turnId}
                        className="w-full px-2 py-1.5 rounded-md border border-red-100 dark:border-red-500/20 bg-red-50/30 dark:bg-red-500/5 text-xs hover:bg-red-100/50 dark:hover:bg-red-500/10 transition-colors cursor-pointer text-left"
                        onClick={() => {
                          setSelectedTurnId(t.turnId)
                          if (t.isSubagent) setHighlightSubagentTurnId(t.turnId)
                          setScrollToTurnId(t.turnId)
                          setActiveTab("turns")
                        }}
                      >
                        <div className="flex items-center gap-2 mb-0.5">
                          <span className="font-mono text-muted-foreground">#{t.turnIndex}</span>
                          <Badge variant="outline" className="text-xs">{t.role}</Badge>
                          {t.isSubagent && t.subagentName && <Badge variant="orange" className="text-xs">🔗 {t.subagentName}</Badge>}
                          {errors.cancelled > 0 && <Badge variant="orange" className="text-xs">{errors.cancelled} cancelled</Badge>}
                          {errors.failed > 0 && <Badge variant="red" className="text-xs">{errors.failed} failed</Badge>}
                          {errors.skillFail > 0 && <Badge variant="red" className="text-xs">{errors.skillFail} skill_fail</Badge>}
                          {t.model && <span className="text-muted-foreground ml-auto">{t.model}</span>}
                        </div>
                        {t.contentSummary && (
                          <p className="text-foreground/80 truncate mb-0.5">{t.contentSummary.substring(0, 80)}</p>
                        )}
                        <div className="flex flex-wrap gap-x-3 gap-y-0.5 text-muted-foreground">
                          {errors.details.map((d, i) => (
                            <span key={i} className={d.type === "failed" ? "text-red-500 dark:text-red-400" : d.type === "cancelled" ? "text-orange-500" : "text-red-500"}>
                              {d.type === "skill_fail" ? "⚡" : "🔧"} {d.toolName}
                            </span>
                          ))}
                          {t.toolCalls.filter(tc => {
                            const r = tc.resultJson ?? ""
                            return r.includes("Exit code") || r.includes("<tool_use_error>") || tc.state === "error" || tc.state === "failed"
                          }).map(tc => {
                            const r = tc.resultJson ?? ""
                            const exitMatch = r.match(/Exit code (\d+)/)
                            const errMsg = r.includes("<tool_use_error>") ? r.replace(/.*<tool_use_error>/, "").replace(/<\/tool_use_error>.*/, "").substring(0, 60) : exitMatch ? `exit ${exitMatch[1]}` : ""
                            return errMsg ? <span key={tc.toolCallId} className="text-red-500/80 dark:text-red-400/80 truncate max-w-[180px]">{tc.toolName}: {errMsg}</span> : null
                          })}
                          {t.skillEvents.filter(se => !se.success && se.errorMessage).map(se => (
                            <span key={se.skillName} className="text-red-500/80 dark:text-red-400/80 truncate max-w-[180px]">⚡ {se.skillName}: {se.errorMessage!.substring(0, 60)}</span>
                          ))}
                        </div>
                      </button>
                    ))}
                  </div>
                </CardContent>
              </Card>
            )
          })()}
        </div>

        <AgentRelationGraph agents={s.agents} bridges={bridges} onViewTurns={(agentSessionId) => {
          if (agentSessionId) {
            setHighlightSubagentTurnId(agentSessionId)
            const firstSubTurn = turns.find(t => t.isSubagent && t.subagentSessionId === agentSessionId)
            if (firstSubTurn) {
              setSelectedTurnId(firstSubTurn.turnId)
              setScrollToTurnId(firstSubTurn.turnId)
            }
          }
          setActiveTab("turns")
        }} />

        <Card size="sm">
          <CardHeader>
            <CardTitle>Skills ({s.skills.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {s.skills.length === 0 ? (
              <div className="text-sm text-muted-foreground">No skills loaded</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {s.skills.map(sn => {
                  const skillEvents = allSkillEvents.filter(e => e.skillName === sn.skillName)
                  const skillTokens = skillEvents.reduce((sum, e) => sum + e.turnTokens.totalTokens, 0)
                  const formatT = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}k` : `${n}`
                  return (
                    <div key={sn.skillName} className="flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs">
                      <span className="font-medium">{sn.skillName}</span>
                      {sn.version != null && <span className="text-muted-foreground">v{sn.version}</span>}
                      <Badge variant="green" className="text-xs">{sn.invocationCount}x</Badge>
                      {skillTokens > 0 && (
                        <Badge variant="outline" className="text-xs">{formatT(skillTokens)}t</Badge>
                      )}
                    </div>
                  )
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card size="sm">
          <CardHeader>
            <CardTitle>Tool Calls ({(() => { const n = turns.flatMap(t => t.toolCalls ?? []).length; return n })()})</CardTitle>
          </CardHeader>
          <CardContent>
            {(() => {
              const allToolCalls = turns.flatMap(t => t.toolCalls ?? [])
              if (allToolCalls.length === 0) return <div className="text-xs text-muted-foreground">No tool calls</div>
              const grouped = new Map<string, { count: number; avgDuration: number; errorCount: number }>()
              for (const tc of allToolCalls) {
                const existing = grouped.get(tc.toolName) ?? { count: 0, avgDuration: 0, errorCount: 0 }
                existing.count++
                existing.avgDuration += tc.durationMs
                if (tc.state === "error") existing.errorCount++
                grouped.set(tc.toolName, existing)
              }
              const sorted = [...grouped.entries()].sort((a, b) => b[1].count - a[1].count)
              const totalErrors = sorted.reduce((s, [, st]) => s + st.errorCount, 0)
              return (
                <div className="space-y-2">
                  <div className="flex flex-wrap gap-2">
                    {sorted.map(([name, stats]) => (
                      <div key={name} className="flex items-center gap-1.5 px-2 py-1 rounded-md border text-xs">
                        <span className="font-medium">{name}</span>
                        <Badge variant="outline" className="text-xs">{stats.count}x</Badge>
                        <span className="text-muted-foreground">{formatLatency(Math.round(stats.avgDuration / stats.count))}</span>
                        {stats.errorCount > 0 && <Badge variant="red" className="text-xs">{stats.errorCount} err</Badge>}
                      </div>
                    ))}
                  </div>
                  {totalErrors > 0 && (
                    <div className="flex items-center gap-1.5 text-xs">
                      <Badge variant="red">{totalErrors} errors total</Badge>
                    </div>
                  )}
                </div>
              )
            })()}
          </CardContent>
        </Card>
      </div>
    )
  }

  function renderTurns() {
    const bridgeItems = bridges.map(b => ({
      bridgeId: b.bridgeId,
      dispatchTurnId: b.dispatchTurnId,
      dispatchContent: b.dispatchContent,
      subagentSessionId: b.subagentSessionId,
      subagentType: b.subagentType,
      subagentName: b.subagentName,
      agentName: b.agentName,
      status: b.status,
      subagentTokens: b.subagentTokens,
      subagentLatencyMs: b.subagentLatencyMs,
    }))

    // Build context data for selected turn
    const selectedTurnItem = selectedTurnId ? turns.find(t => t.turnId === selectedTurnId) : null

    // Root context: the selected turn itself (if it's a root turn)
    const rootContext = selectedTurnItem && !selectedTurnItem.isSubagent ? {
      label: "Root Agent",
      agentName: selectedTurnItem.agentName ?? "root",
      model: selectedTurnItem.model ?? null,
      inputMessagesJson: null as string | null,
      inputMessagesCount: selectedTurnDetail?.inputMessagesCount ?? 0,
      inputMessagesTokens: selectedTurnDetail?.inputMessagesTokens ?? 0,
      contextWindowPct: selectedTurnDetail?.contextWindowPct ?? null,
      endContextWindowPct: computeEndPct(selectedTurnItem),
      contextWindowLimit: selectedTurnDetail?.contextWindowLimit ?? 200000,
      systemOverheadTokens: selectedTurnDetail?.systemOverheadTokens ?? 0,
      cacheReadTokens: selectedTurnDetail?.cacheReadTokens ?? 0,
      cacheWriteTokens: selectedTurnDetail?.cacheWriteTokens ?? 0,
      isSubagent: false,
      subagentName: null,
    } : selectedTurnItem && selectedTurnItem.isSubagent ? (() => {
      // If a subagent turn is selected, find the root turn at same time
      const rootTurns = turns.filter(t => !t.isSubagent)
      const rootTurn = rootTurns.reduce((best, t) => {
        if (!t.createdAt) return best
        if (!selectedTurnItem.createdAt) return best
        const rootTime = new Date(t.createdAt).getTime()
        const selectedTime = new Date(selectedTurnItem.createdAt).getTime()
        if (rootTime <= selectedTime && (!best || new Date(best.createdAt!).getTime() < rootTime)) return t
        return best
      }, null as TurnRowItem | null)
      return {
        label: "Root Agent",
        agentName: rootTurn?.agentName ?? "root",
        model: rootTurn?.model ?? null,
        inputMessagesJson: null as string | null,
        inputMessagesCount: rootTurn?.inputMessagesCount ?? 0,
        inputMessagesTokens: rootTurn?.inputMessagesTokens ?? 0,
        contextWindowPct: rootTurn?.contextWindowPct ?? null,
        endContextWindowPct: computeEndPct(rootTurn ?? null),
        contextWindowLimit: rootTurn?.contextWindowLimit ?? 200000,
        systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
        isSubagent: false,
        subagentName: null,
      }
    })() : null

    // Subagent contexts: find bridges where selected turn dispatched a subagent
    const subagentContexts: Array<{
      label: string
      agentName: string | null
      model: string | null
      inputMessagesJson: string | null
      inputMessagesCount: number
      inputMessagesTokens: number
      contextWindowPct: number | null
      endContextWindowPct: number | null
      contextWindowLimit: number
      systemOverheadTokens: number
      cacheReadTokens: number
      cacheWriteTokens: number
      isSubagent: boolean
      subagentName: string | null
    }> = []

    if (selectedTurnId) {
      const dispatchBridges = bridges.filter(b => b.dispatchTurnId === selectedTurnId)
      for (const bridge of dispatchBridges) {
        if (bridge.subagentSessionId) {
          const subTurns = turns.filter(t => t.subagentSessionId === bridge.subagentSessionId)
          const lastSubTurn = subTurns[subTurns.length - 1]
          subagentContexts.push({
            label: bridge.subagentName ?? bridge.subagentType ?? "subagent",
            agentName: lastSubTurn?.agentName ?? null,
            model: lastSubTurn?.model ?? null,
            inputMessagesJson: null as string | null,
            inputMessagesCount: 0,
            inputMessagesTokens: 0,
            contextWindowPct: lastSubTurn?.contextWindowPct ?? null,
            endContextWindowPct: computeEndPct(lastSubTurn ?? null),
            contextWindowLimit: selectedTurnDetail?.contextWindowLimit ?? 200000,
            systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
            isSubagent: true,
            subagentName: bridge.subagentName ?? bridge.subagentType ?? null,
          })
        }
      }

      // Also: if selected turn IS a subagent turn, show its own context
      if (selectedTurnItem?.isSubagent && selectedTurnDetail) {
        subagentContexts.push({
          label: selectedTurnItem.subagentName ?? "subagent",
          agentName: selectedTurnItem.agentName ?? null,
          model: selectedTurnDetail.model ?? null,
          inputMessagesJson: null as string | null,
          inputMessagesCount: selectedTurnDetail.inputMessagesCount ?? 0,
          inputMessagesTokens: selectedTurnDetail.inputMessagesTokens ?? 0,
          contextWindowPct: selectedTurnDetail.contextWindowPct ?? null,
          endContextWindowPct: computeEndPct(selectedTurnItem),
          contextWindowLimit: selectedTurnDetail.contextWindowLimit ?? 200000,
          systemOverheadTokens: selectedTurnDetail.systemOverheadTokens ?? 0,
          cacheReadTokens: 0,
          cacheWriteTokens: 0,
          isSubagent: true,
          subagentName: selectedTurnItem.subagentName ?? null,
        })
      }
    }

    // Load detailed context for selected turn
    if (selectedTurnItem && selectedTurnDetail) {
      // Only update rootContext if the selected turn IS a root turn
      if (rootContext && !selectedTurnItem.isSubagent) {
        rootContext.inputMessagesJson = selectedTurnDetail.inputMessagesJson ?? null
        rootContext.inputMessagesCount = selectedTurnDetail.inputMessagesCount ?? 0
        rootContext.inputMessagesTokens = selectedTurnDetail.inputMessagesTokens ?? 0
        rootContext.contextWindowPct = selectedTurnDetail.contextWindowPct ?? null
      }
      // Also update subagent context if selected turn is a subagent
      const selfSubCtx = selectedTurnItem.isSubagent
        ? subagentContexts.find(c => c.label === (selectedTurnItem.subagentName ?? "subagent") && c.contextWindowPct === selectedTurnDetail.contextWindowPct)
        : null
      if (selfSubCtx) {
        selfSubCtx.inputMessagesJson = selectedTurnDetail.inputMessagesJson ?? null
        selfSubCtx.inputMessagesCount = selectedTurnDetail.inputMessagesCount ?? 0
        selfSubCtx.inputMessagesTokens = selectedTurnDetail.inputMessagesTokens ?? 0
      }
    }

    // Find previous root assistant turn's context for comparison
    const prevRootPct = (() => {
      if (!selectedTurnItem) return null
      const rootAssistantTurns = turns.filter(t => !t.isSubagent && t.role === 'assistant')
      const currentIdx = rootAssistantTurns.findIndex(t => t.turnId === selectedTurnId)
      if (currentIdx < 0) {
        // Selected turn might be a user/system turn, find nearest assistant
        const sorted = turns.filter(t => !t.isSubagent).sort((a, b) => a.turnIndex - b.turnIndex)
        const selIdx = sorted.findIndex(t => t.turnId === selectedTurnId)
        if (selIdx < 0) return null
        const prevAssistant = sorted.slice(0, selIdx).reverse().find(t => t.role === 'assistant')
        return prevAssistant?.contextWindowPct ?? null
      }
      const prevAssistant = rootAssistantTurns[currentIdx - 1]
      return prevAssistant?.contextWindowPct ?? null
    })()

    return (
      <div className="flex flex-1 h-full min-h-0">
        <div className="w-[400px] border-r flex flex-col min-h-0 overflow-y-auto">
          <TurnTimeline
            turns={turns}
            bridges={bridgeItems}
            selectedTurnId={selectedTurnId}
            highlightSubagentTurnId={highlightSubagentTurnId}
            scrollToTurnId={scrollToTurnId}
            onSelectTurn={(turnId) => {
              setSelectedTurnId(turnId)
              setHighlightSubagentTurnId(null)
              setScrollToTurnId(null)
            }}
          />
        </div>

        <div ref={turnDetailRef} className="flex-1 min-h-0 overflow-y-auto">
          {selectedTurnDetail ? (
            <TurnDetail turn={selectedTurnDetail} />
          ) : (
            <div className="flex items-center justify-center h-full text-muted-foreground">
              Select a turn from the timeline
            </div>
          )}
        </div>

        <div className="w-[320px] border-l flex flex-col min-h-0 overflow-hidden">
          <TurnContextPanel
            selectedTurn={selectedTurnItem ? { turnId: selectedTurnItem.turnId, turnIndex: selectedTurnItem.turnIndex, role: selectedTurnItem.role } : null}
            rootContext={rootContext}
            subagentContexts={subagentContexts}
            prevContextPct={prevRootPct}
          />
        </div>
      </div>
    )
  }

  function renderSubagents() {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <SubagentCards
          subagents={subagents}
          bridges={bridges}
          highlightAgent={highlightAgent}
          onViewTurns={(agentSessionId) => {
            if (agentSessionId) {
              setHighlightSubagentTurnId(agentSessionId)
              const firstSubTurn = turns.find(t => t.isSubagent && t.subagentSessionId === agentSessionId)
              if (firstSubTurn) {
                setSelectedTurnId(firstSubTurn.turnId)
                setScrollToTurnId(firstSubTurn.turnId)
              }
              setActiveTab("turns")
            }
          }}
        />
      </div>
    )
  }

  function renderSkills() {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <SkillDetail
          sessionSkills={s.skills}
          skillEvents={allSkillEvents}
          onNavigateToTurn={(turnIndex) => {
            const turn = turns.find(t => t.turnIndex === turnIndex)
            if (turn) {
              setSelectedTurnId(turn.turnId)
              if (turn.isSubagent) {
                setHighlightSubagentTurnId(turn.turnId)
              } else {
                setHighlightSubagentTurnId(null)
              }
              setScrollToTurnId(turn.turnId)
              setActiveTab("turns")
            }
          }}
        />
      </div>
    )
  }

  function renderInteractions() {
    const rootAgentName = s.agents.find(a => !a.isSubagent)?.agentName ?? null
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <InteractionGraph
          bridges={bridges}
          rootAgentName={rootAgentName}
          sessionStartTime={s.startTime}
          sessionLatencyMs={s.totalLatencyMs}
        />
      </div>
    )
  }

  function renderWorkflow() {
    if (!workflowData) {
      return (
        <div className="flex items-center justify-center h-full text-muted-foreground">
          No workflow data available
        </div>
      )
    }
    const workflowBridges: Array<{
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
    }> = bridges.map(b => ({
      bridgeId: b.bridgeId,
      dispatchContent: b.dispatchContent,
      dispatchTimestamp: b.dispatchTimestamp,
      responseContent: b.responseContent,
      responseTimestamp: b.responseTimestamp,
      subagentName: b.subagentName,
      subagentType: b.subagentType,
      status: b.status,
      subagentTokens: b.subagentTokens,
      subagentLatencyMs: b.subagentLatencyMs,
    }))
    const workflowTurnRanges: Array<{
      turnIndex: number
      subagentSessionId: string | null
      isSubagent: boolean
      role: string
    }> = turns.map(t => ({
      turnIndex: t.turnIndex,
      subagentSessionId: t.subagentSessionId,
      isSubagent: t.isSubagent,
      role: t.role,
    }))

    const workflowTurn = workflowSelectedTurnId ? turns.find(t => t.turnId === workflowSelectedTurnId) : null

    const workflowRootContext = workflowTurn && !workflowTurn.isSubagent ? {
      agentName: workflowTurn.agentName ?? "root",
      model: workflowTurn.model ?? null,
      inputMessagesJson: null as string | null,
      inputMessagesCount: 0,
      inputMessagesTokens: workflowTurn.inputTokens,
      contextWindowPct: workflowTurn.contextWindowPct ?? null,
      endContextWindowPct: computeEndPct(workflowTurn),
      contextWindowLimit: 200000,
      systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      isSubagent: false,
      subagentName: null,
    } : workflowTurn && workflowTurn.isSubagent ? (() => {
      const rootTurns = turns.filter(t => !t.isSubagent)
      const rootTurn = rootTurns.reduce((best, t) => {
        if (!t.createdAt || !workflowTurn.createdAt) return best
        const rootTime = new Date(t.createdAt).getTime()
        const selectedTime = new Date(workflowTurn.createdAt).getTime()
        if (rootTime <= selectedTime && (!best || new Date(best.createdAt!).getTime() < rootTime)) return t
        return best
      }, null as TurnRowItem | null)
      return {
        agentName: rootTurn?.agentName ?? "root",
        model: rootTurn?.model ?? null,
        inputMessagesJson: null as string | null,
        inputMessagesCount: 0,
        inputMessagesTokens: 0,
        contextWindowPct: null,
        endContextWindowPct: null,
        contextWindowLimit: 200000,
        systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
        isSubagent: false,
        subagentName: null,
      }
    })() : null

    const workflowSubagentContexts: Array<{
      agentName: string | null
      model: string | null
      inputMessagesJson: string | null
      inputMessagesCount: number
      inputMessagesTokens: number
      contextWindowPct: number | null
      endContextWindowPct: number | null
      contextWindowLimit: number
      systemOverheadTokens: number
      cacheReadTokens: number
      cacheWriteTokens: number
      isSubagent: boolean
      subagentName: string | null
    }> = []

    if (workflowTurn?.isSubagent) {
      workflowSubagentContexts.push({
        agentName: workflowTurn.agentName ?? null,
        model: workflowTurn.model ?? null,
        inputMessagesJson: null,
        inputMessagesCount: 0,
        inputMessagesTokens: workflowTurn.inputTokens,
        contextWindowPct: workflowTurn.contextWindowPct ?? null,
        endContextWindowPct: computeEndPct(workflowTurn),
        contextWindowLimit: 200000,
        systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
        isSubagent: true,
        subagentName: workflowTurn.subagentName ?? null,
      })
    } else if (workflowTurn) {
      const dispatchBridges = bridges.filter(b => b.dispatchTurnId === workflowTurn.turnId)
      for (const bridge of dispatchBridges) {
        if (bridge.subagentSessionId) {
          const subTurns = turns.filter(t => t.subagentSessionId === bridge.subagentSessionId)
          const lastSubTurn = subTurns[subTurns.length - 1]
          workflowSubagentContexts.push({
            agentName: lastSubTurn?.agentName ?? null,
            model: lastSubTurn?.model ?? null,
            inputMessagesJson: null,
            inputMessagesCount: 0,
            inputMessagesTokens: lastSubTurn?.inputTokens ?? 0,
            contextWindowPct: lastSubTurn?.contextWindowPct ?? null,
            endContextWindowPct: computeEndPct(lastSubTurn ?? null),
            contextWindowLimit: 200000,
            systemOverheadTokens: 0,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
            isSubagent: true,
            subagentName: bridge.subagentName ?? bridge.subagentType ?? null,
          })
        }
      }
    }

    const workflowPrevPct = (() => {
      if (!workflowTurn) return null
      const rootAssistantTurns = turns.filter(t => !t.isSubagent && t.role === 'assistant')
      const currentIdx = rootAssistantTurns.findIndex(t => t.turnId === workflowTurn.turnId)
      if (currentIdx > 0) return rootAssistantTurns[currentIdx - 1]?.contextWindowPct ?? null
      const sorted = turns.filter(t => !t.isSubagent).sort((a, b) => a.turnIndex - b.turnIndex)
      const selIdx = sorted.findIndex(t => t.turnId === workflowTurn.turnId)
      if (selIdx < 0) return null
      const prevAssistant = sorted.slice(0, selIdx).reverse().find(t => t.role === 'assistant')
      return prevAssistant?.contextWindowPct ?? null
    })()

    return (
      <div className="flex flex-1 h-full min-h-0">
        <div className="flex-1 min-h-0 overflow-y-auto border-r">
          <WorkflowTreeView
            workflow={workflowData}
            bridges={workflowBridges}
            turns={workflowTurnRanges}
            taskId={taskId}
            onViewTurnsInteraction={() => {
              setActiveTab("turns")
            }}
            onSelectTurn={(turnId) => {
              setWorkflowSelectedTurnId(turnId)
            }}
          />
        </div>

        <div className="w-[320px] border-l flex flex-col min-h-0 overflow-hidden">
          <TurnContextPanel
            selectedTurn={workflowTurn ? { turnId: workflowTurn.turnId, turnIndex: workflowTurn.turnIndex, role: workflowTurn.role } : null}
            rootContext={workflowRootContext}
            subagentContexts={workflowSubagentContexts}
            prevContextPct={workflowPrevPct}
          />
          {workflowTurn && (
            <div className="px-3 pb-3 shrink-0">
              <Button
                variant="outline"
                size="sm"
                className="text-xs w-full"
                onClick={() => {
                  setSelectedTurnId(workflowTurn.turnId)
                  setActiveTab("turns")
                }}
              >
                View in Turns tab →
              </Button>
            </div>
          )}
        </div>
      </div>
    )
  }

  function renderWorkflowAI() {
    const workflowBridges = bridges.map(b => ({
      bridgeId: b.bridgeId,
      dispatchContent: b.dispatchContent,
      dispatchTimestamp: b.dispatchTimestamp,
      responseContent: b.responseContent,
      responseTimestamp: b.responseTimestamp,
      subagentName: b.subagentName,
      subagentType: b.subagentType,
      status: b.status,
      subagentTokens: b.subagentTokens,
      subagentLatencyMs: b.subagentLatencyMs,
    }))
    const workflowTurns = turns.map(t => ({
      turnIndex: t.turnIndex,
      subagentSessionId: t.subagentSessionId,
      isSubagent: t.isSubagent,
      role: t.role,
    }))
    return (
      <WorkflowAIView
        taskId={taskId}
        turnsCount={turns.length}
        bridgesCount={bridges.length}
        bridges={workflowBridges}
        turns={workflowTurns}
        result={workflowAIResult}
        isAnalyzing={workflowAIAnalyzing}
        error={workflowAIError}
        onAnalyze={handleWorkflowAIAnalyze}
        onClearResult={() => {
          setWorkflowAIResult(null)
          setWorkflowAIError(null)
        }}
      />
    )
  }

  async function handleWorkflowAIAnalyze(provider: AIProviderConfig) {
    setWorkflowAIAnalyzing(true)
    setWorkflowAIError(null)
    try {
      const res = await fetch("/api/ai/analyze-workflow", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ taskId, provider }),
      })
      if (!res.ok) {
        const err = await res.json()
        setWorkflowAIError(err.error ?? "Analysis failed")
        return
      }
      const data = await res.json()
      setWorkflowAIResult(data.result)
    } catch (e) {
      setWorkflowAIError(e instanceof Error ? e.message : "Network error")
    } finally {
      setWorkflowAIAnalyzing(false)
    }
  }

  function navigateToTab(tab: string, turnId?: string | null, bridgeId?: string | null) {
    setActiveTab(tab as TabKey)
    if (turnId) {
      setSelectedTurnId(turnId)
      setScrollToTurnId(turnId)
      const turn = turns.find(t => t.turnId === turnId)
      if (turn?.isSubagent) {
        setHighlightSubagentTurnId(turnId)
      } else {
        setHighlightSubagentTurnId(null)
      }
    }
    if (bridgeId) {
      const bridge = bridges.find(b => b.bridgeId === bridgeId)
      if (bridge) {
        setSelectedBridgeId(bridgeId)
      }
    }
  }

  function renderTrace() {
    return (
      <TraceView
        turns={turns}
        bridges={bridges}
        taskId={taskId}
        sessionQuery={s.query}
        navigateToTab={navigateToTab}
      />
    )
  }

  function renderContext() {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <ContextTracker
          turns={turns}
          sessionModel={s.model}
          onNavigateToTurn={(turnId) => {
            setSelectedTurnId(turnId)
            const turn = turns.find(t => t.turnId === turnId)
            if (turn?.isSubagent) {
              setHighlightSubagentTurnId(turnId)
            } else {
              setHighlightSubagentTurnId(null)
            }
            setActiveTab("turns")
          }}
        />
      </div>
    )
  }

  function renderFileReads() {
    return (
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        <FileReadAnalysis
          taskId={taskId}
          onNavigateToTurn={(turnId) => {
            setSelectedTurnId(turnId)
            const turn = turns.find(t => t.turnId === turnId)
            if (turn?.isSubagent) {
              setHighlightSubagentTurnId(turnId)
            } else {
              setHighlightSubagentTurnId(null)
            }
            setActiveTab("turns")
          }}
        />
      </div>
    )
  }

  function renderReplay() {
    const replayTurns = turns.map(t => ({
      turnId: t.turnId,
      turnIndex: t.turnIndex,
      role: t.role,
      contentSummary: t.contentSummary,
      content: null as string | null,
      agentName: t.agentName,
      isSubagent: t.isSubagent,
      subagentName: t.subagentName,
      subagentSessionId: t.subagentSessionId,
      totalTokens: t.totalTokens,
      inputTokens: t.inputTokens,
      inputMessagesCount: t.inputMessagesCount,
      inputMessagesTokens: t.inputMessagesTokens,
      outputTokens: t.outputTokens,
      latencyMs: t.latencyMs,
      createdAt: t.createdAt,
      model: t.model,
      toolCalls: t.toolCalls.map(tc => ({
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        argsJson: tc.argsJson ?? null,
        resultJson: tc.resultJson ?? null,
        state: tc.state,
        durationMs: tc.durationMs,
      })),
      skillEvents: t.skillEvents,
    }))
    return (
      <ChatReplayView
        turns={replayTurns}
        sessionModel={s.model}
        onNavigateToTurn={(turnId) => {
          setSelectedTurnId(turnId)
          const turn = turns.find(t => t.turnId === turnId)
          if (turn?.isSubagent) {
            setHighlightSubagentTurnId(turnId)
          } else {
            setHighlightSubagentTurnId(null)
          }
          setActiveTab("turns")
        }}
      />
    )
  }

  function renderWorkflowAnalyse() {
    return <WorkflowAnalyseTab taskId={taskId} />
  }

  const TAB_RENDERERS: Record<TabKey, () => React.ReactNode> = {
    overview: renderOverview,
    turns: renderTurns,
    workflow: renderWorkflow,
    trace: renderTrace,
    subagents: renderSubagents,
    skills: renderSkills,
    interactions: renderInteractions,
    workflowAI: renderWorkflowAI,
    workflowAnalyse: renderWorkflowAnalyse,
    context: renderContext,
    fileReads: renderFileReads,
    replay: renderReplay,
  }

  return (
    <div className="flex flex-col h-screen">
      <div className="shrink-0 border-b px-4 py-3">
        <div className="flex items-center gap-3 mb-2">
          <Button variant="ghost" size="sm" className="gap-1" onClick={() => window.location.href = "/"}>
            <ArrowLeftIcon className="size-4" />
            Back
          </Button>
          <div className="flex items-center gap-1.5">
            <span className="text-2xl font-bold tracking-tight text-foreground">{BRAND_NAME}</span>
            <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
          </div>
          <h1 className="text-xl font-bold truncate max-w-[400px]">Session: {s.label ?? s.query ?? taskId}</h1>
          <Button
            variant="outline"
            size="sm"
            className="gap-1"
            onClick={handleRefresh}
            disabled={refreshing || !session?.sourcePath}
          >
            <RefreshCwIcon className={`size-4 ${refreshing ? "animate-spin" : ""}`} />
            {refreshing ? "刷新中..." : "刷新"}
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="gap-1 font-semibold"
            onClick={handleExportMd}
            disabled={exportingMd}
          >
            <FileTextIcon className="size-4" />
            {exportingMd ? "导出中..." : "Export MD"}
          </Button>
          <Button
            variant="default"
            size="sm"
            className="gap-1 font-semibold"
            onClick={handleExport}
            disabled={exporting}
          >
            <UploadIcon className="size-4" />
            {exporting ? "导出中..." : "Export DB"}
          </Button>
        </div>
        <div className="flex items-center gap-4 text-sm mb-2">
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">Tool:</span>
            <span className="font-medium">{s.framework === "opencode" ? `OpenCode${s.frameworkVersion ? ` v${s.frameworkVersion}` : ""}` : s.framework === "claude-code" ? `Claude Code${s.frameworkVersion ? ` v${s.frameworkVersion}` : ""}` : s.framework ?? "N/A"}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">Model:</span>
            <span className="font-medium">{s.model ?? "N/A"}</span>
          </div>
          {s.summaryFiles > 0 && (
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground">Code:</span>
              <span className="font-medium text-green-600">+{s.summaryAdditions}</span>
              <span className="font-medium text-red-500">-{s.summaryDeletions}</span>
              <span className="text-muted-foreground">{s.summaryFiles} files</span>
            </div>
          )}
          {s.sourcePath && (
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground">Source:</span>
              <span className="font-medium truncate max-w-[330px]">{s.sourcePath}</span>
            </div>
          )}
          <div className="flex items-center gap-1.5">
            <span className="text-muted-foreground">Start:</span>
            <span>{formatTimestamp(s.startTime)}</span>
          </div>
          {s.endTime && (
            <div className="flex items-center gap-1.5">
              <span className="text-muted-foreground">End:</span>
              <span>{formatTimestamp(s.endTime)}</span>
            </div>
          )}
        </div>

        <div className="flex gap-1 mt-3 border-b -mb-px">
          {TABS.map(tab => (
            <button
              key={tab.key}
              className={cn(
                "px-3 py-1.5 text-sm font-medium transition-colors cursor-pointer border-b-2",
                activeTab === tab.key
                  ? "border-primary text-primary"
                  : tab.highlight
                  ? "border-transparent text-violet-600 dark:text-violet-400 hover:text-violet-700"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
              onClick={() => setActiveTab(tab.key)}
            >
              <span className="flex items-center gap-1">{tab.icon}{tab.label}</span>
            </button>
          ))}
        </div>
      </div>

      {activeTab !== "trace" && (
        <div className="flex-1 min-h-0">
          {TAB_RENDERERS[activeTab]()}
        </div>
      )}
      <div className={cn("flex-1 min-h-0 flex flex-col", activeTab === "trace" ? "" : "hidden")}>
        {renderTrace()}
      </div>
    </div>
  )
}
