"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useMemo, useRef, useState } from "react"
import { use } from "react"
import { useSearchParams } from "next/navigation"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { CopyButton } from "@/components/observe/CopyButton"
import { isClaudeFormatSession } from "@/lib/shared/session-format"
import { Button } from "@/components/ui/button"
import { ArrowLeftIcon, LayoutDashboardIcon, MessageSquareIcon, SearchIcon, SparklesIcon, BarChart3Icon, FileSearchIcon, FileTextIcon, PlayIcon, CheckCircleIcon, RefreshCwIcon, WifiIcon, ShieldCheckIcon, GaugeIcon, GlobeIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { toast } from "sonner"
import { VERSION_DISPLAY } from "@/lib/version"
import { BRAND_NAME } from "@/lib/branding"
import type { TurnHighlight } from "@/lib/shared/highlight"
import { TurnTimeline } from "@/components/observe/TurnTimeline"
import { TurnDetail } from "@/components/observe/TurnDetail"
import { WireRounds } from "@/components/observe/WireRounds"
import { TurnContextPanel } from "@/components/observe/TurnContextPanel"
import { SkillDetail } from "@/components/observe/SkillDetail"
import { AuditBoardTab } from "@/components/observe/AuditBoardTab"
import { PerfPanorama } from "@/components/observe/PerfPanorama"
import { TraceView } from "@/components/observe/TraceView"
import { ContextTracker } from "@/components/observe/ContextTracker"
import { FileReadAnalysis } from "@/components/observe/FileReadAnalysis"
import { AgentCallGraph } from "@/components/observe/AgentCallGraph"
import { ChatReplayView } from "@/components/observe/ChatReplayView"
import { summarizeToolCallErrors } from "@/lib/tool-call-errors"
import { dispatchOnlySkillNames } from "@/lib/skill-eval-audit"

type TabKey = "overview" | "turns" | "wireRounds" | "trace" | "skills" | "workflowAnalyse" | "performance" | "context" | "fileReads" | "replay"

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
  systemPrompt?: string | null
  fullContext?: {
    tools: Array<{ name: string; description: string }>
    memoryFiles: string
    skills: string
  } | null
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

const HIDDEN_TABS_DEFAULT = ["replay", "wireRounds"]

const ALL_TABS: Array<{ key: TabKey; label: string; icon: React.ReactNode; highlight?: boolean }> = [
  { key: "overview", label: "Overview", icon: <LayoutDashboardIcon className="size-3.5 text-blue-500" /> },
  { key: "turns", label: "Turns", icon: <MessageSquareIcon className="size-3.5 text-emerald-500" /> },
  { key: "wireRounds", label: "Turns(proxy)", icon: <FileTextIcon className="size-3.5 text-sky-500" /> },
  { key: "trace", label: "Trace", icon: <SearchIcon className="size-3.5 text-yellow-500" /> },
  { key: "context", label: "Context", icon: <BarChart3Icon className="size-3.5 text-pink-500" /> },
  { key: "workflowAnalyse", label: "Audit", icon: <ShieldCheckIcon className="size-3.5 text-emerald-500" />, highlight: true },
  { key: "performance", label: "Perf", icon: <GaugeIcon className="size-3.5 text-amber-500" /> },
  { key: "skills", label: "Skills", icon: <SparklesIcon className="size-3.5 text-orange-500" /> },
  { key: "fileReads", label: "Files", icon: <FileSearchIcon className="size-3.5 text-teal-500" /> },
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

// ─── HTML 导出：客户端 DOM 快照（1:1 复刻已渲染的 insight 界面 + 展开交互） ───

function captureStylesheets(): string {
  const out: string[] = []
  for (let i = 0; i < document.styleSheets.length; i++) {
    const sheet = document.styleSheets[i]
    try {
      const rules = sheet.cssRules
      if (!rules || rules.length === 0) continue
      for (let j = 0; j < rules.length; j++) out.push(rules[j].cssText)
    } catch { /* 跨源样式表跳过 */ }
  }
  return out.join("\n")
}

// 预取会话视图依赖的所有 /api/observe/* 端点，按 path 存入 {path: json} 供 fetch 拦截器返回
// 规范化全 URL 为匹配 key（path + 排序后 query），与 entry.tsx 的 urlKey 一致
function urlKeyOf(u: string): string {
  try {
    const parsed = new URL(u, "http://e")
    const params = [...parsed.searchParams.entries()].sort().map(([k, v]) => `${k}=${v}`).join("&")
    return params ? `${parsed.pathname}?${params}` : parsed.pathname
  } catch {
    return u.split("?")[0]
  }
}

// 收集：核心端点按 path 存 data；skill-content/agent-content 按 全 URL 存 param（per-参数区分）；
// sessionStorage 的 skill 对账结果存 skillAudits（迁移到导出 HTML 供 hydrate）
async function gatherExportData(
  taskId: string,
  framework: string | undefined,
  skills: Array<{ skillName: string }>,
  agents: Array<{ agentName: string | null; isSubagent: boolean }>,
): Promise<{ data: Record<string, unknown>; param: Record<string, unknown>; skillAudits: Record<string, string> }> {
  const fw = framework ? `&framework=${encodeURIComponent(framework)}` : ""
  const data: Record<string, unknown> = {}
  const param: Record<string, unknown> = {}

  // 核心端点 → 按 path 存（一路径一份数据，容忍 query 差异）
  const coreUrls = [
    `/api/observe/session?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/session/turns?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/session/bridges?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/executions?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/session/main-agent-workflow?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/session/workflow?taskId=${encodeURIComponent(taskId)}`,
    `/api/observe/stats?taskId=${encodeURIComponent(taskId)}`,
    `/api/observe/session/file-reads?taskId=${encodeURIComponent(taskId)}`,
    `/api/observe/session/skill-content-audit?taskId=${encodeURIComponent(taskId)}${fw}`,
    `/api/observe/session/llm-workflow-extract?taskId=${encodeURIComponent(taskId)}`,
  ]
  // skill-content（每个 skill 一份，"全文"按钮按需取，预取内嵌）+ agent-content（每个 dispatched 子 agent 一份）
  const paramUrls = [
    ...skills.map(s => `/api/observe/session/skill-content?taskId=${encodeURIComponent(taskId)}&skillName=${encodeURIComponent(s.skillName)}`),
    ...[...new Set(agents.filter(a => a.isSubagent && a.agentName).map(a => a.agentName!))].map(name =>
      `/api/observe/session/agent-content?taskId=${encodeURIComponent(taskId)}&agentName=${encodeURIComponent(name)}&framework=${encodeURIComponent(framework ?? "")}`,
    ),
  ]

  const fetchOne = async (u: string, store: Record<string, unknown>, keyFn: (x: string) => string) => {
    try {
      const res = await fetch(u)
      if (!res.ok) return
      store[keyFn(u)] = await res.json()
    } catch { /* 忽略单个端点失败 */ }
  }
  await Promise.all([
    ...coreUrls.map(u => fetchOne(u, data, x => x.split("?")[0])),
    ...paramUrls.map(u => fetchOne(u, param, urlKeyOf)),
  ])

  // skill/agent 对账结果（sessionStorage 里 skill-audit-${taskId}-${kind}-${name}）
  const skillAudits: Record<string, string> = {}
  const prefix = `skill-audit-${taskId}-`
  try {
    for (let i = 0; i < sessionStorage.length; i++) {
      const k = sessionStorage.key(i) ?? ""
      if (k.startsWith(prefix)) {
        const v = sessionStorage.getItem(k)
        if (!v) continue
        // 剥掉 _html（skill-eval 原始 HTML 报告，体积大且含 </script> 会破坏内联脚本；保留 findings/summary 等结构化结果）
        try {
          const parsed = JSON.parse(v)
          if (parsed && typeof parsed === "object" && "_html" in parsed) delete (parsed as Record<string, unknown>)._html
          skillAudits[k] = JSON.stringify(parsed)
        } catch {
          skillAudits[k] = v
        }
      }
    }
  } catch { /* sessionStorage 不可用 */ }

  return { data, param, skillAudits }
}

// 组装嵌入式 SPA HTML：内联 CSS + 三块数据全局 + bundle + #root，离线打开即跑真实 React 应用
function assembleEmbeddedHtml(
  cssText: string,
  bundleJs: string,
  taskId: string,
  framework: string | undefined,
  payload: { data: Record<string, unknown>; param: Record<string, unknown>; skillAudits: Record<string, string> },
): string {
  const esc = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
  // 鲁棒转义：内联进 <script> 的内容里，任何 </script 序列（大小写不敏感、任意后缀）
  // 都会提前截断 script 标签；把 </script 改成 <\/script（HTML 解析器见 <\ 不视为结束，JS/JSON 里 \/ 即 /）
  const safeForScript = (s: string) => s.replace(/<\/script/gi, "<\\/script")
  const dataJson = safeForScript(JSON.stringify(payload.data))
  const paramJson = safeForScript(JSON.stringify(payload.param))
  const skillAuditsJson = safeForScript(JSON.stringify(payload.skillAudits))
  const bundleSafe = safeForScript(bundleJs)
  return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session ${esc(taskId)} — ${esc(BRAND_NAME)}</title>
<style>
html, body, #root { height: 100%; margin: 0; }
${cssText}
</style>
</head>
<body>
<div id="root"></div>
<script>
window.__EXPORT_TASK_ID = ${JSON.stringify(taskId)};
window.__EXPORT_FRAMEWORK = ${JSON.stringify(framework ?? "")};
window.__EXPORT_DATA = ${dataJson};
window.__EXPORT_PARAM = ${paramJson};
window.__EXPORT_SKILL_AUDITS = ${skillAuditsJson};
</script>
<script>${bundleSafe}</script>
</body>
</html>`
}

function downloadHtml(html: string, filename: string): void {
  const blob = new Blob([html], { type: "text/html;charset=utf-8" })
  const opts = {
    suggestedName: filename,
    types: [{ description: "HTML", accept: { "text/html": [".html"] } }],
  }
  const w = window as unknown as {
    showSaveFilePicker?: (o: typeof opts) => Promise<{ createWritable: () => Promise<{ write: (d: Blob) => Promise<void>; close: () => Promise<void> }> }>
  }
  if (typeof w.showSaveFilePicker === "function") {
    w.showSaveFilePicker(opts).then(async (handle) => {
      const writable = await handle.createWritable()
      await writable.write(blob)
      await writable.close()
    }).catch((e: unknown) => {
      if (e instanceof DOMException && e.name === "AbortError") return
      fallbackDownload(blob, filename)
    })
    return
  }
  fallbackDownload(blob, filename)
}

function fallbackDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const a = document.createElement("a")
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  setTimeout(() => URL.revokeObjectURL(url), 15000)
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
  const [showAllErrorTurns, setShowAllErrorTurns] = useState(false)
  const [scrollToTurnId, setScrollToTurnId] = useState<string | null>(null)
  const [traceHighlight, setTraceHighlight] = useState<TurnHighlight | null>(null)
  const [selectedBridgeId, setSelectedBridgeId] = useState<string | null>(null)
  const [selectedTurnDetail, setSelectedTurnDetail] = useState<TurnDetailData | null>(null)
  const [executions, setExecutions] = useState<ExecutionItem[]>([])
  const [bridges, setBridges] = useState<BridgeItem[]>([])
  const [allSkillEvents, setAllSkillEvents] = useState<SkillEventForDetail[]>([])
  // Audit 板块受控状态：子 tab（workflow|skill）+ Skill 子 tab 选中 {name, kind}。
  // Skills tab 的"对账 ↗"跳转时由 onAuditSkill 一并 set，跨 tab 一气呵成（无需 effect/外部 store）。
  const [auditSub, setAuditSub] = useState<"workflow" | "skill">("workflow")
  const [auditSelected, setAuditSelected] = useState<{ name: string; kind: "skill" | "agent" | "root" | "llm-root" } | null>(null)
  // 主 agent 编排 对账目标是否可用：session 首条 user turn(isSubagent=0)内容达阈值
  // （≥500 字）即注入的 workflow skill 声明。主 agent 通常只 dispatch、不 invoke skill，
  // 其 workflow 声明只能从 turn0 取，故单独 gate（见 audit-skilleval kind=root）。
  const [hasMainAgentWorkflow, setHasMainAgentWorkflow] = useState(false)
  // 主 agent 编排 的真名（扫盘按 turn0 body 匹配 disk SKILL.md 的 frontmatter name）。
  // null=未取/无 workflow；MAIN_AGENT_WORKFLOW_NAME 回退由端点返回。
  const [mainAgentWorkflowName, setMainAgentWorkflowName] = useState<string | null>(null)
  // 主 agent 编排 声明的来源标识（如 "STATE.md" / "PLAN.md" / "scan:work-plan.md"），
  // 来自 main-agent-workflow 端点（recoverPlanFileDeclaration 的 source）。供 Skills tab root 行显示。
  const [mainAgentWorkflowSource, setMainAgentWorkflowSource] = useState<string | null>(null)
  // 主 agent 编排 合成行的逐栏数据：dispatch 计数（编排动作）+ 主 agent turn 的 token 汇总。
  const mainAgentWorkflowStats = useMemo(() => {
    const dispatchCount = allSkillEvents.filter(e => !e.isSubagent && e.eventType === "dispatch").length
    let inputTokens = 0, outputTokens = 0, reasoningTokens = 0, cacheReadTokens = 0, totalTokens = 0
    for (const t of turns) {
      if (t.isSubagent) continue
      inputTokens += t.inputTokens ?? 0
      outputTokens += t.outputTokens ?? 0
      reasoningTokens += t.reasoningTokens ?? 0
      cacheReadTokens += t.cacheReadTokens ?? 0
      totalTokens += t.totalTokens ?? 0
    }
    return { dispatchCount, inputTokens, outputTokens, reasoningTokens, cacheReadTokens, totalTokens }
  }, [allSkillEvents, turns])


  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [exportingMd, setExportingMd] = useState(false)
  const [exportingHtml, setExportingHtml] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const refreshingRef = useRef(false)
  const lastMaxTimeUpdatedRef = useRef<number>(-1)
  const pollInitializedRef = useRef(false)

  async function handleRefresh() {
    if (refreshing || refreshingRef.current) return
    setRefreshing(true)
    refreshingRef.current = true
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
      setLoading(true)
      loadAllData()
    } catch {
      toast.error("刷新失败", { description: "网络错误" })
    } finally {
      setRefreshing(false)
      refreshingRef.current = false
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

  async function handleExportHtml() {
    if (exportingHtml) return
    setExportingHtml(true)
    try {
      // 1. 捕获当前页 CSS（Tailwind v4 + shadcn 变量，覆盖会话视图用到的类）
      const cssText = captureStylesheets()
      // 2. 收集：核心数据 + skill 全文/agent 全文（param）+ skill 对账结果（sessionStorage 迁移）
      const payload = await gatherExportData(taskId, framework, session?.skills ?? [], session?.agents ?? [])
      // 3. 取预构建的单 JS bundle（/public/export-view.js）
      const bundleRes = await fetch("/export-view.js")
      if (!bundleRes.ok) throw new Error("export-view.js 未构建，请先运行 npm run build:export-view")
      const bundleJs = await bundleRes.text()
      // 4. 组装内联 HTML 并下载
      const html = assembleEmbeddedHtml(cssText, bundleJs, taskId, framework, payload)
      downloadHtml(html, `session_${taskId}.html`)
      toast.success("HTML exported", {
        description: `session_${taskId}.html — 嵌入式 SPA（离线可交互）`,
        icon: <CheckCircleIcon className="size-4" />,
        duration: 5000,
      })
    } catch (e) {
      toast.error("Export HTML failed", { description: e instanceof Error ? e.message : "Unknown error" })
    } finally {
      setExportingHtml(false)
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

    setLoading(true)
    setError(null)
    // allSettled + 每个自带 10s 超时：冷编译/WSL2 网络抖动时单个 fetch 卡住
    // 不会永久阻塞整个页面（Promise.all 会等最慢的那个）
    const fetchTimeout = 10000
    const timed = (p: Promise<void>) => Promise.race([
      p,
      new Promise<void>(resolve => setTimeout(resolve, fetchTimeout)),
    ])
    Promise.all([
      timed(fetchSessionData()),
      timed(fetchTurns()),
      timed(fetchExecutions()),
      timed(fetchBridges()),
      timed(fetchSkillEvents()),
    ]).finally(() => setLoading(false))
  }

  useEffect(() => {
    loadAllData()
  }, [taskId])

  // 主 agent 编排 对账目标：从端点取可用性（STATE.md 可恢复）+ 真名（扫盘反查）。
  // STATE.md 是具体工作流（任务清单），root 对账送它；turn0 长度不再是 gate（之前误把角色当工作流）。
  useEffect(() => {
    let cancelled = false
    void (async () => {
      try {
        const r = await fetch(
          `/api/observe/session/main-agent-workflow?taskId=${encodeURIComponent(taskId)}&framework=${encodeURIComponent(framework ?? "")}`,
        )
        if (!r.ok) return
        const d = (await r.json()) as { available?: boolean; name?: string | null; source?: string | null }
        if (!cancelled) {
          setHasMainAgentWorkflow(!!d.available)
          setMainAgentWorkflowName(d.name ?? null)
          setMainAgentWorkflowSource(d.source ?? null)
        }
      } catch {
        /* 取不到不影响主流程 */
      }
    })()
    return () => { cancelled = true }
  }, [taskId, framework])

  useEffect(() => {
    if (!session || session.framework !== "opencode" || !session.sourcePath) return

    const POLL_INTERVAL = 5000
    let active = true

    const poll = async () => {
      if (!active || refreshingRef.current) return
      try {
        const res = await fetch(`/api/observe/auto-refresh?taskId=${encodeURIComponent(taskId)}`)
        if (!res.ok || !active) return
        const data = await res.json()
        if (!pollInitializedRef.current) {
          pollInitializedRef.current = true
          lastMaxTimeUpdatedRef.current = data.maxTimeUpdated ?? -1
          return
        }
        if (!data.settled) return
        const countChanged = data.countChanged === true
        const timeChanged = typeof data.maxTimeUpdated === 'number' && data.maxTimeUpdated !== lastMaxTimeUpdatedRef.current
        const needRefresh = countChanged || timeChanged
        if (needRefresh && active) {
          lastMaxTimeUpdatedRef.current = data.maxTimeUpdated ?? -1
          refreshingRef.current = true
          setRefreshing(true)
          const refreshRes = await fetch("/api/ingest/refresh-session", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ taskId, framework }),
          })
          const refreshData = await refreshRes.json()
          if (refreshRes.ok && active) {
            const added = refreshData.addedTurns ?? 0
            const updated = refreshData.updatedTurns ?? 0
            const msg = added > 0 ? `自动同步: +${added} 轮` : (updated > 0 ? `自动同步: 更新 ${updated} 轮` : "自动同步完成")
            toast.success(msg, { duration: 3000 })
            setLoading(true)
            loadAllData().finally(() => {
              refreshingRef.current = false
              setRefreshing(false)
            })
          } else {
            refreshingRef.current = false
            setRefreshing(false)
          }
        }
      } catch {
        refreshingRef.current = false
        setRefreshing(false)
      }
    }

    const intervalId = setInterval(poll, POLL_INTERVAL)

    return () => {
      active = false
      clearInterval(intervalId)
    }
  }, [taskId, session?.framework, session?.sourcePath, framework])

  useEffect(() => {
    // proxy 捕获是扩展 claude 格式：framework 可能是 opencode（version 带 -proxy）
    if (!session || !isClaudeFormatSession(session.framework, session.frameworkVersion) || !session.sourcePath) return

    let active = true
    let es: EventSource | null = null
    try {
      es = new EventSource(`/api/observe/auto-refresh-stream?taskId=${encodeURIComponent(taskId)}`)
    } catch {
      return
    }

    es.onmessage = async (e) => {
      if (!active) return
      let data: { settled?: boolean; maxTimeUpdated?: number }
      try {
        data = JSON.parse(e.data)
      } catch {
        return
      }
      if (!pollInitializedRef.current) {
        pollInitializedRef.current = true
        lastMaxTimeUpdatedRef.current = data.maxTimeUpdated ?? -1
        return
      }
      if (refreshingRef.current) return
      if (!data.settled) return
      const timeChanged = typeof data.maxTimeUpdated === "number" && data.maxTimeUpdated !== lastMaxTimeUpdatedRef.current
      if (!timeChanged) return
      lastMaxTimeUpdatedRef.current = data.maxTimeUpdated ?? -1
      refreshingRef.current = true
      setRefreshing(true)
      try {
        const refreshRes = await fetch("/api/ingest/refresh-session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ taskId, framework }),
        })
        const refreshData = await refreshRes.json()
        if (refreshRes.ok && active) {
          const added = refreshData.addedTurns ?? 0
          const updated = refreshData.updatedTurns ?? 0
          const msg = added > 0 ? `自动同步: +${added} 轮` : (updated > 0 ? `自动同步: 更新 ${updated} 轮` : "自动同步完成")
          toast.success(msg, { duration: 3000 })
          setLoading(true)
          loadAllData().finally(() => {
            refreshingRef.current = false
            setRefreshing(false)
          })
        } else {
          refreshingRef.current = false
          setRefreshing(false)
        }
      } catch {
        refreshingRef.current = false
        setRefreshing(false)
      }
    }

    es.onerror = () => {
      pollInitializedRef.current = false
    }

    return () => {
      active = false
      es?.close()
    }
  }, [taskId, session?.framework, session?.sourcePath, framework])

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

  // Overview 的 Skills 计数/列表也排除仅 dispatch 的子代理（如 blackbox-designer），
  // 与 Skills 表/图表、Audit 的 agent/skill 划分对齐。
  const dispatchOnlyNames = dispatchOnlySkillNames(allSkillEvents)
  const displaySkills = s ? s.skills.filter(sn => !dispatchOnlyNames.has(sn.skillName)) : []

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
        {/* 会话元信息：Tool/Model/时间/来源（从页头移入，仅 Overview 展示） */}
        <Card size="sm">
          <CardContent className="py-2.5 px-4">
            <div className="flex items-center gap-x-6 text-sm whitespace-nowrap">
              <div className="flex items-center gap-1.5">
                <span className="text-muted-foreground">Tool:</span>
                <span className="font-medium inline-flex items-center gap-1.5">
                  {s.framework === "opencode" ? "OpenCode" : s.framework === "claude-code" ? "Claude Code" : s.framework ?? "N/A"}
                  {(() => {
                    // proxy 捕获的 version = '<agent版本>-<marker>'：版本号与 proxy 徽标都显示
                    const ver = s.frameworkVersion?.replace(/-(claude|opencode)-proxy$/, "")
                    return (
                      <>
                        {ver ? <span className="text-muted-foreground">v{ver}</span> : null}
                        {s.frameworkVersion?.endsWith("-proxy") ? <Badge variant="yellow">proxy</Badge> : null}
                      </>
                    )
                  })()}
                </span>
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
              {s.sourcePath && (
                <div className="flex items-center gap-1.5 min-w-0 flex-1">
                  <span className="text-muted-foreground shrink-0">Source:</span>
                  <span className="font-medium truncate" title={s.sourcePath}>
                    {s.sourcePath}
                  </span>
                  <CopyButton text={s.sourcePath} className="size-3.5 shrink-0 text-muted-foreground hover:text-foreground" />
                </div>
              )}
            </div>
          </CardContent>
        </Card>
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
        </div>

        <Card size="sm">
          <CardHeader>
            <CardTitle>Skills ({displaySkills.length})</CardTitle>
          </CardHeader>
          <CardContent>
            {displaySkills.length === 0 ? (
              <div className="text-sm text-muted-foreground">No skills loaded</div>
            ) : (
              <div className="flex flex-wrap gap-2">
                {displaySkills.map(sn => {
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
            {dispatchOnlyNames.size > 0 && (
              <div className="text-[11px] text-muted-foreground mt-2">
                已排除 {dispatchOnlyNames.size} 个仅分派的子代理（见 Audit · agent 面）
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

        {(() => {
          const errorTurns = turns
            .map(t => ({ turn: t, errors: summarizeToolCallErrors(t.toolCalls, t.skillEvents) }))
            .filter(({ errors }) => errors.total > 0)

          if (errorTurns.length === 0) return <div />

          const visibleErrorTurns = showAllErrorTurns ? errorTurns : errorTurns.slice(0, 3)

          return (
            <Card size="sm" className="border-red-200 dark:border-red-500/30">
              <CardHeader>
                <CardTitle className="text-red-600 dark:text-red-400">⚠ Error Turns ({errorTurns.length})</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-2">
                  {visibleErrorTurns.map(({ turn: t, errors }) => (
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
                {errorTurns.length > 3 && (
                  <button
                    className="w-full mt-1 py-1 rounded-md border border-red-100 dark:border-red-500/20 bg-red-50/20 dark:bg-red-500/5 text-xs text-red-600 dark:text-red-400 hover:bg-red-100/50 dark:hover:bg-red-500/10 transition-colors cursor-pointer"
                    onClick={() => setShowAllErrorTurns(v => !v)}
                  >
                    {showAllErrorTurns ? "收起" : `展开全部 (${errorTurns.length})`}
                  </button>
                )}
              </CardContent>
            </Card>
          )
        })()}
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
      systemPrompt: selectedTurnDetail?.systemPrompt ?? null,
      fullContext: selectedTurnDetail?.fullContext ?? null,
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
        systemPrompt: null,
        fullContext: null,
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
      systemPrompt?: string | null
      fullContext?: {
        tools: Array<{ name: string; description: string }>
        memoryFiles: string
        skills: string
      } | null
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
          systemPrompt: selectedTurnDetail.systemPrompt ?? null,
          fullContext: selectedTurnDetail.fullContext ?? null,
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
              setTraceHighlight(null)
            }}
            onJumpToTurnIndex={(turnIndex) => {
              const turn = turns.find(t => t.turnIndex === turnIndex)
              if (turn) {
                setSelectedTurnId(turn.turnId)
                if (turn.isSubagent) {
                  setHighlightSubagentTurnId(turn.turnId)
                } else {
                  setHighlightSubagentTurnId(null)
                }
                setScrollToTurnId(turn.turnId)
                setTraceHighlight(null)
                setActiveTab("turns")
              }
            }}
          />
        </div>

        <div ref={turnDetailRef} className="flex-1 min-h-0 overflow-y-auto">
          {selectedTurnDetail ? (
            <TurnDetail turn={selectedTurnDetail} highlight={traceHighlight} />
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

  function renderSkills() {
    return (
      <div className="flex-1 min-h-0 overflow-y-auto">
        <SkillDetail
          taskId={taskId}
          sessionSkills={s.skills}
          skillEvents={allSkillEvents}
          hasMainAgentWorkflow={hasMainAgentWorkflow}
          mainAgentWorkflowName={mainAgentWorkflowName}
          mainAgentWorkflowSource={mainAgentWorkflowSource}
          mainAgentWorkflowStats={mainAgentWorkflowStats}
          framework={framework}
          onAuditSkill={(skillName, kind) => {
            setAuditSub("skill")
            setAuditSelected({ name: skillName, kind })
            setActiveTab("workflowAnalyse")
          }}
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

  function navigateToTab(tab: string, turnId?: string | null, bridgeId?: string | null, highlight?: TurnHighlight) {
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
    if (highlight) setTraceHighlight(highlight)
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
    return (
      <AuditBoardTab
        taskId={taskId}
        framework={framework}
        skillEvents={allSkillEvents}
        hasMainAgentWorkflow={hasMainAgentWorkflow}
        mainAgentWorkflowName={mainAgentWorkflowName}
        sub={auditSub}
        onSubChange={setAuditSub}
        skillSelected={auditSelected}
        onSkillSelectedChange={setAuditSelected}
        onJumpToTurn={(turn) => {
          // §N in analysis evidence maps to DB turnIndex (1-based, same as errorTurnParam).
          // Prefer a non-subagent turn so subagent turnIndexes don't shadow main turns.
          const t =
            turns.find(x => x.turnIndex === turn && !x.isSubagent) ??
            turns.find(x => x.turnIndex === turn)
          if (t) navigateToTab("turns", t.turnId)
        }}
      />
    )
  }

  function renderPerformance() {
    const handleTurnIndex = (turn: number) => {
      const t =
        turns.find(x => x.turnIndex === turn && !x.isSubagent) ??
        turns.find(x => x.turnIndex === turn)
      if (t) navigateToTab("turns", t.turnId)
    }
    return (
      <div className="h-full overflow-auto p-4 space-y-4">
        <PerfPanorama taskId={taskId} framework={framework} onJumpToTurn={handleTurnIndex} />
      </div>
    )
  }

  const TAB_RENDERERS: Record<TabKey, () => React.ReactNode> = {
    overview: renderOverview,
    turns: renderTurns,
    wireRounds: () => <WireRounds taskId={taskId ?? ""} />,
    trace: renderTrace,
    skills: renderSkills,
    workflowAnalyse: renderWorkflowAnalyse,
    performance: renderPerformance,
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
          <div className="flex items-center gap-1.5 text-sm">
            {refreshing ? (
              <span className="flex items-center gap-1 text-blue-600 dark:text-blue-400">
                <RefreshCwIcon className="size-3.5 animate-spin" />
                同步中...
              </span>
            ) : (session?.framework === "opencode" || session?.framework === "claude-code") && session?.sourcePath ? (
              <span className="flex items-center gap-1 text-emerald-600 dark:text-emerald-400">
                <WifiIcon className="size-3.5" />
                自动同步
              </span>
            ) : (
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
            )}
            <Button
              variant="ghost"
              size="sm"
              className="gap-1 text-muted-foreground"
              onClick={handleRefresh}
              disabled={refreshing || !session?.sourcePath}
            >
              <RefreshCwIcon className="size-3.5" />
            </Button>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1 text-muted-foreground"
            onClick={handleExportMd}
            disabled={exportingMd}
            title="Export MD"
          >
            <FileTextIcon className="size-4" />
            {exportingMd ? "导出中..." : ""}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1 text-muted-foreground"
            onClick={handleExportHtml}
            disabled={exportingHtml}
            title="Export HTML"
          >
            <GlobeIcon className="size-4" />
            {exportingHtml ? "导出中..." : ""}
          </Button>
        </div>

        <div className="border-b mt-2" />
        <div className="flex gap-1 mt-2 border-b -mb-px">
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

      {activeTab !== "trace" && activeTab !== "performance" && (
        <div className="flex-1 min-h-0" data-export-panel>
          {TAB_RENDERERS[activeTab]()}
        </div>
      )}
      <div className={cn("flex-1 min-h-0 flex flex-col", activeTab === "trace" ? "" : "hidden")} data-export-panel>
        {renderTrace()}
      </div>
      <div className={cn("flex-1 min-h-0 flex flex-col", activeTab === "performance" ? "" : "hidden")} data-export-panel>
        {renderPerformance()}
      </div>
    </div>
  )
}
