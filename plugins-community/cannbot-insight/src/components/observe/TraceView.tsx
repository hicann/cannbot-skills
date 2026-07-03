"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useMemo, useCallback } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"

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
  latencyMs: number
  createdAt: string | null
  completedAt: string | null
  model: string | null
  toolCalls: Array<{ toolCallId: string; toolName: string; state: string; durationMs: number }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
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
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface SearchResultItem {
  turnId: string
  turnIndex: number
  role: string
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  contentSummary: string | null
  matchContext: string
  matchField: "content" | "contentSummary" | "toolResult" | "toolError"
  toolName?: string
  createdAt: string
  hasDispatchBridge: boolean
}

type SourceType = "user_input" | "model_output" | "tool_output" | "root_agent_dispatch" | "subagent_output" | "bridge_response"
type PropagationMedium = "model_reasoning" | "task_dispatch" | "subagent_return" | "tool_output" | "user_input"
type ViewMode = "chain" | "list" | "graph"

const SOURCE_TYPE_CONFIG: Record<SourceType, { icon: string; label: string; badgeVariant: "blue" | "green" | "purple" | "orange" | "gray" | "yellow" }> = {
  user_input: { icon: "👤", label: "用户引入", badgeVariant: "blue" },
  model_output: { icon: "🤖", label: "模型产出", badgeVariant: "green" },
  tool_output: { icon: "🔧", label: "工具输出", badgeVariant: "purple" },
  root_agent_dispatch: { icon: "📤", label: "父agent指令", badgeVariant: "orange" },
  subagent_output: { icon: "🤖", label: "子agent产出", badgeVariant: "green" },
  bridge_response: { icon: "📥", label: "子agent返回", badgeVariant: "gray" },
}

const MEDIUM_CONFIG: Record<PropagationMedium, { icon: string; label: string }> = {
  model_reasoning: { icon: "↓", label: "模型推理" },
  task_dispatch: { icon: "↓", label: "task dispatch" },
  subagent_return: { icon: "↓", label: "subagent返回" },
  tool_output: { icon: "↓", label: "工具输出" },
  user_input: { icon: "↓", label: "用户引入" },
}

const SOURCE_BORDER_COLORS: Record<SourceType, string> = {
  user_input: "border-l-blue-500",
  model_output: "border-l-emerald-500",
  tool_output: "border-l-purple-500",
  root_agent_dispatch: "border-l-orange-500",
  subagent_output: "border-l-emerald-500",
  bridge_response: "border-l-yellow-500",
}

const SOURCE_NODE_COLORS: Record<SourceType, string> = {
  user_input: "#3b82f6",
  model_output: "#10b981",
  tool_output: "#8b5cf6",
  root_agent_dispatch: "#f97316",
  subagent_output: "#10b981",
  bridge_response: "#eab308",
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return ""
  try {
    const d = new Date(ts)
    const h = String(d.getHours()).padStart(2, "0")
    const m = String(d.getMinutes()).padStart(2, "0")
    const s = String(d.getSeconds()).padStart(2, "0")
    return `${h}:${m}:${s}`
  } catch {
    return ts
  }
}

function classifyTurnSource(item: SearchResultItem): SourceType {
  if (item.role === "user") return "user_input"
  if (item.role === "tool_result") return "tool_output"
  if (item.role === "assistant" && item.isSubagent) return "subagent_output"
  if (item.role === "assistant" && item.hasDispatchBridge) return "root_agent_dispatch"
  if (item.role === "assistant") return "model_output"
  return "model_output"
}

function inferMedium(prevItem: SearchResultItem, currItem: SearchResultItem): PropagationMedium {
  const currSource = classifyTurnSource(currItem)
  if (currSource === "user_input") return "user_input"
  if (currSource === "tool_output") return "tool_output"
  if (currSource === "root_agent_dispatch") return "task_dispatch"
  if (currSource === "bridge_response") return "subagent_return"
  return "model_reasoning"
}

function highlightKeyword(text: string, keyword: string): React.ReactNode {
  if (!keyword.trim()) return text
  const lowerText = text.toLowerCase()
  const lowerKeyword = keyword.trim().toLowerCase()
  const parts: Array<{ text: string; isKeyword: boolean }> = []
  let lastIndex = 0
  let idx = lowerText.indexOf(lowerKeyword)
  while (idx !== -1) {
    if (idx > lastIndex) {
      parts.push({ text: text.substring(lastIndex, idx), isKeyword: false })
    }
    parts.push({ text: text.substring(idx, idx + keyword.trim().length), isKeyword: true })
    lastIndex = idx + keyword.trim().length
    idx = lowerText.indexOf(lowerKeyword, lastIndex)
  }
  if (lastIndex < text.length) {
    parts.push({ text: text.substring(lastIndex), isKeyword: false })
  }
  return parts.map((p, i) =>
    p.isKeyword
      ? <mark key={i} className="bg-yellow-200 dark:bg-yellow-500/30 text-foreground rounded px-0.5">{p.text}</mark>
      : p.text
  )
}

interface TraceViewProps {
  turns: TurnRowItem[]
  bridges: BridgeItem[]
  taskId: string
  sessionQuery: string | null
  navigateToTab: (tab: string, turnId?: string | null, bridgeId?: string | null) => void
}

export function TraceView({ turns, bridges, taskId, sessionQuery, navigateToTab }: TraceViewProps) {
  const [keyword, setKeyword] = useState("")
  const [searching, setSearching] = useState(false)
  const [searchResults, setSearchResults] = useState<SearchResultItem[]>([])
  const [lastKeyword, setLastKeyword] = useState("")
  const [viewMode, setViewMode] = useState<ViewMode>("chain")
  const [expandedTurns, setExpandedTurns] = useState<Set<string>>(new Set())
  const [recentSearches, setRecentSearches] = useState<string[]>(() => {
    try {
      const stored = sessionStorage.getItem("trace-recent-searches")
      return stored ? JSON.parse(stored) : []
    } catch {
      return []
    }
  })

  const bridgeByDispatchTurnId = useMemo(() => {
    const map = new Map<string, BridgeItem>()
    for (const b of bridges) {
      if (b.dispatchTurnId) map.set(b.dispatchTurnId, b)
      if (b.responseTurnId) map.set(b.responseTurnId, b)
    }
    return map
  }, [bridges])

  const classifiedResults = useMemo(() => {
    return searchResults.map(item => {
      const source = classifyTurnSource(item)
      const bridge = bridgeByDispatchTurnId.get(item.turnId)
      const bridgeId = bridge?.bridgeId ?? null
      let correctedSource = source
      if (item.role === "assistant" && item.isSubagent) {
        const isResponse = bridges.some(b => b.responseTurnId === item.turnId)
        if (isResponse) correctedSource = "bridge_response"
      }
      return { ...item, sourceType: correctedSource, bridgeId }
    })
  }, [searchResults, bridgeByDispatchTurnId, bridges])

  const sourceDistribution = useMemo(() => {
    const dist: Record<SourceType, number> = {
      user_input: 0, model_output: 0, tool_output: 0,
      root_agent_dispatch: 0, subagent_output: 0, bridge_response: 0,
    }
    for (const item of classifiedResults) {
      dist[item.sourceType]++
    }
    return dist
  }, [classifiedResults])

  const propagationChain = useMemo(() => {
    if (classifiedResults.length === 0) return []
    const chain: Array<{
      item: SearchResultItem & { sourceType: SourceType; bridgeId: string | null }
      medium: PropagationMedium | null
    }> = []
    for (let i = 0; i < classifiedResults.length; i++) {
      const medium = i === 0 ? null : inferMedium(classifiedResults[i - 1], classifiedResults[i])
      chain.push({ item: classifiedResults[i], medium })
    }
    return chain
  }, [classifiedResults, bridges])

  function toggleExpanded(turnId: string) {
    setExpandedTurns(prev => {
      const next = new Set(prev)
      if (next.has(turnId)) next.delete(turnId)
      else next.add(turnId)
      return next
    })
  }

  function addRecentSearch(kw: string) {
    setRecentSearches(prev => {
      const next = [kw, ...prev.filter(s => s !== kw)].slice(0, 8)
      try { sessionStorage.setItem("trace-recent-searches", JSON.stringify(next)) } catch {}
      return next
    })
  }

  const handleSearch = useCallback(async () => {
    const kw = keyword.trim()
    if (!kw) return
    setSearchResults([])
    setSearching(true)
    try {
      const res = await fetch(`/api/observe/session/turns/search?taskId=${encodeURIComponent(taskId)}&keyword=${encodeURIComponent(kw)}`)
      if (res.ok) {
        const data = await res.json()
        setSearchResults(data.items ?? [])
        setLastKeyword(kw)
        addRecentSearch(kw)
      }
    } catch {}
    setSearching(false)
  }, [keyword, taskId])

  const handleRecentClick = useCallback((kw: string) => {
    setKeyword(kw)
    setSearchResults([])
    setSearching(true)
    fetch(`/api/observe/session/turns/search?taskId=${encodeURIComponent(taskId)}&keyword=${encodeURIComponent(kw)}`)
      .then(res => res.ok ? res.json() : { items: [] })
      .then(data => {
        setSearchResults(data.items ?? [])
        setLastKeyword(kw)
        addRecentSearch(kw)
      })
      .catch(() => setSearchResults([]))
      .finally(() => setSearching(false))
  }, [taskId])

  const exampleKeywords = useMemo(() => {
    const kws: string[] = []
    if (sessionQuery) {
      const words = sessionQuery.split(/[\s,，。.、/\\]+/).filter(w => w.length >= 2 && w.length <= 30)
      kws.push(...words.slice(0, 3))
    }
    const toolNames = [...new Set(turns.flatMap(t => t.toolCalls.map(tc => tc.toolName)))]
    kws.push(...toolNames.slice(0, 2))
    return kws.slice(0, 5)
  }, [sessionQuery, turns])

  const hasResults = searchResults.length > 0

  return (
    <div className="flex flex-col h-full">
      <div className="shrink-0 px-3 py-2 border-b space-y-1.5">
        <div className="flex gap-2 items-center">
          <Input
            placeholder="输入关键词搜索..."
            value={keyword}
            onChange={(e) => setKeyword(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") handleSearch() }}
            className="h-7 text-sm"
          />
          <Button size="sm" onClick={handleSearch} disabled={searching || !keyword.trim()} className="h-7 text-xs">
            {searching ? "搜索中..." : "搜索"}
          </Button>
        </div>
        {recentSearches.length > 0 && (
          <div className="flex gap-1.5 items-center overflow-hidden">
            <span className="text-xs text-muted-foreground shrink-0">最近:</span>
            {recentSearches.map(kw => (
              <span
                key={kw}
                role="button"
                tabIndex={0}
                className="text-xs px-1.5 py-0.5 rounded bg-muted cursor-pointer hover:bg-accent transition-colors truncate max-w-[120px]"
                onClick={() => handleRecentClick(kw)}
                onKeyDown={(e) => { if (e.key === "Enter") handleRecentClick(kw) }}
              >
                {kw}
              </span>
            ))}
          </div>
        )}
      </div>

      {!hasResults && !searching && (
        <div className="flex-1 flex items-center justify-center">
          <div className="text-center space-y-3 py-12">
            <p className="text-sm text-muted-foreground">输入关键词追踪概念在对话中的传播路径</p>
            {exampleKeywords.length > 0 && (
              <div className="flex gap-1.5 justify-center">
                {exampleKeywords.map(kw => (
                  <span
                    key={kw}
                    role="button"
                    tabIndex={0}
                    className="text-xs px-2 py-1 rounded border cursor-pointer hover:bg-accent transition-colors"
                    onClick={() => { setKeyword(kw); handleSearch() }}
                    onKeyDown={(e) => { if (e.key === "Enter") { setKeyword(kw); handleSearch() } }}
                  >
                    {kw}
                  </span>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {searching && !hasResults && (
        <div className="flex-1 flex items-center justify-center">
          <p className="text-sm text-muted-foreground">搜索中...</p>
        </div>
      )}

      {hasResults && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
          <div className="shrink-0 px-3 py-2 border-b">
            <div className="flex items-center gap-2 mb-1.5">
              <span className="text-sm font-medium">搜索: &quot;{lastKeyword}&quot;</span>
              <Badge variant="blue">{classifiedResults.length} 命中</Badge>
            </div>
            <div className="flex gap-1.5 items-center mb-1.5">
              {Object.entries(sourceDistribution).filter(([, count]) => count > 0).map(([type, count]) => {
                const cfg = SOURCE_TYPE_CONFIG[type as SourceType]
                return (
                  <Badge key={type} variant={cfg.badgeVariant} className="text-xs">
                    {cfg.icon} {cfg.label}: {count}
                  </Badge>
                )
              })}
            </div>
            <div className="flex gap-1">
              {(["chain", "list", "graph"] as ViewMode[]).map(mode => (
                <Button
                  key={mode}
                  size="sm"
                  variant={viewMode === mode ? "default" : "outline"}
                  className="h-6 text-xs"
                  onClick={() => setViewMode(mode)}
                >
                  {mode === "chain" ? "📜 传播链" : mode === "list" ? "📋 列表" : "🗺️ 传播图"}
                </Button>
              ))}
            </div>
          </div>

          <div className="flex-1 min-h-0 overflow-y-auto px-3 py-2">
            {viewMode === "chain" && <PropagationChainView
              chain={propagationChain}
              keyword={lastKeyword}
              expandedTurns={expandedTurns}
              onToggleExpanded={toggleExpanded}
              onViewTurn={(turnId) => navigateToTab("turns", turnId)}
              onViewBridge={(bridgeId) => navigateToTab("interactions", null, bridgeId)}
            />}
            {viewMode === "list" && <ListView
              results={classifiedResults}
              keyword={lastKeyword}
              onViewTurn={(turnId) => navigateToTab("turns", turnId)}
              onViewBridge={(bridgeId) => navigateToTab("interactions", null, bridgeId)}
            />}
            {viewMode === "graph" && <DAGGraphView
              chain={propagationChain}
              keyword={lastKeyword}
              onViewTurn={(turnId) => navigateToTab("turns", turnId)}
            />}
          </div>
        </div>
      )}
    </div>
  )
}

interface PropagationChainViewItem {
  item: SearchResultItem & { sourceType: SourceType; bridgeId: string | null }
  medium: PropagationMedium | null
}

function PropagationChainView({ chain, keyword, expandedTurns, onToggleExpanded, onViewTurn, onViewBridge }: {
  chain: PropagationChainViewItem[]
  keyword: string
  expandedTurns: Set<string>
  onToggleExpanded: (turnId: string) => void
  onViewTurn: (turnId: string) => void
  onViewBridge: (bridgeId: string | null) => void
}) {
  const origin = chain[0]
  const originCfg = SOURCE_TYPE_CONFIG[origin.item.sourceType]

  return (
    <div className="space-y-0">
      {origin && (
        <div className={cn("rounded-lg border-l-4 p-3 bg-emerald-50/30 dark:bg-emerald-500/5", "border-l-emerald-500")}>
          <div className="flex items-center gap-1.5 mb-1">
            <Badge variant="green" className="text-xs">🟢 起源</Badge>
            <Badge variant={originCfg.badgeVariant} className="text-xs">{originCfg.icon} {originCfg.label}</Badge>
            <span className="text-xs font-mono text-muted-foreground">Turn #{origin.item.turnIndex}</span>
            {origin.item.agentName && (
              <Badge variant="outline" className="text-xs">{origin.item.agentName === "build" ? "root agent" : origin.item.agentName}</Badge>
            )}
          {origin.item.matchField === "toolResult" && <Badge variant="purple" className="text-xs">🔧 工具结果</Badge>}
          {origin.item.matchField === "toolError" && <Badge variant="red" className="text-xs">❌ 工具错误</Badge>}
          <span className="text-xs text-muted-foreground">{formatTimestamp(origin.item.createdAt)}</span>
          </div>
          <p className="text-xs text-foreground/80 line-clamp-2">
            {highlightKeyword(origin.item.matchContext, keyword)}
          </p>
          <div className="flex gap-1.5 mt-1.5">
            <Button size="sm" variant="outline" className="h-5 text-xs" onClick={() => onViewTurn(origin.item.turnId)}>
              View Turn →
            </Button>
            {origin.item.bridgeId && (
              <Button size="sm" variant="outline" className="h-5 text-xs" onClick={() => onViewBridge(origin.item.bridgeId)}>
                View Bridge →
              </Button>
            )}
          </div>
        </div>
      )}

      {chain.slice(1).map(({ item, medium }, i) => {
        const cfg = SOURCE_TYPE_CONFIG[item.sourceType]
        const medCfg = medium ? MEDIUM_CONFIG[medium] : null
        const isExpanded = expandedTurns.has(item.turnId)
        const borderColor = SOURCE_BORDER_COLORS[item.sourceType]

        return (
          <div key={`${item.turnId}-${i}`}>
            {medCfg && (
              <div className="flex items-center gap-2 py-1 ml-4">
                <div className="w-3 h-3 border-l-2 border-b-2 border-muted-foreground/40 rounded-bl-md" />
                <span className="text-xs text-muted-foreground">{medCfg.icon} {medCfg.label}</span>
              </div>
            )}
            <div className={cn("rounded-lg border-l-4 p-2.5 ml-2 cursor-pointer hover:bg-accent/30 transition-colors", borderColor)}
              onClick={() => onToggleExpanded(item.turnId)}
            >
              <div className="flex items-center gap-1.5 mb-1">
                <Badge variant={cfg.badgeVariant} className="text-xs">{cfg.icon} {cfg.label}</Badge>
                <span className="text-xs font-mono text-muted-foreground">Turn #{item.turnIndex}</span>
                {item.agentName && (
                  <Badge variant="outline" className="text-xs">{item.agentName === "build" ? "root agent" : item.agentName}</Badge>
                )}
                {item.isSubagent && item.subagentName && (
                  <Badge variant="orange" className="text-xs">{item.subagentName}</Badge>
                )}
                {item.matchField === "toolResult" && <Badge variant="purple" className="text-xs">🔧 {item.toolName ?? "工具结果"}</Badge>}
                {item.matchField === "toolError" && <Badge variant="red" className="text-xs">❌ {item.toolName ?? "工具错误"}</Badge>}
                <span className="text-xs text-muted-foreground">{formatTimestamp(item.createdAt)}</span>
              </div>

              <p className="text-xs text-foreground/80 line-clamp-2">
                {highlightKeyword(isExpanded ? item.contentSummary ?? item.matchContext : item.matchContext, keyword)}
              </p>

              {isExpanded && (
                <div className="flex gap-1.5 mt-1.5">
                  <Button size="sm" variant="outline" className="h-5 text-xs" onClick={(e) => { e.stopPropagation(); onViewTurn(item.turnId) }}>
                    View Turn →
                  </Button>
                  {item.bridgeId && (
                    <Button size="sm" variant="outline" className="h-5 text-xs" onClick={(e) => { e.stopPropagation(); onViewBridge(item.bridgeId) }}>
                      View Bridge →
                    </Button>
                  )}
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ListView({ results, keyword, onViewTurn, onViewBridge }: {
  results: Array<SearchResultItem & { sourceType: SourceType; bridgeId: string | null }>
  keyword: string
  onViewTurn: (turnId: string) => void
  onViewBridge: (bridgeId: string | null) => void
}) {
  return (
    <div className="space-y-1">
      {results.map(item => {
        const cfg = SOURCE_TYPE_CONFIG[item.sourceType]
        return (
          <div key={item.turnId} className="flex items-center gap-2 px-2 py-1.5 rounded-lg border hover:bg-accent/30 transition-colors">
            <span className="text-xs font-mono text-muted-foreground shrink-0 w-8">#{item.turnIndex}</span>
            <Badge variant={cfg.badgeVariant} className="text-xs shrink-0">{cfg.icon}</Badge>
            <span className="text-xs text-muted-foreground shrink-0">
              {item.agentName === "build" ? "root" : item.agentName ?? "?"}
            </span>
            <span className="text-xs text-muted-foreground shrink-0">{formatTimestamp(item.createdAt)}</span>
            <p className="text-xs text-foreground/80 truncate flex-1 min-w-0">
              {highlightKeyword(item.matchContext, keyword)}
            </p>
            <Button size="sm" variant="outline" className="h-5 text-xs shrink-0" onClick={() => onViewTurn(item.turnId)}>
              →
            </Button>
          </div>
        )
      })}
    </div>
  )
}

interface DAGNode {
  turnIndex: number
  sourceType: SourceType
  agentName: string | null
  subagentName: string | null
  medium: PropagationMedium | null
  turnId: string
}

function DAGGraphView({ chain, keyword, onViewTurn }: {
  chain: PropagationChainViewItem[]
  keyword: string
  onViewTurn: (turnId: string) => void
}) {
  const nodes: DAGNode[] = chain.map(({ item, medium }) => ({
    turnIndex: item.turnIndex,
    sourceType: item.sourceType,
    agentName: item.agentName,
    subagentName: item.subagentName,
    medium,
    turnId: item.turnId,
  }))

  const NODE_RADIUS = 10
  const NODE_SPACING_Y = 56
  const BRANCH_OFFSET_X = 80
  const ORIGIN_RADIUS = 14
  const LABEL_OFFSET_X = 14
  const LEGEND_HEIGHT = 76
  const SVG_WIDTH = Math.max(320, nodes.length > 0 ? 280 + (nodes.some(n => n.sourceType === "root_agent_dispatch" || n.sourceType === "subagent_output" || n.sourceType === "bridge_response") ? BRANCH_OFFSET_X : 0) : 280)
  const SVG_HEIGHT = LEGEND_HEIGHT + nodes.length * NODE_SPACING_Y + 30

  const mainX = SVG_WIDTH / 2 - (nodes.some(n => n.sourceType === "root_agent_dispatch" || n.sourceType === "subagent_output" || n.sourceType === "bridge_response") ? BRANCH_OFFSET_X / 2 : 0)
  const branchX = mainX + BRANCH_OFFSET_X

  function getNodeX(node: DAGNode): number {
    if (node.sourceType === "root_agent_dispatch") return branchX
    if (node.sourceType === "subagent_output" || node.sourceType === "bridge_response") return branchX
    return mainX
  }

  function getNodeY(index: number): number {
    return LEGEND_HEIGHT + 20 + index * NODE_SPACING_Y
  }

  const usedSourceTypes = new Set(nodes.map(n => n.sourceType))
  const hasCrossColumn = nodes.some((n, i) => i > 0 && getNodeX(n) !== getNodeX(nodes[i - 1]))

  const legendSourceTypes = [...usedSourceTypes]
  const legendX = 10
  const legendY0 = 8

  const SOURCE_LEGEND_LABELS: Record<SourceType, string> = {
    user_input: "用户输入",
    model_output: "模型回复",
    tool_output: "工具结果",
    root_agent_dispatch: "父agent调度",
    subagent_output: "子agent产出",
    bridge_response: "子agent返回",
  }

  return (
    <svg
      width={SVG_WIDTH}
      height={SVG_HEIGHT}
      className="w-full max-w-[600px] mx-auto"
      viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`}
    >
      <g>
        <rect x={legendX} y={legendY0} width={SVG_WIDTH - legendX * 2} height={LEGEND_HEIGHT - 4} fill="#f9fafb" stroke="#e5e7eb" strokeWidth={1} rx={6} />
        <text x={legendX + 8} y={legendY0 + 16} fontSize={10} fill="#374151" fontWeight="bold">图例</text>
        <text x={legendX + 34} y={legendY0 + 16} fontSize={9} fill="#6b7280">节点类型:</text>
        {legendSourceTypes.map((st, si) => {
          const cfg = SOURCE_TYPE_CONFIG[st]
          const color = SOURCE_NODE_COLORS[st]
          const cx = legendX + 12 + si * 80
          return (
            <g key={st}>
              <circle cx={cx} cy={legendY0 + 36} r={7} fill={color + "20"} stroke={color} strokeWidth={2} />
              <text x={cx + 9} y={legendY0 + 36} fontSize={9} fill="#6b7280" dominantBaseline="central">{cfg.icon} {SOURCE_LEGEND_LABELS[st]}</text>
            </g>
          )
        })}
        <text x={legendX + 34} y={legendY0 + 58} fontSize={9} fill="#6b7280">连线:</text>
        <g>
          <line x1={legendX + 56} y1={legendY0 + 58} x2={legendX + 56 + 30} y2={legendY0 + 58} stroke="#6b7280" strokeWidth={2} />
          <text x={legendX + 56 + 34} y={legendY0 + 58} fontSize={9} fill="#6b7280" dominantBaseline="central">顺序流转</text>
        </g>
        {hasCrossColumn && (
          <g>
            <line x1={legendX + 130} y1={legendY0 + 58} x2={legendX + 130 + 30} y2={legendY0 + 58} stroke="#f97316" strokeWidth={2} strokeDasharray="4 2" />
            <text x={legendX + 130 + 34} y={legendY0 + 58} fontSize={9} fill="#f97316" dominantBaseline="central">派发子agent</text>
          </g>
        )}
      </g>

      {nodes.map((node, i) => {
        const x = getNodeX(node)
        const y = getNodeY(i)
        const prevNode = i > 0 ? nodes[i - 1] : null
        const prevX = prevNode ? getNodeX(prevNode) : x
        const prevY = prevNode ? getNodeY(i - 1) : y
        const color = SOURCE_NODE_COLORS[node.sourceType]
        const isOrigin = i === 0
        const r = isOrigin ? ORIGIN_RADIUS : NODE_RADIUS
        const isCrossColumn = prevNode && prevX !== x

        return (
          <g key={node.turnId}>
            {prevNode && (
              <line
                x1={prevX}
                y1={prevY + (i - 1 === 0 ? ORIGIN_RADIUS : NODE_RADIUS)}
                x2={x}
                y2={y - r}
                stroke={isCrossColumn ? "#f97316" : "#6b7280"}
                strokeWidth={2}
                strokeDasharray={isCrossColumn ? "4 2" : undefined}
              />
            )}
            <circle
              cx={x}
              cy={y}
              r={r}
              fill={isOrigin ? color : color + "20"}
              stroke={color}
              strokeWidth={isOrigin ? 3 : 2}
              className="cursor-pointer hover:stroke-width-4"
              onClick={() => onViewTurn(node.turnId)}
            />
            <text
              x={x}
              y={y}
              textAnchor="middle"
              dominantBaseline="central"
              fontSize={isOrigin ? 10 : 9}
              fill={isOrigin ? "#fff" : color}
              fontWeight="bold"
            >
              {node.turnIndex}
            </text>
            <text
              x={x + r + LABEL_OFFSET_X}
              y={y}
              textAnchor="start"
              dominantBaseline="central"
              fontSize={9}
              fill="#6b7280"
            >
              {SOURCE_TYPE_CONFIG[node.sourceType].icon}
              {node.agentName === "build" ? "root" : node.agentName ?? ""}
            </text>
            <text
              x={x - r - LABEL_OFFSET_X}
              y={y}
              textAnchor="end"
              dominantBaseline="central"
              fontSize={isOrigin ? 10 : 9}
              fill={isOrigin ? "#10b981" : "#9ca3af"}
              fontWeight={isOrigin ? "bold" : "normal"}
            >
              {isOrigin ? "起源 #" : "#"}{node.turnIndex}
            </text>
            {node.medium && i > 0 && (
              <text
                x={(prevX + x) / 2}
                y={(prevY + NODE_SPACING_Y / 2) + (i - 1 === 0 ? ORIGIN_RADIUS : NODE_RADIUS) - 4}
                textAnchor="middle"
                fontSize={7}
                fill="#9ca3af"
              >
                {MEDIUM_CONFIG[node.medium].label}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
