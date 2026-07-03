"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface SubagentExecution {
  executionId: string
  agentName: string | null
  agentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  parentExecutionId: string | null
  rootExecutionId: string | null
  depth: number
  tokens: number
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
  dispatchContent: string | null
  dispatchTimestamp: string | null
  responseExecutionId: string | null
  responseContent: string | null
  responseTimestamp: string | null
  subagentSessionId: string | null
  subagentName: string | null
  subagentType: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface SubagentCardsProps {
  subagents: SubagentExecution[]
  bridges: BridgeItem[]
  onViewTurns?: (agentSessionId: string | null) => void
  highlightAgent?: string | null
}

function formatTokenCount(n: number): string {
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
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

function formatTime(ts: string): string {
  try {
    const d = new Date(ts)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
  } catch {
    return ts
  }
}

function computeEndTime(createdAt: string, latencyMs: number): string {
  const start = new Date(createdAt).getTime()
  return new Date(start + latencyMs).toISOString()
}

const STATUS_BADGE: Record<string, "green" | "red" | "orange" | "gray"> = {
  completed: "green",
  failed: "red",
  running: "orange",
  dispatched: "gray",
  timeout: "orange",
}

interface TreeNode {
  agent: SubagentExecution
  children: TreeNode[]
}

function buildTree(subagents: SubagentExecution[]): TreeNode[] {
  const byParentId = new Map<string, TreeNode[]>()
  const nodes: TreeNode[] = []

  for (const sub of subagents) {
    const node: TreeNode = { agent: sub, children: [] }
    nodes.push(node)
    const parentId = sub.parentExecutionId ?? "root"
    if (!byParentId.has(parentId)) byParentId.set(parentId, [])
    byParentId.get(parentId)!.push(node)
  }

  for (const node of nodes) {
    node.children = byParentId.get(node.agent.executionId) ?? []
  }

  const roots = byParentId.get("root") ?? []
  return roots.length > 0 ? roots : nodes
}

function TreeNodeRow({ node, bridges, onViewTurns, top3LatencyIds, top3TokenIds, highlightAgent }: {
  node: TreeNode
  bridges: BridgeItem[]
  onViewTurns?: (agentSessionId: string | null) => void
  top3LatencyIds: Set<string>
  top3TokenIds: Set<string>
  highlightAgent?: string | null
}) {
  const sub = node.agent
  const depth = sub.depth > 0 ? sub.depth : 1
  const matchingBridges = bridges.filter(
    b => b.responseExecutionId === sub.executionId || b.subagentSessionId === sub.agentSessionId
  )
  const bridgeStatus = matchingBridges[0]?.status ?? "dispatched"
  const indentPx = depth * 24
  const isTop3Latency = top3LatencyIds.has(sub.executionId)
  const isTop3Token = top3TokenIds.has(sub.executionId)
  const isHighlighted = highlightAgent != null && (sub.agentName === highlightAgent || sub.subagentName === highlightAgent)
  const highlightClass = isTop3Latency && isTop3Token
    ? "border-l-4 border-l-amber-500 bg-amber-50/30 dark:bg-amber-500/5"
    : isTop3Latency
    ? "border-l-4 border-l-red-500 bg-red-50/30 dark:bg-red-500/5"
    : isTop3Token
    ? "border-l-4 border-l-violet-500 bg-violet-50/30 dark:bg-violet-500/5"
    : ""

  return (
    <>
      <div
        data-agent-id={sub.agentName ?? sub.subagentName ?? undefined}
        className={cn(
          "flex items-start gap-2 py-2 border-b last:border-0",
          highlightClass,
          isHighlighted && "ring-2 ring-blue-500 rounded-sm"
        )}
        style={{ paddingLeft: indentPx }}
      >
        {depth > 1 && (
          <div className="flex items-center self-stretch">
            <div className="w-3 h-full border-l border-muted-foreground/30 ml-0" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <Badge variant="orange">{sub.agentName ?? sub.subagentName ?? "Subagent"}</Badge>
            <Badge variant={STATUS_BADGE[bridgeStatus] ?? "gray"}>{bridgeStatus}</Badge>
            {sub.subagentType && <Badge variant="outline">{sub.subagentType}</Badge>}
            {isTop3Latency && <Badge variant="red">⏱ Top3 Latency</Badge>}
            {isTop3Token && <Badge variant="purple">🔥 Top3 Tokens</Badge>}
          </div>
          <div className="flex items-center gap-3 text-xs">
            <span className="text-muted-foreground">
              {formatTime(sub.createdAt)} → {formatTime(computeEndTime(sub.createdAt, sub.latencyMs))}
            </span>
            <span className={isTop3Token ? "text-violet-600 dark:text-violet-400 font-medium" : "text-muted-foreground"}>
              {formatTokenCount(sub.tokens)} tokens
            </span>
            <span className="text-muted-foreground">{formatCost(sub.cost)}</span>
            <span className={isTop3Latency ? "text-red-600 dark:text-red-400 font-medium" : "text-muted-foreground"}>
              {formatLatency(sub.latencyMs)}
            </span>
            <span className="text-muted-foreground">{sub.toolCallCount} tools</span>
            <span className="text-muted-foreground">{sub.skillLoadCount + sub.skillInvokeCount} skills</span>
            {sub.model && <span className="text-muted-foreground truncate max-w-[160px]">{sub.model}</span>}
          </div>
          {matchingBridges.length > 0 && (
            <div className="text-xs text-muted-foreground mt-1 line-clamp-2">
              {matchingBridges[0].dispatchContent ?? "Dispatched"}
            </div>
          )}
        </div>
        {onViewTurns && (
          <Button variant="ghost" size="sm" className="text-xs shrink-0" onClick={() => onViewTurns(sub.agentSessionId)}>
            View turns
          </Button>
        )}
      </div>
      {node.children.map(child => (
        <TreeNodeRow key={child.agent.executionId} node={child} bridges={bridges} onViewTurns={onViewTurns} top3LatencyIds={top3LatencyIds} top3TokenIds={top3TokenIds} highlightAgent={highlightAgent} />
      ))}
    </>
  )
}

export function SubagentCards({ subagents, bridges, onViewTurns, highlightAgent }: SubagentCardsProps) {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!highlightAgent || !containerRef.current) return
    const timer = setTimeout(() => {
      const el = containerRef.current?.querySelector(`[data-agent-id="${highlightAgent}"]`)
      if (el) {
        el.scrollIntoView({ behavior: "smooth", block: "center" })
      }
    }, 100)
    return () => clearTimeout(timer)
  }, [highlightAgent])

  if (subagents.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No subagent executions found
      </div>
    )
  }

  const top3LatencyIds = new Set(
    [...subagents].sort((a, b) => b.latencyMs - a.latencyMs).slice(0, 3).map(s => s.executionId)
  )
  const top3TokenIds = new Set(
    [...subagents].sort((a, b) => b.tokens - a.tokens).slice(0, 3).map(s => s.executionId)
  )

  const tree = buildTree(subagents)

  return (
    <div className="p-4" ref={containerRef}>
      <div className="flex items-center gap-3 mb-3">
        <Badge variant="outline">Subagents</Badge>
        <span className="text-xs text-muted-foreground">
          {subagents.length} subagent executions
        </span>
        <span className="text-xs text-red-600 dark:text-red-400">⏱ Top3 latency</span>
        <span className="text-xs text-violet-600 dark:text-violet-400">🔥 Top3 tokens</span>
        <span className="text-xs text-amber-600 dark:text-amber-400">⏱🔥 Both</span>
      </div>

      <div className="border rounded-lg px-3">
        {tree.map(node => (
          <TreeNodeRow key={node.agent.executionId} node={node} bridges={bridges} onViewTurns={onViewTurns} top3LatencyIds={top3LatencyIds} top3TokenIds={top3TokenIds} highlightAgent={highlightAgent} />
        ))}
      </div>
    </div>
  )
}
