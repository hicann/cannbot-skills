"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Audit v4 renderer — agent-centric, three-dimension (completion / efficiency / quality).
// Receives a V4Analysis whose agents[] is a flat list; rebuilds the parent→child tree via parentId.

import { useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export type V4Rating = "pass" | "weak" | "fail" | "n-a"

export interface V4DimRating {
  rating: V4Rating
  note: string
  evidence?: string
  diagnosis?: string
  suggestion?: string
}

export interface V4AgentAction {
  turn: number
  tool: string
  arg: string
  state: "ok" | "error"
  durationMs?: number
  result?: string
}

export interface V4Envelope {
  latencySec: number
  tokensKt: number
  turnCount: number
  toolCallCount: number
  errorCount: number
  retryCount: number
  reasoningTokensKt?: number
}

export interface V4AgentAudit {
  id: string
  parentId?: string | null
  role: "main" | "subagent"
  name: string
  inputSummary: string
  outputSummary: string
  artifacts: string[]
  actions?: V4AgentAction[]
  turns?: number[]
  envelope: V4Envelope
  dimensions: {
    completion: V4DimRating
    efficiency: V4DimRating
    quality: V4DimRating
  }
}

export interface V4Problem {
  type: string
  severity: "high" | "medium" | "low"
  title: string
  detail?: string
  suggestion?: string
}

export interface V4OptimizationPriority {
  priority: number
  target: string
  action: string
  expectedGain: string
}

export interface V4AuditMeta {
  generatedAt: string   // ISO 时间
  elapsedSec: number    // 审计耗时
}

export interface V4Analysis {
  sessionSummary: string
  agents: V4AgentAudit[]
  crossIssues?: V4Problem[]
  optimizationPriorities?: V4OptimizationPriority[]
  _auditMeta?: V4AuditMeta
}

export function isV4Analysis(obj: unknown): obj is V4Analysis {
  return (
    !!obj &&
    typeof obj === "object" &&
    Array.isArray((obj as { agents?: unknown }).agents) &&
    typeof (obj as { sessionSummary?: unknown }).sessionSummary === "string"
  )
}

type BadgeVariant = "green" | "yellow" | "red" | "gray"

function ratingVariant(r: V4Rating): BadgeVariant {
  if (r === "pass") return "green"
  if (r === "weak") return "yellow"
  if (r === "fail") return "red"
  return "gray"
}

function ratingLabel(r: V4Rating): string {
  return { pass: "通过", weak: "瑕疵", fail: "未达标", "n-a": "不适用" }[r]
}

function severityVariant(s: V4Problem["severity"]): BadgeVariant {
  if (s === "high") return "red"
  if (s === "medium") return "yellow"
  return "gray"
}

interface TreeNode {
  agent: V4AgentAudit
  children: TreeNode[]
}

function buildTree(agents: V4AgentAudit[]): TreeNode[] {
  const byParent = new Map<string | null, V4AgentAudit[]>()
  for (const a of agents) {
    const key = a.parentId ?? null
    const arr = byParent.get(key) ?? []
    arr.push(a); byParent.set(key, arr)
  }
  function node(a: V4AgentAudit): TreeNode {
    return { agent: a, children: (byParent.get(a.id) ?? []).map(node) }
  }
  return (byParent.get(null) ?? []).map(node)
}

function DimCell({ label, dim }: { label: string; dim: V4DimRating }) {
  const [open, setOpen] = useState(false)
  // 依据始终可见（审计结论必须可追溯）；诊断/建议折叠
  const hasMore = !!(dim.diagnosis || dim.suggestion)
  return (
    <div className="flex flex-col gap-0.5">
      <div className="flex items-center gap-1 text-xs">
        <span className="text-muted-foreground shrink-0">{label}</span>
        <Badge variant={ratingVariant(dim.rating)} className="text-[10px] px-1 py-0 h-4">
          {ratingLabel(dim.rating)}
        </Badge>
        {hasMore && (
          <button type="button" onClick={() => setOpen(v => !v)} className="text-[10px] text-muted-foreground hover:underline">
            {open ? "▼ 诊断/建议" : "▶ 诊断/建议"}
          </button>
        )}
      </div>
      {dim.evidence && (
        <div className="text-[11px] text-foreground/70 leading-snug">
          <span className="text-muted-foreground font-medium">依据：</span>{dim.evidence}
        </div>
      )}
      {dim.note && <span className="text-[11px] text-muted-foreground line-clamp-2">{dim.note}</span>}
      {open && hasMore && (
        <div className="text-[11px] text-muted-foreground space-y-1 mt-0.5 pl-2 border-l border-border">
          {dim.diagnosis && <div><span className="font-medium text-foreground/70">诊断：</span>{dim.diagnosis}</div>}
          {dim.suggestion && <div><span className="font-medium text-foreground/70">建议：</span>{dim.suggestion}</div>}
        </div>
      )}
    </div>
  )
}

function EnvelopeChips({ env }: { env: V4Envelope }) {
  const items: Array<[string, string, BadgeVariant?]> = [
    ["耗时", `${env.latencySec.toFixed(1)}s`],
    ["token", `${env.tokensKt.toFixed(1)}K`],
    ["turn", String(env.turnCount)],
    ["工具", String(env.toolCallCount)],
    ["错误", String(env.errorCount), env.errorCount > 0 ? "red" : undefined],
    ["重试", String(env.retryCount), env.retryCount > 0 ? "yellow" : undefined],
  ]
  if (env.reasoningTokensKt) items.push(["推理", `${env.reasoningTokensKt.toFixed(1)}K`])
  return (
    <div className="flex flex-wrap gap-1">
      {items.map(([label, val, v]) => (
        <Badge key={label} variant={v ?? "gray"} className="text-[10px] px-1 py-0 h-4 font-mono">
          {label} {val}
        </Badge>
      ))}
    </div>
  )
}

function TurnChips({ turns, onJump }: { turns: number[]; onJump?: (t: number) => void }) {
  const MAX = 14
  const shown = turns.slice(0, MAX)
  const rest = turns.length - shown.length
  return (
    <div className="flex flex-wrap items-center gap-1">
      <span className="text-[10px] text-muted-foreground shrink-0">turns:</span>
      {shown.map(t => (
        <button
          key={t}
          type="button"
          disabled={!onJump}
          onClick={() => onJump?.(t)}
          className={cn(
            "text-[10px] font-mono px-1 rounded border",
            onJump ? "cursor-pointer hover:bg-accent text-foreground/70 border-border" : "text-muted-foreground/60 border-transparent"
          )}
          title={onJump ? `跳转到 turn #${t}` : undefined}
        >
          #{t}
        </button>
      ))}
      {rest > 0 && <span className="text-[10px] text-muted-foreground">+{rest}</span>}
    </div>
  )
}

function AgentCard({ node, depth, onJumpToTurn }: { node: TreeNode; depth: number; onJumpToTurn?: (turn: number) => void }) {
  const { agent, children } = node
  const [expanded, setExpanded] = useState(false)
  const [showActions, setShowActions] = useState(false)
  const hasIo = !!(agent.inputSummary || agent.outputSummary)
  const actions = agent.actions ?? []
  return (
    <div className={cn("rounded-lg border bg-card", depth === 0 ? "border-l-4 border-l-emerald-500" : "border-l-4 border-l-blue-500")}>
      <div className="p-2.5 space-y-1.5">
        <div className="flex flex-wrap items-center gap-1.5">
          <Badge variant={depth === 0 ? "green" : "blue"} className="text-xs">
            {depth === 0 ? "🟢 主 agent" : `🤖 ${agent.name}`}
          </Badge>
          <span className="text-[10px] font-mono text-muted-foreground truncate max-w-[200px]">{agent.id}</span>
          {agent.artifacts.length > 0 && (
            <Badge variant="purple" className="text-[10px] px-1 py-0 h-4">工件 {agent.artifacts.length}</Badge>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2">
          <DimCell label="完成" dim={agent.dimensions.completion} />
          <DimCell label="效率" dim={agent.dimensions.efficiency} />
          <DimCell label="质量" dim={agent.dimensions.quality} />
        </div>

        <EnvelopeChips env={agent.envelope} />

        {agent.turns && agent.turns.length > 0 && (
          <TurnChips turns={agent.turns} onJump={onJumpToTurn} />
        )}

        {hasIo && (
          <div className="mt-1">
            <button
              type="button"
              onClick={() => setExpanded(v => !v)}
              className="text-[11px] text-muted-foreground hover:underline"
            >
              {expanded ? "▼ 收起 I/O" : "▶ 展开 I/O"}
            </button>
            {expanded && (
              <div className="text-[11px] space-y-1 mt-1 pl-2 border-l border-border">
                {agent.inputSummary && (
                  <div><span className="font-medium text-foreground/70">输入：</span><span className="text-muted-foreground whitespace-pre-wrap">{agent.inputSummary.slice(0, 600)}{agent.inputSummary.length > 600 ? "…" : ""}</span></div>
                )}
                {agent.outputSummary && (
                  <div><span className="font-medium text-foreground/70">输出：</span><span className="text-muted-foreground whitespace-pre-wrap">{agent.outputSummary.slice(0, 600)}{agent.outputSummary.length > 600 ? "…" : ""}</span></div>
                )}
                {agent.artifacts.length > 0 && (
                  <div><span className="font-medium text-foreground/70">工件：</span>{agent.artifacts.join("， ")}</div>
                )}
              </div>
            )}
          </div>
        )}

        {actions.length > 0 && (
          <div className="mt-1">
            <button
              type="button"
              onClick={() => setShowActions(v => !v)}
              className="text-[11px] text-muted-foreground hover:underline"
            >
              {showActions ? `▼ 收起动作 (${actions.length})` : `▶ 展开动作 (${actions.length})`}
            </button>
            {showActions && (
              <div className="text-[10px] mt-1 pl-2 border-l border-border space-y-0.5 max-h-64 overflow-y-auto">
                {actions.map((a, i) => (
                  <div key={i} className="flex gap-1.5">
                    <span className="font-mono text-muted-foreground/60 shrink-0">#{a.turn}</span>
                    <span className={cn("font-mono shrink-0", a.state === "error" ? "text-red-600 dark:text-red-400" : "text-foreground/70")}>{a.tool}</span>
                    <span className="text-muted-foreground truncate flex-1">{a.arg}</span>
                    {a.durationMs != null && <span className="text-muted-foreground/50 shrink-0">{(a.durationMs / 1000).toFixed(1)}s</span>}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      {children.length > 0 && (
        <div className="ml-3 mt-1.5 pl-3 border-l-2 border-border space-y-1.5">
          <div className="text-[10px] text-muted-foreground -ml-3 mb-0.5">
            ↳ {agent.name} 派发 {children.length} 个子 agent
          </div>
          {children.map(c => <AgentCard key={c.agent.id} node={c} depth={depth + 1} onJumpToTurn={onJumpToTurn} />)}
        </div>
      )}
    </div>
  )
}

export function WorkflowAgentAudit({ analysis, onJumpToTurn }: { analysis: V4Analysis; onJumpToTurn?: (turn: number) => void }) {
  const tree = useMemo(() => buildTree(analysis.agents), [analysis.agents])
  const summary = useMemo(() => {
    const counts: Record<V4Rating, number> = { pass: 0, weak: 0, fail: 0, "n-a": 0 }
    for (const a of analysis.agents) {
      counts[a.dimensions.completion.rating]++
      counts[a.dimensions.efficiency.rating]++
      counts[a.dimensions.quality.rating]++
    }
    return counts
  }, [analysis.agents])

  return (
    <div className="h-full overflow-auto">
      <div className="sticky top-0 bg-background z-10 border-b px-4 py-2 space-y-1.5">
        <div className="flex items-center justify-between gap-2">
          <p className="text-sm text-foreground/80 flex-1">{analysis.sessionSummary}</p>
          {analysis._auditMeta && (
            <span className="text-[10px] text-muted-foreground font-mono shrink-0">
              审计 {new Date(analysis._auditMeta.generatedAt).toLocaleString("zh-CN", { hour12: false })} · 耗时 {analysis._auditMeta.elapsedSec}s
            </span>
          )}
        </div>
        <div className="flex gap-1.5 flex-wrap">
          {(["pass", "weak", "fail", "n-a"] as V4Rating[]).map(r => (
            <Badge key={r} variant={ratingVariant(r)} className="text-[10px]">
              {ratingLabel(r)} {summary[r]}
            </Badge>
          ))}
          <Badge variant="blue" className="text-[10px]">agents {analysis.agents.length}</Badge>
        </div>
      </div>

      <div className="p-3 space-y-2">
        {tree.map(n => <AgentCard key={n.agent.id} node={n} depth={0} onJumpToTurn={onJumpToTurn} />)}
      </div>

      {analysis.crossIssues && analysis.crossIssues.length > 0 && (
        <div className="p-3 pt-0 space-y-1.5">
          <h4 className="text-xs font-semibold text-muted-foreground">跨 agent 问题</h4>
          {analysis.crossIssues.map((p, i) => (
            <div key={i} className="rounded border p-2 text-xs space-y-0.5">
              <div className="flex items-center gap-1.5">
                <Badge variant={severityVariant(p.severity)} className="text-[10px]">{p.severity}</Badge>
                <span className="font-medium">{p.title}</span>
                <span className="text-muted-foreground text-[10px]">{p.type}</span>
              </div>
              {p.detail && <p className="text-muted-foreground">{p.detail}</p>}
              {p.suggestion && <p className="text-muted-foreground"><span className="font-medium text-foreground/70">建议：</span>{p.suggestion}</p>}
            </div>
          ))}
        </div>
      )}

      {analysis.optimizationPriorities && analysis.optimizationPriorities.length > 0 && (
        <div className="p-3 pt-0 space-y-1.5">
          <h4 className="text-xs font-semibold text-muted-foreground">优化优先级</h4>
          {analysis.optimizationPriorities.map((p, i) => (
            <div key={i} className="rounded border p-2 text-xs space-y-0.5">
              <div className="flex items-center gap-1.5">
                <Badge variant="orange" className="text-[10px]">P{p.priority}</Badge>
                <span className="font-medium font-mono">{p.target}</span>
              </div>
              <p className="text-muted-foreground">{p.action}</p>
              <p className="text-emerald-600 dark:text-emerald-400 text-[11px]">预期收益：{p.expectedGain}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
