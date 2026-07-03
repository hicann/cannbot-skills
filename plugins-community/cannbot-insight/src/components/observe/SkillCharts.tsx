"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState } from "react"
import stringWidth from "string-width"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

function truncateVisual(str: string, maxWidth: number): string {
  if (stringWidth(str) <= maxWidth) return str
  let result = ""
  let width = 0
  for (const char of str) {
    const charW = stringWidth(char)
    if (width + charW > maxWidth - 1) break
    result += char
    width += charW
  }
  return result + "…"
}

interface SkillEventItem {
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
  turnTokens: {
    totalTokens: number
    inputTokens: number
    outputTokens: number
    reasoningTokens: number
    cacheReadTokens: number
    cacheWriteTokens: number
  }
}

interface SkillChartsProps {
  skillEvents: SkillEventItem[]
}

function formatTokens(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

const TOKEN_COLORS = {
  input: { bg: "bg-blue-500", fill: "#3b82f6", label: "Input" },
  output: { bg: "bg-emerald-500", fill: "#10b981", label: "Output" },
  reasoning: { bg: "bg-purple-500", fill: "#8b5cf6", label: "Reasoning" },
  cacheRead: { bg: "bg-yellow-500", fill: "#eab308", label: "Cache Read" },
}

const AGENT_PALETTE = [
  { bg: "bg-blue-500/15", border: "border-blue-400/40", text: "text-blue-700 dark:text-blue-400", dot: "#3b82f6" },
  { bg: "bg-emerald-500/15", border: "border-emerald-400/40", text: "text-emerald-700 dark:text-emerald-400", dot: "#10b981" },
  { bg: "bg-purple-500/15", border: "border-purple-400/40", text: "text-purple-700 dark:text-purple-400", dot: "#8b5cf6" },
  { bg: "bg-orange-500/15", border: "border-orange-400/40", text: "text-orange-700 dark:text-orange-400", dot: "#f97316" },
  { bg: "bg-pink-500/15", border: "border-pink-400/40", text: "text-pink-700 dark:text-pink-400", dot: "#ec4899" },
  { bg: "bg-cyan-500/15", border: "border-cyan-400/40", text: "text-cyan-700 dark:text-cyan-400", dot: "#06b6d4" },
  { bg: "bg-amber-500/15", border: "border-amber-400/40", text: "text-amber-700 dark:text-amber-400", dot: "#f59e0b" },
  { bg: "bg-indigo-500/15", border: "border-indigo-400/40", text: "text-indigo-700 dark:text-indigo-400", dot: "#6366f1" },
]

function TokenUsageBySkillChart({ skillEvents }: SkillChartsProps) {
  const data = useMemo(() => {
    const bySkill = new Map<string, { input: number; output: number; reasoning: number; cacheRead: number; calls: number }>()
    for (const se of skillEvents) {
      if (se.eventType !== "invoke" && se.eventType !== "use") continue
      const existing = bySkill.get(se.skillName) ?? { input: 0, output: 0, reasoning: 0, cacheRead: 0, calls: 0 }
      existing.input += se.turnTokens.inputTokens
      existing.output += se.turnTokens.outputTokens
      existing.reasoning += se.turnTokens.reasoningTokens
      existing.cacheRead += se.turnTokens.cacheReadTokens
      existing.calls++
      bySkill.set(se.skillName, existing)
    }
    return [...bySkill.entries()]
      .sort((a, b) => (b[1].input + b[1].output + b[1].reasoning + b[1].cacheRead) - (a[1].input + a[1].output + a[1].reasoning + a[1].cacheRead))
  }, [skillEvents])

  const totalTokens = useMemo(() => {
    let input = 0, output = 0, reasoning = 0, cacheRead = 0, calls = 0
    for (const se of skillEvents) {
      if (se.eventType !== "invoke" && se.eventType !== "use") continue
      input += se.turnTokens.inputTokens
      output += se.turnTokens.outputTokens
      reasoning += se.turnTokens.reasoningTokens
      cacheRead += se.turnTokens.cacheReadTokens
      calls++
    }
    return { input, output, reasoning, cacheRead, calls }
  }, [skillEvents])

  if (data.length === 0 || totalTokens.calls === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Token Usage by Skill</CardTitle></CardHeader>
        <CardContent><div className="text-sm text-muted-foreground">No invocation data</div></CardContent>
      </Card>
    )
  }

  const maxTotal = Math.max(...data.map(d => d[1].input + d[1].output + d[1].reasoning + d[1].cacheRead))
  const labelWidth = 200
  const barWidth = 180
  const rowHeight = 28

  return (
    <Card size="sm">
      <CardHeader><CardTitle>Token Usage by Skill</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-0.5">
          <div className="flex items-center gap-3 font-semibold border-b pb-1 mb-1" style={{ height: rowHeight }}>
            <span className="text-xs" style={{ width: labelWidth }}>Total ({totalTokens.calls} invocations)</span>
            <div className="flex gap-0 h-5 rounded overflow-hidden bg-muted" style={{ width: barWidth }}>
              {([
                { ...TOKEN_COLORS.input, value: totalTokens.input },
                { ...TOKEN_COLORS.output, value: totalTokens.output },
                { ...TOKEN_COLORS.reasoning, value: totalTokens.reasoning },
                { ...TOKEN_COLORS.cacheRead, value: totalTokens.cacheRead },
              ]).filter(s => s.value > 0).map(s => (
                <div
                  key={s.label}
                  className={cn("h-full", s.bg)}
                  style={{ width: `${(s.value / maxTotal) * 100}%`, minWidth: s.value > 0 ? "1px" : "0" }}
                />
              ))}
            </div>
            <span className="text-xs tabular-nums">{formatTokens(totalTokens.input + totalTokens.output + totalTokens.reasoning + totalTokens.cacheRead)}</span>
            <span className="text-xs tabular-nums text-muted-foreground ml-auto">
              avg {formatTokens(Math.round((totalTokens.input + totalTokens.output + totalTokens.reasoning + totalTokens.cacheRead) / totalTokens.calls))}
            </span>
          </div>
          {data.map(([name, tokens]) => {
            const total = tokens.input + tokens.output + tokens.reasoning + tokens.cacheRead
            const segments = [
              { ...TOKEN_COLORS.input, value: tokens.input },
              { ...TOKEN_COLORS.output, value: tokens.output },
              { ...TOKEN_COLORS.reasoning, value: tokens.reasoning },
              { ...TOKEN_COLORS.cacheRead, value: tokens.cacheRead },
            ].filter(s => s.value > 0)

            return (
              <div key={name} className="flex items-center gap-3" style={{ height: rowHeight }}>
                <span className="text-xs font-medium" style={{ width: labelWidth }}>{truncateVisual(name, 36)}</span>
                <div className="flex gap-0 h-5 rounded overflow-hidden bg-muted" style={{ width: barWidth }}>
                  {segments.map(s => (
                    <div
                      key={s.label}
                      className={cn("h-full", s.bg)}
                      style={{ width: `${(s.value / maxTotal) * 100}%`, minWidth: s.value > 0 ? "1px" : "0" }}
                    />
                  ))}
                </div>
                <span className="text-xs tabular-nums text-muted-foreground">{formatTokens(total)}</span>
                <span className="text-xs tabular-nums text-muted-foreground ml-auto">
                  {tokens.calls} · avg {formatTokens(Math.round(total / tokens.calls))}
                </span>
              </div>
            )
          })}
        </div>
        <div className="flex items-center gap-3 mt-3 pt-2 border-t">
          {Object.values(TOKEN_COLORS).map(c => (
            <div key={c.label} className="flex items-center gap-1.5 text-xs">
              <div className={cn("w-3 h-3 rounded", c.bg)} />
              <span>{c.label}</span>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

function SkillsPerAgentChart({ skillEvents }: SkillChartsProps) {
  const [expandedAgent, setExpandedAgent] = useState<string | null>(null)

  const data = useMemo(() => {
    const byAgent = new Map<string, Map<string, { calls: number; input: number; output: number; reasoning: number; cacheRead: number }>>()
    for (const se of skillEvents) {
      const agent = se.agentName ?? (se.isSubagent ? "subagent" : "root")
      if (!byAgent.has(agent)) byAgent.set(agent, new Map())
      const skillMap = byAgent.get(agent)!
      if (!skillMap.has(se.skillName)) skillMap.set(se.skillName, { calls: 0, input: 0, output: 0, reasoning: 0, cacheRead: 0 })
      const s = skillMap.get(se.skillName)!
      s.calls++
      s.input += se.turnTokens.inputTokens
      s.output += se.turnTokens.outputTokens
      s.reasoning += se.turnTokens.reasoningTokens
      s.cacheRead += se.turnTokens.cacheReadTokens
    }
    const relMap = new Map<string, string[]>()
    for (const se of skillEvents) {
      if (se.isSubagent) {
        const subAgent = se.agentName ?? "subagent"
        const rootSe = skillEvents.find(r => !r.isSubagent && r.turnIndex < se.turnIndex)
        if (rootSe) {
          const root = rootSe.agentName ?? "root"
          if (root !== subAgent) {
            if (!relMap.has(subAgent)) relMap.set(subAgent, [])
            if (!relMap.get(subAgent)!.includes(root)) relMap.get(subAgent)!.push(root)
          }
        }
      }
    }
    return [...byAgent.entries()].map(([name, skills]) => ({
      name,
      skills: [...skills.entries()].map(([skillName, t]) => ({
        name: skillName,
        ...t,
        total: t.input + t.output + t.reasoning + t.cacheRead,
      })),
      calledBy: relMap.get(name) ?? [],
      totalCalls: [...skills.values()].reduce((s, t) => s + t.calls, 0),
      totalTokens: [...skills.values()].reduce((s, t) => s + t.input + t.output + t.reasoning + t.cacheRead, 0),
    }))
  }, [skillEvents])

  if (data.length === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Skills per Agent</CardTitle></CardHeader>
        <CardContent><div className="text-sm text-muted-foreground">No agent data</div></CardContent>
      </Card>
    )
  }

  return (
    <Card size="sm">
      <CardHeader><CardTitle>Skills per Agent</CardTitle></CardHeader>
      <CardContent>
        <div className="space-y-1">
          {data.map((agent, i) => {
            const palette = AGENT_PALETTE[i % AGENT_PALETTE.length]
            const isExpanded = expandedAgent === agent.name

            return (
              <div key={agent.name}>
                <div
                  className={cn("flex items-center gap-2 px-2 py-1.5 rounded-lg border cursor-pointer select-none", palette.bg, palette.border)}
                  onClick={() => setExpandedAgent(isExpanded ? null : agent.name)}
                >
                  <div className={cn("w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold shrink-0", palette.bg, palette.text)} style={{ backgroundColor: palette.dot + "22", color: palette.dot }}>
                    {agent.name.charAt(0).toUpperCase()}
                  </div>
                  <span className={cn("text-xs font-medium", palette.text)}>{truncateVisual(agent.name, 28)}</span>
                  <span className="text-xs tabular-nums text-muted-foreground shrink-0">{agent.totalCalls} calls</span>
                  <span className="text-xs tabular-nums text-muted-foreground shrink-0">{formatTokens(agent.totalTokens)} tok</span>
                  {agent.calledBy.length > 0 && (
                    <span className="text-[10px] text-muted-foreground/60 shrink-0">← {agent.calledBy.join(", ")}</span>
                  )}
                  <span className="ml-auto text-xs text-muted-foreground transition-transform shrink-0" style={{ transform: isExpanded ? "rotate(90deg)" : "none" }}>▸</span>
                </div>
                {isExpanded && (
                  <div className="ml-8 mt-1 space-y-0.5 mb-1">
                    <div className="flex items-center gap-3 text-xs text-muted-foreground px-2 py-0.5 font-medium">
                      <span style={{ minWidth: 160 }}>Skill</span>
                      <span className="tabular-nums text-center" style={{ width: 36 }}>#</span>
                      <span className="tabular-nums text-right" style={{ width: 60 }}>Input</span>
                      <span className="tabular-nums text-right" style={{ width: 60 }}>Output</span>
                      <span className="tabular-nums text-right" style={{ width: 60 }}>Reason</span>
                      <span className="tabular-nums text-right" style={{ width: 60 }}>Cache</span>
                      <span className="tabular-nums text-right" style={{ width: 60 }}>Total</span>
                    </div>
                    {agent.skills.map(skill => (
                      <div key={skill.name} className="flex items-center gap-3 px-2 py-1 rounded">
                        <span className="text-xs" style={{ minWidth: 160 }}>{truncateVisual(skill.name, 36)}</span>
                        <span className="text-xs tabular-nums text-muted-foreground text-center" style={{ width: 36 }}>{skill.calls}</span>
                        <span className="text-xs tabular-nums text-right" style={{ width: 60 }}>{formatTokens(skill.input)}</span>
                        <span className="text-xs tabular-nums text-right" style={{ width: 60 }}>{formatTokens(skill.output)}</span>
                        <span className="text-xs tabular-nums text-right" style={{ width: 60 }}>{formatTokens(skill.reasoning)}</span>
                        <span className="text-xs tabular-nums text-right" style={{ width: 60 }}>{formatTokens(skill.cacheRead)}</span>
                        <span className="text-xs tabular-nums text-right font-medium" style={{ width: 60 }}>{formatTokens(skill.total)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      </CardContent>
    </Card>
  )
}

function FailedSkillsChart({ skillEvents }: SkillChartsProps) {
  const data = useMemo(() => {
    const bySkill = new Map<string, { name: string; count: number; messages: string[] }>()
    for (const se of skillEvents) {
      if (!se.success) {
        const existing = bySkill.get(se.skillName) ?? { name: se.skillName, count: 0, messages: [] }
        existing.count++
        if (se.errorMessage && !existing.messages.includes(se.errorMessage)) {
          existing.messages.push(se.errorMessage)
        }
        bySkill.set(se.skillName, existing)
      }
    }
    return [...bySkill.values()].sort((a, b) => b.count - a.count)
  }, [skillEvents])

  if (data.length === 0) {
    return (
      <Card size="sm">
        <CardHeader><CardTitle>Failed Skills</CardTitle></CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
            <span>✅</span>
            <span>All skill calls succeeded</span>
          </div>
        </CardContent>
      </Card>
    )
  }

  const totalFails = data.reduce((s, d) => s + d.count, 0)

  return (
    <Card size="sm">
      <CardHeader>
        <CardTitle>Failed Skills <Badge variant="red" className="ml-2">{totalFails} errors</Badge></CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {data.map(item => (
            <div key={item.name} className="px-2 py-1.5 rounded-md border bg-red-50/30 dark:bg-red-500/5 border-red-200/30 dark:border-red-600/20">
              <div className="flex items-center gap-2">
                <span className="text-xs font-medium text-red-700 dark:text-red-400">{item.name}</span>
                <Badge variant="red" className="text-xs">{item.count} errors</Badge>
              </div>
              {item.messages.length > 0 && (
                <div className="mt-1 text-xs text-red-600/80 dark:text-red-400/80 space-y-0.5">
                  {item.messages.slice(0, 3).map((msg, i) => (
                    <div key={i} className="truncate">{msg}</div>
                  ))}
                  {item.messages.length > 3 && (
                    <span className="text-muted-foreground">+{item.messages.length - 3} more</span>
                  )}
                </div>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}

export function SkillCharts({ skillEvents }: SkillChartsProps) {
  if (skillEvents.length === 0) {
    return (
      <div className="flex items-center justify-center h-32 text-muted-foreground text-sm">
        No skill event data for charts
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <TokenUsageBySkillChart skillEvents={skillEvents} />
        <SkillsPerAgentChart skillEvents={skillEvents} />
      </div>
      <FailedSkillsChart skillEvents={skillEvents} />
    </div>
  )
}
