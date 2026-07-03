"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { SkillCharts } from "@/components/observe/SkillCharts"
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table"

interface SessionSkillItem {
  skillName: string
  version: number | null
  invocationCount: number
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

interface SkillDetailProps {
  sessionSkills: SessionSkillItem[]
  skillEvents: SkillEventItem[]
  onNavigateToTurn?: (turnIndex: number) => void
}

const EVENT_TYPE_BADGE: Record<string, "blue" | "green" | "orange" | "gray"> = {
  load: "blue",
  invoke: "green",
  use: "green",
  unload: "gray",
}

function formatDuration(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatTokens(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

export function SkillDetail({ sessionSkills, skillEvents, onNavigateToTurn }: SkillDetailProps) {
  const [expandedSkills, setExpandedSkills] = useState<Set<string>>(new Set())

  function toggleExpanded(skillName: string) {
    setExpandedSkills(prev => {
      const next = new Set(prev)
      if (next.has(skillName)) next.delete(skillName)
      else next.add(skillName)
      return next
    })
  }

  const sharedTurnKeys = (() => {
    const m = new Map<string, number>()
    for (const se of skillEvents) {
      const key = `${se.turnIndex}-${se.isSubagent}`
      m.set(key, (m.get(key) ?? 0) + 1)
    }
    return m
  })()

  const skillAggregates = (() => {
    const byName = new Map<string, {
      skillName: string
      version: number | null
      invocationCount: number
      totalEvents: number
      successCount: number
      failCount: number
      avgDuration: number
      totalTokens: number
      inputTokens: number
      outputTokens: number
      reasoningTokens: number
      cacheReadTokens: number
      events: SkillEventItem[]
    }>()

    for (const ss of sessionSkills) {
      const events = skillEvents.filter(se => se.skillName === ss.skillName)
      const invokeEvents = events.filter(se => se.eventType === "invoke" || se.eventType === "use")
      const successCount = events.filter(se => se.success).length
      const failCount = events.filter(se => !se.success).length
      const avgDuration = invokeEvents.length > 0
        ? Math.round(invokeEvents.reduce((sum, e) => sum + e.durationMs, 0) / invokeEvents.length)
        : 0
      const totalTokens = events.reduce((sum, e) => sum + e.turnTokens.totalTokens, 0)
      const inputTokens = events.reduce((sum, e) => sum + e.turnTokens.inputTokens, 0)
      const outputTokens = events.reduce((sum, e) => sum + e.turnTokens.outputTokens, 0)
      const reasoningTokens = events.reduce((sum, e) => sum + e.turnTokens.reasoningTokens, 0)
      const cacheReadTokens = events.reduce((sum, e) => sum + e.turnTokens.cacheReadTokens, 0)

      byName.set(ss.skillName, {
        skillName: ss.skillName,
        version: ss.version ?? events[0]?.skillVersion ?? null,
        invocationCount: ss.invocationCount,
        totalEvents: events.length,
        successCount,
        failCount,
        avgDuration,
        totalTokens,
        inputTokens,
        outputTokens,
        reasoningTokens,
        cacheReadTokens,
        events,
      })
    }

    for (const se of skillEvents) {
      if (byName.has(se.skillName)) continue
      byName.set(se.skillName, {
        skillName: se.skillName,
        version: se.skillVersion ?? null,
        invocationCount: 1,
        totalEvents: 1,
        successCount: se.success ? 1 : 0,
        failCount: se.success ? 0 : 1,
        avgDuration: se.durationMs,
        totalTokens: se.turnTokens.totalTokens,
        inputTokens: se.turnTokens.inputTokens,
        outputTokens: se.turnTokens.outputTokens,
        reasoningTokens: se.turnTokens.reasoningTokens,
        cacheReadTokens: se.turnTokens.cacheReadTokens,
        events: [se],
      })
    }

    return Array.from(byName.values())
  })()

  if (skillAggregates.length === 0 && skillEvents.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-muted-foreground">
        No skill data found
      </div>
    )
  }

  return (
    <div className="p-4 space-y-4">
      <div className="flex items-center gap-3">
        <Badge variant="outline">Skills</Badge>
        <span className="text-sm text-muted-foreground">
          {skillAggregates.length} skills, {skillEvents.length} events
        </span>
      </div>

      <SkillCharts skillEvents={skillEvents} />

      <Card size="sm">
        <CardHeader>
          <CardTitle>Skill Summary</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="text-xs"></TableHead>
                <TableHead className="text-xs">Skill Name</TableHead>
                <TableHead className="text-xs">Version</TableHead>
                <TableHead className="text-xs">Invocations</TableHead>
                <TableHead className="text-xs">Success</TableHead>
                <TableHead className="text-xs">Fail</TableHead>
                <TableHead className="text-xs">Avg Duration</TableHead>
                <TableHead className="text-xs">Input Tok</TableHead>
                <TableHead className="text-xs">Output Tok</TableHead>
                <TableHead className="text-xs">Reason Tok</TableHead>
                <TableHead className="text-xs">Cache Read Tok</TableHead>
                <TableHead className="text-xs">Total Tok</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {skillAggregates.map(sa => {
                const isExpanded = expandedSkills.has(sa.skillName)
                const rows = [
                  <TableRow key={sa.skillName} className="cursor-pointer hover:bg-accent/30" onClick={() => toggleExpanded(sa.skillName)}>
                    <TableCell className="text-xs select-none w-6">
                      {isExpanded ? "▼" : "▶"}
                    </TableCell>
                    <TableCell className="text-xs font-medium">{sa.skillName}</TableCell>
                    <TableCell className="text-xs">{sa.version != null ? `v${sa.version}` : "N/A"}</TableCell>
                    <TableCell className="text-xs tabular-nums">{sa.invocationCount}</TableCell>
                    <TableCell>
                      <Badge variant="green">{sa.successCount}</Badge>
                    </TableCell>
                    <TableCell>
                      {sa.failCount > 0 ? (
                        <Badge variant="red">{sa.failCount}</Badge>
                      ) : (
                        <span className="text-muted-foreground">0</span>
                      )}
                    </TableCell>
                    <TableCell className="tabular-nums">
                      {sa.avgDuration > 0 ? formatDuration(sa.avgDuration) : "N/A"}
                    </TableCell>
                    <TableCell className="text-xs tabular-nums">{formatTokens(sa.inputTokens)}</TableCell>
                    <TableCell className="text-xs tabular-nums">{formatTokens(sa.outputTokens)}</TableCell>
                    <TableCell className="text-xs tabular-nums">{formatTokens(sa.reasoningTokens)}</TableCell>
                    <TableCell className="text-xs tabular-nums">{formatTokens(sa.cacheReadTokens)}</TableCell>
                    <TableCell className="text-xs tabular-nums font-medium">
                      {formatTokens(sa.totalTokens)}
                      {(() => {
                        const hasShared = sa.events.some(se => {
                          const k = `${se.turnIndex}-${se.isSubagent}`
                          return (sharedTurnKeys.get(k) ?? 0) > 1
                        })
                        if (hasShared) return (
                          <Tooltip>
                            <TooltipTrigger render={<Badge variant="outline" className="text-xs ml-1 cursor-help">shared</Badge>} delay={0} closeDelay={0} />
                            <TooltipContent side="top">Token 总和包含被多个 Skill 共享的 Turn 的完整消耗，不精确。同一个 Turn 里的多个 Skill 调用共享该 Turn 的全部 Token，无法单独计算。</TooltipContent>
                          </Tooltip>
                        )
                        return null
                      })()}
                    </TableCell>
                  </TableRow>,
                ]
                if (isExpanded) {
                  rows.push(
                    <TableRow key={`${sa.skillName}-detail`}>
                      <TableCell colSpan={12} className="p-3 bg-muted/20">
                        {sa.events.length === 0 ? (
                          <div className="text-sm text-muted-foreground">No events recorded</div>
                        ) : (
                          <div className="space-y-1.5">
                            {sa.events.map(se => {
                              const turnKey = `${se.turnIndex}-${se.isSubagent}`
                              const isSharedTurn = (sharedTurnKeys.get(turnKey) ?? 0) > 1
                              return (
                                <div key={se.id} className={cn(
                                  "flex items-center gap-2 px-2 py-1.5 border rounded-md text-sm",
                                  se.success ? "bg-emerald-50/30 dark:bg-emerald-500/5" : "bg-red-50/30 dark:bg-red-500/5"
                                )}>
                                  <span className="text-xs text-muted-foreground">Turn {onNavigateToTurn ? <button className="text-blue-600 dark:text-blue-400 hover:underline cursor-pointer" onClick={(e) => { e.stopPropagation(); onNavigateToTurn(se.turnIndex) }}>{se.turnIndex}</button> : se.turnIndex} · Agent: {se.agentName ?? "root"}{se.isSubagent ? " (sub)" : ""}</span>
                                  <Badge variant={EVENT_TYPE_BADGE[se.eventType] ?? "gray"}>
                                    {se.eventType}
                                  </Badge>
                                  <Badge variant={se.success ? "green" : "red"}>
                                    {se.success ? "ok" : "fail"}
                                  </Badge>
                                  {isSharedTurn && (
                                    <Tooltip>
                                      <TooltipTrigger render={<Badge variant="outline" className="text-xs cursor-help">shared</Badge>} delay={0} closeDelay={0} />
                                      <TooltipContent side="top">Token 数为整个 Turn 的消耗，不精确。该 Turn 内还有其他 Skill 调用共享了这些 Token，无法单独计算本 Skill 的精确消耗。</TooltipContent>
                                    </Tooltip>
                                  )}
                                  {se.skillVersion != null && (
                                    <span className="text-xs text-muted-foreground">v{se.skillVersion}</span>
                                  )}
                                  {se.durationMs > 0 && (
                                    <span className="text-xs text-muted-foreground">{formatDuration(se.durationMs)}</span>
                                  )}
                                  {se.turnTokens.totalTokens > 0 && (
                                    <span className="text-xs text-muted-foreground tabular-nums">
                                      {formatTokens(se.turnTokens.totalTokens)} tok ({formatTokens(se.turnTokens.inputTokens)} in / {formatTokens(se.turnTokens.outputTokens)} out / {formatTokens(se.turnTokens.reasoningTokens)} reason / {formatTokens(se.turnTokens.cacheReadTokens)} cache)
                                    </span>
                                  )}
                                  {se.errorMessage && (
                                    <span className="text-xs text-red-600 dark:text-red-400 truncate">{se.errorMessage}</span>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        )}
                      </TableCell>
                    </TableRow>
                  )
                }
                return rows
              })}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  )
}
