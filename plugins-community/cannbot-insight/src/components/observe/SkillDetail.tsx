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
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { cn } from "@/lib/utils"
import { auditKindsForEvents, MAIN_AGENT_WORKFLOW_NAME } from "@/lib/skill-eval-audit"
import { SkillCharts } from "@/components/observe/SkillCharts"
import { BookOpenIcon } from "lucide-react"
import { useSkillContent } from "@/components/observe/use-skill-content"
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
  taskId: string
  sessionSkills: SessionSkillItem[]
  skillEvents: SkillEventItem[]
  /** 主 agent workflow 对账目标是否可用（session 首条 user turn 达阈值 → 顶部合成 root 行）。 */
  hasMainAgentWorkflow?: boolean
  /** 主 agent workflow 的真名（扫盘反查的 identifier）；未取/回退时用合成名。 */
  mainAgentWorkflowName?: string | null
  /** 主 agent workflow 合成行的逐栏数据（dispatch 计数 + 主 agent turn token 汇总）。 */
  mainAgentWorkflowStats?: {
    dispatchCount: number
    inputTokens: number
    outputTokens: number
    reasoningTokens: number
    cacheReadTokens: number
    totalTokens: number
  }
  onNavigateToTurn?: (turnIndex: number) => void
  /** 跳转到 Audit 板块的 Skill 子 tab 并预选该 {name, kind} 进行对账。按行 eventType+isSubagent 路由 kind。 */
  onAuditSkill?: (skillName: string, kind: "skill" | "agent" | "root") => void
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

export function SkillDetail({ taskId, sessionSkills, skillEvents, hasMainAgentWorkflow, mainAgentWorkflowName, mainAgentWorkflowStats, onNavigateToTurn, onAuditSkill }: SkillDetailProps) {
  const [expandedSkills, setExpandedSkills] = useState<Set<string>>(new Set())
  const sc = useSkillContent(taskId)
  // 主 agent workflow 显示名：真名（扫盘）→ 回退合成名。对账按钮传此 name（与 SkillAuditTab target 一致）。
  // 全文 fetch 仍用 MAIN_AGENT_WORKFLOW_NAME sentinel（skill-content 路由按 sentinel 识别走 turn0 恢复）。
  const workflowName = mainAgentWorkflowName || MAIN_AGENT_WORKFLOW_NAME

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

      <SkillCharts taskId={taskId} skillEvents={skillEvents} />

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
                  <TableHead className="text-xs w-20 text-right">全文</TableHead>
                  <TableHead className="text-xs">对账</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
              {hasMainAgentWorkflow && onAuditSkill && [
                <TableRow key="__main_agent_workflow__" className="bg-amber-500/5">
                  <TableCell className="text-xs select-none w-6">◆</TableCell>
                  <TableCell className="text-xs font-medium truncate max-w-[20ch]" title={workflowName}>
                    <Badge variant="yellow" className="mr-1 h-4 px-1 text-[10px]">root</Badge>
                    {workflowName}
                  </TableCell>
                  <TableCell className="text-xs">N/A</TableCell>
                  <TableCell className="text-xs tabular-nums" title="主 agent 的 dispatch 次数（编排动作数）">
                    {mainAgentWorkflowStats?.dispatchCount ?? 0}
                  </TableCell>
                  <TableCell><span className="text-muted-foreground">—</span></TableCell>
                  <TableCell><span className="text-muted-foreground">—</span></TableCell>
                  <TableCell className="tabular-nums text-muted-foreground">N/A</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatTokens(mainAgentWorkflowStats?.inputTokens ?? 0)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatTokens(mainAgentWorkflowStats?.outputTokens ?? 0)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatTokens(mainAgentWorkflowStats?.reasoningTokens ?? 0)}</TableCell>
                  <TableCell className="text-xs tabular-nums">{formatTokens(mainAgentWorkflowStats?.cacheReadTokens ?? 0)}</TableCell>
                  <TableCell className="text-xs tabular-nums font-medium">{formatTokens(mainAgentWorkflowStats?.totalTokens ?? 0)}</TableCell>
                  <TableCell className="text-right w-20">
                    <span
                      role="button"
                      title="查看主 agent workflow 声明全文（session 首条 user turn 注入提示）"
                      className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[11px] font-semibold border transition-colors ${
                        sc.content.has(MAIN_AGENT_WORKFLOW_NAME) || sc.error.has(MAIN_AGENT_WORKFLOW_NAME)
                          ? "border-amber-500 bg-amber-500/25 text-amber-700 dark:text-amber-200"
                          : "border-amber-500/50 bg-amber-500/15 text-amber-600 dark:text-amber-300 hover:bg-amber-500/25 hover:border-amber-500"
                      }`}
                      onClick={(e) => {
                        e.stopPropagation()
                        if (sc.content.has(MAIN_AGENT_WORKFLOW_NAME) || sc.error.has(MAIN_AGENT_WORKFLOW_NAME)) {
                          sc.clear(MAIN_AGENT_WORKFLOW_NAME)
                        } else if (!sc.loading.has(MAIN_AGENT_WORKFLOW_NAME)) {
                          sc.fetchOne(MAIN_AGENT_WORKFLOW_NAME)
                        }
                      }}
                    >
                      <BookOpenIcon className="size-3" />
                      {sc.loading.has(MAIN_AGENT_WORKFLOW_NAME) ? "加载中…" : (sc.content.has(MAIN_AGENT_WORKFLOW_NAME) || sc.error.has(MAIN_AGENT_WORKFLOW_NAME)) ? "收起" : "全文"}
                    </span>
                  </TableCell>
                  <TableCell className="text-xs">
                    <Button
                      size="sm"
                      variant="ghost"
                      className="text-xs h-6 px-2"
                      title="跳转到 Audit · Skill 子 tab 用 --kind root 对账主 agent 编排（声明从 session 首条 user turn 恢复）"
                      onClick={(e) => { e.stopPropagation(); onAuditSkill(workflowName, "root") }}
                    >
                      root ↗
                    </Button>
                  </TableCell>
                </TableRow>,
                (sc.content.has(MAIN_AGENT_WORKFLOW_NAME) || sc.error.has(MAIN_AGENT_WORKFLOW_NAME)) && (
                  <TableRow key="__main_agent_workflow__-content">
                    <TableCell colSpan={14} className="p-3 bg-amber-500/5 border-x border-amber-400/30">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <span className="inline-flex items-center gap-1 text-xs font-medium text-amber-600 dark:text-amber-400">
                            <BookOpenIcon className="size-3.5" />
                            {workflowName} · workflow 声明
                            {sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.source && (
                              <span className="text-muted-foreground">
                                （来源：{sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.source === "session-turn0" ? "session 首条 user turn" : sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.source}，{sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.length} 字符）
                              </span>
                            )}
                          </span>
                          <span className="flex items-center gap-3">
                            {sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.content && (
                              <span
                                role="button"
                                className="text-xs font-medium text-blue-500 cursor-pointer hover:underline"
                                onClick={(e) => { e.stopPropagation(); sc.download(MAIN_AGENT_WORKFLOW_NAME) }}
                              >
                                Download
                              </span>
                            )}
                            <span
                              role="button"
                              title="收起"
                              className="text-muted-foreground cursor-pointer hover:text-foreground text-sm leading-none px-1"
                              onClick={(e) => { e.stopPropagation(); sc.clear(MAIN_AGENT_WORKFLOW_NAME) }}
                            >
                              ×
                            </span>
                          </span>
                        </div>
                        <div className="text-[11px] text-muted-foreground leading-snug">
                          主 agent 的 workflow 级 SKILL.md 声明，来自 session 首条 user turn 的注入系统提示（主 agent 通常只 dispatch、不 invoke skill，其 workflow 声明不在 skillEvents）。即 --kind root 对账所用的声明。
                        </div>
                        {sc.error.get(MAIN_AGENT_WORKFLOW_NAME) ? (
                          <div className="text-xs text-destructive">Error: {sc.error.get(MAIN_AGENT_WORKFLOW_NAME)}</div>
                        ) : sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.content ? (
                          <pre className="max-h-[28rem] overflow-auto rounded border bg-background p-2 text-xs font-mono whitespace-pre-wrap break-words">
                            {sc.content.get(MAIN_AGENT_WORKFLOW_NAME)?.content}
                          </pre>
                        ) : (
                          <div className="text-xs text-muted-foreground">此 session 无可对账的主 agent workflow 声明（首条 user turn 过短或缺失）。</div>
                        )}
                      </div>
                    </TableCell>
                  </TableRow>
                ),
              ]}
              {skillAggregates.map(sa => {
                const isExpanded = expandedSkills.has(sa.skillName)
                const rows = [
                  <TableRow key={sa.skillName} className="cursor-pointer hover:bg-accent/30" onClick={() => toggleExpanded(sa.skillName)}>
                    <TableCell className="text-xs select-none w-6">
                      {isExpanded ? "▼" : "▶"}
                    </TableCell>
                    <TableCell className="text-xs font-medium truncate max-w-[20ch]">
                      {sa.skillName}
                    </TableCell>
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
                    <TableCell className="text-right w-20">
                      <span
                        role="button"
                        title="查看 SKILL.md 全文"
                        className={`inline-flex items-center gap-1 px-1.5 py-0.5 rounded-md text-[11px] font-semibold border transition-colors ${
                          sc.content.has(sa.skillName) || sc.error.has(sa.skillName)
                            ? "border-teal-500 bg-teal-500/25 text-teal-700 dark:text-teal-200"
                            : "border-teal-500/50 bg-teal-500/15 text-teal-600 dark:text-teal-300 hover:bg-teal-500/25 hover:border-teal-500"
                        }`}
                        onClick={(e) => {
                          e.stopPropagation()
                          if (sc.content.has(sa.skillName) || sc.error.has(sa.skillName)) {
                            sc.clear(sa.skillName)
                          } else if (!sc.loading.has(sa.skillName)) {
                            sc.fetchOne(sa.skillName)
                          }
                        }}
                      >
                        <BookOpenIcon className="size-3" />
                        {sc.loading.has(sa.skillName) ? "加载中…" : (sc.content.has(sa.skillName) || sc.error.has(sa.skillName)) ? "收起" : "全文"}
                      </span>
                    </TableCell>
                    <TableCell className="text-xs">
                      {onAuditSkill && (() => {
                        const kinds = auditKindsForEvents(sa.events)
                        return kinds.map(k => (
                          <Button
                            key={k}
                            size="sm"
                            variant="ghost"
                            className="text-xs h-6 px-2"
                            title={`跳转到 Audit · Skill 子 tab 对账此 ${k}（${k === "agent" ? ".md 从本地扫描" : "正文从 session 恢复"}）`}
                            onClick={(e) => { e.stopPropagation(); onAuditSkill(sa.skillName, k) }}
                          >
                            {k} ↗
                          </Button>
                        ))
                      })()}
                    </TableCell>
                  </TableRow>,
                ]
                const contentData = sc.content.get(sa.skillName)
                const contentError = sc.error.get(sa.skillName)
                if (contentData || contentError) {
                  rows.push(
                    <TableRow key={`${sa.skillName}-content`}>
                      <TableCell colSpan={14} className="p-3 bg-teal-500/5 border-x border-teal-400/30">
                        <div className="space-y-2">
                          <div className="flex items-center justify-between">
                            <span className="inline-flex items-center gap-1 text-xs font-medium text-teal-600 dark:text-teal-400">
                              <BookOpenIcon className="size-3.5" />
                              {sa.skillName} · SKILL.md
                              {contentData?.source && (
                                <span className="text-muted-foreground">
                                  （来源：{contentData.source === "skill-tool" ? "Skill 工具注入" : "Read 读取"}，{contentData.length} 字符）
                                </span>
                              )}
                            </span>
                            <span className="flex items-center gap-3">
                              {contentData?.content && (
                                <span
                                  role="button"
                                  className="text-xs font-medium text-blue-500 cursor-pointer hover:underline"
                                  onClick={(e) => { e.stopPropagation(); sc.download(sa.skillName) }}
                                >
                                  Download
                                </span>
                              )}
                              <span
                                role="button"
                                title="收起"
                                className="text-muted-foreground cursor-pointer hover:text-foreground text-sm leading-none px-1"
                                onClick={(e) => { e.stopPropagation(); sc.clear(sa.skillName) }}
                              >
                                ×
                              </span>
                            </span>
                          </div>
                          <div className="text-[11px] text-muted-foreground leading-snug">
                            按时间顺序取本会话内该 skill 的注入内容或 SKILL.md 读取结果；若多次加载，取最长版本。仅含已采集内容，非运行时实时文件。
                          </div>
                          {contentError ? (
                            <div className="text-xs text-destructive">Error: {contentError}</div>
                          ) : contentData?.content ? (
                            <pre className="max-h-[28rem] overflow-auto rounded border bg-background p-2 text-xs font-mono whitespace-pre-wrap break-words">
                              {contentData.content}
                            </pre>
                          ) : (
                            <div className="text-xs text-muted-foreground">未采集到该 skill 的 SKILL.md 内容（本会话可能未通过 Skill 工具加载或读取 SKILL.md）。</div>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  )
                }
                if (isExpanded) {
                  rows.push(
                    <TableRow key={`${sa.skillName}-detail`}>
                      <TableCell colSpan={14} className="p-3 bg-muted/20">
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
