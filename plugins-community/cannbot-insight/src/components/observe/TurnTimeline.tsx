"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect, useRef } from "react"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import { isCommandTurn, isCommandCaveat, isCommandStdout, parseCommandTurns, formatCommandDisplay, isContinuationTurn, parseContinuationTurn } from "@/lib/shared/command-parser"

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
  dispatchTurnId: string | null
  dispatchContent: string | null
  subagentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  agentName: string | null
  status: string
  subagentTokens: number
  subagentLatencyMs: number
}

interface TurnTimelineProps {
  turns: TurnRowItem[]
  bridges: BridgeItem[]
  selectedTurnId: string | null
  onSelectTurn: (turnId: string) => void
  highlightSubagentTurnId?: string | null
  scrollToTurnId?: string | null
}

const ROLE_COLORS: Record<string, string> = {
  user: "border-l-blue-500 bg-blue-50/50 dark:bg-blue-500/5",
  assistant: "border-l-emerald-500 bg-emerald-50/50 dark:bg-emerald-500/5",
  system: "border-l-gray-400 bg-gray-50/50 dark:bg-gray-500/5",
  tool_result: "border-l-teal-500 bg-teal-50/50 dark:bg-teal-500/5",
  command: "border-l-gray-500 bg-gray-50/50 dark:bg-gray-500/5",
  continuation: "border-l-purple-500 bg-purple-50/50 dark:bg-purple-500/5",
}

const ROLE_ICONS: Record<string, string> = {
  user: "👤",
  assistant: "🤖",
  system: "⚙️",
  tool_result: "🔧",
  command: "⚡",
  continuation: "📋",
}

const ROLE_BADGE_VARIANTS: Record<string, "blue" | "green" | "gray" | "purple" | "orange"> = {
  user: "blue",
  assistant: "green",
  system: "gray",
  tool_result: "purple",
  command: "gray",
  continuation: "purple",
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatTokenCount(n: number): string {
  if (n === 0) return ""
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

interface SubagentLane {
  bridgeId: string
  sessionId: string | null
  name: string
  type: string | null
  summary: string | null
  turns: TurnRowItem[]
  status: string
  totalTokens: number
  latencyMs: number
  turnCount: number
}

export function TurnTimeline({ turns, bridges, selectedTurnId, onSelectTurn, highlightSubagentTurnId, scrollToTurnId }: TurnTimelineProps) {
  const [expandedSubagents, setExpandedSubagents] = useState<Set<string>>(new Set())
  const [filterRole, setFilterRole] = useState<string | null>(null)
  const [searchKeyword, setSearchKeyword] = useState("")
  const containerRef = useRef<HTMLDivElement>(null)
  const selectedRef = useRef<HTMLDivElement>(null)
  const selectedSubRef = useRef<HTMLButtonElement>(null)

  // Expand subagent block when highlightSubagentTurnId is set (cross-tab navigation)
  // highlightSubagentTurnId may be either a turnId or a subagentSessionId
  useEffect(() => {
    if (!highlightSubagentTurnId) return
    let subTurn = turns.find(t => t.turnId === highlightSubagentTurnId)
    if (!subTurn) {
      subTurn = turns.find(t => t.isSubagent && t.subagentSessionId === highlightSubagentTurnId)
    }
    if (subTurn?.isSubagent && subTurn?.subagentSessionId) {
      const matchingBridges = bridges.filter(b => b.subagentSessionId === subTurn.subagentSessionId)
      for (const bridge of matchingBridges) {
        if (!expandedSubagents.has(bridge.bridgeId)) {
          setExpandedSubagents(prev => new Set(prev).add(bridge.bridgeId))
        }
      }
    }
  }, [highlightSubagentTurnId, turns, bridges])

  const dispatchTurnIdForSelected = (() => {
    if (!selectedTurnId) return null
    const selTurn = turns.find(t => t.turnId === selectedTurnId)
    if (selTurn?.isSubagent && selTurn?.subagentSessionId) {
      const bridge = bridges.find(b => b.subagentSessionId === selTurn.subagentSessionId)
      return bridge?.dispatchTurnId ?? null
    }
    return null
  })()

  // Scroll to the targeted turn after expansion + selection are committed
  useEffect(() => {
    if (!scrollToTurnId) return
    const timer = setTimeout(() => {
      const isSub = turns.find(t => t.turnId === scrollToTurnId)?.isSubagent
      const ref = isSub ? selectedSubRef.current : selectedRef.current
      if (ref) ref.scrollIntoView({ behavior: "smooth", block: "nearest" })
    }, 200)
    return () => clearTimeout(timer)
  }, [scrollToTurnId, turns])

  const rootTurns = turns.filter(t => !t.isSubagent)

  // Process command turns: group consecutive command-related turns, parse each group
  const commandInfoMap = new Map<string, { display: string; output: string | null }>()
  let commandGroupTexts: string[] = []
  let commandGroupLeadId: string | null = null

  for (const t of rootTurns) {
    const text = t.contentSummary ?? ""
    const isCmd = isCommandTurn(text)
    const isCav = isCommandCaveat(text)
    const isStd = isCommandStdout(text)

    if (isCmd || isCav || isStd) {
      commandGroupTexts.push(text)
      if (isCmd) commandGroupLeadId = t.turnId
    } else {
      // Flush the group when we hit a non-command turn
      if (commandGroupTexts.length > 0 && commandGroupLeadId) {
        const info = parseCommandTurns(commandGroupTexts)
        commandInfoMap.set(commandGroupLeadId, { display: formatCommandDisplay(info), output: info.output })
      }
      commandGroupTexts = []
      commandGroupLeadId = null
    }
  }
  // Flush any remaining group at end
  if (commandGroupTexts.length > 0 && commandGroupLeadId) {
    const info = parseCommandTurns(commandGroupTexts)
    commandInfoMap.set(commandGroupLeadId, { display: formatCommandDisplay(info), output: info.output })
  }

  // Build display-enhanced root turns: skip caveat/stdout, tag command turns.
  // Then reorder compact groups: opencode assigns the continuation summary a
  // lower turnIndex than the /compact command that produced it, so naive
  // turnIndex order shows the summary BEFORE the command. Swap each adjacent
  // [continuation, /compact command] pair so the command leads its summary.
  const displayRootTurns = (() => {
    const enhanced = rootTurns
      .filter(t => {
        const text = t.contentSummary ?? ""
        // Hide caveat and stdout companion turns
        return !(isCommandCaveat(text) || isCommandStdout(text))
      })
      .map(t => {
        const text = t.contentSummary ?? ""
        if (isCommandTurn(text)) {
          const cmdInfo = commandInfoMap.get(t.turnId)
          return {
            ...t,
            displayRole: "command" as string,
            displayContent: cmdInfo?.display ?? "/unknown",
            commandOutput: cmdInfo?.output ?? null,
            continuationSummary: null as string | null,
          }
        }
        if (isContinuationTurn(text)) {
          const info = parseContinuationTurn(text)
          return {
            ...t,
            displayRole: "continuation" as string,
            displayContent: info.summaryLine ?? "Compact summary",
            commandOutput: null as string | null,
            continuationSummary: info.fullSummary,
          }
        }
        return { ...t, displayRole: t.role, displayContent: t.contentSummary ?? "", commandOutput: null as string | null, continuationSummary: null as string | null }
      })
    const result = [...enhanced]
    for (let i = 0; i < result.length - 1; i++) {
      const cur = result[i]
      const next = result[i + 1]
      if (
        cur.displayRole === "continuation" &&
        next.displayRole === "command" &&
        (next.displayContent.includes("/compact") || next.displayContent === "compact")
      ) {
        [result[i], result[i + 1]] = [result[i + 1], result[i]]
        i++
      }
    }
    return result
  })()

  const subTurnsBySession = new Map<string, TurnRowItem[]>()
  for (const t of turns) {
    if (t.isSubagent && t.subagentSessionId) {
      const arr = subTurnsBySession.get(t.subagentSessionId) ?? []
      arr.push(t)
      subTurnsBySession.set(t.subagentSessionId, arr)
    }
  }

  const bridgesByTurnId = new Map<string, BridgeItem[]>()
  for (const b of bridges) {
    if (b.dispatchTurnId) {
      const arr = bridgesByTurnId.get(b.dispatchTurnId) ?? []
      arr.push(b)
      bridgesByTurnId.set(b.dispatchTurnId, arr)
    }
  }

  const subagentBlocksByTurnId = new Map<string, SubagentLane[]>()
  for (const [turnId, bs] of bridgesByTurnId) {
    const lanes: SubagentLane[] = []
    for (const b of bs) {
      const sid = b.subagentSessionId
      const sturns = sid ? (subTurnsBySession.get(sid) ?? []) : []
      lanes.push({
        bridgeId: b.bridgeId,
        sessionId: sid,
        name: b.agentName ?? b.subagentName ?? b.subagentType ?? "subagent",
        type: b.subagentType,
        summary: b.dispatchContent,
        turns: sturns,
        status: b.status,
        totalTokens: sturns.reduce((s, t) => s + t.totalTokens, 0) + b.subagentTokens,
        latencyMs: b.subagentLatencyMs,
        turnCount: sturns.length,
      })
    }
    subagentBlocksByTurnId.set(turnId, lanes)
  }

  const filteredRootTurns = displayRootTurns.filter(t => {
    if (filterRole && t.displayRole !== filterRole) return false
    if (searchKeyword.trim()) {
      const search = searchKeyword.trim().toLowerCase()
      const summary = t.displayContent.toLowerCase()
      const tools = t.toolCalls.map(tc => tc.toolName.toLowerCase()).join(" ")
      if (!summary.includes(search) && !tools.includes(search)) return false
    }
    return true
  })

  const rootRoles = [...new Set(displayRootTurns.map(t => t.displayRole))]

  return (
    <div className="flex flex-col h-full">
      <div className="shrink-0 px-3 py-2 border-b space-y-1.5">
        <div className="flex flex-wrap gap-1.5 items-center">
          <button
            className={cn(
              "px-2 py-1 rounded text-xs font-medium transition-colors",
              !filterRole ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-accent"
            )}
            onClick={() => setFilterRole(null)}
          >
            All ({displayRootTurns.length})
          </button>
          {rootRoles.map(role => (
            <button
              key={role}
              className={cn(
                "px-2 py-1 rounded text-xs font-medium transition-colors",
                filterRole === role ? "bg-primary text-primary-foreground" : "bg-muted text-muted-foreground hover:bg-accent"
              )}
              onClick={() => setFilterRole(role === filterRole ? null : role)}
            >
              {ROLE_ICONS[role] ?? role} ({displayRootTurns.filter(t => t.displayRole === role).length})
            </button>
          ))}
        </div>

        <Input
          placeholder="Search content or tool name..."
          value={searchKeyword}
          onChange={(e) => setSearchKeyword(e.target.value)}
          className="h-6 text-xs"
        />
      </div>

      <div className="flex-1 overflow-y-auto min-h-0" ref={containerRef}>
        <div className="px-3 py-2 space-y-1.5">
          {filteredRootTurns.map(turn => {
            const borderColor = ROLE_COLORS[turn.displayRole] ?? "border-l-gray-300 bg-gray-50/50"
            const lanes = subagentBlocksByTurnId.get(turn.turnId) ?? []
            const hasTaskCalls = turn.toolCalls.some(tc => tc.toolName === "task")

            return (
              <div key={turn.turnId} ref={selectedTurnId === turn.turnId || turn.turnId === dispatchTurnIdForSelected ? selectedRef : null}>
                <button

                  className={cn(
                    "w-full text-left rounded-lg border-l-3 p-2.5 transition-colors cursor-pointer",
                    borderColor,
                    selectedTurnId === turn.turnId || turn.turnId === dispatchTurnIdForSelected ? "ring-2 ring-primary/50" : "hover:bg-accent/50"
                  )}
                  onClick={() => onSelectTurn(turn.turnId)}
                >
                  <div className="flex items-center gap-1.5 mb-1">
                    <span className="text-xs font-mono text-muted-foreground">#{turn.turnIndex}</span>
                    <Badge variant={ROLE_BADGE_VARIANTS[turn.displayRole] ?? "gray"}>
                      {ROLE_ICONS[turn.displayRole] ?? turn.displayRole} {turn.displayRole}
                    </Badge>
                    {turn.toolCalls.length > 0 && (
                      <Badge variant="outline">{turn.toolCalls.length} tools</Badge>
                    )}
                    {hasTaskCalls && (
                      <Badge variant="orange" className="text-xs">🔗 {turn.toolCalls.filter(tc => tc.toolName === "task").length} subagent</Badge>
                    )}
                    {turn.skillEvents.length > 0 && (
                      <Badge variant="yellow" className="text-xs">
                        {turn.skillEvents.length === 1
                          ? `⚡ ${turn.skillEvents[0].skillName}`
                          : `⚡ ${turn.skillEvents[0].skillName} +${turn.skillEvents.length - 1}`}
                      </Badge>
                    )}
                  </div>

                  {turn.displayContent && (
                    <p className="text-xs text-foreground/80 line-clamp-2 mb-1">
                      {turn.displayRole === "command"
                        ? <span className="font-mono font-medium">{turn.displayContent}</span>
                        : turn.displayRole === "continuation"
                          ? <span className="italic">{turn.displayContent}</span>
                          : turn.displayContent}
                      {turn.commandOutput && (
                        <span className="block opacity-60 mt-0.5">{turn.commandOutput}</span>
                      )}
                    </p>
                  )}

                  <div className="flex items-center gap-3 text-xs text-muted-foreground">
                    {turn.totalTokens > 0 && (
                      <span>{formatTokenCount(turn.totalTokens)} tokens</span>
                    )}
                    {turn.latencyMs > 0 && (
                      <span>{formatLatency(turn.latencyMs)}</span>
                    )}
                    {turn.model && (
                      <span className="truncate">{turn.model}</span>
                    )}
                  </div>
                </button>

                {lanes.length > 0 && (
                  <div className={cn(
                    "ml-4 mt-1",
                    lanes.length === 1 ? "" : "grid gap-1.5",
                    lanes.length === 2 ? "grid-cols-2" : lanes.length === 3 ? "grid-cols-3" : "grid-cols-2"
                  )}>
                    {lanes.map(lane => {
                      const isExpanded = expandedSubagents.has(lane.bridgeId)
                      const isError = lane.status === "error"

                      return (
                        <div key={lane.bridgeId} className={cn(
                          "border rounded-lg",
                          isError ? "border-red-300 bg-red-50/30 dark:bg-red-500/5" : "border-orange-200 bg-orange-50/20 dark:bg-orange-500/5"
                        )}>
                          <button
                            className={cn(
                              "w-full flex items-center gap-1.5 px-2 py-1.5 text-left cursor-pointer hover:bg-accent/30 transition-colors",
                            )}
                            onClick={() => {
                              setExpandedSubagents(prev => {
                                const next = new Set(prev)
                                if (next.has(lane.bridgeId)) next.delete(lane.bridgeId)
                                else next.add(lane.bridgeId)
                                return next
                              })
                            }}
                          >
                            <span className="text-xs select-none">{isExpanded ? "▼" : "▶"}</span>
                            <Badge variant="orange" className="text-xs">{lane.name}</Badge>
                            {lane.summary && (
                              <span className="text-xs text-foreground/80 truncate">{lane.summary}</span>
                            )}
                            {isError && <Badge variant="red" className="text-xs">error</Badge>}
                          </button>

                          <div className="flex items-center gap-2 px-2 pb-1 text-xs text-muted-foreground">
                            <span>{lane.turnCount} turns</span>
                            {lane.totalTokens > 0 && <span>{formatTokenCount(lane.totalTokens)} tok</span>}
                            {lane.latencyMs > 0 && <span>{formatLatency(lane.latencyMs)}</span>}
                          </div>

                          {isExpanded && lane.turns.length > 0 && (
                            <div className="px-2 pb-2 space-y-1">
                              {lane.turns.map(st => (
                                <button
                                  key={st.turnId}
                                  ref={selectedTurnId === st.turnId ? selectedSubRef : null}
                                  className={cn(
                                    "w-full text-left flex items-center gap-1.5 px-2 py-1 rounded border-l-2 text-xs transition-colors cursor-pointer",
                                    "border-l-orange-400 bg-orange-50/30 dark:bg-orange-500/10",
                                    selectedTurnId === st.turnId ? "ring-1 ring-primary/50" : "hover:bg-accent/30"
                                  )}
                                  onClick={() => onSelectTurn(st.turnId)}
                                >
                                  <span className="font-mono text-muted-foreground">#{st.turnIndex}</span>
                                  <Badge variant={ROLE_BADGE_VARIANTS[st.role] ?? "gray"} className="text-xs">
                                    {ROLE_ICONS[st.role]} {st.role}
                                  </Badge>
                                  {st.toolCalls.length > 0 && (
                                    <Badge variant="outline" className="text-xs">{st.toolCalls.length} tools</Badge>
                                  )}
                                  {st.contentSummary && (
                                    <span className="text-foreground/80 truncate max-w-[200px]">
                                      {st.contentSummary.replace(/^<thinking>/, "").substring(0, 40)}
                                    </span>
                                  )}
                                  {st.totalTokens > 0 && (
                                    <span className="text-muted-foreground">{formatTokenCount(st.totalTokens)}</span>
                                  )}
                                </button>
                              ))}
                            </div>
                          )}
                        </div>
                      )
                    })}
                  </div>
                )}
              </div>
            )
          })}

          {filteredRootTurns.length === 0 && displayRootTurns.length > 0 && (
            <div className="text-center text-muted-foreground py-8">
              No turns match the current filters
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
