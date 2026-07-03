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
import { cn } from "@/lib/utils"

interface ToolCallEntry {
  name: string
  args: string | null
  result: string | null
}

interface InputMessage {
  role: string
  content: string | null
  tokenCount?: number
  name?: string
  tool_calls?: ToolCallEntry[]
}

interface ContextInfo {
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
}

interface TurnContextPanelProps {
  selectedTurn: {
    turnId: string
    turnIndex: number
    role: string
  } | null
  rootContext: ContextInfo | null
  subagentContexts: ContextInfo[]
  prevContextPct: number | null
}

const ROLE_COLORS: Record<string, string> = {
  system: "bg-purple-500",
  user: "bg-blue-500",
  assistant: "bg-emerald-500",
  tool_result: "bg-teal-500",
  tool: "bg-teal-500",
}

const ROLE_TEXT_COLORS: Record<string, string> = {
  system: "text-purple-600 dark:text-purple-400",
  user: "text-blue-600 dark:text-blue-400",
  assistant: "text-emerald-600 dark:text-emerald-400",
  tool_result: "text-teal-600 dark:text-teal-400",
  tool: "text-teal-600 dark:text-teal-400",
}

const ROLE_ICONS: Record<string, string> = {
  system: "⚙️",
  user: "👤",
  assistant: "🤖",
  tool_result: "🔧",
  tool: "🔧",
}

const ROLE_BADGE_VARIANTS: Record<string, "blue" | "green" | "gray" | "purple" | "orange"> = {
  system: "purple",
  user: "blue",
  assistant: "green",
  tool_result: "gray",
  tool: "gray",
}

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

function parseInputMessages(json: string | null): InputMessage[] {
  if (!json) return []
  try {
    const parsed = JSON.parse(json)
    if (Array.isArray(parsed)) return parsed
    return []
  } catch {
    return []
  }
}

function classifySkillHeader(msg: InputMessage): { isSkillHeader: boolean; skillName: string | null } {
  if (msg.role !== "system") return { isSkillHeader: false, skillName: null }
  const content = msg.content ?? ""
  // Skill headers typically look like: "You have the following skills available: ..." or contain skill names
  if (content.includes("skill") && (content.includes("available") || content.includes("loaded") || content.includes("invoke"))) {
    // Try to extract skill name
    const skillMatch = content.match(/(\w+-\w+-\w+|\w+\/\w+|ops-\w+-\w+)/)
    return { isSkillHeader: true, skillName: skillMatch?.[1] ?? null }
  }
  return { isSkillHeader: false, skillName: null }
}

function truncate(text: string | null, maxLen: number): string | null {
  if (!text) return null
  if (text.length <= maxLen) return text
  return text.substring(0, maxLen) + "..."
}

interface MessageCategory {
  role: string
  label: string
  icon: string
  color: string
  textColor: string
  messages: InputMessage[]
  totalTokens: number
  skillHeaders: Array<{ msg: InputMessage; skillName: string | null }>
}

const ROLE_LABELS: Record<string, string> = {
  system: "System / Skills",
  user: "User",
  assistant: "Assistant",
  tool_result: "Tool Results",
  tool: "Tool Calls",
}

function categorizeMessages(messages: InputMessage[]): MessageCategory[] {
  // Group consecutive same-role messages, preserving interleaved order
  const categories: MessageCategory[] = []

  for (const msg of messages) {
    const cls = classifySkillHeader(msg)
    const isSkill = cls.isSkillHeader && msg.role === "system"
    const last = categories[categories.length - 1]

    if (last && last.role === msg.role) {
      last.messages.push(msg)
      last.totalTokens += msg.tokenCount ?? 0
      if (isSkill) last.skillHeaders.push({ msg, skillName: cls.skillName })
    } else {
      categories.push({
        role: msg.role,
        label: ROLE_LABELS[msg.role] ?? msg.role,
        icon: ROLE_ICONS[msg.role] ?? "?",
        color: ROLE_COLORS[msg.role] ?? "bg-gray-400",
        textColor: ROLE_TEXT_COLORS[msg.role] ?? "text-gray-600",
        messages: [msg],
        totalTokens: msg.tokenCount ?? 0,
        skillHeaders: isSkill ? [{ msg, skillName: cls.skillName }] : [],
      })
    }
  }

  return categories
}

function ContextSection({ info, prevPct }: { info: ContextInfo; prevPct: number | null }) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [expandedCategories, setExpandedCategories] = useState<Set<string>>(new Set())
  const [expandedMsgs, setExpandedMsgs] = useState<Set<string>>(new Set())

  const pctValue = info.contextWindowPct ?? 0
  const endPctValue = info.endContextWindowPct ?? pctValue
  const maxPct = Math.max(pctValue, endPctValue)
  const pctColor =
    maxPct > 80 ? "bg-red-500" :
    maxPct > 50 ? "bg-orange-500" :
    maxPct > 0 ? "bg-blue-500" :
    "bg-gray-300"

  const pctLabel =
    maxPct > 80 ? "text-red-600 dark:text-red-400" :
    maxPct > 50 ? "text-orange-600 dark:text-orange-400" :
    "text-blue-600 dark:text-blue-400"

  const deltaColor =
    maxPct > 80 ? "bg-red-300 dark:bg-red-700" :
    maxPct > 50 ? "bg-orange-300 dark:bg-orange-700" :
    "bg-blue-300 dark:bg-blue-700"

  const hasDelta = info.endContextWindowPct != null && info.endContextWindowPct !== info.contextWindowPct

  const messages = parseInputMessages(info.inputMessagesJson)
  const categories = categorizeMessages(messages)
  const visibleTokens = categories.reduce((s, c) => s + c.totalTokens, 0)
  const hasInputMessages = !!info.inputMessagesJson
  const cacheInputTokens = info.cacheReadTokens + info.cacheWriteTokens
  // stableHidden = system overhead if available; otherwise the portion of inputMessagesTokens
  // that can't be attributed to visible messages (includes cache + system prompt)
  const stableHidden = info.systemOverheadTokens > 0
    ? info.systemOverheadTokens
    : hasInputMessages
      ? Math.max(0, info.inputMessagesTokens - visibleTokens - cacheInputTokens)
      : cacheInputTokens
  const deltaTokens = hasInputMessages ? Math.max(0, info.inputMessagesTokens - visibleTokens - stableHidden - cacheInputTokens) : 0
  const unclassifiedTokens = hasInputMessages ? 0 : Math.max(0, info.inputMessagesTokens - stableHidden - cacheInputTokens)

  function toggleCategory(role: string) {
    setExpandedCategories(prev => {
      const next = new Set(prev)
      if (next.has(role)) next.delete(role)
      else next.add(role)
      return next
    })
  }

  function toggleMsg(key: string) {
    setExpandedMsgs(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  return (
    <div className={cn(
      "border rounded-lg",
      info.isSubagent ? "border-orange-200 dark:border-orange-500/20" : "border-emerald-200 dark:border-emerald-500/20"
    )}>
      <div className="px-3 py-2">
        {/* Agent header */}
        <div className="flex items-center gap-2 mb-2">
          <Badge variant={info.isSubagent ? "orange" : "blue"}>
            {info.isSubagent ? `🤖 ${info.subagentName ?? "subagent"}` : "🤖 root"}
          </Badge>
          {info.model && (
            <span className="text-xs text-muted-foreground truncate max-w-[160px]">{info.model}</span>
          )}
        </div>

        {/* Context window usage bar */}
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs text-muted-foreground w-[52px]">Context</span>
          <div className="flex-1 h-3 bg-muted rounded-full overflow-hidden relative">
            {(() => {
              const segments = []
              let leftPct = 0
              for (const [i, cat] of categories.entries()) {
                if (cat.totalTokens > 0) {
                  const w = (cat.totalTokens / info.contextWindowLimit) * 100
                  segments.push({ key: `${cat.role}-${i}`, color: cat.color, left: leftPct, width: w, isFirst: leftPct === 0 })
                  leftPct += w
                }
              }
              if (cacheInputTokens > 0) {
                const w = (cacheInputTokens / info.contextWindowLimit) * 100
                segments.push({ key: "cached", color: "bg-yellow-400 dark:bg-yellow-500", left: leftPct, width: w, isFirst: segments.length === 0 })
                leftPct += w
              }
              if (stableHidden > 0) {
                const w = (stableHidden / info.contextWindowLimit) * 100
                segments.push({ key: "hidden", color: "bg-gray-400 dark:bg-gray-500", left: leftPct, width: w, isFirst: segments.length === 0 })
                leftPct += w
              }
              if (deltaTokens > 0) {
                const w = (deltaTokens / info.contextWindowLimit) * 100
                segments.push({ key: "delta_tokens", color: "bg-yellow-400 dark:bg-yellow-500", left: leftPct, width: w, isFirst: segments.length === 0 })
                leftPct += w
              }
              if (hasDelta && endPctValue > pctValue) {
                const deltaLeft = Math.max(pctValue, leftPct)
                const deltaWidth = endPctValue - deltaLeft
                if (deltaWidth > 0) {
                  segments.push({ key: "delta", color: cn(deltaColor), left: deltaLeft, width: deltaWidth, isFirst: false })
                }
              }
              // Fallback: when no detailed messages, fill based on contextWindowPct
              if (segments.length === 0 && pctValue > 0) {
                segments.push({ key: "fill", color: pctColor, left: 0, width: Math.min(pctValue, 100), isFirst: true })
                if (hasDelta && endPctValue > pctValue) {
                  segments.push({ key: "delta", color: cn(deltaColor), left: Math.min(pctValue, 100), width: Math.min(Math.max(endPctValue - pctValue, 0), 100 - pctValue), isFirst: false })
                }
              }
              return segments.map(seg => (
                <div
                  key={seg.key}
                  className={cn(
                    "h-full absolute",
                    seg.color,
                    seg.isFirst ? "rounded-l-full" : "",
                  )}
                  style={{ left: `${seg.left}%`, width: `${seg.width}%` }}
                />
              ))
            })()}
          </div>
          <span className={cn("text-xs font-medium tabular-nums", pctLabel)}>
            {info.contextWindowPct != null
              ? hasDelta
                ? `${pctValue.toFixed(1)}% → ${endPctValue.toFixed(1)}%`
                : `${pctValue.toFixed(1)}%`
              : "N/A"
            }
          </span>
        </div>

        {/* Context compression alert — only on first significant drop from high context */}
        {prevPct != null && info.contextWindowPct != null && prevPct > 10 && prevPct > info.contextWindowPct + 5 && (
          <div className="flex items-center gap-2 text-xs text-orange-600 dark:text-orange-400 mb-2 px-1 py-1 bg-orange-50/50 dark:bg-orange-500/10 rounded">
            <span>⚠️ Context compressed: {prevPct.toFixed(1)}% → {info.contextWindowPct.toFixed(1)}%</span>
            <span className="text-muted-foreground">history truncated, recent turns only</span>
          </div>
        )}
        {/* Low context indicator — context below 2% and previous was also low */}
        {info.contextWindowPct != null && info.contextWindowPct < 2 && (prevPct == null || prevPct < 2) && (
          <div className="flex items-center gap-1.5 text-xs text-muted-foreground mb-1">
            <span>📦 Minimal context — agent sending only recent tool output to LLM</span>
          </div>
        )}

        {/* Category breakdown */}
        <div className="space-y-1.5 mb-2">
          {categories.map((cat, i) => {
            const pct = info.contextWindowLimit > 0 ? (cat.totalTokens / info.contextWindowLimit * 100).toFixed(1) : "0"
            const skillCount = cat.skillHeaders.length

            return (
              <div key={`cat-${i}`} className="flex items-center gap-2 text-xs">
                <div className={cn("w-2 h-2 rounded-sm shrink-0", cat.color)} />
                <span className={cn("font-medium min-w-[90px]", cat.textColor)}>{cat.label}</span>
                <span className="text-muted-foreground">{cat.messages.length} msg</span>
                <span className="text-muted-foreground">{formatTokenCount(cat.totalTokens)}t</span>
                <span className="text-muted-foreground">{pct}%</span>
                {skillCount > 0 && (
                  <Badge variant="yellow" className="text-xs">⚡ {skillCount} skill header</Badge>
                )}
              </div>
            )
          })}
          {cacheInputTokens > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-sm shrink-0 bg-yellow-400 dark:bg-yellow-500" />
              <span className="font-medium min-w-[90px] text-yellow-600 dark:text-yellow-400">Cached</span>
              <span className="text-muted-foreground">{formatTokenCount(cacheInputTokens)}t</span>
              <span className="text-muted-foreground">{info.contextWindowLimit > 0 ? (cacheInputTokens / info.contextWindowLimit * 100).toFixed(1) : "0"}%</span>
            </div>
          )}
          {stableHidden > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-sm shrink-0 bg-gray-400 dark:bg-gray-500" />
              <span className="font-medium min-w-[90px] text-gray-600 dark:text-gray-400">System (hidden)</span>
              <span className="text-muted-foreground">{formatTokenCount(stableHidden)}t</span>
              <span className="text-muted-foreground">{info.contextWindowLimit > 0 ? (stableHidden / info.contextWindowLimit * 100).toFixed(1) : "0"}%</span>
            </div>
          )}
          {deltaTokens > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-sm shrink-0 bg-yellow-400 dark:bg-yellow-500" />
              <span className="font-medium min-w-[90px] text-yellow-600 dark:text-yellow-400">Other context</span>
              <span className="text-muted-foreground">{formatTokenCount(deltaTokens)}t</span>
              <span className="text-muted-foreground">{info.contextWindowLimit > 0 ? (deltaTokens / info.contextWindowLimit * 100).toFixed(1) : "0"}%</span>
            </div>
          )}
          {unclassifiedTokens > 0 && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-2 h-2 rounded-sm shrink-0 bg-blue-300 dark:bg-blue-500" />
              <span className="font-medium min-w-[90px] text-blue-600 dark:text-blue-400">Unclassified</span>
              <span className="text-muted-foreground">{formatTokenCount(unclassifiedTokens)}t</span>
              <span className="text-muted-foreground">{info.contextWindowLimit > 0 ? (unclassifiedTokens / info.contextWindowLimit * 100).toFixed(1) : "0"}%</span>
              <span className="text-muted-foreground italic">context data not available</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 text-xs text-muted-foreground border-t pt-1.5">
          <span>Total: {info.inputMessagesCount} messages</span>
          <span>{formatTokenCount(info.inputMessagesTokens)} tokens</span>
        </div>

        {messages.length > 0 && (
          <button
            className="mt-1 text-xs text-primary cursor-pointer hover:underline"
            onClick={() => setIsExpanded(!isExpanded)}
          >
            {isExpanded ? "▼ Hide full context" : "▶ Show full context"}
          </button>
        )}
      </div>

      {/* Full context expansion */}
      {isExpanded && categories.map((cat, i) => (
        <div key={`cat-${i}`} className="border-t">
          <button
            className="w-full flex items-center gap-2 px-3 py-1.5 hover:bg-accent/30 transition-colors cursor-pointer"
            onClick={() => toggleCategory(cat.role)}
          >
            <div className={cn("w-2 h-2 rounded-sm", cat.color)} />
            <span className={cn("font-medium text-sm", cat.textColor)}>{cat.icon} {cat.label}</span>
            <span className="text-xs text-muted-foreground">{cat.messages.length} msg · {formatTokenCount(cat.totalTokens)}t</span>
            {cat.skillHeaders.length > 0 && (
              <Badge variant="yellow" className="text-xs">⚡ {cat.skillHeaders.length}</Badge>
            )}
            <span className="text-xs text-muted-foreground ml-auto">
              {expandedCategories.has(cat.role) ? "▼" : "▶"}
            </span>
          </button>

          {expandedCategories.has(cat.role) && (
            <div className="px-3 pb-2 space-y-1 max-h-[300px] overflow-y-auto">
              {cat.messages.map((msg, index) => {
                const key = `${cat.role}-${index}`
                const isMsgExpanded = expandedMsgs.has(key)
                const summary = truncate(msg.content ?? "", 80)
                const showFull = isMsgExpanded && msg.content && msg.content.length > 80
                const isSkill = cat.skillHeaders.some(sh => sh.msg === msg)

                return (
                  <div key={key} className={cn("border rounded-md overflow-hidden", isSkill && "border-l-3 border-l-yellow-400 bg-yellow-50/10 dark:bg-yellow-500/5")}>
                    <button
                      className="w-full flex items-center gap-1.5 px-2 py-1 hover:bg-accent/30 transition-colors text-xs cursor-pointer"
                      onClick={() => toggleMsg(key)}
                    >
                      <Badge variant={ROLE_BADGE_VARIANTS[msg.role] ?? "gray"} className="text-xs">
                        {msg.role}
                      </Badge>
                      {msg.name && <span className="text-muted-foreground">{msg.name}</span>}
                      {isSkill && <Badge variant="yellow" className="text-xs">⚡ skill</Badge>}
                      <span className="flex-1 text-muted-foreground truncate">
                        {summary ?? "(empty)"}
                      </span>
                      {msg.tokenCount != null && msg.tokenCount > 0 && (
                        <span className="text-muted-foreground">{formatTokenCount(msg.tokenCount)}t</span>
                      )}
                      {msg.content && msg.content.length > 80 && (
                        <span className="text-muted-foreground">{isMsgExpanded ? "▼" : "▶"}</span>
                      )}
                    </button>

                    {showFull && (
                      <div className="px-2 pb-1.5 text-xs whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto bg-muted/30">
                        {msg.content}
                      </div>
                    )}

                    {/* Tool calls attached to assistant message */}
                    {(isMsgExpanded || showFull) && msg.tool_calls && msg.tool_calls.length > 0 && (
                      <div className="px-2 pb-1.5 space-y-1">
                        {msg.tool_calls.map((tc, tcIdx) => {
                          const tcTokens = Math.round(((tc.args?.length ?? 0) + (tc.result?.length ?? 0)) / 3.5)
                          return (
                          <div key={tcIdx} className="border rounded-md overflow-hidden bg-orange-50/30 dark:bg-orange-500/5">
                            <div className="flex items-center gap-2 px-2 py-1 text-xs">
                              <Badge variant="orange" className="text-xs">{tc.name}</Badge>
                              <span className="text-muted-foreground">tool call</span>
                              {tcTokens > 0 && (
                                <span className="text-muted-foreground">{formatTokenCount(tcTokens)}t</span>
                              )}
                            </div>
                            {tc.args && (
                              <div className="px-2 py-1 text-xs whitespace-pre-wrap break-words max-h-[100px] overflow-y-auto border-t bg-muted/20">
                                <span className="font-medium text-muted-foreground">args:</span> {tc.args}
                              </div>
                            )}
                            {tc.result && (
                              <div className="px-2 py-1 text-xs whitespace-pre-wrap break-words max-h-[100px] overflow-y-auto border-t bg-muted/20">
                                <span className="font-medium text-muted-foreground">result:</span> {tc.result}
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
            </div>
          )}
        </div>
      ))}
    </div>
  )
}

export function TurnContextPanel({ selectedTurn, rootContext, subagentContexts, prevContextPct }: TurnContextPanelProps) {
  if (!selectedTurn) {
    return (
      <div className="flex items-center justify-center h-full text-muted-foreground text-sm p-4">
        Select a turn to view context
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full overflow-y-auto p-3 space-y-3">
      <div className="flex items-center gap-2 text-sm shrink-0">
        <span className="font-mono text-muted-foreground">#{selectedTurn.turnIndex}</span>
        <Badge variant={ROLE_BADGE_VARIANTS[selectedTurn.role] ?? "gray"}>
          {ROLE_ICONS[selectedTurn.role] ?? "?"} {selectedTurn.role}
        </Badge>
        <span className="text-muted-foreground">Context</span>
      </div>

      {rootContext && (
        <ContextSection info={rootContext} prevPct={prevContextPct} />
      )}

      {subagentContexts.length > 0 && subagentContexts.map((ctx, i) => (
        <ContextSection key={`sub-${i}`} info={ctx} prevPct={null} />
      ))}

      {!rootContext && subagentContexts.length === 0 && (
        <div className="text-xs text-muted-foreground py-4 text-center">
          No context data available for this turn
        </div>
      )}
    </div>
  )
}
