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
import { CopyButton } from "./CopyButton"

interface ToolCallEntry {
  name: string
  args: string | null
  result: string | null
  isSkillRelated?: boolean
}

interface InputMessage {
  role: string
  content: string | null
  tokenCount?: number
  name?: string
  tool_calls?: ToolCallEntry[]
}

interface LlmContextViewProps {
  inputMessagesJson: string | null
  inputMessagesCount: number
  inputMessagesTokens: number
  contextWindowPct: number | null
  systemOverheadTokens?: number
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

function truncate(text: string | null, maxLen: number): string | null {
  if (!text) return null
  if (text.length <= maxLen) return text
  return text.substring(0, maxLen) + "..."
}

function formatTokenCount(n: number): string {
  if (n === 0) return ""
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

// Estimate tokens from character count (rough heuristic for mixed CJK/English)
// ~1 token per ~3.5 characters for typical LLM tokenization
function estimateTokensFromChars(charLen: number): number {
  return Math.round(charLen / 3.5)
}

export function LlmContextView({
  inputMessagesJson,
  inputMessagesCount,
  inputMessagesTokens,
  contextWindowPct,
  systemOverheadTokens,
}: LlmContextViewProps) {
  if (inputMessagesCount === 0 && !inputMessagesJson) {
    return null
  }

  const messages = parseInputMessages(inputMessagesJson)
  const totalVisibleTokens = messages.reduce((s, m) => s + (m.tokenCount ?? estimateTokensFromChars(m.content?.length ?? 0)), 0)
  const autoExpand = totalVisibleTokens < 6000

  const [isExpanded, setIsExpanded] = useState(autoExpand)
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(() =>
    autoExpand ? new Set(messages.map((_, i) => i)) : new Set()
  )

  // System overhead: stable from first turn + delta for subsequent turns
  const stableHidden = systemOverheadTokens ?? 0
  const deltaTokens = Math.max(0, inputMessagesTokens - totalVisibleTokens - stableHidden)

  const pctValue = contextWindowPct ?? 0
  const pctColor =
    pctValue > 80 ? "bg-red-500" :
    pctValue > 50 ? "bg-orange-500" :
    pctValue > 0 ? "bg-blue-500" :
    "bg-gray-300"

  function toggleMessage(index: number) {
    setExpandedMessages(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  const ROLE_BADGE_VARIANTS_INLINE: Record<string, "blue" | "green" | "gray" | "purple" | "orange"> = {
    system: "purple",
    user: "blue",
    assistant: "green",
    tool_result: "gray",
    tool: "gray",
  }
  const ROLE_DOT_COLOR: Record<string, string> = {
    system: "bg-purple-500",
    user: "bg-blue-500",
    assistant: "bg-emerald-500",
    tool_result: "bg-teal-500",
    tool: "bg-gray-400",
  }
  const ROLE_TEXT_COLOR: Record<string, string> = {
    system: "text-purple-600 dark:text-purple-400",
    user: "text-blue-600 dark:text-blue-400",
    assistant: "text-emerald-600 dark:text-emerald-400",
    tool_result: "text-teal-600 dark:text-teal-400",
    tool: "text-gray-600",
  }

  return (
    <div className="border rounded-lg">
      <span
        role="button"
        tabIndex={0}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-accent/50 transition-colors cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsExpanded(!isExpanded) }}
      >
        <div className="flex items-center gap-2 text-sm">
          <span className="font-medium">LLM Input</span>
          <span className="text-muted-foreground">
            {inputMessagesCount} messages, {formatTokenCount(inputMessagesTokens)} tokens
            {contextWindowPct != null && contextWindowPct > 0 && ` (${contextWindowPct.toFixed(1)}% context)`}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {contextWindowPct != null && contextWindowPct > 0 && (
            <div className="w-24 h-2 bg-muted rounded-full overflow-hidden">
              <div
                className={cn("h-full rounded-full transition-all", pctColor)}
                style={{ width: `${Math.min(pctValue, 100)}%` }}
              />
            </div>
          )}
          <span className="text-xs text-muted-foreground">
            {isExpanded ? "▼" : "▶"}
          </span>
        </div>
      </span>

      {isExpanded && (
        <div className="border-t px-3 py-2 space-y-1.5">
          {/* Hidden system context notice */}
          {stableHidden > 100 && (
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-purple-50/50 dark:bg-purple-500/10 border border-purple-200 dark:border-purple-500/20">
              <div className="w-2 h-2 rounded-sm bg-purple-500 shrink-0" />
              <span className="text-xs font-medium text-purple-600 dark:text-purple-400">System (hidden)</span>
              <span className="text-xs text-muted-foreground">≈{formatTokenCount(stableHidden)}t</span>
            </div>
          )}
          {deltaTokens > 100 && (
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-yellow-50/50 dark:bg-yellow-500/10 border border-yellow-200 dark:border-yellow-500/20">
              <div className="w-2 h-2 rounded-sm bg-yellow-400 shrink-0" />
              <span className="text-xs font-medium text-yellow-600 dark:text-yellow-400">Other context</span>
              <span className="text-xs text-muted-foreground">{formatTokenCount(deltaTokens)}t</span>
            </div>
          )}

          {/* Visible messages in interleaved order */}
          {messages.map((msg, index) => {
            const isMsgExpanded = expandedMessages.has(index)
            const msgTokens = msg.tokenCount ?? estimateTokensFromChars(msg.content?.length ?? 0)
            const isLongContent = msg.content && msg.content.length > 500

            return (
              <div key={index} className="border rounded-md overflow-hidden">
                <span
                  role="button"
                  tabIndex={0}
                  className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent/30 transition-colors text-sm cursor-pointer"
                  onClick={() => toggleMessage(index)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleMessage(index) }}
                >
                  <div className={cn("w-2 h-2 rounded-sm shrink-0", ROLE_DOT_COLOR[msg.role] ?? "bg-gray-400")} />
                  <Badge variant={ROLE_BADGE_VARIANTS_INLINE[msg.role] ?? "gray"} className="text-xs">
                    {msg.role}
                  </Badge>
                  {msg.name && (
                    <span className="text-xs text-muted-foreground">{msg.name}</span>
                  )}
                  {msgTokens > 0 && (
                    <span className="text-xs text-muted-foreground">{formatTokenCount(msgTokens)}t</span>
                  )}
                  {isLongContent && (
                    <span className="text-xs text-muted-foreground">
                      {isMsgExpanded ? "▼" : "▶"}
                    </span>
                  )}
                  {msg.content && <CopyButton text={msg.content} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />}
                </span>

                {isMsgExpanded && msg.content && (
                  <div className="px-2 pb-2 text-sm whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto bg-muted/30">
                    {msg.content}
                  </div>
                )}

                {/* Tool calls attached to assistant message */}
                {isMsgExpanded && msg.tool_calls && msg.tool_calls.length > 0 && (
                  <div className="px-2 pb-2 space-y-1.5">
                    {msg.tool_calls.map((tc, tcIdx) => {
                      const tcTokens = Math.round(((tc.args?.length ?? 0) + (tc.result?.length ?? 0)) / 3.5)
                      return (
                        <div key={tcIdx} className={cn("border rounded-md overflow-hidden bg-orange-50/30 dark:bg-orange-500/5", tc.isSkillRelated && "border-l-3 border-l-yellow-400")}>
                          <div className="flex items-center gap-2 px-2 py-1 text-xs">
                            <Badge variant={tc.isSkillRelated ? "yellow" : "orange"} className="text-xs">{tc.isSkillRelated ? "⚡" : tc.name}</Badge>
                            {!tc.isSkillRelated && <span className="text-muted-foreground">tool call</span>}
                            {tcTokens > 0 && (
                              <span className="text-muted-foreground">{formatTokenCount(tcTokens)}t</span>
                            )}
                            {(tc.args || tc.result) && <CopyButton text={[tc.args, tc.result].filter(Boolean).join("\n\n---\n\n")} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />}
                          </div>
                          {tc.args && (
                            <div className="px-2 py-1 text-xs whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto border-t bg-muted/20">
                              <span className="font-medium text-muted-foreground">args:</span> {tc.args}
                            </div>
                          )}
                          {tc.result && (
                            <div className="px-2 py-1 text-xs whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto border-t bg-muted/20">
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
  )
}
