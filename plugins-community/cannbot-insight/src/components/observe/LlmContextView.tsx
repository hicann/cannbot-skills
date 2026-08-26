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
  agentName?: string | null
  tool_calls?: ToolCallEntry[]
}

// LLM Input mirrors the WIRE: messages render verbatim in original order
// (reminder blocks stay inside their user message, registry + skills stay in
// their system message) — no extracted copies. The System / System tools
// panels above the list are wire TOP-LEVEL fields (system prompt, tools),
// which are not part of any message, so they never duplicate message content.
// reminderPrefix is used by TurnContextPanel to keep its compact summaries
// prompt-only.
export function reminderPrefix(content: string | null): { reminder: string; prompt: string } | null {
  if (!content?.startsWith("<system-reminder>")) return null
  const end = content.lastIndexOf("</system-reminder>")
  if (end < 0) return null
  return {
    reminder: content.slice(0, end + "</system-reminder>".length),
    prompt: content.slice(end + "</system-reminder>".length).trim(),
  }
}

interface LlmContextViewProps {
  inputMessagesJson: string | null
  inputMessagesCount: number
  inputMessagesTokens: number
  contextWindowPct: number | null
  systemOverheadTokens?: number
  systemPrompt?: string | null
  fullContext?: {
    tools: Array<{ name: string; description: string }>
  } | null
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
  systemPrompt,
  fullContext,
}: LlmContextViewProps) {
  if (inputMessagesCount === 0 && !inputMessagesJson) {
    return null
  }

  const messages = parseInputMessages(inputMessagesJson)
  const totalVisibleTokens = messages.reduce((s, m) => s + (m.tokenCount ?? estimateTokensFromChars(m.content?.length ?? 0)), 0)
  const autoExpand = totalVisibleTokens < 6000

  const [isExpanded, setIsExpanded] = useState(autoExpand)
  const [expandedMessages, setExpandedMessages] = useState<Set<number>>(() => {
    const set = new Set<number>()
    if (autoExpand) {
      for (let i = 0; i < messages.length; i++) set.add(i)
    } else {
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].agentName === 'continuation') set.add(i)
      }
    }
    return set
  })

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
          {/* System context: real verbatim content (proxy capture) or hidden residual (log-imported) */}
          {systemPrompt ? (
            <ContextSection title="System" text={systemPrompt} />
          ) : stableHidden > 100 && (
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-purple-50/50 dark:bg-purple-500/10 border border-purple-200 dark:border-purple-500/20">
              <div className="w-2 h-2 rounded-sm bg-purple-500 shrink-0" />
              <span className="text-xs font-medium text-purple-600 dark:text-purple-400">System (hidden)</span>
              <span className="text-xs text-muted-foreground">≈{formatTokenCount(stableHidden)}t</span>
            </div>
          )}
          {fullContext?.tools && fullContext.tools.length > 0 && (
            <ToolsPanel tools={fullContext.tools} />
          )}
          {deltaTokens > 100 && (
            <div className="flex items-center gap-2 px-2 py-1.5 rounded-md bg-yellow-50/50 dark:bg-yellow-500/10 border border-yellow-200 dark:border-yellow-500/20">
              <div className="w-2 h-2 rounded-sm bg-yellow-400 shrink-0" />
              <span className="text-xs font-medium text-yellow-600 dark:text-yellow-400">Other context</span>
              <span className="text-xs text-muted-foreground">{formatTokenCount(deltaTokens)}t</span>
            </div>
          )}

          {messages.map((msg, index) => {
            const isMsgExpanded = expandedMessages.has(index)
            const msgTokens = msg.tokenCount ?? estimateTokensFromChars(msg.content?.length ?? 0)
            const isContinuation = msg.agentName === 'continuation'

            return (
              <div key={index} className={cn("border rounded-md overflow-hidden", isContinuation && "border-l-3 border-l-purple-500 bg-purple-50/20 dark:bg-purple-500/10")}>
                <span
                  role="button"
                  tabIndex={0}
                  className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent/30 transition-colors text-sm cursor-pointer"
                  onClick={() => toggleMessage(index)}
                  onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleMessage(index) }}
                >
                  <div className={cn("w-2 h-2 rounded-sm shrink-0", isContinuation ? "bg-purple-500" : (ROLE_DOT_COLOR[msg.role] ?? "bg-gray-400"))} />
                  <Badge variant={isContinuation ? "purple" : (ROLE_BADGE_VARIANTS_INLINE[msg.role] ?? "gray")} className="text-xs">
                    {isContinuation ? "continuation" : msg.role}
                  </Badge>
                  {msg.name && (
                    <span className="text-xs text-muted-foreground">{msg.name}</span>
                  )}
                  {msgTokens > 0 && (
                    <span className="text-xs text-muted-foreground">{formatTokenCount(msgTokens)}t</span>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {isMsgExpanded ? "▼" : "▶"}
                  </span>
                  {msg.content && <CopyButton text={msg.content} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />}
                </span>

                {isMsgExpanded && msg.content && (
                  (() => {
                    const rp = reminderPrefix(msg.content)
                    if (!rp) {
                      return (
                        <div className="px-2 pb-2 text-sm whitespace-pre-wrap break-words overflow-y-auto bg-muted/30 max-h-[300px]">
                          {msg.content}
                        </div>
                      )
                    }
                    // wire keeps the reminder inside the user message; render
                    // it as its own visually separated sub-block, prompt below
                    return (
                      <div className="px-2 pb-2 space-y-2 overflow-y-auto bg-muted/30 max-h-[300px]">
                        <div className="rounded border border-purple-200 dark:border-purple-500/30 bg-purple-50/30 dark:bg-purple-500/5 overflow-hidden">
                          <div className="flex items-center gap-2 px-2 py-1 bg-purple-50/50 dark:bg-purple-500/10">
                            <Badge variant="purple" className="text-[10px]">system-reminder</Badge>
                            <span className="text-[10px] text-muted-foreground">{formatTokenCount(estimateTokensFromChars(rp.reminder.length))}t</span>
                          </div>
                          <div className="px-2 py-1.5 text-xs whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto font-mono text-foreground/70">
                            {rp.reminder}
                          </div>
                        </div>
                        {rp.prompt && (
                          <div className="text-sm whitespace-pre-wrap break-words">
                            {rp.prompt}
                          </div>
                        )}
                      </div>
                    )
                  })()
                )}

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

const SYSTEM_SECTION_STYLE = {
  badge: "purple" as const,
  dot: "bg-purple-500",
  border: "border-purple-200 dark:border-purple-500/20",
  bg: "bg-purple-50/30 dark:bg-purple-500/5",
  headerBg: "bg-purple-50/50 dark:bg-purple-500/10",
}

function ContextSection({ title, text }: { title: string; text: string }) {
  const [isOpen, setIsOpen] = useState(false)
  const style = SYSTEM_SECTION_STYLE
  const tokenCount = estimateTokensFromChars(text.length)
  const preview = text.length > 120 ? text.substring(0, 120) + "..." : text
  return (
    <div className={cn("border rounded-md overflow-hidden", style.border, style.bg)}>
      <span
        role="button"
        tabIndex={0}
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent/30 transition-colors text-sm cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsOpen(!isOpen) }}
      >
        <div className={cn("w-2 h-2 rounded-sm shrink-0", style.dot)} />
        <Badge variant={style.badge} className="text-xs">{title}</Badge>
        <span className="text-xs text-muted-foreground">{formatTokenCount(tokenCount)}t · {text.length.toLocaleString()} chars</span>
        <span className="ml-auto text-xs text-muted-foreground">{isOpen ? "▼" : "▶"}</span>
      </span>
      {isOpen && (
        <div className={cn("border-t", style.border)}>
          <div className={cn("flex items-center justify-between px-2 py-1", style.headerBg)}>
            <span className="text-[10px] text-muted-foreground truncate">{preview}</span>
            <CopyButton text={text} />
          </div>
          <pre className="px-3 py-2 text-xs whitespace-pre-wrap break-words max-h-[400px] overflow-y-auto font-mono text-foreground/80">
            {text}
          </pre>
        </div>
      )}
    </div>
  )
}

function ToolsPanel({ tools }: { tools: Array<{ name: string; description: string }> }) {
  const [isOpen, setIsOpen] = useState(false)
  return (
    <div className="border rounded-md overflow-hidden bg-orange-50/30 dark:bg-orange-500/5 border-orange-200 dark:border-orange-500/20">
      <span
        role="button"
        tabIndex={0}
        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-accent/30 transition-colors text-sm cursor-pointer"
        onClick={() => setIsOpen(!isOpen)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsOpen(!isOpen) }}
      >
        <div className="w-2 h-2 rounded-sm bg-orange-400 shrink-0" />
        <Badge variant="orange" className="text-xs">System tools</Badge>
        <span className="text-xs text-muted-foreground">{tools.length} tools</span>
        <span className="ml-auto text-xs text-muted-foreground">{isOpen ? "▼" : "▶"}</span>
      </span>
      {isOpen && (
        <div className="border-t border-orange-200 dark:border-orange-500/20 space-y-1 p-1.5 max-h-[400px] overflow-y-auto">
          {tools.map((t, i) => (
            <div key={i} className="border rounded bg-background/60 px-2 py-1">
              <div className="flex items-center gap-2">
                <Badge variant="orange" className="text-[10px]">{t.name}</Badge>
                {t.description && <CopyButton text={t.description} className="ml-auto size-3 text-muted-foreground hover:text-foreground" />}
              </div>
              {t.description && (
                <div className="mt-0.5 text-[11px] text-muted-foreground whitespace-pre-wrap break-words">
                  {t.description}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
