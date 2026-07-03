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

interface TurnData {
  turnId: string
  turnIndex: number
  role: string
  content: string | null
  contentSummary: string | null
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  latencyMs: number
  model: string | null
  toolCalls: Array<{ toolCallId: string; toolName: string; state: string; durationMs: number }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
}

interface CompareTurnsProps {
  turnsA: TurnData[]
  turnsB: TurnData[]
}

const ROLE_ICONS: Record<string, string> = {
  user: "👤",
  assistant: "🤖",
  system: "⚙️",
}

const ROLE_COLORS: Record<string, string> = {
  user: "bg-blue-50/50 dark:bg-blue-500/5",
  assistant: "bg-emerald-50/50 dark:bg-emerald-500/5",
  system: "bg-gray-50/50 dark:bg-gray-500/5",
}

function formatTokens(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

function formatMs(ms: number): string {
  if (ms === 0) return ""
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function pairTurns(turnsA: TurnData[], turnsB: TurnData[]): Array<{ index: number; a: TurnData | null; b: TurnData | null }> {
  const maxLen = Math.max(turnsA.length, turnsB.length)
  const pairs: Array<{ index: number; a: TurnData | null; b: TurnData | null }> = []
  for (let i = 0; i < maxLen; i++) {
    const a = turnsA.find(t => t.turnIndex === i) ?? null
    const b = turnsB.find(t => t.turnIndex === i) ?? null
    pairs.push({ index: i, a, b })
  }
  return pairs
}

function tokenDiff(a: number, b: number): { diff: number; winner: "a" | "b" | "none" } {
  if (a === b) return { diff: 0, winner: "none" }
  return { diff: b - a, winner: a < b ? "a" : "b" }
}

function truncateContent(content: string | null, maxLen: number): string | null {
  if (!content) return null
  if (content.length <= maxLen) return content
  return content.substring(0, maxLen) + `... (${content.length} chars total)`
}

function TurnPanel({ turn, side }: { turn: TurnData; side: "A" | "B" }) {
  const [showFull, setShowFull] = useState(false)
  const badgeVariant = side === "A" ? "blue" : "orange"
  const content = turn.content
  const displayContent = showFull ? content : truncateContent(content, 500)
  const thinkingMatch = content?.match(/<thinking>([\s\S]*?)<\/thinking>/)
  const thinkingContent = thinkingMatch?.[1] ?? null
  const textContent = thinkingMatch ? content?.replace(/<thinking>[\s\S]*?<\/thinking>/, "").trim() : content

  return (
    <div className={cn("px-3 py-2 space-y-2", ROLE_COLORS[turn.role] ?? "")}>
      <div className="flex items-center gap-1.5">
        <Badge variant={badgeVariant} className="text-xs">{ROLE_ICONS[turn.role] ?? turn.role} {turn.role}</Badge>
        {turn.totalTokens > 0 && <span className="text-xs tabular-nums">{formatTokens(turn.totalTokens)} tok</span>}
        {turn.latencyMs > 0 && <span className="text-xs text-muted-foreground">{formatMs(turn.latencyMs)}</span>}
        {turn.model && <span className="text-xs text-muted-foreground truncate">{turn.model}</span>}
      </div>

      {thinkingContent && (
        <div className="border rounded-md p-1.5 bg-purple-50/30 dark:bg-purple-500/5">
          <div className="flex items-center gap-1 mb-1">
            <Badge variant="purple" className="text-xs">thinking</Badge>
            <span className="text-xs text-muted-foreground">{formatTokens(turn.reasoningTokens)} reasoning tokens</span>
          </div>
          <pre className="text-xs whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto">
            {showFull ? thinkingContent : truncateContent(thinkingContent, 300)}
          </pre>
        </div>
      )}

      {(textContent || !thinkingContent) && (
        <div>
          <pre className="text-xs whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto">
            {displayContent ?? turn.contentSummary ?? "(empty)"}
          </pre>
          {content && content.length > 500 && !showFull && (
            <button
              className="text-xs text-blue-500 hover:text-blue-600 cursor-pointer mt-1"
              onClick={() => setShowFull(true)}
            >
              Show full content ({content.length} chars)
            </button>
          )}
          {showFull && content && content.length > 500 && (
            <button
              className="text-xs text-muted-foreground hover:text-foreground cursor-pointer mt-1"
              onClick={() => setShowFull(false)}
            >
              Collapse
            </button>
          )}
        </div>
      )}

      {turn.inputTokens > 0 && (
        <div className="flex gap-2 text-xs text-muted-foreground">
          <span>in:{formatTokens(turn.inputTokens)}</span>
          <span>out:{formatTokens(turn.outputTokens)}</span>
          {turn.reasoningTokens > 0 && <span>reasoning:{formatTokens(turn.reasoningTokens)}</span>}
        </div>
      )}

      {turn.toolCalls.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {turn.toolCalls.map(tc => (
            <Badge key={tc.toolCallId} variant="outline" className="text-xs">{tc.toolName}</Badge>
          ))}
        </div>
      )}

      {turn.skillEvents.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {turn.skillEvents.map(se => (
            <Badge key={se.skillName + se.eventType} variant="yellow" className="text-xs">{se.skillName}</Badge>
          ))}
        </div>
      )}
    </div>
  )
}

export function CompareTurns({ turnsA, turnsB }: CompareTurnsProps) {
  const pairs = pairTurns(turnsA, turnsB)

  if (pairs.length === 0) {
    return <p className="text-sm text-muted-foreground">No turn data available.</p>
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span className="inline-block w-3 h-3 rounded bg-blue-500" /> Session A ({turnsA.length} turns)
        <span className="inline-block w-3 h-3 rounded bg-orange-500 ml-2" /> Session B ({turnsB.length} turns)
      </div>

      {pairs.map(({ index, a, b }) => {
        const roleA = a?.role ?? "—"
        const roleB = b?.role ?? "—"
        const sameRole = roleA === roleB
        const tokensA = a?.totalTokens ?? 0
        const tokensB = b?.totalTokens ?? 0
        const td = tokenDiff(tokensA, tokensB)

        return (
          <div key={index} className="border rounded-lg">
            <div className="flex items-center gap-2 px-2 py-1 bg-muted/30 border-b text-xs">
              <span className="font-mono text-muted-foreground">Turn #{index}</span>
              {!sameRole && (
                <Badge variant="red" className="text-xs">role mismatch: {roleA} vs {roleB}</Badge>
              )}
              <span className="ml-auto text-muted-foreground">
                {td.diff !== 0 && (
                  <span className={td.winner === "a" ? "text-blue-600 dark:text-blue-400" : "text-orange-600 dark:text-orange-400"}>
                    {td.diff > 0 ? "+" : ""}{formatTokens(Math.abs(td.diff))} tokens {td.winner === "a" ? "A wins" : "B wins"}
                  </span>
                )}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-0 divide-x">
              {a ? <TurnPanel turn={a} side="A" /> : (
                <div className="px-3 py-2 bg-muted/20">
                  <p className="text-xs text-muted-foreground italic">No turn at this index in A</p>
                </div>
              )}
              {b ? <TurnPanel turn={b} side="B" /> : (
                <div className="px-3 py-2 bg-muted/20">
                  <p className="text-xs text-muted-foreground italic">No turn at this index in B</p>
                </div>
              )}
            </div>
          </div>
        )
      })}
    </div>
  )
}
