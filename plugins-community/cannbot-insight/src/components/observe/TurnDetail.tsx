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
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { CopyButton } from "./CopyButton"
import { BashEscapeView, parseBashEscape } from "./BashEscapeView"
import { LlmContextView } from "./LlmContextView"
import { LlmOutputView } from "./LlmOutputView"
import { TokenBarChart } from "./TokenBarChart"
import { ToolCallList } from "./ToolCallList"
import { SkillEventList } from "./SkillEventList"
import type { TurnHighlight } from "@/lib/shared/highlight"

interface TurnDetailData {
  turnId: string
  turnIndex: number
  role: string
  content: string | null
  contentJson: string | null
  contentSummary: string | null
  inputMessagesJson: string | null
  inputMessagesCount: number
  inputMessagesTokens: number
  contextWindowPct: number | null
  systemOverheadTokens?: number
  systemPrompt?: string | null
  fullContext?: {
    tools: Array<{ name: string; description: string }>
    memoryFiles: string
    skills: string
  } | null
  agentName: string | null
  subagentName: string | null
  isSubagent: boolean
  totalTokens: number
  inputTokens: number
  outputTokens: number
  reasoningTokens: number
  cacheReadTokens: number
  cacheWriteTokens: number
  contextWindowLimit?: number
  latencyMs: number
  ttftMs: number | null
  createdAt: string | null
  completedAt: string | null
  model: string | null
  modelId: string | null
  providerId: string | null
  finishReason: string | null
  toolCalls: Array<{
    id: string
    toolCallId: string
    toolName: string
    argsJson: string | null
    resultJson: string | null
    state: string
    errorType: string | null
    errorMessage: string | null
    durationMs: number
    isSkillRelated: boolean
  }>
  skillEvents: Array<{
    id: string
    skillName: string
    skillVersion: number | null
    eventType: string
    success: boolean
    errorMessage: string | null
    argsJson: string | null
    durationMs: number
  }>
}

const ROLE_ICONS: Record<string, string> = {
  user: "👤",
  assistant: "🤖",
  system: "⚙️",
  tool_result: "🔧",
}

interface WireInputMessage {
  role: string
  content: string | Array<Record<string, unknown>>
}

// proxy wire-round 输入 turn：contentJson 存 verbatim 消息数组
// （{wireInput:true, messages:[{role, content}]}）。命中时用逐消息
// wire 渲染替代普通 User Input 面板。
function parseWireInput(contentJson: string | null): WireInputMessage[] | null {
  if (!contentJson) return null
  try {
    const parsed = JSON.parse(contentJson)
    if (parsed && typeof parsed === "object" && parsed.wireInput === true && Array.isArray(parsed.messages)) {
      return parsed.messages as WireInputMessage[]
    }
  } catch { /* not wire-input */ }
  return null
}

// 模型读到的是各 block 的文本内容（渲染层丢掉 JSON 外壳与 cache_control
// 这类协议标记），显示层按同样方式展开，替代转义 JSON 直出。verbatim
// 原文仍在捕获文件里。
function blockText(b: Record<string, unknown>): string {
  if (b?.type === "text" || b?.type === "thinking" || b?.type === "reasoning") {
    const t = b.text ?? b.thinking ?? b.content
    return typeof t === "string" ? t : ""
  }
  if (b?.type === "tool_use") return `[tool_use ${typeof b.name === "string" ? b.name : "?"}] ${JSON.stringify(b.input ?? {})}`
  if (b?.type === "tool_result") {
    const c = b.content
    const inner = typeof c === "string" ? c : Array.isArray(c) ? c.map(x => blockText(x as Record<string, unknown>)).join("\n") : c == null ? "" : JSON.stringify(c)
    return `[tool_result${b.is_error ? " error" : ""}] ${inner}`
  }
  if (b?.type === "image") return "[image]"
  return JSON.stringify(b)
}

function wireMessageText(m: WireInputMessage): { text: string; chars: number } {
  const raw = typeof m.content === "string" ? m.content : (m.content ?? []).map(b => blockText(b)).join("\n\n")
  return { text: raw, chars: raw.length }
}

function WireInputMessages({ messages }: { messages: WireInputMessage[] }) {
  const [openMsgs, setOpenMsgs] = useState<Set<number>>(() => {
    const set = new Set<number>()
    if (messages.length <= 3) for (let i = 0; i < messages.length; i++) set.add(i)
    return set
  })
  const toggle = (i: number) => setOpenMsgs(prev => {
    const next = new Set(prev)
    if (next.has(i)) next.delete(i); else next.add(i)
    return next
  })
  const ROLE_VARIANT: Record<string, "blue" | "green" | "purple" | "gray"> = {
    user: "blue",
    assistant: "green",
    system: "purple",
  }
  return (
    <div className="border rounded-lg">
      <div className="flex items-center gap-2 px-3 py-2 text-sm border-b">
        <span className="font-medium">Wire 输入</span>
        <span className="text-muted-foreground">{messages.length} 条消息 · 本轮新增（verbatim）</span>
      </div>
      <div className="px-3 py-2 space-y-1.5">
        {messages.map((m, i) => {
          const { text, chars } = wireMessageText(m)
          const isOpen = openMsgs.has(i)
          const isToolResult = typeof m.content !== "string" && Array.isArray(m.content) && m.content.length > 0 && (m.content as Array<Record<string, unknown>>).every(b => b?.type === "tool_result")
          const esc = parseBashEscape(text)
          return (
            <div key={i} className="border rounded bg-muted/20 overflow-hidden">
              <span
                role="button"
                tabIndex={0}
                className="w-full flex items-center gap-2 px-2 py-1 hover:bg-accent/30 cursor-pointer"
                onClick={() => toggle(i)}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggle(i) }}
              >
                <span className="text-[10px] text-muted-foreground">{isOpen ? "▼" : "▶"}</span>
                <Badge variant={isToolResult ? "gray" : (ROLE_VARIANT[m.role] ?? "gray")} className="text-xs">
                  {isToolResult ? "tool_result" : m.role}
                </Badge>
                <span className="text-[10px] text-muted-foreground">{chars.toLocaleString()} chars{isOpen ? "" : " · 点击展开全文"}</span>
                <CopyButton text={text} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />
              </span>
              <div className={`px-2 pb-1.5 ${isOpen ? "max-h-[480px] overflow-y-auto" : "max-h-[54px] overflow-hidden"}`}>
                {esc ? <BashEscapeView esc={esc} /> : (
                  <pre className="text-[11px] whitespace-pre-wrap break-all font-mono text-foreground/75">{text}</pre>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

const ROLE_BADGE_VARIANTS: Record<string, "blue" | "green" | "gray" | "purple" | "orange"> = {
  user: "blue",
  assistant: "green",
  system: "gray",
  tool_result: "purple",
}

function formatLatency(ms: number): string {
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function formatTimestamp(ts: string | null): string {
  if (!ts) return "N/A"
  try {
    return new Date(ts).toLocaleString()
  } catch {
    return ts
  }
}

export function TurnDetail({ turn, highlight }: { turn: TurnDetailData; highlight?: TurnHighlight | null }) {
  if (!turn) return null

  const wireMessages = parseWireInput(turn.contentJson)
  const contentLength = (turn.content ?? "").length + (turn.contentJson ?? "").length
  const isLongContent = contentLength > 10000
  const toolOverheadTokens = Math.round(
    turn.toolCalls.reduce(function (s, tc) { return s + (tc.argsJson?.length ?? 0) + (tc.resultJson?.length ?? 0); }, 0) / 3.5
  )

  return (
    <div id="turn-detail-top" className="flex flex-col gap-4 p-4 scroll-mt-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-sm font-mono text-muted-foreground">#{turn.turnIndex}</span>
        <Badge variant={ROLE_BADGE_VARIANTS[turn.role] ?? "gray"}>
          {ROLE_ICONS[turn.role]} {turn.role}
        </Badge>
        {turn.isSubagent && (
          <Badge variant="orange">{turn.subagentName ?? "subagent"}</Badge>
        )}
        {turn.agentName && (
          <Badge variant="outline">{turn.agentName}</Badge>
        )}
        {isLongContent && (
          <Badge variant="outline">long content</Badge>
        )}
      </div>

      <Card size="sm">
        <CardHeader>
          <CardTitle>Overview</CardTitle>
        </CardHeader>
        <CardContent>
          <TokenBarChart
            totalTokens={turn.totalTokens}
            inputTokens={turn.inputTokens}
            outputTokens={turn.outputTokens}
            reasoningTokens={turn.reasoningTokens}
            cacheReadTokens={turn.cacheReadTokens}
            cacheWriteTokens={turn.cacheWriteTokens}
            toolOverheadTokens={toolOverheadTokens}
            contextWindowLimit={turn.contextWindowLimit ?? 200000}
          />

          <div className="flex items-center gap-3 text-xs text-muted-foreground mt-3 pt-2 border-t flex-wrap">
            {turn.model && <span className="font-medium text-foreground">{turn.model}</span>}
            {turn.latencyMs > 0 && <span>{formatLatency(turn.latencyMs)}</span>}
            {turn.createdAt && <span>{formatTimestamp(turn.createdAt)}</span>}
            {turn.finishReason && <span>finish: {turn.finishReason}</span>}
          </div>
        </CardContent>
      </Card>

      <LlmContextView
        inputMessagesJson={turn.inputMessagesJson}
        inputMessagesCount={turn.inputMessagesCount}
        inputMessagesTokens={turn.inputMessagesTokens}
        contextWindowPct={turn.contextWindowPct}
        systemOverheadTokens={turn.systemOverheadTokens ?? 0}
        systemPrompt={turn.systemPrompt ?? null}
        fullContext={turn.fullContext ?? null}
      />

      {wireMessages ? (
        <WireInputMessages messages={wireMessages} />
      ) : (
        <LlmOutputView
          content={turn.content}
          contentJson={turn.contentJson}
          contentSummary={turn.contentSummary ?? (turn.content ? (turn.content.length > 200 ? turn.content.substring(0, 200) + "..." : turn.content) : null)}
          outputTokens={turn.outputTokens}
          reasoningTokens={turn.reasoningTokens}
          role={turn.role}
          highlight={highlight}
        />
      )}

      {turn.toolCalls.length > 0 && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Tool Calls ({turn.toolCalls.length}{turn.toolCalls.some(tc => tc.isSkillRelated) ? `, ${turn.toolCalls.filter(tc => tc.isSkillRelated).length} skill` : ""})</CardTitle>
          </CardHeader>
          <CardContent>
            <ToolCallList toolCalls={turn.toolCalls} highlight={highlight} />
          </CardContent>
        </Card>
      )}

      {turn.skillEvents.length > 0 && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Skills ({new Set(turn.skillEvents.map(se => se.skillName)).size})</CardTitle>
          </CardHeader>
          <CardContent>
            <SkillEventList
              skillEvents={turn.skillEvents}
              skillToolCalls={turn.toolCalls.filter(function (tc) { return tc.isSkillRelated; }).map(function (tc) {
                return {
                  id: tc.id,
                  toolCallId: tc.toolCallId,
                  toolName: tc.toolName,
                  argsJson: tc.argsJson,
                  resultJson: tc.resultJson,
                  state: tc.state,
                  durationMs: tc.durationMs,
                };
              })}
            />
          </CardContent>
        </Card>
      )}

    </div>
  )
}
