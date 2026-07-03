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
import { LlmContextView } from "./LlmContextView"
import { LlmOutputView } from "./LlmOutputView"
import { TokenBarChart } from "./TokenBarChart"
import { ToolCallList } from "./ToolCallList"
import { SkillEventList } from "./SkillEventList"

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

export function TurnDetail({ turn }: { turn: TurnDetailData }) {
  if (!turn) return null

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
      />

      <LlmOutputView
        content={turn.content}
        contentJson={turn.contentJson}
        contentSummary={turn.contentSummary ?? (turn.content ? (turn.content.length > 200 ? turn.content.substring(0, 200) + "..." : turn.content) : null)}
        outputTokens={turn.outputTokens}
        reasoningTokens={turn.reasoningTokens}
        role={turn.role}
      />

      {turn.toolCalls.length > 0 && (
        <Card size="sm">
          <CardHeader>
            <CardTitle>Tool Calls ({turn.toolCalls.length}{turn.toolCalls.some(tc => tc.isSkillRelated) ? `, ${turn.toolCalls.filter(tc => tc.isSkillRelated).length} skill` : ""})</CardTitle>
          </CardHeader>
          <CardContent>
            <ToolCallList toolCalls={turn.toolCalls} />
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
