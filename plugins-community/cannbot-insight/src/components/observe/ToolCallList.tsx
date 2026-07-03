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
import { CopyButton } from "./CopyButton"

interface ToolCallItem {
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
}

interface ToolCallListProps {
  toolCalls: ToolCallItem[]
}

function hasResultError(resultJson: string | null): boolean {
  if (!resultJson) return false
  return resultJson.includes('<tool_use_error>') || resultJson.includes('Exit code')
}

function toolCallDisplayState(state: string, errorType: string | null, resultJson: string | null): { label: string; variant: "green" | "red" | "gray" | "orange" } {
  if ((state === "ok" || state === "completed") && (errorType || hasResultError(resultJson))) return { label: "error", variant: "red" }
  if (state === "ok" || state === "completed") return { label: state, variant: "green" }
  if (state === "error" || state === "failed") return { label: state, variant: "red" }
  return { label: state, variant: "orange" }
}

function formatDuration(ms: number): string {
  if (ms === 0) return ""
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function truncateJson(json: string | null, maxLen: number): string | null {
  if (!json) return null
  if (json.length <= maxLen) return json
  return json.substring(0, maxLen) + `... (${json.length} chars total)`
}

function extractSkillLabel(argsJson: string | null): string | null {
  if (!argsJson) return null
  try {
    const args = JSON.parse(argsJson)
    return args.skill_name ?? args.name ?? null
  } catch { /* ignore */ }
  return null
}

export function ToolCallList({ toolCalls }: ToolCallListProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  const skillCalls = toolCalls.filter(tc => tc.isSkillRelated)
  const nonSkillCalls = toolCalls.filter(tc => !tc.isSkillRelated)

  function toggleExpanded(id: string) {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <div className="space-y-1.5">
      {skillCalls.map(tc => {
        const displayState = toolCallDisplayState(tc.state, tc.errorType, tc.resultJson)
        const skillLabel = extractSkillLabel(tc.argsJson)

        return (
          <div key={tc.id} className="flex items-center gap-2 px-2 py-1 rounded-md border-l-3 border-l-yellow-400 bg-yellow-50/20 dark:bg-yellow-500/5 text-xs">
            <Badge variant="yellow" className="text-xs">⚡</Badge>
            {skillLabel ? (
              <span className="font-medium text-foreground/80">{skillLabel}</span>
            ) : (
              <span className="font-medium text-foreground/80">{tc.toolName}</span>
            )}
            <Badge variant={displayState.variant} className="text-xs">{displayState.label}</Badge>
            {tc.durationMs > 0 && (
              <span className="text-muted-foreground">{formatDuration(tc.durationMs)}</span>
            )}
            <span className="text-muted-foreground ml-auto">→ Skills</span>
          </div>
        )
      })}
      {nonSkillCalls.map(tc => {
        const isExpanded = expandedIds.has(tc.id)
        const displayState = toolCallDisplayState(tc.state, tc.errorType, tc.resultJson)

        return (
          <div key={tc.id} className="border rounded-md">
            <span
              role="button"
              tabIndex={0}
              className="w-full flex items-center gap-2 px-2 py-1.5 text-sm cursor-pointer hover:bg-accent/30 transition-colors"
              onClick={() => toggleExpanded(tc.id)}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') toggleExpanded(tc.id) }}
            >
              <span className="font-medium">{tc.toolName}</span>
              <Badge variant={displayState.variant}>{displayState.label}</Badge>
              {tc.durationMs > 0 && (
                <span className="text-xs text-muted-foreground">{formatDuration(tc.durationMs)}</span>
              )}
              {tc.errorType && (
                <Badge variant="red">{tc.errorType}</Badge>
              )}
              {(tc.argsJson || tc.resultJson) && <CopyButton text={[tc.argsJson, tc.resultJson].filter(Boolean).join("\n\n---\n\n")} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />}
              {!tc.argsJson && !tc.resultJson && <span className="ml-auto" />}
              <span className="text-xs text-muted-foreground shrink-0">
                {isExpanded ? "▼" : "▶"}
              </span>
            </span>

            {isExpanded && (
              <div className="px-2 pb-2 space-y-2">
                {tc.errorMessage && (
                  <div className="text-xs text-red-600 dark:text-red-400 p-1.5 bg-red-50/50 dark:bg-red-500/10 rounded">
                    {tc.errorMessage}
                  </div>
                )}

                {tc.argsJson && (
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="text-xs font-medium text-muted-foreground">Args</span>
                      <CopyButton text={tc.argsJson} className="size-4 text-muted-foreground hover:text-foreground" />
                    </div>
                    <pre className="text-xs whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto bg-muted/30 p-1.5 rounded">
                      {truncateJson(tc.argsJson, 2000) ?? tc.argsJson}
                    </pre>
                  </div>
                )}

                {tc.resultJson && (
                  <div>
                    <div className="flex items-center gap-1.5 mb-1">
                      <span className="text-xs font-medium text-muted-foreground">Result</span>
                      <CopyButton text={tc.resultJson} className="size-4 text-muted-foreground hover:text-foreground" />
                    </div>
                    <pre className="text-xs whitespace-pre-wrap break-words max-h-[200px] overflow-y-auto bg-muted/30 p-1.5 rounded">
                      {truncateJson(tc.resultJson, 2000) ?? tc.resultJson}
                    </pre>
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
