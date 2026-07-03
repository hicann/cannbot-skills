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
import { type SkillEventItem, type SkillToolCallItem, type SkillGroup, extractSkillNameFromArgs, groupSkillEvents } from "@/lib/skill-event-grouping"
import { CopyButton } from "./CopyButton"

const EVENT_TYPE_BADGE_VARIANTS: Record<string, "blue" | "green" | "orange" | "gray"> = {
  load: "blue",
  invoke: "green",
  use: "green",
  dispatch: "orange",
  unload: "gray",
}

function formatDuration(ms: number): string {
  if (ms === 0) return ""
  if (ms < 1000) return `${ms}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

function formatTokenCount(n: number): string {
  if (n === 0) return ""
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

interface SkillEventListProps {
  skillEvents: SkillEventItem[]
  skillToolCalls?: SkillToolCallItem[]
}

function renderJsonSection(label: string, json: string | null) {
  if (!json) return null

  let displayText = json
  try {
    const parsed = JSON.parse(json)
    displayText = JSON.stringify(parsed, null, 2)
  } catch {
    displayText = json
  }

  return (
    <div>
      <div className="flex items-center gap-1.5 mb-1">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <CopyButton text={displayText} className="size-4 text-muted-foreground hover:text-foreground" />
      </div>
      <pre className="text-xs whitespace-pre-wrap break-words overflow-y-auto bg-muted/30 p-1.5 rounded max-h-[300px]">
        {displayText}
      </pre>
    </div>
  )
}

export function SkillEventList({ skillEvents, skillToolCalls }: SkillEventListProps) {
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  function toggleExpanded(id: string) {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  const groups = groupSkillEvents(skillEvents, skillToolCalls)

  return (
    <div className="space-y-2">
      {groups.map(group => {
        const isExpanded = expandedIds.has(group.compositeId)
        const hasLifecycle = group.loadEvent && group.invokeEvent
        const invokeErrorMsg = group.invokeEvent?.errorMessage ?? group.otherEvents.find(e => e.errorMessage)?.errorMessage ?? null

        return (
          <div key={group.compositeId} className="border rounded-md">
            <button
              className="w-full flex items-center gap-2 px-2 py-1.5 text-sm cursor-pointer hover:bg-accent/30 transition-colors"
              onClick={() => toggleExpanded(group.compositeId)}
            >
              <span className="font-medium">{group.skillName}</span>
              {group.skillVersion != null && (
                <span className="text-xs text-muted-foreground">v{group.skillVersion}</span>
              )}
              {hasLifecycle ? (
                <Badge variant="blue">load → invoke</Badge>
              ) : group.loadEvent ? (
                <Badge variant="blue">load</Badge>
              ) : group.invokeEvent ? (
                <Badge variant="green">invoke</Badge>
              ) : group.dispatchEvent ? (
                <Badge variant="orange">dispatch</Badge>
              ) : (
                group.otherEvents.map(se => (
                  <Badge key={se.id} variant={EVENT_TYPE_BADGE_VARIANTS[se.eventType] ?? "gray"}>
                    {se.eventType}
                  </Badge>
                ))
              )}
              <Badge variant={group.allSuccess ? "green" : "red"}>
                {group.allSuccess ? "success" : "fail"}
              </Badge>
              {(() => {
                const matchingTcs = skillToolCalls ? skillToolCalls.filter(tc => {
                  const sn = extractSkillNameFromArgs(tc.argsJson)
                  return sn === group.skillName
                }) : []
                const overhead = Math.round(matchingTcs.reduce(function (s, tc) { return s + (tc.argsJson?.length ?? 0) + (tc.resultJson?.length ?? 0); }, 0) / 3.5)
                return overhead > 0 ? (
                  <span className="text-xs text-muted-foreground tabular-nums">{formatTokenCount(overhead)}t</span>
                ) : null
              })()}
              <span className="text-xs text-muted-foreground ml-auto">
                {isExpanded ? "▼" : "▶"}
              </span>
            </button>

            {isExpanded && (
              <div className="px-2 pb-2 space-y-2">
                {invokeErrorMsg && (
                  <div className="text-xs text-red-600 dark:text-red-400 p-1.5 bg-red-50/50 dark:bg-red-500/10 rounded">
                    {invokeErrorMsg}
                  </div>
                )}

                {renderJsonSection("Load Args:", group.loadArgsJson)}
                {renderJsonSection("Invoke Args:", group.invokeArgsJson)}
                {renderJsonSection("Result:", group.invokeResultJson)}
                {renderJsonSection("Dispatch Args:", group.dispatchArgsJson)}
                {renderJsonSection("Dispatch Result:", group.dispatchResultJson)}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
