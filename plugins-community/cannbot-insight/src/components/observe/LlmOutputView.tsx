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
import { cn } from "@/lib/utils"
import { CopyButton } from "./CopyButton"
import { highlightKeyword, type TurnHighlight } from "@/lib/shared/highlight"

interface LlmOutputViewProps {
  content: string | null
  contentJson: string | null
  contentSummary: string | null
  outputTokens: number
  reasoningTokens?: number
  role?: string | null
  highlight?: TurnHighlight | null
}

function formatTokenCount(n: number): string {
  if (n === 0) return ""
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`
  return `${n}`
}

interface ContentSection {
  type: "thinking" | "text"
  content: string
}

function parseContentSections(content: string | null, contentJson: string | null): ContentSection[] {
  const sections: ContentSection[] = []

  let rawContent = content ?? ""

  if (contentJson) {
    try {
      const parsed = JSON.parse(contentJson)
      if (typeof parsed === "string") {
        rawContent = parsed
      } else if (Array.isArray(parsed)) {
        for (const block of parsed) {
          if (block.type === "thinking" || block.type === "reasoning") {
            sections.push({ type: "thinking", content: block.content ?? block.thinking ?? String(block) })
          } else if (block.type === "text") {
            sections.push({ type: "text", content: block.content ?? String(block) })
          }
        }
        if (sections.length > 0) return sections
      } else if (typeof parsed === "object" && parsed.content) {
        rawContent = parsed.content
      }
    } catch {
      rawContent = content ?? ""
    }
  }

  if (sections.length === 0 && rawContent) {
    const thinkingPattern = new RegExp("<thinking>(.*?)<\\/thinking>", "gs")
    const matches = [...rawContent.matchAll(thinkingPattern)]

    if (matches.length > 0) {
      let remaining = rawContent
      for (const match of matches) {
        sections.push({ type: "thinking", content: match[1] })
        remaining = remaining.replace(match[0], "")
      }
      const trimmed = remaining.trim()
      if (trimmed) {
        sections.push({ type: "text", content: trimmed })
      }
    } else {
      sections.push({ type: "text", content: rawContent })
    }
  }

  return sections
}

export function LlmOutputView({
  content,
  contentJson,
  contentSummary,
  outputTokens,
  reasoningTokens,
  role,
  highlight,
}: LlmOutputViewProps) {
  const [isExpanded, setIsExpanded] = useState(false)
  const [isThinkingExpanded, setIsThinkingExpanded] = useState(false)
  const [prevKw, setPrevKw] = useState<string | null>(null)
  const rootRef = useRef<HTMLDivElement>(null)

  const sections = parseContentSections(content, contentJson)

  // Trace-search jump: when the highlight keyword changes, auto-expand so the
  // highlighted match is visible. Uses the render-phase prop-change pattern
  // (https://react.dev/learn/you-might-not-need-an-effect#adjusting-some-state-when-a-prop-changes)
  // instead of setState-in-effect.
  const currentKw = highlight?.keyword ?? null
  if (currentKw !== prevKw) {
    setPrevKw(currentKw)
    if (currentKw && highlight) {
      const isContentMatch = !highlight.matchField || highlight.matchField === "content" || highlight.matchField === "contentSummary"
      if (isContentMatch) {
        setIsExpanded(true)
        const lower = currentKw.trim().toLowerCase()
        if (sections.some(s => s.type === "thinking" && s.content.toLowerCase().includes(lower))) {
          setIsThinkingExpanded(true)
        }
      }
    }
  }

  // Scroll the first highlighted match into view once expanded.
  useEffect(() => {
    if (!highlight?.keyword || !isExpanded) return
    const t = setTimeout(() => {
      rootRef.current?.querySelector("mark")?.scrollIntoView({ behavior: "smooth", block: "center" })
    }, 60)
    return () => clearTimeout(t)
  }, [highlight, isExpanded])

  if (!content && !contentJson) return null

  const summary = contentSummary ?? (content ? (content.length > 200 ? content.substring(0, 200) + "..." : content) : null)

  const HEADER_LABELS: Record<string, string> = {
    user: "User Input",
    assistant: "LLM Output",
    system: "System Message",
    tool_result: "Tool Result",
  }
  const headerLabel = HEADER_LABELS[role ?? ""] ?? "Content"

  return (
    <div ref={rootRef} className="border rounded-lg">
      <span
        role="button"
        tabIndex={0}
        className="w-full flex items-center justify-between px-3 py-2 hover:bg-accent/50 transition-colors cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsExpanded(!isExpanded) }}
      >
        <div className="flex items-center gap-2 text-sm">
          <span className="font-medium">{headerLabel}</span>
          {summary && (
            <span className="text-muted-foreground truncate max-w-[400px]">{summary}</span>
          )}
          {((outputTokens + (reasoningTokens ?? 0)) > 0) && (
            <Badge variant="outline">
              {(reasoningTokens ?? 0) > 0
                ? `${formatTokenCount(reasoningTokens ?? 0)}t reasoning + ${formatTokenCount(outputTokens)}t output`
                : `${formatTokenCount(outputTokens)} tokens`
              }
            </Badge>
          )}
        </div>
        <span className="text-xs text-muted-foreground">
          {isExpanded ? "▼" : "▶"}
        </span>
      </span>

      {isExpanded && sections.length > 0 && (
        <div className="border-t px-3 py-2 space-y-3">
          {sections.map((section, index) => {
            if (section.type === "thinking") {
              return (
                <div key={index} className="border rounded-md">
                  <span
                    role="button"
                    tabIndex={0}
                    className="w-full flex items-center gap-2 px-2 py-1.5 text-sm cursor-pointer hover:bg-accent/30 transition-colors"
                    onClick={() => setIsThinkingExpanded(!isThinkingExpanded)}
                    onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') setIsThinkingExpanded(!isThinkingExpanded) }}
                  >
                    <Badge variant="purple">thinking</Badge>
                    {(reasoningTokens ?? 0) > 0 && (
                      <span className="text-muted-foreground text-xs">{formatTokenCount(reasoningTokens ?? 0)}t</span>
                    )}
                    <span className="text-muted-foreground text-xs">
                      {isThinkingExpanded ? "▼ hide" : "▶ show"}
                    </span>
                    <CopyButton text={section.content} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />
                  </span>
                  {isThinkingExpanded && (
                    <div className="px-2 pb-2 text-sm whitespace-pre-wrap break-words max-h-[400px] overflow-y-auto bg-purple-50/30 dark:bg-purple-500/5">
                      {highlightKeyword(section.content, highlight?.keyword)}
                    </div>
                  )}
                </div>
              )
            }

            return (
              <div key={index} className="border rounded-md overflow-hidden">
                <div className="flex items-center gap-2 px-2 py-1.5 bg-emerald-50/30 dark:bg-emerald-500/5">
                  <Badge variant="green">text</Badge>
                  {outputTokens > 0 && (
                    <span className="text-muted-foreground text-xs">{formatTokenCount(outputTokens)}t</span>
                  )}
                  <CopyButton text={section.content} className="ml-auto size-4 text-muted-foreground hover:text-foreground" />
                </div>
                <div className="px-2 pb-2 text-sm whitespace-pre-wrap break-words max-h-[600px] overflow-y-auto">
                  {highlightKeyword(section.content, highlight?.keyword)}
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
