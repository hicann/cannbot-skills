"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useMemo, useEffect, useRef, useCallback } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"
import { isCommandTurn, isCommandCaveat, isCommandStdout, parseCommandTurns, formatCommandDisplay, isContinuationTurn, parseContinuationTurn } from "@/lib/shared/command-parser"

type ChatReplayTurn = {
  turnId: string
  turnIndex: number
  role: string
  contentSummary: string | null
  content: string | null
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  totalTokens: number
  inputTokens: number
  inputMessagesCount: number
  inputMessagesTokens: number
  outputTokens: number
  latencyMs: number
  createdAt: string | null
  model: string | null
  toolCalls: Array<{
    toolCallId: string
    toolName: string
    argsJson: string | null
    resultJson: string | null
    state: string
    durationMs: number
  }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean }>
}

interface ChatReplayViewProps {
  turns: ChatReplayTurn[]
  sessionModel: string | null
  onNavigateToTurn: (turnId: string) => void
}

// Each message assigned to a panel: "left" (User↔Assistant) or "right" (dispatch↔LLM)
type ChatMessage = {
  id: string
  type: "user" | "assistant_tool" | "dispatch" | "continue" | "llm" | "llm_empty" | "command" | "continuation"
  panel: "left" | "right"
  role: string
  content: string | null
  agentName: string | null
  isSubagent: boolean
  subagentName: string | null
  model: string | null
  // command-specific
  commandName?: string
  commandOutput?: string | null
  // dispatch-specific
  inputMessagesTokens?: number
  inputMessagesCount?: number
  // tool call-specific
  toolName?: string
  toolArgs?: string | null
  toolResult?: string | null
  toolState?: string
  toolDuration?: number
  turnIndex: number
  turnId: string
  tokens: number
  latencyMs: number
  createdAt: string | null
}

const SPEED_OPTIONS: Array<{ value: number; label: string }> = [
  { value: 1, label: "1x" },
  { value: 2, label: "2x" },
  { value: 4, label: "4x" },
]

const BASE_DELAY_MS = 300
const MAX_CONTENT_LEN = 200
const MAX_TOOL_RESULT_LEN = 300

function formatTokenCount(n: number): string {
  if (n === 0) return "0"
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`
  if (n >= 1000) return `${(n / 1000).toFixed(1)}K`
  return `${n}`
}

function formatLatency(ms: number): string {
  if (ms === 0) return "0ms"
  if (ms < 1000) return `${ms}ms`
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60000).toFixed(1)}min`
}

function truncateText(text: string | null, maxLen: number): string | null {
  if (!text) return null
  if (text.length <= maxLen) return text
  return text.substring(0, maxLen) + "..."
}

function extractToolArgSummary(argsJson: string | null): string | null {
  if (!argsJson) return null
  try {
    const args = JSON.parse(argsJson)
    if (typeof args === "string") return truncateText(args, 60)
    if (typeof args.command === "string") return truncateText(args.command, 60)
    if (typeof args.cmd === "string") return truncateText(args.cmd, 60)
    const entries = Object.entries(args)
    if (entries.length === 0) return null
    const firstVal = entries[0][1]
    if (typeof firstVal === "string") return truncateText(firstVal, 60)
    return `${entries.length} params`
  } catch {
    return truncateText(argsJson, 60)
  }
}

const TOOL_ICON_MAP: Record<string, string> = {
  Bash: "⌨", Read: "📖", Write: "📝", Edit: "✏️",
  Glob: "🔍", Grep: "🔎", WebFetch: "🌐", WebSearch: "🔎", Agent: "🤖",
}

function getToolIcon(name: string): string {
  return TOOL_ICON_MAP[name] ?? "🔧"
}

// Each assistant turn → left panel: tool calls; right panel: dispatch + LLM text
// user turn → left panel: user message (or command)
function groupTurnsIntoMessages(turns: ChatReplayTurn[]): ChatMessage[] {
  const messages: ChatMessage[] = []
  let lastRole: string | null = null
  // Buffer for grouping consecutive command turns
  let commandBuffer: { turn: ChatReplayTurn; text: string }[] = []

  function flushCommandBuffer() {
    if (commandBuffer.length === 0) return
    const texts = commandBuffer.map(b => b.text)
    const info = parseCommandTurns(texts)
    const display = formatCommandDisplay(info)
    const first = commandBuffer[0]
    messages.push({
      id: `cmd-${first.turn.turnId}`,
      type: "command",
      panel: "left",
      role: "user",
      content: display,
      commandName: info.name,
      commandOutput: info.output,
      agentName: first.turn.agentName ?? null,
      isSubagent: first.turn.isSubagent,
      subagentName: first.turn.subagentName ?? null,
      model: null,
      turnIndex: first.turn.turnIndex,
      turnId: first.turn.turnId,
      tokens: commandBuffer.reduce((s, b) => s + b.turn.totalTokens, 0),
      latencyMs: commandBuffer.reduce((s, b) => s + b.turn.latencyMs, 0),
      createdAt: first.turn.createdAt ?? null,
    })
    lastRole = "user"
    commandBuffer = []
  }

  for (const turn of turns) {
    const text = turn.contentSummary ?? turn.content ?? ""

    // Only user turns can be command messages — assistant turns reference them in prose
    const isCommand = turn.role === "user" && isCommandTurn(text)
    const isCaveat = turn.role === "user" && isCommandCaveat(text)
    const isStdout = turn.role === "user" && isCommandStdout(text)

    if (isCommand || isCaveat || isStdout) {
      commandBuffer.push({ turn, text })
      continue
    }

    // Non-command turn: flush any buffered command turns first
    flushCommandBuffer()

    if (turn.role === "user" && isContinuationTurn(text)) {
      const info = parseContinuationTurn(text)
      messages.push({
        id: `continuation-${turn.turnId}`,
        type: "continuation",
        panel: "left",
        role: "user",
        content: info.summaryLine ?? "Compact summary",
        agentName: turn.agentName ?? null,
        isSubagent: turn.isSubagent,
        subagentName: turn.subagentName ?? null,
        model: null,
        turnIndex: turn.turnIndex,
        turnId: turn.turnId,
        tokens: turn.totalTokens,
        latencyMs: turn.latencyMs,
        createdAt: turn.createdAt ?? null,
      })
      lastRole = "user"
    } else if (turn.role === "user") {
      messages.push({
        id: `user-${turn.turnId}`,
        type: "user",
        panel: "left",
        role: "user",
        content: text || "User message",
        agentName: turn.agentName ?? null,
        isSubagent: turn.isSubagent,
        subagentName: turn.subagentName ?? null,
        model: null,
        turnIndex: turn.turnIndex,
        turnId: turn.turnId,
        tokens: turn.totalTokens,
        latencyMs: turn.latencyMs,
        createdAt: turn.createdAt ?? null,
      })
      lastRole = "user"
    } else if (turn.role === "assistant") {
      const isSub = turn.isSubagent
      const hasText = (turn.contentSummary ?? turn.content ?? "").length > 0
      const ctxTokens = turn.inputMessagesTokens > 0 ? turn.inputMessagesTokens : turn.inputTokens
      const isFirst = lastRole !== "assistant"

      // --- Right panel: dispatch/continue ---
      if (isFirst) {
        messages.push({
          id: `dispatch-${turn.turnId}`,
          type: "dispatch",
          panel: "right",
          role: "dispatch",
          content: `→ ${turn.model ?? "LLM"}: forwarding ${formatTokenCount(ctxTokens)} context (${turn.inputMessagesCount} messages)`,
          agentName: turn.agentName ?? "Assistant",
          isSubagent: isSub,
          subagentName: turn.subagentName ?? null,
          model: turn.model ?? null,
          inputMessagesTokens: ctxTokens,
          inputMessagesCount: turn.inputMessagesCount,
          turnIndex: turn.turnIndex,
          turnId: turn.turnId,
          tokens: ctxTokens,
          latencyMs: 0,
          createdAt: turn.createdAt ?? null,
        })
      } else {
        messages.push({
          id: `continue-${turn.turnId}`,
          type: "continue",
          panel: "right",
          role: "dispatch",
          content: `→ continue (${formatTokenCount(ctxTokens)} ctx)`,
          agentName: turn.agentName ?? "Assistant",
          isSubagent: isSub,
          subagentName: turn.subagentName ?? null,
          model: turn.model ?? null,
          inputMessagesTokens: ctxTokens,
          inputMessagesCount: turn.inputMessagesCount,
          turnIndex: turn.turnIndex,
          turnId: turn.turnId,
          tokens: ctxTokens,
          latencyMs: 0,
          createdAt: turn.createdAt ?? null,
        })
      }

      // --- Right panel: LLM text ---
      if (hasText) {
        messages.push({
          id: `llm-${turn.turnId}`,
          type: "llm",
          panel: "right",
          role: "assistant",
          content: turn.contentSummary ?? turn.content ?? "...",
          agentName: turn.agentName ?? "Assistant",
          isSubagent: isSub,
          subagentName: turn.subagentName ?? null,
          model: turn.model ?? null,
          turnIndex: turn.turnIndex,
          turnId: turn.turnId,
          tokens: turn.totalTokens,
          latencyMs: turn.latencyMs,
          createdAt: turn.createdAt ?? null,
        })
      } else if (turn.toolCalls.length === 0) {
        messages.push({
          id: `llm-empty-${turn.turnId}`,
          type: "llm_empty",
          panel: "right",
          role: "assistant",
          content: "...",
          agentName: turn.agentName ?? "Assistant",
          isSubagent: isSub,
          subagentName: turn.subagentName ?? null,
          model: turn.model ?? null,
          turnIndex: turn.turnIndex,
          turnId: turn.turnId,
          tokens: turn.totalTokens,
          latencyMs: turn.latencyMs,
          createdAt: turn.createdAt ?? null,
        })
      }

      // --- Left panel: tool calls ---
      for (const tc of turn.toolCalls) {
        messages.push({
          id: `tool-${tc.toolCallId}`,
          type: "assistant_tool",
          panel: "left",
          role: "tool",
          content: null,
          agentName: turn.agentName ?? "Assistant",
          isSubagent: isSub,
          subagentName: turn.subagentName ?? null,
          model: null,
          toolName: tc.toolName,
          toolArgs: extractToolArgSummary(tc.argsJson),
          toolResult: truncateText(tc.resultJson, MAX_TOOL_RESULT_LEN),
          toolState: tc.state,
          toolDuration: tc.durationMs,
          turnIndex: turn.turnIndex,
          turnId: turn.turnId,
          tokens: turn.totalTokens,
          latencyMs: turn.latencyMs,
          createdAt: turn.createdAt ?? null,
        })
      }

      lastRole = "assistant"
    } else {
      lastRole = turn.role
    }
  }

  flushCommandBuffer()

  return messages
}

// --- Sub-components for each panel ---

function ThinkingDots({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-1 px-3 py-1.5">
      <span className="text-xs text-muted-foreground">{label}</span>
      <div className="flex gap-0.5">
        {[0, 1, 2].map(i => (
          <span key={i} className="inline-block w-1.5 h-1.5 rounded-full bg-muted-foreground/60"
            style={{ animation: `chat-replay-dot-bounce 1.4s ease-in-out ${i * 0.2}s infinite` }}
          />
        ))}
      </div>
    </div>
  )
}

function LeftPanelBubble({ msg, expanded, onToggleExpand, onNavigate }: {
  msg: ChatMessage; expanded: boolean; onToggleExpand: (id: string) => void; onNavigate: (id: string) => void
}) {
  if (msg.type === "user") {
    // User: left-aligned, blue
    return (
      <div className="flex gap-2 items-start">
        <div className="shrink-0 w-6 h-6 rounded-full bg-blue-500 text-white flex items-center justify-center text-xs">👤</div>
        <div className="max-w-[85%]">
          <span className="text-xs text-muted-foreground">User</span>
          <div className="rounded-2xl rounded-bl-sm px-3 py-2 text-sm bg-blue-500 text-white cursor-pointer whitespace-pre-wrap break-words"
            onClick={() => onNavigate(msg.turnId)}>
            {truncateText(msg.content ?? "", MAX_CONTENT_LEN) ?? "User message"}
          </div>
          {(msg.content ?? "").length > MAX_CONTENT_LEN && !expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Show full ({(msg.content ?? "").length}c)</button>
          )}
          {(msg.content ?? "").length > MAX_CONTENT_LEN && expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Collapse</button>
          )}
        </div>
      </div>
    )
  }

  if (msg.type === "command") {
    return (
      <div className="flex gap-2 items-start">
        <div className="shrink-0 w-6 h-6 rounded-full bg-gray-500 text-white flex items-center justify-center text-xs">⚡</div>
        <div className="max-w-[85%]">
          <span className="text-xs text-muted-foreground">Command</span>
          <div className="rounded-xl px-3 py-2 text-sm bg-gray-100 dark:bg-gray-800/40 text-foreground cursor-pointer"
            onClick={() => onNavigate(msg.turnId)}>
            <span className="font-medium font-mono">{msg.commandName ?? msg.content}</span>
            {msg.content && msg.content !== msg.commandName && (
              <span className="ml-1 text-xs opacity-70 font-mono">{msg.content.replace(msg.commandName ?? "", "").trim()}</span>
            )}
            {msg.commandOutput && (
              <div className="text-xs opacity-60 mt-1 border-t border-current/10 pt-1 whitespace-pre-wrap">{msg.commandOutput}</div>
            )}
          </div>
        </div>
      </div>
    )
  }

  if (msg.type === "continuation") {
    return (
      <div className="flex gap-2 items-start">
        <div className="shrink-0 w-6 h-6 rounded-full bg-purple-500 text-white flex items-center justify-center text-xs">📋</div>
        <div className="max-w-[85%]">
          <span className="text-xs text-muted-foreground">Compact summary</span>
          <div className="rounded-xl px-3 py-2 text-sm bg-purple-100 dark:bg-purple-900/30 text-foreground cursor-pointer"
            onClick={() => onNavigate(msg.turnId)}>
            <span className="italic">{truncateText(msg.content ?? "", MAX_CONTENT_LEN)}</span>
          </div>
          {(msg.content ?? "").length > MAX_CONTENT_LEN && !expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Show full ({(msg.content ?? "").length}c)</button>
          )}
          {(msg.content ?? "").length > MAX_CONTENT_LEN && expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Collapse</button>
          )}
        </div>
      </div>
    )
  }

  if (msg.type === "assistant_tool") {
    // Assistant tool call: right-aligned, orange
    return (
      <div className="flex gap-2 items-start flex-row-reverse">
        <div className="shrink-0 w-6 h-6 rounded-full bg-orange-500 text-white flex items-center justify-center text-xs">
          {getToolIcon(msg.toolName ?? "")}
        </div>
        <div className="max-w-[85%] flex flex-col items-end">
          <span className="text-xs text-muted-foreground">
            {msg.isSubagent ? `${msg.subagentName ?? "subagent"}` : `${msg.agentName ?? "Assistant"}`} · Tool
          </span>
          <div className="rounded-2xl rounded-br-sm px-3 py-2 text-sm bg-orange-100 dark:bg-orange-900/20 text-foreground cursor-pointer"
            onClick={() => onNavigate(msg.turnId)}>
            <div className="space-y-1">
              <div className="flex items-center gap-1.5 font-medium">
                <span>{getToolIcon(msg.toolName ?? "")}</span>
                <span>{msg.toolName}</span>
                {(msg.toolDuration ?? 0) > 0 && <span className="text-xs opacity-60">{formatLatency(msg.toolDuration ?? 0)}</span>}
                {msg.toolState === "error" && <Badge variant="red" className="text-xs">error</Badge>}
              </div>
              {msg.toolArgs && <div className="text-xs opacity-80 font-mono break-all">{msg.toolArgs}</div>}
              {msg.toolResult && (
                <div className="text-xs opacity-70 font-mono break-all border-t border-current/10 pt-1 mt-1">{msg.toolResult}</div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return null
}

function RightPanelBubble({ msg, expanded, onToggleExpand, onNavigate }: {
  msg: ChatMessage; expanded: boolean; onToggleExpand: (id: string) => void; onNavigate: (id: string) => void
}) {
  if (msg.type === "dispatch" || msg.type === "continue") {
    // Dispatch/Continue: left-aligned, orange, compact
    return (
      <div className="flex gap-2 items-start">
        <div className="shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs"
          style={{ backgroundColor: msg.isSubagent ? "#8b5cf6" : "#f97316" }}>→</div>
        <div className="max-w-[85%]">
          <span className="text-xs text-muted-foreground">
            {msg.isSubagent ? `${msg.subagentName ?? "subagent"} → LLM` : `${msg.agentName ?? "Assistant"} → LLM`}
            {msg.model && <span className="ml-1 opacity-60 font-mono">{msg.model}</span>}
          </span>
          <div className="rounded-xl px-2.5 py-1.5 text-xs bg-orange-100 dark:bg-orange-900/20 text-foreground cursor-pointer"
            onClick={() => onNavigate(msg.turnId)}>
            {msg.content}
          </div>
        </div>
      </div>
    )
  }

  if (msg.type === "llm" || msg.type === "llm_empty") {
    // LLM: right-aligned, green
    const isSub = msg.isSubagent
    const bubbleBg = isSub ? "bg-purple-100 dark:bg-purple-900/30" : "bg-emerald-100 dark:bg-emerald-900/30"
    const avatarBg = isSub ? "#8b5cf6" : "#10b981"
    const displayContent = truncateText(msg.content ?? "", MAX_CONTENT_LEN) ?? "..."
    const needsExpand = (msg.content ?? "").length > MAX_CONTENT_LEN && msg.type === "llm"

    return (
      <div className="flex gap-2 items-start flex-row-reverse">
        <div className="shrink-0 w-6 h-6 rounded-full text-white flex items-center justify-center text-xs"
          style={{ backgroundColor: avatarBg }}>🧠</div>
        <div className="max-w-[85%] flex flex-col items-end">
          <span className="text-xs text-muted-foreground">
            {isSub ? `${msg.subagentName ?? "subagent"} · LLM` : `LLM (${msg.agentName ?? "root"})`}
            {msg.model && <span className="ml-1 opacity-60 font-mono">{msg.model}</span>}
          </span>
          <div className={cn("rounded-2xl rounded-br-sm px-3 py-2 text-sm cursor-pointer whitespace-pre-wrap break-words", bubbleBg, "text-foreground")}
            onClick={() => onNavigate(msg.turnId)}>
            {needsExpand && !expanded ? displayContent : msg.content ?? "..."}
          </div>
          {needsExpand && !expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Show full ({(msg.content ?? "").length}c)</button>
          )}
          {needsExpand && expanded && (
            <button className="text-xs text-muted-foreground hover:text-foreground cursor-pointer"
              onClick={() => onToggleExpand(msg.id)}>Collapse</button>
          )}
        </div>
      </div>
    )
  }

  return null
}

// --- Main component with split-panel layout ---

export function ChatReplayView({ turns, sessionModel, onNavigateToTurn }: ChatReplayViewProps) {
  const messages = useMemo(() => groupTurnsIntoMessages(turns), [turns])
  const totalMessages = messages.length

  const [currentIndex, setCurrentIndex] = useState(-1)
  const [isPlaying, setIsPlaying] = useState(false)
  const [speed, setSpeed] = useState(1)
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set())

  const animRef = useRef<number | null>(null)
  const lastTickRef = useRef<number>(0)
  const leftScrollRef = useRef<HTMLDivElement>(null)
  const rightScrollRef = useRef<HTMLDivElement>(null)

  const toggleExpand = useCallback((id: string) => {
    setExpandedIds(prev => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }, [])

  // Derive "thinking" state from current animation position
  const showThinking = useMemo(() => {
    if (!isPlaying || currentIndex < 0 || currentIndex >= totalMessages - 1) return null
    const nextMsg = messages[currentIndex + 1]
    if (nextMsg?.panel === "right" && (nextMsg.type === "dispatch" || nextMsg.type === "continue")) return "dispatch"
    if (nextMsg?.panel === "right" && (nextMsg.type === "llm" || nextMsg.type === "llm_empty")) return "llm"
    return null
  }, [isPlaying, currentIndex, totalMessages, messages])

  // Animation tick with adaptive delay
  useEffect(() => {
    if (!isPlaying) {
      if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null }
      return
    }

    const nextIdx = currentIndex + 1
    const nextMsg = nextIdx < totalMessages ? messages[nextIdx] : null
    const isLlm = nextMsg?.type === "llm" || nextMsg?.type === "llm_empty"
    const isDispatch = nextMsg?.type === "dispatch" || nextMsg?.type === "continue"
    const delay = isLlm ? (2 * BASE_DELAY_MS) / speed
      : isDispatch ? (BASE_DELAY_MS / 2) / speed
      : BASE_DELAY_MS / speed

    function tick(now: number) {
      if (now - lastTickRef.current >= delay) {
        lastTickRef.current = now
        setCurrentIndex(prev => {
          const next = prev + 1
          if (next >= totalMessages - 1) setTimeout(() => setIsPlaying(false), 0)
          return next
        })
      }
      animRef.current = requestAnimationFrame(tick)
    }

    lastTickRef.current = performance.now()
    animRef.current = requestAnimationFrame(tick)
    return () => { if (animRef.current) { cancelAnimationFrame(animRef.current); animRef.current = null } }
  }, [isPlaying, speed, currentIndex, totalMessages, messages])

  // Auto-scroll both panels
  useEffect(() => {
    if (leftScrollRef.current) leftScrollRef.current.scrollTop = leftScrollRef.current.scrollHeight
    if (rightScrollRef.current) rightScrollRef.current.scrollTop = rightScrollRef.current.scrollHeight
  }, [currentIndex, showThinking])

  function handlePlay() {
    if (currentIndex >= totalMessages - 1) setCurrentIndex(-1)
    setIsPlaying(true)
  }
  function handlePause() { setIsPlaying(false) }
  function handleReset() { setIsPlaying(false); setCurrentIndex(-1); setExpandedIds(new Set()) }

  const visibleMessages = messages.slice(0, currentIndex + 1)
  const leftMessages = visibleMessages.filter(m => m.panel === "left")
  const rightMessages = visibleMessages.filter(m => m.panel === "right")

  const currentMsg = currentIndex >= 0 ? messages[currentIndex] : null
  const currentStats = currentMsg ? (() => {
    const relatedTurns = turns.filter(t => t.turnId === currentMsg.turnId)
    const totalT = relatedTurns.reduce((s, t) => s + t.totalTokens, 0)
    const totalIn = relatedTurns.reduce((s, t) => s + t.inputTokens, 0)
    const totalOut = relatedTurns.reduce((s, t) => s + t.outputTokens, 0)
    const totalLat = relatedTurns.reduce((s, t) => s + t.latencyMs, 0)
    return { tokens: totalT, input: totalIn, output: totalOut, latency: totalLat }
  })() : null

  // Thinking animation position (in right panel)
  const thinkingLabel = showThinking === "llm" ? "LLM thinking" : showThinking === "dispatch" ? "Forwarding" : null
  const thinkingAvatarBg = showThinking === "llm" ? "#10b981" : showThinking === "dispatch" ? "#f97316" : null
  const thinkingIcon = showThinking === "llm" ? "🧠" : showThinking === "dispatch" ? "→" : null
  const thinkingAlign = showThinking === "llm" ? "flex-row-reverse" : "flex-row"

  if (totalMessages === 0) {
    return <div className="flex items-center justify-center h-full text-muted-foreground text-sm">No turns data available for replay</div>
  }

  return (
    <div className="flex flex-col h-full min-h-0">
      {/* Control bar */}
      <div className="shrink-0 border-b px-3 py-2 flex items-center gap-2 bg-background">
        <Button size="sm" variant="outline" className="h-7 text-xs gap-1" onClick={isPlaying ? handlePause : handlePlay}>
          {isPlaying ? "⏸ Pause" : "▶ Play"}
        </Button>
        <Button size="sm" variant="outline" className="h-7 text-xs" onClick={handleReset}>↺</Button>
        <div className="flex items-center gap-1">
          {SPEED_OPTIONS.map(opt => (
            <Button key={opt.value} size="sm" variant={speed === opt.value ? "default" : "outline"} className="h-7 text-xs px-2"
              onClick={() => setSpeed(opt.value)}>{opt.label}</Button>
          ))}
        </div>
        <div className="flex-1 flex items-center gap-2">
          <span className="text-xs text-muted-foreground tabular-nums shrink-0">
            {currentIndex >= 0 ? `#${currentIndex + 1}` : "—"} / {totalMessages}
          </span>
          <input type="range" min={-1} max={totalMessages - 1} value={currentIndex}
            onChange={(e) => { setIsPlaying(false); setCurrentIndex(Number(e.target.value)) }}
            className="flex-1 h-1.5 accent-emerald-500" />
        </div>
      </div>

      {/* Split-panel chat area */}
      <div className="flex-1 min-h-0 flex gap-0">
        {/* Left panel: User ↔ Assistant */}
        <div className="w-1/2 border-r flex flex-col min-h-0">
          <div className="shrink-0 px-3 py-1.5 border-b bg-muted/30 text-xs text-muted-foreground text-center font-medium">
            User ↔ Assistant
          </div>
          <div ref={leftScrollRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-2 space-y-2">
            {leftMessages.map(msg => (
              <LeftPanelBubble key={msg.id} msg={msg} expanded={expandedIds.has(msg.id)}
                onToggleExpand={toggleExpand} onNavigate={onNavigateToTurn} />
            ))}
            {currentIndex < 0 && !isPlaying && (
              <div className="flex items-center justify-center h-full min-h-[150px] text-muted-foreground text-xs">
                Click ▶ Play
              </div>
            )}
          </div>
        </div>

        {/* Right panel: dispatch ↔ LLM */}
        <div className="w-1/2 flex flex-col min-h-0">
          <div className="shrink-0 px-3 py-1.5 border-b bg-muted/30 text-xs text-muted-foreground text-center font-medium">
            Assistant ↔ LLM
          </div>
          <div ref={rightScrollRef} className="flex-1 min-h-0 overflow-y-auto px-3 py-2 space-y-2">
            {rightMessages.map(msg => (
              <RightPanelBubble key={msg.id} msg={msg} expanded={expandedIds.has(msg.id)}
                onToggleExpand={toggleExpand} onNavigate={onNavigateToTurn} />
            ))}
            {thinkingLabel && (
              <div className={cn("flex gap-2 items-start", thinkingAlign)}>
                <div className="shrink-0 w-6 h-6 rounded-full text-white flex items-center justify-center text-xs"
                  style={{ backgroundColor: thinkingAvatarBg ?? "#6b7280" }}>{thinkingIcon}</div>
                <ThinkingDots label={thinkingLabel} />
              </div>
            )}
            {currentIndex < 0 && !isPlaying && (
              <div className="flex items-center justify-center h-full min-h-[150px] text-muted-foreground text-xs">
                Click ▶ Play
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Stats bar */}
      <div className="shrink-0 border-t px-3 py-1.5 flex items-center gap-3 text-xs text-muted-foreground bg-background">
        {currentStats ? (
          <>
            <span>📊 Tokens: <span className="font-medium tabular-nums text-foreground">{formatTokenCount(currentStats.tokens)}</span></span>
            <span>In: <span className="font-medium tabular-nums text-foreground">{formatTokenCount(currentStats.input)}</span></span>
            <span>Out: <span className="font-medium tabular-nums text-foreground">{formatTokenCount(currentStats.output)}</span></span>
            <span>⏱ <span className="font-medium tabular-nums text-foreground">{formatLatency(currentStats.latency)}</span></span>
            {currentMsg?.model && <span>Model: <span className="font-medium text-foreground">{currentMsg.model}</span></span>}
          </>
        ) : <span>Waiting to start...</span>}
        {sessionModel && <span className="ml-auto opacity-60">Session: {sessionModel}</span>}
      </div>

      <style>{`
        @keyframes chat-replay-dot-bounce {
          0%, 80%, 100% { transform: translateY(0); opacity: 0.4; }
          40% { transform: translateY(-6px); opacity: 1; }
        }
      `}</style>
    </div>
  )
}
