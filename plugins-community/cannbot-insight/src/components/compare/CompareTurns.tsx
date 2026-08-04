"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useRef, useCallback, useMemo, useEffect, memo } from "react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"
import { computeAlignStats, computeContentDiff, estimateAlignTimeBreakdown, type AlignedPair, type TurnData, type DiffRange, type ManualAlignment } from "@/lib/compare/turn-align"
import { useTurnAlignWorker } from "@/lib/compare/use-turn-align-worker"

interface CompareTurnsProps {
  turnsA: TurnData[]
  turnsB: TurnData[]
}

const ROLE_ICONS: Record<string, string> = {
  user: "👤",
  assistant: "🤖",
  system: "⚙️",
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

function DiffContent({ ranges }: { ranges: DiffRange[] }) {
  return (
    <pre className="text-xs whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto">
      {ranges.map((range, i) => (
        <span
          key={i}
          className={cn(
            range.type === "equal" && "",
            range.type === "added" && "bg-emerald-100/60 dark:bg-emerald-500/15 text-emerald-800 dark:text-emerald-300",
            range.type === "removed" && "bg-red-100/60 dark:bg-red-500/15 text-red-800 dark:text-red-300",
          )}
        >
          {range.type === "removed" && "− "}
          {range.type === "added" && "+ "}
          {range.text}
          {i < ranges.length - 1 ? "\n" : ""}
        </span>
      ))}
    </pre>
  )
}

const TurnPanel = memo(function TurnPanel({
  turn,
  side,
  diffRanges,
  showDiff,
  selectable,
  selected,
  onSelectTurn,
}: {
  turn: TurnData
  side: "A" | "B"
  // Pre-computed by the parent PairCard: A side gets equal+removed ranges,
  // B side gets equal+added. null when diff is off or content missing.
  // Hoisted out of TurnPanel so each pair computes LCS once, not twice.
  diffRanges: DiffRange[] | null
  showDiff: boolean
  selectable?: boolean
  selected?: boolean
  onSelectTurn?: (turnIndex: number) => void
}) {
  const [showFull, setShowFull] = useState(false)
  const badgeVariant = side === "A" ? "blue" : "orange"
  const accentBorder = side === "A"
    ? "border-l-2 border-l-blue-400 rounded-tl-xl rounded-bl-xl"
    : "border-r-2 border-r-orange-400 rounded-tr-xl rounded-br-xl"
  const content = turn.content
  const thinkingMatch = content?.match(/<thinking>([\s\S]*?)<\/thinking>/)
  const thinkingContent = thinkingMatch?.[1] ?? null
  const textContent = thinkingMatch ? content?.replace(/<thinking>[\s\S]*?<\/thinking>/, "").trim() : content
  const displayContent = showFull ? textContent : textContent?.substring(0, 500)

  const handleSelect = useCallback(() => {
    onSelectTurn?.(turn.turnIndex)
  }, [onSelectTurn, turn.turnIndex])

  return (
    <div
      className={cn(
        "px-3 py-2 space-y-2 transition-all",
        accentBorder,
        selectable && "cursor-pointer hover:bg-primary/5",
        selected && "ring-2 ring-purple-400 bg-purple-50/40 dark:bg-purple-500/10",
      )}
      onClick={selectable ? handleSelect : undefined}
    >
      <div className="flex items-center gap-1.5">
        <Badge variant={badgeVariant} className="text-xs">{ROLE_ICONS[turn.role] ?? turn.role} {turn.role}</Badge>
        <span className="text-xs tabular-nums font-medium">{formatTokens(turn.totalTokens)} tok</span>
        {turn.latencyMs > 0 && <span className="text-xs text-muted-foreground">{formatMs(turn.latencyMs)}</span>}
        {turn.model && <span className="text-xs text-muted-foreground truncate max-w-[120px]">{turn.model}</span>}
        {selected && <Badge variant="purple" className="text-xs ml-1">已选</Badge>}
      </div>

      {thinkingContent && (
        <div className="ring-1 ring-foreground/10 rounded-md p-1.5 bg-purple-50/30 dark:bg-purple-500/5">
          <div className="flex items-center gap-1 mb-1">
            <Badge variant="purple" className="text-xs">thinking</Badge>
            <span className="text-xs text-muted-foreground">{formatTokens(turn.reasoningTokens)} reasoning</span>
          </div>
          <pre className="text-xs whitespace-pre-wrap break-words max-h-[150px] overflow-y-auto">
            {showFull ? thinkingContent : thinkingContent.substring(0, 300)}
          </pre>
        </div>
      )}

      <div>
        {showDiff && diffRanges && diffRanges.length > 0 ? (
          <DiffContent ranges={diffRanges} />
        ) : (
          <pre className="text-xs whitespace-pre-wrap break-words max-h-[300px] overflow-y-auto">
            {displayContent ?? turn.contentSummary ?? "(empty)"}
          </pre>
        )}
        {textContent && textContent.length > 500 && !showFull && !showDiff && (
          <button
            className="text-xs text-blue-500 hover:text-blue-600 cursor-pointer mt-1"
            onClick={() => setShowFull(true)}
          >
            Show full ({textContent.length} chars)
          </button>
        )}
        {showFull && textContent && textContent.length > 500 && (
          <button
            className="text-xs text-muted-foreground hover:text-foreground cursor-pointer mt-1"
            onClick={() => setShowFull(false)}
          >
            Collapse
          </button>
        )}
      </div>

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
            <Badge key={tc.toolCallId} variant={tc.state === "ok" ? "outline" : "red"} className="text-xs">
              {tc.toolName}
            </Badge>
          ))}
        </div>
      )}

      {turn.skillEvents.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {turn.skillEvents.map(se => (
            <Badge key={se.skillName + se.eventType} variant={se.success ? "yellow" : "red"} className="text-xs">
              {se.skillName}
            </Badge>
          ))}
        </div>
      )}
    </div>
  )
 })

// PairCard: a single aligned-pair row. Extracted from the inline map callback
// so React.memo can skip re-renders when the parent CompareTurns re-renders
// due to scroll-driven setVisibleStartIdx/End changes — those don't affect
// PairCard props, so the memo boundary short-circuits ~8000 PairCards that
// used to re-render on every scroll tick.
//
// Also hoists computeContentDiff up from TurnPanel: each pair's LCS is now
// computed once here and the filtered A/B halves are passed down, instead of
// each TurnPanel running LCS independently (2× LCS work per pair).
interface PairCardProps {
  pair: AlignedPair
  globalIdx: number
  showDiff: boolean
  selectMode: "off" | "selecting"
  selectedSide: "A" | "B" | null
  selectedTurn: number | null
  manuallyUsedA: Set<number>
  manuallyUsedB: Set<number>
  onRemoveManual: (indexA: number, indexB: number) => void
  onSelectTurnA: (indexA: number) => void
  onSelectTurnB: (indexB: number) => void
  registerRef: (globalIdx: number, el: HTMLDivElement | null) => void
}

const PairCard = memo(function PairCard({
  pair,
  globalIdx,
  showDiff,
  selectMode,
  selectedSide,
  selectedTurn,
  manuallyUsedA,
  manuallyUsedB,
  onRemoveManual,
  onSelectTurnA,
  onSelectTurnB,
  registerRef,
}: PairCardProps) {
  const isSimilar = pair.type === "match" && pair.similarity >= 0.4

  const ringClass = pair.isManual
    ? "border-2 border-purple-400"
    : isSimilar
      ? "border-2 border-red-500"
      : "border border-foreground/10"

  const bgClass = pair.isManual
    ? "bg-purple-50/40 dark:bg-purple-500/10"
    : isSimilar
      ? "bg-red-50/50 dark:bg-red-500/10"
      : ""

  const headerBg = pair.isManual
    ? "bg-purple-100/50 dark:bg-purple-800/30"
    : isSimilar
      ? "bg-red-100/60 dark:bg-red-800/30"
      : "bg-muted/30"

  const aSelectable = !!(selectMode !== "off" && pair.a && !manuallyUsedA.has(pair.a.turnIndex) && (selectedTurn === null || selectedSide === "B"))
  const bSelectable = !!(selectMode !== "off" && pair.b && !manuallyUsedB.has(pair.b.turnIndex) && (selectedTurn === null || selectedSide === "A"))
  const aSelected = !!(selectedSide === "A" && selectedTurn !== null && pair.a?.turnIndex === selectedTurn)
  const bSelected = !!(selectedSide === "B" && selectedTurn !== null && pair.b?.turnIndex === selectedTurn)

  // Shared LCS: compute once, filter for A (equal+removed) and B (equal+added).
  // Previously each TurnPanel ran LCS on the same content pair independently.
  const { aDiff, bDiff } = useMemo<{ aDiff: DiffRange[] | null; bDiff: DiffRange[] | null }>(() => {
    if (pair.type !== "match" || !pair.a || !pair.b || !showDiff) {
      return { aDiff: null, bDiff: null }
    }
    const aContent = pair.a.content ?? pair.a.contentSummary
    const bContent = pair.b.content ?? pair.b.contentSummary
    if (!aContent || !bContent) return { aDiff: null, bDiff: null }
    const all = computeContentDiff(aContent, bContent)
    return {
      aDiff: all.filter(r => r.type === "equal" || r.type === "removed"),
      bDiff: all.filter(r => r.type === "equal" || r.type === "added"),
    }
  }, [pair.a, pair.b, pair.type, showDiff])

  return (
    <div
      ref={(el) => registerRef(globalIdx, el)}
      data-global-idx={globalIdx}
      className={cn("rounded-xl", ringClass, bgClass)}
      style={{ contentVisibility: "auto", containIntrinsicSize: "500px" }}
    >
      <div className={cn("flex items-center gap-2 px-3 py-1.5 border-b text-xs rounded-t-xl", headerBg)}>
        {pair.isManual && pair.a && pair.b && (
          <>
            <Badge variant="purple" className="text-xs">手动</Badge>
            <span className="font-mono text-muted-foreground">
              #{pair.a.turnIndex}/{pair.b.turnIndex}
            </span>
            {pair.a.role === pair.b.role && (
              <Badge variant={isSimilar ? "red" : "outline"} className="text-xs">
                {isSimilar ? "相似" : "不相似"}
              </Badge>
            )}
            {pair.a.role !== pair.b.role && (
              <Badge variant="red" className="text-xs">role: {pair.a.role} vs {pair.b.role}</Badge>
            )}
            <button
              className="ml-auto text-xs text-purple-500 hover:text-red-500 cursor-pointer"
              onClick={() => onRemoveManual(pair.indexA!, pair.indexB!)}
              title="取消手动对齐"
            >
              ✕ 取消对齐
            </button>
          </>
        )}
        {!pair.isManual && pair.type === "match" && pair.a && pair.b && (
          <>
            <span className="font-mono text-muted-foreground">
              #{pair.a.turnIndex}/{pair.b.turnIndex}
            </span>
            {pair.a.role !== pair.b.role && (
              <Badge variant="red" className="text-xs">role: {pair.a.role} vs {pair.b.role}</Badge>
            )}
            {pair.a.role === pair.b.role && (
              <Badge variant={isSimilar ? "red" : "outline"} className="text-xs">
                {isSimilar ? "相似" : "不相似"}
              </Badge>
            )}
            <span className="ml-auto text-muted-foreground tabular-nums">
              {pair.a.totalTokens !== pair.b.totalTokens && (
                <span className={pair.a.totalTokens < pair.b.totalTokens ? "text-blue-600 dark:text-blue-400" : "text-orange-600 dark:text-orange-400"}>
                  {formatTokens(Math.abs(pair.b.totalTokens - pair.a.totalTokens))} tok diff
                </span>
              )}
            </span>
          </>
        )}
        {pair.type === "aOnly" && pair.a && (
          <span className="font-mono">
            <Badge variant="blue" className="text-xs">A #{pair.a.turnIndex}</Badge>
            <span className="text-muted-foreground ml-1">{ROLE_ICONS[pair.a.role]} {pair.a.role}</span>
          </span>
        )}
        {pair.type === "bOnly" && pair.b && (
          <span className="font-mono">
            <Badge variant="orange" className="text-xs">B #{pair.b.turnIndex}</Badge>
            <span className="text-muted-foreground ml-1">{ROLE_ICONS[pair.b.role]} {pair.b.role}</span>
          </span>
        )}
      </div>

      {pair.type === "match" && pair.a && pair.b ? (
        <div className="grid grid-cols-2 gap-0 divide-x divide-foreground/10 rounded-b-xl">
          <TurnPanel
            turn={pair.a}
            side="A"
            showDiff={showDiff}
            diffRanges={aDiff}
            selectable={aSelectable}
            selected={aSelected}
            onSelectTurn={onSelectTurnA}
          />
          <TurnPanel
            turn={pair.b}
            side="B"
            showDiff={showDiff}
            diffRanges={bDiff}
            selectable={bSelectable}
            selected={bSelected}
            onSelectTurn={onSelectTurnB}
          />
        </div>
      ) : pair.type === "aOnly" && pair.a ? (
        <div className="grid grid-cols-2 gap-0 divide-x divide-foreground/10 rounded-b-xl">
          <TurnPanel
            turn={pair.a}
            side="A"
            showDiff={false}
            diffRanges={null}
            selectable={aSelectable}
            selected={aSelected}
            onSelectTurn={onSelectTurnA}
          />
          <div className="px-3 py-2 bg-muted/20">
            <p className="text-xs text-muted-foreground italic">— B 无对应 turn —</p>
          </div>
        </div>
      ) : pair.type === "bOnly" && pair.b ? (
        <div className="grid grid-cols-2 gap-0 divide-x divide-foreground/10 rounded-b-xl">
          <div className="px-3 py-2 bg-muted/20">
            <p className="text-xs text-muted-foreground italic">— A 无对应 turn —</p>
          </div>
          <TurnPanel
            turn={pair.b}
            side="B"
            showDiff={false}
            diffRanges={null}
            selectable={bSelectable}
            selected={bSelected}
            onSelectTurn={onSelectTurnB}
          />
        </div>
      ) : null}
    </div>
  )
})

function OverviewGutter({
  pairs,
  scrollToPair,
  visibleStartIdx,
  visibleEndIdx,
  gutterHeight,
  draggingRef,
  onDragEnd,
}: {
  pairs: { pair: AlignedPair; globalIdx: number }[]
  scrollToPair: (globalIdx: number, smooth?: boolean) => void
  visibleStartIdx: number
  visibleEndIdx: number
  gutterHeight: number
  draggingRef: React.MutableRefObject<boolean>
  onDragEnd: () => void
}) {
  const INPUT_H = 32
  const BUTTON_H = 32
  const PROGRESS_H = 22
  const DETAIL_ROW_H = 28

  const bgForPair = (pair: AlignedPair) =>
    pair.isManual
      ? "bg-purple-500"
      : pair.type === "match" && pair.similarity >= 0.4
        ? "bg-red-400"
        : "bg-muted-foreground/20"

  const totalPairs = pairs.length
  const viewportTopPct = totalPairs > 0 ? visibleStartIdx / totalPairs * 100 : 0
  const viewportH = totalPairs > 0 ? (visibleEndIdx - visibleStartIdx + 1) / totalPairs * 100 : 100

  // Track the last jump target so consecutive clicks always advance to the
  // NEXT match — visibleStartIdx (React state) may not have updated yet
  // between rapid clicks, causing jumpToType to find the same pair again.
  // When the user manually scrolls, visibleStartIdx changes and no longer
  // matches lastJumpIdxRef, so we fall back to using visibleStartIdx.
  const lastJumpIdxRef = useRef(-1)

  const counts = useMemo(() => {
    let manual = 0, similar = 0, unsimilar = 0, aOnly = 0, bOnly = 0
    for (const { pair } of pairs) {
      if (pair.isManual) manual++
      else if (pair.type === "match" && pair.similarity >= 0.4) similar++
      else if (pair.type === "match") unsimilar++
      else if (pair.type === "aOnly") aOnly++
      else bOnly++
    }
    return { manual, similar, unsimilar, aOnly, bOnly }
  }, [pairs])

  const pairType = useCallback((pair: AlignedPair): string => {
    if (pair.isManual) return "manual"
    if (pair.type === "match" && pair.similarity >= 0.4) return "similar"
    if (pair.type === "match") return "unsimilar"
    if (pair.type === "aOnly") return "aOnly"
    return "bOnly"
  }, [])

  const jumpToType = useCallback((type: string, forward: boolean) => {
    if (pairs.length === 0) return
    // Always use lastJumpIdxRef if set (synced immediately on last jump).
    // visibleStartIdx (React state) lags behind on rapid consecutive clicks
    // because it waits for React flush + scroll event DOM measurement. Using
    // the ref directly guarantees each click advances past the previous target.
    // Falls back to visibleStartIdx only on the very first click (ref = -1).
    const baseIdx = lastJumpIdxRef.current >= 0 ? lastJumpIdxRef.current : visibleStartIdx
    const start = forward ? baseIdx + 1 : baseIdx - 1
    const findAndJump = (from: number, to: number, step: number) => {
      for (let i = from; step > 0 ? i < to : i >= to; i += step) {
        if (pairType(pairs[i].pair) === type) {
          lastJumpIdxRef.current = pairs[i].globalIdx
          scrollToPair(pairs[i].globalIdx)
          return true
        }
      }
      return false
    }
    if (forward) {
      if (findAndJump(start, pairs.length, 1)) return
      findAndJump(0, Math.min(start, pairs.length), 1)
    } else {
      if (findAndJump(start, -1, -1)) return
      findAndJump(pairs.length - 1, start, -1)
    }
  }, [pairs, visibleStartIdx, pairType, scrollToPair])

  const detailListRef = useRef<HTMLDivElement>(null)

  // When visibleStartIdx changes (user scrolled the right-side turns), auto-scroll
  // the detail list to keep the current pair centered. Before v1.31 the detail
  // list only showed ~28 pairs around visibleStartIdx (content auto-shifted), but
  // now it shows ALL pairs for unlimited scrolling — so we must scroll the list
  // explicitly to follow the right side.
  useEffect(() => {
    const el = detailListRef.current
    if (!el) return
    const targetTop = visibleStartIdx * DETAIL_ROW_H - el.clientHeight / 2 + DETAIL_ROW_H / 2
    el.scrollTo({ top: Math.max(0, targetTop), behavior: "auto" })
  }, [visibleStartIdx])

  const [jumpInput, setJumpInput] = useState("")

  const handleJump = useCallback(() => {
    const idx = parseInt(jumpInput, 10)
    if (isNaN(idx)) return
    const found = pairs.find(p => p.pair.a?.turnIndex === idx || p.pair.b?.turnIndex === idx)
    if (found) scrollToPair(found.globalIdx)
  }, [jumpInput, pairs, scrollToPair])

  const onJumpKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleJump()
  }, [handleJump])

  const minimapRef = useRef<HTMLDivElement>(null)
  const thumbRef = useRef<HTMLDivElement>(null)
  const dragRafRef = useRef<number | null>(null)
  const lastDragClientY = useRef(0)

  const yToGlobalIdx = useCallback((clientY: number): number | null => {
    const container = minimapRef.current
    if (!container || pairs.length === 0) return null
    const rect = container.getBoundingClientRect()
    const y = Math.max(0, Math.min(rect.height, clientY - rect.top))
    const pct = y / rect.height
    const idx = Math.floor(pct * pairs.length)
    return pairs[Math.min(idx, pairs.length - 1)]?.globalIdx ?? null
  }, [pairs])

  const onMinimapPointerDown = useCallback((e: React.PointerEvent) => {
    draggingRef.current = true
    lastDragClientY.current = e.clientY
    ;(e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId)
    const idx = yToGlobalIdx(e.clientY)
    if (idx !== null) scrollToPair(idx, false)
  }, [yToGlobalIdx, scrollToPair])

  const onMinimapPointerMove = useCallback((e: React.PointerEvent) => {
    if (!draggingRef.current) return
    lastDragClientY.current = e.clientY
    const container = minimapRef.current
    const thumb = thumbRef.current
    if (container && thumb) {
      const rect = container.getBoundingClientRect()
      const y = Math.max(0, Math.min(rect.height, e.clientY - rect.top))
      const pct = rect.height > 0 ? y / rect.height : 0
      const thumbHRatio = Math.max(viewportH, 5) / 100
      const top = Math.min(pct, Math.max(0, 1 - thumbHRatio))
      thumb.style.top = `${top * 100}%`
    }
    if (dragRafRef.current !== null) return
    dragRafRef.current = requestAnimationFrame(() => {
      dragRafRef.current = null
      const idx = yToGlobalIdx(lastDragClientY.current)
      if (idx !== null) scrollToPair(idx, false)
    })
  }, [draggingRef, viewportH, yToGlobalIdx, scrollToPair])

  const onMinimapPointerUp = useCallback((e: React.PointerEvent) => {
    draggingRef.current = false
    if (dragRafRef.current !== null) {
      cancelAnimationFrame(dragRafRef.current)
      dragRafRef.current = null
    }
    ;(e.currentTarget as HTMLElement).releasePointerCapture?.(e.pointerId)
    onDragEnd()
  }, [onDragEnd])

  // Show all pairs in the detail list (not just a slice around visibleStartIdx)
  // so the user can scroll the gutter indefinitely. content-visibility:auto on
  // each row keeps the cost low — off-screen rows skip layout/paint.
  const curIdx = pairs.length > 0 ? visibleStartIdx + 1 : 0
  const pctV = totalPairs > 0 ? Math.round(curIdx / totalPairs * 100) : 0

  const TYPE_BUTTONS: Array<{ key: string; label: string; color: string; count: number }> = [
    { key: "manual", label: "手动", color: "bg-purple-500", count: counts.manual },
    { key: "similar", label: "相似", color: "bg-red-400", count: counts.similar },
    { key: "aOnly", label: "A only", color: "bg-blue-400", count: counts.aOnly },
    { key: "bOnly", label: "B only", color: "bg-orange-400", count: counts.bOnly },
  ]

  return (
    <div className="flex flex-col w-20 shrink-0 rounded-md ring-1 ring-foreground/10 bg-muted/10 overflow-hidden" style={{ height: gutterHeight }}>
      <div className="flex items-center gap-1 px-1 border-b border-foreground/10 shrink-0" style={{ height: INPUT_H }}>
        <span className="text-[10px] text-muted-foreground">#</span>
        <input
          type="text"
          inputMode="numeric"
          value={jumpInput}
          onChange={(e) => setJumpInput(e.target.value.replace(/[^0-9]/g, ""))}
          onKeyDown={onJumpKeyDown}
          placeholder="1234"
          className="flex-1 min-w-0 bg-transparent text-[10px] tabular-nums outline-none border-b border-transparent focus:border-foreground/30"
        />
        <button
          className="text-[10px] px-1 rounded-sm cursor-pointer text-muted-foreground hover:bg-accent hover:text-foreground"
          onClick={handleJump}
          title="跳转到 turn #"
        >↗</button>
      </div>

      <div className="grid grid-cols-2 gap-0.5 p-1 border-b border-foreground/10 shrink-0" style={{ height: BUTTON_H }}>
        {TYPE_BUTTONS.map(btn => {
          const disabled = btn.count === 0
          return (
            <button
              key={btn.key}
              className={cn(
                "flex items-center gap-1 text-[10px] px-1 rounded-sm text-left tabular-nums",
                disabled
                  ? "text-muted-foreground/40 cursor-default"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground cursor-pointer"
              )}
              onClick={(e) => !disabled && jumpToType(btn.key, !e.shiftKey)}
              title={disabled ? `${btn.label}: 无` : `${btn.label}: ${btn.count} (Shift+向前)`}
            >
              <span className={cn("inline-block w-1.5 h-1.5 rounded-sm shrink-0", btn.color)} />
              <span className="truncate">{btn.label}</span>
              <span className="font-mono ml-auto">{btn.count}</span>
            </button>
          )
        })}
      </div>

      <div className="flex flex-1 min-h-0">
        <div
          ref={detailListRef}
          className="flex-1 overflow-y-auto min-h-0 [scrollbar-width:none] [ms-overflow-style:none] [&::-webkit-scrollbar]:hidden"
        >
          {pairs.map(({ pair, globalIdx }, i) => {
            const isCurrent = i >= visibleStartIdx && i <= visibleEndIdx
            const sim = pair.similarity
            const aIdx = pair.a?.turnIndex ?? "?"
            const bIdx = pair.b?.turnIndex ?? "?"
            return (
              <div
                key={globalIdx}
                onClick={() => scrollToPair(globalIdx)}
                className={cn(
                  "flex items-center gap-1 px-1 cursor-pointer text-[10px] tabular-nums border-b border-foreground/5",
                  isCurrent ? "bg-primary/10 text-foreground" : "text-muted-foreground hover:bg-accent/50",
                )}
                style={{ height: DETAIL_ROW_H, contentVisibility: "auto", containIntrinsicSize: "28px" }}
                title={`#A${aIdx}/B${bIdx}`}
              >
                <span className={cn("inline-block w-1.5 h-1.5 rounded-sm shrink-0", bgForPair(pair))} />
                <span className="font-mono truncate">{aIdx}/{bIdx}</span>
                <span className="ml-auto truncate">{pair.isManual ? "手动" : pair.type === "match" ? `${Math.round(sim * 100)}%` : ""}</span>
              </div>
            )
          })}
        </div>
        <div
          ref={minimapRef}
          className="shrink-0 w-2 relative cursor-pointer touch-none bg-muted/30 rounded-sm"
          onPointerDown={onMinimapPointerDown}
          onPointerMove={onMinimapPointerMove}
          onPointerUp={onMinimapPointerUp}
          title="拖拽快速滚动"
        >
          <div
            ref={thumbRef}
            className="absolute left-0 right-0 bg-foreground/50 rounded-full pointer-events-none"
            style={{ top: draggingRef.current ? undefined : `${viewportTopPct}%`, height: `${Math.max(viewportH, 5)}%` }}
          />
        </div>
      </div>

      <div className="flex items-center justify-between px-1 py-0.5 border-t border-foreground/10 text-[10px] text-muted-foreground tabular-nums shrink-0" style={{ height: PROGRESS_H }}>
        <span>{curIdx}/{totalPairs}</span>
        <span>{pctV}%</span>
      </div>
    </div>
  )
}

function AlignStatsBar({ stats, manualCount }: { stats: ReturnType<typeof computeAlignStats>; manualCount: number }) {
  return (
    <div className="flex items-center gap-2 text-xs flex-wrap">
      {manualCount > 0 && (
        <span className="flex items-center gap-1">
          <span className="inline-block w-2 h-2 rounded-full bg-purple-500" />
          手动 <span className="tabular-nums font-medium">{manualCount}</span>
        </span>
      )}
      <span className="flex items-center gap-1">
        <span className="inline-block w-2 h-2 rounded-full bg-red-400" />
        相似 <span className="tabular-nums font-medium">{stats.highSimilarity + stats.mediumSimilarity}</span>
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/30 ring-1 ring-muted-foreground/50" />
        不相似 <span className="tabular-nums font-medium">{stats.lowSimilarity}</span>
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/30 ring-1 ring-muted-foreground/50" />
        A only <span className="tabular-nums font-medium">{stats.aOnly}</span>
      </span>
      <span className="flex items-center gap-1">
        <span className="inline-block w-2 h-2 rounded-full bg-muted-foreground/30 ring-1 ring-muted-foreground/50" />
        B only <span className="tabular-nums font-medium">{stats.bOnly}</span>
      </span>
      <span className="text-muted-foreground">平均 <span className="tabular-nums font-medium">{Math.round(stats.avgSimilarity * 100)}%</span></span>
    </div>
  )
}

type FilterKey = "all" | "similar" | "unsimilar" | "aOnly" | "bOnly" | "manual"

const FILTER_ITEMS: Array<{ key: FilterKey; label: string; icon: string }> = [
  { key: "all", label: "全部", icon: "" },
  { key: "manual", label: "手动", icon: "📌" },
  { key: "similar", label: "相似", icon: "" },
  { key: "unsimilar", label: "不相似", icon: "" },
  { key: "aOnly", label: "A only", icon: "" },
  { key: "bOnly", label: "B only", icon: "" },
]

export function CompareTurns({ turnsA, turnsB }: CompareTurnsProps) {
  const [manualAlignments, setManualAlignments] = useState<ManualAlignment[]>([])
  const [selectMode, setSelectMode] = useState<"off" | "selecting">("off")
  const [selectedSide, setSelectedSide] = useState<"A" | "B" | null>(null)
  const [selectedTurn, setSelectedTurn] = useState<number | null>(null)

  const { pairs, stats, loading, progress } = useTurnAlignWorker(turnsA, turnsB, manualAlignments)
  const [showDiff, setShowDiff] = useState(true)
  const [filter, setFilter] = useState<FilterKey>("all")
  // Phase 6: removed renderedCount + setTimeout +100/帧 batching. Replaced by
  // virtualRange window driven by scroll position — only PairCards inside
  // [visibleStartIdx - BUFFER, visibleEndIdx + BUFFER] are mounted at all.
  const pairRefs = useRef<Map<number, HTMLDivElement>>(new Map())
  const scrollRef = useRef<HTMLDivElement>(null)
  const contentRef = useRef<HTMLDivElement>(null)
  // When scrollToPair targets an unmounted pair, we force virtualRange to
  // include it, then this ref remembers the target so a useEffect can
  // precisely center it via getBoundingClientRect once it mounts + layouts.
  const pendingScrollRef = useRef<{ globalIdx: number; smooth: boolean } | null>(null)

  const manuallyUsedA = useMemo(() => new Set(manualAlignments.map(m => m.indexA)), [manualAlignments])
  const manuallyUsedB = useMemo(() => new Set(manualAlignments.map(m => m.indexB)), [manualAlignments])

  const selStateRef = useRef({ selectMode, selectedSide, selectedTurn, manuallyUsedA, manuallyUsedB })
  useEffect(() => {
    selStateRef.current = { selectMode, selectedSide, selectedTurn, manuallyUsedA, manuallyUsedB }
  })
  const handleSelectTurnA = useCallback((indexA: number) => {
    const { selectMode, selectedSide, selectedTurn, manuallyUsedA } = selStateRef.current
    if (manuallyUsedA.has(indexA)) return
    if (selectMode === "selecting" && selectedTurn === null) {
      setSelectedSide("A")
      setSelectedTurn(indexA)
    } else if (selectMode === "selecting" && selectedSide === "B" && selectedTurn !== null) {
      setManualAlignments(prev => [...prev, { indexA, indexB: selectedTurn }])
      setSelectedSide(null)
      setSelectedTurn(null)
      setSelectMode("off")
    }
  }, [])

  const handleSelectTurnB = useCallback((indexB: number) => {
    const { selectMode, selectedSide, selectedTurn, manuallyUsedB } = selStateRef.current
    if (manuallyUsedB.has(indexB)) return
    if (selectMode === "selecting" && selectedTurn === null) {
      setSelectedSide("B")
      setSelectedTurn(indexB)
    } else if (selectMode === "selecting" && selectedSide === "A" && selectedTurn !== null) {
      setManualAlignments(prev => [...prev, { indexA: selectedTurn, indexB }])
      setSelectedSide(null)
      setSelectedTurn(null)
      setSelectMode("off")
    }
  }, [])

  const handleRemoveManual = useCallback((indexA: number, indexB: number) => {
    setManualAlignments(prev => prev.filter(m => m.indexA !== indexA || m.indexB !== indexB))
  }, [])

  const enterSelectMode = useCallback(() => {
    setSelectMode("selecting")
    setSelectedSide(null)
    setSelectedTurn(null)
  }, [])

  const cancelSelectMode = useCallback(() => {
    setSelectMode("off")
    setSelectedSide(null)
    setSelectedTurn(null)
  }, [])

  useEffect(() => {
    if (selectMode === "off") return
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") cancelSelectMode()
    }
    window.addEventListener("keydown", onKeyDown)
    return () => window.removeEventListener("keydown", onKeyDown)
  }, [selectMode, cancelSelectMode])

  const filteredPairs = useMemo(() => {
    const indexed = pairs.map((p, i) => ({ pair: p, globalIdx: i }))
    if (filter === "all") return indexed
    if (filter === "similar") return indexed.filter(({ pair }) => pair.type === "match" && !pair.isManual && pair.similarity >= 0.4)
    if (filter === "unsimilar") return indexed.filter(({ pair }) => pair.type === "match" && !pair.isManual && pair.similarity < 0.4)
    if (filter === "aOnly") return indexed.filter(({ pair }) => pair.type === "aOnly")
    if (filter === "bOnly") return indexed.filter(({ pair }) => pair.type === "bOnly")
    if (filter === "manual") return indexed.filter(({ pair }) => pair.isManual)
    return indexed
  }, [pairs, filter])

  const filterCounts = useMemo(() => ({
    all: pairs.length,
    manual: manualAlignments.length,
    similar: pairs.filter(p => p.type === "match" && !p.isManual && p.similarity >= 0.4).length,
    unsimilar: pairs.filter(p => p.type === "match" && !p.isManual && p.similarity < 0.4).length,
    aOnly: stats.aOnly,
    bOnly: stats.bOnly,
  }), [pairs, manualAlignments, stats])

  const [gutterHeight, setGutterHeight] = useState(600)
  const gutterRef = useRef<HTMLDivElement>(null)
  const draggingRef = useRef(false)
  const [visibleStartIdx, setVisibleStartIdx] = useState(0)
  const [visibleEndIdx, setVisibleEndIdx] = useState(filteredPairs.length - 1)

  // Phase 6 virtualization. Previously every scroll re-rendered all 12000
  // PairCards because the scroll listener used DOM binary search over the
  // full child list. Now we estimate the visible window from scrollTop
  // directly (PAIR_ROW_HEIGHT matches containIntrinsicSize), then render
  // only [vStart - BUFFER, vEnd + BUFFER] — typically ~20-40 PairCards
  // instead of 12000. The rest are pure placeholder divs with no children.
  const PAIR_ROW_HEIGHT = 500
  const VIRTUAL_BUFFER = 10
  const virtualRange = useMemo(() => {
    const total = filteredPairs.length
    if (total === 0) return { start: 0, end: -1 }
    const start = Math.max(0, visibleStartIdx - VIRTUAL_BUFFER)
    const end = Math.min(total - 1, visibleEndIdx + VIRTUAL_BUFFER)
    return { start, end }
  }, [filteredPairs.length, visibleStartIdx, visibleEndIdx])

  useEffect(() => {
    const el = gutterRef.current
    if (el) setGutterHeight(el.clientHeight)
  }, [filteredPairs])

  useEffect(() => {
    const container = scrollRef.current
    const content = contentRef.current
    if (!container || !content) return
    let rafId: number | null = null
    let lastStart = -1
    let lastEnd = -1
    let lastDragSetTime = 0
    const onScroll = () => {
      if (rafId !== null) return
      rafId = requestAnimationFrame(() => {
        rafId = null
        if (draggingRef.current) {
          const now = Date.now()
          if (now - lastDragSetTime < 150) return
          lastDragSetTime = now
        }
        const total = filteredPairs.length
        if (total === 0) return

        // Measure actual visible PairCards using getBoundingClientRect
        // (viewport-relative, not affected by offsetParent or
        // content-visibility:auto layout skipping). offsetTop/offsetHeight
        // were unreliable for off-screen content-visibility:auto elements.
        const containerRect = container.getBoundingClientRect()
        const children = content.children
        let start = -1
        let end = -1
        for (let i = 0; i < children.length; i++) {
          const child = children[i] as HTMLElement
          if (child.getAttribute("aria-hidden") === "true") continue
          const idxAttr = child.getAttribute("data-global-idx")
          if (!idxAttr) continue
          const idx = parseInt(idxAttr, 10)
          const childRect = child.getBoundingClientRect()
          if (childRect.height === 0) continue // not laid out yet
          const childTopRel = childRect.top - containerRect.top
          const childBotRel = childTopRel + childRect.height
          if (childBotRel > 0 && start === -1) start = idx
          if (childTopRel < containerRect.height) end = idx
          else break
        }

        const scrollTop = container.scrollTop
        const viewportH = container.clientHeight
        if (start === -1) start = Math.max(0, Math.floor(scrollTop / PAIR_ROW_HEIGHT))
        if (end === -1) end = Math.min(total - 1, start + Math.ceil(viewportH / PAIR_ROW_HEIGHT))
        end = Math.min(total - 1, Math.max(start, end))

        if (start !== lastStart || end !== lastEnd) {
          lastStart = start
          lastEnd = end
          setVisibleStartIdx(start)
          setVisibleEndIdx(end)
        }
      })
    }
    onScroll()
    container.addEventListener("scroll", onScroll, { passive: true })
    return () => {
      container.removeEventListener("scroll", onScroll)
      if (rafId !== null) cancelAnimationFrame(rafId)
    }
  }, [filteredPairs])

  const scrollToPair = useCallback((globalIdx: number, smooth = true) => {
    const container = scrollRef.current
    if (!container) return
    // If already mounted and laid out, center it immediately.
    const el = pairRefs.current.get(globalIdx)
    if (el && el.offsetHeight > 0) {
      const elRect = el.getBoundingClientRect()
      const containerRect = container.getBoundingClientRect()
      const elTop = elRect.top - containerRect.top + container.scrollTop
      const targetTop = elTop - container.clientHeight / 2 + elRect.height / 2
      container.scrollTo({ top: Math.max(0, targetTop), behavior: smooth ? "smooth" : "auto" })
      return
    }
    // Not mounted: estimate scroll + force virtualRange to include target.
    // Set visibleStartIdx to globalIdx (not globalIdx - BUFFER) so that:
    // 1. virtualRange = [globalIdx - BUFFER, globalIdx + viewport + BUFFER]
    //    → target PairCard mounts
    // 2. jumpToType's start point (visibleStartIdx + 1) is right after the
    //    target, so consecutive clicks find the NEXT match, not the same one
    pendingScrollRef.current = { globalIdx, smooth }
    const estTop = globalIdx * 500 - container.clientHeight / 2
    container.scrollTo({ top: Math.max(0, estTop), behavior: "auto" })
    const viewportPairs = Math.ceil(container.clientHeight / PAIR_ROW_HEIGHT)
    setVisibleStartIdx(globalIdx)
    setVisibleEndIdx(Math.min(filteredPairs.length - 1, globalIdx + viewportPairs))
  }, [filteredPairs.length])

  // After the scroll event updates visibleStartIdx/End → virtualRange →
  // PairCard mounts, center it precisely. Uses multiple rAF attempts because
  // content-visibility:auto elements have no geometry until they enter the
  // viewport — getBoundingClientRect may return 0 on the first frame.
  useEffect(() => {
    const pending = pendingScrollRef.current
    if (!pending) return
    const el = pairRefs.current.get(pending.globalIdx)
    if (!el) return
    pendingScrollRef.current = null

    const centerEl = (attempt: number) => {
      const container = scrollRef.current
      if (!container) return
      const elRect = el.getBoundingClientRect()
      if (elRect.height === 0 && attempt < 5) {
        requestAnimationFrame(() => centerEl(attempt + 1))
        return
      }
      const containerRect = container.getBoundingClientRect()
      let targetTop: number
      if (elRect.height > 0) {
        const elTop = elRect.top - containerRect.top + container.scrollTop
        targetTop = elTop - container.clientHeight / 2 + elRect.height / 2
      } else {
        targetTop = el.offsetTop - container.clientHeight / 2
      }
      container.scrollTo({ top: Math.max(0, targetTop), behavior: pending.smooth ? "smooth" : "auto" })
    }
    requestAnimationFrame(() => centerEl(0))
  }, [virtualRange])

  const onDragEnd = useCallback(() => {
    setTimeout(() => {
      scrollRef.current?.dispatchEvent(new Event("scroll"))
    }, 0)
  }, [])

  // Isolate gutter wheel events from the turns scroll container. Without this,
  // wheel over the gutter's non-scrollable areas (or after reaching the detail
  // list's scroll boundary) propagates to the turns container, making the
  // right side jump when the user scrolls inside the gutter.
  useEffect(() => {
    const el = gutterRef.current
    if (!el) return
    const onWheel = (e: WheelEvent) => {
      // Walk up from the wheel target to find the nearest scrollable ancestor
      // within the gutter. If it can still scroll in the wheel direction, let
      // the browser handle it natively (don't break inner scrolling).
      let node = e.target as HTMLElement | null
      while (node && node !== el) {
        const canScroll = node.scrollHeight > node.clientHeight
        if (canScroll) {
          const atTop = node.scrollTop <= 0 && e.deltaY < 0
          const atBottom = node.scrollTop + node.clientHeight >= node.scrollHeight && e.deltaY > 0
          if (!atTop && !atBottom) return // inner scroll, allow
          break // hit boundary, fall through to block propagation
        }
        node = node.parentElement
      }
      // No inner scrollable consumer (or reached its boundary): block the
      // wheel so it doesn't chain to the turns container on the right.
      e.preventDefault()
      e.stopPropagation()
    }
    el.addEventListener("wheel", onWheel, { passive: false })
    return () => el.removeEventListener("wheel", onWheel)
  }, [])

  // Stable ref callback: the inline arrow `ref={(el) => pairRefs.current.set(...)}`
  // was rebuilt every render, which made React call ref(null) then ref(el) on
  // every parent re-render — visible as 8000 ref churns on scroll. useCallback
  // keeps the function reference stable so React skips ref reattachment.
  const registerRef = useCallback((globalIdx: number, el: HTMLDivElement | null) => {
    if (el) pairRefs.current.set(globalIdx, el)
    else pairRefs.current.delete(globalIdx)
  }, [])

  if (loading && pairs.length === 0) {
    const N = turnsA.length, M = turnsB.length
    // estimateAlignTimeBreakdown covers worker transfer + align CPU + render.
    // Phase A (turns API fetch) is shown separately by page.tsx under
    // "正在获取 turn 数据..." so we don't double-count it here.
    const est = estimateAlignTimeBreakdown(N, M)
    const estLabel = est.totalMs < 1000
      ? "<1s"
      : `${Math.max(1, Math.round(est.totalMs / 1000))}s`
    const fmtStage = (ms: number) => ms < 1000 ? `${Math.round(ms)}ms` : `${(ms / 1000).toFixed(1)}s`
    const pct = Math.round(progress * 100)
    return (
      <div className="px-4 py-4 space-y-3">
        <div className="flex items-center justify-between">
          <p className="text-sm text-muted-foreground animate-pulse">正在对齐 turn，预计 ~{estLabel}</p>
          <span className="text-sm font-mono tabular-nums text-muted-foreground">{pct}%</span>
        </div>
        <div className="h-1.5 rounded-full bg-muted overflow-hidden">
          <div
            className="h-full bg-primary transition-all duration-150 rounded-full"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="text-xs text-muted-foreground">{N} × {M} = {(N * M).toLocaleString()} 对</p>
        <p className="text-xs text-muted-foreground tabular-nums">
          传输 {fmtStage(est.transferMs)} · 对齐 {fmtStage(est.alignMs)} · 渲染 {fmtStage(est.renderMs)}
        </p>
      </div>
    )
  }

  if (pairs.length === 0) {
    return <p className="text-sm text-muted-foreground">No turn data available.</p>
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <AlignStatsBar stats={stats} manualCount={manualAlignments.length} />
        <div className="flex items-center gap-2">
          {selectMode !== "off" ? (
            <>
              <span className="text-xs text-purple-600 dark:text-purple-400 animate-pulse">
                {selectedTurn === null
                  ? "请点击任意侧 turn"
                  : selectedSide === "A"
                    ? `已选 A #${selectedTurn}，请点击 B 侧 turn`
                    : `已选 B #${selectedTurn}，请点击 A 侧 turn`}
              </span>
              <button
                className="px-2 py-1 text-xs rounded-md ring-1 ring-foreground/10 bg-muted text-muted-foreground cursor-pointer hover:ring-red-400 hover:text-red-500 transition-colors"
                onClick={cancelSelectMode}
              >
                取消
              </button>
            </>
          ) : (
            <button
              className={cn(
                "px-2 py-1 text-xs rounded-md cursor-pointer transition-colors",
                manualAlignments.length > 0
                  ? "bg-purple-500/15 text-purple-700 dark:text-purple-400 ring-1 ring-purple-500/30"
                  : "ring-1 ring-foreground/10 text-muted-foreground hover:ring-purple-400",
              )}
              onClick={enterSelectMode}
            >
              手动对齐 {manualAlignments.length > 0 ? `(${manualAlignments.length})` : ""}
            </button>
          )}
          <button
            className={cn(
              "px-2 py-1 text-xs rounded-md cursor-pointer transition-colors",
              showDiff
                ? "bg-primary text-primary-foreground ring-1 ring-primary"
                : "ring-1 ring-foreground/10 text-muted-foreground hover:ring-primary",
            )}
            onClick={() => setShowDiff(!showDiff)}
          >
            内容对比
          </button>
        </div>
      </div>

      <div className="flex items-center gap-1.5">
        {FILTER_ITEMS.map(item => {
          const count = filterCounts[item.key]
          const isActive = filter === item.key
          return (
            <button
              key={item.key}
              className={cn(
                "px-2 py-1 text-xs rounded-md cursor-pointer transition-colors font-medium",
                isActive
                  ? "bg-primary text-primary-foreground"
                  : "bg-muted text-muted-foreground hover:bg-accent hover:text-foreground",
              )}
              onClick={() => setFilter(item.key)}
            >
              {item.icon}{item.label} {count}
            </button>
          )
        })}
      </div>

      <div className="flex gap-1.5 relative" style={{ height: "calc(100vh - 180px)" }}>
        <div
          ref={gutterRef}
          className="shrink-0 flex flex-col"
          style={{ height: "calc(100vh - 180px)" }}
        >
          <OverviewGutter pairs={filteredPairs} scrollToPair={scrollToPair} visibleStartIdx={visibleStartIdx} visibleEndIdx={visibleEndIdx} gutterHeight={gutterHeight} draggingRef={draggingRef} onDragEnd={onDragEnd} />
        </div>
        <div ref={scrollRef} className="flex-1 overflow-y-auto min-h-0">
          <div ref={contentRef} className="space-y-2 relative">
          {virtualRange.start > 0 && (
            <div
              key="__virtual-top"
              style={{ height: virtualRange.start * 500, flexShrink: 0 }}
              aria-hidden
            />
          )}
          {filteredPairs.slice(virtualRange.start, virtualRange.end + 1).map((item) => {
            const globalIdx = item.globalIdx
            return (
              <PairCard
                key={globalIdx}
                pair={item.pair}
                globalIdx={globalIdx}
                showDiff={showDiff}
                selectMode={selectMode}
                selectedSide={selectedSide}
                selectedTurn={selectedTurn}
                manuallyUsedA={manuallyUsedA}
                manuallyUsedB={manuallyUsedB}
                onRemoveManual={handleRemoveManual}
                onSelectTurnA={handleSelectTurnA}
                onSelectTurnB={handleSelectTurnB}
                registerRef={registerRef}
              />
            )
          })}
          {virtualRange.end < filteredPairs.length - 1 && (
            <div
              key="__virtual-bottom"
              style={{ height: (filteredPairs.length - 1 - virtualRange.end) * 500, flexShrink: 0 }}
              aria-hidden
            />
          )}
        </div>
        </div>
      </div>
    </div>
  )
}
