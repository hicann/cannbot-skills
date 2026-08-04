"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect, useMemo, Suspense } from "react"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { VERSION_DISPLAY } from "@/lib/version"
import { CompareOverviewCards } from "@/components/compare/CompareOverviewCards"
import { CompareTokenChart } from "@/components/compare/CompareTokenChart"
import { CompareToolTable } from "@/components/compare/CompareToolTable"
import { CompareTurns } from "@/components/compare/CompareTurns"

interface SessionData {
  taskId: string
  framework: string
  query: string | null
  model: string | null
  totalTokens: number
  totalInputTokens: number
  totalOutputTokens: number
  totalReasoningTokens: number
  totalCacheReadTokens: number
  totalCacheWriteTokens: number
  totalCost: number
  totalLatencyMs: number
  totalLlmCallCount: number
  totalToolCallCount: number
  totalSubagentCount: number
  totalSkillLoadCount: number
  skills: Array<{ skillName: string; version: number | null; invocationCount: number }>
}

interface TurnItem {
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

function computeToolStats(turns: TurnItem[]): Record<string, { count: number; successCount: number }> {
  const stats: Record<string, { count: number; successCount: number }> = {}
  for (const turn of turns) {
    for (const tc of turn.toolCalls) {
      if (!stats[tc.toolName]) stats[tc.toolName] = { count: 0, successCount: 0 }
      stats[tc.toolName].count++
      if (tc.state === "ok" || tc.state === "completed") stats[tc.toolName].successCount++
    }
  }
  return stats
}

type Tab = "overview" | "turns"

export default function ComparePage() {
  return (
    <Suspense fallback={<Loading />}>
      <ComparePageContent />
    </Suspense>
  )
}

function Loading() {
  return (
    <div className="flex h-screen flex-col bg-background">
      <div className="shrink-0 border-b px-4 py-3">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight">Session Compare</h1>
          <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
        </div>
      </div>
      <div className="flex-1 min-h-0 px-4 py-6 overflow-y-auto">
        <p className="text-muted-foreground">Loading...</p>
      </div>
    </div>
  )
}

function ComparePageContent() {
  const searchParams = useSearchParams()
  const sessionsParam = searchParams.get("sessions") ?? ""
  const idsParam = searchParams.get("ids") ?? ""
  const [activeTab, setActiveTab] = useState<Tab>("overview")

  const parsedIds = useMemo(() => {
    if (idsParam) return idsParam.split(",").filter(Boolean)
    return sessionsParam.split(",").filter(Boolean)
  }, [idsParam, sessionsParam])
  const isSessionIdMode = idsParam.length > 0
  const uniqueIds = useMemo(() => new Set(parsedIds), [parsedIds])
  const hasDuplicates = uniqueIds.size < parsedIds.length
  const paramError = hasDuplicates
    ? "Duplicate session IDs detected — please select two different sessions"
    : parsedIds.length !== 2
      ? "Please select exactly 2 sessions to compare"
      : null

  const [sessionA, setSessionA] = useState<SessionData | null>(null)
  const [sessionB, setSessionB] = useState<SessionData | null>(null)
  const [turnsA, setTurnsA] = useState<TurnItem[]>([])
  const [turnsB, setTurnsB] = useState<TurnItem[]>([])
  const [turnsLoading, setTurnsLoading] = useState(false)
  const [turnsLoaded, setTurnsLoaded] = useState(false)
  const [fetchLoading, setFetchLoading] = useState(parsedIds.length === 2)
  const [fetchError, setFetchError] = useState<string | null>(null)

  useEffect(() => {
    if (parsedIds.length !== 2) return

    async function fetchData() {
      try {
        const [resA, resB] = await Promise.all([
          fetch(`/api/observe/session?${isSessionIdMode ? `sessionId=${parsedIds[0]}` : `taskId=${parsedIds[0]}`}`),
          fetch(`/api/observe/session?${isSessionIdMode ? `sessionId=${parsedIds[1]}` : `taskId=${parsedIds[1]}`}`),
        ])

        if (!resA.ok || !resB.ok) {
          setFetchError("Failed to fetch session data")
          setFetchLoading(false)
          return
        }

        setSessionA(await resA.json())
        setSessionB(await resB.json())
        setFetchLoading(false)
      } catch {
        setFetchError("Error fetching data")
        setFetchLoading(false)
      }
    }

    fetchData()
  }, [parsedIds])

  useEffect(() => {
    if (parsedIds.length !== 2 || activeTab !== "turns") return
    if (turnsLoaded) return

    async function fetchTurns() {
      setTurnsLoading(true)
      try {
        const qA = sessionA ? `taskId=${encodeURIComponent(sessionA.taskId)}&framework=${encodeURIComponent(sessionA.framework ?? 'unknown')}` : `taskId=${parsedIds[0]}`
        const qB = sessionB ? `taskId=${encodeURIComponent(sessionB.taskId)}&framework=${encodeURIComponent(sessionB.framework ?? 'unknown')}` : `taskId=${parsedIds[1]}`
        const [turnsResA, turnsResB] = await Promise.all([
          fetch(`/api/observe/session/turns?${qA}&includeContent=true&skipOverhead=true&maxContentLen=0`),
          fetch(`/api/observe/session/turns?${qB}&includeContent=true&skipOverhead=true&maxContentLen=0`),
        ])

        if (!turnsResA.ok || !turnsResB.ok) {
          setFetchError("Failed to fetch turn data")
          setTurnsLoading(false)
          return
        }

        const turnsDataA = await turnsResA.json()
        const turnsDataB = await turnsResB.json()

        setTurnsA(turnsDataA.items ?? [])
        setTurnsB(turnsDataB.items ?? [])
        setTurnsLoading(false)
        setTurnsLoaded(true)
      } catch {
        setFetchError("Error fetching turn data")
        setTurnsLoading(false)
      }
    }

    fetchTurns()
  }, [parsedIds, activeTab, turnsLoaded])

  const error = paramError ?? fetchError
  const loading = fetchLoading

  const TABS: Array<{ key: Tab; label: string }> = [
    { key: "overview", label: "Overview" },
    { key: "turns", label: "Turn-by-Turn" },
  ]

  if (loading) {
    return (
      <div className="flex h-screen flex-col bg-background">
        <div className="shrink-0 border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight">Session Compare</h1>
            <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
          </div>
        </div>
        <div className="flex-1 min-h-0 px-4 py-6 overflow-y-auto">
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    )
  }

  if (error || !sessionA || !sessionB) {
    return (
      <div className="flex h-screen flex-col bg-background">
        <div className="shrink-0 border-b px-4 py-3">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold tracking-tight">Session Compare</h1>
            <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
          </div>
        </div>
        <div className="flex-1 min-h-0 px-4 py-6 overflow-y-auto">
          <p className="text-red-600 dark:text-red-400">{error ?? "Missing session data"}</p>
          <Link href="/">
            <Button variant="outline" size="sm" className="mt-4">← Back to Home</Button>
          </Link>
        </div>
      </div>
    )
  }

  const toolStatsA = computeToolStats(turnsA)
  const toolStatsB = computeToolStats(turnsB)

  return (
    <div className="flex h-screen flex-col bg-background">
      <div className="shrink-0 border-b px-4 py-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Link href="/">
              <Button variant="ghost" size="sm" className="h-7 px-2">←</Button>
            </Link>
            <h1 className="text-lg font-bold tracking-tight">Session Compare</h1>
            <span className="text-xs text-muted-foreground">{VERSION_DISPLAY}</span>
            <div className="flex items-center gap-2 ml-2">
              <Badge variant="blue" className="text-xs">A: {sessionA.query ?? sessionA.taskId}</Badge>
              <span className="text-muted-foreground text-xs">vs</span>
              <Badge variant="orange" className="text-xs">B: {sessionB.query ?? sessionB.taskId}</Badge>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-4 text-sm mt-2">
          <span className="text-muted-foreground">Model: <span className="font-medium">{sessionA.model ?? "unknown"}</span></span>
          <span className="text-muted-foreground">Tokens: <span className="font-medium tabular-nums">{sessionA.totalTokens}</span></span>
          <span className="text-muted-foreground">Tools: <span className="font-medium tabular-nums">{sessionA.totalToolCallCount}</span></span>
          <span className="text-muted-foreground">→</span>
          <span className="text-muted-foreground">Model: <span className="font-medium">{sessionB.model ?? "unknown"}</span></span>
          <span className="text-muted-foreground">Tokens: <span className="font-medium tabular-nums">{sessionB.totalTokens}</span></span>
          <span className="text-muted-foreground">Tools: <span className="font-medium tabular-nums">{sessionB.totalToolCallCount}</span></span>
        </div>

        <div className="flex items-center gap-1 border-b -mb-px mt-3">
          {TABS.map(tab => (
            <button
              key={tab.key}
              className={cn(
                "px-3 py-1.5 text-sm font-medium cursor-pointer border-b-2 -mb-px transition-colors",
                activeTab === tab.key
                  ? "border-primary text-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground"
              )}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto">
        {activeTab === "overview" && (
          <div className="px-6 py-4 space-y-4">
            <CompareOverviewCards sessionA={sessionA} sessionB={sessionB} />
            <CompareTokenChart
              tokenA={{
                inputTokens: sessionA.totalInputTokens,
                outputTokens: sessionA.totalOutputTokens,
                reasoningTokens: sessionA.totalReasoningTokens,
                cacheReadTokens: sessionA.totalCacheReadTokens,
                cacheWriteTokens: sessionA.totalCacheWriteTokens,
              }}
              tokenB={{
                inputTokens: sessionB.totalInputTokens,
                outputTokens: sessionB.totalOutputTokens,
                reasoningTokens: sessionB.totalReasoningTokens,
                cacheReadTokens: sessionB.totalCacheReadTokens,
                cacheWriteTokens: sessionB.totalCacheWriteTokens,
              }}
            />
            <CompareToolTable toolStatsA={toolStatsA} toolStatsB={toolStatsB} />
          </div>
        )}

        <div className={cn("px-4 py-4 h-full", activeTab !== "turns" && "hidden")}>
          {turnsLoading ? (
            <p className="text-muted-foreground">正在获取 turn 数据...</p>
          ) : (
            <CompareTurns turnsA={turnsA} turnsB={turnsB} />
          )}
        </div>
      </div>
    </div>
  )
}
