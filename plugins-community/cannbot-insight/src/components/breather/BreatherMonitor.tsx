"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useEffect, useRef, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Checkbox } from "@/components/ui/checkbox"
import { cn } from "@/lib/utils"
import {
  ActivityIcon,
  ArrowRightIcon,
  PlugIcon,
  AlertTriangleIcon,
  CheckCircleIcon,
  RefreshCwIcon,
  BarChart3Icon,
} from "lucide-react"

interface SessionInfo {
  id: string
  title: string | null
  createdAt: string
  directory: string | null
  parentID: string | null
  model: string | null
  agent: string | null
}

interface StatusData {
  status: {
    sessionId: string
    sessionTitle: string | null
    currentContextPct: number
    currentInputTokens: number
    contextWindowLimit: number
    model: string | null
  } | null
  history: Array<{ turnIndex: number; contextPct: number; timestamp: string; hasSubagentCall: boolean }>
  handoffLinks?: HandoffLinks
  childSessionIds?: string[]
  childSessions?: Array<{ id: string; title: string | null }>
  projectPath?: string
}

interface ConnectResult {
  connected: boolean
  baseUrl?: string
  error?: string
  hint?: string
  sessions?: SessionInfo[]
}

interface HandoffResult {
  success: boolean
  error?: string
  handoffMode?: "pipeline" | "general"
  handoffFilePath?: string
  handoffSession?: { id: string; title: string }
  newSession?: { id: string; title: string }
  continuationSession?: { id: string; title: string }
  originalSessionId?: string
  projectPath?: string
  operatorName?: string
  originalTitle?: string
  handoffNum?: number
  stageInfo?: {
    stageName: string
    lastCompletedStep: string
    lastCompletedSubagent: string
    nextStep: string
  }
  generalInfo?: {
    taskSummary: string
    filesModifiedCount: number
  }
}

interface HandoffLinks {
  parent: { id: string; title: string | null } | null
  children: Array<{ id: string; title: string | null }>
}

type Phase = "idle" | "connecting" | "selecting" | "monitoring" | "pending" | "handoff-executing" | "handoff-running" | "handoff-done"
type HandoffStep = "handoff-working" | "creating-continuation"

const REFRESH_TRIGGER_EVENTS = [
  "message.updated",
  "session.idle",
  "session.updated",
  "session.created",
  "session.next.tool.success",
  "session.next.tool.failed",
  "session.next.text.ended",
  "session.next.step.ended",
  "session.next.step.failed",
]

export function BreatherMonitor() {
  const [phase, setPhase] = useState<Phase>("idle")
  const [host, setHost] = useState("localhost")
  const [port, setPort] = useState("15031")
  const [threshold, setThreshold] = useState("70")
  const [autoHandoff, setAutoHandoff] = useState(false)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [contextPct, setContextPct] = useState(0)
  const [inputTokens, setInputTokens] = useState(0)
  const [contextWindowLimit, setContextWindowLimit] = useState(0)
  const [history, setHistory] = useState<Array<{ turnIndex: number; contextPct: number; timestamp: string; hasSubagentCall: boolean }>>([])
  const [activeChildTitles, setActiveChildTitles] = useState<Map<string, string | null>>(new Map())
  const [sessionTitle, setSessionTitle] = useState<string | null>(null)
  const [model, setModel] = useState<string | null>(null)
  const [connectError, setConnectError] = useState<string | null>(null)
  const [availableSessions, setAvailableSessions] = useState<SessionInfo[]>([])
  const [handoffResult, setHandoffResultState] = useState<HandoffResult | null>(null)
  const setHandoffResult = useCallback((v: HandoffResult | null | ((prev: HandoffResult | null) => HandoffResult | null)) => {
    if (typeof v === "function") {
      setHandoffResultState(prev => {
        const next = v(prev)
        handoffResultRef.current = next
        return next
      })
    } else {
      handoffResultRef.current = v
      setHandoffResultState(v)
    }
  }, [])
  const [handoffLinks, setHandoffLinks] = useState<HandoffLinks>({ parent: null, children: [] })
  const [safePointInfo, setSafePointInfo] = useState<{ canHandoff: boolean; reason: string } | null>(null)
  const [handoffStep, setHandoffStep] = useState<HandoffStep>("handoff-working")
  const [lastEvent, setLastEvent] = useState<string>("")
  const eventSourceRef = useRef<EventSource | null>(null)
  const phaseRef = useRef<Phase>(phase)
  const sessionIdRef = useRef<string | null>(sessionId)
  const isCreatingContinuationRef = useRef(false)
  const pollAbortRef = useRef(false)
  const handoffWaitingSidRef = useRef<string | null>(null)

  const activeChildIdsRef = useRef<Set<string>>(new Set())
  const handoffResultRef = useRef<HandoffResult | null>(null)
  const hostRef = useRef(host)
  const portRef = useRef(port)

  phaseRef.current = phase
  sessionIdRef.current = sessionId
  hostRef.current = host
  portRef.current = port

  const eventsUrl = `/api/breather/events?host=${encodeURIComponent(host)}&port=${encodeURIComponent(port)}`

  const refreshStatus = useCallback(async () => {
    if (!sessionId) return
    try {
      const res = await fetch(`/api/breather/status?host=${encodeURIComponent(host)}&port=${encodeURIComponent(port)}&sessionId=${encodeURIComponent(sessionId)}`)
      if (res.ok) {
        const data: StatusData = await res.json()
        if (data.status) {
          setContextPct(data.status.currentContextPct)
          setInputTokens(data.status.currentInputTokens)
          setContextWindowLimit(data.status.contextWindowLimit)
          setSessionTitle(data.status.sessionTitle)
          setModel(data.status.model)
        }
        setHistory(data.history)
        setHandoffLinks(data.handoffLinks ?? { parent: null, children: [] })
        if (data.childSessionIds) {
          const childIds = new Set(data.childSessionIds)
          for (const id of [...activeChildIdsRef.current]) {
            if (!childIds.has(id)) {
              activeChildIdsRef.current.delete(id)
            }
          }
        }
        if (data.childSessions) {
          const titleMap = new Map<string, string | null>()
          for (const cs of data.childSessions) {
            if (activeChildIdsRef.current.has(cs.id)) {
              titleMap.set(cs.id, cs.title)
            }
          }
          setActiveChildTitles(titleMap)
        }
      }
    } catch {}
  }, [host, port, sessionId])

  const startSSE = useCallback(() => {
    if (eventSourceRef.current) eventSourceRef.current.close()

    const es = new EventSource(eventsUrl)
    eventSourceRef.current = es

    es.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        const eventType = payload?.type ?? payload?.payload?.type ?? ""
        const eventSessionId = payload?.payload?.properties?.sessionID ?? payload?.properties?.sessionID ?? ""

        setLastEvent(`${eventType} | sid=${eventSessionId}`)

        const isMainSession = eventSessionId === sessionIdRef.current
        const isActiveChild = activeChildIdsRef.current.has(eventSessionId)

        if ((isMainSession || isActiveChild) && REFRESH_TRIGGER_EVENTS.includes(eventType)) refreshStatus()

        if (eventType === "session.created") {
          refreshStatus()
          if (autoHandoff && phaseRef.current === "pending") {
            activeChildIdsRef.current.add(eventSessionId)
          }
        }

        if (autoHandoff && eventType === "session.idle" && isActiveChild && phaseRef.current === "pending") {
          activeChildIdsRef.current.delete(eventSessionId)
          if (activeChildIdsRef.current.size === 0 && !isCreatingContinuationRef.current) {
            isCreatingContinuationRef.current = true
            checkSafePointAndHandoff()
          }
        }

        if (autoHandoff && eventType === "session.idle" && isMainSession && phaseRef.current === "pending" && !isCreatingContinuationRef.current) {
          activeChildIdsRef.current = new Set()
          isCreatingContinuationRef.current = true
          checkSafePointAndHandoff()
        }

        if (eventType === "session.idle" && handoffWaitingSidRef.current && eventSessionId === handoffWaitingSidRef.current) {
          fetch("/api/breather/handoff-status", {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ sessionId: eventSessionId, status: "idle", host: hostRef.current, port: parseInt(portRef.current) }),
          }).catch(() => {})
          if (phaseRef.current === "handoff-running" && !pollAbortRef.current && !isCreatingContinuationRef.current) {
            checkFlagAndComplete(eventSessionId)
          }
        }
      } catch {}
    }

    es.onerror = () => {
      es.close()
      eventSourceRef.current = null
      setTimeout(() => {
        const p = phaseRef.current
        if (p === "monitoring" || p === "pending" || p === "selecting" || p === "handoff-running") startSSE()
      }, 5000)
    }
  }, [eventsUrl, refreshStatus, phase])

  const stopSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
  }, [])

  async function checkSafePointAndHandoff() {
    const sid = sessionIdRef.current
    if (!sid) { isCreatingContinuationRef.current = false; return }

    try {
      const res = await fetch(`/api/breather/safe-point?host=${encodeURIComponent(hostRef.current)}&port=${encodeURIComponent(portRef.current)}&sessionId=${encodeURIComponent(sid)}`)
      if (res.ok) {
        const data = await res.json()
        setSafePointInfo(data)
        if (data.canHandoff) {
          try {
            await fetch(`http://${hostRef.current}:${parseInt(portRef.current)}/session/${encodeURIComponent(sid)}/abort`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
            })
          } catch {}
          handleHandoff()
        } else {
          isCreatingContinuationRef.current = false
        }
      } else {
        isCreatingContinuationRef.current = false
      }
    } catch {
      isCreatingContinuationRef.current = false
    }
  }

  // Auto-mark pending when context exceeds threshold (only if autoHandoff enabled)
  useEffect(() => {
    if (!autoHandoff) return
    const thresholdValue = parseInt(threshold) || 70
    if (phase === "monitoring" && contextPct >= thresholdValue) {
      setPhase("pending")
    }
    if (phase === "pending" && contextPct < thresholdValue) {
      setPhase("monitoring")
    }
  }, [contextPct, phase, threshold, autoHandoff])

  useEffect(() => {
    if (phase === "monitoring" || phase === "pending" || phase === "handoff-running") {
      refreshStatus()
      startSSE()
    } else {
      stopSSE()
    }
    return stopSSE
  }, [phase, refreshStatus, startSSE, stopSSE])

  async function handleConnect() {
    setPhase("connecting")
    setConnectError(null)
    try {
      const res = await fetch("/api/breather/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ host, port: parseInt(port) }),
      })
      const data: ConnectResult = await res.json()
      if (data.connected && data.sessions?.length) {
        setAvailableSessions(data.sessions)
        setPhase("selecting")
      } else if (data.connected) {
        setConnectError("没有找到可用的 session")
        setPhase("idle")
      } else {
        setConnectError(data.error ?? "连接失败")
        setPhase("idle")
      }
    } catch {
      setConnectError("网络请求失败")
      setPhase("idle")
    }
  }

  function handleSelectSession(sid: string) {
    isCreatingContinuationRef.current = false
    pollAbortRef.current = false
    handoffWaitingSidRef.current = null
    activeChildIdsRef.current = new Set()
    setHandoffResult(null)
    setHandoffLinks({ parent: null, children: [] })
    setSafePointInfo(null)
    setHandoffStep("handoff-working")
    setSessionId(sid)
    setContextPct(0)
    setInputTokens(0)
    setContextWindowLimit(0)
    setHistory([])
    setPhase("monitoring")
    startSSE()
  }

  async function checkFlagAndComplete(eventSessionId: string) {
    if (isCreatingContinuationRef.current) return
    isCreatingContinuationRef.current = true
    const result = handoffResultRef.current
    const pp = result?.projectPath ?? ""
    const op = result?.operatorName ?? ""
    const hNum = result?.handoffNum ?? 0
    const origId = result?.originalSessionId ?? ""
    const doneFilePath = `${pp}/operators/${op}/docs/SESSION-HANDOFF-${origId.slice(0, 12)}-${hNum}.done`
    try {
      const checkRes = await fetch(`/api/breather/check-file?path=${encodeURIComponent(doneFilePath)}`)
      const checkData = await checkRes.json()
      if (!checkData.exists) {
        isCreatingContinuationRef.current = false
        return
      }
    } catch {
      isCreatingContinuationRef.current = false
      return
    }
    const sid = handoffWaitingSidRef.current
    handoffWaitingSidRef.current = null
    onHandoffSessionIdle(sid!)
  }

  async function onHandoffSessionIdle(handoffSid: string) {
    setHandoffStep("creating-continuation")
    try {
      const res = await fetch("/api/breather/complete", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ handoffSessionId: handoffSid, originalSessionId: sessionIdRef.current, host: hostRef.current, port: parseInt(portRef.current) }),
      })
      const data = await res.json()
      if (data.success) {
        const contSessionId = data.continuationSession?.id
        setHandoffResult(prev => prev ? { ...prev, continuationSession: data.continuationSession } : data)
        if (contSessionId) {
          setSessionId(contSessionId)
          setSessionTitle(data.continuationSession?.title ?? null)
          setContextPct(0)
          setHistory([])
        }
        handoffWaitingSidRef.current = null
        isCreatingContinuationRef.current = false
        pollAbortRef.current = false
        activeChildIdsRef.current = new Set()
        setPhase("handoff-done")
        refreshStatus()
      } else {
        isCreatingContinuationRef.current = false
        setHandoffResult(prev => prev ? { ...prev, error: data.error } : { success: false, error: data.error })
        setPhase("monitoring")
      }
    } catch {
      isCreatingContinuationRef.current = false
      setHandoffResult({ success: false, error: "触发 complete 失败: 网络错误" })
      setPhase("monitoring")
    }
  }

  async function handleHandoff() {
    if (!sessionId) return
    setPhase("handoff-executing")
    try {
      const res = await fetch("/api/breather/handoff", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sessionId, host, port: parseInt(port), contextPct }),
      })
      const data: HandoffResult = await res.json()
      setHandoffResult(data)
      if (data.success) {
        const handoffSid = data.handoffSession?.id ?? data.newSession?.id
        if (handoffSid) {
          setHandoffStep("handoff-working")
          setPhase("handoff-running")
          isCreatingContinuationRef.current = false
          handoffWaitingSidRef.current = handoffSid
          pollAbortRef.current = false
          startSSE()
        } else {
          setPhase("handoff-done")
          isCreatingContinuationRef.current = false
          refreshStatus()
        }
      } else {
        setPhase(autoHandoff ? "pending" : "monitoring")
      }
    } catch {
      setHandoffResult({ success: false, error: "网络请求失败" })
      setPhase(autoHandoff ? "pending" : "monitoring")
      isCreatingContinuationRef.current = false
    }
  }

  function handleDisconnect() {
    stopSSE()
    pollAbortRef.current = true
    handoffWaitingSidRef.current = null
    setPhase("idle")
    setSessionId(null)
    setContextPct(0)
    setInputTokens(0)
    setContextWindowLimit(0)
    setHistory([])
    setConnectError(null)
    setHandoffResult(null)
    setHandoffLinks({ parent: null, children: [] })
    setAvailableSessions([])
    setSafePointInfo(null)
  }

  function handleChangeSession() {
    stopSSE()
    isCreatingContinuationRef.current = false
    pollAbortRef.current = false
    handoffWaitingSidRef.current = null
    activeChildIdsRef.current = new Set()
    setSafePointInfo(null)
    setHandoffStep("handoff-working")
    setSessionId(null)
    setContextPct(0)
    setInputTokens(0)
    setContextWindowLimit(0)
    setHistory([])
    setHandoffResult(null)
    setHandoffLinks({ parent: null, children: [] })
    setPhase("selecting")
    handleConnect()
  }

  const thresholdNum = parseInt(threshold) || 70
  const overThreshold = contextPct >= thresholdNum
  const pctColor = contextPct >= 90 ? "text-red-600 dark:text-red-400" : contextPct >= thresholdNum ? "text-yellow-600 dark:text-yellow-400" : "text-emerald-600 dark:text-emerald-400"
  const barColor = contextPct >= 90 ? "bg-red-500" : contextPct >= thresholdNum ? "bg-yellow-500" : "bg-emerald-500"

  const formatTokenCount = (n: number) => n >= 1000 ? `${(n / 1000).toFixed(1)}K` : String(n)

  return (
    <div className="flex flex-col min-h-screen bg-background">
      <main className="flex w-full max-w-5xl flex-col gap-4 px-6 py-6 mx-auto">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <BarChart3Icon className="size-5 text-pink-500" />
            <h1 className="text-xl font-bold tracking-tight text-foreground">上下文监控</h1>
          </div>
          {phase !== "idle" && (
            <Button variant="outline" size="sm" className="text-xs gap-1" onClick={handleDisconnect}>
              <PlugIcon className="size-3" />断开连接
            </Button>
          )}
        </div>

        {phase === "idle" && (
          <Card>
            <CardHeader><CardTitle>连接 OpenCode 实例</CardTitle></CardHeader>
            <CardContent className="space-y-3">
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="text-xs text-muted-foreground">地址</label>
                  <Input value={host} onChange={(e) => setHost(e.target.value)} placeholder="localhost" className="mt-1" />
                </div>
                <div>
                  <label className="text-xs text-muted-foreground">端口</label>
                  <Input value={port} onChange={(e) => setPort(e.target.value)} placeholder="15031" className="mt-1" />
                </div>
              </div>
              <div>
                <label className="text-xs text-muted-foreground">上下文阈值 (%)</label>
                <Input value={threshold} onChange={(e) => setThreshold(e.target.value)} placeholder="70" type="number" min="10" max="100" className="mt-1" />
              </div>
              <div className="flex items-start gap-2">
                <Checkbox checked={autoHandoff} onCheckedChange={(v) => setAutoHandoff(v === true)} className="mt-1" />
                <div>
                  <label className="text-sm font-medium">启动自动 handoff</label>
                  <div className="text-xs text-muted-foreground mt-0.5">试验性功能，还没完成</div>
                </div>
              </div>
              {connectError && (
                <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/20 px-3 py-2 rounded-md">
                  <AlertTriangleIcon className="size-4" />{connectError}
                </div>
              )}
              <div className="flex items-center gap-2 text-xs text-muted-foreground bg-muted px-3 py-2 rounded-md">
                <AlertTriangleIcon className="size-3" />
                请确保 OpenCode 使用指定端口启动: <code className="font-mono bg-background px-1 rounded">opencode --port {port}</code>
              </div>
              <Button onClick={handleConnect} className="gap-1">
                <PlugIcon className="size-4" />确认连接
              </Button>
            </CardContent>
          </Card>
        )}

        {phase === "connecting" && (
          <Card>
            <CardContent className="flex items-center justify-center py-8">
              <RefreshCwIcon className="size-5 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">正在连接...</span>
            </CardContent>
          </Card>
        )}

        {phase === "selecting" && availableSessions.length > 0 && (
          <Card>
            <CardHeader><CardTitle>选择要监控的 Session</CardTitle></CardHeader>
            <CardContent className="space-y-2 max-h-[400px] overflow-y-auto">
              {availableSessions.map((s) => (
                <button
                  key={s.id}
                  onClick={() => handleSelectSession(s.id)}
                  className="flex items-center justify-between w-full px-3 py-2 text-sm rounded-md border hover:bg-accent transition-colors"
                >
                  <div className="flex items-center gap-2 min-w-0">
                    <ActivityIcon className="size-4 text-muted-foreground shrink-0" />
                    <span className="truncate font-medium">{s.title ?? s.id.slice(0, 12)}</span>
                  </div>
                  <div className="flex items-center gap-2 shrink-0 ml-2">
                    {s.model && <Badge variant="blue" className="text-xs">{s.model}</Badge>}
                    {s.agent && <Badge variant="outline" className="text-xs">{s.agent}</Badge>}
                  </div>
                </button>
              ))}
            </CardContent>
          </Card>
        )}

        {(phase === "monitoring" || phase === "pending" || phase === "handoff-executing" || phase === "handoff-running" || phase === "handoff-done") && sessionId && (
          <>
            <Card size="sm">
              <CardContent className="py-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <ActivityIcon className={cn("size-4", phase === "monitoring" ? "text-emerald-500 animate-pulse" : phase === "pending" ? "text-yellow-500 animate-pulse" : phase === "handoff-executing" ? "text-orange-500 animate-pulse" : "text-blue-500")} />
                    <span className="text-sm font-medium truncate">{sessionTitle ?? sessionId.slice(0, 12)}</span>
                    {model && <Badge variant="blue" className="text-xs">{model}</Badge>}
                    {phase === "pending" && autoHandoff && <Badge variant="outline" className="text-xs text-yellow-600 border-yellow-400">等待安全点</Badge>}
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="ghost" size="sm" className="text-xs gap-1" onClick={refreshStatus}>
                      <RefreshCwIcon className="size-3" />刷新
                    </Button>
                    <Button variant="outline" size="sm" className="text-xs gap-1" onClick={handleChangeSession}>
                      <ArrowRightIcon className="size-3" />切换 Session
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader><CardTitle>实时上下文占用</CardTitle></CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className={cn("text-3xl font-bold tabular-nums", pctColor)}>{contextPct.toFixed(1)}%</span>
                  <span className="text-sm text-muted-foreground">阈值: {thresholdNum}% | {formatTokenCount(inputTokens)}/{formatTokenCount(contextWindowLimit)} tokens</span>
                </div>
                <div className="relative h-4 rounded-full bg-muted overflow-hidden">
                  <div className={cn("h-full rounded-full transition-all duration-500", barColor)} style={{ width: `${Math.min(contextPct, 100)}%` }} />
                  <div className="absolute top-0 bottom-0 w-0.5 bg-yellow-500" style={{ left: `${thresholdNum}%` }} />
                </div>
                {phase === "pending" && autoHandoff && (
                  <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950/20 px-3 py-2 rounded-md">
                    <AlertTriangleIcon className="size-4 animate-pulse" />
                    上下文已达 {contextPct.toFixed(1)}%，超过阈值 {thresholdNum}%。正在等待安全点后自动执行 handoff...
                  </div>
                )}
                {phase === "pending" && activeChildTitles.size > 0 && (
                  <div className="text-xs text-muted-foreground bg-muted px-3 py-2 rounded-md">
                    等待子 session idle: {[...activeChildTitles.entries()].map(([id, title]) => title ?? id.slice(0, 12)).join(", ")}
                  </div>
                )}
                {overThreshold && phase === "monitoring" && !autoHandoff && (
                  <div className="flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-950/20 px-3 py-2 rounded-md">
                    <AlertTriangleIcon className="size-4" />
                    上下文已达 {contextPct.toFixed(1)}%，超过阈值 {thresholdNum}%。如需 handoff 请启用自动 handoff 选项。
                  </div>
                )}
                {safePointInfo && phase === "pending" && autoHandoff && !safePointInfo.canHandoff && (
                  <div className="text-xs text-muted-foreground bg-muted px-3 py-2 rounded-md">
                    安全点检查: {safePointInfo.reason}
                  </div>
                )}
                {!overThreshold && phase === "monitoring" && (
                  <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
                    <CheckCircleIcon className="size-4" />上下文正常，事件驱动监控中
                  </div>
                )}
              </CardContent>
            </Card>

            {history.length > 0 && (
              <Card size="sm">
                <CardHeader><CardTitle>上下文增长趋势</CardTitle></CardHeader>
                <CardContent>
                  <div className="flex items-end gap-2 mb-1 text-xs text-muted-foreground">
                    <span>纵轴: 上下文占用 (%)</span>
                    <span className="ml-auto">横轴: 时间</span>
                  </div>
                  <div className="flex items-center gap-3 text-xs mb-2">
                    <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald-500 inline-block" />上下文占用</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-yellow-500 inline-block" style={{ borderTop: "1px dashed #eab308" }} />阈值 ({thresholdNum}%)</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-emerald-500 inline-block" />普通回复</span>
                    <span className="flex items-center gap-1"><span className="w-2 h-2 inline-block" style={{ background: "#8b5cf6" }} />发起 subagent</span>
                  </div>
                  <svg width="100%" height={220} viewBox="0 0 440 220" preserveAspectRatio="xMidYMid meet" className="min-w-[300px]">
                    {(() => {
                      const maxPct = Math.max(...history.map((h) => h.contextPct), thresholdNum + 10, 100)
                      const padTop = 15, padBottom = 40, padLeft = 45, padRight = 25
                      const plotH = 220 - padTop - padBottom, plotW = 440 - padLeft - padRight
                      const toY = (pct: number) => padTop + plotH - (pct / maxPct) * plotH
                      const toX = (i: number) => padLeft + (i / Math.max(history.length - 1, 1)) * plotW
                      const thresholdY = toY(thresholdNum)
                      const polyline = history.map((h, i) => `${toX(i)},${toY(h.contextPct)}`).join(" ")

                      const xTicks = history.length <= 8
                        ? history.map((h, i) => ({ x: toX(i), label: new Date(h.timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }) }))
                        : [0, Math.floor(history.length / 4), Math.floor(history.length / 2), Math.floor(3 * history.length / 4), history.length - 1].map(i => ({
                          x: toX(i),
                          label: new Date(history[i].timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" }),
                        }))

                      return (
                        <>
                          {[0, 25, 50, 75, 100].map((v) => (
                            <g key={v}>
                              <line x1={padLeft} y1={toY(v)} x2={440 - padRight} y2={toY(v)} stroke="#e5e7eb" strokeWidth={0.5} />
                              <text x={padLeft - 5} y={toY(v) + 4} textAnchor="end" fontSize={9} fill="#6b7280">{v}%</text>
                            </g>
                          ))}
                          <line x1={padLeft} y1={thresholdY} x2={440 - padRight} y2={thresholdY} stroke="#eab308" strokeWidth={1.5} strokeDasharray="6 3" />
                          <text x={440 - padRight + 3} y={thresholdY + 4} fontSize={8} fill="#eab308">阈值</text>
                          <polyline points={polyline} fill="none" stroke="#10b981" strokeWidth={2} strokeLinejoin="round" />
                          {xTicks.map((tick, idx) => (
                            <g key={idx}>
                              <line x1={tick.x} y1={padTop} x2={tick.x} y2={220 - padBottom} stroke="#e5e7eb" strokeWidth={0.5} />
                              <text x={tick.x} y={220 - padBottom + 14} textAnchor="middle" fontSize={8} fill="#6b7280">{tick.label}</text>
                            </g>
                          ))}
                          {history.map((h, i) => {
                            const isSub = h.hasSubagentCall
                            const fill = h.contextPct >= thresholdNum ? "#eab308" : "#10b981"
                            const subFill = isSub ? "#8b5cf6" : fill
                            const r = history.length > 20 ? 2 : history.length > 10 ? 3 : 4
                            const timeLabel = new Date(h.timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", second: "2-digit" })
                            const label = isSub ? `轮次 ${i + 1} | ${timeLabel} | ${h.contextPct}% | 发起 subagent` : `轮次 ${i + 1} | ${timeLabel} | ${h.contextPct}%`
                            return (
                              <g key={i}>
                                {isSub
                                  ? <rect x={toX(i) - r} y={toY(h.contextPct) - r} width={2 * r} height={2 * r} fill={subFill}>
                                      <title>{label}</title>
                                    </rect>
                                  : <circle cx={toX(i)} cy={toY(h.contextPct)} r={r} fill={subFill}>
                                      <title>{label}</title>
                                    </circle>
                                }
                              </g>
                            )
                          })}
                        </>
                      )
                    })()}
                  </svg>
                </CardContent>
              </Card>
            )}

            {phase === "handoff-executing" && (
              <Card>
                <CardContent className="flex items-center justify-center py-6">
                  <RefreshCwIcon className="size-5 animate-spin text-orange-500" />
                  <span className="ml-2 text-muted-foreground">Handoff 执行中：abort 旧 session → 提取上下文 → 写交接文件 → 创建新 session → 发送 resume prompt</span>
                </CardContent>
              </Card>
            )}

            {phase === "handoff-running" && (
              <Card>
                <CardContent className="space-y-3 py-4">
                  <div className="flex items-center gap-2 text-sm text-orange-600 dark:text-orange-400">
                    <RefreshCwIcon className="size-4 animate-spin" />
                    <span className="font-medium">Handoff 进行中</span>
                  </div>
                  <div className="w-full bg-muted rounded-full h-2">
                    <div className={cn("h-2 rounded-full transition-all duration-500", handoffStep === "handoff-working" ? "bg-orange-500 w-1/2" : "bg-orange-500 w-full")} />
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {handoffStep === "handoff-working" && "等待 handoff session 完成..."}
                    {handoffStep === "creating-continuation" && "正在创建 continuation session..."}
                  </div>
                  {lastEvent && <div className="text-xs text-muted-foreground font-mono">SSE: {lastEvent}</div>}
                </CardContent>
              </Card>
            )}

            {phase === "handoff-done" && handoffResult?.success && (
              <Card>
                <CardHeader><CardTitle>Handoff 交接信息已生成</CardTitle></CardHeader>
                <CardContent className="space-y-2">
                  <div className="flex items-center gap-2 text-sm text-emerald-600 dark:text-emerald-400">
                    <CheckCircleIcon className="size-4" />交接文件已写入项目目录
                  </div>
                  <div className="text-xs text-muted-foreground">
                    模式: {handoffResult.handoffMode === "pipeline" ? "流水线 (LOG.md)" : "通用 (消息提取)"}
                  </div>
                  {handoffResult.handoffFilePath && (
                    <div className="text-xs text-muted-foreground">
                      交接文件: <span className="font-mono">{handoffResult.handoffFilePath}</span>
                    </div>
                  )}
                  {handoffResult.stageInfo && (
                    <div className="bg-muted px-3 py-2 rounded-md text-xs space-y-1">
                      <div><span className="text-muted-foreground">恢复阶段:</span> {handoffResult.stageInfo.stageName}</div>
                      <div><span className="text-muted-foreground">已完成到:</span> {handoffResult.stageInfo.lastCompletedSubagent}（{handoffResult.stageInfo.lastCompletedStep}）</div>
                      <div><span className="text-muted-foreground">下一步:</span> {handoffResult.stageInfo.nextStep}</div>
                    </div>
                  )}
                  {handoffResult.generalInfo && (
                    <div className="bg-muted px-3 py-2 rounded-md text-xs space-y-1">
                      <div><span className="text-muted-foreground">任务:</span> {handoffResult.generalInfo.taskSummary}</div>
                      <div><span className="text-muted-foreground">已修改文件数:</span> {handoffResult.generalInfo.filesModifiedCount}</div>
                    </div>
                  )}
                  <div className="bg-blue-50 dark:bg-blue-950 px-3 py-2 rounded-md text-xs space-y-2 border border-blue-200 dark:border-blue-800">
                    <div className="font-semibold text-blue-700 dark:text-blue-300">Handoff 已自动完成</div>
                    <div>新 session 已创建并自动发送了 resume prompt，请在 OpenCode TUI 中查看新 session 的运行状态。</div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" className="text-xs" onClick={() => { 
                       const origId = handoffResult?.originalSessionId ?? sessionId
                       setSessionId(origId)
                       setPhase("monitoring"); setHandoffResult(null) 
                       refreshStatus()
                       startSSE()
                    }}>
                      继续监控原 session
                    </Button>
                    <Button variant="default" size="sm" className="text-xs gap-1" onClick={() => {
                       const newId = handoffResult?.continuationSession?.id ?? handoffResult?.newSession?.id
                       if (newId) {
                         setSessionId(newId)
                         setSessionTitle(handoffResult?.continuationSession?.title ?? handoffResult?.newSession?.title ?? null)
                         setPhase("monitoring")
                         setHandoffResult(null)
                         setSafePointInfo(null)
                         setContextPct(0)
                         setHistory([])
                         refreshStatus()
                         startSSE()
                       }
                     }}>
                      <ArrowRightIcon className="size-3" />监控新 session
                    </Button>
                  </div>
                </CardContent>
              </Card>
            )}

            {(handoffLinks.parent || handoffLinks.children.length > 0) && (
              <Card size="sm">
                <CardHeader><CardTitle>Handoff 关系</CardTitle></CardHeader>
                <CardContent className="space-y-2">
                  {handoffLinks.parent && (
                    <div className="flex items-center gap-1 text-xs">
                      <span className="text-muted-foreground">前序:</span>
                      <Badge variant="outline" className="text-xs font-mono">{handoffLinks.parent.title ?? handoffLinks.parent.id.slice(0, 12)}</Badge>
                      <ArrowRightIcon className="size-3 text-muted-foreground" />
                      <Badge variant="blue" className="text-xs font-mono">{sessionTitle ?? sessionId.slice(0, 8)}</Badge>
                    </div>
                  )}
                  {handoffLinks.children.length > 0 && (
                    <div className="space-y-1">
                      <span className="text-xs text-muted-foreground">后续 session:</span>
                      {handoffLinks.children.map((child) => (
                        <div key={child.id} className="flex items-center gap-1 text-xs">
<Badge variant="blue" className="text-xs font-mono">{sessionTitle ?? sessionId.slice(0, 12)}</Badge>
                          <ArrowRightIcon className="size-3 text-muted-foreground" />
                          <Badge variant="outline" className="text-xs font-mono">{child.title ?? child.id.slice(0, 12)}</Badge>
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            )}
          </>
        )}
      </main>
    </div>
  )
}
