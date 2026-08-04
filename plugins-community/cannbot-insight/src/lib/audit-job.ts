// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { Analysis } from "@/components/observe/WorkflowFlowChart"
import type { V4Analysis, V4AuditMeta } from "@/components/observe/WorkflowAgentAudit"
import type { AgentVersion } from "@/lib/agent-version"

export type AnyAnalysis = Analysis | V4Analysis

export interface AuditProgressItem {
  stage: string
  msg: string
  round?: number
}

export interface AuditJobState {
  taskId: string
  generating: boolean
  genError: string | null
  progress: AuditProgressItem[]
  elapsed: number
  startedAt: number | null
  result: AnyAnalysis | null
}

export interface AuditProvider {
  apiKey: string
  baseUrl: string
  model: string
}

export interface StartAuditJobOpts {
  taskId: string
  framework?: string
  agentVersion: AgentVersion
  provider: AuditProvider
  onResult?: (analysis: AnyAnalysis) => void
}

const EMPTY_STATE: AuditJobState = {
  taskId: "",
  generating: false,
  genError: null,
  progress: [],
  elapsed: 0,
  startedAt: null,
  result: null,
}

const states = new Map<string, AuditJobState>()
const snapshots = new Map<string, AuditJobState>()
const listeners = new Set<() => void>()
let timer: ReturnType<typeof setInterval> | null = null

function emit(): void {
  for (const l of listeners) l()
}

function update(taskId: string, patch: Partial<AuditJobState>): void {
  const prev = states.get(taskId)
  const next: AuditJobState = { ...(prev ?? { ...EMPTY_STATE, taskId }), ...patch, taskId }
  states.set(taskId, next)
  snapshots.set(taskId, next)
  emit()
}

export function subscribeAuditJob(cb: () => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function getAuditJobSnapshot(taskId: string): AuditJobState {
  return snapshots.get(taskId) ?? EMPTY_STATE
}

export function getAuditJobServerSnapshot(): AuditJobState {
  return EMPTY_STATE
}

function ensureTimer(): void {
  if (timer) return
  timer = setInterval(() => {
    const now = Date.now()
    let changed = false
    for (const [taskId, st] of states) {
      if (st.generating && st.startedAt != null) {
        const elapsed = Math.floor((now - st.startedAt) / 1000)
        if (elapsed !== st.elapsed) {
          update(taskId, { elapsed })
          changed = true
        }
      }
    }
    if (changed) emit()
  }, 1000)
}

function maybeStopTimer(): void {
  if (!timer) return
  let any = false
  for (const st of states.values()) {
    if (st.generating) {
      any = true
      break
    }
  }
  if (!any) {
    clearInterval(timer)
    timer = null
  }
}

export function startAuditJob(opts: StartAuditJobOpts): void {
  const { taskId, provider } = opts
  if (states.get(taskId)?.generating) return
  if (!provider?.apiKey || !provider?.baseUrl || !provider?.model) {
    update(taskId, {
      generating: false,
      genError: "请先配置并保存 LLM API（Base URL / API Key / Model），或前往「/settings」页配置",
      progress: [],
      result: null,
      startedAt: null,
      elapsed: 0,
    })
    return
  }
  update(taskId, {
    generating: true,
    genError: null,
    progress: [],
    elapsed: 0,
    startedAt: Date.now(),
    result: null,
  })
  ensureTimer()
  void runAuditJob(opts).catch((e) => {
    update(taskId, { generating: false, genError: e instanceof Error ? e.message : "网络错误" })
    maybeStopTimer()
  })
}

async function runAuditJob(opts: StartAuditJobOpts): Promise<void> {
  const { taskId, framework, agentVersion, provider } = opts
  const startedAt = states.get(taskId)?.startedAt ?? Date.now()
  try {
    const url = agentVersion === "v1" ? "/api/ai/audit-session" : "/api/ai/audit-session-py"
    const body =
      agentVersion === "v3"
        ? { taskId, framework, provider, mode: "claude" }
        : agentVersion === "v4"
          ? { taskId, framework, provider, mode: "v4" }
          : { taskId, framework, provider }
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(1800000),
    })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: "生成失败" }))
      update(taskId, { generating: false, genError: err.error ?? "生成失败" })
      maybeStopTimer()
      return
    }
    const reader = res.body?.getReader()
    if (!reader) {
      update(taskId, { generating: false, genError: "无法读取流" })
      maybeStopTimer()
      return
    }
    const decoder = new TextDecoder()
    let buffer = ""
    let analysis: AnyAnalysis | null = null
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split("\n")
      buffer = lines.pop() ?? ""
      for (const line of lines) {
        if (!line.trim()) continue
        try {
          const evt = JSON.parse(line)
          handleEvent(taskId, evt, (a) => {
            analysis = a
          })
        } catch {
          /* 忽略半行 */
        }
      }
    }
    if (buffer.trim()) {
      try {
        const evt = JSON.parse(buffer)
        handleEvent(taskId, evt, (a) => {
          analysis = a
        })
      } catch {
        /* 忽略 */
      }
    }
    if (analysis) {
      const result = analysis as AnyAnalysis & { _auditMeta?: V4AuditMeta }
      result._auditMeta = {
        generatedAt: new Date().toISOString(),
        elapsedSec: Math.floor((Date.now() - startedAt) / 1000),
      }
      update(taskId, { generating: false, result })
      opts.onResult?.(result)
    } else {
      const cur = states.get(taskId)
      update(taskId, {
        generating: false,
        genError: cur?.genError ?? "返回数据缺 analysis 字段",
      })
    }
  } catch (e) {
    update(taskId, { generating: false, genError: e instanceof Error ? e.message : "网络错误" })
  } finally {
    maybeStopTimer()
  }
}

function handleEvent(
  taskId: string,
  evt: { stage?: string; msg?: string; round?: number; analysis?: AnyAnalysis },
  onResult: (a: AnyAnalysis) => void,
): void {
  if (evt.stage === "result" && evt.analysis) {
    onResult(evt.analysis)
    return
  }
  if (evt.stage === "error") {
    update(taskId, { genError: evt.msg ?? "生成失败" })
    return
  }
  const cur = states.get(taskId)
  if (!cur) return
  if (evt.stage === "ping") {
    const next = [...cur.progress]
    const last = next[next.length - 1]
    if (last && last.stage === "ping" && last.round === evt.round) {
      next[next.length - 1] = { stage: "ping", msg: evt.msg ?? "", round: evt.round }
    } else {
      next.push({ stage: "ping", msg: evt.msg ?? "", round: evt.round })
    }
    update(taskId, { progress: next })
    return
  }
  if (evt.msg) {
    update(taskId, {
      progress: [...cur.progress, { stage: evt.stage ?? "", msg: evt.msg, round: evt.round }],
    })
  }
}

export function clearAuditJob(taskId: string): void {
  states.delete(taskId)
  snapshots.delete(taskId)
  emit()
  maybeStopTimer()
}

export function __resetAuditJobsForTests(): void {
  states.clear()
  snapshots.clear()
  listeners.clear()
  if (timer) {
    clearInterval(timer)
    timer = null
  }
}
