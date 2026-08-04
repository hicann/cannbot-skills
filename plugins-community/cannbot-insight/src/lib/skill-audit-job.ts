// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Skill audit job store：把单个 target 的 running/result/error 提升到模块级，让组件 unmount
 *（切到别的 tab）后正在跑的 fetch 继续在后台、状态保留；切回来 useSyncExternalStore 读回
 * running/result。镜像 audit-job.ts 的模式（cross-tab resume），按 selKey 分键支持多 target 并发。
 *
 * result 同步写 sessionStorage（reload 后可恢复）；首次挂载时 hydrate 把 sessionStorage 灌进 store。
 */

import type { AuditReport, SkillAuditStoredResult } from "@/lib/skill-eval-audit-types"

export type AuditKind = "skill" | "agent" | "root"

/**
 * store 侧的 result：AuditReport + 可选 _html（"在新页打开原始 HTML"逃生口）。
 * 从 sessionStorage 恢复 / NDJSON result 事件物化都是此 shape。
 */
export type SkillAuditResult = SkillAuditStoredResult

export interface SkillAuditJobState {
  running: boolean
  error: string | null
  result: SkillAuditResult | null
  /** 最新进度消息（skill-eval on_progress 输出的一行）。 */
  progress: string | null
  /** 0-100，按 skill-eval 的批次/完成解析的粗略百分比。 */
  percent: number
}

const EMPTY_STATE: SkillAuditJobState = { running: false, error: null, result: null, progress: null, percent: 0 }

export function skillAuditKey(taskId: string, kind: AuditKind, name: string): string {
  return `${kind}:${taskId}:${name}`
}

const MAX_HTML_SIZE = 2_000_000
const STORAGE_KEY = (taskId: string, kind: AuditKind, name: string) => `skill-audit-${taskId}-${kind}-${name}`

const states = new Map<string, SkillAuditJobState>()
const snapshots = new Map<string, SkillAuditJobState>()
const listeners = new Set<() => void>()

function emit(): void {
  for (const l of listeners) l()
}

function update(key: string, patch: Partial<SkillAuditJobState>): void {
  const prev = states.get(key) ?? EMPTY_STATE
  const next: SkillAuditJobState = { ...prev, ...patch }
  states.set(key, next)
  snapshots.set(key, next)
  emit()
}

export function subscribeSkillAudit(cb: () => void): () => void {
  listeners.add(cb)
  return () => {
    listeners.delete(cb)
  }
}

export function getSkillAuditSnapshot(key: string): SkillAuditJobState {
  return snapshots.get(key) ?? EMPTY_STATE
}

export function getSkillAuditServerSnapshot(): SkillAuditJobState {
  return EMPTY_STATE
}

function persistToStorage(taskId: string, kind: AuditKind, name: string, r: SkillAuditResult): void {
  let toStore = r
  if (r._html && r._html.length > MAX_HTML_SIZE) toStore = { ...r, _html: undefined }
  try {
    sessionStorage.setItem(STORAGE_KEY(taskId, kind, name), JSON.stringify(toStore))
  } catch {
    /* quota */
  }
}

export function loadSkillAuditFromStorage(taskId: string, kind: AuditKind, name: string): SkillAuditResult | null {
  try {
    const raw = sessionStorage.getItem(STORAGE_KEY(taskId, kind, name))
    return raw ? (JSON.parse(raw) as SkillAuditResult) : null
  } catch {
    return null
  }
}

/**
 * 首次挂载/reload 时把 sessionStorage 已有结果灌进 store（store 已有 live state 则跳过，免覆盖正在跑的）。
 * entries = 本 session 所有可对账 target，逐个尝试 hydrate。
 */
export function hydrateSkillAuditFromStorage(
  taskId: string,
  entries: Array<{ name: string; kind: AuditKind }>,
): void {
  for (const e of entries) {
    const key = skillAuditKey(taskId, e.kind, e.name)
    if (states.has(key)) continue
    const loaded = loadSkillAuditFromStorage(taskId, e.kind, e.name)
    if (loaded) update(key, { result: loaded })
  }
}

export function startSkillAudit(opts: {
  taskId: string
  kind: AuditKind
  name: string
  framework?: string
}): void {
  const { taskId, kind, name, framework } = opts
  const key = skillAuditKey(taskId, kind, name)
  if (states.get(key)?.running) return // 去重：已在跑则不重发（重跑 finished 时 running=false 可重启）
  update(key, { running: true, error: null, result: null, progress: null, percent: 0 })
  void runSkillAudit(taskId, kind, name, framework).catch((e: unknown) => {
    update(key, { running: false, error: e instanceof Error ? e.message : String(e), progress: null })
  })
}

interface SkillEvalStreamEvent {
  stage: "progress" | "result" | "error"
  percent?: number
  msg?: string
  report?: AuditReport
  _html?: string
}

async function runSkillAudit(
  taskId: string,
  kind: AuditKind,
  name: string,
  framework?: string,
): Promise<void> {
  const key = skillAuditKey(taskId, kind, name)
  // skill / root 都走 audit-skilleval（root 的声明同 skill——从 session 恢复 SKILL.md，
  // 只是 --kind root 切主 agent 作用域）；agent 走 audit-agenteval。skill 路由按 body.kind 选 --kind。
  const useSkillRoute = kind === "skill" || kind === "root"
  try {
    const res = await fetch(useSkillRoute ? "/api/ai/audit-skilleval" : "/api/ai/audit-agenteval", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(
        useSkillRoute ? { taskId, skillName: name, kind, framework } : { taskId, agentName: name, framework },
      ),
    })
    if (!res.ok) {
      // 路由 prep 失败返回 JSON {error}；流式端点正常时 res.ok 且 body 是 NDJSON
      let msg = `HTTP ${res.status}`
      try {
        const j = (await res.json()) as { error?: string }
        if (j?.error) msg = j.error
      } catch {
        /* 非 JSON 错误体则用 HTTP 状态 */
      }
      throw new Error(msg)
    }
    const reader = res.body?.getReader()
    if (!reader) throw new Error("无法读取对账流")
    const decoder = new TextDecoder()
    let buf = ""
    const handle = (evt: SkillEvalStreamEvent): void => {
      if (evt.stage === "progress") {
        update(key, { running: true, progress: evt.msg ?? null, percent: evt.percent ?? 0 })
      } else if (evt.stage === "result") {
        // evt.report 是 AuditReport（runner 在 result 事件必带）；?? {} 仅为类型兜底，
        // 实际无 report 时 runner 走 error 分支。as 收声到 SkillAuditResult。
        const r = { ...(evt.report ?? {}), _html: evt._html } as SkillAuditResult
        persistToStorage(taskId, kind, name, r)
        update(key, { running: false, result: r, progress: null, percent: 100 })
      } else if (evt.stage === "error") {
        update(key, { running: false, error: evt.msg ?? "对账失败", progress: null })
      }
    }
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += decoder.decode(value, { stream: true })
      const lines = buf.split("\n")
      buf = lines.pop() ?? ""
      for (const line of lines) {
        if (!line.trim()) continue
        try {
          handle(JSON.parse(line) as SkillEvalStreamEvent)
        } catch {
          /* 半行 / 非 JSON 忽略 */
        }
      }
    }
    if (buf.trim()) {
      try {
        handle(JSON.parse(buf) as SkillEvalStreamEvent)
      } catch {
        /* 尾行残缺忽略 */
      }
    }
    // 流结束若仍是 running（未收 result/error 事件）→ 视为异常结束
    if (states.get(key)?.running) {
      update(key, { running: false, error: "对账流结束但未收到结果", progress: null })
    }
  } catch (e: unknown) {
    update(key, { running: false, error: e instanceof Error ? e.message : String(e), progress: null })
  }
}

export function __resetSkillAuditForTests(): void {
  states.clear()
  snapshots.clear()
  listeners.clear()
}
