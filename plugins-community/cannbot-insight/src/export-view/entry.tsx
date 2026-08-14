// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// 嵌入式导出 bundle 的入口：挂载 SessionDetailPage，fetch 拦截器返回内嵌数据
import React from "react"
import { createRoot } from "react-dom/client"
import SessionDetailPage from "../app/session/[taskId]/page"

// file:// 下 EventSource/SSE 会报安全错误；导出快照无需实时刷新，no-op 掉
class EventSourceShim {
  onmessage: ((ev: { data: string }) => void) | null = null
  onerror: ((ev: unknown) => void) | null = null
  onopen: ((ev: unknown) => void) | null = null
  readyState = 0
  constructor() {}
  close() {}
  addEventListener() {}
  removeEventListener() {}
}
;(window as unknown as { EventSource: typeof EventSourceShim }).EventSource = EventSourceShim

declare global {
  interface Window {
    __EXPORT_TASK_ID?: string
    __EXPORT_FRAMEWORK?: string
    __EXPORT_DATA?: Record<string, unknown>
    __EXPORT_PARAM?: Record<string, unknown>
    __EXPORT_SKILL_AUDITS?: Record<string, string>
  }
}

// 迁移 sessionStorage 里的 skill/agent 对账结果（live insight 导出时捕获），
// 让 skill-audit-job 的 hydrateSkillAuditFromStorage 在挂载时读到，直接展示无需离线重跑
try {
  if (window.__EXPORT_SKILL_AUDITS) {
    for (const [k, v] of Object.entries(window.__EXPORT_SKILL_AUDITS)) {
      try { sessionStorage.setItem(k, v) } catch { /* quota */ }
    }
  }
} catch { /* 隐私模式 sessionStorage 不可用 */ }

// fetch 拦截：先按全 URL（含规范化 query）查 PARAM（skill-content/agent-content 等
// per-参数端点，同类不同参需区分），未命中再按 path 查 DATA（核心端点，一路径一份数据）
const DATA: Record<string, unknown> = window.__EXPORT_DATA || {}
const PARAM: Record<string, unknown> = window.__EXPORT_PARAM || {}
const origFetch = window.fetch.bind(window)
function pathKey(url: string): string {
  return url.replace(/^https?:\/\/[^/]+/, "").split("?")[0].split("#")[0]
}
function urlKey(url: string): string {
  try {
    const u = new URL(url, "http://e")
    const params = [...u.searchParams.entries()].sort().map(([k, v]) => `${k}=${v}`).join("&")
    return params ? `${u.pathname}?${params}` : u.pathname
  } catch {
    return pathKey(url)
  }
}
window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url
  // ① per-参数端点：全 URL 精确匹配
  const full = urlKey(url)
  if (full in PARAM) {
    const body = JSON.stringify(PARAM[full])
    return Promise.resolve(new Response(body, {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
  }
  // ② 核心端点：按 path 匹配（容忍 query 差异）
  const key = pathKey(url)
  if (key in DATA) {
    const body = JSON.stringify(DATA[key])
    return Promise.resolve(new Response(body, {
      status: 200,
      headers: { "content-type": "application/json" },
    }))
  }
  // SSE / auto-refresh / 未知 /api 端点：返回空响应，避免 fetch 悬挂
  if (key.includes("/auto-refresh") || key.includes("/events") || key.startsWith("/api/")) {
    return Promise.resolve(new Response("", { status: 200, headers: { "content-type": "text/plain" } }))
  }
  return origFetch(input, init)
}) as typeof window.fetch

// React 19 的 use(promise) 要求 thenable 带 status 字段；native Promise 没有，
// 首次渲染会挂起但根无 Suspense 边界 → 崩溃。构造 fulfilled thenable，use() 同步返回，不挂起。
function makeFulfilledThenable<T>(value: T): Promise<T> {
  const thenable = {
    status: "fulfilled" as const,
    value,
    then(onFulfilled?: (v: T) => unknown) {
      if (typeof onFulfilled === "function") {
        Promise.resolve(value).then(onFulfilled)
      }
      return this
    },
  }
  return thenable as unknown as Promise<T>
}

function ExportRoot() {
  const params = React.useMemo(
    () => makeFulfilledThenable({ taskId: window.__EXPORT_TASK_ID ?? "" }),
    [],
  )
  return (
    <React.Suspense fallback={<div className="p-4 text-muted-foreground">Loading…</div>}>
      <SessionDetailPage params={params} />
    </React.Suspense>
  )
}

const el = document.getElementById("root")
if (el) {
  createRoot(el).render(<ExportRoot />)
}
