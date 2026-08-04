// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Server-only（spawn/child_process）：spawn skill-eval audit，流式读 stdout，解析 on_progress
 * 输出算百分比，事件以 NDJSON 回传前端。管道模式（非 tty）下 rich 输出纯文本，可可靠解析。
 *
 * 进度结构（auditor.py 的 on_progress）：
 *   "对账 transcript k/N: name"
 *   "  [条件性|禁令|步骤] 批 k/total 判定 N 条…"   （多批时带 批 k/total）
 *   "  [条件性|禁令|步骤] 批 k/total 完成 ⏱ Xs"     （完成）
 *   "  [精筛] …" / refine 完成信号
 * 三个 LLM 类并发跑，% = (refineDone + 3 类各自进度) / 4。粗但反映真实进展。
 */
import { spawn, type ChildProcess } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

export interface SkillEvalEvent {
  stage: "progress" | "result" | "error"
  percent?: number
  msg?: string
  report?: unknown
  _html?: string
}

type CatKey = "条件性" | "禁令" | "步骤"
const CATS: CatKey[] = ["条件性", "禁令", "步骤"]
const ANSI_RE = /\x1b\[[0-9;]*[a-zA-Z]/g

interface CatState {
  done: boolean
  k: number
  total: number
}

function computePercent(refineDone: boolean, cats: Record<CatKey, CatState>): number {
  let sum = refineDone ? 1 : 0
  for (const c of CATS) {
    const s = cats[c]
    sum += s.done ? 1 : s.total > 0 ? s.k / s.total : 0
  }
  return Math.round((sum / 4) * 100)
}

function runSkillEvalAuditStreaming(opts: {
  args: string[]
  outputDir: string
  onEvent: (e: SkillEvalEvent) => void
  timeoutMs?: number
}): Promise<void> {
  const { args, outputDir, onEvent, timeoutMs = 1_800_000 } = opts
  return new Promise<void>((resolve) => {
    const cats: Record<CatKey, CatState> = {
      条件性: { done: false, k: 0, total: 0 },
      禁令: { done: false, k: 0, total: 0 },
      步骤: { done: false, k: 0, total: 0 },
    }
    let refineDone = false

    function handleLine(raw: string): void {
      const line = raw.replace(ANSI_RE, "").replace(/\r/g, "").trim()
      if (!line) return
      if (line.includes("精筛") && (line.includes("完成") || line.includes("滤除") || line.includes("无候选"))) {
        refineDone = true
      }
      const bm = line.match(/\[(条件性|禁令|步骤)\][^\n]*?批\s*(\d+)\/(\d+)/)
      if (bm) {
        const cat = bm[1] as CatKey
        cats[cat].k = parseInt(bm[2], 10)
        cats[cat].total = parseInt(bm[3], 10)
      }
      for (const c of CATS) {
        if (line.includes(`[${c}]`) && line.includes("完成")) cats[c].done = true
      }
      onEvent({ stage: "progress", percent: computePercent(refineDone, cats), msg: line })
    }

    let proc: ChildProcess
    try {
      proc = spawn("skill-eval", args, {
        env: { ...process.env, PYTHONUNBUFFERED: "1" },
        stdio: ["ignore", "pipe", "pipe"],
      })
    } catch (e: unknown) {
      onEvent({ stage: "error", msg: e instanceof Error ? e.message : String(e) })
      resolve()
      return
    }

    let stdoutBuf = ""
    let stderrBuf = ""
    proc.stdout?.on("data", (chunk: Buffer) => {
      stdoutBuf += chunk.toString("utf8")
      const lines = stdoutBuf.split(/\n/)
      stdoutBuf = lines.pop() ?? ""
      for (const l of lines) handleLine(l)
    })
    proc.stderr?.on("data", (chunk: Buffer) => {
      stderrBuf += chunk.toString("utf8")
    })

    const timer = setTimeout(() => {
      try {
        proc.kill("SIGKILL")
      } catch {
        /* ignore */
      }
    }, timeoutMs)

    proc.on("error", (e: Error & { code?: string }) => {
      clearTimeout(timer)
      if (e.code === "ENOENT") {
        onEvent({ stage: "error", msg: "skill-eval 未找到：装好 skill-eval 并确保 `skill-eval` 在 PATH。" })
      } else {
        onEvent({ stage: "error", msg: e.message })
      }
      resolve()
    })

    proc.on("close", (code: number | null) => {
      clearTimeout(timer)
      if (stdoutBuf) handleLine(stdoutBuf)
      if (code !== 0) {
        const tail = stderrBuf.trim().slice(-500)
        onEvent({
          stage: "error",
          msg: `skill-eval audit 失败${code != null ? ` (exit ${code})` : ""}${tail ? `: ${tail}` : ""}`,
        })
        resolve()
        return
      }
      const reportPath = path.join(outputDir, "audit-report.json")
      if (!fs.existsSync(reportPath)) {
        onEvent({ stage: "error", msg: "skill-eval 跑完但没产出 audit-report.json。" })
        resolve()
        return
      }
      let report: unknown
      try {
        report = JSON.parse(fs.readFileSync(reportPath, "utf8"))
      } catch (e: unknown) {
        onEvent({ stage: "error", msg: `audit-report.json 解析失败：${e instanceof Error ? e.message : String(e)}` })
        resolve()
        return
      }
      const htmlPath = path.join(outputDir, "audit-report.html")
      const _html = fs.existsSync(htmlPath) ? fs.readFileSync(htmlPath, "utf8") : undefined
      onEvent({ stage: "result", percent: 100, report, _html })
      resolve()
    })
  })
}

/**
 * 把一次 skill-eval audit 跑成 NDJSON 流式 Response（progress → result/error）。
 * 路由 prep 好 args + outputDir 后调用本函数返回给前端。
 * cleanup 在流结束（无论成功/失败）后调用——tmp 目录的清理必须等流消费完（skill-eval 跑完），
 * 不能在路由 return 时就删（否则 skill-eval 还没读到 SKILL.md/session.json）。
 */
export function makeStreamingAuditResponse(
  args: string[],
  outputDir: string,
  cleanup?: () => void,
): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: SkillEvalEvent): void => {
        controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"))
      }
      try {
        await runSkillEvalAuditStreaming({ args, outputDir, onEvent: send })
      } catch (e: unknown) {
        send({ stage: "error", msg: e instanceof Error ? e.message : String(e) })
      } finally {
        try {
          cleanup?.()
        } catch {
          /* cleanup 失败忽略 */
        }
        controller.close()
      }
    },
  })
  return new Response(stream, {
    headers: {
      "Content-Type": "application/x-ndjson; charset=utf-8",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
    },
  })
}
