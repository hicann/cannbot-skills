// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Server-only（spawn/child_process）：spawn sift audit，流式读 stdout，解析 on_progress
 * 输出算百分比，事件以 NDJSON 回传前端。管道模式（非 tty）下 rich 输出纯文本，可可靠解析。
 *
 * 进度结构（auditor.py 的 on_progress）：
 *   "对账 transcript k/N: name"                          — transcript 层进度
 *   "  [精筛] 批 k/total 判定 N 条…" / "…完成 ⏱ Xs"      — 精筛(refine)
 *   "  [headline] 批 k/total 判定 N 条…" / "…完成 ⏱ Xs"  — headline
 *   "  [条件性|禁令|步骤] 批 k/total 判定 N 条…"          — 3 类 LLM 判定并发
 *   "  [条件性|禁令|步骤] 批 k/total 完成 ⏱ Xs"
 * 5 阶段统一用 [label] 批 k/total 跟踪，% = sum(5 stages) / 5。
 * output 类(PROGRAMMATIC) 不调 LLM、瞬间完成，不需要跟踪。
 * 加最低 5% 初始值（进程启动→首条输出之间不再是 0%）。
 */
import { spawn, type ChildProcess } from "node:child_process"
import fs from "node:fs"
import path from "node:path"

export interface SiftEvent {
  stage: "progress" | "result" | "error"
  percent?: number
  msg?: string
  report?: unknown
  _html?: string
}

/** 5 个有 LLM 调用的阶段（output 类 PROGRAMMATIC 瞬间完成不跟踪）。 */
type StageKey = "精筛" | "headline" | "条件性" | "禁令" | "步骤"
const STAGES: StageKey[] = ["精筛", "headline", "条件性", "禁令", "步骤"]
const ANSI_RE = /\x1b\[[0-9;]*[a-zA-Z]/g

interface StageState {
  done: boolean
  k: number
  total: number
}

/**
 * 分阶段权重进度（贴合真实耗时分布）：
 * - 预处理（精筛 + headline）：0% → 15%（通常快，缓存命中秒完成）
 * - 判定（3 类 LLM）：15% → 95%（主要耗时，40-257s/批）
 * - 完成：100%
 * 多 transcript 按 k/N 缩放判定段。
 */
function stageProgress(s: StageState): number {
  if (s.done) return 1
  return s.total > 0 ? s.k / s.total : 0
}

function computePercent(
  stages: Record<StageKey, StageState>,
  started: boolean,
  transcriptK: number,
  transcriptTotal: number,
): number {
  if (!started) return 0

  // 预处理段：0% → 15%（精筛 + headline 各占一半）
  const refinePct = stageProgress(stages["精筛"]) * 0.5
  const headlinePct = stageProgress(stages["headline"]) * 0.5
  const preprocessPct = (refinePct + headlinePct) * 15

  // 判定段：15% → 95%（80% 总量，3 类均分）
  const judgeRaw =
    (stageProgress(stages["条件性"]) + stageProgress(stages["禁令"]) + stageProgress(stages["步骤"])) / 3
  // 多 transcript 缩放：transcript k/N 决定判定段在 15-95% 中的位置
  const transcriptFrac = transcriptTotal > 1 ? (transcriptK - 1) / transcriptTotal : 0
  const judgePct = (transcriptFrac + judgeRaw / transcriptTotal) * 80

  const pct = Math.round(preprocessPct + 15 + judgePct)
  return Math.max(pct, started ? 3 : 0)
}

function runSiftAuditStreaming(opts: {
  args: string[]
  outputDir: string
  onEvent: (e: SiftEvent) => void
  timeoutMs?: number
}): Promise<void> {
  const { args, outputDir, onEvent, timeoutMs = 1_800_000 } = opts
  return new Promise<void>((resolve) => {
    const stages: Record<StageKey, StageState> = {
      "精筛": { done: false, k: 0, total: 0 },
      "headline": { done: false, k: 0, total: 0 },
      "条件性": { done: false, k: 0, total: 0 },
      "禁令": { done: false, k: 0, total: 0 },
      "步骤": { done: false, k: 0, total: 0 },
    }
    let started = false
    let transcriptK = 1
    let transcriptTotal = 1

    function handleLine(raw: string): void {
      const line = raw.replace(ANSI_RE, "").replace(/\r/g, "").trim()
      if (!line) return
      started = true

      // transcript k/N（多 transcript 进度）
      const tm = line.match(/对账 transcript\s+(\d+)\/(\d+)/)
      if (tm) {
        transcriptK = parseInt(tm[1], 10)
        transcriptTotal = parseInt(tm[2], 10)
      }

      // [label] 批 k/total — 统一解析 5 个阶段
      const bm = line.match(/\[(精筛|headline|条件性|禁令|步骤)\][^\n]*?批\s*(\d+)\/(\d+)/)
      if (bm) {
        const stage = bm[1] as StageKey
        stages[stage].k = parseInt(bm[2], 10)
        stages[stage].total = parseInt(bm[3], 10)
      }
      // 完成检测（精确匹配 "完成 ⏱" 避免误匹配"未完成"）
      for (const s of STAGES) {
        if (line.includes(`[${s}]`) && line.includes("完成")) stages[s].done = true
      }
      // 精筛的旧式完成信号（缓存 hit 时不输出批次，只输出滤除/无候选）
      if (line.includes("精筛") && (line.includes("滤除") || line.includes("无候选") || line.includes("完成"))) {
        stages["精筛"].done = true
      }

      onEvent({ stage: "progress", percent: computePercent(stages, started, transcriptK, transcriptTotal), msg: line })
    }

    let proc: ChildProcess
    try {
      proc = spawn("sift", args, {
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
        onEvent({ stage: "error", msg: "sift 未找到：装好 sift 并确保 `sift` 在 PATH。" })
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
          msg: `sift audit 失败${code != null ? ` (exit ${code})` : ""}${tail ? `: ${tail}` : ""}`,
        })
        resolve()
        return
      }
      const reportPath = path.join(outputDir, "audit-report.json")
      if (!fs.existsSync(reportPath)) {
        onEvent({ stage: "error", msg: "sift 跑完但没产出 audit-report.json。" })
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
 * 把一次 sift audit 跑成 NDJSON 流式 Response（progress → result/error）。
 * 路由 prep 好 args + outputDir 后调用本函数返回给前端。
 * cleanup 在流结束（无论成功/失败）后调用——tmp 目录的清理必须等流消费完（sift 跑完），
 * 不能在路由 return 时就删（否则 sift 还没读到 SKILL.md/session.json）。
 */
export function makeStreamingAuditResponse(
  args: string[],
  outputDir: string,
  cleanup?: () => void,
): Response {
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const send = (obj: SiftEvent): void => {
        controller.enqueue(encoder.encode(JSON.stringify(obj) + "\n"))
      }
      try {
        await runSiftAuditStreaming({ args, outputDir, onEvent: send })
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
