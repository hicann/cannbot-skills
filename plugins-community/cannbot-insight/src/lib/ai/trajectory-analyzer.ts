// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from "node:fs"
import path from "node:path"
import type { PrismaClient } from "@prisma/client"
import type { AIProviderConfig } from "@/lib/ai/analyzer"
import { exportSessionToMarkdown } from "@/lib/export/markdown-exporter"
import {
  extractSkeleton,
  extractSkillContent,
  parseStats,
  extractGates,
  extractErrors,
  readSection,
  type ReadRequest,
} from "@/lib/ai/trajectory-parser"

const MAX_ROUNDS = 3

export class AnalysisError extends Error {}
export class SchemaError extends Error {}

export interface TrajectorySummary {
  skeleton: ReturnType<typeof extractSkeleton>
  skillContent: ReturnType<typeof extractSkillContent>
  stats: ReturnType<typeof parseStats>
  gates: ReturnType<typeof extractGates>
  errors: ReturnType<typeof extractErrors>
}

export function buildTrajectorySummary(text: string): TrajectorySummary {
  return {
    skeleton: extractSkeleton(text),
    skillContent: extractSkillContent(text),
    stats: parseStats(text),
    gates: extractGates(text),
    errors: extractErrors(text),
  }
}

const REQUIRED_TOP_KEYS = [
  "sessionSummary",
  "workflowMeta",
  "sessionMeta",
  "flow",
  "skillQuality",
  "workflowLevelIssues",
  "optimizationPriorities",
] as const

const REQUIRED_ARRAY_KEYS_IN_SESSION_META = ["cpsExecuted", "cpsMissing", "phasesNotReached"] as const

export function validateSchema(json: unknown): void {
  if (typeof json !== "object" || json === null) {
    throw new SchemaError("analysis output must be a JSON object")
  }
  const obj = json as Record<string, unknown>
  for (const key of REQUIRED_TOP_KEYS) {
    if (!(key in obj)) throw new SchemaError(`missing top-level key: ${key}`)
  }
  if (!Array.isArray(obj.flow)) throw new SchemaError("flow must be an array")
  if (!Array.isArray(obj.skillQuality)) throw new SchemaError("skillQuality must be an array")
  if (!Array.isArray(obj.workflowLevelIssues)) throw new SchemaError("workflowLevelIssues must be an array")
  if (!Array.isArray(obj.optimizationPriorities)) throw new SchemaError("optimizationPriorities must be an array")

  const sm = obj.sessionMeta
  if (typeof sm !== "object" || sm === null || Array.isArray(sm)) {
    throw new SchemaError("sessionMeta must be an object")
  }
  const smObj = sm as Record<string, unknown>
  for (const key of REQUIRED_ARRAY_KEYS_IN_SESSION_META) {
    const v = smObj[key]
    if (!Array.isArray(v)) throw new SchemaError(`sessionMeta.${key} must be an array`)
  }
}

function tryParseJson(content: string): unknown | null {
  try {
    return JSON.parse(content)
  } catch {
    return null
  }
}

function tryParseReadRequest(content: string): ReadRequest | null {
  const parsed = tryParseJson(content)
  if (parsed && typeof parsed === "object" && parsed !== null && "read" in parsed) {
    const read = (parsed as { read: unknown }).read
    if (read && typeof read === "object") {
      const r = read as Record<string, unknown>
      if (Array.isArray(r.lines) && r.lines.length === 2) {
        return { lines: [Number(r.lines[0]), Number(r.lines[1])] } as { lines: [number, number] }
      }
      if (typeof r.section === "string") {
        return { section: r.section }
      }
    }
  }
  return null
}

function buildInitialUserMessage(summary: TrajectorySummary): string {
  const gatesStr = summary.gates.length > 0
    ? JSON.stringify(summary.gates.slice(0, 50).map(g => ({ turn: g.turn, result: g.result, line: g.line })))
    : "[]"
  const errorsStr = summary.errors.length > 0
    ? JSON.stringify(summary.errors.slice(0, 20).map(e => ({ turn: e.turn, lineRange: e.lineRange, snippet: e.snippet.slice(0, 500) })))
    : "[]"

  return [
    "## 轨迹骨架",
    JSON.stringify({
      skeleton: summary.skeleton.skeleton,
      occurrences: summary.skeleton.occurrences,
      turnCount: summary.skeleton.turnCount,
    }),
    "",
    "## Workflow SKILL.md 原文",
    summary.skillContent.skillMd || "(未找到 skill_content)",
    "",
    "## Stats",
    JSON.stringify(summary.stats),
    "",
    "## 门控结果",
    gatesStr,
    "",
    "## 异常段落",
    errorsStr,
    "",
    "请按 prompt 指令输出分析 JSON。",
    '若需更多上下文，输出 {"read": {"lines": [from, to]}} 或 {"read": {"section": "§N.M"}}。',
    "最多 3 轮读请求。",
  ].join("\n")
}

export async function callLLM(
  provider: AIProviderConfig,
  system: string,
  messages: Array<{ role: string; content: string }>,
): Promise<string> {
  const apiBase = provider.baseUrl.replace(/\/+$/, "")
  const chatUrl =
    apiBase.endsWith("/v1") || apiBase.includes("/v1/")
      ? `${apiBase}/chat/completions`
      : `${apiBase}/v1/chat/completions`

  const body = JSON.stringify({
    model: provider.model,
    messages: [{ role: "system", content: system }, ...messages],
    temperature: 0.3,
    max_tokens: 16384,
    response_format: { type: "json_object" },
  })

  let lastErr: unknown
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      const response = await fetch(chatUrl, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${provider.apiKey}`,
          "Content-Type": "application/json",
        },
        body,
        signal: AbortSignal.timeout(900000),
      })

      if (!response.ok) {
        const errText = await response.text()
        throw new Error(`LLM API error: ${response.status} ${errText}`)
      }

      const data = (await response.json()) as {
        choices?: Array<{ message?: { content?: string } }>
      }
      const content = data.choices?.[0]?.message?.content
      if (!content) throw new Error("LLM API returned empty content")
      return content
    } catch (e) {
      lastErr = e
      if (attempt === 0) {
        await new Promise(r => setTimeout(r, 2000))
        continue
      }
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("LLM call failed")
}

export function writeAnalysisJson(outputDir: string, trajectoryBasename: string, json: unknown): string {
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true })
  }
  const outPath = path.join(outputDir, `${trajectoryBasename}-analysis.json`)
  fs.writeFileSync(outPath, JSON.stringify(json), "utf-8")
  return outPath
}

export interface AnalysisResult {
  outputPath: string
  rounds: number
  analysis: unknown
}

export interface ProgressEvent {
  stage: "skeleton" | "llm" | "read" | "schema-retry" | "validate" | "done" | "ping"
  round?: number
  rounds?: number
  msg: string
  detail?: unknown
}

export async function runAnalysisPipeline(opts: {
  trajectoryText: string
  promptMd: string
  provider: AIProviderConfig
  outputDir: string
  outputBasename: string
  onProgress?: (e: ProgressEvent) => void
}): Promise<AnalysisResult> {
  const { trajectoryText, promptMd, provider, outputDir, outputBasename, onProgress } = opts

  const summary = buildTrajectorySummary(trajectoryText)
  onProgress?.({
    stage: "skeleton",
    msg: `骨架：${summary.skeleton.skeleton.length} calls / ${summary.skeleton.turnCount} turns`,
    detail: { gates: summary.gates.length, errors: summary.errors.length },
  })

  const messages: Array<{ role: string; content: string }> = [
    { role: "user", content: buildInitialUserMessage(summary) },
  ]

  let rounds = 0
  let result: unknown = null

  for (let round = 0; round < MAX_ROUNDS; round++) {
    rounds = round + 1
    onProgress?.({ stage: "llm", round: rounds, msg: `LLM 第 ${rounds}/${MAX_ROUNDS} 轮…` })

    // 心跳：LLM 调用期间每 20s 发 ping，保持流活跃 + 反馈耗时
    let hbSecs = 0
    const heartbeat = onProgress
      ? setInterval(() => { hbSecs += 20; onProgress({ stage: "ping", round: rounds, msg: `LLM 调用中… ${hbSecs}s` }) }, 20000)
      : null
    let resp: string
    try {
      resp = await callLLM(provider, promptMd, messages)
    } finally {
      if (heartbeat) clearInterval(heartbeat)
    }

    const parsed = tryParseJson(resp)
    if (parsed !== null) {
      try {
        validateSchema(parsed)
        onProgress?.({ stage: "validate", msg: "schema 校验通过" })
        result = parsed
        break
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e)
        onProgress?.({ stage: "schema-retry", round: rounds, msg: `schema 校验失败：${msg}` })
        messages.push({ role: "assistant", content: resp })
        messages.push({ role: "user", content: `Schema 校验失败：${msg}。请修正后重新输出完整 JSON。` })
        continue
      }
    }

    const readReq = tryParseReadRequest(resp)
    if (readReq) {
      const target = "lines" in readReq ? `lines ${readReq.lines[0]}-${readReq.lines[1]}` : readReq.section
      onProgress?.({ stage: "read", round: rounds, msg: `LLM 请求读取 ${target}，补上下文` })
      const snippet = readSection(trajectoryText, readReq)
      messages.push({ role: "assistant", content: resp })
      messages.push({ role: "user", content: `已读取：\n${snippet}` })
      continue
    }

    messages.push({
      role: "user",
      content: '输出格式错误，请输出完整 JSON 或 {"read": {"lines": [from, to]}} 或 {"read": {"section": "§N.M"}}',
    })
  }

  if (result === null) {
    throw new AnalysisError(`LLM ${MAX_ROUNDS} 轮未产出有效 JSON`)
  }

  const outputPath = writeAnalysisJson(outputDir, outputBasename, result)
  onProgress?.({ stage: "done", rounds, msg: `完成（${rounds} 轮）` })

  return { outputPath, rounds, analysis: result }
}

export async function analyzeTrajectory(opts: {
  trajectoryPath: string
  promptPath?: string
  outputDir?: string
  provider: AIProviderConfig
}): Promise<AnalysisResult> {
  const {
    trajectoryPath,
    promptPath = path.resolve(process.cwd(), "prompts/session-trajectory-analyse.md"),
    outputDir = path.resolve(process.cwd(), "logs"),
    provider,
  } = opts

  if (!fs.existsSync(trajectoryPath)) {
    throw new AnalysisError(`trajectory file not found: ${trajectoryPath}`)
  }
  if (!fs.existsSync(promptPath)) {
    throw new AnalysisError(`prompt file not found: ${promptPath}`)
  }

  const trajectoryText = fs.readFileSync(trajectoryPath, "utf-8")
  const promptMd = fs.readFileSync(promptPath, "utf-8")
  const basename = path.basename(trajectoryPath, ".md")

  return runAnalysisPipeline({
    trajectoryText,
    promptMd,
    provider,
    outputDir,
    outputBasename: basename,
  })
}

export async function analyzeTrajectoryByTask(opts: {
  taskId: string
  framework?: string
  prisma: PrismaClient
  promptPath?: string
  outputDir?: string
  provider: AIProviderConfig
  onProgress?: (e: ProgressEvent) => void
}): Promise<AnalysisResult> {
  const {
    taskId,
    framework,
    prisma,
    promptPath = path.resolve(process.cwd(), "prompts/session-trajectory-analyse.md"),
    outputDir = path.resolve(process.cwd(), "logs"),
    provider,
    onProgress,
  } = opts

  if (!fs.existsSync(promptPath)) {
    throw new AnalysisError(`prompt file not found: ${promptPath}`)
  }

  onProgress?.({ stage: "skeleton", msg: "组装轨迹 MD…" })
  const trajectoryText = await exportSessionToMarkdown(taskId, prisma, framework)
  const promptMd = fs.readFileSync(promptPath, "utf-8")

  return runAnalysisPipeline({
    trajectoryText,
    promptMd,
    provider,
    outputDir,
    outputBasename: `session-${taskId}`,
    onProgress,
  })
}
