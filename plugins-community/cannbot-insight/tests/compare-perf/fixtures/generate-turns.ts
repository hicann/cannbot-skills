// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Deterministic, seedable PRNG (mulberry32) so bench fixtures are reproducible.
// We avoid Node's crypto or Math.random for stable cross-run comparisons.
function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return function () {
    a = (a + 0x6D2B79F5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

import type { TurnData } from "@/lib/compare/turn-align"

const USER_PROMPTS_CN = [
  "帮我看一下登录认证模块为什么报错",
  "为什么 mybatis 的分页查询不生效",
  "我想给 cannbot-insight 加一个 export 功能",
  "整理一下最近一周的 PR 变更",
  "测试一下这个算子在 NPU 上的执行时间",
  "把 src/lib 目录下的文件梳理一下",
  "对比一下两次 session 的 turn 对齐效果",
  "重构成 hook 形式，并把状态提升到父组件",
  "修一下 Prisma N+1 查询的性能问题",
  "把中英混合 UI 的英文部分统一改成中文",
]

const USER_PROMPTS_EN = [
  "Please investigate the authentication module error",
  "Why does mybatis pagination not work as expected",
  "Add an export feature to cannbot-insight",
  "Summarize this week's pull requests",
  "Benchmark the operator execution time on NPU",
  "Walk me through the files under src/lib",
  "Compare turn alignment across two sessions",
  "Refactor into hooks and lift state up",
  "Fix the Prisma N+1 query performance issue",
  "Translate the English UI labels to Chinese",
]

const ASSISTANT_BODY_CN = [
  "我先读取相关文件，定位问题所在",
  "从堆栈看，问题出在密码校验逻辑",
  "我建议增加一个 context 字段，避免重复计算",
  "把这段逻辑抽到一个独立的 hook 里更清晰",
  "需要先确认数据库索引是否生效",
  "这里的 N+1 主要来自循环里的 await findMany",
  "可以用 banded DP 把复杂度降到线性级",
  "我用 vitest bench 跑了基线，老的是 2.3 秒",
  "合并 preparedTurn 的 cache 之后快了 40%",
  "改完跑一下 npm run test 看有没有 regression",
]

const ASSISTANT_BODY_EN = [
  "Let me read the relevant files first to locate the issue",
  "The stack trace points to password validation logic",
  "I suggest adding a context field to avoid recomputation",
  "Extracting this into a dedicated hook would be cleaner",
  "We should verify the database index is effective",
  "The N+1 pattern comes from the await findMany inside the loop",
  "We can use banded DP to reduce the complexity to linear",
  "I ran the vitest bench baseline, the old one is 2.3 seconds",
  "After merging the preparedTurn cache, it is 40% faster",
  "Run npm run test after the change to check for regressions",
]

const THINKING_FRAGMENTS = [
  "The user wants me to look at the authentication module carefully",
  "I should check the prisma schema for relations before querying",
  "Considering whether to introduce a worker for the heavy DP",
  "Mapping out the data flow from session list to compare page",
  "Need to keep the content field for diff but skip it for alignment",
  "Estimating the alignment cost: N*M is too large for 1000 turns",
  "Maybe a token-based shingle approach would be cheaper than cosine",
  "The user mentioned CJK characters, need to handle char granularity",
]

const TOOL_NAMES = [
  "Read", "Write", "Bash", "Glob", "Grep", "Edit", "Task", "Agent",
  "skill/load_skill", "skill/invoke",
]

const SKILL_NAMES = [
  "cannbot-skill-review", "gitcode-pr-handler", "customize-opencode",
  "gitcode-issue-handler", "gitcode-toolkit",
]

function pick<T>(rng: () => number, arr: T[]): T {
  return arr[Math.floor(rng() * arr.length)]
}

function randomInt(rng: () => number, min: number, max: number): number {
  return Math.floor(rng() * (max - min + 1)) + min
}

interface BuildOptions {
  seed: number
  count: number
  scenario: "similar" | "divergent" | "different-size" | "many-subagents"
  language: "cn" | "en" | "mixed"
}

function buildContent(rng: () => number, role: string, language: BuildOptions["language"], withThinking: boolean): string {
  const useCN = language === "cn" || (language === "mixed" && rng() < 0.5)
  const userArr = useCN ? USER_PROMPTS_CN : USER_PROMPTS_EN
  const bodyArr = useCN ? ASSISTANT_BODY_CN : ASSISTANT_BODY_EN

  let content = ""
  if (role === "user") {
    content = pick(rng, userArr)
  } else if (role === "assistant") {
    if (withThinking) {
      content = `<thinking>${pick(rng, THINKING_FRAGMENTS)}</thinking>\n${pick(rng, bodyArr)}`
    } else {
      content = pick(rng, bodyArr)
    }
  } else if (role === "system") {
    content = "You are a helpful coding agent. Follow the user's instructions carefully."
  } else {
    content = `<result>${pick(rng, bodyArr)}</result>`
  }

  if (rng() < 0.4) {
    const extraLines = randomInt(rng, 1, 4)
    for (let i = 0; i < extraLines; i++) {
      content += "\n" + pick(rng, bodyArr)
    }
  }

  return content
}

function buildTurn(rng: () => number, index: number, opts: BuildOptions, roleBias?: string): TurnData {
  const role = roleBias ?? (index === 0 ? "user" : pick(rng, ["user", "assistant", "assistant", "tool_result"]))
  const withThinking = role === "assistant" && rng() < 0.6
  const content = buildContent(rng, role, opts.language, withThinking)

  const toolCallCount = role === "assistant" ? randomInt(rng, 0, 5) : 0
  const toolCalls = Array.from({ length: toolCallCount }, (_, i) => ({
    toolCallId: `tc-${index}-${i}`,
    toolName: pick(rng, TOOL_NAMES),
    state: rng() < 0.9 ? "ok" : "error",
    durationMs: randomInt(rng, 10, 3000),
  }))

  const skillEventCount = role === "assistant" && rng() < 0.3 ? randomInt(rng, 0, 2) : 0
  const skillEvents = Array.from({ length: skillEventCount }, () => ({
    skillName: pick(rng, SKILL_NAMES),
    eventType: pick(rng, ["load", "invoke", "dispatch"]),
    success: rng() < 0.85,
  }))

  const isSubagent = opts.scenario === "many-subagents" && index % 18 === 0

  return {
    turnId: `t-${opts.seed}-${index}`,
    turnIndex: index,
    role,
    content,
    contentSummary: content.substring(0, 200),
    totalTokens: randomInt(rng, 100, 8000),
    inputTokens: randomInt(rng, 50, 6000),
    outputTokens: randomInt(rng, 20, 2000),
    reasoningTokens: withThinking ? randomInt(rng, 50, 1500) : 0,
    latencyMs: randomInt(rng, 0, 10000),
    model: pick(rng, ["claude-sonnet-4-5", "gpt-4o", "glm-5.2", "deepseek-v3"]),
    toolCalls,
    skillEvents,
    // Surface subagent info as a hidden channel for downstream transforms (unused by turn-align).
    ...isSubagent ? { _isSubagent: true } : {},
  } as TurnData
}

export function buildSession(opts: BuildOptions): TurnData[] {
  const rng = mulberry32(opts.seed)
  const turns: TurnData[] = []

  if (opts.scenario === "many-subagents") {
    for (let i = 0; i < opts.count; i++) {
      const role = i === 0 ? "user" : (i % 18 === 0 ? "assistant" : pick(rng, ["user", "assistant", "tool_result"]))
      turns.push(buildTurn(rng, i, opts, role))
    }
    return turns
  }

  for (let i = 0; i < opts.count; i++) {
    let role: string | undefined
    if (i === 0) role = "user"
    else if (i === 1) role = "assistant"
    else if (i === 2) role = "user"
    turns.push(buildTurn(rng, i, opts, role))
  }

  return turns
}

export interface ScenarioPair {
  label: string
  description: string
  turnsA: TurnData[]
  turnsB: TurnData[]
  expectedAlignment: "high" | "medium" | "low"
}

export function buildScenarioPair(
  scenario: "similar" | "divergent" | "different-size" | "many-subagents",
  countA: number,
  countB: number,
  seedA = 42,
  seedB = 43,
): ScenarioPair {
  const descriptionMap: Record<typeof scenario, string> = {
    similar: "Same query, similar model — turns should align with high similarity",
    divergent: "Same query, different execution paths — partial alignment with many gaps",
    "different-size": "Asymmetric session sizes — tests gap handling",
    "many-subagents": "Heavy subagent usage — exercises role/tool variety",
  }

  const langA = scenario === "similar" ? "cn" : scenario === "divergent" ? "en" : "mixed"
  const langB = scenario === "similar" ? "cn" : scenario === "divergent" ? "cn" : "mixed"

  const turnsA = buildSession({ seed: seedA, count: countA, scenario, language: langA })
  const turnsB = buildSession({ seed: scenario === "similar" ? seedA : seedB, count: countB, scenario, language: langB })

  const expectedAlignment: ScenarioPair["expectedAlignment"] =
    scenario === "similar" ? "high" : scenario === "divergent" ? "medium" : "low"

  return {
    label: `${scenario} ${countA}x${countB}`,
    description: descriptionMap[scenario],
    turnsA,
    turnsB,
    expectedAlignment,
  }
}

export function buildManualAlignments(turnsA: TurnData[], turnsB: TurnData[], count: number): Array<{ indexA: number; indexB: number }> {
  const out: Array<{ indexA: number; indexB: number }> = []
  const stepA = Math.max(1, Math.floor(turnsA.length / (count + 1)))
  const stepB = Math.max(1, Math.floor(turnsB.length / (count + 1)))
  for (let i = 1; i <= count; i++) {
    const ia = Math.min(turnsA.length - 1, i * stepA)
    const ib = Math.min(turnsB.length - 1, i * stepB)
    if (ia >= 0 && ib >= 0) out.push({ indexA: ia, indexB: ib })
  }
  return out
}

export { mulberry32 }
