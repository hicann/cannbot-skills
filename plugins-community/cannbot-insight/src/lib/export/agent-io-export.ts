// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
//
// Deterministic agent-IO extraction (Audit v4 Step 1).
// Walks the session's turns + InteractionBridge tree to build, per agent (main + every subagent,
// nested), its input / output / artifacts / envelope metrics. No LLM. Full fidelity (no truncation
// of dispatchContent/responseContent/tool args beyond a generous summary cap for the LLM).

import type { PrismaClient } from "@prisma/client"

export interface AgentEnvelope {
  latencySec: number
  tokensKt: number
  turnCount: number
  toolCallCount: number
  errorCount: number
  retryCount: number
  reasoningTokensKt: number
}

export interface AgentAction {
  turn: number
  tool: string
  arg: string            // 关键参数截断（file_path/path/command/pattern 等）
  state: "ok" | "error"
  durationMs?: number
  result?: string         // 结果/错误摘要截断（尤其错误或 verifier 输出）
}

export interface AgentNode {
  id: string                  // "main" or subagentSessionId
  parentId: string | null     // null for main, else the owning agent's id
  role: "main" | "subagent"
  name: string
  inputSummary: string
  outputSummary: string
  artifacts: string[]
  actions: AgentAction[]      // 该 agent 自己的工具调用时间线（按 turn 顺序）
  turns: number[]             // 该 agent 拥有的具体 turnIndex 列表（用于定位/跳转）
  envelope: AgentEnvelope
}

export interface AgentIO {
  sessionId: string
  taskQuery: string
  agents: AgentNode[]   // flat list (main first); tree rebuildable via parentId
}

const SUMMARY_CAP = 4000
const ARG_CAP = 160
const RESULT_CAP = 320
const MAX_ACTIONS_PER_AGENT = 80

function cap(s: string | null | undefined, n = SUMMARY_CAP): string {
  if (!s) return ""
  return s.length > n ? s.slice(0, n) + `\n…(截断 ${s.length - n} 字符)` : s
}

function extractFilePath(argsJson: string | null): string | null {
  if (!argsJson) return null
  try {
    const a = JSON.parse(argsJson) as Record<string, unknown>
    const c =
      (a.file_path as string) ??
      (a.path as string) ??
      (a.filePath as string) ??
      (a.command as string) // for Bash, keep command as artifact hint
    return typeof c === "string" ? c : null
  } catch {
    return null
  }
}

function extractKeyArg(argsJson: string | null, toolName: string): string {
  if (!argsJson) return ""
  let a: Record<string, unknown>
  try { a = JSON.parse(argsJson) as Record<string, unknown> } catch { return cap(argsJson, ARG_CAP) }
  const n = toolName.toLowerCase()
  const pick = (v: unknown): string => (typeof v === "string" ? v : "")
  if (n === "bash" || n === "bashoutput") {
    return cap(pick(a.command), ARG_CAP)
  }
  if (n === "grep" || n === "glob") {
    const p = pick(a.pattern) || pick(a.path) || pick(a.glob)
    return cap(p, ARG_CAP)
  }
  if (n === "edit" || n === "multiedit") {
    const fp = pick(a.file_path) || pick(a.path)
    const old = pick(a.old_string) || pick(a.oldString)
    const ne = pick(a.new_string) || pick(a.newString)
    return cap(`${fp}${old ? `  old=${old.slice(0, 40)}` : ""}${ne ? `  new=${ne.slice(0, 40)}` : ""}`, ARG_CAP)
  }
  const fp = pick(a.file_path) || pick(a.path) || pick(a.filePath) || pick(a.command) || pick(a.pattern) || pick(a.query)
  if (fp) return cap(fp, ARG_CAP)
  // fallback: compact json
  return cap(argsJson, ARG_CAP)
}

function resultSnippet(tc: { state: string; errorType: string | null; errorMessage: string | null; resultJson: string | null }): string | undefined {
  if (tc.state === "error" || tc.errorType || tc.errorMessage) {
    const e = tc.errorMessage || tc.errorType || ""
    if (e) return cap(e, RESULT_CAP)
    if (tc.resultJson) return cap(`(error) ${tc.resultJson}`, RESULT_CAP)
    return "(error)"
  }
  if (!tc.resultJson) return undefined
  return cap(tc.resultJson, RESULT_CAP)
}

function isWriteEditTool(name: string): boolean {
  const n = name.toLowerCase()
  return n === "write" || n === "edit" || n === "multiedit" || n === "create"
}

interface TurnRow {
  id: string
  turnIndex: number
  role: string
  content: string | null
  isSubagent: boolean
  subagentName: string | null
  subagentSessionId: string | null
  model: string | null
  latencyMs: number | null
  totalTokens: number | null
  reasoningTokens: number | null
  toolCalls: Array<{
    id: string
    toolName: string
    argsJson: string | null
    resultJson: string | null
    state: string
    errorType: string | null
    errorMessage: string | null
    durationMs: number | null
  }>
  skillEvents: Array<{ skillName: string; eventType: string; success: boolean; errorMessage: string | null }>
}

interface BridgeRow {
  id: string
  dispatchTurnId: string | null
  dispatchToolCallId: string | null
  dispatchContent: string | null
  responseTurnId: string | null
  responseContent: string | null
  subagentSessionId: string | null
  subagentType: string | null
  subagentName: string | null
  status: string
  subagentTokens: number | null
  subagentLatencyMs: number | null
}

function ownerAgentId(t: { isSubagent: boolean; subagentSessionId: string | null }): string {
  return t.isSubagent && t.subagentSessionId ? t.subagentSessionId : "main"
}

function emptyEnvelope(): AgentEnvelope {
  return { latencySec: 0, tokensKt: 0, turnCount: 0, toolCallCount: 0, errorCount: 0, retryCount: 0, reasoningTokensKt: 0 }
}

function accumulateEnvelope(env: AgentEnvelope, t: TurnRow, bridge: BridgeRow | null): void {
  env.turnCount++
  if (bridge) {
    env.latencySec += (bridge.subagentLatencyMs ?? 0) / 1000
    env.tokensKt += (bridge.subagentTokens ?? 0) / 1000
  } else {
    env.latencySec += (t.latencyMs ?? 0) / 1000
    env.tokensKt += (t.totalTokens ?? 0) / 1000
  }
  env.reasoningTokensKt += (t.reasoningTokens ?? 0) / 1000
  for (const tc of t.toolCalls) {
    env.toolCallCount++
    if (tc.state === "error" || tc.errorMessage || tc.errorType) env.errorCount++
  }
  for (const se of t.skillEvents) {
    if (!se.success) env.retryCount++
  }
}

export async function buildAgentIO(taskId: string, prisma: PrismaClient, framework?: string): Promise<AgentIO> {
  const where: Record<string, string> = { taskId }
  if (framework) where.framework = framework

  const session = await prisma.session.findFirst({ where })
  if (!session) throw new Error(`Session not found: "${taskId}"`)

  const bridges = (await prisma.interactionBridge.findMany({
    where: { sessionId: session.id },
    orderBy: [{ dispatchTimestamp: "asc" }],
  })) as BridgeRow[]

  const allTurns = (await prisma.turn.findMany({
    where: { sessionId: session.id },
    orderBy: [{ turnIndex: "asc" }],
    include: {
      toolCalls: {
        select: {
          id: true,
          toolName: true,
          argsJson: true,
          resultJson: true,
          state: true,
          errorType: true,
          errorMessage: true,
          durationMs: true,
        },
        orderBy: [{ id: "asc" }],
      },
      skillEvents: {
        select: { skillName: true, eventType: true, success: true, errorMessage: true },
      },
    },
  })) as unknown as TurnRow[]

  const turnById = new Map<string, TurnRow>()
  for (const t of allTurns) turnById.set(t.id, t)

  // The Task/Agent toolCall that dispatched each bridge (full prompt lives in its argsJson)
  const dispatchArgsByBridgeId = new Map<string, string | null>()
  for (const b of bridges) {
    if (!b.dispatchToolCallId) { dispatchArgsByBridgeId.set(b.id, null); continue }
    const parentTurn = b.dispatchTurnId ? turnById.get(b.dispatchTurnId) : null
    const tc = parentTurn?.toolCalls.find(x => x.id === b.dispatchToolCallId)
    dispatchArgsByBridgeId.set(b.id, tc?.argsJson ?? null)
  }

  // Group turns by owning agent
  const turnsByAgent = new Map<string, TurnRow[]>()
  for (const t of allTurns) {
    const owner = ownerAgentId(t)
    const arr = turnsByAgent.get(owner) ?? []
    arr.push(t); turnsByAgent.set(owner, arr)
  }

  // Map: child agentId -> its bridge (each subagent has exactly one inbound bridge)
  const bridgeByChild = new Map<string, BridgeRow>()
  for (const b of bridges) {
    if (b.subagentSessionId) bridgeByChild.set(b.subagentSessionId, b)
  }

  // Children bridges per parent agent
  const childBridgesByParent = new Map<string, BridgeRow[]>()
  for (const b of bridges) {
    const parentTurn = b.dispatchTurnId ? turnById.get(b.dispatchTurnId) : null
    const parentAgent = parentTurn ? ownerAgentId(parentTurn) : "main"
    const arr = childBridgesByParent.get(parentAgent) ?? []
    arr.push(b); childBridgesByParent.set(parentAgent, arr)
  }

  const firstUserTurn = allTurns.find(t => !t.isSubagent && t.role === "user")

  const flat: AgentNode[] = []

  function buildNode(agentId: string, parentId: string | null) {
    const isMain = agentId === "main"
    const turns = turnsByAgent.get(agentId) ?? []
    const bridge = isMain ? null : (bridgeByChild.get(agentId) ?? null)

    // input
    let inputSummary: string
    if (isMain) {
      inputSummary = cap(firstUserTurn?.content ?? "")
    } else {
      const dispatchArgs = bridge ? (dispatchArgsByBridgeId.get(bridge.id) ?? null) : null
      const parts: string[] = []
      if (bridge?.dispatchContent) parts.push(bridge.dispatchContent)
      if (dispatchArgs) parts.push(cap(dispatchArgs, SUMMARY_CAP))
      inputSummary = parts.join("\n\n")
    }

    // output
    let outputSummary: string
    if (isMain) {
      const lastAssistant = [...turns].reverse().find(t => t.role === "assistant")
      outputSummary = cap(lastAssistant?.content ?? "")
    } else {
      outputSummary = cap(bridge?.responseContent ?? "")
    }

    // artifacts: Write/Edit file paths (and Bash command hints) from this agent's own turns
    const artifacts: string[] = []
    for (const t of turns) {
      for (const tc of t.toolCalls) {
        if (isWriteEditTool(tc.toolName)) {
          const fp = extractFilePath(tc.argsJson)
          if (fp) artifacts.push(`${tc.toolName}: ${fp}`)
        }
      }
    }

    // actions: ordered tool-call timeline (this agent's own work) — truncated key arg + state + result snippet
    const actions: AgentAction[] = []
    for (const t of turns) {
      for (const tc of t.toolCalls) {
        if (actions.length >= MAX_ACTIONS_PER_AGENT) break
        actions.push({
          turn: t.turnIndex,
          tool: tc.toolName,
          arg: extractKeyArg(tc.argsJson, tc.toolName),
          state: (tc.state === "error" || tc.errorMessage || tc.errorType) ? "error" : "ok",
          durationMs: tc.durationMs ?? undefined,
          result: resultSnippet(tc),
        })
      }
      if (actions.length >= MAX_ACTIONS_PER_AGENT) break
    }

    // envelope
    const env = emptyEnvelope()
    for (const t of turns) accumulateEnvelope(env, t, isMain ? null : bridge)

    flat.push({
      id: agentId,
      parentId,
      role: isMain ? "main" : "subagent",
      name: isMain
        ? "main"
        : (bridge?.subagentName ?? bridge?.subagentType ?? agentId),
      inputSummary,
      outputSummary,
      artifacts,
      actions,
      turns: turns.map(t => t.turnIndex),
      envelope: env,
    })

    // recurse into children
    const childBridges = childBridgesByParent.get(agentId) ?? []
    for (const b of childBridges) {
      if (b.subagentSessionId) buildNode(b.subagentSessionId, agentId)
    }
  }

  buildNode("main", null)

  return {
    sessionId: session.id,
    taskQuery: cap(firstUserTurn?.content ?? ""),
    agents: flat,
  }
}
