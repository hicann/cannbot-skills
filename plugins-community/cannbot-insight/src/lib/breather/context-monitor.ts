// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { OpenCodeClient, V1Message } from "@/lib/breather/opencode-api"
import { getContextWindowLimit } from "@/lib/context-window-config"

function calcContextInput(tokens: { total?: number; input: number; output: number; reasoning: number; cache: { read: number; write: number } } | null): number {
  if (!tokens) return 0
  return tokens.input + (tokens.cache?.read ?? 0)
}

function hasSubagentCall(msg: V1Message): boolean {
  return msg.parts.some((p: any) => p.type === "tool" && p.tool === "task")
}

interface ContextStatus {
  sessionId: string
  sessionTitle: string | null
  currentContextPct: number
  currentInputTokens: number
  contextWindowLimit: number
  model: string | null
}

export interface HistoryPoint {
  turnIndex: number
  contextPct: number
  timestamp: string
  hasSubagentCall: boolean
}

export async function getLatestSessionContext(
  client: OpenCodeClient,
  sessionId: string
): Promise<ContextStatus | null> {
  const messages = await client.getMessages(sessionId)
  if (messages.length === 0) return null

  const assistantMsgs = messages.filter((m: V1Message) => m.info.role === "assistant")
  if (assistantMsgs.length === 0) return null

  const lastAssistantMsg = assistantMsgs[assistantMsgs.length - 1]

  let lastValidMsg: V1Message | null = null
  for (let i = assistantMsgs.length - 1; i >= 0; i--) {
    const inputTokens = calcContextInput(assistantMsgs[i].info.tokens)
    if (inputTokens > 0) {
      lastValidMsg = assistantMsgs[i]
      break
    }
  }
  if (!lastValidMsg) return {
    sessionId,
    sessionTitle: null,
    currentContextPct: 0,
    currentInputTokens: 0,
    contextWindowLimit: 0,
    model: lastAssistantMsg.info.modelID ?? null,
  }

  const modelId = lastValidMsg.info.modelID ?? null
  const inputTokens = calcContextInput(lastValidMsg.info.tokens)
  const apiContextLimit = modelId ? await client.getContextLimit(modelId) : null
  const contextWindowLimit = apiContextLimit ?? getContextWindowLimit(modelId)

  let contextPct = 0
  if (inputTokens > 0 && contextWindowLimit > 0) {
    contextPct = (inputTokens / contextWindowLimit) * 100
  }

  return {
    sessionId,
    sessionTitle: null,
    currentContextPct: Math.round(contextPct * 10) / 10,
    currentInputTokens: inputTokens,
    contextWindowLimit,
    model: modelId,
  }
}

export async function getSessionContextHistory(
  client: OpenCodeClient,
  sessionId: string
): Promise<HistoryPoint[]> {
  const messages = await client.getMessages(sessionId)
  const assistantMsgs = messages.filter((m: V1Message) => m.info.role === "assistant" && m.info.tokens)

  const firstValidMsg = assistantMsgs.find(m => calcContextInput(m.info.tokens) > 0)
  const firstModelId = firstValidMsg?.info.modelID ?? null
  const apiContextLimit = firstModelId ? await client.getContextLimit(firstModelId) : null
  const contextWindowLimit = apiContextLimit ?? getContextWindowLimit(firstModelId)

  const history: HistoryPoint[] = []
  let turnIdx = 0

  for (const msg of assistantMsgs) {
    const inputTokens = calcContextInput(msg.info.tokens)
    if (inputTokens <= 0) continue

    let contextPct = 0
    if (inputTokens > 0 && contextWindowLimit > 0) {
      contextPct = (inputTokens / contextWindowLimit) * 100
    }

    history.push({
      turnIndex: turnIdx++,
      contextPct: Math.round(contextPct * 10) / 10,
      timestamp: msg.info.time?.created
        ? new Date(msg.info.time.created).toISOString()
        : new Date().toISOString(),
      hasSubagentCall: hasSubagentCall(msg),
    })
  }

  return history
}
