// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { TurnRow, ToolCallRow, SkillEventRow } from './turn-split';

export interface ExecutionRow {
  id: string;
  sessionId: string;
  agentName: string | null;
  agentSessionId: string | null;
  isSubagent: boolean;
  subagentType: string | null;
  subagentName: string | null;
  parentExecutionId: string | null;
  rootExecutionId: string | null;
  depth: number;
  tokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadInputTokens: number;
  cacheCreationInputTokens: number;
  maxSingleCallTokens: number;
  cost: number;
  latencyMs: number;
  createdAt: string;
  toolCallCount: number;
  toolCallErrorCount: number;
  llmCallCount: number;
  skillLoadCount: number;
  skillInvokeCount: number;
  finalResult: string | null;
  model: string | null;
}

let idCounter = 0;

function generateId(): string {
  idCounter++;
  const ts = Date.now().toString(36);
  const rand = Math.random().toString(36).substring(2, 8);
  const cnt = idCounter.toString(36);
  return `c${ts}${rand}${cnt}`;
}

export function resetIdCounter(): void {
  idCounter = 0;
}

export function splitExecutions(
  turns: TurnRow[],
  toolCalls: ToolCallRow[],
  skillEvents: SkillEventRow[],
  sessionId: string,
): ExecutionRow[] {
  const executions: ExecutionRow[] = [];

  const rootTurns = turns.filter(t => !t.isSubagent);
  const rootId = generateId();

  const rootAgg = aggregateTurns(rootTurns, toolCalls, skillEvents);
  const rootModel = rootTurns.find(t => t.model)?.model ?? null;
  const rootAgentName = rootTurns.find(t => t.agentName)?.agentName ?? null;

  const rootFinalResult = rootTurns.length > 0
    ? (rootTurns[rootTurns.length - 1]?.contentSummary ?? null)
    : null;

  executions.push({
    id: rootId,
    sessionId,
    agentName: rootAgentName,
    agentSessionId: sessionId,
    isSubagent: false,
    subagentType: null,
    subagentName: null,
    parentExecutionId: null,
    rootExecutionId: rootId,
    depth: 0,
    tokens: rootAgg.tokens,
    inputTokens: rootAgg.inputTokens,
    outputTokens: rootAgg.outputTokens,
    reasoningTokens: rootAgg.reasoningTokens,
    cacheReadInputTokens: rootAgg.cacheReadInputTokens,
    cacheCreationInputTokens: rootAgg.cacheCreationInputTokens,
    maxSingleCallTokens: rootAgg.maxSingleCallTokens,
    cost: rootAgg.cost,
    latencyMs: rootAgg.latencyMs,
    createdAt: rootAgg.createdAt,
    toolCallCount: rootAgg.toolCallCount,
    toolCallErrorCount: rootAgg.toolCallErrorCount,
    llmCallCount: rootAgg.llmCallCount,
    skillLoadCount: rootAgg.skillLoadCount,
    skillInvokeCount: rootAgg.skillInvokeCount,
    finalResult: rootFinalResult,
    model: rootModel,
  });

  const subagentSessionIds = new Set<string>();
  for (const turn of turns) {
    if (turn.subagentSessionId) {
      subagentSessionIds.add(turn.subagentSessionId);
    }
  }

  for (const subSessionId of subagentSessionIds) {
    const subTurns = turns.filter(t => t.subagentSessionId === subSessionId);
    if (subTurns.length === 0) continue;

    const subId = generateId();
    const subAgg = aggregateTurns(subTurns, toolCalls, skillEvents);
    const subModel = subTurns.find(t => t.model)?.model ?? null;
    const subAgentName = subTurns.find(t => t.agentName)?.agentName ?? null;
    const subSubagentName = subTurns.find(t => t.subagentName)?.subagentName ?? null;
    const subSubagentType = subTurns.find(t => t.subagentType)?.subagentType ?? null;

    const subFinalResult = subTurns[subTurns.length - 1]?.contentSummary ?? null;

    // Use explicit subagent_type from interactions if available, otherwise derive from name
    const subagentNameField = subSubagentName ?? subAgentName;
    let subagentType: string | null = subSubagentType ?? null;
    if (!subagentType) {
      if (subagentNameField) {
        const typeMatch = subagentNameField.match(/^@(\w+)/);
        subagentType = typeMatch ? typeMatch[1] : subagentNameField.toLowerCase();
      }
    }

    executions.push({
      id: subId,
      sessionId,
      agentName: subAgentName,
      agentSessionId: subSessionId,
      isSubagent: true,
      subagentType,
      subagentName: subagentNameField,
      parentExecutionId: rootId,
      rootExecutionId: rootId,
      depth: 1,
      tokens: subAgg.tokens,
      inputTokens: subAgg.inputTokens,
      outputTokens: subAgg.outputTokens,
      reasoningTokens: subAgg.reasoningTokens,
      cacheReadInputTokens: subAgg.cacheReadInputTokens,
      cacheCreationInputTokens: subAgg.cacheCreationInputTokens,
      maxSingleCallTokens: subAgg.maxSingleCallTokens,
      cost: subAgg.cost,
      latencyMs: subAgg.latencyMs,
      createdAt: subAgg.createdAt,
      toolCallCount: subAgg.toolCallCount,
      toolCallErrorCount: subAgg.toolCallErrorCount,
      llmCallCount: subAgg.llmCallCount,
      skillLoadCount: subAgg.skillLoadCount,
      skillInvokeCount: subAgg.skillInvokeCount,
      finalResult: subFinalResult,
      model: subModel,
    });
  }

  return executions;
}

interface TurnAggregation {
  tokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadInputTokens: number;
  cacheCreationInputTokens: number;
  maxSingleCallTokens: number;
  cost: number;
  latencyMs: number;
  createdAt: string;
  toolCallCount: number;
  toolCallErrorCount: number;
  llmCallCount: number;
  skillLoadCount: number;
  skillInvokeCount: number;
}

function aggregateTurns(
  turns: TurnRow[],
  allToolCalls: ToolCallRow[],
  allSkillEvents: SkillEventRow[],
): TurnAggregation {
  const turnIds = new Set(turns.map(t => t.id));

  const tokens = turns.reduce((s, t) => s + t.totalTokens, 0);
  const inputTokens = turns.reduce((s, t) => s + t.inputTokens, 0);
  const outputTokens = turns.reduce((s, t) => s + t.outputTokens, 0);
  const reasoningTokens = turns.reduce((s, t) => s + t.reasoningTokens, 0);
  const cacheReadInputTokens = turns.reduce((s, t) => s + t.cacheReadTokens, 0);
  const cacheCreationInputTokens = turns.reduce((s, t) => s + t.cacheWriteTokens, 0);
  const cost = turns.reduce((s, t) => s + t.cost, 0);
  const maxSingleCallTokens = turns.length > 0
    ? Math.max(...turns.map(t => t.totalTokens))
    : 0;

  const timestamps = turns
    .filter(t => t.createdAt_ts)
    .map(t => new Date(t.createdAt_ts!).getTime());
  const endTimestamps = turns
    .filter(t => t.completedAt)
    .map(t => new Date(t.completedAt!).getTime());

  const earliestStart = timestamps.length > 0 ? Math.min(...timestamps) : Date.now();
  const latestEnd = endTimestamps.length > 0
    ? Math.max(...endTimestamps)
    : (timestamps.length > 0 ? Math.max(...timestamps) + turns[turns.length - 1].latencyMs : Date.now());

  const latencyMs = Math.max(0, latestEnd - earliestStart);
  const createdAt = new Date(earliestStart).toISOString();

  const llmCallCount = turns.filter(t => t.role === 'assistant').length;

  const relatedToolCalls = allToolCalls.filter(tc => turnIds.has(tc.turnId));
  const toolCallCount = relatedToolCalls.length;
  const toolCallErrorCount = relatedToolCalls.filter(tc => tc.state !== 'ok' && tc.state !== 'completed').length;

  const relatedSkillEvents = allSkillEvents.filter(se => turnIds.has(se.turnId));
  const skillLoadCount = relatedSkillEvents.filter(se => se.eventType === 'load').length;
  const skillInvokeCount = relatedSkillEvents.filter(se => se.eventType === 'invoke' || se.eventType === 'use').length;

  return {
    tokens,
    inputTokens,
    outputTokens,
    reasoningTokens,
    cacheReadInputTokens,
    cacheCreationInputTokens,
    maxSingleCallTokens,
    cost,
    latencyMs,
    createdAt,
    toolCallCount,
    toolCallErrorCount,
    llmCallCount,
    skillLoadCount,
    skillInvokeCount,
  };
}
