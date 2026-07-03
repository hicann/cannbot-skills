// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { RawInteraction } from '../shared/types';
import type { ToolCallRow, TurnRow } from './turn-split';

export interface InteractionBridgeRow {
  id: string;
  sessionId: string;
  dispatchExecutionId: string;
  dispatchTurnId: string | null;
  dispatchToolCallId: string | null;
  dispatchContent: string | null;
  dispatchTimestamp: string | null;
  responseExecutionId: string | null;
  responseTurnId: string | null;
  responseContent: string | null;
  responseTimestamp: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  subagentTokens: number;
  subagentLatencyMs: number;
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

const TIMEOUT_THRESHOLD_MS = 30 * 60 * 1000;

interface TaskArgs {
  subagent_session_id?: string;
  subagent_name?: string;
  subagent_type?: string;
  summary?: string;
  description?: string;
  prompt?: string;
}

function parseTaskArgs(argsJson: string | null, toolName?: string): TaskArgs {
  if (!argsJson) return {};
  try {
    const args = JSON.parse(argsJson);
    if (toolName === 'Agent') {
      return {
        subagent_session_id: args.subagent_session_id ?? null,
        subagent_name: args.name ?? args.subagent_name ?? null,
        subagent_type: args.subagent_type ?? args.mode ?? null,
        summary: args.description ?? null,
        description: args.description ?? null,
        prompt: args.prompt ?? null,
      };
    }
    return args;
  } catch {
    return {};
  }
}

function extractSubagentSessionIdFromResult(resultJson: string | null): string | null {
  if (!resultJson) return null;
  const match = resultJson.match(/task_id:\s*(\S+)/);
  return match ? match[1] : null;
}

function isSubagentError(turn: TurnRow): boolean {
  if (turn.finishReason === 'error' || turn.finishReason === 'failed') return true;
  return false;
}

export function buildBridges(
  interactions: RawInteraction[],
  toolCalls: ToolCallRow[],
  turns: TurnRow[],
  sessionId: string,
  rootExecutionId: string,
  toolUseIdToSubagentSessionId?: Map<string, string>,
): InteractionBridgeRow[] {
  const bridges: InteractionBridgeRow[] = [];

  const dispatchToolCalls = toolCalls.filter(tc => tc.toolName === 'task' || tc.toolName === 'Agent');

  const subagentInteractionMap = new Map<string, RawInteraction[]>();
  for (const interaction of interactions) {
    const sid = interaction.subagent_session_id;
    if (sid) {
      const list = subagentInteractionMap.get(sid) ?? [];
      list.push(interaction);
      subagentInteractionMap.set(sid, list);
    }
  }

  const subagentTurnMap = new Map<string, TurnRow[]>();
  for (const turn of turns) {
    const sid = turn.subagentSessionId;
    if (sid) {
      const list = subagentTurnMap.get(sid) ?? [];
      list.push(turn);
      subagentTurnMap.set(sid, list);
    }
  }

  const usedSessionIds = new Set<string>();

  for (const dispatchTc of dispatchToolCalls) {
    const args = parseTaskArgs(dispatchTc.argsJson, dispatchTc.toolName);
    let matchedSessionId: string | null = args.subagent_session_id ?? null;

    // For Agent tool calls, use the meta.json toolUseId mapping for precise matching
    if (!matchedSessionId && dispatchTc.toolName === 'Agent' && toolUseIdToSubagentSessionId) {
      matchedSessionId = toolUseIdToSubagentSessionId.get(dispatchTc.toolCallId) ?? null;
    }

    if (!matchedSessionId) {
      matchedSessionId = extractSubagentSessionIdFromResult(dispatchTc.resultJson);
    }

    if (!matchedSessionId) {
      const dispatchTime = dispatchTc.startedAt ? new Date(dispatchTc.startedAt).getTime() : 0;
      let closestSessionId: string | null = null;
      let closestDiff = Infinity;
      for (const [sid, subInteractions] of subagentInteractionMap) {
        if (usedSessionIds.has(sid)) continue;
        const firstTime = subInteractions[0]?.timeInfo?.created ?? 0;
        const diff = Math.abs(firstTime - dispatchTime);
        if (diff < closestDiff) {
          closestDiff = diff;
          closestSessionId = sid;
        }
      }
      if (closestSessionId && closestDiff < TIMEOUT_THRESHOLD_MS) {
        matchedSessionId = closestSessionId;
      }
    }

    const dispatchContent = args.summary ?? args.description ?? args.prompt ?? null;
    const dispatchTimestamp = dispatchTc.startedAt ?? null;

    const subagentName = args.subagent_name ?? null;
    const subagentType = args.subagent_type ?? null;

    if (matchedSessionId) {
      usedSessionIds.add(matchedSessionId);

      const subTurns = subagentTurnMap.get(matchedSessionId) ?? [];

      if (subTurns.length > 0) {
        const lastTurn = subTurns[subTurns.length - 1];
        const responseContent = lastTurn.content ?? null;
        const responseTimestamp = lastTurn.completedAt ?? null;
        const responseTurnId = lastTurn.id ?? null;

        const subagentTokens = subTurns.reduce((sum, t) => sum + t.totalTokens, 0);
        const firstSubTime = subTurns[0].createdAt_ts
          ? new Date(subTurns[0].createdAt_ts).getTime()
          : 0;
        const lastSubTime = responseTimestamp
          ? new Date(responseTimestamp).getTime()
          : firstSubTime;
        const dispatchTimeMs = dispatchTimestamp
          ? new Date(dispatchTimestamp).getTime()
          : firstSubTime;
        const subagentLatencyMs = lastSubTime > 0 && firstSubTime > 0
          ? Math.max(lastSubTime - Math.min(dispatchTimeMs, firstSubTime), 0)
          : 0;

        const hasError = subTurns.some(t => isSubagentError(t));

        let status: string;
        if (!hasError) {
          status = 'completed';
        } else {
          status = 'failed';
        }

        bridges.push({
          id: generateId(),
          sessionId,
          dispatchExecutionId: rootExecutionId,
          dispatchTurnId: dispatchTc.turnId,
          dispatchToolCallId: dispatchTc.id,
          dispatchContent,
          dispatchTimestamp,
          responseExecutionId: null,
          responseTurnId,
          responseContent,
          responseTimestamp,
          subagentSessionId: matchedSessionId,
          subagentType,
          subagentName: subagentName ?? (lastTurn.subagentName ?? null),
          status,
          subagentTokens,
          subagentLatencyMs,
        });
      } else {
        const dispatchTimeMs = dispatchTimestamp
          ? new Date(dispatchTimestamp).getTime()
          : 0;
        const nowMs = Date.now();
        const elapsed = nowMs - dispatchTimeMs;

        let status: string;
        if (elapsed > TIMEOUT_THRESHOLD_MS) {
          status = 'timeout';
        } else {
          status = 'dispatched';
        }

        bridges.push({
          id: generateId(),
          sessionId,
          dispatchExecutionId: rootExecutionId,
          dispatchTurnId: dispatchTc.turnId,
          dispatchToolCallId: dispatchTc.id,
          dispatchContent,
          dispatchTimestamp,
          responseExecutionId: null,
          responseTurnId: null,
          responseContent: null,
          responseTimestamp: null,
          subagentSessionId: matchedSessionId,
          subagentType,
          subagentName,
          status,
          subagentTokens: 0,
          subagentLatencyMs: 0,
        });
      }
    } else {
      const dispatchTimeMs = dispatchTimestamp
        ? new Date(dispatchTimestamp).getTime()
        : 0;
      const nowMs = Date.now();
      const elapsed = nowMs - dispatchTimeMs;

      let status: string;
      if (elapsed > TIMEOUT_THRESHOLD_MS) {
        status = 'timeout';
      } else {
        status = 'dispatched';
      }

      bridges.push({
        id: generateId(),
        sessionId,
        dispatchExecutionId: rootExecutionId,
        dispatchTurnId: dispatchTc.turnId,
        dispatchToolCallId: dispatchTc.id,
        dispatchContent,
        dispatchTimestamp,
        responseExecutionId: null,
        responseTurnId: null,
        responseContent: null,
        responseTimestamp: null,
        subagentSessionId: null,
        subagentType,
        subagentName,
        status,
        subagentTokens: 0,
        subagentLatencyMs: 0,
      });
    }
  }

  return bridges;
}
