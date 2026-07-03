// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { RawInteraction } from '../shared/types';
import { getContextWindowLimit } from '@/lib/context-window-config';
import { isContinuationTurn } from '@/lib/shared/command-parser';

export interface TurnRow {
  id: string;
  sessionId: string;
  turnIndex: number;
  role: string;
  content: string | null;
  contentJson: string | null;
  contentSummary: string | null;
  inputMessagesJson: string | null;
  inputMessagesCount: number;
  inputMessagesTokens: number;
  contextWindowPct: number | null;
  agentName: string | null;
  subagentName: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  cost: number;
  createdAt_ts: string | null;
  completedAt: string | null;
  latencyMs: number;
  ttftMs: number | null;
  model: string | null;
  modelId: string | null;
  providerId: string | null;
  temperature: number | null;
  maxTokens: number | null;
  finishReason: string | null;
  isSubagent: boolean;
  parentExecutionId: string | null;
}

export interface ToolCallRow {
  id: string;
  turnId: string;
  toolCallId: string;
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
  errorType: string | null;
  errorMessage: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number;
  dispatchBridgeId: string | null;
  isSkillRelated: boolean;
}

export interface SkillEventRow {
  id: string;
  turnId: string;
  skillName: string;
  skillVersion: number | null;
  eventType: string;
  success: boolean;
  errorMessage: string | null;
  argsJson: string | null;
  startedAt: string | null;
  completedAt: string | null;
  durationMs: number;
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

function truncateTo200(text: string | null): string | null {
  if (!text) return null;
  if (text.length <= 200) return text;
  return text.substring(0, 200);
}

function isSkillToolCall(toolName: string): boolean {
  const lower = toolName.toLowerCase();
  return lower.startsWith('skill/') || lower === 'skill' || lower === 'load_skill';
}

function getSkillEventType(toolName: string): string {
  const lower = toolName.toLowerCase();
  if (lower === 'skill/load_skill' || lower === 'load_skill') return 'load';
  if (lower === 'skill/invoke' || lower === 'skill') return 'invoke';
  return 'use';
}

function extractSkillName(toolName: string, argsJson: string | null): string {
  if (!argsJson) return toolName.replace(/^skill\//i, '');
  try {
    const args = JSON.parse(argsJson);
    if (args.skill) return args.skill;
    if (args.skill_name) return args.skill_name;
    if (args.name) return args.name;
  } catch { /* ignore */ }
  return toolName.replace(/^skill\//i, '');
}

function extractSkillVersion(argsJson: string | null): number | null {
  if (!argsJson) return null;
  try {
    const args = JSON.parse(argsJson);
    if (typeof args.version === 'number') return args.version;
  } catch { /* ignore */ }
  return null;
}

function isAgentSkillDispatch(toolName: string, argsJson: string | null): boolean {
  const lower = toolName.toLowerCase();
  if (lower !== 'agent' && lower !== 'task') return false;
  if (!argsJson) return false;
  try {
    const args = JSON.parse(argsJson);
    const subagentType = args.subagent_type ?? args.subagent_name ?? null;
    if (!subagentType) return false;
    // Exclude generic agent types (not skill-driven)
    if (subagentType === 'general-purpose' || subagentType === 'general') return false;
    return true;
  } catch { return false }
}

function extractDispatchSkillName(argsJson: string | null): string {
  if (!argsJson) return 'unknown-dispatch';
  try {
    const args = JSON.parse(argsJson);
    const subagentType = args.subagent_type ?? args.subagent_name ?? null;
    if (subagentType) return subagentType;
    if (args.description) return String(args.description).substring(0, 40);
  } catch { /* ignore */ }
  return 'unknown-dispatch';
}

export function extractErrorMessage(resultJson: string | null): string | null {
  if (!resultJson) return null;
  if (resultJson.includes('<tool_use_error>')) return truncateTo200(resultJson);
  if (resultJson.includes('Exit code')) return truncateTo200('Command failed: ' + resultJson);
  return null;
}

function classifyError(state: string, errorMessage: string | null): string | null {
  if (state === 'ok' || state === 'completed') {
    return errorMessage ? 'tool_error' : null;
  }
  if (!errorMessage) return 'unknown';
  const msg = errorMessage.toLowerCase();
  if (msg.includes('timeout') || msg.includes('timed out')) return 'timeout';
  if (msg.includes('permission') || msg.includes('denied') || msg.includes('forbidden')) return 'permission';
  if (msg.includes('format') || msg.includes('invalid') || msg.includes('parse')) return 'format';
  return 'server_error';
}

export function splitIntoTurns(
  interactions: RawInteraction[],
  sessionId: string,
  _parentSessionId?: string
): { turns: TurnRow[], toolCalls: ToolCallRow[], skillEvents: SkillEventRow[] } {
  void _parentSessionId;
  const turns: TurnRow[] = [];
  const toolCalls: ToolCallRow[] = [];
  const skillEvents: SkillEventRow[] = [];
  let prevInputMessagesTokens = 0;
  let prevContextKey = '';
  // Loop index of the most recent /compact continuation turn in the current
  // execution context. A continuation replaces the conversation history with a
  // summary, so the next assistant turn's prompt legitimately shrinks and its
  // input-message count must restart from the summary, not accumulate all
  // pre-compact turns.
  let prevCompactBoundaryIdx = -1;

  for (let i = 0; i < interactions.length; i++) {
    const interaction = interactions[i];
    const turnId = generateId();
    const turnIndex = i;

    const role = interaction.role === 'subagent' ? 'assistant' : interaction.role;

    const content = interaction.content;
    const contentSummary = truncateTo200(content);

    const usage = interaction.usage;
    const totalTokens = usage?.total ?? 0;
    const inputTokens = usage?.input ?? 0;
    const outputTokens = usage?.output ?? 0;
    const reasoningTokens = usage?.reasoning ?? 0;
    const cacheReadTokens = usage?.cacheRead ?? 0;
    const cacheWriteTokens = usage?.cacheWrite ?? 0;
    const cost = usage?.cost ?? 0;

    const timeInfo = interaction.timeInfo;
    const createdAt_ts = timeInfo
      ? new Date(timeInfo.created).toISOString()
      : new Date(interaction.timestamp).toISOString();
    const completedAt = timeInfo?.completed
      ? new Date(timeInfo.completed).toISOString()
      : null;
    const latencyMs = timeInfo?.completed && timeInfo?.created
      ? timeInfo.completed - timeInfo.created
      : (interaction.latency ?? 0);

    const agentName = interaction.agent ?? interaction.subagent_name ?? null;
    const subagentName = interaction.subagent_name ?? null;
    const subagentSessionId = interaction.subagent_session_id ?? null;
    const subagentType = interaction.subagent_type ?? null;
    const isSubagent = !!interaction.subagent_session_id;

    const model = interaction.model ?? null;
    const modelId = interaction.modelID ?? null;
    const providerId = interaction.providerID ?? null;
    const finishReason = interaction.finish_reason ?? null;

    let inputMessagesJson: string | null = null;
    let inputMessagesCount = 0;
    let inputMessagesTokens = 0;
    let contextWindowPct: number | null = null;

    if (role === 'assistant') {
      inputMessagesJson = null;
      let count = 0;
      const mySubagentSessionId = interaction.subagent_session_id ?? null;
      // After a /compact, the history is replaced by the continuation summary,
      // so count only turns from the last compact boundary onward (plus the
      // summary itself) instead of every pre-compact turn.
      const startJ = prevCompactBoundaryIdx >= 0 ? prevCompactBoundaryIdx : 0;
      for (let j = startJ; j < i; j++) {
        const prev = interactions[j];
        const prevRole = prev.role === 'subagent' ? 'assistant' : prev.role;
        // For subagent turns, only count prior turns in the same subagent session
        // For root turns, count all prior root turns (skip subagent turns)
        if (mySubagentSessionId) {
          if (prev.subagent_session_id === mySubagentSessionId &&
              (prevRole === 'user' || prevRole === 'assistant' || prevRole === 'system')) count++;
        } else {
          if (!prev.subagent_session_id &&
              (prevRole === 'user' || prevRole === 'assistant' || prevRole === 'system')) count++;
        }
      }
      inputMessagesCount = count;

      // Use totalTokens (the authoritative prompt size reported by the agent)
      // as the context-size base. It correctly reflects /compact — the prompt
      // shrinks when the history is replaced by a summary. The adapter's
      // input+cacheRead+cacheWrite proxy is unreliable on cache-cold turns
      // (post-compact cacheRead=0) and under-reports the real prompt.
      const adapterInputMessagesTokens = usage?.inputMessagesTokens ?? 0;
      inputMessagesTokens = totalTokens > 0 ? totalTokens : adapterInputMessagesTokens;

      // Monotonic floor: within a compact segment the context should not
      // decrease (smooths cache-read noise / reporting dips). The floor resets
      // to 0 at each /compact boundary (set when a continuation turn is seen),
      // so the legitimate post-compact drop passes through. This handles a
      // session with multiple compactions: each continuation starts a fresh
      // growing segment.
      const contextKey = isSubagent ? (subagentSessionId ?? 'sub') : 'root';
      if (contextKey !== prevContextKey) {
        prevInputMessagesTokens = 0;
        prevCompactBoundaryIdx = -1;
        prevContextKey = contextKey;
      }
      if (inputMessagesTokens > 0 && inputMessagesTokens < prevInputMessagesTokens) {
        inputMessagesTokens = prevInputMessagesTokens;
      }
      prevInputMessagesTokens = Math.max(prevInputMessagesTokens, inputMessagesTokens);

      const contextWindowLimit = getContextWindowLimit(model);
      contextWindowPct = inputMessagesTokens > 0
        ? (inputMessagesTokens / contextWindowLimit) * 100
        : null;
    }

    // A continuation turn ("This session is being continued...") is the boundary
    // produced by /compact: the conversation history is replaced by the summary
    // that follows. Mark it so the next assistant turn in this context resets
    // its monotonic floor and input-message count.
    if (role === 'user' && !isSubagent && content && isContinuationTurn(content)) {
      prevInputMessagesTokens = 0;
      prevCompactBoundaryIdx = i;
    }

    const turn: TurnRow = {
      id: turnId,
      sessionId,
      turnIndex,
      role,
      content,
      contentJson: null,
      contentSummary,
      inputMessagesJson,
      inputMessagesCount,
      inputMessagesTokens,
      contextWindowPct,
      agentName,
      subagentName,
      subagentSessionId,
      subagentType,
      totalTokens,
      inputTokens,
      outputTokens,
      reasoningTokens,
      cacheReadTokens,
      cacheWriteTokens,
      cost,
      createdAt_ts,
      completedAt,
      latencyMs,
      ttftMs: null,
      model,
      modelId,
      providerId,
      temperature: null,
      maxTokens: null,
      finishReason,
      isSubagent,
      parentExecutionId: null,
    };

    turns.push(turn);

    if (interaction.tool_calls) {
      for (const tc of interaction.tool_calls) {
        const toolCallRowId = generateId();
        const isSkillRelated = isSkillToolCall(tc.toolName) || isAgentSkillDispatch(tc.toolName, tc.argsJson);
        const errMsg = extractErrorMessage(tc.resultJson);

        const toolCallRow: ToolCallRow = {
          id: toolCallRowId,
          turnId,
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          argsJson: tc.argsJson,
          resultJson: tc.resultJson,
          state: tc.state,
          errorType: classifyError(tc.state, errMsg),
          errorMessage: errMsg,
          startedAt: createdAt_ts,
          completedAt: completedAt ?? createdAt_ts,
          durationMs: 0,
          dispatchBridgeId: null,
          isSkillRelated,
        };

        toolCalls.push(toolCallRow);

        if (isSkillRelated) {
          const skillEventRowId = generateId();
          const isDispatch = isAgentSkillDispatch(tc.toolName, tc.argsJson);
          const eventType = isDispatch ? 'dispatch' : getSkillEventType(tc.toolName);
          const skillName = isDispatch ? extractDispatchSkillName(tc.argsJson) : extractSkillName(tc.toolName, tc.argsJson);
          const skillVersion = isDispatch ? null : extractSkillVersion(tc.argsJson);
          const success = (tc.state === 'ok' || tc.state === 'completed') && !errMsg;

          const skillEventRow: SkillEventRow = {
            id: skillEventRowId,
            turnId,
            skillName,
            skillVersion,
            eventType,
            success,
            errorMessage: success ? null : (errMsg || tc.state),
            argsJson: tc.argsJson,
            startedAt: createdAt_ts,
            completedAt: completedAt ?? createdAt_ts,
            durationMs: 0,
          };

          skillEvents.push(skillEventRow);
        }
      }
    }
  }

  return { turns, toolCalls, skillEvents };
}
