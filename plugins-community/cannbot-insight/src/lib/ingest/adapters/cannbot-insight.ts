// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import Database from 'better-sqlite3';
import type { SessionListItem, RawInteraction, ToolCallInfo, TokenUsage } from '../../shared/types';

export function listSessions(dbPath: string): SessionListItem[] {
  let db: Database.Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
  } catch {
    return [];
  }

  try {
    const sessions = db.prepare(
      'SELECT id, taskId, query, model, startTime, totalLlmCallCount FROM "Session" ORDER BY startTime DESC'
    ).all() as {
      id: string; taskId: string; query: string | null; model: string | null;
      startTime: string; totalLlmCallCount: number;
    }[];

    let versionBySession: Map<string, string | null> = new Map();
    try {
      const versionRows = db.prepare(
        'SELECT taskId, version FROM "Session"'
      ).all() as { taskId: string; version: string | null }[];
      for (const r of versionRows) versionBySession.set(r.taskId, r.version);
    } catch { /* column may not exist in older exports */ }

    return sessions.map(s => ({
      id: s.taskId,
      createdAt: new Date(s.startTime).toISOString(),
      firstQuery: s.query,
      turnCount: s.totalLlmCallCount,
      modelName: s.model,
      version: versionBySession.get(s.taskId) ?? null,
    }));
  } finally {
    db.close();
  }
}

export function readSession(dbPath: string, sessionId: string): RawInteraction[] {
  let db: Database.Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
  } catch {
    return [];
  }

  try {
    const session = db.prepare(
      'SELECT id, taskId FROM "Session" WHERE taskId = ?'
    ).get(sessionId) as { id: string; taskId: string } | undefined;

    if (!session) return [];

    const turns = db.prepare(
      'SELECT * FROM "Turn" WHERE sessionId = ? ORDER BY turnIndex'
    ).all(session.id) as TurnRow[];

    if (turns.length === 0) return [];

    const turnIds = turns.map(t => t.id);

    const toolCalls = turnIds.length > 0
      ? (db.prepare(
          `SELECT * FROM "ToolCall" WHERE turnId IN (${turnIds.map(() => '?').join(',')})`
        ).all(...turnIds) as ToolCallRow[])
      : [];

    const skillEvents = turnIds.length > 0
      ? (db.prepare(
          `SELECT * FROM "SkillEvent" WHERE turnId IN (${turnIds.map(() => '?').join(',')})`
        ).all(...turnIds) as SkillEventRow[])
      : [];

    const toolCallsByTurn = new Map<string, ToolCallInfo[]>();
    for (const tc of toolCalls) {
      const info: ToolCallInfo = {
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        argsJson: tc.argsJson,
        resultJson: tc.resultJson,
        state: tc.state,
      };
      if (!toolCallsByTurn.has(tc.turnId)) toolCallsByTurn.set(tc.turnId, []);
      toolCallsByTurn.get(tc.turnId)!.push(info);
    }

    const result: RawInteraction[] = [];

    for (const t of turns) {
      const createdTs = (t.createdAt_ts ?? t.createdAt) ? new Date(t.createdAt_ts ?? t.createdAt).getTime() : 0;
      const completedTs = t.completedAt ? new Date(t.completedAt).getTime() : undefined;

      const timeInfo = {
        created: createdTs,
        completed: completedTs,
      };

      const latency = (t.latencyMs ?? 0) || (completedTs && createdTs ? completedTs - createdTs : null);

      const totalTokens = t.totalTokens ?? 0;
      const inputTokens = t.inputTokens ?? 0;
      const outputTokens = t.outputTokens ?? 0;
      const reasoningTokens = t.reasoningTokens ?? 0;
      const cacheReadTokens = t.cacheReadTokens ?? 0;
      const cacheWriteTokens = t.cacheWriteTokens ?? 0;

      const usage: TokenUsage | null = {
        total: totalTokens,
        input: inputTokens,
        output: outputTokens,
        reasoning: reasoningTokens,
        cacheRead: cacheReadTokens,
        cacheWrite: cacheWriteTokens,
        cost: 0,
        inputMessagesTokens: inputTokens + cacheReadTokens + cacheWriteTokens,
      };

      let model: string | null = null;
      if (t.modelId) {
        model = t.providerId && t.modelId ? `${t.providerId}/${t.modelId}` : t.modelId;
      } else if (t.model) {
        model = t.model;
      }

      result.push({
        role: t.role,
        content: t.content ?? t.contentSummary ?? null,
        timestamp: new Date(createdTs || Date.now()).toISOString(),
        timeInfo,
        agent: t.agentName ?? null,
        subagent_name: t.subagentName ?? null,
        subagent_session_id: t.subagentSessionId ?? null,
        subagent_type: null,
        tool_calls: toolCallsByTurn.has(t.id) ? toolCallsByTurn.get(t.id)! : null,
        usage,
        model,
        modelID: t.modelId ?? null,
        providerID: t.providerId ?? null,
        latency,
        finish_reason: t.finishReason ?? null,
      });
    }

    return result;
  } finally {
    db.close();
  }
}

interface TurnRow {
  id: string;
  sessionId: string;
  turnIndex: number;
  role: string;
  content: string | null;
  contentJson: string | null;
  contentSummary: string | null;
  agentName: string | null;
  subagentName: string | null;
  subagentSessionId: string | null;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  createdAt_ts: string | null;
  completedAt: string | null;
  latencyMs: number;
  ttftMs: number | null;
  model: string | null;
  modelId: string | null;
  providerId: string | null;
  finishReason: string | null;
  isSubagent: number;
  createdAt: string;
}

interface ToolCallRow {
  id: string;
  turnId: string;
  toolCallId: string;
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
}

interface SkillEventRow {
  id: string;
  turnId: string;
  skillName: string;
  eventType: string;
}
