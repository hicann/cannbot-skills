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
      'SELECT id, title, version, time_created FROM session WHERE parent_id IS NULL OR parent_id = \'\' ORDER BY time_created DESC'
    ).all() as { id: string; title: string; version: string; time_created: number }[];

    if (sessions.length === 0) return [];

    const sessionIds = sessions.map(s => s.id);

    const userMsgs = db.prepare(
      `SELECT session_id, id, data FROM message WHERE session_id IN (${sessionIds.map(() => '?').join(',')}) AND json_extract(data, '$.role') = 'user' ORDER BY time_created`
    ).all(...sessionIds) as { session_id: string; id: string; data: string }[];

    const msgCounts = db.prepare(
      `SELECT session_id, COUNT(*) as cnt FROM message WHERE session_id IN (${sessionIds.map(() => '?').join(',')}) GROUP BY session_id`
    ).all(...sessionIds) as { session_id: string; cnt: number }[];

    const countBySession = new Map<string, number>();
    for (const c of msgCounts) {
      countBySession.set(c.session_id, c.cnt);
    }

    const assistantMsgs = db.prepare(
      `SELECT session_id, data FROM message WHERE session_id IN (${sessionIds.map(() => '?').join(',')}) AND json_extract(data, '$.role') = 'assistant' ORDER BY time_created`
    ).all(...sessionIds) as { session_id: string; data: string }[];

    const assistantMsgBySession = new Map<string, { data: string }>();
    for (const m of assistantMsgs) {
      if (!assistantMsgBySession.has(m.session_id)) {
        assistantMsgBySession.set(m.session_id, { data: m.data });
      }
    }

    // Only need first user message per session — filter parts by those specific message IDs
    const firstUserMsgBySession = new Map<string, string>();
    for (const m of userMsgs) {
      if (!firstUserMsgBySession.has(m.session_id)) {
        firstUserMsgBySession.set(m.session_id, m.id);
      }
    }

    const firstUserMsgIds = [...firstUserMsgBySession.values()];

    // Filtered text parts — only for first user messages, not the entire DB
    const textParts = firstUserMsgIds.length > 0
      ? (db.prepare(
          `SELECT message_id, data FROM part WHERE message_id IN (${firstUserMsgIds.map(() => '?').join(',')}) AND json_extract(data, '$.type') = 'text' ORDER BY time_created`
        ).all(...firstUserMsgIds) as { message_id: string; data: string }[])
      : [];

    const textByMsgId = new Map<string, string[]>();
    for (const p of textParts) {
      try {
        const pd = JSON.parse(p.data);
        const text = pd.text || '';
        if (!textByMsgId.has(p.message_id)) textByMsgId.set(p.message_id, []);
        textByMsgId.get(p.message_id)!.push(text);
      } catch { /* skip */ }
    }

    // Store full user msg data (id + data) for fallback text extraction
    const userMsgDataBySession = new Map<string, { id: string; data: string }>();
    for (const m of userMsgs) {
      if (!userMsgDataBySession.has(m.session_id)) {
        userMsgDataBySession.set(m.session_id, { id: m.id, data: m.data });
      }
    }

    const result: SessionListItem[] = [];
    for (const session of sessions) {
      let firstQuery: string | null = null;
      const userMsgData = userMsgDataBySession.get(session.id);
      if (userMsgData) {
        try {
          const msgData = JSON.parse(userMsgData.data);
          if (msgData.content && typeof msgData.content === 'string') {
            firstQuery = msgData.content;
          } else {
            const texts = textByMsgId.get(userMsgData.id) ?? [];
            firstQuery = texts.join('\n').trim() || null;
          }
        } catch { /* skip */ }
      }

      const turnCount = countBySession.get(session.id) ?? 0;

      let modelName: string | null = null;
      const assistantMsgData = assistantMsgBySession.get(session.id);
      if (assistantMsgData) {
        try {
          const msgData = JSON.parse(assistantMsgData.data);
          if (msgData.modelID) {
            modelName = msgData.providerID && msgData.modelID
              ? `${msgData.providerID}/${msgData.modelID}`
              : msgData.modelID;
          } else if (msgData.model) {
            modelName = msgData.model.providerID && msgData.model.modelID
              ? `${msgData.model.providerID}/${msgData.model.modelID}`
              : msgData.model.modelID || null;
          }
        } catch { /* skip */ }
      }

      result.push({
        id: session.id,
        createdAt: new Date(session.time_created).toISOString(),
        firstQuery,
        turnCount,
        modelName,
        version: session.version,
      });
    }

    return result;
  } finally {
    db.close();
  }
}

export function listSubagentSessions(dbPath: string, parentSessionId: string): string[] {
  let db: Database.Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
  } catch {
    return [];
  }

  try {
    return _listSubagentSessions(db, parentSessionId);
  } finally {
    db.close();
  }
}

function _listSubagentSessions(db: Database.Database, parentSessionId: string): string[] {
  const children = db.prepare(
    'SELECT id FROM session WHERE parent_id = ?'
  ).all(parentSessionId) as { id: string }[];

  return children.map(c => c.id);
}

export function readSession(dbPath: string, sessionId: string): RawInteraction[] {
  let db: Database.Database | null = null;
  try {
    db = new Database(dbPath, { readonly: true });
  } catch {
    return [];
  }

  try {
    return _readSession(db, sessionId);
  } finally {
    db.close();
  }
}

function _readSession(db: Database.Database, sessionId: string): RawInteraction[] {
  const sessionRow = db.prepare(
    'SELECT id, parent_id FROM session WHERE id = ?'
  ).get(sessionId) as { id: string; parent_id: string | null } | undefined;

  if (!sessionRow) return [];

  const isSubagent = sessionRow.parent_id && sessionRow.parent_id !== '';

  const subagentInfo = isSubagent ? db.prepare(
    'SELECT id, title, parent_id FROM session WHERE id = ?'
  ).get(sessionId) as { id: string; title: string; parent_id: string } : null;

  let subagent_name: string | null = null;
  if (subagentInfo) {
    const titleMatch = subagentInfo.title.match(/@\(\w+\)\s+subagent/);
    if (titleMatch) {
      subagent_name = titleMatch[1];
    } else {
      const agentFromTitle = subagentInfo.title.match(/@(\w+)/);
      if (agentFromTitle) subagent_name = agentFromTitle[1];
    }
  }

  const messages = db.prepare(
    'SELECT id, time_created, data FROM message WHERE session_id = ? ORDER BY time_created'
  ).all(sessionId) as { id: string; time_created: number; data: string }[];

  const msgIds = messages.map(m => m.id);
  if (msgIds.length === 0) return [];

  const allTextParts = db.prepare(
    `SELECT message_id, data FROM part WHERE message_id IN (${msgIds.map(() => '?').join(',')}) AND json_extract(data, '$.type') = 'text' ORDER BY time_created`
  ).all(...msgIds) as { message_id: string; data: string }[];

  const allReasoningParts = db.prepare(
    `SELECT message_id, data FROM part WHERE message_id IN (${msgIds.map(() => '?').join(',')}) AND json_extract(data, '$.type') = 'reasoning' ORDER BY time_created`
  ).all(...msgIds) as { message_id: string; data: string }[];

  const allToolParts = db.prepare(
    `SELECT message_id, data FROM part WHERE message_id IN (${msgIds.map(() => '?').join(',')}) AND json_extract(data, '$.type') = 'tool' ORDER BY time_created`
  ).all(...msgIds) as { message_id: string; data: string }[];

  const textByMsg = new Map<string, string[]>();
  for (const p of allTextParts) {
    try {
      const pd = JSON.parse(p.data);
      if (pd.text) {
        if (!textByMsg.has(p.message_id)) textByMsg.set(p.message_id, []);
        textByMsg.get(p.message_id)!.push(pd.text);
      }
    } catch { /* skip */ }
  }

  const reasoningByMsg = new Map<string, string[]>();
  for (const p of allReasoningParts) {
    try {
      const pd = JSON.parse(p.data);
      if (pd.text) {
        if (!reasoningByMsg.has(p.message_id)) reasoningByMsg.set(p.message_id, []);
        reasoningByMsg.get(p.message_id)!.push(pd.text);
      }
    } catch { /* skip */ }
  }

  const toolCallsByMsg = new Map<string, ToolCallInfo[]>();
  for (const p of allToolParts) {
    try {
      const pd = JSON.parse(p.data);
      const callID = pd.callID || '';
      const toolName = pd.tool || '';
      const state = pd.state?.status || 'unknown';

      let argsJson: string | null = null;
      if (pd.input) {
        argsJson = JSON.stringify(pd.input);
      } else if (pd.state?.input) {
        const stateInput = pd.state.input;
        const mergedArgs: Record<string, unknown> = { ...stateInput };
        if (pd.state.metadata?.sessionId) {
          mergedArgs.subagent_session_id = pd.state.metadata.sessionId;
        }
        if (pd.state.metadata?.model) {
          mergedArgs.subagent_model = pd.state.metadata.model;
        }
        if (pd.state.title) {
          mergedArgs.summary = pd.state.title;
        }
        argsJson = JSON.stringify(mergedArgs);
      }

      let resultJson: string | null = null;
      if (pd.output) {
        resultJson = typeof pd.output === 'string' ? pd.output : JSON.stringify(pd.output);
      } else if (pd.state?.output) {
        resultJson = typeof pd.state.output === 'string' ? pd.state.output : JSON.stringify(pd.state.output);
      }

      const tc: ToolCallInfo = { toolCallId: callID, toolName, argsJson, resultJson, state };
      if (!toolCallsByMsg.has(p.message_id)) toolCallsByMsg.set(p.message_id, []);
      toolCallsByMsg.get(p.message_id)!.push(tc);
    } catch { /* skip */ }
  }

  const result: RawInteraction[] = [];

  for (const msg of messages) {
    try {
      const msgData = JSON.parse(msg.data);
      const role = msgData.role || 'unknown';
      const content = extractMessageContentBulk(msg.id, msgData, textByMsg, reasoningByMsg);
      const agent = msgData.agent || null;

      const timeInfo = msgData.time
        ? {
            created: msgData.time.created ?? msg.time_created,
            completed: msgData.time.completed ?? undefined,
          }
        : { created: msg.time_created };

      let latency: number | null = null;
      if (timeInfo.completed && timeInfo.created) {
        latency = timeInfo.completed - timeInfo.created;
      }

      let usage: TokenUsage | null = null;
      if (msgData.tokens) {
        const input = msgData.tokens.input ?? 0;
        const cacheRead = msgData.tokens.cache?.read ?? 0;
        const cacheWrite = msgData.tokens.cache?.write ?? 0;
        usage = {
          total: msgData.tokens.total ?? 0,
          input,
          output: msgData.tokens.output ?? 0,
          reasoning: msgData.tokens.reasoning ?? 0,
          cacheRead,
          cacheWrite,
          cost: msgData.cost ?? 0,
          inputMessagesTokens: input + cacheRead + cacheWrite,
        };
      }

      let model: string | null = null;
      let modelID: string | null = null;
      let providerID: string | null = null;
      if (msgData.modelID) {
        modelID = msgData.modelID;
        providerID = msgData.providerID;
        model = providerID && modelID ? `${providerID}/${modelID}` : modelID;
      } else if (msgData.model) {
        providerID = msgData.model.providerID;
        modelID = msgData.model.modelID;
        model = providerID && modelID ? `${providerID}/${modelID}` : modelID;
      }

      const finish_reason = msgData.finish || null;

      const tool_calls = toolCallsByMsg.has(msg.id)
        ? toolCallsByMsg.get(msg.id)!
        : null;

      result.push({
        role,
        content,
        timestamp: new Date(msg.time_created).toISOString(),
        timeInfo,
        agent,
        subagent_name: isSubagent ? subagent_name : null,
        subagent_session_id: isSubagent ? sessionId : null,
        subagent_type: null,
        tool_calls,
        usage,
        model,
        modelID,
        providerID,
        latency,
        finish_reason,
      });
    } catch {
      result.push({
        role: 'unknown',
        content: null,
        timestamp: new Date(msg.time_created).toISOString(),
        timeInfo: { created: msg.time_created },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        subagent_type: null,
        tool_calls: null,
        usage: null,
        model: null,
        modelID: null,
        providerID: null,
        latency: null,
        finish_reason: null,
      });
    }
  }

  return result;
}

function extractMessageContentBulk(
  messageId: string,
  msgData: Record<string, unknown>,
  textByMsg: Map<string, string[]>,
  reasoningByMsg: Map<string, string[]>,
): string | null {
  if (msgData.content && typeof msgData.content === 'string') {
    return msgData.content;
  }

  const textContent = textByMsg.get(messageId) ?? [];
  const reasoningContent = reasoningByMsg.get(messageId) ?? [];

  const parts: string[] = [];
  if (reasoningContent.length > 0) {
    parts.push(`<thinking>${reasoningContent.join('\n')}</thinking>`);
  }
  if (textContent.length > 0) {
    parts.push(textContent.join('\n'));
  }

  return parts.length > 0 ? parts.join('\n\n') : null;
}

// ── WithDb variants: reuse an existing DB connection ──

export function listSubagentSessionsWithDb(db: Database.Database, parentSessionId: string): string[] {
  return _listSubagentSessions(db, parentSessionId);
}

export function readSessionWithDb(db: Database.Database, sessionId: string): RawInteraction[] {
  return _readSession(db, sessionId);
}

export interface SessionMeta {
  parentId: string | null;
  version: string | null;
  directory: string | null;
  summaryAdditions: number;
  summaryDeletions: number;
  summaryFiles: number;
}

export function readSessionMeta(db: Database.Database, sessionId: string): SessionMeta {
  const row = db.prepare(
    'SELECT parent_id, version, directory, summary_additions, summary_deletions, summary_files FROM session WHERE id = ?'
  ).get(sessionId) as {
    parent_id: string | null;
    version: string | null;
    directory: string | null;
    summary_additions: number | null;
    summary_deletions: number | null;
    summary_files: number | null;
  } | undefined;

  if (!row) return { parentId: null, version: null, directory: null, summaryAdditions: 0, summaryDeletions: 0, summaryFiles: 0 };

  return {
    parentId: row.parent_id && row.parent_id !== '' ? row.parent_id : null,
    version: row.version,
    directory: row.directory,
    summaryAdditions: row.summary_additions ?? 0,
    summaryDeletions: row.summary_deletions ?? 0,
    summaryFiles: row.summary_files ?? 0,
  };
}
