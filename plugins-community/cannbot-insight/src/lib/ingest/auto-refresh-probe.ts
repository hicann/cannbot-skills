// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';
import type { PrismaClient } from '@prisma/client';
import { isClaudeFormatSession } from '../shared/session-format';

export interface AutoRefreshProbe {
  countChanged: boolean;
  sourceMessageCount: number;
  ourTurnCount: number;
  maxTimeUpdated: number;
  streaming: boolean;
  pendingInput: boolean;
  settled: boolean;
}

export interface ProbeSession {
  id: string;
  taskId: string;
  sourcePath: string | null;
  framework: string | null;
  // proxy 捕获判定需要（framework=opencode + version 带 -proxy 是 claude 格式）
  version?: string | null;
}

const NO_CHANGE: AutoRefreshProbe = {
  countChanged: false,
  sourceMessageCount: 0,
  ourTurnCount: 0,
  maxTimeUpdated: 0,
  streaming: false,
  pendingInput: false,
  settled: false,
};

// Claude Code metadata line types that don't carry conversational content.
// Mirrors NON_BREAKING_TYPES in the claude-jsonl adapter.
const CLAUDE_METADATA_TYPES = new Set([
  'ai-title',
  'attachment',
  'mode',
  'permission-mode',
  'file-history-snapshot',
  'last-prompt',
  'system',
]);

export async function probeAutoRefresh(
  session: ProbeSession,
  prisma: PrismaClient,
): Promise<AutoRefreshProbe> {
  if (!session.sourcePath) return { ...NO_CHANGE };
  // claude 格式判定必须先于 opencode-db 分支：proxy 捕获的 opencode 会话
  // sourcePath 是 jsonl，按 sqlite 打开只会抛错退化为 NO_CHANGE（永不触发
  // 自动刷新）。原生 opencode（version 无 -proxy）不受影响。
  if (isClaudeFormatSession(session.framework, session.version)) {
    return probeClaudeCodeRefresh(session);
  }
  if (session.framework === 'opencode') {
    return probeOpencodeRefresh(session, prisma);
  }
  return { ...NO_CHANGE };
}

async function probeOpencodeRefresh(
  session: ProbeSession,
  prisma: PrismaClient,
): Promise<AutoRefreshProbe> {
  const taskId = session.taskId;
  let db: Database.Database | null = null;
  try {
    db = new Database(session.sourcePath!, { readonly: true });

    const parentStats = db.prepare(
      'SELECT COUNT(*) as cnt, MAX(time_updated) as max_updated FROM message WHERE session_id = ?'
    ).get(taskId) as { cnt: number; max_updated: number | null };

    const subagentStats = db.prepare(
      'SELECT COUNT(*) as cnt, MAX(time_updated) as max_updated FROM message WHERE session_id IN (SELECT id FROM session WHERE parent_id = ?)'
    ).get(taskId) as { cnt: number; max_updated: number | null };

    const sourceMessageCount = parentStats.cnt + subagentStats.cnt;
    const maxTimeUpdated = Math.max(
      parentStats.max_updated ?? 0,
      subagentStats.max_updated ?? 0
    );

    const incompleteParent = db.prepare(
      `SELECT COUNT(*) as cnt FROM message WHERE session_id = ? AND json_extract(data, '$.role') = 'assistant' AND json_extract(data, '$.time.completed') IS NULL`
    ).get(taskId) as { cnt: number };

    const incompleteSub = db.prepare(
      `SELECT COUNT(*) as cnt FROM message WHERE session_id IN (SELECT id FROM session WHERE parent_id = ?) AND json_extract(data, '$.role') = 'assistant' AND json_extract(data, '$.time.completed') IS NULL`
    ).get(taskId) as { cnt: number };

    const streaming = (incompleteParent.cnt + incompleteSub.cnt) > 0;

    const latestParent = db.prepare(
      `SELECT json_extract(data, '$.role') as role FROM message WHERE session_id = ? ORDER BY time_created DESC LIMIT 1`
    ).get(taskId) as { role: string | null } | undefined;

    const latestSub = db.prepare(
      `SELECT json_extract(data, '$.role') as role FROM message WHERE session_id IN (SELECT id FROM session WHERE parent_id = ?) ORDER BY time_created DESC LIMIT 1`
    ).get(taskId) as { role: string | null } | undefined;

    const pendingInput = (latestParent?.role === 'user' || latestSub?.role === 'user');

    const ourTurnCount = await prisma.turn.count({ where: { sessionId: session.id } });
    const countChanged = sourceMessageCount > ourTurnCount;
    const settled = !streaming && !pendingInput;

    return { countChanged, sourceMessageCount, ourTurnCount, maxTimeUpdated, streaming, pendingInput, settled };
  } catch {
    return { ...NO_CHANGE };
  } finally {
    if (db) db.close();
  }
}

async function probeClaudeCodeRefresh(
  session: ProbeSession,
): Promise<AutoRefreshProbe> {
  const filePath = resolveClaudeSessionFile(session.sourcePath!, session.taskId);
  if (!filePath) return { ...NO_CHANGE };

  let stat: fs.Stats;
  try {
    stat = fs.statSync(filePath);
  } catch {
    return { ...NO_CHANGE };
  }
  if (!stat.isFile()) return { ...NO_CHANGE };

  const maxTimeUpdated = stat.mtimeMs;

  let rawContent: string;
  try {
    rawContent = fs.readFileSync(filePath, 'utf-8');
  } catch {
    return { ...NO_CHANGE };
  }

  const lines = rawContent.length > 0
    ? rawContent.split('\n').filter(l => l.trim())
    : [];

  const parsed: Array<{ type?: string; message?: { role?: string; content?: unknown; stop_reason?: string | null }; stopReason?: string | null }> = [];
  for (const line of lines) {
    try {
      parsed.push(JSON.parse(line));
    } catch {
      // skip malformed lines
    }
  }

  // Find the last meaningful (non-metadata) line to determine session state.
  let lastMeaningful: { type?: string; message?: { content?: unknown; stop_reason?: string | null }; stopReason?: string | null } | null = null;
  for (let i = parsed.length - 1; i >= 0; i--) {
    const t = parsed[i].type;
    if (t && !CLAUDE_METADATA_TYPES.has(t)) {
      lastMeaningful = parsed[i];
      break;
    }
  }

  let streaming = false;
  let pendingInput = false;

  if (lastMeaningful) {
    const t = lastMeaningful.type;
    if (t === 'assistant') {
      // Claude Code emits a `stop_reason` (end_turn / tool_use / ...) on a finalized
      // assistant response segment. Mirrors opencode's `time.completed IS NULL` check:
      // a present stop_reason means the response is complete (not streaming); an
      // absent stop_reason means the response is still in progress. Proxy captures
      // carry it as a TOP-LEVEL `stopReason` extension field (v1.74+) — read both,
      // otherwise proxy captures look streaming forever and never auto-refresh.
      const stopReason = lastMeaningful.message?.stop_reason ?? lastMeaningful.stopReason;
      streaming = !stopReason;
    } else if (t === 'result') {
      // Final result marker emitted after a completed assistant response.
      // settled = true (neither streaming nor pendingInput).
    } else if (t === 'user') {
      const content = lastMeaningful.message?.content;
      const hasRealText =
        typeof content === 'string'
          ? content.trim().length > 0
          : Array.isArray(content) &&
            content.some(
              (b): b is { type: string; text?: string } =>
                typeof b === 'object' && b !== null && (b as { type: string }).type === 'text' &&
                Boolean(((b as { text?: string }).text ?? '').trim())
            );
      if (hasRealText) {
        // User prompt waiting for an assistant response.
        pendingInput = true;
      } else {
        // tool_result-only user line: mid tool loop, assistant response pending.
        streaming = true;
      }
    }
  }

  const settled = !streaming && !pendingInput;
  // claude-code detects change via mtime; count-based fields are opencode's
  // mechanism and not consumed by the claude client (left inert to skip the
  // prisma query on every file-change event).
  return { countChanged: false, sourceMessageCount: 0, ourTurnCount: 0, maxTimeUpdated, streaming, pendingInput, settled };
}

export function resolveClaudeSessionFile(sourcePath: string, sessionId: string): string | null {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(sourcePath);
  } catch {
    return null;
  }
  if (stat.isFile()) return sourcePath;
  if (stat.isDirectory()) {
    const direct = path.join(sourcePath, sessionId + '.jsonl');
    try {
      if (fs.existsSync(direct) && fs.statSync(direct).isFile()) return direct;
    } catch {
      // fall through to recursive search
    }
    return findJsonlRecursive(sourcePath, sessionId);
  }
  return null;
}

function findJsonlRecursive(dirPath: string, sessionId: string): string | null {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(dirPath, { withFileTypes: true });
  } catch {
    return null;
  }
  for (const entry of entries) {
    const full = path.join(dirPath, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'subagents') continue;
      const found = findJsonlRecursive(full, sessionId);
      if (found) return found;
    } else if (entry.isFile() && entry.name === sessionId + '.jsonl') {
      return full;
    }
  }
  return null;
}
