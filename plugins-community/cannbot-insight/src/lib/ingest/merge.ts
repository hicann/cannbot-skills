// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { TurnRow, ToolCallRow, SkillEventRow } from './turn-split';

export interface DedupSessionResult {
  shouldImport: boolean;
  existingSessionId: string | null;
}

export function dedupSession(
  existingSessionId: string | null,
  newTaskId: string
): DedupSessionResult {
  if (existingSessionId) {
    return { shouldImport: false, existingSessionId };
  }
  return { shouldImport: true, existingSessionId: null };
}

export function mergeTurns(existingTurns: TurnRow[], newTurns: TurnRow[]): TurnRow[] {
  const existingKeys = new Set(
    existingTurns.map(t => `${t.turnIndex}:${t.role}`)
  );

  const merged = [...existingTurns];
  for (const turn of newTurns) {
    const key = `${turn.turnIndex}:${turn.role}`;
    if (!existingKeys.has(key)) {
      merged.push(turn);
      existingKeys.add(key);
    }
  }

  merged.sort((a, b) => a.turnIndex - b.turnIndex || a.role.localeCompare(b.role));
  return merged;
}

export function mergeToolCalls(existing: ToolCallRow[], incoming: ToolCallRow[]): ToolCallRow[] {
  const existingIds = new Set(existing.map(tc => tc.toolCallId));

  const merged = [...existing];
  for (const tc of incoming) {
    if (!existingIds.has(tc.toolCallId)) {
      merged.push(tc);
      existingIds.add(tc.toolCallId);
    }
  }

  return merged;
}

export function mergeSkillEvents(existing: SkillEventRow[], incoming: SkillEventRow[]): SkillEventRow[] {
  const existingKeys = new Set(
    existing.map(se => `${se.turnId}:${se.skillName}:${se.eventType}`)
  );

  const merged = [...existing];
  for (const se of incoming) {
    const key = `${se.turnId}:${se.skillName}:${se.eventType}`;
    if (!existingKeys.has(key)) {
      merged.push(se);
      existingKeys.add(key);
    }
  }

  return merged;
}

export interface TurnUpdate {
  dbId: string;
  data: Record<string, unknown>;
}

export interface ToolCallUpdate {
  dbId: string;
  data: Record<string, unknown>;
}

const TURN_UPDATE_FIELDS: (keyof TurnRow)[] = [
  'content', 'contentSummary',
  'totalTokens', 'inputTokens', 'outputTokens', 'reasoningTokens',
  'cacheReadTokens', 'cacheWriteTokens', 'completedAt', 'latencyMs',
  'ttftMs', 'model', 'modelId', 'providerId', 'finishReason',
];

const TC_UPDATE_FIELDS: (keyof ToolCallRow)[] = [
  'resultJson', 'state', 'errorType', 'errorMessage',
  'completedAt', 'durationMs', 'dispatchBridgeId', 'isSkillRelated',
];

export function diffTurns(
  existingByKey: Map<string, TurnRow>,
  incoming: TurnRow[],
): { toInsert: TurnRow[]; toUpdate: TurnUpdate[] } {
  const toInsert: TurnRow[] = [];
  const toUpdate: TurnUpdate[] = [];

  for (const turn of incoming) {
    const key = `${turn.turnIndex}:${turn.role}`;
    const existing = existingByKey.get(key);
    if (!existing) {
      toInsert.push(turn);
      continue;
    }
    const changes: Record<string, unknown> = {};
    for (const field of TURN_UPDATE_FIELDS) {
      const oldVal = existing[field];
      const newVal = turn[field];
      if (oldVal !== newVal && (oldVal !== null || newVal !== null)) {
        changes[field] = newVal;
      }
    }
    if (Object.keys(changes).length > 0) {
      toUpdate.push({ dbId: existing.id, data: changes });
    }
  }

  return { toInsert, toUpdate };
}

export function diffToolCalls(
  existingById: Map<string, ToolCallRow>,
  incoming: ToolCallRow[],
): { toInsert: ToolCallRow[]; toUpdate: ToolCallUpdate[] } {
  const toInsert: ToolCallRow[] = [];
  const toUpdate: ToolCallUpdate[] = [];

  for (const tc of incoming) {
    const existing = existingById.get(tc.toolCallId);
    if (!existing) {
      toInsert.push(tc);
      continue;
    }
    const changes: Record<string, unknown> = {};
    for (const field of TC_UPDATE_FIELDS) {
      const oldVal = existing[field];
      const newVal = tc[field];
      if (oldVal !== newVal && (oldVal !== null || newVal !== null)) {
        changes[field] = newVal;
      }
    }
    if (Object.keys(changes).length > 0) {
      toUpdate.push({ dbId: existing.id, data: changes });
    }
  }

  return { toInsert, toUpdate };
}
