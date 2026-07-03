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
