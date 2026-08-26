// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { isContinuationTurn } from '@/lib/shared/command-parser';

export interface ContextTurn {
  id: string;
  turnIndex: number;
  role: string;
  content: string | null;
  agentName?: string | null;
  isSubagent?: boolean;
  subagentSessionId?: string | null;
}

// A real local-command turn (command/caveat/stdout) STARTS with its tag.
// A /compact continuation summary may CONTAIN those tags mid-text (it
// summarizes a conversation that used commands), so an anywhere-match would
// wrongly drop the summary itself — hence the start-anchored check.
export function isLocalCommandNoise(text: string | null): boolean {
  if (!text) return false;
  return text.startsWith('<command-name>')
    || text.startsWith('<local-command-caveat>')
    || text.startsWith('<local-command-stdout>');
}

// Select the context turns that form the LLM input for the assistant turn at
// `targetTurnIndex`. A /compact continuation replaces the prior conversation
// history with a summary, so the window starts at the most recent continuation
// before the target — not at session start. Handles multiple compactions: each
// continuation truncates the window again. Local CLI command noise (command,
// caveat, stdout) is excluded since it is not sent to the LLM.
export function selectInputContextTurns(
  contextTurns: ContextTurn[],
  targetTurnIndex: number,
  targetAgentName?: string | null,
): ContextTurn[] {
  const isCompactionTurn = targetAgentName === 'compaction';

  let startTurnIndex = 0;
  for (const ct of contextTurns) {
    if (ct.turnIndex < targetTurnIndex) {
      if (ct.content && isContinuationTurn(ct.content)) {
        startTurnIndex = Math.max(startTurnIndex, ct.turnIndex);
      }
      if (ct.agentName === 'compaction' && !isCompactionTurn) {
        startTurnIndex = Math.max(startTurnIndex, ct.turnIndex);
      }
      if (ct.agentName === 'compaction-boundary' && !isCompactionTurn) {
        startTurnIndex = Math.max(startTurnIndex, ct.turnIndex + 1);
      }
    }
  }
  return contextTurns.filter(ct =>
    ct.turnIndex < targetTurnIndex &&
    ct.turnIndex >= startTurnIndex &&
    ['user', 'assistant', 'system', 'tool_result'].includes(ct.role) &&
    !isLocalCommandNoise(ct.content) &&
    ct.agentName !== 'compaction-boundary',
  );
}

// Wire-fidelity pass for LLM Input reconstruction. On the wire, a
// <system-reminder> context block rides INSIDE a user message as a text block
// (reminder block + prompt block); the claude-jsonl adapter splits it into a
// system turn + user turn for the conversation view. Merge it back here so the
// reconstructed message list mirrors the original request: a reminder-split
// system turn directly before a user turn folds into that user message, and a
// standalone reminder turn (pure context injection, no prompt) becomes a user
// message — exactly what was sent. Registry / skill system turns (no reminder
// wrapper) pass through unchanged.
export function toWireOrder(contextTurns: ContextTurn[]): ContextTurn[] {
  const out: ContextTurn[] = [];
  for (let i = 0; i < contextTurns.length; i++) {
    const ct = contextTurns[i];
    const isReminder = ct.role === 'system' && (ct.content ?? '').startsWith('<system-reminder>');
    if (isReminder) {
      const next = contextTurns[i + 1];
      if (next?.role === 'user') {
        // the reminder turn preserves the original trailing whitespace, so
        // plain concatenation reproduces the exact wire message text
        out.push({ ...next, content: `${ct.content}${next.content ?? ''}` });
        i++;
        continue;
      }
      out.push({ ...ct, role: 'user' });
      continue;
    }
    out.push(ct);
  }
  return out;
}
