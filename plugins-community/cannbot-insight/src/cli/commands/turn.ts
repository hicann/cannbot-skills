// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { InsightClient } from '../client';
import { renderTable, TableColumn } from '../utils/table';
import { formatTokens, formatDuration, formatDate, truncate } from '../utils/format';
import { formatHeader, formatLabel, formatDivider, theme } from '../utils/colors';
import type { ApiTurnDetailResponse, ApiToolCallDetail, ApiSkillEventDetail } from '../types';

const TOOL_CALL_COLUMNS: TableColumn[] = [
  { key: 'toolName', label: 'Tool', width: 20 },
  { key: 'state', label: 'State', width: 10 },
  { key: 'durationMs', label: 'Duration', width: 10 },
];

const SKILL_COLUMNS: TableColumn[] = [
  { key: 'skillName', label: 'Skill', width: 20 },
  { key: 'eventType', label: 'Event', width: 10 },
  { key: 'success', label: 'Success', width: 8 },
  { key: 'durationMs', label: 'Duration', width: 10 },
];

function renderToolCallRow(row: ApiToolCallDetail, key: string): string {
  switch (key) {
    case 'toolName': return row.toolName ?? '—';
    case 'state': return row.state ?? '—';
    case 'durationMs': return formatDuration(row.durationMs);
    default: return String(row[key as keyof ApiToolCallDetail] ?? '—');
  }
}

function renderSkillRow(row: ApiSkillEventDetail, key: string): string {
  switch (key) {
    case 'skillName': return row.skillName ?? '—';
    case 'eventType': return row.eventType ?? '—';
    case 'success': return row.success === null ? '—' : row.success ? '✓' : '✗';
    case 'durationMs': return formatDuration(row.durationMs);
    default: return String(row[key as keyof ApiSkillEventDetail] ?? '—');
  }
}

function displayTurnDetail(turn: ApiTurnDetailResponse): string {
  const lines: string[] = [];

  lines.push(formatHeader(`Turn #${turn.turnIndex}: ${turn.role}`));
  lines.push(formatDivider());
  lines.push('');

  lines.push(formatLabel('Turn ID', turn.turnId));
  lines.push(formatLabel('Session ID', turn.sessionId));
  lines.push(formatLabel('Turn Index', String(turn.turnIndex)));
  lines.push(formatLabel('Role', turn.role));
  lines.push(formatLabel('Model', turn.model ?? '—'));
  lines.push(formatLabel('Model ID', turn.modelId ?? '—'));
  lines.push(formatLabel('Provider', turn.providerId ?? '—'));
  lines.push(formatLabel('Agent', turn.agentName ?? '—'));
  if (turn.isSubagent) {
    lines.push(formatLabel('Subagent', turn.subagentName ?? '—'));
    lines.push(formatLabel('Sub Session', turn.subagentSessionId ?? '—'));
  }
  lines.push(formatLabel('Finish Reason', turn.finishReason ?? '—'));
  lines.push(formatLabel('Created', formatDate(turn.createdAt)));
  lines.push(formatLabel('Completed', formatDate(turn.completedAt)));
  lines.push(formatLabel('Latency', formatDuration(turn.latencyMs)));
  if (turn.ttftMs !== null) {
    lines.push(formatLabel('TTFT', formatDuration(turn.ttftMs)));
  }
  lines.push('');

  lines.push(formatHeader('Token Usage'));
  lines.push(formatDivider());
  lines.push('');
  lines.push(formatLabel('Total Tokens', formatTokens(turn.totalTokens)));
  lines.push(formatLabel('Input Tokens', formatTokens(turn.inputTokens)));
  lines.push(formatLabel('Output Tokens', formatTokens(turn.outputTokens)));
  lines.push(formatLabel('Reasoning Tokens', formatTokens(turn.reasoningTokens)));
  lines.push(formatLabel('Cache Read', formatTokens(turn.cacheReadTokens)));
  lines.push(formatLabel('Cache Write', formatTokens(turn.cacheWriteTokens)));
  if (turn.inputMessagesCount !== null) {
    lines.push(formatLabel('Input Messages', String(turn.inputMessagesCount)));
    lines.push(formatLabel('Input Msg Tokens', formatTokens(turn.inputMessagesTokens)));
  }
  if (turn.contextWindowPct !== null) {
    lines.push(formatLabel('Context Window', `${turn.contextWindowPct.toFixed(1)}%`));
  }
  lines.push('');

  if (turn.contentSummary) {
    lines.push(formatHeader('Content Summary'));
    lines.push(formatDivider());
    lines.push('');
    lines.push(truncate(turn.contentSummary, 300));
    lines.push('');
  }

  if (turn.toolCalls?.length > 0) {
    lines.push(formatHeader('Tool Calls'));
    lines.push(renderTable(
      TOOL_CALL_COLUMNS,
      turn.toolCalls as unknown as Record<string, unknown>[],
      renderToolCallRow as unknown as (row: Record<string, unknown>, key: string) => string,
    ));
    lines.push('');
  }

  if (turn.skillEvents?.length > 0) {
    lines.push(formatHeader('Skill Events'));
    lines.push(renderTable(
      SKILL_COLUMNS,
      turn.skillEvents as unknown as Record<string, unknown>[],
      renderSkillRow as unknown as (row: Record<string, unknown>, key: string) => string,
    ));
    lines.push('');
  }

  return lines.join('\n');
}

export function turnCommand(): Command {
  const cmd = new Command('turn');
  cmd
    .description('Show turn detail')
    .argument('<turnId>', 'Turn ID')
    .option('--json', 'Output as JSON')
    .action(async (turnId, opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const turn = await client.getTurnDetail(turnId);

      if (opts.json) {
        console.log(JSON.stringify(turn, null, 2));
        return;
      }

      const output = displayTurnDetail(turn);
      console.log(output);
    });

  return cmd;
}
