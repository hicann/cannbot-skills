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
import { formatTokens, formatCost, formatDuration, formatDate } from '../utils/format';
import { formatHeader, formatLabel, formatDivider } from '../utils/colors';
import type { ApiSessionDetailResponse } from '../types';

function displayDetail(session: ApiSessionDetailResponse): string {
  const lines: string[] = [];

  lines.push(formatHeader(`Session: ${session.taskId}`));
  lines.push(formatDivider());
  lines.push('');

  lines.push(formatLabel('Task ID', session.taskId));
  lines.push(formatLabel('Session ID', session.sessionId));
  lines.push(formatLabel('Model', session.model ?? '—'));
  lines.push(formatLabel('User', session.user ?? '—'));
  lines.push(formatLabel('Query', session.query ?? '—'));
  lines.push(formatLabel('Start', formatDate(session.startTime)));
  lines.push(formatLabel('End', formatDate(session.endTime)));
  lines.push(formatLabel('Duration', formatDuration(session.totalLatencyMs)));
  lines.push('');

  lines.push(formatHeader('Token Usage'));
  lines.push(formatDivider());
  lines.push('');
  lines.push(formatLabel('Total Tokens', formatTokens(session.totalTokens)));
  lines.push(formatLabel('Input Tokens', formatTokens(session.totalInputTokens)));
  lines.push(formatLabel('Output Tokens', formatTokens(session.totalOutputTokens)));
  lines.push(formatLabel('Reasoning Tokens', formatTokens(session.totalReasoningTokens)));
  lines.push(formatLabel('Cache Read', formatTokens(session.totalCacheReadTokens)));
  lines.push(formatLabel('Cache Write', formatTokens(session.totalCacheWriteTokens)));
  lines.push('');

  lines.push(formatHeader('Cost & Activity'));
  lines.push(formatDivider());
  lines.push('');
  lines.push(formatLabel('Total Cost', formatCost(session.totalCost)));
  lines.push(formatLabel('Tool Calls', String(session.totalToolCallCount ?? 0)));
  lines.push(formatLabel('LLM Calls', String(session.totalLlmCallCount ?? 0)));
  lines.push(formatLabel('Skill Loads', String(session.totalSkillLoadCount ?? 0)));
  lines.push(formatLabel('Subagents', String(session.totalSubagentCount ?? 0)));
  lines.push('');

  if (session.agents?.length > 0) {
    const AGENT_COLUMNS: TableColumn[] = [
      { key: 'agentName', label: 'Agent', width: 15 },
      { key: 'isSubagent', label: 'Sub', width: 6 },
      { key: 'model', label: 'Model', width: 18 },
      { key: 'tokens', label: 'Tokens', width: 10 },
      { key: 'cost', label: 'Cost', width: 10 },
      { key: 'latency', label: 'Latency', width: 10 },
    ];

    lines.push(formatHeader('Agents'));
    lines.push(renderTable(AGENT_COLUMNS, session.agents as unknown as Record<string, unknown>[], (row: Record<string, unknown>, key: string) => {
      const a = row as unknown as ApiSessionDetailResponse['agents'][0];
      switch (key) {
        case 'tokens': return formatTokens(a.tokens);
        case 'cost': return formatCost(a.cost);
        case 'latency': return formatDuration(a.latencyMs);
        case 'isSubagent': return a.isSubagent ? '✓' : '—';
        case 'agentName': return a.agentName ?? '—';
        case 'model': return a.model ?? '—';
        default: return String(a[key as keyof typeof a] ?? '—');
      }
    }));
    lines.push('');
  }

  if (session.skills?.length > 0) {
    const SKILL_COLUMNS: TableColumn[] = [
      { key: 'skillName', label: 'Skill', width: 25 },
      { key: 'version', label: 'Version', width: 10 },
      { key: 'invocations', label: 'Count', width: 8 },
    ];

    lines.push(formatHeader('Skills'));
    lines.push(renderTable(SKILL_COLUMNS, session.skills as unknown as Record<string, unknown>[]));
    lines.push('');
  }

  return lines.join('\n');
}

export function sessionCommand(): Command {
  const cmd = new Command('session');
  cmd
    .description('Show session detail')
    .argument('<taskId>', 'Session task ID')
    .option('--json', 'Output as JSON')
    .action(async (taskId, opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const session = await client.getSession(taskId);
      const stats = await client.getStats(taskId);

      if (opts.json) {
        console.log(JSON.stringify({ session, stats }, null, 2));
        return;
      }

      const output = displayDetail(session);
      console.log(output);
    });

  return cmd;
}
