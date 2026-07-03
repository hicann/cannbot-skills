// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { InsightClient } from '../client';
import { formatTokens, formatCost, formatDuration } from '../utils/format';
import { formatHeader, formatLabel, formatDivider } from '../utils/colors';
import type { ApiGlobalStatsResponse } from '../types';

function displayGlobalStats(stats: ApiGlobalStatsResponse): string {
  const lines: string[] = [];

  lines.push(formatHeader('Global Statistics'));
  lines.push(formatDivider());
  lines.push('');

  lines.push(formatLabel('Total Sessions', String(stats.totalSessions)));
  lines.push(formatLabel('Total Tokens', formatTokens(stats.totalTokens)));
  lines.push(formatLabel('Total Cost', formatCost(stats.totalCost)));
  lines.push(formatLabel('Total Duration', formatDuration(stats.totalLatencyMs)));
  lines.push(formatLabel('Avg Duration', formatDuration(stats.avgLatencyMs)));
  lines.push('');

  if (stats.totalSessions > 0) {
    const avgTokens = stats.totalTokens / stats.totalSessions;
    const avgCost = stats.totalCost / stats.totalSessions;
    lines.push(formatLabel('Avg Tokens/Session', formatTokens(avgTokens)));
    lines.push(formatLabel('Avg Cost/Session', formatCost(avgCost)));
  }

  return lines.join('\n');
}

export function statsCommand(): Command {
  const cmd = new Command('stats');
  cmd
    .description('Show global or session statistics')
    .option('--session <taskId>', 'Session task ID')
    .option('--json', 'Output as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const stats = await client.getStats(opts.session);

      if (opts.json) {
        console.log(JSON.stringify(stats, null, 2));
        return;
      }

      if (opts.session) {
        const s = stats as import('../types').ApiSessionStatsResponse;
        const lines: string[] = [];
        lines.push(formatHeader(`Session Stats: ${s.taskId}`));
        lines.push(formatDivider());
        lines.push('');
        lines.push(formatLabel('Total Tokens', formatTokens(s.totalTokens)));
        lines.push(formatLabel('Input Tokens', formatTokens(s.totalInputTokens)));
        lines.push(formatLabel('Output Tokens', formatTokens(s.totalOutputTokens)));
        lines.push(formatLabel('Reasoning Tokens', formatTokens(s.totalReasoningTokens)));
        lines.push(formatLabel('Cache Read', formatTokens(s.totalCacheReadTokens)));
        lines.push(formatLabel('Cache Write', formatTokens(s.totalCacheWriteTokens)));
        lines.push(formatLabel('Total Cost', formatCost(s.totalCost)));
        lines.push(formatLabel('Duration', formatDuration(s.totalLatencyMs)));
        lines.push(formatLabel('Tool Calls', String(s.totalToolCallCount ?? 0)));
        lines.push(formatLabel('LLM Calls', String(s.totalLlmCallCount ?? 0)));
        lines.push(formatLabel('Skill Loads', String(s.totalSkillLoadCount ?? 0)));
        lines.push(formatLabel('Subagents', String(s.totalSubagentCount ?? 0)));
        console.log(lines.join('\n'));
      } else {
        console.log(displayGlobalStats(stats as ApiGlobalStatsResponse));
      }
    });

  return cmd;
}
