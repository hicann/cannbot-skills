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
import { formatTokens, formatCost, formatDuration } from '../utils/format';
import { formatHeader, formatDivider } from '../utils/colors';
import type { ApiSessionDetailResponse } from '../types';

const COMPARE_COLUMNS: TableColumn[] = [
  { key: 'metric', label: 'Metric', width: 20 },
  { key: 'session1', label: 'Session A', width: 25 },
  { key: 'session2', label: 'Session B', width: 25 },
];

interface CompareRow {
  metric: string;
  session1: string;
  session2: string;
}

function buildCompareRows(s1: ApiSessionDetailResponse, s2: ApiSessionDetailResponse): CompareRow[] {
  return [
    { metric: 'Task ID', session1: s1.taskId, session2: s2.taskId },
    { metric: 'Model', session1: s1.model ?? '—', session2: s2.model ?? '—' },
    { metric: 'User', session1: s1.user ?? '—', session2: s2.user ?? '—' },
    { metric: 'Query', session1: s1.query ?? '—', session2: s2.query ?? '—' },
    { metric: 'Tokens', session1: formatTokens(s1.totalTokens), session2: formatTokens(s2.totalTokens) },
    { metric: 'Input Tokens', session1: formatTokens(s1.totalInputTokens), session2: formatTokens(s2.totalInputTokens) },
    { metric: 'Output Tokens', session1: formatTokens(s1.totalOutputTokens), session2: formatTokens(s2.totalOutputTokens) },
    { metric: 'Cost', session1: formatCost(s1.totalCost), session2: formatCost(s2.totalCost) },
    { metric: 'Duration', session1: formatDuration(s1.totalLatencyMs), session2: formatDuration(s2.totalLatencyMs) },
    { metric: 'Tool Calls', session1: String(s1.totalToolCallCount ?? 0), session2: String(s2.totalToolCallCount ?? 0) },
    { metric: 'LLM Calls', session1: String(s1.totalLlmCallCount ?? 0), session2: String(s2.totalLlmCallCount ?? 0) },
    { metric: 'Subagents', session1: String(s1.totalSubagentCount ?? 0), session2: String(s2.totalSubagentCount ?? 0) },
    { metric: 'Skill Loads', session1: String(s1.totalSkillLoadCount ?? 0), session2: String(s2.totalSkillLoadCount ?? 0) },
  ];
}

export function compareCommand(): Command {
  const cmd = new Command('compare');
  cmd
    .description('Compare two sessions')
    .argument('<taskId1>', 'First session task ID')
    .argument('<taskId2>', 'Second session task ID')
    .option('--json', 'Output as JSON')
    .action(async (taskId1, taskId2, opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const [s1, s2] = await Promise.all([
        client.getSession(taskId1),
        client.getSession(taskId2),
      ]);

      if (opts.json) {
        console.log(JSON.stringify({ session1: s1, session2: s2 }, null, 2));
        return;
      }

      const rows = buildCompareRows(s1, s2);
      const table = renderTable(COMPARE_COLUMNS, rows as unknown as Record<string, unknown>[]);

      console.log(formatHeader(`Compare: ${taskId1} vs ${taskId2}`));
      console.log(formatDivider());
      console.log(table);
    });

  return cmd;
}
