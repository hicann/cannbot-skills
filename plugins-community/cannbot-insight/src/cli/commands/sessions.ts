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
import { formatHeader, theme } from '../utils/colors';
import type { ApiSessionListItem } from '../types';

const SESSION_COLUMNS: TableColumn[] = [
  { key: 'taskId', label: 'Task ID', width: 20 },
  { key: 'model', label: 'Model', width: 18 },
  { key: 'user', label: 'User', width: 12 },
  { key: 'query', label: 'Query', width: 30 },
  { key: 'tokens', label: 'Tokens', width: 10 },
  { key: 'cost', label: 'Cost', width: 10 },
  { key: 'duration', label: 'Duration', width: 10 },
  { key: 'startTime', label: 'Start', width: 16 },
  { key: 'tools', label: 'Tools', width: 6 },
  { key: 'subs', label: 'Subs', width: 6 },
];

function renderSessionRow(row: ApiSessionListItem, key: string): string {
  switch (key) {
    case 'tokens': return formatTokens(row.totalTokens);
    case 'cost': return formatCost(row.totalCost);
    case 'duration': return formatDuration(row.totalLatencyMs);
    case 'startTime': return formatDate(row.startTime);
    case 'tools': return String(row.totalToolCallCount ?? 0);
    case 'subs': return String(row.totalSubagentCount ?? 0);
    case 'query': return row.query ?? '—';
    case 'model': return row.model ?? '—';
    case 'user': return row.user ?? '—';
    default: return String(row[key as keyof ApiSessionListItem] ?? '—');
  }
}

export function sessionsCommand(): Command {
  const cmd = new Command('sessions');
  cmd
    .description('List sessions')
    .option('--limit <number>', 'Items per page', '20')
    .option('--offset <number>', 'Offset (skip first N items)', '0')
    .option('--user <string>', 'Filter by user')
    .option('--json', 'Output as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const limit = +opts.limit;
      const offset = +opts.offset;
      const page = Math.floor(offset / limit) + 1;

      const response = await client.listSessions({
        page,
        pageSize: limit,
        user: opts.user,
      });

      if (opts.json) {
        console.log(JSON.stringify(response, null, 2));
        return;
      }

      const rows = response.items as ApiSessionListItem[];
      const table = renderTable(SESSION_COLUMNS, rows as unknown as Record<string, unknown>[], renderSessionRow as unknown as (row: Record<string, unknown>, key: string) => string);

      console.log(formatHeader('Sessions'));
      console.log(table);

      const totalTokens = rows.reduce((sum, r) => sum + (r.totalTokens ?? 0), 0);
      const totalCost = rows.reduce((sum, r) => sum + (r.totalCost ?? 0), 0);

      console.log('');
      console.log(theme.muted(`Total: ${response.total} sessions │ Page ${response.page} │ Tokens: ${formatTokens(totalTokens)} │ Cost: ${formatCost(totalCost)}`));
    });

  return cmd;
}
