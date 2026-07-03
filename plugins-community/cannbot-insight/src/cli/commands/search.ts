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
import { formatDate } from '../utils/format';
import { formatHeader, theme } from '../utils/colors';
import type { ApiSearchResult } from '../types';

const SEARCH_COLUMNS: TableColumn[] = [
  { key: 'turnId', label: 'Turn ID', width: 15 },
  { key: 'role', label: 'Role', width: 10 },
  { key: 'agent', label: 'Agent', width: 15 },
  { key: 'content', label: 'Content', width: 40 },
  { key: 'field', label: 'Match', width: 10 },
  { key: 'time', label: 'Time', width: 16 },
];

function renderSearchRow(row: ApiSearchResult, key: string): string {
  switch (key) {
    case 'content': return row.matchContext ?? row.contentSummary ?? '—';
    case 'field': return row.matchField ?? '—';
    case 'agent': return row.agentName ?? (row.isSubagent ? row.subagentName ?? '—' : '—');
    case 'time': return formatDate(row.createdAt);
    default: return String(row[key as keyof ApiSearchResult] ?? '—');
  }
}

export function searchCommand(): Command {
  const cmd = new Command('search');
  cmd
    .description('Search turns in a session')
    .argument('<taskId>', 'Session task ID')
    .option('--keyword <string>', 'Search keyword')
    .option('--limit <number>', 'Max results to show', '50')
    .option('--json', 'Output as JSON')
    .action(async (taskId, opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      if (!opts.keyword) {
        console.error('Error: --keyword is required');
        process.exit(1);
      }

      const response = await client.searchTurns(taskId, opts.keyword);

      if (opts.json) {
        console.log(JSON.stringify(response, null, 2));
        return;
      }

      const limit = +opts.limit;
      const items = response.items.slice(0, limit) as ApiSearchResult[];
      const table = renderTable(SEARCH_COLUMNS, items as unknown as Record<string, unknown>[], renderSearchRow as unknown as (row: Record<string, unknown>, key: string) => string);

      console.log(formatHeader(`Search: "${opts.keyword}" in ${taskId}`));
      console.log(table);
      console.log('');
      console.log(theme.muted(`Found ${response.total} results (showing ${items.length})`));
    });

  return cmd;
}
