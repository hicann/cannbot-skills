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
import { formatTokens, formatDate } from '../utils/format';
import { formatHeader, formatDivider, formatSuccess, formatWarning, theme } from '../utils/colors';
import type { ApiImportableSession, ApiImportableSessionsResponse } from '../types';
import readline from 'node:readline';

const IMPORT_COLUMNS: TableColumn[] = [
  { key: 'id', label: 'Session ID', width: 20 },
  { key: 'firstQuery', label: 'Query', width: 30 },
  { key: 'turnCount', label: 'Turns', width: 8 },
  { key: 'model', label: 'Model', width: 18 },
  { key: 'createdAt', label: 'Created', width: 16 },
];

function renderImportableRow(row: ApiImportableSession, key: string): string {
  switch (key) {
    case 'firstQuery': return row.firstQuery ?? '—';
    case 'model': return row.model ?? '—';
    case 'createdAt': return formatDate(row.createdAt);
    default: return String(row[key as keyof ApiImportableSession] ?? '—');
  }
}

async function confirmPrompt(message: string): Promise<boolean> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${message} [y/N]: `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

export function importCommand(): Command {
  const cmd = new Command('import');
  cmd
    .description('Import sessions from external sources')
    .option('--source <type>', 'Source type (opencode-db, claude-jsonl)', 'opencode-db')
    .option('--file <path>', 'File path to import')
    .option('--dir <path>', 'Directory to scan (recursive)')
    .option('--list', 'List importable sessions only')
    .option('--session-id <id>', 'Import specific session by ID')
    .option('--all', 'Import all importable sessions')
    .option('--yes', 'Skip confirmation prompt')
    .option('--json', 'Output as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      const source = opts.source;
      const filePath = opts.file || opts.dir || '';

      if (!filePath) {
        console.error(formatWarning('Error: --file or --dir is required'));
        process.exit(1);
      }

      const importable: ApiImportableSessionsResponse = await client.listImportableSessions(source, filePath);

      if (opts.list) {
        if (opts.json) {
          console.log(JSON.stringify(importable, null, 2));
          return;
        }

        const sessions = importable.sessions as ApiImportableSession[];
        if (sessions.length === 0) {
          console.log(theme.muted('No importable sessions found.'));
          return;
        }

        console.log(formatHeader(`Importable Sessions (${source})`));
        console.log(renderTable(
          IMPORT_COLUMNS,
          sessions as unknown as Record<string, unknown>[],
          renderImportableRow as unknown as (row: Record<string, unknown>, key: string) => string,
        ));
        console.log('');
        console.log(theme.muted(`Total: ${sessions.length} sessions available for import`));
        return;
      }

      const sessions = importable.sessions as ApiImportableSession[];
      let toImport: ApiImportableSession[];

      if (opts.sessionId) {
        const found = sessions.find(s => s.id === opts.sessionId);
        if (!found) {
          console.error(formatWarning(`Error: Session "${opts.sessionId}" not found in importable list`));
          process.exit(1);
        }
        toImport = [found];
      } else if (opts.all) {
        toImport = sessions;
      } else {
        console.error(formatWarning('Error: Specify --session-id <id> or --all to import'));
        process.exit(1);
      }

      if (toImport.length === 0) {
        console.log(theme.muted('No sessions to import.'));
        return;
      }

      if (!opts.json) {
        console.log(formatHeader(`Import Sessions (${source})`));
        console.log(formatDivider());
        for (const s of toImport) {
          console.log(`  ${s.id} — ${s.firstQuery ?? '—'} (${s.turnCount} turns, ${s.model ?? '—'})`);
        }
        console.log('');
      }

      if (!opts.yes) {
        const confirmed = await confirmPrompt(`Import ${toImport.length} session(s)?`);
        if (!confirmed) {
          console.log(theme.muted('Import cancelled.'));
          return;
        }
      }

      const results = [];
      for (const s of toImport) {
        const result = await client.importSession(source, filePath, s.id);
        results.push({ ...result, importSourceId: s.id });
      }

      if (opts.json) {
        console.log(JSON.stringify(results, null, 2));
        return;
      }

      for (const r of results) {
        console.log(formatSuccess(`✓ Imported session ${r.sessionId}`));
      }
      console.log('');
      console.log(theme.muted(`Imported ${results.length} session(s)`));
    });

  return cmd;
}
