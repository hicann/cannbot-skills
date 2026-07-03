// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { InsightClient } from '../client';
import { formatHeader, formatDivider, formatSuccess, formatWarning, theme } from '../utils/colors';
import readline from 'node:readline';

async function confirmPrompt(message: string): Promise<boolean> {
  const rl = readline.createInterface({ input: process.stdin, output: process.stdout });
  return new Promise((resolve) => {
    rl.question(`${message} [y/N]: `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

export function deleteCommand(): Command {
  const cmd = new Command('delete');
  cmd
    .description('Delete sessions')
    .option('--session <taskId>', 'Delete specific session')
    .option('--all', 'Delete all sessions')
    .option('--yes', 'Skip confirmation prompt')
    .option('--json', 'Output as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      if (!opts.session && !opts.all) {
        console.error(formatWarning('Error: Specify --session <taskId> or --all'));
        process.exit(1);
      }

      if (opts.all) {
        if (!opts.json) {
          console.log(formatHeader('Delete All Sessions'));
          console.log(formatDivider());
          console.log(formatWarning('⚠ This will delete ALL sessions!'));
        }

        if (!opts.yes) {
          const confirmed = await confirmPrompt('Delete all sessions? This cannot be undone.');
          if (!confirmed) {
            console.log(theme.muted('Delete cancelled.'));
            return;
          }
        }

        const result = await client.deleteSession(undefined, true);

        if (opts.json) {
          console.log(JSON.stringify(result, null, 2));
          return;
        }

        console.log(formatSuccess(`✓ Deleted ${result.deleted} session(s)`));
        return;
      }

      const taskId = opts.session;

      if (!opts.json) {
        console.log(formatHeader(`Delete Session: ${taskId}`));
        console.log(formatDivider());
      }

      if (!opts.yes) {
        const confirmed = await confirmPrompt(`Delete session "${taskId}"?`);
        if (!confirmed) {
          console.log(theme.muted('Delete cancelled.'));
          return;
        }
      }

      const result = await client.deleteSession(taskId);

      if (opts.json) {
        console.log(JSON.stringify(result, null, 2));
        return;
      }

      console.log(formatSuccess(`✓ Deleted session ${taskId} (${result.deleted} record(s))`));
    });

  return cmd;
}
