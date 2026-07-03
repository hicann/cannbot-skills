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
import fs from 'node:fs';
import path from 'node:path';

export function exportCommand(): Command {
  const cmd = new Command('export');
  cmd
    .description('Export session to a standalone SQLite database file')
    .option('--session <taskId>', 'Session taskId to export')
    .option('--output <path>', 'Output file path (default: ./cannbot_session_<taskId>.db)')
    .option('--json', 'Output result as JSON')
    .action(async (opts, command) => {
      const globalOpts = command.parent?.opts() ?? {};
      const client = new InsightClient(globalOpts.server, {
        timeout: +globalOpts.timeout,
      });

      if (!opts.session) {
        console.error(formatWarning('Error: --session <taskId> is required'));
        process.exit(1);
      }

      const taskId = opts.session;
      const outputPath = opts.output ?? path.join(process.cwd(), `cannbot_session_${taskId}.db`);

      if (!opts.json) {
        console.log(formatHeader(`Export Session: ${taskId}`));
        console.log(formatDivider());
      }

      const result = await client.exportSession(taskId, outputPath);

      if (opts.json) {
        console.log(JSON.stringify({ taskId, filePath: outputPath, size: result.size }, null, 2));
        return;
      }

      console.log(formatSuccess(`✓ Exported session ${taskId} → ${outputPath}`));
      console.log(theme.muted(`  File size: ${result.size} bytes`));
    });

  return cmd;
}
