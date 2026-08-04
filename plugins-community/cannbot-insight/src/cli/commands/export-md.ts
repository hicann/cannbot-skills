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
import path from 'node:path';

export function exportMdCommand(): Command {
  const cmd = new Command('export-md');
  cmd
    .description('Export session to a Markdown file (same output as the web UI export & audit step 1)')
    .option('--session <taskId>', 'Session taskId to export')
    .option('--framework <name>', 'Source framework: claude-code | opencode')
    .option('--output <path>', 'Output file path (default: ./session_<taskId>.md)')
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
      const framework = opts.framework as string | undefined;
      const outputPath = opts.output ?? path.join(process.cwd(), `session_${taskId}.md`);

      if (!opts.json) {
        console.log(formatHeader(`Export Markdown: ${taskId}`));
        console.log(formatDivider());
      }

      const result = await client.exportSessionMarkdown(taskId, outputPath, framework);

      if (opts.json) {
        console.log(JSON.stringify({ taskId, framework: framework ?? null, filePath: outputPath, size: result.size }, null, 2));
        return;
      }

      console.log(formatSuccess(`✓ Exported markdown ${taskId} → ${outputPath}`));
      console.log(theme.muted(`  File size: ${result.size} bytes`));
    });

  return cmd;
}
