// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { VERSION } from '@/lib/version';
import { BRAND_SLUG, BRAND_CLI_ALIAS, BRAND_DESCRIPTION } from '@/lib/branding';
import { DEFAULT_SERVER_URL, loadConfig } from './config';
import { runTui } from './tui/App';
import { sessionsCommand } from './commands/sessions';
import { sessionCommand } from './commands/session';
import { statsCommand } from './commands/stats';
import { compareCommand } from './commands/compare';
import { searchCommand } from './commands/search';
import { turnCommand } from './commands/turn';
import { importCommand } from './commands/import';
import { deleteCommand } from './commands/delete';
import { configCommand } from './commands/config';
import { analyzeCommand } from './commands/analyze';
import { exportCommand } from './commands/export';
import { uploadCommand } from './commands/upload';
import { InsightError } from './errors';

export function createProgram(): Command {
  const program = new Command();
  program
    .name(BRAND_SLUG)
    .alias(BRAND_CLI_ALIAS)
    .description(BRAND_DESCRIPTION)
    .version(VERSION)
    .option('--server <url>', 'Backend server URL', DEFAULT_SERVER_URL)
    .option('--timeout <ms>', 'Request timeout in ms', '15000');

  program
    .command('tui')
    .description('Launch interactive TUI mode')
    .action(async (opts, cmd) => {
      const globalOpts = cmd.parent?.opts() ?? {};
      await runTui(loadConfig(globalOpts));
    });

  program.addCommand(sessionsCommand());
  program.addCommand(sessionCommand());
  program.addCommand(statsCommand());
  program.addCommand(compareCommand());
  program.addCommand(searchCommand());
  program.addCommand(turnCommand());
  program.addCommand(importCommand());
  program.addCommand(deleteCommand());
  program.addCommand(configCommand());
  program.addCommand(analyzeCommand());
  program.addCommand(exportCommand());
  program.addCommand(uploadCommand());

  return program;
}

const program = createProgram();
program.parseAsync(process.argv).catch((err: Error) => {
  if (err instanceof InsightError) {
    console.error(`\n❌ ${err.message}`);
  } else {
    console.error(err);
  }
  process.exit(1);
});
