// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Command } from 'commander';
import { loadConfig, saveConfig, resetConfig, DEFAULT_CONFIG } from '../config';
import { formatHeader, formatLabel, formatDivider, formatSuccess, formatWarning, theme } from '../utils/colors';
import { getAllContextWindows, getDefaultContextWindow } from '@/lib/context-window-config';
import { formatTokens } from '../utils/format';

export function configCommand(): Command {
  const cmd = new Command('config');
  cmd.description('Manage CLI configuration');

  cmd
    .command('set <key> <value>')
    .description('Set a configuration value')
    .action((key, value) => {
      const validKeys = Object.keys(DEFAULT_CONFIG);
      if (!validKeys.includes(key)) {
        console.error(formatWarning(`Error: Unknown config key "${key}". Valid keys: ${validKeys.join(', ')}`));
        process.exit(1);
      }

      let parsedValue: string | number | Record<string, string> = value;
      if (key === 'timeout') {
        parsedValue = +value;
        if (isNaN(parsedValue) || parsedValue <= 0) {
          console.error(formatWarning('Error: timeout must be a positive number'));
          process.exit(1);
        }
      }

      saveConfig({ [key]: parsedValue });
      console.log(formatSuccess(`✓ Set ${key} = ${parsedValue}`));
    });

  cmd
    .command('get <key>')
    .description('Get a configuration value')
    .action((key) => {
      const config = loadConfig();
      const value = config[key as keyof typeof config];
      if (value === undefined) {
        console.error(formatWarning(`Error: Unknown config key "${key}"`));
        process.exit(1);
      }
      console.log(JSON.stringify(value));
    });

  cmd
    .command('list')
    .description('List all configuration values')
    .action(() => {
      const config = loadConfig();
      console.log(formatHeader('CLI Configuration'));
      console.log(formatDivider());
      console.log('');
      for (const [key, value] of Object.entries(config)) {
        if (typeof value === 'object') {
          console.log(formatLabel(key, '{...}'));
          for (const [subKey, subValue] of Object.entries(value as Record<string, string>)) {
            console.log(`  ${theme.muted(subKey)}: ${subValue}`);
          }
        } else {
          console.log(formatLabel(key, String(value)));
        }
      }
    });

  cmd
    .command('context-windows')
    .description('View context window configuration for models')
    .action(() => {
      const allWindows = getAllContextWindows();
      const defaultWindow = getDefaultContextWindow();

      console.log(formatHeader('Context Window Configuration'));
      console.log(formatDivider());
      console.log('');
      console.log(formatLabel('Default Window', formatTokens(defaultWindow)));
      console.log('');

      for (const [model, window] of Object.entries(allWindows)) {
        console.log(formatLabel(model, formatTokens(window)));
      }
    });

  cmd
    .command('reset')
    .description('Reset configuration to defaults')
    .action(() => {
      resetConfig();
      console.log(formatSuccess('✓ Configuration reset to defaults'));
    });

  return cmd;
}
