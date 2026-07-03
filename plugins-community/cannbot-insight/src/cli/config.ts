// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { BRAND_CONFIG_DIR_SUFFIX } from '@/lib/branding';
import { ConfigError } from './errors';

export const DEFAULT_SERVER_URL = 'http://localhost:21025';

export interface CliConfig {
  server: string;
  timeout: number;
  theme: 'dark' | 'light' | 'auto';
  keybindings: Record<string, string>;
}

export const DEFAULT_CONFIG: CliConfig = {
  server: DEFAULT_SERVER_URL,
  timeout: 15000,
  theme: 'auto',
  keybindings: {
    quit: 'q',
    help: '?',
    search: '/',
    refresh: 'r',
    navigateUp: 'k',
    navigateDown: 'j',
    enter: 'Enter',
    tabSwitch: 'Tab',
  },
};

const CONFIG_DIR = path.join(os.homedir(), BRAND_CONFIG_DIR_SUFFIX);
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

export function loadConfig(globalOpts?: { server?: string; timeout?: string }): CliConfig {
  let config = { ...DEFAULT_CONFIG };

  if (fs.existsSync(CONFIG_FILE)) {
    try {
      const saved = JSON.parse(fs.readFileSync(CONFIG_FILE, 'utf-8'));
      config = { ...config, ...saved };
    } catch { /* ignore invalid config */ }
  }

  if (process.env.CANNBOT_SERVER) {
    config.server = process.env.CANNBOT_SERVER;
  }

  if (globalOpts?.server) {
    config.server = globalOpts.server;
  }

  if (process.env.CANNBOT_TIMEOUT) {
    config.timeout = +process.env.CANNBOT_TIMEOUT;
  }

  if (globalOpts?.timeout) {
    config.timeout = +globalOpts.timeout;
  }

  return config;
}

export function saveConfig(config: Partial<CliConfig>): void {
  if (!fs.existsSync(CONFIG_DIR)) {
    fs.mkdirSync(CONFIG_DIR, { recursive: true });
  }
  const current = loadConfig();
  const merged = { ...current, ...config };
  fs.writeFileSync(CONFIG_FILE, JSON.stringify(merged, null, 2));
}

export function resetConfig(): void {
  if (fs.existsSync(CONFIG_FILE)) {
    try {
      fs.unlinkSync(CONFIG_FILE);
    } catch (e) {
      throw new ConfigError(`Failed to reset config: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
}
