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

// Computed lazily (not at module load) so tests can isolate HOME without
// nuking the real ~/.cannbot-insight (which holds the proxy/ capture dir).
function configDir(): string {
  return path.join(os.homedir(), BRAND_CONFIG_DIR_SUFFIX);
}
function configFilePath(): string {
  return path.join(configDir(), 'config.json');
}

export function loadConfig(globalOpts?: { server?: string; timeout?: string }): CliConfig {
  let config = { ...DEFAULT_CONFIG };

  if (fs.existsSync(configFilePath())) {
    try {
      const saved = JSON.parse(fs.readFileSync(configFilePath(), 'utf-8'));
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
  const dir = configDir();
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
  const current = loadConfig();
  const merged = { ...current, ...config };
  fs.writeFileSync(configFilePath(), JSON.stringify(merged, null, 2));
}

export function resetConfig(): void {
  if (fs.existsSync(configFilePath())) {
    try {
      fs.unlinkSync(configFilePath());
    } catch (e) {
      throw new ConfigError(`Failed to reset config: ${e instanceof Error ? e.message : String(e)}`);
    }
  }
}
