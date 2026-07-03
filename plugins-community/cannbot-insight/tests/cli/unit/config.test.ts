// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { loadConfig, saveConfig, resetConfig, DEFAULT_CONFIG, DEFAULT_SERVER_URL } from '@/cli/config';
import { BRAND_CONFIG_DIR_SUFFIX } from '@/lib/branding';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const CONFIG_DIR = path.join(os.homedir(), BRAND_CONFIG_DIR_SUFFIX);
const CONFIG_FILE = path.join(CONFIG_DIR, 'config.json');

describe('config', () => {
  beforeEach(() => {
    resetConfig();
  });

  afterEach(() => {
    resetConfig();
  });

  describe('DEFAULT_CONFIG', () => {
    it('has correct default server URL', () => {
      expect(DEFAULT_CONFIG.server).toBe(DEFAULT_SERVER_URL);
    });

    it('has correct default timeout', () => {
      expect(DEFAULT_CONFIG.timeout).toBe(15000);
    });

    it('has default keybindings', () => {
      expect(DEFAULT_CONFIG.keybindings.quit).toBe('q');
      expect(DEFAULT_CONFIG.keybindings.help).toBe('?');
    });
  });

  describe('loadConfig', () => {
    it('returns defaults when no config file exists', () => {
      const config = loadConfig();
      expect(config.server).toBe(DEFAULT_SERVER_URL);
      expect(config.timeout).toBe(15000);
    });

    it('overrides defaults with command line opts', () => {
      const config = loadConfig({ server: 'http://custom:8080', timeout: '5000' });
      expect(config.server).toBe('http://custom:8080');
      expect(config.timeout).toBe(5000);
    });

    it('overrides defaults with environment variables', () => {
      process.env.CANNBOT_SERVER = 'http://env-server:9999';
      process.env.CANNBOT_TIMEOUT = '30000';
      const config = loadConfig();
      expect(config.server).toBe('http://env-server:9999');
      expect(config.timeout).toBe(30000);
      delete process.env.CANNBOT_SERVER;
      delete process.env.CANNBOT_TIMEOUT;
    });

    it('command line opts override environment variables', () => {
      process.env.CANNBOT_SERVER = 'http://env-server:9999';
      const config = loadConfig({ server: 'http://cli-server:8080' });
      expect(config.server).toBe('http://cli-server:8080');
      delete process.env.CANNBOT_SERVER;
    });

    it('loads saved config from file', () => {
      saveConfig({ server: 'http://saved:7777', theme: 'dark' });
      const config = loadConfig();
      expect(config.server).toBe('http://saved:7777');
      expect(config.theme).toBe('dark');
    });

    it('ignores invalid config file', () => {
      fs.mkdirSync(CONFIG_DIR, { recursive: true });
      fs.writeFileSync(CONFIG_FILE, 'not json');
      const config = loadConfig();
      expect(config.server).toBe(DEFAULT_SERVER_URL);
    });
  });

  describe('saveConfig', () => {
    it('creates config directory if not exists', () => {
      if (fs.existsSync(CONFIG_DIR)) {
        fs.rmSync(CONFIG_DIR, { recursive: true });
      }
      saveConfig({ server: 'http://new:1234' });
      expect(fs.existsSync(CONFIG_DIR)).toBe(true);
      expect(fs.existsSync(CONFIG_FILE)).toBe(true);
    });

    it('merges with existing config', () => {
      saveConfig({ server: 'http://first:1111' });
      saveConfig({ timeout: 5000 });
      const config = loadConfig();
      expect(config.server).toBe('http://first:1111');
      expect(config.timeout).toBe(5000);
    });
  });

  describe('resetConfig', () => {
    it('removes config file', () => {
      saveConfig({ server: 'http://test:1234' });
      expect(fs.existsSync(CONFIG_FILE)).toBe(true);
      resetConfig();
      expect(fs.existsSync(CONFIG_FILE)).toBe(false);
    });

    it('does nothing when config file does not exist', () => {
      resetConfig();
      expect(fs.existsSync(CONFIG_FILE)).toBe(false);
    });
  });
});
