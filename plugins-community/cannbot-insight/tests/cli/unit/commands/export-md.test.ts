// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { InsightClient } from '@/cli/client';
import { exportMdCommand } from '@/cli/commands/export-md';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

const mockFetch = vi.fn();
const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
const mockProcessExit = vi.spyOn(process, 'exit').mockImplementation((() => {}) as never);

const MARKDOWN = '# Session\n\n## §1 Assistant\n\nhello world\n';

describe('exportMdCommand', () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
    mockProcessExit.mockClear();
    vi.stubGlobal('fetch', mockFetch);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = exportMdCommand();
    expect(cmd.name()).toBe('export-md');
    expect(cmd.description()).toContain('Markdown');
  });

  it('has --session, --framework, --output, --json options', () => {
    const cmd = exportMdCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--session');
    expect(options).toContain('--framework');
    expect(options).toContain('--output');
    expect(options).toContain('--json');
  });

  it('exportSessionMarkdown GETs export-md endpoint with taskId + framework and writes the file', async () => {
    let fetchedUrl = '';
    mockFetch.mockImplementationOnce((url: string) => {
      fetchedUrl = url;
      return Promise.resolve({ ok: true, text: () => Promise.resolve(MARKDOWN) });
    });

    const tmp = path.join(os.tmpdir(), `cbi-export-md-${Date.now()}.md`);
    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const result = await client.exportSessionMarkdown('f3bb027a-xyz', tmp, 'claude-code');

    // URL hits the same endpoint the web UI uses, with taskId + framework params
    expect(fetchedUrl).toContain('/api/observe/session/export-md');
    expect(fetchedUrl).toContain('taskId=f3bb027a-xyz');
    expect(fetchedUrl).toContain('framework=claude-code');

    // File written byte-identical to the endpoint response
    expect(fs.existsSync(tmp)).toBe(true);
    expect(fs.readFileSync(tmp, 'utf-8')).toBe(MARKDOWN);
    expect(result.size).toBe(Buffer.byteLength(MARKDOWN, 'utf-8'));

    fs.unlinkSync(tmp);
  });

  it('omits framework param when not provided', async () => {
    let fetchedUrl = '';
    mockFetch.mockImplementationOnce((url: string) => {
      fetchedUrl = url;
      return Promise.resolve({ ok: true, text: () => Promise.resolve(MARKDOWN) });
    });

    const tmp = path.join(os.tmpdir(), `cbi-export-md-nofw-${Date.now()}.md`);
    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    await client.exportSessionMarkdown('sess-1', tmp);

    expect(fetchedUrl).toContain('taskId=sess-1');
    expect(fetchedUrl).not.toContain('framework=');
    fs.unlinkSync(tmp);
  });

  it('throws ApiError on non-OK response', async () => {
    mockFetch.mockImplementationOnce(() =>
      Promise.resolve({ ok: false, status: 500, text: () => Promise.resolve(JSON.stringify({ error: 'boom' })) }),
    );

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    await expect(client.exportSessionMarkdown('sess-1', '/tmp/unused.md')).rejects.toThrow('boom');
  });

  it('JSON output shape for --json mode', () => {
    const json = JSON.stringify({ taskId: 'sess-1', framework: null, filePath: './session_sess-1.md', size: 42 }, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.taskId).toBe('sess-1');
    expect(parsed.size).toBe(42);
  });
});
