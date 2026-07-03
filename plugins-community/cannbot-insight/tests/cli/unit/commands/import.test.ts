// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { importCommand } from '@/cli/commands/import';
import { InsightClient } from '@/cli/client';
import { renderTable } from '@/cli/utils/table';
import type { ApiImportableSession, ApiImportableSessionsResponse, ApiImportResponse } from '@/cli/types';

const mockListImportableSessions = vi.fn();
const mockImportSession = vi.fn();
vi.spyOn(InsightClient.prototype, 'listImportableSessions').mockImplementation(mockListImportableSessions);
vi.spyOn(InsightClient.prototype, 'importSession').mockImplementation(mockImportSession);

const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

const sampleImportableSession: ApiImportableSession = {
  id: 'sess-001',
  createdAt: '2026-06-14T10:00:00Z',
  firstQuery: 'Fix the login bug',
  turnCount: 15,
  model: 'claude-3.5-sonnet',
};

const sampleImportableResponse: ApiImportableSessionsResponse = {
  sessions: [sampleImportableSession],
};

const sampleImportResult: ApiImportResponse = {
  sessionId: 'sess-001',
  imported: 1,
};

describe('importCommand', () => {
  beforeEach(() => {
    mockListImportableSessions.mockReset();
    mockImportSession.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = importCommand();
    expect(cmd.name()).toBe('import');
    expect(cmd.description()).toContain('Import sessions');
  });

  it('has --source, --file, --dir, --list, --session-id, --all, --yes, --json options', () => {
    const cmd = importCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--source');
    expect(options).toContain('--file');
    expect(options).toContain('--dir');
    expect(options).toContain('--list');
    expect(options).toContain('--session-id');
    expect(options).toContain('--all');
    expect(options).toContain('--yes');
    expect(options).toContain('--json');
  });

  it('formats importable sessions list', () => {
    const IMPORT_COLUMNS = [
      { key: 'id', label: 'Session ID', width: 20 },
      { key: 'firstQuery', label: 'Query', width: 30 },
      { key: 'turnCount', label: 'Turns', width: 8 },
      { key: 'model', label: 'Model', width: 18 },
    ];

    const table = renderTable(
      IMPORT_COLUMNS,
      [sampleImportableSession] as unknown as Record<string, unknown>[],
    );

    expect(table).toContain('sess-001');
    expect(table).toContain('15');
  });

  it('listImportableSessions called with correct args', async () => {
    mockListImportableSessions.mockResolvedValueOnce(sampleImportableResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const result = await client.listImportableSessions('opencode-db', '/path/to/db');

    expect(mockListImportableSessions).toHaveBeenCalledWith('opencode-db', '/path/to/db');
    expect(result.sessions.length).toBe(1);
    expect(result.sessions[0].id).toBe('sess-001');
  });

  it('importSession called with correct args', async () => {
    mockImportSession.mockResolvedValueOnce(sampleImportResult);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const result = await client.importSession('opencode-db', '/path/to/db', 'sess-001');

    expect(mockImportSession).toHaveBeenCalledWith('opencode-db', '/path/to/db', 'sess-001');
    expect(result.imported).toBe(1);
  });

  it('JSON output for importable sessions', () => {
    const json = JSON.stringify(sampleImportableResponse, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.sessions.length).toBe(1);
    expect(parsed.sessions[0].id).toBe('sess-001');
    expect(parsed.sessions[0].turnCount).toBe(15);
  });

  it('JSON output for import results', () => {
    const json = JSON.stringify(sampleImportResult, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.sessionId).toBe('sess-001');
    expect(parsed.imported).toBe(1);
  });

  it('default source is opencode-db', () => {
    const cmd = importCommand();
    const sourceOption = cmd.options.find(o => o.long === '--source');
    expect(sourceOption?.defaultValue).toBe('opencode-db');
  });
});
