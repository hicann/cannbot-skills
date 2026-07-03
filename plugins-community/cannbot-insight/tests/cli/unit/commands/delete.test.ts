// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { deleteCommand } from '@/cli/commands/delete';
import { InsightClient } from '@/cli/client';
import type { ApiDeleteResponse } from '@/cli/types';

const mockDeleteSession = vi.fn();
vi.spyOn(InsightClient.prototype, 'deleteSession').mockImplementation(mockDeleteSession);

const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

const sampleDeleteResponse: ApiDeleteResponse = {
  deleted: 1,
  taskId: 'task-001',
};

const sampleDeleteAllResponse: ApiDeleteResponse = {
  deleted: 10,
};

describe('deleteCommand', () => {
  beforeEach(() => {
    mockDeleteSession.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = deleteCommand();
    expect(cmd.name()).toBe('delete');
    expect(cmd.description()).toContain('Delete sessions');
  });

  it('has --session, --all, --yes, --json options', () => {
    const cmd = deleteCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--session');
    expect(options).toContain('--all');
    expect(options).toContain('--yes');
    expect(options).toContain('--json');
  });

  it('deleteSession called with taskId', async () => {
    mockDeleteSession.mockResolvedValueOnce(sampleDeleteResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const result = await client.deleteSession('task-001');

    expect(mockDeleteSession).toHaveBeenCalledWith('task-001');
    expect(result.deleted).toBe(1);
    expect(result.taskId).toBe('task-001');
  });

  it('deleteSession called with deleteAll', async () => {
    mockDeleteSession.mockResolvedValueOnce(sampleDeleteAllResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const result = await client.deleteSession(undefined, true);

    expect(mockDeleteSession).toHaveBeenCalledWith(undefined, true);
    expect(result.deleted).toBe(10);
  });

  it('JSON output for single session delete', () => {
    const json = JSON.stringify(sampleDeleteResponse, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.deleted).toBe(1);
    expect(parsed.taskId).toBe('task-001');
  });

  it('JSON output for delete all', () => {
    const json = JSON.stringify(sampleDeleteAllResponse, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.deleted).toBe(10);
  });

  it('requires either --session or --all', () => {
    const cmd = deleteCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--session');
    expect(options).toContain('--all');
  });
});
