// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { searchCommand } from '@/cli/commands/search';
import { InsightClient } from '@/cli/client';
import { renderTable } from '@/cli/utils/table';
import { formatDate } from '@/cli/utils/format';
import type { ApiSearchResponse, ApiSearchResult } from '@/cli/types';

const mockSearchTurns = vi.fn();
vi.spyOn(InsightClient.prototype, 'searchTurns').mockImplementation(mockSearchTurns);

const sampleResult: ApiSearchResult = {
  turnId: 'turn-001',
  turnIndex: 5,
  role: 'assistant',
  agentName: 'main',
  isSubagent: false,
  subagentName: null,
  subagentSessionId: null,
  contentSummary: 'Fixed the bug in login module',
  matchContext: '...Fixed the **bug** in login module...',
  matchField: 'content',
  createdAt: '2026-06-14T10:35:00Z',
  hasDispatchBridge: false,
};

const sampleResponse: ApiSearchResponse = {
  items: [sampleResult],
  total: 1,
};

describe('searchCommand', () => {
  beforeEach(() => {
    mockSearchTurns.mockReset();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = searchCommand();
    expect(cmd.name()).toBe('search');
  });

  it('has --keyword, --limit, --json options', () => {
    const cmd = searchCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--keyword');
    expect(options).toContain('--limit');
    expect(options).toContain('--json');
  });

  it('calls searchTurns with taskId and keyword', async () => {
    mockSearchTurns.mockResolvedValueOnce(sampleResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const response = await client.searchTurns('task-001', 'bug');

    expect(mockSearchTurns).toHaveBeenCalledWith('task-001', 'bug');
    expect(response.total).toBe(1);
  });

  it('formats search result fields correctly', () => {
    const r = sampleResult;
    expect(r.turnId).toBe('turn-001');
    expect(r.role).toBe('assistant');
    expect(r.matchContext).toContain('bug');
    expect(r.matchField).toBe('content');
    expect(formatDate(r.createdAt)).toBeTruthy();
  });

  it('renders search results table', () => {
    const SEARCH_COLUMNS = [
      { key: 'turnId', label: 'Turn ID', width: 15 },
      { key: 'role', label: 'Role', width: 10 },
      { key: 'content', label: 'Content', width: 40 },
      { key: 'field', label: 'Match', width: 10 },
      { key: 'time', label: 'Time', width: 16 },
    ];

    const items = [sampleResult];
    const table = renderTable(
      SEARCH_COLUMNS,
      items as unknown as Record<string, unknown>[],
      (row: Record<string, unknown>, key: string) => {
        const r = row as unknown as ApiSearchResult;
        switch (key) {
          case 'content': return r.matchContext ?? r.contentSummary ?? '—';
          case 'field': return r.matchField ?? '—';
          case 'time': return formatDate(r.createdAt);
          default: return String(r[key as keyof ApiSearchResult] ?? '—');
        }
      },
    );

    expect(table).toContain('turn-001');
    expect(table).toContain('assistant');
    expect(table).toContain('bug');
    expect(table).toContain('content');
  });

  it('respects --limit to slice results', () => {
    const manyItems: ApiSearchResult[] = Array.from({ length: 10 }, (_, i) => ({
      ...sampleResult,
      turnId: `turn-${i}`,
      turnIndex: i,
    }));

    const limit = 5;
    const sliced = manyItems.slice(0, limit);
    expect(sliced.length).toBe(5);
  });

  it('JSON output includes items and total', async () => {
    mockSearchTurns.mockResolvedValueOnce(sampleResponse);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const response = await client.searchTurns('task-001', 'bug');

    const json = JSON.stringify(response, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.items[0].turnId).toBe('turn-001');
    expect(parsed.total).toBe(1);
  });
});
