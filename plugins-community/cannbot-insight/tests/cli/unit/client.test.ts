// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InsightClient } from '@/cli/client';
import { ApiError, NetworkError } from '@/cli/errors';

const mockFetch = vi.fn();
vi.stubGlobal('fetch', mockFetch);

describe('InsightClient', () => {
  describe('constructor', () => {
    it('uses default baseUrl when none provided', () => {
      const client = new InsightClient();
      expect(client.getConfig().baseUrl).toBe('http://localhost:21025');
    });

    it('accepts custom baseUrl', () => {
      const client = new InsightClient('http://custom:8080');
      expect(client.getConfig().baseUrl).toBe('http://custom:8080');
    });

    it('merges partial config with defaults', () => {
      const client = new InsightClient('http://localhost:21025', { timeout: 5000 });
      expect(client.getConfig().timeout).toBe(5000);
      expect(client.getConfig().retries).toBe(2);
      expect(client.getConfig().retryDelay).toBe(1000);
    });

    it('accepts authToken', () => {
      const client = new InsightClient('http://localhost:21025', { authToken: 'test-token' });
      expect(client.getConfig().authToken).toBe('test-token');
    });
  });

  describe('API methods', () => {
    let client: InsightClient;

    beforeEach(() => {
      client = new InsightClient('http://localhost:21025', { retries: 0, retryDelay: 10 });
      mockFetch.mockReset();
    });

    it('listSessions sends GET to /api/observe/data', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0, page: 1 }),
      });
      await client.listSessions({ page: 1, pageSize: 20 });
      expect(mockFetch).toHaveBeenCalledTimes(1);
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/data');
      expect(url).toContain('page=1');
      expect(url).toContain('pageSize=20');
    });

    it('getSession sends GET to /api/observe/session with taskId', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ sessionId: 's1', taskId: 't1' }),
      });
      await client.getSession('test-task');
      expect(mockFetch).toHaveBeenCalledTimes(1);
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session');
      expect(url).toContain('taskId=test-task');
    });

    it('getStats without taskId returns global stats', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ totalSessions: 42, totalTokens: 1000 }),
      });
      const result = await client.getStats();
      expect(result).toEqual({ totalSessions: 42, totalTokens: 1000 });
    });

    it('getStats with taskId returns session stats', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ taskId: 't1', totalTokens: 500 }),
      });
      const result = await client.getStats('t1');
      expect(result).toEqual({ taskId: 't1', totalTokens: 500 });
    });

    it('getTurns sends GET to /api/observe/session/turns', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0 }),
      });
      await client.getTurns('t1', { role: 'assistant' });
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session/turns');
      expect(url).toContain('taskId=t1');
      expect(url).toContain('role=assistant');
    });

    it('getTurnDetail sends GET to /api/observe/session/turns/{turnId}', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ turnId: 'turn1', role: 'assistant' }),
      });
      await client.getTurnDetail('turn1');
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session/turns/turn1');
    });

    it('searchTurns sends GET to /api/observe/session/turns/search', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0 }),
      });
      await client.searchTurns('t1', 'bug');
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session/turns/search');
      expect(url).toContain('taskId=t1');
      expect(url).toContain('keyword=bug');
    });

    it('getWorkflow sends GET to /api/observe/session/workflow', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ phases: [], summary: {} }),
      });
      await client.getWorkflow('t1');
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session/workflow');
    });

    it('getBridges sends GET to /api/observe/session/bridges', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ items: [], total: 0 }),
      });
      await client.getBridges('t1');
      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('/api/observe/session/bridges');
    });

    it('importSession sends POST to /api/ingest/import-file', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ sessionId: 's1', imported: true }),
      });
      await client.importSession('opencode-db', '/path/to/db', 's1');
      expect(mockFetch).toHaveBeenCalledTimes(1);
      const call = mockFetch.mock.calls[0][1];
      expect(call.method).toBe('POST');
      expect(JSON.parse(call.body)).toEqual({
        source: 'opencode-db',
        filePath: '/path/to/db',
        sessionId: 's1',
      });
    });

    it('listImportableSessions sends POST to /api/ingest/import-file/sessions', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ sessions: [] }),
      });
      await client.listImportableSessions('opencode-db', '/path/to/db');
      const call = mockFetch.mock.calls[0][1];
      expect(call.method).toBe('POST');
    });

    it('deleteSession sends DELETE to /api/ingest/delete-session', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ deleted: 1, taskId: 't1' }),
      });
      await client.deleteSession('t1');
      const call = mockFetch.mock.calls[0][1];
      expect(call.method).toBe('DELETE');
    });

    it('analyzeWorkflow sends POST to /api/ai/analyze-workflow', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ result: { phases: [], summary: {} } }),
      });
      await client.analyzeWorkflow('t1', { baseUrl: 'http://ai', apiKey: 'key', model: 'gpt-4' });
      const call = mockFetch.mock.calls[0][1];
      expect(call.method).toBe('POST');
    });

    it('testProvider sends POST to /api/ai/test-provider', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ success: true, message: 'OK' }),
      });
      await client.testProvider('http://ai', 'key');
      const call = mockFetch.mock.calls[0][1];
      expect(call.method).toBe('POST');
    });
  });

  describe('retry logic', () => {
    let client: InsightClient;

    beforeEach(() => {
      client = new InsightClient('http://localhost:21025', { retries: 2, retryDelay: 10 });
      mockFetch.mockReset();
    });

    it('retries on 5xx and succeeds on second attempt', async () => {
      mockFetch
        .mockResolvedValueOnce({
          ok: false, status: 500,
          text: async () => 'Internal Server Error',
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({ totalSessions: 42 }),
        });

      const result = await client.getStats();
      expect(result).toEqual({ totalSessions: 42 });
      expect(mockFetch).toHaveBeenCalledTimes(2);
    });

    it('throws after all retries exhausted on 5xx', async () => {
      mockFetch.mockResolvedValue({
        ok: false, status: 500,
        text: async () => 'Internal Server Error',
      });

      await expect(client.getStats()).rejects.toThrow(ApiError);
      expect(mockFetch).toHaveBeenCalledTimes(3);
    });

    it('does NOT retry on 4xx errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false, status: 400,
        text: async () => JSON.stringify({ error: 'Bad request' }),
      });

      await expect(client.listSessions({})).rejects.toThrow(ApiError);
      expect(mockFetch).toHaveBeenCalledTimes(1);
    });

    it('throws NetworkError on network failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('ECONNREFUSED'));

      await expect(client.getStats()).rejects.toThrow(NetworkError);
    });
  });

  describe('auth header', () => {
    it('includes Authorization header when authToken is set', async () => {
      const client = new InsightClient('http://localhost:21025', { authToken: 'my-token', retries: 0 });
      mockFetch.mockReset();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ totalSessions: 0 }),
      });

      await client.getStats();
      const call = mockFetch.mock.calls[0][1];
      expect(call.headers.Authorization).toBe('Bearer my-token');
    });

    it('does not include Authorization header when authToken is absent', async () => {
      const client = new InsightClient('http://localhost:21025', { retries: 0 });
      mockFetch.mockReset();
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: async () => ({ totalSessions: 0 }),
      });

      await client.getStats();
      const call = mockFetch.mock.calls[0][1];
      expect(call.headers.Authorization).toBeUndefined();
    });
  });
});
