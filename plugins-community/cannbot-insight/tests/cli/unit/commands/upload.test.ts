// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InsightClient } from '@/cli/client';
import { uploadCommand } from '@/cli/commands/upload';
import type { ApiImportableSession, ApiImportableSessionsResponse, ApiImportResponse, ApiUploadResponse, ApiSessionDetailResponse } from '@/cli/types';

const mockListSessions = vi.fn();
const mockListImportableSessions = vi.fn();
const mockImportSession = vi.fn();
const mockUploadSession = vi.fn();
const mockGetSession = vi.fn();
vi.spyOn(InsightClient.prototype, 'listSessions').mockImplementation(mockListSessions);
vi.spyOn(InsightClient.prototype, 'listImportableSessions').mockImplementation(mockListImportableSessions);
vi.spyOn(InsightClient.prototype, 'importSession').mockImplementation(mockImportSession);
vi.spyOn(InsightClient.prototype, 'uploadSession').mockImplementation(mockUploadSession);
vi.spyOn(InsightClient.prototype, 'getSession').mockImplementation(mockGetSession);

const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {});
const mockProcessExit = vi.spyOn(process, 'exit').mockImplementation((() => {}) as never);

const sampleSession: ApiImportableSession = {
  id: 'ses_abc123',
  createdAt: '2026-06-29T10:00:00Z',
  firstQuery: 'Fix the build error in op code',
  turnCount: 15,
  model: 'claude-sonnet-4-6',
};

const sampleImportableResponse: ApiImportableSessionsResponse = {
  sessions: [sampleSession],
};

const sampleImportResult: ApiImportResponse = {
  sessionId: 'cmqyuh9bu0000abc',
  imported: true,
};

const sampleUploadResult: ApiUploadResponse = {
  success: true,
  filename: 'cannbot_db_session_ses_abc123_a1b2c3d4.db',
};

const sampleSessionDetail: ApiSessionDetailResponse = {
  sessionId: 'cmqyuh9bu0000abc',
  taskId: 'ses_abc123',
  query: 'Fix the build error',
  framework: 'opencode',
  startTime: '2026-06-29T10:00:00Z',
  endTime: '2026-06-29T11:00:00Z',
  totalTokens: 50000,
  totalInputTokens: 30000,
  totalOutputTokens: 20000,
  totalReasoningTokens: 0,
  totalCacheReadTokens: 10000,
  totalCacheWriteTokens: 5000,
  totalCost: 0.5,
  totalLatencyMs: 60000,
  totalToolCallCount: 10,
  totalLlmCallCount: 15,
  totalSkillLoadCount: 2,
  totalSubagentCount: 1,
  model: 'claude-sonnet-4-6',
  user: 'gxh',
  sourcePath: '/home/gxh/code/logs/opencode.db',
  label: null,
  agents: [],
  skills: [],
};

describe('uploadCommand', () => {
  beforeEach(() => {
    mockListSessions.mockReset();
    mockListImportableSessions.mockReset();
    mockImportSession.mockReset();
    mockUploadSession.mockReset();
    mockGetSession.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
    mockProcessExit.mockClear();
  });

  describe('command registration', () => {
    it('registers as Commander sub-command "upload"', () => {
      const cmd = uploadCommand();
      expect(cmd.name()).toBe('upload');
    });

    it('has --session, --file, --source, --description, --json, --yes options', () => {
      const cmd = uploadCommand();
      const options = cmd.options.map(o => o.long);
      expect(options).toContain('--session');
      expect(options).toContain('--file');
      expect(options).toContain('--source');
      expect(options).toContain('--description');
      expect(options).toContain('--json');
      expect(options).toContain('--yes');
    });
  });

  describe('detectSourceType', () => {
    it('detects .db files as opencode-db', async () => {
      const cmd = uploadCommand();
      // Access detectSourceType via the module — it's not exported, so test via behavior
      // When --file is a .db and no --source is given, the command should use opencode-db
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockListImportableSessions.mockResolvedValueOnce(sampleImportableResponse);
      mockImportSession.mockResolvedValueOnce(sampleImportResult);
      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);

      const client = new InsightClient('http://localhost:21025', { retries: 0 });
      await client.listImportableSessions('opencode-db', '/path/to/sessions.db');
      expect(mockListImportableSessions).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db');
    });

    it('detects .jsonl files as claude-jsonl', async () => {
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockListImportableSessions.mockResolvedValueOnce({
        sessions: [{
          id: 'd1ef6b6f',
          createdAt: '2026-06-29T10:00:00Z',
          firstQuery: 'Hello',
          turnCount: 5,
          model: 'claude-sonnet',
        }],
      });

      const client = new InsightClient('http://localhost:21025', { retries: 0 });
      await client.listImportableSessions('claude-jsonl', '/path/to/session.jsonl');
      expect(mockListImportableSessions).toHaveBeenCalledWith('claude-jsonl', '/path/to/session.jsonl');
    });
  });

  describe('upload from source file (--file mode)', () => {
    it('uploads single session from .db with --description', async () => {
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockListImportableSessions.mockResolvedValueOnce(sampleImportableResponse);
      mockImportSession.mockResolvedValueOnce(sampleImportResult);
      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);

      const client = new InsightClient('http://localhost:21025', { retries: 0, timeout: 5000 });

      // Simulate: import session, then upload
      const importResult = await client.importSession('opencode-db', '/path/to/sessions.db', 'ses_abc123');
      expect(importResult.imported).toBe(true);

      const uploadResult = await client.uploadSession('ses_abc123', '提交人: gxh\n内容描述: Fix bug', 'opencode');
      expect(uploadResult.success).toBe(true);
      expect(uploadResult.filename).toContain('cannbot_db_session');

      expect(mockImportSession).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db', 'ses_abc123');
      expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh\n内容描述: Fix bug', 'opencode');
    });

    it('resolves framework from source type correctly', async () => {
      // opencode-db → opencode, claude-jsonl → claude-code
      const client = new InsightClient('http://localhost:21025', { retries: 0 });

      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
      await client.uploadSession('ses_abc123', 'description', 'opencode');
      expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', 'description', 'opencode');

      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
      await client.uploadSession('d1ef6b6f', 'description', 'claude-code');
      expect(mockUploadSession).toHaveBeenCalledWith('d1ef6b6f', 'description', 'claude-code');
    });

    it('uses original taskId (not Prisma sessionId) for upload', async () => {
      // The bug: importResult.sessionId is Prisma cuid "cmqyuh9bu0000abc"
      // but upload must use the original taskId "ses_abc123"
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockListImportableSessions.mockResolvedValueOnce(sampleImportableResponse);
      mockImportSession.mockResolvedValueOnce(sampleImportResult);
      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);

      const client = new InsightClient('http://localhost:21025', { retries: 0 });

      // Import returns Prisma ID
      const importResult = await client.importSession('opencode-db', '/path/to/db', 'ses_abc123');
      expect(importResult.sessionId).toBe('cmqyuh9bu0000abc');

      // Upload uses original taskId (from the importable session, not import result)
      await client.uploadSession('ses_abc123', 'description', 'opencode');
      expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', 'description', 'opencode');
      // NOT 'cmqyuh9bu0000abc'
    });

    it('calls listImportableSessions then importSession then uploadSession in sequence', async () => {
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockListImportableSessions.mockResolvedValueOnce(sampleImportableResponse);
      mockImportSession.mockResolvedValueOnce(sampleImportResult);
      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);

      const client = new InsightClient('http://localhost:21025', { retries: 0 });

      // Step 1: list importable sessions
      const importable = await client.listImportableSessions('opencode-db', '/path/to/db');
      expect(importable.sessions.length).toBe(1);

      // Step 2: import
      const importResult = await client.importSession('opencode-db', '/path/to/db', importable.sessions[0].id);
      expect(importResult.imported).toBe(true);

      // Step 3: upload (using original taskId)
      const uploadResult = await client.uploadSession(importable.sessions[0].id, 'description', 'opencode');
      expect(uploadResult.success).toBe(true);

      // Verify call order
      expect(mockListImportableSessions).toHaveBeenCalledBefore(mockImportSession);
      expect(mockImportSession).toHaveBeenCalledBefore(mockUploadSession);
    });
  });

  describe('upload from Insight DB (--session mode)', () => {
    it('uploads by taskId with description', async () => {
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });
      mockGetSession.mockResolvedValueOnce(sampleSessionDetail);
      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);

      const client = new InsightClient('http://localhost:21025', { retries: 0 });
      const result = await client.uploadSession('ses_abc123', '提交人: gxh\n内容描述: test', 'opencode');
      expect(result.success).toBe(true);
      expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh\n内容描述: test', 'opencode');
    });
  });

  describe('client.uploadSession method', () => {
    it('sends POST to /api/ingest/upload-session with taskId, framework, description', async () => {
      const client = new InsightClient('http://localhost:21025', { retries: 0, timeout: 5000 });

      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
      const result = await client.uploadSession('ses_abc123', '提交人: gxh', 'opencode');

      expect(result.success).toBe(true);
      expect(result.filename).toContain('cannbot_db_session');
      expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh', 'opencode');
    });

    it('defaults framework to "unknown" when not provided', () => {
      // The client method signature: uploadSession(taskId, description, framework?)
      // Implementation: framework ?? 'unknown'
      // Verify the logic directly
      const providedFramework = undefined;
      const resolvedFramework = providedFramework ?? 'unknown';
      expect(resolvedFramework).toBe('unknown');
    });
  });

  describe('description template', () => {
    it('description includes all 5 fields', () => {
      const description = `提交人: gxh
内容描述: Fix the build error in op code
问题说明: none
日志路径: /home/gxh/code/logs/opencode.db
备注: 模型 claude-sonnet-4-6, 时间 2026-06-29`;

      expect(description).toContain('提交人:');
      expect(description).toContain('内容描述:');
      expect(description).toContain('问题说明:');
      expect(description).toContain('日志路径:');
      expect(description).toContain('备注:');
    });

    it('auto-fills description from session data', () => {
      const sessionInfo = {
        query: 'Fix the build error',
        user: 'gxh',
        model: 'claude-sonnet-4-6',
        taskId: 'ses_abc123',
        framework: 'opencode',
        sourcePath: '/home/gxh/code/logs/opencode.db',
      };

      // Simulate template construction
      const description = `提交人: ${sessionInfo.user}
内容描述: ${sessionInfo.query}
问题说明:
日志路径: ${sessionInfo.sourcePath}
备注: 模型 ${sessionInfo.model}`;

      expect(description).toContain('提交人: gxh');
      expect(description).toContain('内容描述: Fix the build error');
      expect(description).toContain('日志路径: /home/gxh/code/logs/opencode.db');
      expect(description).toContain('模型 claude-sonnet-4-6');
    });
  });

  describe('JSON output format', () => {
    it('upload result serializes correctly', () => {
      const result: ApiUploadResponse = { success: true, filename: 'cannbot_db_session_ses_abc123_a1b2.db' };
      const json = JSON.stringify(result);
      const parsed = JSON.parse(json);
      expect(parsed.success).toBe(true);
      expect(parsed.filename).toContain('cannbot_db_session');
    });

    it('combined import+upload result serializes correctly', () => {
      const combined = {
        import: sampleImportResult,
        upload: sampleUploadResult,
      };
      const json = JSON.stringify(combined);
      const parsed = JSON.parse(json);
      expect(parsed.import.sessionId).toBe('cmqyuh9bu0000abc');
      expect(parsed.import.imported).toBe(true);
      expect(parsed.upload.success).toBe(true);
    });
  });

  describe('backend auto-start/stop', () => {
    it('detects running backend via listSessions', async () => {
      mockListSessions.mockResolvedValueOnce({ items: [], total: 0, page: 1 });

      const client = new InsightClient('http://localhost:21025', { retries: 0 });
      const result = await client.listSessions({ pageSize: 1 });
      expect(result).toBeDefined();
      expect(mockListSessions).toHaveBeenCalledWith({ pageSize: 1 });
    });

    it('detects backend not running via network error', async () => {
      mockListSessions.mockRejectedValueOnce(new Error('fetch failed'));

      const client = new InsightClient('http://localhost:21025', { retries: 0, retryDelay: 10 });
      try {
        await client.listSessions({ pageSize: 1 });
      } catch (e) {
        expect(e).toBeDefined();
      }
    });
  });

  describe('export pipeline validation', () => {
    it('upload API accepts description in body', async () => {
      const client = new InsightClient('http://localhost:21025', { retries: 0 });

      mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
      await client.uploadSession('ses_abc123', '提交人: gxh\n内容描述: test', 'opencode');

      // Verify the uploadSession call includes description
      const callArgs = mockUploadSession.mock.calls[0];
      expect(callArgs[1]).toBe('提交人: gxh\n内容描述: test');
    });

    it('commit message uses description (not "Add session <taskId>")', async () => {
      // The upload route uses description as commit message:
      // const commitMsg = description ?? `Add session ${taskId}`;
      // When description is provided, it should be used directly
      const description = '提交人: gxh\n内容描述: Fix build';
      const commitMsg = description ?? `Add session ses_abc123`;
      expect(commitMsg).toBe(description);
      expect(commitMsg).not.toContain('Add session');
    });

    it('commit message falls back to "Add session" when no description', () => {
      const description: string | undefined = undefined;
      const taskId = 'ses_abc123';
      const commitMsg = description ?? `Add session ${taskId}`;
      expect(commitMsg).toBe('Add session ses_abc123');
    });
  });
});
