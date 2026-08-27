// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// CLI upload command 端到端 IT：用 commander.parseAsync 真实驱动 uploadCommand
// 的 action 流程（与 upload.test.ts 互补——后者直接调 client 方法，不走
// commander 解析）。vi.spyOn(InsightClient.prototype) mock HTTP 避免真起后端，
// ensureBackend mock listSessions 成功 → 跳过 spawnBackend（绝不真起 next dev）。
// readline 用 vi.mock 整体替换（vi.spyOn 对 node: 内置模块的 ESM namespace
// frozen 属性不生效）。
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { InsightClient } from '@/cli/client';
import { uploadCommand } from '@/cli/commands/upload';
import type {
  ApiImportableSession,
  ApiImportableSessionsResponse,
  ApiImportResponse,
  ApiUploadResponse,
  ApiSessionDetailResponse,
} from '@/cli/types';

// vi.hoisted 暴露 mock 引用给 vi.mock factory 用（factory 是 hoisted，不能
// 引用外部变量）
const { mockCreateInterface } = vi.hoisted(() => ({ mockCreateInterface: vi.fn() }));
vi.mock('node:readline', () => ({
  default: { createInterface: mockCreateInterface },
  createInterface: mockCreateInterface,
}));

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

const sampleImportable: ApiImportableSessionsResponse = { sessions: [sampleSession] };
const sampleImportResult: ApiImportResponse = { sessionId: 'cmqyuh9bu0000abc', imported: true };
const sampleUploadResult: ApiUploadResponse = {
  sid: 'ses_abc123',
  folder: 'sessions/ses_abc123/',
  unchanged: false,
};
const sampleSessionDetail: ApiSessionDetailResponse = {
  sessionId: 'cmqyuh9bu0000abc',
  taskId: 'ses_abc123',
  label: null,
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
  agents: [],
  skills: [],
};

describe('uploadCommand (commander.parseAsync 驱动 action 流程)', () => {
  beforeEach(() => {
    mockListSessions.mockReset();
    mockListImportableSessions.mockReset();
    mockImportSession.mockReset();
    mockUploadSession.mockReset();
    mockGetSession.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
    mockProcessExit.mockClear();
    mockCreateInterface.mockReset();
    // 默认 ensureBackend → listSessions 成功 → 跳过 spawnBackend（绝不真起 next dev）
    mockListSessions.mockResolvedValue({ items: [], total: 0, page: 1 });
    // 默认 readline.createInterface 返回 fakeRl，question 立即 callback('default')
    const fakeRl = {
      question: vi.fn((_label: string, cb: (a: string) => void) => cb('default')),
      close: vi.fn(),
    };
    mockCreateInterface.mockReturnValue(fakeRl);
  });

  it('--session <id> --description <text>：直接调 client.uploadSession + 输出 formatSuccess', async () => {
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--session', 'ses_abc123', '--description', '提交人: gxh\n内容描述: test'],
      { from: 'user' },
    );
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh\n内容描述: test');
    expect(mockConsoleLog).toHaveBeenCalledWith(expect.stringContaining('Uploaded session ses_abc123'));
  });

  it('--session <id> --json：JSON 输出 upload 结果', async () => {
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--session', 'ses_abc123', '--description', 'desc', '--json'],
      { from: 'user' },
    );
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', 'desc');
    const jsonCall = mockConsoleLog.mock.calls.find(c => typeof c[0] === 'string' && c[0].trimStart().startsWith('{'));
    expect(jsonCall).toBeDefined();
    expect(JSON.parse(jsonCall![0] as string)).toEqual(sampleUploadResult);
  });

  it('--file <path> --source opencode-db --session-id <id> --yes --description：listImportable → import → upload 顺序', async () => {
    mockListImportableSessions.mockResolvedValueOnce(sampleImportable);
    mockImportSession.mockResolvedValueOnce(sampleImportResult);
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--file', '/path/to/sessions.db', '--source', 'opencode-db', '--session-id', 'ses_abc123', '--yes', '--description', '提交人: gxh'],
      { from: 'user' },
    );
    expect(mockListImportableSessions).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db');
    expect(mockImportSession).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db', 'ses_abc123');
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh');
    // 顺序：listImportable → import → upload
    expect(mockListImportableSessions).toHaveBeenCalledBefore(mockImportSession);
    expect(mockImportSession).toHaveBeenCalledBefore(mockUploadSession);
  });

  it('--file <path> --list：列出可导入会话 + renderTable 输出', async () => {
    mockListImportableSessions.mockResolvedValueOnce(sampleImportable);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--file', '/path/to/sessions.db', '--source', 'opencode-db', '--list'],
      { from: 'user' },
    );
    expect(mockListImportableSessions).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db');
    // 不应继续 import/upload
    expect(mockImportSession).not.toHaveBeenCalled();
    expect(mockUploadSession).not.toHaveBeenCalled();
    expect(mockConsoleLog).toHaveBeenCalledWith(expect.stringContaining('Importable Sessions'));
  });

  it('缺 --session / --file → process.exit(1) + console.error 提示', async () => {
    const cmd = uploadCommand();
    await cmd.parseAsync([], { from: 'user' }).catch(() => {});
    expect(mockProcessExit).toHaveBeenCalledWith(1);
    expect(mockConsoleError).toHaveBeenCalledWith(expect.stringContaining('Specify --session'));
  });

  it('--file <path> --session-id <不存在> → process.exit(1) + console.error not found', async () => {
    mockListImportableSessions.mockResolvedValueOnce(sampleImportable);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--file', '/path/to/sessions.db', '--source', 'opencode-db', '--session-id', 'nonexistent', '--yes', '--description', 'x'],
      { from: 'user' },
    ).catch(() => {});
    expect(mockProcessExit).toHaveBeenCalledWith(1);
    expect(mockConsoleError).toHaveBeenCalledWith(expect.stringContaining('not found'));
  });

  it('--session <id> --interactive：readline.createInterface mock 收集 5 字段模板', async () => {
    mockGetSession.mockResolvedValueOnce(sampleSessionDetail);
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    // beforeEach 已设 mockCreateInterface 返回 fakeRl（question 立即 cb('default')）
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--session', 'ses_abc123', '--interactive'],
      { from: 'user' },
    );
    // description 模板含全部 5 字段
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('提交人:'));
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('内容描述:'));
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('问题说明:'));
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('日志路径:'));
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('备注:'));
    // collectDescription 调 5 次 inputPrompt → 5 次 createInterface
    expect(mockCreateInterface).toHaveBeenCalledTimes(5);
  });

  it('--file <path> 单会话自动选中（无 --session-id）→ import + upload', async () => {
    mockListImportableSessions.mockResolvedValueOnce(sampleImportable); // 1 个 session
    mockImportSession.mockResolvedValueOnce(sampleImportResult);
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--file', '/path/to/sessions.db', '--source', 'opencode-db', '--yes', '--description', '提交人: gxh'],
      { from: 'user' },
    );
    // 单会话自动选中 target = sessions[0]
    expect(mockImportSession).toHaveBeenCalledWith('opencode-db', '/path/to/sessions.db', 'ses_abc123');
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', '提交人: gxh');
  });

  it('--session <id>：getSession 拿 detail 填 description 默认值（用户/查询/模型）', async () => {
    mockGetSession.mockResolvedValueOnce(sampleSessionDetail);
    mockUploadSession.mockResolvedValueOnce(sampleUploadResult);
    // 覆盖 beforeEach 默认 fakeRl：question 空回车 → inputPrompt 用 defaultVal
    const emptyRl = {
      question: vi.fn((_label: string, cb: (a: string) => void) => cb('')),
      close: vi.fn(),
    };
    mockCreateInterface.mockReturnValue(emptyRl);
    const cmd = uploadCommand();
    await cmd.parseAsync(
      ['--session', 'ses_abc123', '--interactive'],
      { from: 'user' },
    );
    // 空回车 → 提交人用 detail.user='gxh'、内容描述用 detail.query='Fix the build error'
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('提交人: gxh'));
    expect(mockUploadSession).toHaveBeenCalledWith('ses_abc123', expect.stringContaining('内容描述: Fix the build error'));
  });
});
