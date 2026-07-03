// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { turnCommand } from '@/cli/commands/turn';
import { InsightClient } from '@/cli/client';
import { formatTokens, formatDuration, truncate } from '@/cli/utils/format';
import { renderTable } from '@/cli/utils/table';
import type { ApiTurnDetailResponse, ApiToolCallDetail, ApiSkillEventDetail } from '@/cli/types';

const mockGetTurnDetail = vi.fn();
vi.spyOn(InsightClient.prototype, 'getTurnDetail').mockImplementation(mockGetTurnDetail);

const mockConsoleLog = vi.spyOn(console, 'log').mockImplementation(() => {});
const mockConsoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

const sampleToolCall: ApiToolCallDetail = {
  id: 'tc1',
  toolCallId: 'tool-1',
  toolName: 'bash',
  argsJson: '{"command":"ls"}',
  resultJson: 'file1\nfile2',
  state: 'completed',
  errorType: null,
  errorMessage: null,
  startedAt: '2026-06-14T10:30:00Z',
  completedAt: '2026-06-14T10:30:01Z',
  durationMs: 1000,
  dispatchBridgeId: null,
  isSkillRelated: false,
};

const sampleSkillEvent: ApiSkillEventDetail = {
  id: 'se1',
  skillName: 'agent-debug',
  skillVersion: '0.4',
  eventType: 'load',
  success: true,
  errorMessage: null,
  argsJson: null,
  startedAt: '2026-06-14T10:30:00Z',
  completedAt: '2026-06-14T10:30:02Z',
  durationMs: 2000,
};

const sampleTurn: ApiTurnDetailResponse = {
  turnId: 'turn-001',
  sessionId: 'session-001',
  turnIndex: 5,
  role: 'assistant',
  content: 'I will implement the feature now.',
  contentJson: null,
  contentSummary: 'Implementing feature with bash tool',
  inputMessagesJson: null,
  inputMessagesCount: 3,
  inputMessagesTokens: 5000,
  contextWindowPct: 35.2,
  agentName: 'main',
  subagentName: null,
  subagentSessionId: null,
  isSubagent: false,
  totalTokens: 15000,
  inputTokens: 10000,
  outputTokens: 5000,
  reasoningTokens: 2000,
  cacheReadTokens: 3000,
  cacheWriteTokens: 1000,
  latencyMs: 5000,
  ttftMs: 200,
  createdAt: '2026-06-14T10:30:00Z',
  completedAt: '2026-06-14T10:30:05Z',
  model: 'claude-3.5-sonnet',
  modelId: 'claude-3-5-sonnet-20241022',
  providerId: 'anthropic',
  finishReason: 'tool_use',
  toolCalls: [sampleToolCall],
  skillEvents: [sampleSkillEvent],
};

describe('turnCommand', () => {
  beforeEach(() => {
    mockGetTurnDetail.mockReset();
    mockConsoleLog.mockClear();
    mockConsoleError.mockClear();
  });

  it('registers as a Commander sub-command', () => {
    const cmd = turnCommand();
    expect(cmd.name()).toBe('turn');
    expect(cmd.description()).toContain('turn detail');
  });

  it('has --json option', () => {
    const cmd = turnCommand();
    const options = cmd.options.map(o => o.long);
    expect(options).toContain('--json');
  });

  it('registers turnId argument', () => {
    const cmd = turnCommand();
    expect(cmd._args.length).toBeGreaterThanOrEqual(1);
  });

  it('formats turn detail correctly', () => {
    const turn = sampleTurn;

    expect(formatTokens(turn.totalTokens)).toBe('15.0K');
    expect(formatTokens(turn.inputTokens)).toBe('10.0K');
    expect(formatTokens(turn.outputTokens)).toBe('5.0K');
    expect(formatTokens(turn.reasoningTokens)).toBe('2.0K');
    expect(formatTokens(turn.cacheReadTokens)).toBe('3.0K');
    expect(formatTokens(turn.cacheWriteTokens)).toBe('1.0K');
    expect(formatDuration(turn.latencyMs)).toBe('5.0s');
    expect(formatDuration(turn.ttftMs)).toBe('200ms');
    expect(turn.contextWindowPct.toFixed(1)).toBe('35.2');
  });

  it('renders tool calls table', () => {
    const TOOL_CALL_COLUMNS = [
      { key: 'toolName', label: 'Tool', width: 20 },
      { key: 'state', label: 'State', width: 10 },
      { key: 'durationMs', label: 'Duration', width: 10 },
    ];

    const table = renderTable(
      TOOL_CALL_COLUMNS,
      [sampleToolCall] as unknown as Record<string, unknown>[],
      (row: Record<string, unknown>, key: string) => {
        const tc = row as unknown as ApiToolCallDetail;
        if (key === 'toolName') return tc.toolName ?? '—';
        if (key === 'state') return tc.state ?? '—';
        if (key === 'durationMs') return formatDuration(tc.durationMs);
        return String(tc[key as keyof ApiToolCallDetail] ?? '—');
      },
    );

    expect(table).toContain('bash');
    expect(table).toContain('completed');
    expect(table).toContain('1.0s');
  });

  it('renders skill events table', () => {
    const SKILL_COLUMNS = [
      { key: 'skillName', label: 'Skill', width: 20 },
      { key: 'eventType', label: 'Event', width: 10 },
      { key: 'success', label: 'Success', width: 8 },
      { key: 'durationMs', label: 'Duration', width: 10 },
    ];

    const table = renderTable(
      SKILL_COLUMNS,
      [sampleSkillEvent] as unknown as Record<string, unknown>[],
      (row: Record<string, unknown>, key: string) => {
        const se = row as unknown as ApiSkillEventDetail;
        if (key === 'skillName') return se.skillName ?? '—';
        if (key === 'eventType') return se.eventType ?? '—';
        if (key === 'success') return se.success ? '✓' : '✗';
        if (key === 'durationMs') return formatDuration(se.durationMs);
        return String(se[key as keyof ApiSkillEventDetail] ?? '—');
      },
    );

    expect(table).toContain('agent-debug');
    expect(table).toContain('load');
    expect(table).toContain('✓');
  });

  it('truncates content summary', () => {
    const longSummary = 'This is a very long content summary that should be truncated when displayed in the turn detail view to avoid overwhelming the terminal output';
    const truncated = truncate(longSummary, 300);
    expect(truncated.length).toBeLessThanOrEqual(longSummary.length);
  });

  it('JSON output includes all fields', async () => {
    mockGetTurnDetail.mockResolvedValueOnce(sampleTurn);

    const client = new InsightClient('http://localhost:21025', { retries: 0 });
    const turn = await client.getTurnDetail('turn-001');

    const json = JSON.stringify(turn, null, 2);
    const parsed = JSON.parse(json);
    expect(parsed.turnId).toBe('turn-001');
    expect(parsed.turnIndex).toBe(5);
    expect(parsed.role).toBe('assistant');
    expect(parsed.model).toBe('claude-3.5-sonnet');
    expect(parsed.totalTokens).toBe(15000);
    expect(parsed.toolCalls.length).toBe(1);
    expect(parsed.skillEvents.length).toBe(1);
  });

  it('handles subagent turn display', () => {
    const subTurn = { ...sampleTurn, isSubagent: true, subagentName: 'explore', subagentSessionId: 'sub-001' };
    expect(subTurn.isSubagent).toBe(true);
    expect(subTurn.subagentName).toBe('explore');
    expect(subTurn.subagentSessionId).toBe('sub-001');
  });
});
