// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { listSessions, readSession } from '../../src/lib/ingest/adapters/claude-jsonl.ts';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

const FIXTURE_DIR = path.resolve(__dirname, '../data/claude-sessions');
const FIXTURE_FILE = path.join(FIXTURE_DIR, 'abc123.jsonl');
const SIMPLE_FILE = path.join(FIXTURE_DIR, 'simple-session.jsonl');
const EMPTY_FILE = path.join(FIXTURE_DIR, 'empty-session.jsonl');
const PARALLEL_TOOLS_FILE = path.join(FIXTURE_DIR, 'parallel-tools.jsonl');

describe('claude-jsonl adapter', () => {
  describe('listSessions', () => {
    it('returns session list from directory', () => {
      const sessions = listSessions(FIXTURE_DIR);
      expect(sessions.length).toBeGreaterThanOrEqual(2);

      const first = sessions[0];
      expect(first).toHaveProperty('id');
      expect(first).toHaveProperty('createdAt');
      expect(first).toHaveProperty('firstQuery');
      expect(first).toHaveProperty('turnCount');
      expect(first).toHaveProperty('modelName');

      expect(typeof first.id).toBe('string');
      expect(typeof first.createdAt).toBe('string');
      expect(typeof first.turnCount).toBe('number');
      expect(first.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('returns session list from single file', () => {
      const sessions = listSessions(FIXTURE_FILE);
      expect(sessions.length).toBe(1);
      expect(sessions[0].id).toBe('abc123');
      expect(sessions[0].firstQuery).toContain('refactor');
      expect(sessions[0].modelName).toBe('claude-sonnet-4-20250514');
    });

    it('extracts firstQuery from user messages', () => {
      const sessions = listSessions(FIXTURE_DIR);
      const abcSession = sessions.find(s => s.id === 'abc123');
      expect(abcSession?.firstQuery).toContain('refactor');
    });

    it('extracts modelName from assistant messages', () => {
      const sessions = listSessions(FIXTURE_DIR);
      const simpleSession = sessions.find(s => s.id === 'simple-session');
      expect(simpleSession?.modelName).toBe('claude-haiku-3-20250415');
    });

    it('handles nonexistent directory gracefully', () => {
      const sessions = listSessions('/nonexistent/path/that/does/not/exist');
      expect(sessions).toEqual([]);
    });

    it('handles empty string path gracefully', () => {
      const sessions = listSessions('');
      expect(sessions).toEqual([]);
    });

    it('skips empty JSONL files', () => {
      const sessions = listSessions(FIXTURE_DIR);
      const emptyEntry = sessions.find(s => s.id === 'empty-session');
      expect(emptyEntry).toBeUndefined();
    });

    it('derives session id from file name', () => {
      const sessions = listSessions(SIMPLE_FILE);
      expect(sessions[0].id).toBe('simple-session');
    });
  });

  describe('readSession', () => {
    it('returns RawInteraction[] for a real session', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      expect(interactions.length).toBeGreaterThan(0);

      expect(interactions[0]).toHaveProperty('role');
      expect(interactions[0]).toHaveProperty('content');
      expect(interactions[0]).toHaveProperty('timestamp');
      expect(interactions[0]).toHaveProperty('timeInfo');
      expect(interactions[0]).toHaveProperty('agent');
      expect(interactions[0]).toHaveProperty('subagent_name');
      expect(interactions[0]).toHaveProperty('subagent_session_id');
      expect(interactions[0]).toHaveProperty('tool_calls');
      expect(interactions[0]).toHaveProperty('usage');
      expect(interactions[0]).toHaveProperty('model');
      expect(interactions[0]).toHaveProperty('modelID');
      expect(interactions[0]).toHaveProperty('providerID');
      expect(interactions[0]).toHaveProperty('latency');
      expect(interactions[0]).toHaveProperty('finish_reason');
    });

    it('returns correct roles', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const roles = interactions.map(i => i.role);
      expect(roles).toContain('user');
      expect(roles).toContain('assistant');
      expect(roles).toContain('result');
    });

    it('extracts user content as text', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const userMsg = interactions.find(i => i.role === 'user');
      expect(userMsg?.content).toContain('refactor');
    });

    it('extracts assistant text content from content array', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const assistantMsgs = interactions.filter(i => i.role === 'assistant');
      expect(assistantMsgs.length).toBeGreaterThan(0);
      expect(assistantMsgs[0].content).toContain('analyze');
    });

    it('maps usage fields correctly', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const assistantWithUsage = interactions.find(
        i => i.role === 'assistant' && i.usage
      );
      if (assistantWithUsage) {
        expect(assistantWithUsage.usage!.input).toBeGreaterThan(0);
        expect(assistantWithUsage.usage!.output).toBeGreaterThan(0);
        expect(assistantWithUsage.usage!.total).toBe(
          assistantWithUsage.usage!.input + assistantWithUsage.usage!.output + assistantWithUsage.usage!.cacheRead + assistantWithUsage.usage!.cacheWrite
        );
        expect(assistantWithUsage.usage!.reasoning).toBe(0);
        expect(typeof assistantWithUsage.usage!.cacheRead).toBe('number');
        expect(typeof assistantWithUsage.usage!.cacheWrite).toBe('number');
        expect(assistantWithUsage.usage!.cost).toBeGreaterThan(0);
      }
    });

    it('extracts model name from assistant messages', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const assistant = interactions.find(i => i.role === 'assistant' && i.model);
      expect(assistant?.model).toBe('claude-sonnet-4-20250514');
    });

    it('handles tool_use / tool_result pairing', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const assistantWithTools = interactions.find(
        i => i.role === 'assistant' && i.tool_calls && i.tool_calls!.length > 0
      );
      if (assistantWithTools) {
        const tc = assistantWithTools.tool_calls![0];
        expect(tc.toolCallId).toBe('toolu_01ABC');
        expect(tc.toolName).toBe('ReadFile');
        expect(tc.argsJson).toContain('src/auth/module.ts');
        expect(tc.resultJson).toContain('auth module content');
        expect(tc.state).toBe('completed');
      }
    });

    it('handles result type messages', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      const resultMsg = interactions.find(i => i.role === 'result');
      expect(resultMsg).toBeDefined();
      expect(resultMsg?.content).toContain('Refactoring plan');
      expect(resultMsg?.finish_reason).toBe('success');
      expect(resultMsg?.latency).toBe(15000);
      expect(resultMsg?.usage?.cost).toBe(0.01);
    });

    it('handles empty file gracefully', () => {
      const interactions = readSession(EMPTY_FILE, 'empty-session');
      expect(interactions).toEqual([]);
    });

    it('handles nonexistent file gracefully', () => {
      const interactions = readSession('/nonexistent/file.jsonl', 'fake');
      expect(interactions).toEqual([]);
    });

    it('handles empty path gracefully', () => {
      const interactions = readSession('', 'any');
      expect(interactions).toEqual([]);
    });

    it('generates valid timestamps from file mtime', () => {
      const interactions = readSession(SIMPLE_FILE, 'simple-session');
      for (const i of interactions) {
        expect(i.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      }
    });

    it('has null subagent fields', () => {
      const interactions = readSession(FIXTURE_FILE, 'abc123');
      for (const i of interactions) {
        expect(i.subagent_name).toBeNull();
        expect(i.subagent_session_id).toBeNull();
      }
    });

    it('extracts latency from duration_ms', () => {
      const interactions = readSession(SIMPLE_FILE, 'simple-session');
      const assistant = interactions.find(i => i.role === 'assistant');
      expect(assistant?.latency).toBe(1000);
    });

    it('extracts assistant content without tool_use returns null tool_calls', () => {
      const interactions = readSession(SIMPLE_FILE, 'simple-session');
      const assistant = interactions.find(i => i.role === 'assistant');
      expect(assistant?.tool_calls).toBeNull();
    });

    it('sorts sessions by createdAt DESC', () => {
      const sessions = listSessions(FIXTURE_DIR);
      for (let i = 1; i < sessions.length; i++) {
        expect(sessions[i - 1].createdAt >= sessions[i].createdAt).toBe(true);
      }
    });

    it('merges parallel tool calls from same API response into one turn', () => {
      // When Claude Code streams a single API response with multiple tool_use blocks,
      // it writes them as separate assistant lines interleaved with user tool_result lines.
      // These should be merged into one assistant turn with all tool calls.
      const interactions = readSession(PARALLEL_TOOLS_FILE, 'parallel-tools');

      // Should have: 1 user + 1 assistant (with 2 tool calls) + 1 assistant (text) + 1 result = 4
      const assistantTurns = interactions.filter(i => i.role === 'assistant');
      expect(assistantTurns.length).toBe(2);

      // First assistant turn should have BOTH tool calls merged
      const merged = assistantTurns[0];
      expect(merged.tool_calls).not.toBeNull();
      expect(merged.tool_calls!.length).toBe(2);

      // First tool call: Bash
      expect(merged.tool_calls![0].toolCallId).toBe('call_00_bash');
      expect(merged.tool_calls![0].toolName).toBe('Bash');
      expect(merged.tool_calls![0].resultJson).toContain('PermissionError');

      // Second tool call: Agent (cancelled due to parallel Bash error)
      expect(merged.tool_calls![1].toolCallId).toBe('call_01_agent');
      expect(merged.tool_calls![1].toolName).toBe('Agent');
      expect(merged.tool_calls![1].resultJson).toContain('Cancelled');

      // Second assistant turn is the follow-up text (separate API response)
      expect(assistantTurns[1].tool_calls).toBeNull();
      expect(assistantTurns[1].content).toContain('Both tasks completed');
    });

    it('handles malformed JSON lines by skipping them', () => {
      const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'claude-jsonl-test-'));
      const tmpFile = path.join(tmpDir, 'malformed.jsonl');
      fs.writeFileSync(tmpFile, [
        '{"type":"user","message":{"role":"user","content":"hello"}}',
        'not-json-at-all',
        '{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"hi"}],"model":"claude-test","usage":{"input_tokens":10,"output_tokens":5}}}',
      ].join('\n'));

      const interactions = readSession(tmpFile, 'malformed');
      expect(interactions.length).toBe(2);
      expect(interactions[0].role).toBe('user');
      expect(interactions[1].role).toBe('assistant');

      fs.rmSync(tmpDir, { recursive: true });
    });
  });
});
