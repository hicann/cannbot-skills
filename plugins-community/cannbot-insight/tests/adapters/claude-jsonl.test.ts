// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { listSessions, readSession, listSubagentSessions } from '../../src/lib/ingest/adapters/claude-jsonl.ts';
import { readFullContext } from '../../src/lib/ingest/adapters/claude-jsonl-full-context.ts';
import path from 'node:path';
import fs from 'node:fs';
import os from 'node:os';

const FIXTURE_DIR = path.resolve(__dirname, '../data/claude-sessions');
const FIXTURE_FILE = path.join(FIXTURE_DIR, 'abc123.jsonl');
const SIMPLE_FILE = path.join(FIXTURE_DIR, 'simple-session.jsonl');
const EMPTY_FILE = path.join(FIXTURE_DIR, 'empty-session.jsonl');
const PARALLEL_TOOLS_FILE = path.join(FIXTURE_DIR, 'parallel-tools.jsonl');
const SYSREMINDER_FILE = path.join(FIXTURE_DIR, 'system-reminder-strip.jsonl');
const E2E_SAMPLE_FILE = path.resolve(__dirname, '../data/e2e/claude-sample.jsonl');

describe('claude-jsonl adapter', () => {
  describe('listSessions', () => {
    it('returns session list from directory', () => {
      const sessions = listSessions(FIXTURE_DIR);
      expect(sessions.length).toBeGreaterThanOrEqual(2);

      const first = sessions[0];
      expect(first).toHaveProperty('id');
      expect(first).toHaveProperty('createdAt');
      expect(first).toHaveProperty('endedAt');
      expect(first).toHaveProperty('firstQuery');
      expect(first).toHaveProperty('turnCount');
      expect(first).toHaveProperty('modelName');
      expect(first).toHaveProperty('totalTokens');

      expect(typeof first.id).toBe('string');
      expect(typeof first.createdAt).toBe('string');
      expect(typeof first.turnCount).toBe('number');
      expect(typeof first.totalTokens).toBe('number');
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

    it('sums totalTokens across assistant usage blocks', () => {
      const sessions = listSessions(FIXTURE_FILE);
      const abc = sessions.find(s => s.id === 'abc123');
      expect(abc?.totalTokens).toBe(630);

      const simple = listSessions(SIMPLE_FILE).find(s => s.id === 'simple-session');
      expect(simple?.totalTokens).toBe(15);
    });

    it('extracts endedAt from last line timestamp and returns null when absent', () => {
      const sessions = listSessions(E2E_SAMPLE_FILE);
      expect(sessions.length).toBe(1);
      const s = sessions[0];
      expect(s.endedAt).toBe('2026-05-19T09:26:27.511Z');
      expect(s.totalTokens).toBe(1500);

      const noTs = listSessions(FIXTURE_FILE).find(x => x.id === 'abc123');
      expect(noTs?.endedAt).toBeNull();
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

  describe('system-reminder splitting', () => {
    it('listSessions firstQuery excludes <system-reminder> context blocks', () => {
      const sessions = listSessions(SYSREMINDER_FILE);
      expect(sessions.length).toBe(1);
      expect(sessions[0].firstQuery).toBe('看看有哪些没有提交的代码');
      expect(sessions[0].firstQuery).not.toContain('system-reminder');
      expect(sessions[0].firstQuery).not.toContain('claudeMd');
    });

    it('user 消息保留 reminder 全文（wire 保真，不再拆 system turn）', () => {
      const interactions = readSession(SYSREMINDER_FILE, 'system-reminder-strip');
      // reminder 是 user 消息内部的 text block —— 同一条 wire 消息，不拆分
      const withPrompt = interactions.filter(i => i.role === 'user' && i.content?.includes('看看有哪些没有提交的代码'));
      expect(withPrompt.length).toBe(1);
      expect(withPrompt[0].content).toContain('<system-reminder>');
      expect(withPrompt[0].content).toContain('claudeMd');
      // 不再产出 reminder 拆分 system turn
      const sysTurns = interactions.filter(i => i.role === 'system');
      expect(sysTurns.filter(i => i.content?.includes('system-reminder')).length).toBe(0);
    });

    it('纯 reminder 的 user 行（无真实提示）原样成为 user turn', () => {
      const interactions = readSession(SYSREMINDER_FILE, 'system-reminder-strip');
      // line 3 是 reminder-only user 行 —— 原样保留（wire 上就是一条 user 消息）
      expect(interactions.filter(i => i.role === 'user').length).toBe(2);
      expect(interactions.filter(i => i.role === 'assistant').length).toBe(2);
      const pure = interactions.filter(i => i.role === 'user').find(i => i.content?.includes('Some mid-session context injection'));
      expect(pure).toBeDefined();
      expect(pure!.content).toContain('<system-reminder>');
    });
  });

  describe('proxy system lines (cannbot-proxy extended claude-format)', () => {
    const PROXY_FILE = path.join(FIXTURE_DIR, 'proxy-system-line.jsonl');

    it('role:"system" message line (agent-types registry) becomes a system turn', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      // system turn: only the registry line (reminder 不再拆分，留在 user 消息内)
      const sysTurns = interactions.filter(i => i.role === 'system');
      expect(sysTurns.length).toBe(1);
      const registry = sysTurns[0];
      expect(registry?.content).toContain('Available agent types');
      expect(registry?.content).toContain('- Explore:');
      expect(registry?.content).toContain('- init');
      expect(interactions.filter(i => i.role === 'user').length).toBe(1);
      expect(interactions.filter(i => i.role === 'assistant').length).toBe(1);
    });

    it('registry system turn sits between the user prompt and the assistant turn', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      const roles = interactions.map(i => i.role);
      // order must mirror the wire conversation: user(reminder 在消息内), system(registry), assistant
      expect(roles).toEqual(['user', 'system', 'assistant']);
      const user = interactions.find(i => i.role === 'user');
      expect(user?.content).toContain('<system-reminder>');
    });

    it('native claude-code type:"system" lines (no message field) stay skipped', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      // The 4th line is a client-side status event (top-level content string,
      // no message) — NOT model input, must not produce a turn.
      expect(interactions.some(i => i.content?.includes('Caveat:'))).toBe(false);
    });
  });


  describe('subagent full context resolution', () => {
    // Regression: a subagent turn must show the SUBAGENT's own system prompt +
    // restricted toolset (from subagents/<subId>.jsonl), NOT the main session's.
    const MAIN_FILE = path.join(FIXTURE_DIR, 'subctx.jsonl');
    const FILE_SID = 'subctx';

    it('listSubagentSessions resolves the subagent file next to the main jsonl', () => {
      const subs = listSubagentSessions(MAIN_FILE, FILE_SID);
      expect(subs.length).toBe(1);
      expect(subs[0].id).toBe('sub-x');
      expect(subs[0].filePath).toContain('subagents');
    });

    it('readFullContext on the subagent file returns the SUBAGENT context, distinct from main', () => {
      const subs = listSubagentSessions(MAIN_FILE, FILE_SID);
      const subCtx = readFullContext(subs[0].filePath)!;
      const mainCtx = readFullContext(MAIN_FILE)!;
      // Subagent has its own system prompt + restricted toolset.
      expect(subCtx.systemPrompt).toContain('SUB_SYSTEM');
      expect(subCtx.tools.map(t => t.name)).toEqual(['C']);
      // Main has a different system + full toolset.
      expect(mainCtx.systemPrompt).toContain('MAIN_SYSTEM');
      expect(mainCtx.tools.map(t => t.name).sort()).toEqual(['A', 'B']);
      // The bug this guards: subagent context must NOT be the main's.
      expect(subCtx.systemPrompt).not.toBe(mainCtx.systemPrompt);
    });
  });
});

// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

describe('task-notification 重分类（原生 claude jsonl）', () => {
  const NOTIF = `<task-notification>
<task-id>abf48b365b9ec5010</task-id>
<tool-use-id>toolu_d1</tool-use-id>
<output-file>/tmp/x/abf48b365b9ec5010.output</output-file>
<status>completed</status>
<summary>Agent "验收算子开发目录" finished</summary>
<note>A task-notification fires each time this agent stops.</note>
<result>PASS</result>
<usage><subagent_tokens>18621</subagent_tokens></usage>
</task-notification>`;

  it('user 行的 task-notification → system turn（可读摘要，非原始 XML）', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'task-notif-'));
    const file = path.join(tmpDir, 'task-notif.jsonl');
    fs.writeFileSync(file, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: '开始任务' }, timestamp: '2026-08-18T10:00:00.000Z' }),
      JSON.stringify({ type: 'user', message: { role: 'user', content: NOTIF }, timestamp: '2026-08-18T10:00:05.000Z' }),
      JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'a1', content: [{ type: 'text', text: '收到' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '2026-08-18T10:00:10.000Z' }),
    ].join('\n') + '\n');
    const interactions = readSession(file, 'task-notif');
    const notif = interactions.find(i => i.content?.includes('✅ Task'));
    expect(notif).toBeDefined();
    expect(notif!.role).toBe('system');
    expect(notif!.content).toContain('✅ Task: 验收算子开发目录 [completed]');
    expect(notif!.content).not.toContain('<task-id>');
    const userTurns = interactions.filter(i => i.role === 'user');
    expect(userTurns).toHaveLength(1);
    expect(userTurns[0].content).toBe('开始任务');
  });

  it('notification 与真实输入混排时顺序保留', () => {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'task-notif-'));
    const file = path.join(tmpDir, 'task-notif-mixed.jsonl');
    fs.writeFileSync(file, [
      JSON.stringify({ type: 'user', message: { role: 'user', content: NOTIF }, timestamp: '2026-08-18T10:00:00.000Z' }),
      JSON.stringify({ type: 'user', message: { role: 'user', content: '真实输入' }, timestamp: '2026-08-18T10:00:01.000Z' }),
      JSON.stringify({ type: 'assistant', message: { role: 'assistant', id: 'a1', content: [{ type: 'text', text: 'ok' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, timestamp: '2026-08-18T10:00:02.000Z' }),
    ].join('\n') + '\n');
    const interactions = readSession(file, 'task-notif-mixed');
    expect(interactions.map(i => i.role)).toEqual(['system', 'user', 'assistant']);
  });
});
