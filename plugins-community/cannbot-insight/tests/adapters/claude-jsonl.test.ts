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
import { readWireEnrichments, wireEnrichmentKey } from '../../src/lib/ingest/adapters/claude-jsonl-wire.ts';
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

    it('sorts sessions by endedAt DESC (falling back to createdAt)', () => {
      const sessions = listSessions(FIXTURE_DIR);
      for (let i = 1; i < sessions.length; i++) {
        const prev = sessions[i - 1].endedAt ?? sessions[i - 1].createdAt;
        const curr = sessions[i].endedAt ?? sessions[i].createdAt;
        expect(prev >= curr).toBe(true);
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

    it('splits the first user message into a system turn + a user prompt turn', () => {
      const interactions = readSession(SYSREMINDER_FILE, 'system-reminder-strip');
      // system turn carries the injected context (CLAUDE.md / currentDate)
      const sysTurns = interactions.filter(i => i.role === 'system');
      expect(sysTurns.length).toBeGreaterThanOrEqual(1);
      expect(sysTurns[0].content).toContain('system-reminder');
      expect(sysTurns[0].content).toContain('claudeMd');
      // user turn is the real prompt only
      const userTurns = interactions.filter(i => i.role === 'user');
      expect(userTurns.length).toBe(1);
      expect(userTurns[0].content).toBe('看看有哪些没有提交的代码');
      expect(userTurns[0].content).not.toContain('system-reminder');
    });

    it('pure-context user message (no real prompt) becomes a system turn only', () => {
      const interactions = readSession(SYSREMINDER_FILE, 'system-reminder-strip');
      // Line 3 is a <system-reminder>-only user message (no prompt) → emitted
      // as a system turn, NO user turn for it. Total: 1 user + 2 system + 2 assistant.
      expect(interactions.filter(i => i.role === 'user').length).toBe(1);
      expect(interactions.filter(i => i.role === 'system').length).toBe(2);
      expect(interactions.filter(i => i.role === 'assistant').length).toBe(2);
    });
  });

  describe('proxy system lines (cannbot-proxy extended claude-format)', () => {
    const PROXY_FILE = path.join(FIXTURE_DIR, 'proxy-system-line.jsonl');

    it('registry line stays inside the round 输入 turn (verbatim wire messages)', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      // proxy 文件走 round-pair 切分：1 round → 输入(user) + 输出(assistant)
      expect(interactions.map(i => i.role)).toEqual(['user', 'assistant']);
      const input = interactions[0];
      expect(input.content).toBe('你用的什么模型');
      // OCP：RawInteraction 不带 content_json（管线不感知 proxy 数据），
      // verbatim 消息由扩展层 readWireEnrichments 按需读取
      expect(input.content_json).toBeUndefined();
      const enrichments = readWireEnrichments(PROXY_FILE);
      // 稳定键 (role, timeInfo.created) —— 不靠数组下标
      const w = JSON.parse(enrichments.get(wireEnrichmentKey(input.role, input.timeInfo!.created))!.contentJson!);
      expect(w.wireInput).toBe(true);
      // reminder 保持在 user 消息内（不拆分），registry 是第 2 条 wire 消息
      expect(w.messages.map((m: { role: string }) => m.role)).toEqual(['user', 'system']);
      expect(JSON.stringify(w.messages[0].content)).toContain('<system-reminder>');
      expect(JSON.stringify(w.messages[1].content)).toContain('Available agent types');
      expect(JSON.stringify(w.messages[1].content)).toContain('- Explore:');
    });

    it('输出 turn stores the verbatim accumulated request in wire order', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      const output = interactions[1];
      // OCP：RawInteraction 不带 input_messages_json
      expect(output.input_messages_json).toBeUndefined();
      const enrichments = readWireEnrichments(PROXY_FILE);
      const req = JSON.parse(enrichments.get(wireEnrichmentKey(output.role, output.timeInfo!.created))!.inputMessagesJson!);
      // wire 顺序：user(reminder+prompt) → system(registry)，registry 在
      // user 之后、response 之前 —— 与真实 messages 数组一致
      expect(req.map((m: { role: string }) => m.role)).toEqual(['user', 'system']);
      expect(req[0].content).toContain('你用的什么模型');
      expect(req[1].content).toContain('- init');
    });

    it('native claude-code type:"system" lines (no message field) stay skipped', () => {
      const interactions = readSession(PROXY_FILE, 'proxy-system-line');
      // The 4th line is a client-side status event (top-level content string,
      // no message) — NOT model input, must not produce a turn.
      expect(interactions.some(i => i.content?.includes('Caveat:'))).toBe(false);
    });
  });

  describe('proxy compact boundary enrichment alignment', () => {
    // Regression: compact 边界会出现「孤儿 input turn」——其配对 assistant
    // 的 content 为空（compact 摘要生成响应，buildAssistantInteraction 返回
    // null），buildWireRounds 仍 push 该 user input。真实 session d3514ea9
    // 经增量 merge 后管线会丢弃这类孤儿，DB turn 数 < buildWireRounds 数组
    // 长度 → 数组下标 ≠ DB turnIndex → 按 turnIndex 查 enrichment 会让
    // compact 后的 turn 拿到前一个孤儿的数据（output 拿到 input 的
    // contentJson、下一个 input 拿到 output 的 inputMessagesJson）。
    // 修法：enrichment 按 (role, createdAt_ms) 稳定键匹配，不靠下标。
    const PROXY_COMPACT_FILE = path.join(FIXTURE_DIR, 'proxy-compact-boundary.jsonl');

    it('fixture 产生 compact 边界 + 孤儿 input（空 assistant）', () => {
      const interactions = readSession(PROXY_COMPACT_FILE, 'compact-boundary');
      // 8 行 → 7 interaction（空 assistant 的 round：input 五子棋 push、
      // output 因 buildAssistantInteraction 返回 null 而 drop → 孤儿 input）
      expect(interactions.length).toBe(7);
      // arr2 是孤儿 input（compact 前的最后一问），后面紧跟 continuation
      expect(interactions[2].role).toBe('user');
      expect(interactions[2].content).toContain('compact 前的最后一问');
      // arr3 是 continuation（"This session is being continued"）
      expect(interactions[3].role).toBe('user');
      expect(interactions[3].content).toContain('continued from a previous');
    });

    it('enrichment 按稳定键对齐 —— 孤儿被丢弃后 compact 后 turn 仍拿到自己的数据', () => {
      const interactions = readSession(PROXY_COMPACT_FILE, 'compact-boundary');
      const enrichments = readWireEnrichments(PROXY_COMPACT_FILE);
      // 模拟管线丢弃孤儿（arr2，compact 前的最后一问）——真实 merge 会这样。
      // 丢弃后 DB turnIndex = buildWireRounds 数组下标 -1（孤儿之后全部漂移）。
      const dbTurns = interactions.filter((_, i) => i !== 2);
      // DB turnIndex 2 现在是 continuation（arr3），不再是孤儿（arr2）。
      // 按稳定键 (role, createdAt_ts) 查，必须拿到 continuation 自己的
      // contentJson，而非孤儿的数据。
      const continuation = dbTurns[2];
      expect(continuation.content).toContain('continued from a previous');
      const contKey = wireEnrichmentKey(continuation.role, continuation.timeInfo!.created);
      const contEnrich = enrichments.get(contKey);
      expect(contEnrich).toBeDefined();
      expect(contEnrich!.contentJson).not.toBeNull();
      expect(contEnrich!.inputMessagesJson).toBeNull();
      const w = JSON.parse(contEnrich!.contentJson!);
      expect(w.wireInput).toBe(true);
      // continuation 的 verbatim 消息含 compact 摘要文本，不是「compact 前的最后一问」
      expect(JSON.stringify(w.messages)).toContain('continued from a previous');
      expect(JSON.stringify(w.messages)).not.toContain('compact 前的最后一问');
    });

    it('每个 kept turn 的稳定键都解析到自己类型的 enrichment', () => {
      const interactions = readSession(PROXY_COMPACT_FILE, 'compact-boundary');
      const enrichments = readWireEnrichments(PROXY_COMPACT_FILE);
      const dbTurns = interactions.filter((_, i) => i !== 2); // 丢孤儿
      for (const it of dbTurns) {
        const key = wireEnrichmentKey(it.role, it.timeInfo!.created);
        const enrich = enrichments.get(key);
        expect(enrich).toBeDefined();
        if (it.role === 'user') {
          expect(enrich!.contentJson, `user turn @${it.timeInfo!.created} 应有 contentJson`).not.toBeNull();
          expect(enrich!.inputMessagesJson).toBeNull();
        } else {
          expect(enrich!.contentJson).toBeNull();
          expect(enrich!.inputMessagesJson, `assistant turn @${it.timeInfo!.created} 应有 inputMessagesJson`).not.toBeNull();
        }
      }
    });

    it('latency/finishReason 走管线进 interaction，ttftMs 走 enrichment（仅 output turn）', () => {
      const interactions = readSession(PROXY_COMPACT_FILE, 'compact-boundary');
      const enrichments = readWireEnrichments(PROXY_COMPACT_FILE);
      // arr1 = 第一轮的回答（assistant）: fixture duration_ms=1000/stopReason=end_turn/ttftMs=120
      const a1 = interactions[1];
      // latency/finishReason 经标准管线（adapter 读 duration_ms + stopReason →
      // interaction.latency/finish_reason → turn-split → DB），不进 enrichment。
      expect(a1.latency).toBe(1000);
      expect(a1.finish_reason).toBe('end_turn');
      // timeInfo.completed 被 proxy-gate 置 undefined（有 duration_ms），让
      // turn-split fallback 到 interaction.latency 而非 completed-created=0
      expect(a1.timeInfo?.completed).toBeUndefined();
      // ttftMs 无管线字段 → 只在 enrichment（API 层覆盖）
      const a1Enrich = enrichments.get(wireEnrichmentKey(a1.role, a1.timeInfo!.created))!;
      expect(a1Enrich.ttftMs).toBe(120);
      expect(a1Enrich.contentJson).toBeNull();   // output enrichment: imJson not cJson
      expect(a1Enrich.inputMessagesJson).not.toBeNull();
      // input turn（arr0）ttftMs 为 null
      const u0 = interactions[0];
      const u0Enrich = enrichments.get(wireEnrichmentKey(u0.role, u0.timeInfo!.created))!;
      expect(u0Enrich.ttftMs).toBeNull();
      // input turn 不带 latency/finish_reason
      expect(u0.latency).toBeNull();
      expect(u0.finish_reason).toBeNull();
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
