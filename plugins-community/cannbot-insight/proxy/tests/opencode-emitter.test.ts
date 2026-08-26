// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Self-contained opencode-emitter test: verify the emitter's JSON OUTPUT is
// correct extended claude-format (no cannbot-insight / Prisma dependency).
// The cannbot-insight claude-jsonl adapter (its own tests) consumes this format;
// the two sides are coupled only by the format contract — same as the
// claude-emitter tests, but for the OpenAI-wire → claude-format conversion.

import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { emit } from '../src/opencode-emitter.ts';
import { sessionFilePath } from '../src/writer.ts';

const SID = 'sid-main';
const FIXTURE = path.resolve(__dirname, 'data/opencode-wire-records.jsonl');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cpx-opencode-'));
process.env.CANNBOT_PROXY_DIR = tmpDir; // BEFORE sessionFilePath is evaluated at module scope
const mainFile = sessionFilePath(SID); // cpx- prefixed, keyed by real sid
const subagentsDir = path.join(tmpDir, SID, 'subagents');

function readJsonl(file: string): any[] {
  if (!fs.existsSync(file)) return [];
  return fs.readFileSync(file, 'utf-8').split('\n').filter(l => l.trim()).map(l => JSON.parse(l));
}
function subFiles(): string[] {
  if (!fs.existsSync(subagentsDir)) return [];
  return fs.readdirSync(subagentsDir).filter(f => f.endsWith('.jsonl'));
}
function metaFiles(): string[] {
  if (!fs.existsSync(subagentsDir)) return [];
  return fs.readdirSync(subagentsDir).filter(f => f.endsWith('.meta.json'));
}

beforeAll(() => {
  const lines = fs.readFileSync(FIXTURE, 'utf-8').split('\n').filter(l => l.trim());
  for (const line of lines) {
    try { emit(JSON.parse(line)); } catch { /* skip */ }
  }
});

describe('opencode-emitter: OpenAI wire-record → extended claude-format JSON', () => {
  it('main session jsonl has user + assistant lines', () => {
    const rows = readJsonl(mainFile);
    expect(rows.length).toBeGreaterThan(0);
    expect(rows.some(r => r.type === 'user')).toBe(true);
    expect(rows.some(r => r.type === 'assistant')).toBe(true);
  });

  it('title-generation record is skipped', () => {
    const all = [...readJsonl(mainFile), ...subFiles().flatMap(f => readJsonl(path.join(subagentsDir, f)))];
    expect(all.every(r => !JSON.stringify(r).includes('title generator'))).toBe(true);
  });

  it('assistant lines carry extension system + tools fields (OpenAI→claude converted)', () => {
    const asst = readJsonl(mainFile).find(r => r.type === 'assistant');
    expect(asst).toBeDefined();
    expect(asst.tools).toBeDefined();
    expect(asst.tools.map((t: any) => t.name).sort()).toEqual(['bash', 'read']);
    expect(asst.tools.every((t: any) => typeof t.description === 'string')).toBe(true);
    // `system` = the opencode system VERBATIM (capture ≠ interpret): the proxy
    // writes the raw system (persona + `Instructions from:` memory +
    // `<available_skills>` skills) as-is; structural parsing is done by the
    // standalone opencode-context-parser, not the emitter.
    expect(asst.system).toContain('You are opencode');
    expect(asst.system).toContain('Instructions from: /home/user/AGENTS.md');
    expect(asst.system).toContain('<available_skills>');
  });

  it('system is extracted from messages[role:system], NOT emitted as a user turn', () => {
    const users = readJsonl(mainFile).filter(r => r.type === 'user');
    expect(users.every(r => !(typeof r.message?.content === 'string' && r.message.content.includes('You are opencode')))).toBe(true);
  });

  it('does NOT duplicate prior assistant responses (delta skips role:assistant)', () => {
    // rec1, rec3, rec5 each emit one assistant line for the main session.
    const mainAsst = readJsonl(mainFile).filter(r => r.type === 'assistant');
    expect(mainAsst.length).toBe(3);
  });

  it('delta-emits only NEW user messages (rec1 "hello" + rec3 "research..." + rec5 tool result)', () => {
    const mainUsers = readJsonl(mainFile).filter(r => r.type === 'user');
    const texts = mainUsers.map(r => typeof r.message?.content === 'string' ? r.message.content : '');
    expect(texts.some(t => t === 'hello')).toBe(true);
    expect(texts.some(t => t === 'research the codebase and list files')).toBe(true);
  });

  it('routes 1 subagent to <sid>/subagents/<subId>.jsonl by x-session-id', () => {
    expect(subFiles().length).toBe(1);
    expect(subFiles()[0]).toBe('ses_sub_002.jsonl');
  });

  it('subagent file has user(task) + assistant lines', () => {
    for (const f of subFiles()) {
      const rows = readJsonl(path.join(subagentsDir, f));
      expect(rows.some(r => r.type === 'user')).toBe(true);
      expect(rows.some(r => r.type === 'assistant')).toBe(true);
    }
  });

  it('subagent meta.json links toolUseId + name + agentType via Task dispatch matching', () => {
    expect(metaFiles().length).toBe(1);
    const meta = JSON.parse(fs.readFileSync(path.join(subagentsDir, metaFiles()[0]), 'utf-8'));
    expect(meta.toolUseId).toBe('call_task1');
    expect(meta.name).toBe('researcher');
    expect(meta.agentType).toBe('explore');
  });

  it('subagent assistant line carries the SUBAGENT system + restricted toolset', () => {
    const subRows = readJsonl(path.join(subagentsDir, subFiles()[0]));
    const asst = subRows.find(r => r.type === 'assistant');
    expect(asst).toBeDefined();
    expect(asst.system).toContain('subagent');
    expect(asst.tools.map((t: any) => t.name)).toEqual(['read']);
  });

  it('converts OpenAI role:tool result → claude user line with tool_result block', () => {
    // rec5's role:tool (call_bash1 → "file1.txt\nfile2.txt") must surface as a
    // user line carrying a tool_result content block.
    const has = readJsonl(mainFile).some(r =>
      r.type === 'user' &&
      Array.isArray(r.message?.content) &&
      r.message.content.some((b: any) => b.type === 'tool_result' && b.tool_use_id === 'call_bash1' && b.content === 'file1.txt\nfile2.txt')
    );
    expect(has).toBe(true);
  });

  it('main assistant dispatch line carries the Task + bash tool_use blocks', () => {
    const dispatch = readJsonl(mainFile).find(r =>
      r.type === 'assistant' &&
      Array.isArray(r.message?.content) &&
      r.message.content.some((b: any) => b.type === 'tool_use' && b.name === 'task')
    );
    expect(dispatch).toBeDefined();
    const names = dispatch.message.content.filter((b: any) => b.type === 'tool_use').map((b: any) => b.name).sort();
    expect(names).toEqual(['bash', 'task']);
  });

  it('carries the reassembled Anthropic usage verbatim (no double-mapping to 0)', () => {
    // The OpenAIReassembler already converts prompt_tokens/completion_tokens →
    // input_tokens/output_tokens. The emitter must pass that through unchanged.
    // Regression: re-mapping here (reading prompt_tokens, now absent) zeroed all
    // usage → insight showed no System(hidden)/Other context in LLM Input.
    const asst = readJsonl(mainFile).find(r => r.type === 'assistant');
    const u = asst.message?.usage;
    expect(u).toBeDefined();
    expect(u.input_tokens).toBe(50);
    expect(u.output_tokens).toBe(5);
  });

  it('auth/key material is never present in the emitted jsonl', () => {
    const all = [...readJsonl(mainFile), ...subFiles().flatMap(f => readJsonl(path.join(subagentsDir, f)))];
    expect(all.every(r => !JSON.stringify(r).includes('Bearer'))).toBe(true);
    expect(all.every(r => !JSON.stringify(r).includes('apiKey'))).toBe(true);
  });
});

// Header-routed sessions (opencode ≥ 1.17.9, wire-verified): the resolver
// names each record by its x-session-id, and child sessions carry
// x-parent-session-id pointing at the main session — resume/fork keep or mint
// ids exactly like claude, so captures follow the conversation, not the run.
describe('opencode-emitter: header-routed sessions (x-session-id + x-parent-session-id)', () => {
  const mk = (sid: string, xid: string, parent: string | null, userText: string) => ({
    sid,
    protocol: 'openai' as const,
    receivedAt: Date.now(),
    completedAt: Date.now() + 5,
    latencyMs: 5,
    ttftMs: 1,
    request: {
      path: '/rec/chat/completions',
      model: 'rec-model',
      body: { messages: [
        { role: 'system', content: 'You are opencode' },
        { role: 'user', content: userText },
      ] },
    },
    response: { status: 200, model: 'rec-model', stop_reason: 'stop', content: [{ type: 'text', text: 'ok' }], usage: null },
    xSessionId: xid,
    parentSessionId: parent,
  });

  it('main session files under its OWN x-session-id (cpx-<ses>.jsonl)', () => {
    emit(mk('ses_MAIN_A', 'ses_MAIN_A', null, '第一问'));
    const rows = readJsonl(sessionFilePath('ses_MAIN_A'));
    expect(rows.some(r => r.type === 'user' && JSON.stringify(r.message.content).includes('第一问'))).toBe(true);
    expect(rows.some(r => r.type === 'assistant')).toBe(true);
  });

  it('child with x-parent-session-id files under the PARENT tree, own ses id as subId', () => {
    emit(mk('ses_CHILD_B', 'ses_CHILD_B', 'ses_MAIN_A', '子任务：reply ok'));
    const subPath = path.join(tmpDir, 'ses_MAIN_A', 'subagents', 'ses_CHILD_B.jsonl');
    expect(fs.existsSync(subPath)).toBe(true);
    expect(readJsonl(subPath).some(r => r.type === 'user')).toBe(true);
    expect(fs.existsSync(sessionFilePath('ses_CHILD_B'))).toBe(false); // never a top-level file
  });

  it('resume/fork to another main session starts fresh delta state (no cross-session skip)', () => {
    emit(mk('ses_MAIN_C', 'ses_MAIN_C', null, '新会话首问'));
    const users = readJsonl(sessionFilePath('ses_MAIN_C')).filter(r => r.type === 'user');
    expect(users.length).toBe(1);
    expect(JSON.stringify(users[0].message.content)).toContain('新会话首问');
  });
});
