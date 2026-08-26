// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Standalone test for opencode-context-parser: verify it parses the opencode
// system's native structure (instructions / `Instructions from:` memory /
// `<available_skills>` skills / tools) from a proxy-captured jsonl. The fixture
// is the same opencode-wire-records.jsonl the emitter test uses (emitter writes
// the system VERBATIM; this parser interprets it).

import { describe, it, expect, beforeAll } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { emit } from '../src/opencode-emitter.ts';
import { sessionFilePath } from '../src/writer.ts';
import { parseOpencodeContext, parseOpencodeSystem } from '../src/opencode-context-parser.ts';

const SID = 'sid-main';
const FIXTURE = path.resolve(__dirname, 'data/opencode-wire-records.jsonl');
const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cpx-ctx-'));
process.env.CANNBOT_PROXY_DIR = tmpDir; // BEFORE sessionFilePath is evaluated at module scope
const mainFile = sessionFilePath(SID); // cpx- prefixed

beforeAll(() => {
  const lines = fs.readFileSync(FIXTURE, 'utf-8').split('\n').filter(l => l.trim());
  for (const line of lines) {
    try { emit(JSON.parse(line)); } catch { /* skip */ }
  }
});

describe('parseOpencodeSystem (unit)', () => {
  const sys = 'You are opencode.\n\n# Tone\nBe concise.\n\nInstructions from: /p/AGENTS.md\n# Project Rules\nFollow conventions.\n\n<available_skills>\n<skill>\n<name>review</name>\n<description>Reviews skills.</description>\n<location>file:///x/SKILL.md</location>\n</skill>\n</available_skills>';

  it('instructions = persona before injected context', () => {
    const { instructions } = parseOpencodeSystem(sys);
    expect(instructions).toContain('You are opencode');
    expect(instructions).toContain('# Tone');
    expect(instructions).not.toContain('Instructions from');
    expect(instructions).not.toContain('<available_skills>');
  });

  it('memory = parsed Instructions-from blocks [{path, content}]', () => {
    const { memory } = parseOpencodeSystem(sys);
    expect(memory).toHaveLength(1);
    expect(memory[0].path).toBe('/p/AGENTS.md');
    expect(memory[0].content).toContain('# Project Rules');
    expect(memory[0].content).toContain('Follow conventions.');
  });

  it('skills = parsed [{name, description, location}]', () => {
    const { skills } = parseOpencodeSystem(sys);
    expect(skills).toHaveLength(1);
    expect(skills[0].name).toBe('review');
    expect(skills[0].description).toBe('Reviews skills.');
    expect(skills[0].location).toBe('file:///x/SKILL.md');
  });

  it('handles absence: no Instructions-from, no available_skills', () => {
    const r = parseOpencodeSystem('You are a subagent. Focus on exploring.');
    expect(r.instructions).toBe('You are a subagent. Focus on exploring.');
    expect(r.memory).toEqual([]);
    expect(r.skills).toEqual([]);
  });
});

describe('parseOpencodeContext (jsonl file)', () => {
  it('parses the first assistant line carrying a verbatim system', () => {
    const ctx = parseOpencodeContext(mainFile);
    expect(ctx).not.toBeNull();
    expect(ctx!.systemRaw).toContain('You are opencode');
    // system is VERBATIM — the injected markers are still in systemRaw (capture ≠ interpret)
    expect(ctx!.systemRaw).toContain('Instructions from: /home/user/AGENTS.md');
    expect(ctx!.systemRaw).toContain('<available_skills>');
  });

  it('instructions excludes the injected memory/skills', () => {
    const ctx = parseOpencodeContext(mainFile)!;
    expect(ctx.instructions).toContain('You are opencode');
    expect(ctx.instructions).toContain('# Tone and style');
    expect(ctx.instructions).not.toContain('Instructions from');
    expect(ctx.instructions).not.toContain('<available_skills>');
  });

  it('memory = AGENTS.md [{path, content}]', () => {
    const ctx = parseOpencodeContext(mainFile)!;
    expect(ctx.memory).toHaveLength(1);
    expect(ctx.memory[0].path).toBe('/home/user/AGENTS.md');
    expect(ctx.memory[0].content).toContain('# Project Rules');
    expect(ctx.memory[0].content).toContain('Follow CANNBot conventions.');
  });

  it('skills = [{name, description}]', () => {
    const ctx = parseOpencodeContext(mainFile)!;
    expect(ctx.skills.map(s => s.name).sort()).toEqual(['cannbot-skill-review', 'customize-opencode']);
    expect(ctx.skills.every(s => typeof s.description === 'string')).toBe(true);
  });

  it('tools = [{name, description}] from the tools extension field', () => {
    const ctx = parseOpencodeContext(mainFile)!;
    expect(ctx.tools.map(t => t.name).sort()).toEqual(['bash', 'read']);
  });

  it('returns null when no assistant line carries a system', () => {
    const empty = path.join(tmpDir, 'no-context.jsonl');
    fs.writeFileSync(empty, '{"type":"user","message":{"role":"user","content":"hi"}}\n');
    expect(parseOpencodeContext(empty)).toBeNull();
  });
});
