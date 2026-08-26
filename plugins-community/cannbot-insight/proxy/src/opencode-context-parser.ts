// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// opencode-context-parser: standalone, opencode-aware parser for the proxy's
// captured jsonl. Reads the `system` extension field (the opencode system,
// captured VERBATIM by opencode-emitter) from the first assistant line and
// parses opencode's native structure into:
//   instructions — opencode built-in persona (everything before injected context)
//   memory       — `Instructions from: <path>` blocks (AGENTS.md etc.) → [{path, content}]
//   skills       — `<available_skills><skill>…</skill>` block → [{name, description, location}]
//   tools        — the `tools` extension field → [{name, description}]
//
// Design: capture ≠ interpret. The proxy is a pure verbatim capture layer; this
// parser owns opencode-specific structural parsing. jsonl-driven: the jsonl is the
// single source of truth and is re-parseable anytime with updated logic (the raw
// `systemRaw` is always preserved). Independent of cannbot-insight (no insight
// import/DB dependency) — a future merge would call parseOpencodeContext from an
// opencode branch of readFullContext.

import fs from 'node:fs';

export interface MemoryFile { path: string; content: string; }
export interface Skill { name: string; description: string; location?: string; }
export interface Tool { name: string; description: string; }
export interface OpencodeContext {
  systemRaw: string;        // full verbatim opencode system (for reference / re-parse)
  instructions: string;     // opencode built-in persona (before injected context)
  memory: MemoryFile[];     // `Instructions from:` blocks, parsed
  skills: Skill[];          // `<available_skills>` block, parsed
  tools: Tool[];            // from the `tools` extension field
}

interface JsonlLine {
  type: string;
  system?: string;
  tools?: Array<{ name?: string; description?: string }>;
}

function parseLines(filePath: string): JsonlLine[] {
  const content = fs.readFileSync(filePath, 'utf-8');
  const out: JsonlLine[] = [];
  for (const l of content.split('\n')) {
    const t = l.trim();
    if (!t || t.startsWith('//')) continue;
    try { out.push(JSON.parse(t)); } catch { /* skip malformed */ }
  }
  return out;
}

// Split the verbatim opencode system into its three native sections.
export function parseOpencodeSystem(text: string): { instructions: string; memory: MemoryFile[]; skills: Skill[] } {
  const memIdx = text.indexOf('Instructions from:');
  const skillsIdx = text.indexOf('<available_skills>');
  // boundary = earliest injected-context marker; instructions = everything before it
  let boundary = -1;
  if (memIdx >= 0 && (skillsIdx < 0 || memIdx < skillsIdx)) boundary = memIdx;
  else if (skillsIdx >= 0) boundary = skillsIdx;
  const instructions = boundary >= 0 ? text.slice(0, boundary).trimEnd() : text;

  // memory: each `Instructions from: <path>` block → {path, content}. Content runs
  // from after the path line to the next `Instructions from:` or `<available_skills>`.
  const memory: MemoryFile[] = [];
  if (memIdx >= 0) {
    const memEnd = skillsIdx >= 0 && skillsIdx > memIdx ? skillsIdx : text.length;
    const memSeg = text.slice(memIdx, memEnd);
    for (const part of memSeg.split(/(?=Instructions from:)/)) {
      const m = part.match(/^Instructions from:\s*([^\n]+)\n?([\s\S]*)$/);
      if (m) memory.push({ path: m[1].trim(), content: m[2].trim() });
    }
  }

  // skills: <available_skills>…</available_skills> → [{name, description, location}]
  const skills: Skill[] = [];
  if (skillsIdx >= 0) {
    const endTag = text.indexOf('</available_skills>', skillsIdx);
    const block = text.slice(skillsIdx, endTag >= 0 ? endTag + '</available_skills>'.length : text.length);
    const re = /<skill>([\s\S]*?)<\/skill>/g;
    let sm: RegExpExecArray | null;
    while ((sm = re.exec(block)) !== null) {
      const inner = sm[1];
      const name = inner.match(/<name>([\s\S]*?)<\/name>/)?.[1]?.trim() ?? '';
      const desc = inner.match(/<description>([\s\S]*?)<\/description>/)?.[1]?.trim() ?? '';
      const loc = inner.match(/<location>([\s\S]*?)<\/location>/)?.[1]?.trim();
      if (name) skills.push({ name, description: desc, ...(loc ? { location: loc } : {}) });
    }
  }
  return { instructions, memory, skills };
}

// Parse the opencode system context from a proxy-captured jsonl file. Reads the
// first assistant line carrying a `system` field (opencode-emitter writes the
// verbatim system there). Returns null if no such line / no system. Works for
// both main (`<sid>.jsonl`) and subagent (`<sid>/subagents/<subId>.jsonl`) files.
export function parseOpencodeContext(filePath: string): OpencodeContext | null {
  const asst = parseLines(filePath).find(l => l.type === 'assistant' && typeof l.system === 'string');
  if (!asst || !asst.system) return null;
  const { instructions, memory, skills } = parseOpencodeSystem(asst.system);
  const tools: Tool[] = Array.isArray(asst.tools)
    ? asst.tools.map(t => ({ name: t.name ?? '', description: t.description ?? '' })).filter(t => t.name)
    : [];
  return { systemRaw: asst.system, instructions, memory, skills, tools };
}

// CLI: npx tsx opencode-context-parser.ts <jsonl-path> [--memory|--skills|--instructions|--tools]
// default (no flag): print the full OpencodeContext as JSON.
function main(): void {
  const args = process.argv.slice(2);
  const file = args.find(a => !a.startsWith('-'));
  const flag = args.find(a => a.startsWith('--'));
  if (!file) {
    console.error('usage: opencode-context-parser.ts <jsonl-path> [--memory|--skills|--instructions|--tools]');
    process.exit(2);
  }
  const ctx = parseOpencodeContext(file);
  if (!ctx) { console.error(`no opencode system context found in ${file}`); process.exit(1); }
  switch (flag) {
    case '--memory':
      for (const m of ctx.memory) { console.log(`===== ${m.path} =====`); console.log(m.content); }
      break;
    case '--skills': console.log(JSON.stringify(ctx.skills, null, 2)); break;
    case '--instructions': console.log(ctx.instructions); break;
    case '--tools': console.log(JSON.stringify(ctx.tools, null, 2)); break;
    default: console.log(JSON.stringify(ctx, null, 2));
  }
}

if (process.argv[1]?.endsWith('opencode-context-parser.ts')) main();
