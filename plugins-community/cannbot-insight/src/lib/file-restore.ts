// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface RestoreOp {
  kind: 'read' | 'write';
  argsJson: string | null;
  resultJson: string | null;
  ts: Date | null;
}

export interface RestoreLine {
  n: number;
  content: string | null;
  source: 'read' | 'write' | 'gap';
}

export interface RestoreResult {
  lines: RestoreLine[];
  maxLine: number;
  opsUsed: number;
}

export function extractFilePath(argsJson: string | null): string | null {
  if (!argsJson) return null;
  try {
    const p = JSON.parse(argsJson);
    return p.filePath ?? p.file_path ?? null;
  } catch {
    return null;
  }
}

export function extractOffset(argsJson: string | null): number | undefined {
  if (!argsJson) return undefined;
  try {
    const p = JSON.parse(argsJson);
    return p.offset ?? undefined;
  } catch {
    return undefined;
  }
}

const NUMBERED_TAB = /^\s*(\d+)\t(.*)$/;
const NUMBERED_COLON = /^\s*(\d+): (.*)$/;
const CONTENT_OPEN = /<content>\s*$/;
const CONTENT_CLOSE = /<\/content>/;

export function parseReadLines(resultJson: string | null, offset?: number): Map<number, string> {
  const out = new Map<number, string>();
  if (!resultJson) return out;

  const raw = typeof resultJson === 'string' ? resultJson : String(resultJson);
  let lines = raw.split('\n');

  // Strip the opencode/Claude envelope: lines between <content> and </content>.
  // Envelope (<path>/<type>/<content>) must be excluded so it doesn't pollute
  // the restored file; the real content lives inside the <content> block.
  const openIdx = lines.findIndex(l => CONTENT_OPEN.test(l) || l.trim() === '<content>');
  if (openIdx >= 0) {
    const closeIdx = lines.findIndex(l => CONTENT_CLOSE.test(l));
    lines = lines.slice(openIdx + 1, closeIdx < 0 ? undefined : closeIdx);
  }
  if (lines.length === 0) return out;

  // Numbered parse: claude uses `N\tcontent`, opencode uses `N: content`.
  // Both carry the file's real line numbers, so trust them.
  let matchedAny = false;
  for (const line of lines) {
    const mt = line.match(NUMBERED_TAB);
    if (mt) { out.set(parseInt(mt[1], 10), mt[2]); matchedAny = true; continue; }
    const mc = line.match(NUMBERED_COLON);
    if (mc) { out.set(parseInt(mc[1], 10), mc[2]); matchedAny = true; continue; }
  }
  if (matchedAny) return out;

  // Fallback: plain content with no line-number prefix — number from offset.
  const start = offset && offset > 0 ? offset : 1;
  for (let i = 0; i < lines.length; i++) {
    out.set(start + i, lines[i]);
  }
  return out;
}

export function parseWriteLines(argsJson: string | null): Map<number, string> | null {
  if (!argsJson) return null;
  let parsed: unknown;
  try {
    parsed = JSON.parse(argsJson);
  } catch {
    return null;
  }
  if (typeof parsed !== 'object' || parsed === null) return null;
  const content = (parsed as { content?: unknown }).content;
  if (typeof content !== 'string') return null;

  const body = content.endsWith('\n') ? content.slice(0, -1) : content;
  const parts = body.split('\n');
  const out = new Map<number, string>();
  parts.forEach((line, i) => out.set(i + 1, line));
  return out;
}

export function restoreFile(ops: RestoreOp[]): RestoreResult {
  const sorted = [...ops].sort((a, b) => {
    const ta = a.ts?.getTime() ?? -Infinity;
    const tb = b.ts?.getTime() ?? -Infinity;
    return ta - tb;
  });

  const lineMap = new Map<number, { content: string; source: 'read' | 'write' }>();
  let opsUsed = 0;

  for (const op of sorted) {
    let lines: Map<number, string> | null = null;
    if (op.kind === 'read') {
      lines = parseReadLines(op.resultJson, extractOffset(op.argsJson));
    } else {
      lines = parseWriteLines(op.argsJson);
    }
    if (!lines || lines.size === 0) continue;
    opsUsed++;
    for (const [n, content] of lines) {
      lineMap.set(n, { content, source: op.kind });
    }
  }

  let maxLine = 0;
  for (const n of lineMap.keys()) if (n > maxLine) maxLine = n;

  const result: RestoreLine[] = [];
  for (let n = 1; n <= maxLine; n++) {
    const hit = lineMap.get(n);
    if (hit) result.push({ n, content: hit.content, source: hit.source });
    else result.push({ n, content: null, source: 'gap' });
  }

  return { lines: result, maxLine, opsUsed };
}

export function renderRestoredText(lines: RestoreLine[]): string {
  return lines
    .map(l => (l.content === null ? `--line ${l.n} not found --` : l.content))
    .join('\n');
}
