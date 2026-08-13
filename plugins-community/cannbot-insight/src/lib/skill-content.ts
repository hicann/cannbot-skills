// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { parseReadLines } from './file-restore';

export interface SkillToolCall {
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  startedAt: Date | null;
}

export interface SkillContentResult {
  content: string;
  source: 'skill-tool' | 'read';
  length: number;
  /** 是否判定为读取了完整 SKILL.md。skill-tool 注入恒为 true；read 需无 offset/limit 且无截断标记。 */
  fullRead: boolean;
  /** read 来源时实际读到的最大行号（文件真实行号）；skill-tool 为 null。 */
  maxLine: number | null;
}

const SKILL_TOOL_NAMES = new Set(['skill', 'load_skill', 'skill/load_skill', 'skill/invoke']);

export function isSkillToolCall(toolName: string): boolean {
  const lower = toolName.toLowerCase();
  return lower.startsWith('skill/') || SKILL_TOOL_NAMES.has(lower);
}

export function extractSkillName(toolName: string, argsJson: string | null): string {
  if (!argsJson) return toolName.replace(/^skill\//i, '');
  try {
    const args = JSON.parse(argsJson);
    if (args.skill) return args.skill;
    if (args.skill_name) return args.skill_name;
    if (args.name) return args.name;
  } catch { /* ignore */ }
  return toolName.replace(/^skill\//i, '');
}

export function extractSkillNameFromReadPath(filePath: string): string | null {
  if (!filePath) return null;
  const norm = filePath.replace(/\\/g, '/');
  if (!norm.toLowerCase().endsWith('/skill.md')) return null;
  const parts = norm.split('/').filter(Boolean);
  if (parts.length < 2) return null;
  return parts[parts.length - 2];
}

export function stripSkillPreamble(content: string): string {
  if (content.startsWith('Base directory for this skill:')) {
    const nl = content.indexOf('\n');
    if (nl === -1) return '';
    const rest = content.slice(nl + 1);
    return rest.startsWith('\n') ? rest.slice(1) : rest;
  }
  return content;
}

function joinReadContent(resultJson: string): string {
  const m = parseReadLines(resultJson);
  const keys = [...m.keys()].sort((a, b) => a - b);
  return keys.map(k => m.get(k)!).join('\n');
}

function readMaxLine(resultJson: string): number | null {
  const m = parseReadLines(resultJson);
  if (m.size === 0) return null;
  let max = 0;
  for (const n of m.keys()) if (n > max) max = n;
  return max;
}

function hasReadOffsetOrLimit(argsJson: string | null): boolean {
  if (!argsJson) return false;
  try {
    const a = JSON.parse(argsJson);
    return (a.offset != null && a.offset > 0) || a.limit != null;
  } catch { return false; }
}

const TRUNCATION_RE = /(?:only\s+showing\s+(?:the\s+)?first\s+\d+\s+lines)|(?:lines?\s+(?:hidden|omitted|not\s+shown|truncated))|(?:content\s+(?:was\s+)?truncat)|(?:file\s+is\s+(?:large|too\s+large))/i;

function hasReadTruncation(resultJson: string | null): boolean {
  if (!resultJson) return false;
  return TRUNCATION_RE.test(resultJson);
}

export function selectSkillContent(
  toolCalls: SkillToolCall[],
  skillName: string
): SkillContentResult | null {
  const want = (n: string) => n === skillName;

  const skillToolCandidates: SkillToolCall[] = [];
  const readCandidates: SkillToolCall[] = [];

  for (const tc of toolCalls) {
    const lower = tc.toolName.toLowerCase();
    if (isSkillToolCall(tc.toolName)) {
      if (want(extractSkillName(tc.toolName, tc.argsJson)) && tc.resultJson) {
        skillToolCandidates.push(tc);
      }
      continue;
    }
    if (lower === 'read' && tc.argsJson && tc.resultJson) {
      try {
        const args = JSON.parse(tc.argsJson);
        const fp = String(args.file_path ?? args.filePath ?? '');
        const name = extractSkillNameFromReadPath(fp);
        if (name && want(name)) readCandidates.push(tc);
      } catch { /* ignore */ }
    }
  }

  if (skillToolCandidates.length > 0) {
    const best = skillToolCandidates.reduce((a, b) =>
      (b.resultJson?.length ?? 0) > (a.resultJson?.length ?? 0) ? b : a
    );
    const content = stripSkillPreamble(best.resultJson!);
    return { content, source: 'skill-tool', length: content.length, fullRead: true, maxLine: null };
  }

  if (readCandidates.length > 0) {
    const best = readCandidates.reduce((a, b) =>
      (b.resultJson?.length ?? 0) > (a.resultJson?.length ?? 0) ? b : a
    );
    const content = joinReadContent(best.resultJson!);
    const partial = hasReadOffsetOrLimit(best.argsJson) || hasReadTruncation(best.resultJson);
    return {
      content,
      source: 'read',
      length: content.length,
      fullRead: !partial,
      maxLine: readMaxLine(best.resultJson!),
    };
  }

  return null;
}
