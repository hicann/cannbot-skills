// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbot-proxy 扩展层：proxy 在标准 claude 行的 assistant 行上额外挂了
// 扩展字段 `system` / `tools`（verbatim request body），标准 claude-jsonl
// ingest 忽略它们。这里读取并组装 Full Context（System/Tools/Memory/Skills）。
// 依赖核心 claude-jsonl 的行解析 + message 扫描 helper，不改核心。proxy 以后
// 加新扩展字段（metadata/cache/retry/…）在本文件扩展，核心不动。

import {
  parseJsonlLines,
  extractSystemText,
  findMemorySection,
  findSkillsSection,
  proxySourceOfLines,
  type ClaudeJsonlLine,
  type ContentBlock,
} from './claude-jsonl';

export interface FullContextTool {
  name: string;
  description: string;
}
export interface FullContext {
  systemPrompt: string;
  tools: FullContextTool[];
  memoryFiles: string;
  skills: string;
}

// 扫 message content 取 memory(CLAUDE.md) / skills（claude-format 通用逻辑，
// 原始 claude jsonl 与 proxy 扩展都有）。
function scanMessageContext(lines: ClaudeJsonlLine[]): { memoryFiles: string; skills: string } {
  let memoryFiles = '';
  let skills = '';
  for (const line of lines) {
    const c = line.message?.content;
    if (typeof c === 'string') {
      if (!memoryFiles) memoryFiles = findMemorySection(c);
      if (!skills) skills = findSkillsSection(c);
    } else if (Array.isArray(c)) {
      for (const b of c) {
        if (b.type !== 'text' || !b.text) continue;
        if (!memoryFiles) memoryFiles = findMemorySection(b.text);
        if (!skills) skills = findSkillsSection(b.text);
      }
    }
  }
  return { memoryFiles, skills };
}

export function readFullContext(filePath: string): FullContext | null {
  const lines = parseJsonlLines(filePath);
  if (lines.length === 0) return null;

  // 扩展字段（proxy 捕获）：第一条带 tools 的 assistant 行。
  let systemPrompt = '';
  let tools: FullContextTool[] = [];
  let foundExt = false;
  for (const line of lines) {
    if (foundExt || line.type !== 'assistant') continue;
    const sysAny = (line as { system?: unknown }).system;
    const toolsAny = (line as { tools?: unknown }).tools;
    if (Array.isArray(toolsAny) && toolsAny.length > 0) {
      foundExt = true;
      systemPrompt = extractSystemText(sysAny as string | ContentBlock[] | undefined) ?? '';
      tools = (toolsAny as Array<{ name?: string; description?: string }>)
        .map(t => ({ name: t.name ?? '', description: t.description ?? '' }))
        .filter(t => t.name);
    }
  }

  const { memoryFiles, skills } = scanMessageContext(lines);
  if (!foundExt && !memoryFiles && !skills) return null;
  return { systemPrompt, tools, memoryFiles, skills };
}

// cpx 捕获文件自带来源标记：每行顶层 source:"claude-proxy" / "opencode-proxy"
// （保留原始 agent 名 + "-proxy" 后缀）。与路径无关——文件改名/移动/归档后
// 标记仍在。导入时取该值存入 Session.version，用于列表区分来源。
export function proxySourceOf(filePath: string): string | null {
  return proxySourceOfLines(parseJsonlLines(filePath));
}
