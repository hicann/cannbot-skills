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
  // x_cannbay.data（cc-wire-round）优先，legacy 顶层 system/tools fallback。
  let systemPrompt = '';
  let tools: FullContextTool[] = [];
  let foundExt = false;
  for (const line of lines) {
    if (foundExt || line.type !== 'assistant') continue;
    const xb = (line as { x_cannbay?: { data?: { system?: unknown; tools?: unknown } } }).x_cannbay;
    const sysAny = xb?.data?.system ?? (line as { system?: unknown }).system;
    const toolsAny = xb?.data?.tools ?? (line as { tools?: unknown }).tools;
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
  const lines = parseJsonlLines(filePath);
  for (const l of lines) {
    const s = (l as { source?: unknown }).source;
    if (typeof s === 'string' && s.endsWith('-proxy')) return s;
  }
  return null;
}

// post-patch 读 cc-wire-round.requestParams（spec §4.1 导入归宿 "post-patch"）：
// Turn schema 无 temperature/maxTokens 列（zero-schema-change 约束），proxy
// 捕获的请求参数只存在 jsonl 的 x_cannbay.data 里，按需在 turns API 叠加 ——
// 与 readFullContext 同构（API 层调扩展层读 proxy 数据，核心管线不感知）。
export interface TurnRequestParams {
  temperature: number | null;
  maxTokens: number | null;
  model: string | null;
}

export function readTurnRequestParams(filePath: string, turnIndex: number): TurnRequestParams | null {
  const lines = parseJsonlLines(filePath);
  // turnIndex 指向 DB 里的 turn 序号，proxy jsonl 里 assistant 行按出现顺序
  // 对应 turnIndex（assistant 行 = 每个 wire 轮的响应）。roundIndex 也从 0 起，
  // 逐轮 +1，与 turnIndex 对齐 —— 直接按 assistant 行计数定位。
  let assistantSeen = -1;
  for (const l of lines) {
    if (l.type !== 'assistant') continue;
    assistantSeen++;
    if (assistantSeen !== turnIndex) continue;
    const xb = (l as { x_cannbay?: { schema?: string; version?: number; data?: { requestParams?: { temperature?: number; maxTokens?: number; model?: string | null } } } }).x_cannbay;
    // version 门禁（spec §6）：只认 cc-wire-round v1
    if (!xb || xb.schema !== 'cc-wire-round' || xb.version !== 1) continue;
    const rp = xb.data?.requestParams;
    return {
      temperature: rp?.temperature ?? null,
      maxTokens: rp?.maxTokens ?? null,
      model: rp?.model ?? null,
    };
  }
  return null;
}
