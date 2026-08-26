// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Proxy 捕获文件（每行带 source:"-proxy" 标记）的扩展层 —— 与
// claude-jsonl-full-context.ts 同构：核心管线（readSession → normalize →
// turn-split → data-service）完全不感知 proxy 的 verbatim 数据。proxy 特有
// 的 contentJson（输入 turn 的 verbatim 消息）和 inputMessagesJson（输出
// turn 的累积 wire 请求）由本模块在 API 层按需读取，不注入管线。
//
// readWireRounds → 标准 RawInteraction[]（无扩展字段），走管线
// readWireEnrichments → Map<wireEnrichmentKey, {contentJson, inputMessagesJson}>，
//   key = `${role}:${createdAtMs}`（管线不变的稳定身份，不靠数组下标），
//   API 层调用（像 readFullContext），管线不感知。
//
// 为何不按下标：buildWireRounds 产出的 interaction 数组下标 ≠ 入库 turnIndex
// —— 管线（turn-split）在 compact 边界会折叠/丢弃连续同 role interaction
// （pre-compact 尾部 user 输入被 compact 吸收），DB turn 数 < 数组长度。按
// 下标建键会在 compact 后漂移，output turn 拿到 input 的 contentJson。
// `createdAt_ts` 经管线透传不变（已验证），用 (role, createdAtMs) 作稳定键。

import type { RawInteraction } from '../../shared/types';
import fs from 'node:fs';
import {
  type ClaudeJsonlLine,
  type ContentBlock,
  type AssistantGroup,
  parseJsonlLines,
  proxySourceOfLines,
  buildAssistantInteraction,
  collectAllToolResults,
  extractTextContent,
  isValidISO,
  stripSystemReminders,
} from './claude-jsonl';

interface WireMsg {
  role: string;
  content: string | ContentBlock[];
  timestamp: string | null;
}

function estimateTokens(chars: number): number {
  return Math.round(chars / 3.5);
}

function stringifyContent(content: string | ContentBlock[]): string {
  return typeof content === 'string' ? content : JSON.stringify(content, null, 0);
}

// tool_use_id → tool name（assistant 行的 tool_use 块），给 tool_result 消息补显示名
function collectToolUseNames(lines: ClaudeJsonlLine[]): Map<string, string> {
  const names = new Map<string, string>();
  for (const line of lines) {
    if (line.type === 'assistant' && Array.isArray(line.message?.content)) {
      for (const block of line.message.content as ContentBlock[]) {
        if (block.type === 'tool_use' && block.id && block.name) {
          names.set(block.id, block.name);
        }
      }
    }
  }
  return names;
}

// 单条 wire 消息 → LlmContextView 渲染条目（保持 wire 顺序，不折叠不重排）。
// assistant 消息带 tool_calls（result:null —— wire 上 result 是下一条
// tool_result 消息，不属于 assistant 消息本身）。
function wireMsgToDisplay(m: WireMsg, toolNames: Map<string, string>): Record<string, unknown> {
  const tokenCount = estimateTokens(JSON.stringify(m.content).length);
  if (m.role === 'user') {
    if (typeof m.content === 'string') {
      return { role: 'user', content: m.content, tokenCount };
    }
    const blocks = m.content;
    const results = blocks.filter(b => b.type === 'tool_result');
    if (results.length > 0 && results.length === blocks.length) {
      const contents = results.map(r => (typeof r.content === 'string' ? r.content : JSON.stringify(r.content ?? '')));
      return {
        role: 'tool_result',
        content: contents.join('\n---\n\n'),
        name: results.length === 1 ? (toolNames.get(results[0].tool_use_id ?? '') ?? null) : null,
        tokenCount,
      };
    }
    const text = extractTextContent(blocks);
    return { role: 'user', content: text ?? JSON.stringify(blocks, null, 0), tokenCount };
  }
  if (m.role === 'assistant') {
    const blocks = Array.isArray(m.content) ? m.content : [];
    const tool_calls: Array<{ name: string; args: string | null; result: string | null }> = blocks
      .filter(b => b.type === 'tool_use')
      .map(b => ({ name: b.name ?? '', args: b.input ? JSON.stringify(b.input) : null, result: null }));
    const entry: Record<string, unknown> = {
      role: 'assistant',
      content: extractTextContent(blocks) ?? JSON.stringify(blocks, null, 0),
      tokenCount,
    };
    if (tool_calls.length > 0) entry.tool_calls = tool_calls;
    return entry;
  }
  // system（注入的注册表 / skills / [已压缩] 标记行等）
  return {
    role: 'system',
    content: extractTextContent(m.content as ContentBlock[] | string) ?? stringifyContent(m.content),
    tokenCount,
  };
}

// 输入 turn 的展示文本：优先取本轮真正的用户输入（去 system-reminder），
// 否则给消息构成摘要 —— 保证 session label / turn 列表预览可读。
function inputDisplayContent(msgs: WireMsg[]): string {
  const counts = { user: 0, system: 0, tool_result: 0, other: 0 };
  for (const m of msgs) {
    if (m.role === 'system') counts.system++;
    else if (m.role === 'user') {
      if (Array.isArray(m.content) && m.content.every(b => b.type === 'tool_result')) counts.tool_result++;
      else counts.user++;
    } else counts.other++;
  }
  for (const m of msgs) {
    if (m.role !== 'user') continue;
    const text = extractTextContent(m.content as string | ContentBlock[]);
    if (!text) continue;
    const stripped = stripSystemReminders(text);
    if (stripped) return stripped;
  }
  const parts: string[] = [];
  if (counts.user > 0) parts.push(`user ×${counts.user}`);
  if (counts.tool_result > 0) parts.push(`tool_result ×${counts.tool_result}`);
  if (counts.system > 0) parts.push(`system ×${counts.system}`);
  if (counts.other > 0) parts.push(`${counts.other} 条其他`);
  return `（本轮输入 ${msgs.length} 条：${parts.join(' · ')}）`;
}

export interface WireTurnEnrichment {
  contentJson: string | null;
  inputMessagesJson: string | null;
  // ttftMs only (output turn); no pipeline field exists for it, so it stays
  // an API-layer override. latency/finishReason now flow through the standard
  // pipeline (emitter writes duration_ms + stopReason → adapter → turn-split
  // → DB), so they are NOT in this enrichment.
  ttftMs: number | null;
}

// enrichment Map 的稳定键：role + createdAt 毫秒。buildWireRounds 建键与
// turns API 查键都走本函数，避免字符串格式漂移。createdAt_ms 经管线
// （normalize → turn-split → DB Turn.createdAt_ts）透传不变，故同一条
// interaction 在 buildWireRounds 与 DB 两侧算出同一 key。
export function wireEnrichmentKey(role: string, createdAtMs: number): string {
  return `${role}:${createdAtMs}`;
}

// 内部 builder：一次性产出标准 interactions + enrichment map。
// 两个公共函数（readWireRounds / readWireEnrichments）共用此逻辑，
// 避免重复解析文件 + 重复 round-pair 切分。
function buildWireRounds(lines: ClaudeJsonlLine[], fileMtime: number): {
  interactions: RawInteraction[];
  enrichments: Map<string, WireTurnEnrichment>;
} {
  const allToolResults = collectAllToolResults(lines);
  const toolNames = collectToolUseNames(lines);

  const interactions: RawInteraction[] = [];
  const enrichments = new Map<string, WireTurnEnrichment>();

  const history: WireMsg[] = [];
  let pendingNew: WireMsg[] = [];
  let pendingLines: ClaudeJsonlLine[] | null = null;
  let pendingStartIdx = 0;

  const closeRound = () => {
    if (!pendingLines) return;
    if (pendingNew.length > 0) {
      const inputTs = pendingNew.find(m => m.timestamp)?.timestamp ?? null;
      const inputCreatedMs = inputTs && isValidISO(inputTs) ? new Date(inputTs).getTime() : fileMtime;
      interactions.push({
        role: 'user',
        content: inputDisplayContent(pendingNew),
        timestamp: inputTs ?? new Date(fileMtime).toISOString(),
        timeInfo: { created: inputCreatedMs },
        agent: null, subagent_name: null, subagent_session_id: null, subagent_type: null,
        tool_calls: null, usage: null, model: null, modelID: null, providerID: null,
        latency: null, finish_reason: null,
      });
      // enrichment: 输入 turn 的 verbatim 消息数组（API 层按需读取）
      enrichments.set(wireEnrichmentKey('user', inputCreatedMs), {
        contentJson: JSON.stringify({
          wireInput: true,
          messages: pendingNew.map(m => ({ role: m.role, content: m.content })),
        }),
        inputMessagesJson: null,
        ttftMs: null,
      });
    }
    const group: AssistantGroup = { lines: pendingLines, startLineIndex: pendingStartIdx };
    const output = buildAssistantInteraction(group, allToolResults, fileMtime);
    if (output) {
      interactions.push(output);
      // enrichment: 输出 turn 的累积 wire 请求 verbatim（API 层按需读取）
      const outputCreatedMs = output.timeInfo?.created ?? fileMtime;
      // ttftMs 从 assistant 组首行读扩展字段（latency/finishReason 已走管线）。
      const aLine = pendingLines[0];
      enrichments.set(wireEnrichmentKey(output.role ?? 'assistant', outputCreatedMs), {
        contentJson: null,
        inputMessagesJson: JSON.stringify(history.map(m => wireMsgToDisplay(m, toolNames))),
        ttftMs: aLine?.ttftMs ?? null,
      });
      history.push({
        role: 'assistant',
        content: group.lines.flatMap(l => (Array.isArray(l.message?.content) ? l.message.content as ContentBlock[] : [])),
        timestamp: group.lines[0]?.timestamp ?? null,
      });
    }
    pendingLines = null;
    pendingNew = [];
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.type === 'assistant' && line.message) {
      if (!pendingLines) {
        pendingLines = [];
        pendingStartIdx = i;
      }
      pendingLines.push(line);
      continue;
    }
    if (line.type === 'user' || line.type === 'system') {
      if (!line.message) continue;
      // 先关闭当前 round（flush 待处理的 assistant group）—— 此时 history
      // 仍含 compact 前的完整累积，output turn 的 inputMessagesJson 才正确
      closeRound();
      // /compact 边界：claude-code 用 continuation 摘要替换整个 messages
      // 数组，旧历史消失。检测到时重置 history。continuation 文本可能在
      // 第二个 text block（第一个是 system-reminder），逐 block 用 startsWith 判断
      const blocks = Array.isArray(line.message.content)
        ? (line.message.content as ContentBlock[])
        : [];
      const isCompact = line.type === 'user' && blocks.some(
        b => b.type === 'text' && typeof b.text === 'string' &&
          b.text.startsWith('This session is being continued from a previous conversation')
      );
      if (isCompact) {
        history.length = 0;
        pendingNew = [];
      }
      const msg: WireMsg = {
        role: line.message.role ?? line.type,
        content: line.message.content ?? '',
        timestamp: line.timestamp ?? null,
      };
      history.push(msg);
      pendingNew.push(msg);
    }
  }
  closeRound();

  return { interactions, enrichments };
}

// 管线入口：产出标准 RawInteraction[]（无扩展字段）。
// readSession 检测到 proxy 文件（source:"-proxy"）时调用本函数替代原生切分。
// 产出的 interaction 走标准 normalize → turn-split → data-service 管线，
// 管线不感知 proxy 的 verbatim 数据。
export function readWireRounds(lines: ClaudeJsonlLine[], fileMtime: number): RawInteraction[] {
  return buildWireRounds(lines, fileMtime).interactions;
}

// 扩展层入口：按需读取 proxy turn 的 verbatim 数据（像 readFullContext）。
// 返回 Map<wireEnrichmentKey, {contentJson, inputMessagesJson}>，key 由
// wireEnrichmentKey(role, createdAtMs) 构造。turns detail API 用同一 helper
// 以 turn.role + turn.createdAt_ts 查键，为 proxy turn 补充
// contentJson/inputMessagesJson，不经过管线。
// 非 proxy 文件（无 source:"-proxy" 标记）返回空 map —— 不对原生文件跑 wire split。
export function readWireEnrichments(filePath: string): Map<string, WireTurnEnrichment> {
  const lines = parseJsonlLines(filePath);
  if (lines.length === 0) return new Map();
  if (!proxySourceOfLines(lines)) return new Map();
  const fileMtime = fs.statSync(filePath).mtime.getTime();
  return buildWireRounds(lines, fileMtime).enrichments;
}
