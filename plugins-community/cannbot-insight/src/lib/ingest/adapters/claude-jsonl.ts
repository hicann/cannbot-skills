// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from 'node:fs';
import path from 'node:path';
import type { SessionListItem, RawInteraction, ToolCallInfo, TokenUsage } from '../../shared/types';

interface ClaudeJsonlLine {
  type: string;
  timestamp?: string;
  version?: string;
  message?: {
    id?: string;
    role: string;
    content?: string | ContentBlock[];
    model?: string;
    usage?: ClaudeUsage;
  };
  subtype?: string;
  result?: string;
  cost_usd?: number;
  duration_ms?: number;
}

interface ContentBlock {
  type: string;
  text?: string;
  thinking?: string;
  id?: string;
  name?: string;
  input?: Record<string, unknown>;
  tool_use_id?: string;
  content?: string | Record<string, unknown> | unknown[];
}

interface ClaudeUsage {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens?: number;
  cache_creation_input_tokens?: number;
}

function extractTextContent(content: string | ContentBlock[] | undefined): string | null {
  if (!content) return null;
  if (typeof content === 'string') return content;
  const parts: string[] = [];
  for (const block of content) {
    if (block.type === 'text' && block.text) {
      parts.push(block.text);
    } else if (block.type === 'thinking' && block.thinking) {
      parts.push(`<thinking>${block.thinking}</thinking>`);
    }
  }
  return parts.length > 0 ? parts.join('\n') : null;
}

function extractToolCalls(content: ContentBlock[]): ToolCallInfo[] | null {
  const toolUseBlocks = content.filter(b => b.type === 'tool_use');
  if (toolUseBlocks.length === 0) return null;

  const toolResultBlocks = content.filter(b => b.type === 'tool_result');
  const resultMap = new Map<string, string>();
  for (const r of toolResultBlocks) {
    if (r.tool_use_id && r.content) {
      const val = r.content;
      resultMap.set(r.tool_use_id, typeof val === 'string' ? val : JSON.stringify(val));
    }
  }

  return toolUseBlocks.map(b => ({
    toolCallId: b.id ?? '',
    toolName: b.name ?? '',
    argsJson: b.input ? JSON.stringify(b.input) : null,
    resultJson: (() => { const v = resultMap.get(b.id ?? '') ?? null; return v == null ? null : (typeof v === 'string' ? v : JSON.stringify(v)); })(),
    state: 'completed',
  }));
}

function mapUsage(usage: ClaudeUsage | undefined, costUsd?: number): TokenUsage | null {
  if (!usage) return null;
  const input = usage.input_tokens ?? 0;
  const cacheRead = usage.cache_read_input_tokens ?? 0;
  const cacheWrite = usage.cache_creation_input_tokens ?? 0;
  return {
    total: input + (usage.output_tokens ?? 0) + cacheRead + cacheWrite,
    input,
    output: usage.output_tokens ?? 0,
    reasoning: 0,
    cacheRead,
    cacheWrite,
    cost: costUsd ?? 0,
    inputMessagesTokens: input + cacheRead + cacheWrite,
  };
}

function deriveSessionId(filePath: string): string {
  const basename = path.basename(filePath, '.jsonl');
  return basename;
}

function parseJsonlLines(filePath: string): ClaudeJsonlLine[] {
  try {
    const content = fs.readFileSync(filePath, 'utf-8');
    if (!content.trim()) return [];
    const lines = content.split('\n').filter(l => l.trim());
    const result: ClaudeJsonlLine[] = [];
    for (const line of lines) {
      try {
        result.push(JSON.parse(line) as ClaudeJsonlLine);
      } catch {
        console.warn(`claude-jsonl: skipping malformed JSON line in ${filePath}`);
      }
    }
    return result;
  } catch {
    return [];
  }
}

function collectAllToolResults(lines: ClaudeJsonlLine[]): Map<string, string> {
  const resultMap = new Map<string, string>();
  for (const line of lines) {
    if (line.type === 'user' && line.message?.content && Array.isArray(line.message.content)) {
      for (const block of line.message.content as ContentBlock[]) {
        if (block.type === 'tool_result' && block.tool_use_id && block.content) {
          const val = block.content;
          resultMap.set(block.tool_use_id, typeof val === 'string' ? val : JSON.stringify(val));
        }
      }
    }
  }
  return resultMap;
}

function collectJsonlFiles(dirPath: string): string[] {
  const results: string[] = [];
  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      const full = path.join(dirPath, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === 'subagents') continue;
        results.push(...collectJsonlFiles(full));
      } else if (entry.name.endsWith('.jsonl')) {
        results.push(full);
      }
    }
  } catch {}
  return results;
}

export function listSessions(dirPath: string): SessionListItem[] {
  if (!dirPath || !fs.existsSync(dirPath)) return [];

  let stat = fs.statSync(dirPath);
  let files: string[];

  if (stat.isFile() && dirPath.endsWith('.jsonl')) {
    files = [dirPath];
  } else if (stat.isDirectory()) {
    files = collectJsonlFiles(dirPath);
  } else {
    return [];
  }

  const result: SessionListItem[] = [];
  for (const file of files) {
    const sessionId = deriveSessionId(file);
    const lines = parseJsonlLines(file);
    if (lines.length === 0) continue;

    let firstQuery: string | null = null;
    let modelName: string | null = null;
    let createdAt: string;

    try {
      createdAt = fs.statSync(file).mtime.toISOString();
    } catch {
      createdAt = new Date(0).toISOString();
    }

    for (const line of lines) {
      if (line.type === 'user' && line.message) {
        const text = extractTextContent(line.message.content);
        if (text && !firstQuery) {
          firstQuery = text.substring(0, 200);
        }
      }
      if (line.type === 'assistant' && line.message?.model) {
        if (!modelName) {
          modelName = line.message.model;
        }
      }
    }

    result.push({
      id: sessionId,
      createdAt,
      firstQuery,
      turnCount: lines.length,
      modelName,
    });
  }

  return result.sort((a, b) => b.createdAt.localeCompare(a.createdAt));
}

function findSessionFile(dirPath: string, sessionId: string): string | null {
  const directPath = path.join(dirPath, sessionId + '.jsonl');
  if (fs.existsSync(directPath) && fs.statSync(directPath).isFile()) return directPath;

  try {
    const entries = fs.readdirSync(dirPath, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isDirectory() && entry.name !== 'subagents') {
        const found = findSessionFile(path.join(dirPath, entry.name), sessionId);
        if (found) return found;
      }
    }
  } catch {}

  return null;
}

export function listSubagentSessions(dirPath: string, sessionId: string): { id: string, filePath: string }[] {
  // dirPath can be a file (e.g., /path/to/session.jsonl) or a directory.
  // Subagents live at <parentDir>/sessionId/subagents/
  const parentDir = fs.statSync(dirPath).isFile() ? path.dirname(dirPath) : dirPath;
  const subagentsDir = path.join(parentDir, sessionId, 'subagents');
  if (!fs.existsSync(subagentsDir) || !fs.statSync(subagentsDir).isDirectory()) return [];

  const results: { id: string, filePath: string }[] = [];
  try {
    const entries = fs.readdirSync(subagentsDir, { withFileTypes: true });
    for (const entry of entries) {
      if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        const subId = path.basename(entry.name, '.jsonl');
        results.push({ id: subId, filePath: path.join(subagentsDir, entry.name) });
      }
    }
  } catch {}

  return results;
}

export function collectSubagentToolUseMappings(dirPath: string, sessionId: string): Map<string, string> {
  // Maps toolUseId (from meta.json) → subagent session ID (from .jsonl filename)
  const mapping = new Map<string, string>();
  const subagentFiles = listSubagentSessions(dirPath, sessionId);
  for (const sub of subagentFiles) {
    const metaPath = sub.filePath.replace('.jsonl', '.meta.json');
    try {
      if (fs.existsSync(metaPath)) {
        const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
        if (meta.toolUseId) {
          mapping.set(meta.toolUseId, sub.id);
        }
      }
    } catch {}
  }
  return mapping;
}

function isValidISO(timestamp: string): boolean {
  return /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/.test(timestamp);
}

function extractVersionFromLines(lines: ClaudeJsonlLine[]): string | null {
  for (const line of lines) {
    if (line.version) return line.version;
  }
  return null;
}

export function extractVersion(filePath: string): string | null {
  const lines = parseJsonlLines(filePath);
  return extractVersionFromLines(lines);
}

interface AssistantGroup {
  lines: ClaudeJsonlLine[];
  startLineIndex: number;
}

// Non-substantive line types that shouldn't break assistant grouping
// These are Claude Code metadata lines injected during streaming
const NON_BREAKING_TYPES = new Set([
  'ai-title', 'attachment', 'mode', 'permission-mode',
  'file-history-snapshot', 'last-prompt', 'system',
]);

function groupAssistantLines(lines: ClaudeJsonlLine[]): AssistantGroup[] {
  const groups: AssistantGroup[] = [];
  let current: AssistantGroup | null = null;
  let currentMsgIds: Set<string> | null = null;
  let pendingToolResults: ClaudeJsonlLine[] | null = null;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    if (line.type === 'assistant' && line.message) {
      const msgId = line.message?.id;

      if (current && pendingToolResults && msgId && currentMsgIds?.has(msgId)) {
        // This assistant line shares the same message.id as the current group,
        // meaning the interleaved tool_result lines belong to the same API response.
        // Absorb them without breaking the group.
        pendingToolResults = null;
      } else if (pendingToolResults) {
        // The next assistant line has a different message.id (or no id),
        // so the pending tool_results are from a different turn. End the group.
        groups.push(current!);
        current = null;
        currentMsgIds = null;
        pendingToolResults = null;
      }

      if (!current) {
        current = { lines: [], startLineIndex: i };
        currentMsgIds = new Set();
      }
      current.lines.push(line);
      if (msgId && currentMsgIds) currentMsgIds.add(msgId);
    } else if (NON_BREAKING_TYPES.has(line.type)) {
      // Don't break the current assistant group for non-substantive lines
      continue;
    } else if (
      current &&
      line.type === 'user' &&
      line.message?.content &&
      Array.isArray(line.message.content) &&
      extractTextContent(line.message.content) === null
    ) {
      // User line contains only tool_result blocks (no real text prompt).
      // Buffer it — we'll decide when we see the next assistant line:
      // if it shares the same message.id, these results are interleaved
      // within the same API response; otherwise they end the group.
      if (!pendingToolResults) pendingToolResults = [];
      pendingToolResults.push(line);
      continue;
    } else {
      if (current) {
        groups.push(current);
        current = null;
        currentMsgIds = null;
        pendingToolResults = null;
      }
    }
  }
  if (current) groups.push(current);
  return groups;
}

export function readSession(filePath: string, sessionId: string): RawInteraction[] {
  if (!filePath || !fs.existsSync(filePath)) return [];

  const stat = fs.statSync(filePath);

  let resolvedFilePath: string;
  if (stat.isFile()) {
    resolvedFilePath = filePath;
  } else if (stat.isDirectory()) {
    const found = findSessionFile(filePath, sessionId);
    if (!found) return [];
    resolvedFilePath = found;
  } else {
    return [];
  }

  const lines = parseJsonlLines(resolvedFilePath);
  if (lines.length === 0) return [];

  const allToolResults = collectAllToolResults(lines);
  const fileMtime = fs.statSync(resolvedFilePath).mtime.getTime();

  const result: RawInteraction[] = [];

  // Phase 1: process non-assistant lines and group assistant lines
  const assistantGroups = groupAssistantLines(lines);

  // Build a map of line-index → assistant group for interleaving
  const lineToGroup = new Map<number, AssistantGroup>();
  for (const g of assistantGroups) {
    for (let j = 0; j < g.lines.length; j++) {
      lineToGroup.set(g.startLineIndex + j, g);
    }
  }

  // Track which groups we've already emitted
  const emittedGroups = new Set<AssistantGroup>();

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // If this line belongs to an assistant group, emit the whole group once
    const group = lineToGroup.get(i);
    if (group) {
      if (emittedGroups.has(group)) continue; // already emitted
      emittedGroups.add(group);

      // Merge the group into one RawInteraction
      const textParts: string[] = [];
      const allToolCalls: ToolCallInfo[] = [];
      let mergedUsage: TokenUsage | null = null;
      let mergedModel: string | null = null;
      let firstTimestamp: string | null = null;
      let firstTimeCreated: number | null = null;
      let lastTimeCreated: number | null = null;
      let mergedFinishReason: string | null = null;
      let explicitDurationMs: number | null = null;

      // Find the line with the most complete usage (has cache fields = final line)
      // Also track the maximum raw input_tokens across all lines (thinking lines report
      // full cumulative input, while final lines report incremental input only)
      let finalUsageLine: ClaudeJsonlLine | null = null;
      let maxRawInputTokens = 0;

      for (let lineIdx = 0; lineIdx < group.lines.length; lineIdx++) {
        const al = group.lines[lineIdx];
        const alLineIndex = group.startLineIndex + lineIdx;
        const contentBlocks = Array.isArray(al.message?.content)
          ? al.message.content as ContentBlock[]
          : [];
        const text = extractTextContent(al.message?.content);
        if (text) textParts.push(text);

        const toolCalls = extractToolCalls(contentBlocks);
        if (toolCalls) {
          for (const tc of toolCalls) {
            const v = allToolResults.get(tc.toolCallId) ?? tc.resultJson ?? null;
            allToolCalls.push({
              ...tc,
              resultJson: v == null ? null : (typeof v === 'string' ? v : JSON.stringify(v)),
            });
          }
        }

        const usage = al.message?.usage;
        if (usage) {
          // Track the maximum raw input_tokens (thinking lines report full cumulative)
          if ((usage.input_tokens ?? 0) > maxRawInputTokens) {
            maxRawInputTokens = usage.input_tokens ?? 0;
          }
          // The final line of a streaming response includes cache_read/cache_write fields
          // and has actual output_tokens; earlier lines report cumulative input only
          if (usage.cache_read_input_tokens != null || usage.cache_creation_input_tokens != null || (usage.output_tokens > 0 && usage.input_tokens < (finalUsageLine?.message?.usage?.input_tokens ?? Infinity))) {
            finalUsageLine = al;
          }
        }

        const realTimestamp = al.timestamp || null;
        const timeCreated = realTimestamp
          ? (isValidISO(realTimestamp) ? new Date(realTimestamp).getTime() : 0)
          : fileMtime + alLineIndex * 1000;
        const timestamp = realTimestamp
          ? (isValidISO(realTimestamp) ? realTimestamp : new Date(0).toISOString())
          : new Date(timeCreated).toISOString();

        if (firstTimeCreated == null || timeCreated < firstTimeCreated) {
          firstTimeCreated = timeCreated;
          firstTimestamp = timestamp;
        }
        if (lastTimeCreated == null || timeCreated > lastTimeCreated) {
          lastTimeCreated = timeCreated;
        }

        if (!mergedModel && al.message?.model) mergedModel = al.message.model;
        if (al.subtype) mergedFinishReason = al.subtype;
        if (al.duration_ms != null) explicitDurationMs = al.duration_ms;
      }

      // Build merged usage: use final line's cache data, but for total input use the
      // maximum raw input_tokens across all lines (thinking lines report full cumulative
      // input while final lines report incremental only)
      if (finalUsageLine?.message?.usage) {
        const fu = finalUsageLine.message.usage;
        const incrementalInput = fu.input_tokens ?? 0;
        const cacheRead = fu.cache_read_input_tokens ?? 0;
        const cacheWrite = fu.cache_creation_input_tokens ?? 0;
        // inputMessagesTokens = max of thinking line's full input and incremental+cache
        const inputMessagesTokens = Math.max(maxRawInputTokens, incrementalInput + cacheRead + cacheWrite);
        mergedUsage = {
          total: inputMessagesTokens + (fu.output_tokens ?? 0),
          input: incrementalInput,
          output: fu.output_tokens ?? 0,
          reasoning: 0,
          cacheRead,
          cacheWrite,
          cost: 0,
          inputMessagesTokens,
        };
      } else if (group.lines[0]?.message?.usage) {
        // No final line with cache fields — use first line's cumulative input
        const u0 = group.lines[0].message.usage;
        mergedUsage = {
          total: (u0.input_tokens ?? 0) + (u0.output_tokens ?? 0),
          input: u0.input_tokens ?? 0,
          output: u0.output_tokens ?? 0,
          reasoning: 0,
          cacheRead: 0,
          cacheWrite: 0,
          cost: 0,
          inputMessagesTokens: u0.input_tokens ?? 0,
        };
      }

      const mergedContent = textParts.length > 0 ? textParts.join('\n') : null;
      const mergedToolCalls = allToolCalls.length > 0 ? allToolCalls : null;

      // Detect skill invocations from Read tool_use calls to SKILL.md paths
      // When Claude Code loads a skill, it reads <skill-dir>/SKILL.md — this is the
      // definitive signal that a skill was invoked in this turn.
      const skillInvocations: ToolCallInfo[] = [];
      for (const tc of allToolCalls) {
        if (tc.toolName === 'Read' && tc.argsJson) {
          try {
            const args = JSON.parse(tc.argsJson);
            const filePath = String(args.file_path ?? '');
            if (filePath.endsWith('/SKILL.md')) {
              const skillName = filePath.split('/').slice(-2, -1)[0] || filePath.split('/SKILL.md')[0].split('/').pop() || '';
              skillInvocations.push({
                toolCallId: `skill-${tc.toolCallId}`,
                toolName: `skill/${skillName}`,
                argsJson: JSON.stringify({ skill: skillName, file_path: filePath }),
                resultJson: null,
                state: 'completed',
              });
            }
          } catch { /* ignore */ }
        }
      }

      const finalToolCalls = skillInvocations.length > 0
        ? [...allToolCalls, ...skillInvocations]
        : mergedToolCalls;

      // Skip if no content and no tool calls
      if (!mergedContent && !mergedToolCalls) continue;

      const latency = explicitDurationMs ?? (lastTimeCreated && firstTimeCreated ? lastTimeCreated - firstTimeCreated : null);

      result.push({
        role: 'assistant',
        content: mergedContent,
        timestamp: firstTimestamp ?? new Date(fileMtime).toISOString(),
        timeInfo: {
          created: firstTimeCreated ?? fileMtime,
          completed: lastTimeCreated ?? undefined,
        },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        subagent_type: null,
        tool_calls: finalToolCalls,
        usage: mergedUsage,
        model: mergedModel,
        modelID: null,
        providerID: null,
        latency,
        finish_reason: mergedFinishReason,
      });
      continue;
    }

    // Non-assistant lines
    const realTimestamp = line.timestamp || null;
    const timestamp = realTimestamp
      ? (isValidISO(realTimestamp) ? realTimestamp : new Date(0).toISOString())
      : new Date(fileMtime + i * 1000).toISOString();
    const timeCreated = realTimestamp ? new Date(realTimestamp).getTime() : fileMtime + i * 1000;
    const timeCompleted = line.duration_ms && timeCreated ? timeCreated + line.duration_ms : undefined;

    if (line.type === 'user' && line.message) {
      const content = extractTextContent(line.message.content);
      // Skip user messages that only contain tool_result (no real text prompt)
      if (!content) continue;

      // Claude Code injects skill context as user messages — reclassify as system
      // Key marker: <skill-format> tag distinguishes skill injections from /compact etc.
      const isSkillInjection = content.includes('Base directory for this skill') ||
        content.includes('<skill-format>') ||
        content.startsWith('Launching skill:') ||
        content.startsWith('Launching skill ');

      result.push({
        role: isSkillInjection ? 'system' : 'user',
        content,
        timestamp,
        timeInfo: { created: timeCreated },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        subagent_type: null,
        tool_calls: null,
        usage: null,
        model: null,
        modelID: null,
        providerID: null,
        latency: null,
        finish_reason: null,
      });
    } else if (line.type === 'result') {
      const resultUsage = line.cost_usd != null
        ? { total: 0, input: 0, output: 0, reasoning: 0, cacheRead: 0, cacheWrite: 0, cost: line.cost_usd, inputMessagesTokens: 0 }
        : null;
      result.push({
        role: 'result',
        content: line.result ?? null,
        timestamp,
        timeInfo: { created: timeCreated },
        agent: null,
        subagent_name: null,
        subagent_session_id: null,
        subagent_type: null,
        tool_calls: null,
        usage: resultUsage,
        model: null,
        modelID: null,
        providerID: null,
        latency: line.duration_ms ?? null,
        finish_reason: line.subtype ?? null,
      });
    }
  }

  // Post-process: merge skill injection system turns into preceding assistant's tool_call result
  // Claude Code emits skill content as a separate user/system line, but it's really the
  // tool_result of the Skill tool_call in the preceding assistant turn.
  for (let i = result.length - 1; i >= 0; i--) {
    const r = result[i];
    if (r.role !== 'system') continue;
    const content = r.content ?? '';
    const isSkillInjection = content.includes('Base directory for this skill') ||
      content.includes('<skill-format>') ||
      content.startsWith('Launching skill:') ||
      content.startsWith('Launching skill ');
    if (!isSkillInjection) continue;

    // Find the preceding assistant turn
    const prev = i > 0 ? result[i - 1] : null;
    if (prev?.role === 'assistant' && prev.tool_calls) {
      // Find a Skill tool_call whose resultJson is short/incomplete
      const skillTc = prev.tool_calls.find(tc =>
        tc.toolName.toLowerCase() === 'skill' &&
        tc.resultJson != null &&
        tc.resultJson.length < content.length
      );
      if (skillTc) {
        skillTc.resultJson = content;
        // Only remove if successfully merged into tool_call
        result.splice(i, 1);
      }
    }
    // If not merged into a Skill tool_call, keep the system turn
    // — it will be handled by the consecutive-merge step below
  }

  // Post-process: merge consecutive skill-injection system turns into one
  // When multiple skills are loaded at once, Claude Code emits separate lines for each.
  // These appear between a user prompt and the assistant response as separate system turns.
  // Merge them into a single consolidated system turn.
  for (let i = result.length - 1; i >= 0; i--) {
    const r = result[i];
    if (r.role !== 'system') continue;
    const content = r.content ?? '';
    const isSkillInjection = content.includes('Base directory for this skill') ||
      content.includes('<skill-format>') ||
      content.startsWith('Launching skill:') ||
      content.startsWith('Launching skill ');
    if (!isSkillInjection) continue;

    // Look backward for consecutive system/skill-injection turns to merge
    const mergeGroup: number[] = [i];
    let j = i - 1;
    while (j >= 0 && result[j].role === 'system') {
      const prevContent = result[j].content ?? '';
      const prevIsSkill = prevContent.includes('Base directory for this skill') ||
        prevContent.includes('<skill-format>') ||
        prevContent.startsWith('Launching skill:') ||
        prevContent.startsWith('Launching skill ');
      if (prevIsSkill) {
        mergeGroup.push(j);
        j--;
      } else {
        break;
      }
    }

    if (mergeGroup.length > 1) {
      // Merge: combine all content, keep earliest timestamp
      const allContent = mergeGroup
        .sort((a, b) => a - b)
        .map(idx => result[idx].content ?? '')
        .join('\n\n---\n\n');
      const earliestIdx = Math.min(...mergeGroup);
      const earliest = result[earliestIdx];

      // Update the earliest turn with merged content
      earliest.content = allContent;

      // Remove the rest (in reverse order to preserve indices)
      for (const idx of mergeGroup.sort((a, b) => b - a)) {
        if (idx !== earliestIdx) {
          result.splice(idx, 1);
        }
      }
    }
  }

  // Post-process: infer latency from timestamp gaps and estimate cost
  for (let i = 0; i < result.length; i++) {
    const r = result[i];
    // Infer latency: use gap between this turn and the next
    if (r.latency == null && r.timeInfo?.created) {
      const nextTs = result[i + 1]?.timeInfo?.created ?? null;
      if (nextTs && nextTs > r.timeInfo.created) {
        r.latency = nextTs - r.timeInfo.created;
        r.timeInfo.completed = nextTs;
      }
    }
    // Estimate cost if not provided (Claude pricing approximation)
    if (r.usage && r.usage.cost === 0) {
      const model = r.model ?? '';
      const isOpus = model.includes('opus') || model.includes('4.8');
      const isSonnet = model.includes('sonnet') || model.includes('4.6');
      const isHaiku = model.includes('haiku') || model.includes('4.5');
      let inputPrice = 3 / 1_000_000;   // default ~sonnet
      let outputPrice = 15 / 1_000_000;
      let cacheReadPrice = 0.3 / 1_000_000;
      let cacheWritePrice = 3.75 / 1_000_000;
      if (isOpus) {
        inputPrice = 15 / 1_000_000;
        outputPrice = 75 / 1_000_000;
        cacheReadPrice = 1.5 / 1_000_000;
        cacheWritePrice = 18.75 / 1_000_000;
      } else if (isHaiku) {
        inputPrice = 0.8 / 1_000_000;
        outputPrice = 4 / 1_000_000;
        cacheReadPrice = 0.08 / 1_000_000;
        cacheWritePrice = 1 / 1_000_000;
      }
      r.usage.cost =
        r.usage.input * inputPrice +
        r.usage.output * outputPrice +
        r.usage.cacheRead * cacheReadPrice +
        r.usage.cacheWrite * cacheWritePrice;
    }
  }

  return result;
}
