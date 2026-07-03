// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { PrismaClient } from '@prisma/client';

const MAX_CONTENT = 10000;
const MAX_ARGS_JSON = 2000;
const MAX_RESULT_JSON = 5000;
const MAX_SYSTEM_CONTENT = 2000;
const MAX_THINKING = 2000;

function truncate(text: string, maxLen: number, label?: string): string {
  if (text.length <= maxLen) return text;
  const suffix = label ? `\n... [truncated, full: ${text.length} chars]` : `\n... [truncated]`;
  return text.substring(0, maxLen) + suffix;
}

function fmtDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 3600000) return `${(ms / 60000).toFixed(1)}min`;
  return `${(ms / 3600000).toFixed(1)}h`;
}

function fmtTokens(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 1000000) return `${(n / 1000).toFixed(1)}K`;
  return `${(n / 1000000).toFixed(1)}M`;
}

function fmtCost(c: number): string {
  if (c < 0.01) return `$${c.toFixed(4)}`;
  return `$${c.toFixed(2)}`;
}

function detectLang(argsJson: string | null, toolName: string): string {
  if (toolName === 'Bash' || toolName === 'bash') return 'bash';
  if (toolName === 'Write' || toolName === 'write') {
    if (!argsJson) return '';
    try {
      const args = JSON.parse(argsJson);
      const fp = String(args.file_path ?? '');
      if (fp.endsWith('.py')) return 'python';
      if (fp.endsWith('.ts') || fp.endsWith('.tsx')) return 'typescript';
      if (fp.endsWith('.js')) return 'javascript';
      if (fp.endsWith('.json')) return 'json';
      if (fp.endsWith('.yaml') || fp.endsWith('.yml')) return 'yaml';
      if (fp.endsWith('.md')) return 'markdown';
      if (fp.endsWith('.sh')) return 'bash';
      if (fp.endsWith('.sql')) return 'sql';
    } catch {}
  }
  if (toolName === 'Read' || toolName === 'read') return '';
  return '';
}

function extractThinking(content: string | null): { thinking: string | null; body: string | null } {
  if (!content) return { thinking: null, body: null };
  const thinkingMatch = content.match(/<thinking>([\s\S]*?)<\/thinking>/);
  if (thinkingMatch) {
    const thinking = truncate(thinkingMatch[1].trim(), MAX_THINKING);
    const body = content.replace(/<thinking>[\s\S]*?<\/thinking>/, '').trim();
    return { thinking, body: body || null };
  }
  return { thinking: null, body: content };
}

type ToolCallData = {
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
  state: string;
  errorType: string | null;
  errorMessage: string | null;
  durationMs: number;
};

type SkillEventData = {
  skillName: string;
  eventType: string;
  success: boolean;
  errorMessage: string | null;
};

type TurnData = {
  turnIndex: number;
  role: string;
  content: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  subagentSessionId: string | null;
  model: string | null;
  latencyMs: number;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  cacheReadTokens: number;
  cacheWriteTokens: number;
  inputMessagesCount: number;
  inputMessagesTokens: number;
  contextWindowPct: number | null;
  contextWindowLimit: number | null;
  toolCalls: ToolCallData[];
  skillEvents: SkillEventData[];
};

// ─── Numbering scheme ───
// Root turns: 1, 2, 3, 4...
// Subagent sessions: 2.1, 2.2 (under the root turn that dispatched them)
// Subagent turns: 2.1.1, 2.1.2, 2.1.3...

function renderToolCall(tc: ToolCallData, dispatchNums: string[]): string {
  const isErr = tc.state !== 'ok' && tc.state !== 'completed';
  const name = tc.toolName;
  const parts: string[] = [];

  // If this is an Agent dispatch, show cross-reference instead of args
  if ((name === 'Agent' || name === 'agent') && dispatchNums.length > 0) {
    const refs = dispatchNums.map(n => `§${n}`).join(' + §');
    parts.push(`**Tool: ${name}** → §${dispatchNums.join(' + §')}${dispatchNums.length > 1 ? '（并行）' : ''}`);
    return parts.join('\n');
  }

  if (tc.argsJson) {
    if (name === 'Bash' || name === 'bash') {
      try {
        const parsed = JSON.parse(tc.argsJson);
        const cmd = String(parsed.command ?? parsed.description ?? '');
        parts.push(`**Tool: ${name}**`);
        parts.push('');
        parts.push('```bash');
        parts.push(truncate(cmd, 200));
        parts.push('```');
      } catch {
        parts.push(`**Tool: ${name}**`);
        parts.push('');
        parts.push('```');
        parts.push(truncate(tc.argsJson, 200));
        parts.push('```');
      }
    } else {
      parts.push(`**Tool: ${name}**`);
      parts.push('');
      parts.push('**Input:**');
      parts.push('');
      if (name === 'Read' || name === 'read' || name === 'Write' || name === 'write') {
        try {
          const parsed = JSON.parse(tc.argsJson);
          const fp = String(parsed.file_path ?? '');
          if (name === 'Read' || name === 'read') {
            parts.push(`filePath: ${fp}`);
          } else {
            parts.push(`filePath: ${fp}`);
            const contentPreview = String(parsed.content ?? '');
            if (contentPreview.length > 0) {
              parts.push('');
              const lang = detectLang(tc.argsJson, name);
              parts.push(`\`\`\`${lang}`);
              parts.push(truncate(contentPreview, MAX_ARGS_JSON, 'content'));
              parts.push('```');
            }
          }
        } catch {
          parts.push('```json');
          parts.push(truncate(tc.argsJson, MAX_ARGS_JSON));
          parts.push('```');
        }
      } else {
        parts.push('```json');
        parts.push(truncate(tc.argsJson, MAX_ARGS_JSON));
        parts.push('```');
      }
    }
  } else {
    parts.push(`**Tool: ${name}**`);
  }

  if (tc.resultJson) {
    const lang = detectLang(tc.argsJson, name);
    parts.push('');
    parts.push('**Output:**');
    parts.push('');
    const resultText = isErr ? tc.resultJson : truncate(tc.resultJson, MAX_RESULT_JSON);
    if (lang && lang !== 'json') {
      parts.push(`\`\`\`${lang}`);
      parts.push(resultText);
      parts.push('```');
    } else if (tc.resultJson.trimStart().startsWith('{') || tc.resultJson.trimStart().startsWith('[')) {
      parts.push('```json');
      parts.push(resultText);
      parts.push('```');
    } else {
      parts.push('```');
      parts.push(resultText);
      parts.push('```');
    }
  }

  if (isErr) {
    const errType = tc.errorType ?? 'unknown';
    const errMsg = tc.errorMessage ?? '';
    parts.push(`*Error: ${errType}${errMsg ? ` - ${truncate(errMsg, 200)}` : ''}*`);
  }

  if (tc.durationMs > 0) {
    parts.push(`*${fmtDuration(tc.durationMs)}*`);
  }

  return parts.join('\n');
}

function roleLabel(role: string): string {
  return role === 'tool_result' ? 'Tool Result'
    : role === 'system' ? 'System'
    : role === 'result' ? 'Result'
    : role.charAt(0).toUpperCase() + role.slice(1);
}

function contextSuffix(turn: TurnData): string {
  if (turn.role === 'assistant' && turn.contextWindowPct != null && turn.contextWindowPct > 0) {
    const pct = turn.contextWindowPct.toFixed(1);
    let suffix = `· 📦 ${pct}% context`;
    if (turn.inputMessagesCount > 0) {
      suffix += ` (${turn.inputMessagesCount} msgs / ${fmtTokens(turn.inputMessagesTokens)}t)`;
    }
    return suffix;
  }
  if (turn.role === 'user' && turn.inputMessagesTokens > 0) {
    return `· ${fmtTokens(turn.inputMessagesTokens)}t`;
  }
  return '';
}

function renderRootTurn(num: number, turn: TurnData, dispatchNums: string[]): string {
  const parts: string[] = [];

  // Heading: "## §1 User · 51.2Kt" or "## §2 Assistant · glm-5.1 · 9.8s · 📦 25.6%"
  const headingParts: string[] = [`§${num} ${roleLabel(turn.role)}`];
  if (turn.role === 'assistant') {
    if (turn.model) headingParts.push(`· ${turn.model}`);
    if (turn.latencyMs > 0) headingParts.push(`· ${fmtDuration(turn.latencyMs)}`);
  }
  headingParts.push(contextSuffix(turn));
  parts.push(`## ${headingParts.join(' ')}`);
  parts.push('');

  // Content
  if (turn.role === 'user') {
    parts.push(turn.content ?? '(empty)');
  } else if (turn.role === 'system') {
    parts.push(truncate(turn.content ?? '(empty)', MAX_SYSTEM_CONTENT));
  } else if (turn.role === 'assistant') {
    const { thinking, body } = extractThinking(turn.content);
    if (thinking) {
      parts.push('_Thinking:_');
      parts.push('');
      parts.push(thinking);
      parts.push('');
    }
    if (body) {
      parts.push(truncate(body, MAX_CONTENT, 'content'));
      parts.push('');
    }

    if (turn.skillEvents.length > 0) {
      for (const se of turn.skillEvents) {
        const status = se.success ? '✅' : '❌';
        parts.push(`*Skill: ${se.skillName} (${se.eventType}) ${status}*`);
        if (!se.success && se.errorMessage) {
          parts.push(`  Error: ${truncate(se.errorMessage, 200)}`);
        }
      }
      parts.push('');
    }

    if (turn.toolCalls.length > 0) {
      for (const tc of turn.toolCalls) {
        parts.push(renderToolCall(tc, dispatchNums));
        parts.push('');
      }
    }
  } else if (turn.role === 'tool_result' || turn.role === 'result') {
    if (turn.content) {
      parts.push(truncate(turn.content, MAX_CONTENT));
    }
  } else {
    if (turn.content) {
      parts.push(truncate(turn.content, MAX_CONTENT));
    }
  }

  return parts.join('\n');
}

function renderSubagentSession(num: string, name: string, turns: TurnData[], tokens: number, latencyMs: number): string {
  const parts: string[] = [];
  const peakContext = turns.reduce((max, t) => Math.max(max, t.contextWindowPct ?? 0), 0);

  // Section heading: "### **§2.1** general · 11 turns · 368Kt · 3.5min · peak 📦 22.4%"
  const headingParts: string[] = [`**§${num}** ${name} · ${turns.length} turns · ${fmtTokens(tokens)}t · ${fmtDuration(latencyMs)}`];
  if (peakContext > 0) headingParts.push(`· peak 📦 ${peakContext.toFixed(1)}%`);
  parts.push(`### ${headingParts.join(' ')}`);
  parts.push('');

  // Collapsible subagent turns
  parts.push('<details>');
  parts.push(`<summary>展开 §${num} turns (${turns.length})</summary>`);
  parts.push('');

  let subTurnNum = 1;
  for (const turn of turns) {
    const turnNum = `${num}.${subTurnNum}`;
    const heading: string[] = [`§${turnNum} ${roleLabel(turn.role)}`];
    if (turn.role === 'assistant') {
      if (turn.model) heading.push(`· ${turn.model}`);
      if (turn.latencyMs > 0) heading.push(`· ${fmtDuration(turn.latencyMs)}`);
    }
    heading.push(contextSuffix(turn));
    parts.push(`#### ${heading.join(' ')}`);
    parts.push('');

    if (turn.role === 'user') {
      parts.push(truncate(turn.content ?? '(empty)', MAX_CONTENT));
    } else if (turn.role === 'assistant') {
      const { thinking, body } = extractThinking(turn.content);
      if (thinking) {
        parts.push('_Thinking:_');
        parts.push('');
        parts.push(truncate(thinking, MAX_THINKING));
        parts.push('');
      }
      if (body) {
        parts.push(truncate(body, MAX_CONTENT, 'content'));
        parts.push('');
      }
      if (turn.skillEvents.length > 0) {
        for (const se of turn.skillEvents) {
          parts.push(`*Skill: ${se.skillName} (${se.eventType}) ${se.success ? '✅' : '❌'}*`);
        }
        parts.push('');
      }
      if (turn.toolCalls.length > 0) {
        for (const tc of turn.toolCalls) {
          parts.push(renderToolCall(tc, []));
          parts.push('');
        }
      }
    } else {
      if (turn.content) {
        parts.push(truncate(turn.content, MAX_CONTENT));
      }
    }

    parts.push('');
    subTurnNum++;
  }

  parts.push('</details>');
  parts.push('');

  return parts.join('\n');
}

export async function exportSessionToMarkdown(
  taskId: string,
  prisma: PrismaClient,
  framework?: string
): Promise<string> {
  const where: Record<string, string> = { taskId };
  if (framework) where.framework = framework;

  const session = await prisma.session.findFirst({ where });
  if (!session) throw new Error(`Session not found: "${taskId}"`);

  const executions = await prisma.execution.findMany({
    where: { sessionId: session.id },
    orderBy: [{ isSubagent: 'asc' }, { createdAt: 'asc' }],
  });

  const subagentExecs = executions.filter(e => e.isSubagent);

  const bridges = await prisma.interactionBridge.findMany({
    where: { sessionId: session.id },
    orderBy: [{ dispatchTimestamp: 'asc' }],
  });

  // Fetch ALL turns
  const allTurns = await prisma.turn.findMany({
    where: { sessionId: session.id },
    orderBy: [{ turnIndex: 'asc' }],
    include: {
      toolCalls: {
        select: {
          toolName: true,
          argsJson: true,
          resultJson: true,
          state: true,
          errorType: true,
          errorMessage: true,
          durationMs: true,
        },
        orderBy: [{ id: 'asc' }],
      },
      skillEvents: {
        select: {
          skillName: true,
          eventType: true,
          success: true,
          errorMessage: true,
        },
      },
    },
  });

  const turnData: TurnData[] = allTurns.map(t => ({
    turnIndex: t.turnIndex,
    role: t.role,
    content: t.content,
    isSubagent: t.isSubagent,
    subagentName: t.subagentName,
    subagentSessionId: t.subagentSessionId,
    model: t.model,
    latencyMs: t.latencyMs,
    totalTokens: t.totalTokens,
    inputTokens: t.inputTokens,
    outputTokens: t.outputTokens,
    reasoningTokens: t.reasoningTokens,
    cacheReadTokens: t.cacheReadTokens,
    cacheWriteTokens: t.cacheWriteTokens,
    inputMessagesCount: t.inputMessagesCount,
    inputMessagesTokens: t.inputMessagesTokens,
    contextWindowPct: t.contextWindowPct,
    contextWindowLimit: null,
    toolCalls: t.toolCalls.map(tc => ({
      toolName: tc.toolName,
      argsJson: tc.argsJson,
      resultJson: tc.resultJson,
      state: tc.state,
      errorType: tc.errorType,
      errorMessage: tc.errorMessage,
      durationMs: tc.durationMs,
    })),
    skillEvents: t.skillEvents.map(se => ({
      skillName: se.skillName,
      eventType: se.eventType,
      success: se.success,
      errorMessage: se.errorMessage,
    })),
  }));

  // ─── Build numbering ───
  const rootTurns = turnData.filter(t => !t.isSubagent);

  // Group subagent turns by session
  const subTurnsBySessionId = new Map<string, TurnData[]>();
  for (const t of turnData.filter(t => t.isSubagent && t.subagentSessionId)) {
    const arr = subTurnsBySessionId.get(t.subagentSessionId!) ?? [];
    arr.push(t);
    subTurnsBySessionId.set(t.subagentSessionId!, arr);
  }

  // Build dispatch map: subagentSessionId -> root turn index (in rootTurns array)
  const dispatchMap = new Map<string, number>(); // subId -> rootTurns array index
  for (const bridge of bridges) {
    if (bridge.subagentSessionId && bridge.dispatchTurnId) {
      const rawTurn = allTurns.find(t => t.id === bridge.dispatchTurnId);
      if (rawTurn) {
        const rootIdx = rootTurns.findIndex(t => t.turnIndex === rawTurn.turnIndex);
        if (rootIdx >= 0) dispatchMap.set(bridge.subagentSessionId, rootIdx);
      }
    }
  }

  // For unmatched subagents, assign to the nearest Agent tool call turn
  for (const [subId] of subTurnsBySessionId) {
    if (!dispatchMap.has(subId)) {
      // Find the first root assistant turn with an Agent tool call that hasn't been assigned
      for (let i = 0; i < rootTurns.length; i++) {
        const t = rootTurns[i];
        if (t.role === 'assistant' && t.toolCalls.some(tc => tc.toolName === 'Agent' || tc.toolName === 'agent')) {
          const alreadyAssigned = [...dispatchMap.values()].filter(v => v === i).length;
          const agentTcs = t.toolCalls.filter(tc => tc.toolName === 'Agent' || tc.toolName === 'agent').length;
          if (alreadyAssigned < agentTcs) {
            dispatchMap.set(subId, i);
            break;
          }
        }
      }
    }
  }

  // Compute root turn numbers: 1, 2, 3... and subagent numbers: 2.1, 2.2...
  // Each root turn gets a sequential number, and its subagents get X.1, X.2...
  const rootTurnNums = new Map<number, number>(); // rootTurns array index -> display number
  const subNums = new Map<string, string>(); // subagentSessionId -> display number like "2.1"

  let rootNum = 1;
  for (let i = 0; i < rootTurns.length; i++) {
    rootTurnNums.set(i, rootNum);

    // Assign subagent numbers under this root turn
    const subsHere = [...dispatchMap.entries()]
      .filter(([_, rootIdx]) => rootIdx === i)
      .map(([subId]) => subId);

    let subCounter = 1;
    for (const subId of subsHere) {
      subNums.set(subId, `${rootNum}.${subCounter}`);
      subCounter++;
    }

    rootNum++;
  }

  // ─── Assemble Markdown ───
  const md: string[] = [];

  // Header
  const title = session.label ?? session.query ?? session.taskId;
  md.push(`# ${truncate(title, 120)}`);
  md.push('');

  // Numbering scheme legend
  md.push('> **编号规则**');
  md.push('> - §1, §2, §3… → 主 Agent 每个 turn');
  md.push('> - §2.1, §2.2… → 子 Agent session（隶属于 dispatch 的主 Agent turn）');
  md.push('> - §2.1.1, §2.1.2… → 子 Agent 内每个 turn（`<details>` 折叠）');
  md.push('> - 同前缀 = 并行执行（如 §2.1 + §2.2）');
  md.push('> - 📦 = context window 使用率');
  md.push('');

  const durationMs = session.endTime && session.startTime
    ? new Date(session.endTime).getTime() - new Date(session.startTime).getTime()
    : session.totalLatencyMs;

  md.push(`**Session:** ${session.taskId} | **Framework:** ${session.framework} | **Model:** ${session.model ?? 'unknown'}`);
  md.push(`**Duration:** ${fmtDuration(durationMs)} | **Tokens:** ${fmtTokens(session.totalTokens)} (in: ${fmtTokens(session.totalInputTokens)} / out: ${fmtTokens(session.totalOutputTokens)} / reasoning: ${fmtTokens(session.totalReasoningTokens)}) | **Cost:** ${fmtCost(session.totalCost)}`);
  md.push('');
  md.push('---');
  md.push('');

  // Render root turns with subagent sections nested at dispatch points
  const renderedSubagents = new Set<string>();

  for (let i = 0; i < rootTurns.length; i++) {
    const turn = rootTurns[i];
    const num = rootTurnNums.get(i)!;

    // Collect dispatch cross-references for this turn's tool calls
    const subsHere = [...dispatchMap.entries()]
      .filter(([_, rootIdx]) => rootIdx === i)
      .map(([subId]) => subId);
    const dispatchNums = subsHere.map(subId => subNums.get(subId) ?? '?');

    md.push(renderRootTurn(num, turn, dispatchNums));
    md.push('');

    // Insert subagent sections after this root turn
    if (subsHere.length > 0) {
      // Mark parallel relationship when multiple subagents dispatched at same turn
      if (subsHere.length > 1) {
        const parallelRefs = subsHere.map(subId => `§${subNums.get(subId) ?? '?'}`);
        md.push(`> ${parallelRefs.join(' + ')} 并行执行`);
        md.push('');
      }

      for (const subId of subsHere) {
        if (renderedSubagents.has(subId)) continue;
        renderedSubagents.add(subId);

        const subTurns = subTurnsBySessionId.get(subId) ?? [];
        const subName = subTurns[0]?.subagentName ?? 'Unknown';
        const subExec = subagentExecs.find(e => e.agentSessionId === subId);
        const subTokens = subExec?.tokens ?? subTurns.reduce((s, t) => s + t.totalTokens, 0);
        const subLatency = subExec?.latencyMs ?? subTurns.reduce((s, t) => s + t.latencyMs, 0);
        const subNum = subNums.get(subId) ?? '?';

        md.push(renderSubagentSession(subNum, subName, subTurns, subTokens, subLatency));
      }
    }

    md.push('---');
    md.push('');
  }

  // Any remaining subagent sections not matched to dispatch points
  for (const [subId, subTurns] of subTurnsBySessionId) {
    if (renderedSubagents.has(subId)) continue;
    renderedSubagents.add(subId);
    const subName = subTurns[0]?.subagentName ?? 'Unknown';
    const subExec = subagentExecs.find(e => e.agentSessionId === subId);
    const subTokens = subExec?.tokens ?? subTurns.reduce((s, t) => s + t.totalTokens, 0);
    const subLatency = subExec?.latencyMs ?? subTurns.reduce((s, t) => s + t.latencyMs, 0);
    const subNum = subNums.get(subId) ?? '?';

    md.push(renderSubagentSession(subNum, subName, subTurns, subTokens, subLatency));
    md.push('---');
    md.push('');
  }

  // Stats section
  md.push('## Stats');
  md.push('');

  const rootExec = executions.find(e => !e.isSubagent);
  const rootTokens = rootExec?.tokens ?? session.totalTokens;
  const subTokens = subagentExecs.reduce((s, e) => s + e.tokens, 0);
  const rootCost = rootExec?.cost ?? session.totalCost;
  const subCost = subagentExecs.reduce((s, e) => s + e.cost, 0);
  const rootTurnCount = rootTurns.length;
  const subTurnCount = turnData.filter(t => t.isSubagent).length;
  const rootPeakContext = rootTurns.reduce((max, t) => Math.max(max, t.contextWindowPct ?? 0), 0);
  const subPeakContext = turnData.filter(t => t.isSubagent).reduce((max, t) => Math.max(max, t.contextWindowPct ?? 0), 0);

  md.push('| Metric | Root | Subagent(s) | Total |');
  md.push('|--------|------|-------------|-------|');
  md.push(`| Tokens | ${fmtTokens(rootTokens)} | ${fmtTokens(subTokens)} | ${fmtTokens(rootTokens + subTokens)} |`);
  md.push(`| Cost | ${fmtCost(rootCost)} | ${fmtCost(subCost)} | ${fmtCost(rootCost + subCost)} |`);
  md.push(`| Turns | ${rootTurnCount} | ${subTurnCount} | ${rootTurnCount + subTurnCount} |`);
  md.push(`| Context Peak | ${rootPeakContext > 0 ? `${rootPeakContext.toFixed(1)}%` : '—'} | ${subPeakContext > 0 ? `${subPeakContext.toFixed(1)}%` : '—'} | — |`);
  md.push(`| Subagents | — | ${subagentExecs.length} | ${subagentExecs.length} |`);
  md.push(`| Skills | ${session.totalSkillLoadCount} | — | ${session.totalSkillLoadCount} |`);
  md.push('');

  return md.join('\n');
}
