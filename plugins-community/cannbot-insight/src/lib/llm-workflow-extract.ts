// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { spawn } from "node:child_process";
import type { PrismaClient } from "@prisma/client";
import type { PlanFileDeclaration } from "./skill-eval-audit";

/**
 * LLM 提取主 agent 编排规程：读 session 的前几个 turn（dispatch 动作之前），
 * spawn claude CLI 让 LLM 总结编排规程 md。
 *
 * 与确定性提取（recoverWorkflowDeclaration）独立并行，方便对比：
 * - 确定性：从 DB 固定字段恢复原文（Skill invoke / Read / turn0）
 * - LLM：读 session 行为 → LLM 理解总结编排规程（泛化兜底，覆盖确定性取不到的场景）
 *
 * 输入：dispatch 动作之前的主 agent turns（含 user 注入 + assistant 加载 skill/规划）。
 * 无 dispatch → 取前 10 个 turns。
 * 输出：编排规程 md（source="llm-extract"），喂给 audit --kind root 对账。
 */

const MAX_TURNS = 10;
const TURN0_MAX_CHARS = 3000;
const ASSISTANT_MAX_CHARS = 1000;
const SKILL_BODY_MAX_CHARS = 20000;
const READ_BODY_MAX_CHARS = 15000;
const CLAUDE_TIMEOUT_MS = 300_000;

/**
 * 服务端模块级缓存：GET 端点立即返回（不阻塞 HTTP 连接），前端轮询。
 * key = taskId, value = { loading, content, source, error, promise }
 */
interface LlmExtractJob {
  loading: boolean;
  content: string | null;
  source: string | null;
  error: string | null;
  promise: Promise<void> | null;
}
const extractJobs = new Map<string, LlmExtractJob>();

/**
 * 启动或查询 LLM 提取任务。非阻塞：
 * - 首次调用 → 后台启动 claude CLI，立即返回 { loading: true }
 * - 已在运行 → 立即返回 { loading: true }
 * - 已完成 → 立即返回 { content, source } 或 { error }
 */
export function getOrStartLlmExtract(
  taskId: string,
  sessionId: string,
  prisma: PrismaClient,
): LlmExtractJob {
  const existing = extractJobs.get(taskId);
  if (existing && (existing.loading || existing.content)) return existing;
  if (existing && !existing.loading && !existing.content && existing.error) {
    extractJobs.delete(taskId); // 清除失败缓存，允许重试
  }

  const job: LlmExtractJob = { loading: true, content: null, source: null, error: null, promise: null };
  extractJobs.set(taskId, job);

  job.promise = (async () => {
    try {
      const decl = await llmExtractWorkflowDeclaration(sessionId, prisma);
      if (decl) { job.content = decl.content; job.source = decl.source; }
      else { job.error = "LLM 提取无输出（claude CLI 超时/无 dispatch 前 turns）"; }
    } catch (e) { job.error = e instanceof Error ? e.message : String(e); }
    finally { job.loading = false; }
  })();

  return job;
}

export async function llmExtractWorkflowDeclaration(
  sessionId: string,
  prisma: PrismaClient,
): Promise<PlanFileDeclaration | null> {
  // 1. 找第一个 dispatch 的 turnIndex → cutoff
  const firstDispatch = await prisma.skillEvent.findFirst({
    where: { eventType: "dispatch", turn: { sessionId, isSubagent: false } },
    orderBy: { createdAt: "asc" },
    select: { turnId: true },
  });

  let cutoff: number | undefined;
  if (firstDispatch) {
    const dispatchTurn = await prisma.turn.findUnique({
      where: { id: firstDispatch.turnId },
      select: { turnIndex: true },
    });
    cutoff = dispatchTurn?.turnIndex ?? undefined;
  }
  console.log('[llm-workflow-extract] cutoff:', cutoff, 'hasDispatch:', !!firstDispatch);

  // 2. 取 cutoff 之前（或前 MAX_TURNS 个）主 agent turns
  const turns = await prisma.turn.findMany({
    where: {
      sessionId,
      isSubagent: false,
      ...(cutoff != null ? { turnIndex: { lt: cutoff } } : {}),
    },
    orderBy: { turnIndex: "asc" },
    take: MAX_TURNS,
    select: { id: true, turnIndex: true, role: true, content: true, agentName: true },
  });

  console.log('[llm-workflow-extract] turns found:', turns.length);
  if (turns.length === 0) return null;

  // 3. 取这些 turns 的工具调用（含 resultJson——Skill invoke 的 <skill_content> / Read 的 <content>）
  const turnIds = turns.map((t) => t.id);
  const toolCalls = await prisma.toolCall.findMany({
    where: { turnId: { in: turnIds } },
    select: { turnId: true, toolName: true, argsJson: true, resultJson: true },
  });
  const tcByTurn = new Map<string, Array<{ toolName: string; argsJson: string | null; resultJson: string | null }>>();
  for (const tc of toolCalls) {
    const arr = tcByTurn.get(tc.turnId) ?? [];
    arr.push({ toolName: tc.toolName, argsJson: tc.argsJson, resultJson: tc.resultJson });
    tcByTurn.set(tc.turnId, arr);
  }

  // 4. 构造 prompt
  const prompt = buildExtractionPrompt(turns, tcByTurn);
  console.log('[llm-workflow-extract] prompt length:', prompt.length, 'toolCalls:', toolCalls.length);

  // 5. spawn claude CLI
  const result = await runClaudeCli(prompt);
  if (!result) {
    console.error('[llm-workflow-extract] claude CLI returned null. prompt length:', prompt.length, 'turns:', turns.length);
    return null;
  }

  return { content: result, source: "llm-extract" };
}

function buildExtractionPrompt(
  turns: Array<{ id: string; turnIndex: number; role: string; content: string | null; agentName: string | null }>,
  tcByTurn: Map<string, Array<{ toolName: string; argsJson: string | null; resultJson: string | null }>>,
): string {
  let sessionText = "";
  for (const t of turns) {
    const role = t.role;
    const agent = t.agentName ?? "unknown";
    const content = t.content ?? "";
    const maxChars = role === "user" ? TURN0_MAX_CHARS : ASSISTANT_MAX_CHARS;
    const truncated = content.slice(0, maxChars);
    const ellipsis = content.length > maxChars ? " …(截断)" : "";
    sessionText += `### Turn ${t.turnIndex} (${role}, agent=${agent})\n${truncated}${ellipsis}\n`;

    const tcs = tcByTurn.get(t.id) ?? [];
    for (const tc of tcs) {
      const block = formatToolCallFull(tc.toolName, tc.argsJson, tc.resultJson);
      if (block) sessionText += block;
    }
    sessionText += "\n";
  }

  return `阅读以下 session 片段（主 agent dispatch 动作之前的 turns）。

你的任务：找出其中的编排规程原文，选择最完整的那一份输出。

编排规程可能出现在以下位置（按可靠性排序）：
1. Skill 调用的返回结果（<skill_content> 标签内的 SKILL.md 正文）——最可靠
2. 注入的 user message（turn0 本身就是编排规程）
3. Read 调用的返回结果（<content> 标签内的文件内容）
4. assistant 引用的规程片段

要求：
- 择优选择一份最完整的编排规程原文输出，不要拼接多个来源
- 只提取原文，不要总结、重组、补充
- 不要改变原文的标题结构
- 不要添加原文没有的内容
- 如果编排规程在 <skill_content> 里，提取 <skill_content> 内部的正文（去掉标签外壳）
- 如果某段是 assistant 的思考过程，只提取其中引用的规程原文，不要提取思考过程本身
- 只输出 md 正文，不要包裹在代码块里，不要加任何解释

## Session 片段

${sessionText}`;
}

/** 格式化工具调用的完整信息（含 resultJson 原文，供 LLM 提取） */
function formatToolCallFull(toolName: string, argsJson: string | null, resultJson: string | null, includeRead = false): string {
  const tn = toolName.toLowerCase();
  const summary = formatToolCall(toolName, argsJson);
  // Skill invoke：提取 <skill_content> 正文（编排规程原文）
  if (tn === "skill" && resultJson) {
    const m = resultJson.match(/<skill_content[^>]*>([\s\S]*?)<\/skill_content>/);
    if (m) {
      const body = m[1].trim().slice(0, SKILL_BODY_MAX_CHARS);
      return `工具调用：${summary}\n返回（skill_content 正文，${body.length} 字）：\n${body}\n`;
    }
  }
  // Read：如果 includeRead=true（subagent 提取），提取 <content> 正文（可能是 agent.md）
  if (includeRead && tn === "read" && resultJson) {
    const m = resultJson.match(/<content>([\s\S]*?)<\/content>/);
    if (m) {
      const body = m[1].replace(/^\s*\d+:\s/gm, "").trim().slice(0, READ_BODY_MAX_CHARS);
      if (body.length > 100) {
        return `工具调用：${summary}\n返回（文件内容，${body.length} 字）：\n${body}\n`;
      }
    }
  }
  // 其他工具只显示摘要
  return summary ? `工具调用：${summary}\n` : "";
}

function formatToolCall(toolName: string, argsJson: string | null): string {
  if (!argsJson) return toolName;
  try {
    const a = JSON.parse(argsJson);
    const tn = toolName.toLowerCase();
    if (tn === "skill") return `Skill(name=${a.name ?? a.skill ?? "?"})`;
    if (tn === "read") return `Read(filePath=${a.filePath ?? a.file_path ?? "?"})`;
    if (tn === "task" || tn === "agent") {
      return `Task(description=${a.description ?? "?"}, subagent_type=${a.subagent_type ?? a.subagentType ?? a.agent ?? "?"})`;
    }
    if (tn === "todowrite") return `TodoWrite(todos=${a.todos?.length ?? "?"}项)`;
    return `${toolName}(${JSON.stringify(a).slice(0, 120)})`;
  } catch {
    return toolName;
  }
}

function runClaudeCli(prompt: string): Promise<string | null> {
  return new Promise((resolve) => {
    const claudeBin = process.env.CLAUDE_CLI_PATH || "/usr/local/bin/claude";
    const proc = spawn(claudeBin, ["-p", "--output-format", "text"], {
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    const timer = setTimeout(() => {
      proc.kill();
      resolve(null);
    }, CLAUDE_TIMEOUT_MS);

    proc.stdout.on("data", (d: Buffer) => {
      stdout += d.toString();
    });
    proc.stderr.on("data", (d: Buffer) => {
      stderr += d.toString();
    });
    proc.on("close", (code: number | null) => {
      clearTimeout(timer);
      if (code === 0 && stdout.trim()) {
        resolve(stdout.trim());
      } else {
        console.error('[llm-workflow-extract] claude CLI exit code:', code, 'stdout len:', stdout.length, 'stderr:', stderr);
        resolve(null);
      }
    });
    proc.on("error", (err: Error) => {
      clearTimeout(timer);
      console.error('[llm-workflow-extract] spawn error:', err.message);
      resolve(null);
    });
    proc.stdin.write(prompt);
    proc.stdin.end();
  });
}
