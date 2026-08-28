// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { PrismaClient } from "@prisma/client";
import { extractSkillNameFromArgs } from "@/lib/skill-event-grouping";

/**
 * sift 对账桥接：从 insight 的 session 里恢复某 skill 的 SKILL.md 正文，
 * 物化成临时 skill 目录，shell out 调 `sift audit`，读回 audit-report.json。
 *
 * 每个 skill 的完整正文存在 invoke 类 ToolCall 的 resultJson 里，以
 * `<skill_content name="…">…</skill_content>` 包裹（见 skill-event-grouping.ts 的 invokeTc）。
 */

const SKILL_CONTENT_RE = /<skill_content[^>]*>([\s\S]*?)<\/skill_content>/;

/** skill 工具调用里能带 <skill_content> 正文的 toolName（小写）。 */
const SKILL_INVOKE_TOOLS = new Set(["skill", "skill/invoke", "skill/use"]);

/**
 * 从一段文本里剥出 `<skill_content>…</skill_content>` 内部的正文（trim）。
 * 没有外壳 / 空输入 → null。audit 期望的是纯 SKILL.md markdown，不要外壳。
 */
export function stripSkillContent(resultJson: string | null | undefined): string | null {
  if (!resultJson) return null;
  const m = resultJson.match(SKILL_CONTENT_RE);
  return m ? m[1].trim() : null;
}

/** 一个 ToolCall 是不是"带正文的 skill invoke 调用"（区别于 task/agent dispatch）。 */
export function isSkillInvokeToolCall(toolName: string | null | undefined): boolean {
  return !!toolName && SKILL_INVOKE_TOOLS.has(toolName.toLowerCase());
}

/**
 * audit 调用参数（不含二进制名 `sift`，由调用方加）。声明路径作位置参：目录找其下 SKILL.md，传 .md 文件则直读。
 *
 * `--kind skill` 始终带:本路(--db)是 per-skill 对账(只传一个 skill 的 SKILL.md)。不带它,
 * sift 会拿本 skill 的指令去对账整条 session(含别的 skill 跑的 turn)→ 误判。sift
 * 靠传入 SKILL.md 的 name + --kind skill 切出本 skill 被 Skill 调用期间的段,只对账这些段
 * (audit.py;skill 未被调用过 → exit 2,而此处正文必来自一次 invoke,不会触发)。
 */
export function buildAuditArgs(opts: {
  skillPath: string;
  dbPath: string;
  sessionId: string;
  outputDir: string;
  kind?: "skill" | "agent" | "root";
}): string[] {
  return [
    "audit",
    opts.skillPath,
    "--db",
    opts.dbPath,
    "--session",
    opts.sessionId,
    "-o",
    opts.outputDir,
    "--kind",
    opts.kind ?? "skill",
    "--turn-refs",
  ];
}

/**
 * `--transcript` 路的参数(cannbot-insight / claude-code framework 用):
 * 把 session 序列化成结构化 records JSON 喂 sift,绕开 `--db` 的 opencode-native 限制。
 * `kind` 必带:skill=按 Skill 调用切(skill 对账)、agent=按 agent 归属切(被 dispatch 的 agent 对账)。
 */
export function buildTranscriptArgs(opts: {
  skillPath: string;
  transcriptPath: string;
  outputDir: string;
  kind: "skill" | "agent" | "root";
}): string[] {
  return ["audit", opts.skillPath, "--transcript", opts.transcriptPath, "-o", opts.outputDir, "--kind", opts.kind, "--turn-refs"];
}

/**
 * 一个声明单元在 session 里以哪些「面」出现 → 该跑哪些 kind 的对账。
 *
 * Skills tab 每行按 events 的 eventType 路由对账按钮:
 *   - invoke / use 事件 → skill 面(Skill 调用锚点,正文从 session 的 resultJson 恢复)
 *   - dispatch 事件 → agent 面(被 dispatch 的子 agent,.md 声明从 AGENTS_ROOT 读)
 *
 * 注:**主 agent 的 workflow 对账（root 面）不在此派生**——主 agent 通常不 invoke skill
 * （只 dispatch 子 agent），其 workflow skill 声明是 session 首条 user turn 的注入系统
 * 提示（见 audit-skillsift 路由 kind=root + recoverMainAgentWorkflowBody）。root 目标由
 * SkillAuditTab / SkillDetail 按 turn0 长度阈值合成添加，不靠 skillEvents 的 eventType。
 *
 * dispatch-only 行(如 developer)原来也挂 skill 对账按钮,但无 invoke 事件 → recoverSkillBody
 * 返回 null → 必 404。路由到 agent 对账才对。dual-nature(同既 invoke 又 dispatch,如 st-verifier)
 * 两面都返,UI 出两个按钮各审各的(与 sift 的 kind 分路一致——同名 skill 声明 ≠ agent 声明)。
 *
 * 输入只读 eventType,与 SkillEventItem 结构兼容(鸭子类型);未知 eventType → 不计入任何面。
 */
export function auditKindsForEvents(
  events: { eventType: string }[],
): ("skill" | "agent")[] {
  const kinds: ("skill" | "agent")[] = []
  if (events.some((e) => e.eventType === "invoke" || e.eventType === "use")) kinds.push("skill")
  if (events.some((e) => e.eventType === "dispatch")) kinds.push("agent")
  return kinds
}

/**
 * 仅 dispatch（含 dispatch+unload）的条目是被分派的子代理，不是被加载/调用的 skill。
 * Skills 表据此过滤掉子代理，与 Audit 的 agent/skill 划分对齐。
 * 输入只读 eventType，与 SkillEventItem 结构兼容（鸭子类型）。
 */
export function isDispatchOnlyAgent(events: { eventType: string }[]): boolean {
  return !events.some(
    (e) => e.eventType === "load" || e.eventType === "invoke" || e.eventType === "use"
  );
}

/**
 * 返回「仅 dispatch」的 skillName 集合（即被分派的子代理，如 blackbox-designer）。
 * 用于全页统一过滤：Overview 计数/列表、Skills 图表都不展示这些非-skill 条目。
 * 与 isDispatchOnlyAgent 同口径：某 name 无任何 load/invoke/use 事件 → 视为子代理。
 */
export function dispatchOnlySkillNames(events: { skillName: string; eventType: string }[]): Set<string> {
  const hasSkillEvent = new Map<string, boolean>();
  for (const e of events) {
    if (e.eventType === "load" || e.eventType === "invoke" || e.eventType === "use") {
      hasSkillEvent.set(e.skillName, true);
    } else if (!hasSkillEvent.has(e.skillName)) {
      hasSkillEvent.set(e.skillName, false);
    }
  }
  const out = new Set<string>();
  for (const [name, hasSkill] of hasSkillEvent) if (!hasSkill) out.add(name);
  return out;
}

/**
 * In-flight 去重:同 key 的请求在飞时,复用已有 promise、不再调 fetcher。
 *
 * SiftAuditDialog 的 audit POST 在 useEffect 里,dev 下 remount(HMR / Fast Refresh 回退
 * 全页刷新)会重跑 effect → 不去重就再发一次 POST。而服务端 execFileSync 不可中止、单跑可达
 * 30 分钟 → 多轮堆叠抢 LLM 端点 → 限流失败(实测 opdef-developer:两轮重叠 2 分钟,先起的被
 * 529 杀)。去重让重发复用在飞的 promise,从根上杜绝堆叠。settle(resolve/reject)后清条目,
 * 下次 open 重跑无妨(已有结果不再缓存,允许用户重审)。
 *
 * cache 入参可注入(便于测);生产由调用方传模块级单例。泛型 T = 请求结果类型。
 */
export function getInflightAudit<T>(
  key: string,
  fetcher: () => Promise<T>,
  cache: Map<string, Promise<T>>,
): Promise<T> {
  const existing = cache.get(key);
  if (existing) return existing;
  const p = fetcher().finally(() => {
    cache.delete(key);
  });
  cache.set(key, p);
  return p;
}

/** 安全解析 ToolCall.argsJson(JSON 字符串)→ 对象；坏 / 空 → {}。 */
function parseArgs(argsJson: string | null | undefined): Record<string, unknown> {
  if (!argsJson) return {};
  try {
    return JSON.parse(argsJson) as Record<string, unknown>;
  } catch {
    return {};
  }
}

/**
 * 把 insight 的 session(Turn + ToolCall)序列化成 sift 结构化 records JSON：
 * `{format:"sift-records", records:[{role, text?, agent?, parent_tool_use_id, tool_calls?}]}`。
 * sift 收到后展开成 cc 记录、按 skill 切段对账（见 sift evals/structured.py）。
 *
 * scope-aware:子 agent turn 的 parent_tool_use_id = subagentSessionId(缺则 parentExecutionId),
 * 主 turn 恒 null。sift 切片器(slice.py _scope_of)据此归作用域——子 agent 里跑的 skill 段
 * 不再被主 scope 的 last-invoked-owns 污染(主 scope 噪声混进段)。无需合成 Agent tool_use:零嵌套
 * 数据下 parent_tool_use_id 直接归段即正确(slice.py 的 depth-1 closure 在无 Agent tool_use 时是
 * no-op,归段只靠 parent_tool_use_id)。turn 按 turnIndex 升序;turn 内 tool_calls 按 createdAt 升序。
 */
export async function buildStructuredRecords(
  sessionId: string,
  prisma: PrismaClient,
): Promise<{ format: string; records: unknown[] }> {
  const turns = await prisma.turn.findMany({
    where: { sessionId },
    orderBy: { turnIndex: "asc" },
  });
  if (turns.length === 0) return { format: "sift-records", records: [] };

  const turnIds = turns.map((t) => t.id);
  const toolCalls = await prisma.toolCall.findMany({
    where: { turnId: { in: turnIds } },
    orderBy: { createdAt: "asc" },
  });
  const byTurn = new Map<string, typeof toolCalls>();
  for (const tc of toolCalls) {
    const arr = byTurn.get(tc.turnId) ?? [];
    arr.push(tc);
    byTurn.set(tc.turnId, arr);
  }

  const records = turns.map((t) => {
    const rec: Record<string, unknown> = {
      role: t.role,
      // 子 agent turn → subagentSessionId 归独立 scope(缺则 parentExecutionId 兜底);主 turn → null
      // (主 scope)。sift 切片器按此归段,隔离子 agent 的 skill 执行、免主 scope 噪声污染。
      parent_tool_use_id: t.isSubagent ? (t.subagentSessionId || t.parentExecutionId || null) : null,
    };
    if (t.content) rec.text = t.content;
    if (t.agentName) rec.agent = t.agentName;
    const tcs = byTurn.get(t.id) ?? [];
    if (tcs.length > 0) {
      // **不带 result**(`tc.resultJson`):audit 判定只读 user_turns(scene)+ tool 输入
      // (input → tool_trace)+ assistant 文本(transcript_text),**工具结果对 verdict 零贡献**
      // (见 sift audit/auditor.py:rebuild 只取这三项;derive_declarations 只读
      // input_files 的 path,而 path 来自 Read 的 file_path 输入、不靠 result)。真实 session
      // 的 resultJson 常达 10MB+(Read 整文件 / skill 正文 / 大输出),回传它会让 --transcript
      // 的 session.json 撑到 20MB+,序列化 / 落盘 / sift 解析全慢——纯死重量,砍之。
      // sift 结构化 reader 对缺 result 的 tool_call 本就容错(不产 tool_result 块)。
      rec.tool_calls = tcs.map((tc) => ({
        name: tc.toolName,
        input: parseArgs(tc.argsJson),
      }));
    }
    return rec;
  });
  return { format: "sift-records", records };
}

/** 主 agent 编排对账目标的合成名（Skills / Audit skill 子 tab 用；root 的 skillName 无意义，仅作 key + 显示）。 */
export const MAIN_AGENT_WORKFLOW_NAME = "主 agent 编排";

/** 主 agent 编排声明的最小长度（首条 user turn 内容达此才视为注入的 workflow skill 正文）。 */
export const MAIN_AGENT_WORKFLOW_MIN_LEN = 500;

/**
 * 恢复主 agent 的 workflow 声明正文：取 session **首条 user turn**（isSubagent=false、
 * role=user、turnIndex 最小）的 content。opencode/cannbot-insight 里主 agent 配置的
 * workflow skill 会被注入成首条 user 消息（系统提示），主 agent 通常不 invoke skill
 * （只 dispatch 子 agent）→ 其 workflow 声明只能从 turn0 取，不在 skillEvents。
 *
 * 内容过短（< MAIN_AGENT_WORKFLOW_MIN_LEN）→ null（视为普通短用户查询，非 workflow 声明）。
 * 找不到 user turn → null。
 *
 * 注：这是主 agent 的**角色/编排规程**（agent.md-like），不是具体工作流。具体工作流
 * （任务树）是 STATE.md，见 `recoverStateMdPlan`——root 对账改送 STATE.md（具体计划）。
 * 本函数仍用于扫盘反查 workflow skill 真名（skill-md-scan 按 turn0 body 匹配）。
 */
export async function recoverMainAgentWorkflowBody(
  sessionId: string,
  prisma: PrismaClient,
): Promise<string | null> {
  const t = await prisma.turn.findFirst({
    where: { sessionId, role: "user", isSubagent: false },
    orderBy: { turnIndex: "asc" },
  });
  if (!t?.content) return null;
  return t.content.length >= MAIN_AGENT_WORKFLOW_MIN_LEN ? t.content : null;
}

/** 计划文件声明的最小长度（达此才视为 workflow 任务清单，非残片）。 */
export const STATE_MD_MIN_LEN = 100;

/**
 * 常见计划文件名（basename，大小写不敏感）。argsJson.filePath 命中即强信号——
 * 命名即语义，文件作者用这些名字说明它就是计划文件。
 */
export const PLAN_FILE_NAMES = [
  "STATE.md",
  "TODO.md",
  "TASKS.md",
  "WORKFLOW.md",
  "BACKLOG.md",
  "ROADMAP.md",
  "SPRINT.md",
] as const;

/** 计划文件扫描的 take 上限（大 session 防爆）。计划文件早期就写，1000 够覆盖。 */
const PLAN_FILE_SCAN_TAKE = 1000;

/**
 * 计划文件声明（recoverPlanFileDeclaration 的返回）：剥行号 + [x]→[ ] 归一后的
 * plan 态正文 + 来源标识（供 UI 显示）。
 */
export interface PlanFileDeclaration {
  content: string;
  source: string;
}

/** 从 argsJson 提取 filePath（兼容 filePath/file_path/path 三种字段名）。 */
function extractFilePath(argsJson: string | null): string | null {
  if (!argsJson) return null;
  try {
    const a = JSON.parse(argsJson);
    if (a && typeof a === "object") {
      const fp = a.filePath ?? a.file_path ?? a.path ?? a.filename;
      if (typeof fp === "string") return fp;
    }
  } catch {
    /* not JSON */
  }
  const m = argsJson.match(/"(?:filePath|file_path|path|filename)"\s*:\s*"([^"]+)"/);
  return m ? m[1] : null;
}

/** 取路径 basename（兼容 / 和 \）。 */
function basenameOf(p: string): string {
  return p.split(/[\\/]/).pop() ?? p;
}

/** filePath 的 basename 是否命中常见计划文件名（大小写不敏感）。 */
function isPlanFilename(filePath: string | null): boolean {
  if (!filePath) return false;
  const base = basenameOf(filePath).toLowerCase();
  return PLAN_FILE_NAMES.some((n) => base === n.toLowerCase());
}

/**
 * 判断 content 是否"plan-like"（内容特征兜底，用于无文件名命中时）。
 * 保守阈值：宁可漏不可误（false positive → 拿源码当 workflow 审 → 垃圾 finding）。
 *
 * - checklist 项 ≥3（plan 专属，源码注释偶有 1-2 个 TODO → 卡住）
 * - OR 计划标题 ≥1（## Tasks/任务/Plan… 强信号）
 * - OR 编号任务 ≥5（编号列表弱信号，需多条才算 plan）
 */
function isPlanLike(content: string): boolean {
  const checklist = (content.match(/^\s*- \[[ xX]\]/gim) || []).length;
  const heading = (content.match(
    /^## (Tasks|任务|TODO|待办|Plan|计划|工作流|Workflow|Backlog|Roadmap)/im,
  ) || []).length;
  const numbered = (content.match(/^\d+\.\s/gm) || []).length;
  return checklist >= 3 || heading >= 1 || numbered >= 5;
}

/**
 * 从一个文件工具调用（read/write）提取 content + filePath + basename。
 * Read：resultJson 的 `<content>…</content>`，剥行号（opencode 格式 `1: line\n2: line`）。
 * Write：argsJson.content（JSON parse）。无 content → null。
 */
function extractFileContent(tc: {
  toolName: string;
  argsJson: string | null;
  resultJson: string | null;
}): { content: string; basename: string | null } | null {
  const tn = (tc.toolName || "").toLowerCase();
  let content: string | null = null;
  if (tn === "read" && tc.resultJson) {
    const m = tc.resultJson.match(/<content>([\s\S]*?)<\/content>/);
    if (m) content = m[1].replace(/^\s*\d+:\s/gm, "");
  } else if (tn === "write" && tc.argsJson) {
    try {
      const a = JSON.parse(tc.argsJson);
      if (a && typeof a.content === "string") content = a.content;
    } catch {
      /* ignore */
    }
  }
  if (!content) return null;
  const fp = extractFilePath(tc.argsJson);
  return { content, basename: fp ? basenameOf(fp) : null };
}

/** todowrite 类工具名（小写）。主 agent 用它维护任务计划（todos）。 */
const TODO_TOOLS = new Set(["todowrite", "todo_write", "todo"]);

/**
 * 从 todowrite 工具调用提取 todos 数组。
 * argsJson 格式：`{"todos":[{"content":"...","status":"pending",...},...]}`。
 * 返回 todos（仅含 content 字符串的项）；非 todowrite / 无 todos → null。
 */
function extractTodosFromToolCall(tc: {
  toolName: string;
  argsJson: string | null;
}): Array<{ content: string; status: string }> | null {
  const tn = (tc.toolName || "").toLowerCase();
  if (!TODO_TOOLS.has(tn)) return null;
  if (!tc.argsJson) return null;
  try {
    const a = JSON.parse(tc.argsJson);
    if (a && Array.isArray(a.todos) && a.todos.length > 0) {
      const valid = a.todos.filter(
        (t: unknown): t is { content: string; status: string } =>
          typeof t === "object" && t !== null && typeof (t as { content?: unknown }).content === "string",
      );
      return valid.length > 0 ? valid : null;
    }
  } catch {
    /* not JSON */
  }
  return null;
}

/** 把 todos 格式化为 plan 态 checklist（所有 status 归一为 `[ ]`，cancelled 项保留）。 */
function formatTodosAsPlan(todos: Array<{ content: string; status: string }>): string {
  const lines = todos.map((t) => `- [ ] ${t.content}`);
  return `# 主 agent 编排 计划（todowrite）\n\n${lines.join("\n")}`;
}

/**
 * 恢复主 agent 的 workflow 具体计划：从 session 的工具调用取**最完整的计划内容**，
 * [x]→[ ] 归一为 plan 态（声明=该干什么，非执行结果）。
 *
 * **D 混合方案 + plan-like 内容过滤**（见 docs/workflow-plan-extraction-design.md）：
 * 单次 DB 查询（read/write/todowrite）+ 内存四桶分流，按优先级取首个 substantial：
 *
 * 1. **named-Read + plan-like**（计划文件被 Read 且内容是任务清单，最高优先）：
 *    filePath basename 命中 PLAN_FILE_NAMES 且是 Read 调用 且内容 plan-like
 *    （有 ## Tasks 标题或 checklist）。STATE.md 有 ## Tasks + 执行者/验收标准 → 真 workflow。
 *    PLAN.md 迭代穿刺表格不 plan-like → 不进此桶 → 退 todowrite。这是 S1 vs S2 的区分关键。
 * 2. **todowrite**（主 agent 自己的 todos 计划）：toolName=todowrite，≥3 todos。
 *    S2 无 plan-like 的 STATE.md → 用 todowrite（"1.1 开发准备…"）。
 * 3. **named-Write + plan-like**（计划文件被 Write 且 plan-like = 产物兜底，低置信）。
 * 4. **scan**（内容特征 fallback）。
 *
 * named-Read 优先于 todowrite 的理由：STATE.md 有 ## Tasks + 执行者/验收标准（详细任务
 * 清单），是 state-generator 生成 + 主 agent Read 执行的真 workflow。todowrite 是主
 * agent 的编排动作（"Dispatch developer to..."），不如 STATE.md 详细。S2 的 PLAN.md 是
 * 迭代穿刺表格（不 plan-like）→ 不进 named-Read → 退 todowrite（S2 的真 workflow）。
 *
 * 返回 { content, source }：source = 文件名 / "todowrite" / "scan:文件名"。
 */
export async function recoverPlanFileDeclaration(
  sessionId: string,
  prisma: PrismaClient,
): Promise<PlanFileDeclaration | null> {
  const tcs = await prisma.toolCall.findMany({
    where: {
      turn: { sessionId },
      toolName: { in: ["read", "Read", "write", "Write", "todowrite", "TodoWrite", "todo_write"] },
    },
    select: { toolName: true, argsJson: true, resultJson: true },
    take: PLAN_FILE_SCAN_TAKE,
  });
  let bestNamedRead: { content: string; basename: string } | null = null;
  let bestTodo: { todos: Array<{ content: string; status: string }>; count: number } | null = null;
  let bestNamedWrite: { content: string; basename: string } | null = null;
  let bestScan: { content: string; basename: string } | null = null;
  for (const tc of tcs) {
    // todowrite 桶
    const todos = extractTodosFromToolCall(tc);
    if (todos) {
      if (!bestTodo || todos.length > bestTodo.count) {
        bestTodo = { todos, count: todos.length };
      }
      continue;
    }
    // 文件桶（read/write）：只有 plan-like 的计划文件才进 named 桶
    // （S2 的 PLAN.md 是迭代穿刺表格，不 plan-like → 不进 named-Read → 退 todowrite）
    const ex = extractFileContent(tc);
    if (!ex) continue;
    const isRead = (tc.toolName || "").toLowerCase() === "read";
    if (isPlanFilename(ex.basename) && isPlanLike(ex.content)) {
      const base = ex.basename ?? "unknown";
      if (isRead) {
        if (!bestNamedRead || ex.content.length > bestNamedRead.content.length) {
          bestNamedRead = { content: ex.content, basename: base };
        }
      } else {
        if (!bestNamedWrite || ex.content.length > bestNamedWrite.content.length) {
          bestNamedWrite = { content: ex.content, basename: base };
        }
      }
    } else if (!isPlanFilename(ex.basename) && isPlanLike(ex.content)) {
      const base = ex.basename ?? "unknown";
      if (!bestScan || ex.content.length > bestScan.content.length) {
        bestScan = { content: ex.content, basename: base };
      }
    }
  }
  // 优先级 1：named-Read（计划文件被 Read 且 plan-like = 真 workflow，最高优先）
  // STATE.md 有 ## Tasks + checklist（plan-like），PLAN.md 迭代表格不 plan-like → 区分关键
  if (bestNamedRead && bestNamedRead.content.length >= STATE_MD_MIN_LEN) {
    return { content: bestNamedRead.content.replace(/\[x\]/g, "[ ]"), source: bestNamedRead.basename };
  }
  // 优先级 2：todowrite（主 agent 自己的 todos 计划）
  // substantiality 信号是 todo 数量（≥3），不是字符长度（todo 标题短，格式化后可能 <100 字）
  if (bestTodo && bestTodo.count >= 3) {
    return { content: formatTodosAsPlan(bestTodo.todos), source: "todowrite" };
  }
  // 优先级 3：named-Write（计划文件被 Write 且 plan-like = 产物兜底，低置信）
  if (bestNamedWrite && bestNamedWrite.content.length >= STATE_MD_MIN_LEN) {
    return { content: bestNamedWrite.content.replace(/\[x\]/g, "[ ]"), source: bestNamedWrite.basename };
  }
  // 优先级 4：scan（内容特征 fallback）
  if (bestScan && bestScan.content.length >= STATE_MD_MIN_LEN) {
    return { content: bestScan.content.replace(/\[x\]/g, "[ ]"), source: `scan:${bestScan.basename}` };
  }
  return null;
}

/**
 * 找主 agent 的 workflow skill name：主 agent（isSubagent=false）的首个 Skill invoke 事件。
 * S1 → ops-registry-invoke-glacier，S2 → ops-registry-invoke-workflow。
 * 主 agent 通常先 invoke workflow skill（编排规程），再按它 dispatch 子 agent。
 * 比旧方案（turn0 ≥500 反查磁盘）更通用——不依赖 turn0 是注入型还是短用户任务。
 */
export async function resolveWorkflowSkillName(
  sessionId: string,
  prisma: PrismaClient,
): Promise<string | null> {
  const se = await prisma.skillEvent.findFirst({
    where: { eventType: "invoke", turn: { sessionId, isSubagent: false } },
    orderBy: { createdAt: "asc" },
  });
  return se?.skillName ?? null;
}

/**
 * 恢复 workflow skill 的 references 资源文件（task-prompts.md 等，具体编排模板）：
 * 主 agent Read 的文件里 filePath 含 skillName 的，取最长内容。
 * 泛化：不限文件名（task-prompts.md / prompts.md / ...），按 filePath 含 skill name 匹配。
 * 内容含每阶段 Task 调用参数（subagent_type + prompt + 验收标准）→ 外部声明的编排模板。
 */
async function recoverSkillResourceFile(
  sessionId: string,
  skillName: string,
  prisma: PrismaClient,
): Promise<PlanFileDeclaration | null> {
  const tcs = await prisma.toolCall.findMany({
    where: {
      turn: { sessionId },
      toolName: { in: ["read", "Read"] },
      argsJson: { contains: skillName },
    },
    select: { toolName: true, argsJson: true, resultJson: true },
    take: 200,
  });
  let best: { content: string; basename: string } | null = null;
  for (const tc of tcs) {
    const ex = extractFileContent(tc);
    if (!ex) continue;
    const fp = extractFilePath(tc.argsJson);
    // filePath 必须含 skillName 且含 "resources"（编排模板在 resources/ 目录，
    // 非 templates/ 目录的文档模板如 STATE.md.templ / DESIGN.md.templ）
    if (!fp || !fp.includes(skillName) || !fp.includes("resources")) continue;
    if (!best || ex.content.length > best.content.length) {
      best = { content: ex.content, basename: ex.basename ?? "unknown" };
    }
  }
  if (!best || best.content.length < STATE_MD_MIN_LEN) return null;
  return { content: best.content.replace(/\[x\]/g, "[ ]"), source: best.basename };
}

/**
 * 恢复 root 对账的 workflow 声明（编排规程）。
 *
 * 按 sift 设计意图（slice.py 注释："root 声明常是 workflow 级 SKILL.md"），
 * root 对账审的是"编排规程遵循度"，声明应该是 workflow skill 的编排规程——
 * 不是任务清单（STATE.md）或自写计划（todowrite），那些是"任务完成度"视角。
 *
 * 两种 workflow skill 加载方式：
 * - **S2 型（Skill invoke）**：主 agent 调用 Skill 工具加载 workflow skill。
 *   优先级 1：skill resources 文件（task-prompts.md 等，具体编排模板）。
 *   优先级 2：SKILL.md body（编排规程骨架，`<skill_content>`）。
 * - **S1 型（turn0 注入）**：workflow skill 正文注入首条 user turn（≥500 字），
 *   主 agent 不 invoke skill。turn0 = agent.md-like 编排规程。
 *
 * workflow skill 定位：主 agent Skill invoke（`resolveWorkflowSkillName`）；
 * 无 invoke → 退 turn0 ≥500（`recoverMainAgentWorkflowBody`）。
 * S1 → turn0 注入（1809 字编排规程），S2 → Skill invoke（ops-registry-invoke-workflow）。
 *
 * 返回 { content, source }：source = 文件名（resources）或 "skillName (SKILL.md)"。
 */
export async function recoverWorkflowDeclaration(
  sessionId: string,
  prisma: PrismaClient,
): Promise<PlanFileDeclaration | null> {
  const skillName = await resolveWorkflowSkillName(sessionId, prisma);

  // 有 Skill invoke（S2 型）：workflow skill 通过 Skill 调用加载
  if (skillName) {
    // 优先级 1：skill resources 文件（task-prompts.md 等，具体编排模板）
    const resource = await recoverSkillResourceFile(sessionId, skillName, prisma);
    if (resource) return resource;
    // 优先级 2：SKILL.md body（编排规程骨架）
    const body = await recoverSkillBody(sessionId, skillName, prisma);
    if (body && body.length >= STATE_MD_MIN_LEN) {
      return { content: body, source: `${skillName} (SKILL.md)` };
    }
  }

  // 无 Skill invoke（S1 型）：workflow skill 注入 turn0（编排规程在首条 user turn）
  // turn0 ≥500 字 = 注入的 agent.md-like 编排规程，不是短用户任务
  const turn0 = await recoverMainAgentWorkflowBody(sessionId, prisma);
  if (turn0) {
    return { content: turn0, source: "turn0 (注入编排规程)" };
  }

  return null;
}

/**
 * 旧接口（向后兼容）：返回计划文件正文字符串（无 source）。新调用点用
 * `recoverPlanFileDeclaration` 拿 source。本函数是薄包装，逻辑全在
 * `recoverPlanFileDeclaration`。
 */
export async function recoverStateMdPlan(
  sessionId: string,
  prisma: PrismaClient,
): Promise<string | null> {
  const r = await recoverPlanFileDeclaration(sessionId, prisma);
  return r?.content ?? null;
}

/**
 * 按 session + skillName 恢复该 skill 的纯 SKILL.md 正文：
 * 找该 session 内这个 skill 的 invoke SkillEvent → 同 turn 里按 argsJson.name 匹配的
 * skill ToolCall → 从 resultJson 剥 <skill_content>。
 * 找不到 / 无正文 → null。
 */
export async function recoverSkillBody(
  sessionId: string,
  skillName: string,
  prisma: PrismaClient,
): Promise<string | null> {
  const se = await prisma.skillEvent.findFirst({
    where: { skillName, eventType: "invoke", turn: { sessionId } },
    orderBy: { createdAt: "asc" },
  });
  if (!se) return null;

  const tcs = await prisma.toolCall.findMany({
    where: { turnId: se.turnId, isSkillRelated: true },
  });
  const candidates = tcs.filter((tc) => isSkillInvokeToolCall(tc.toolName));
  const byName = candidates.find((tc) => extractSkillNameFromArgs(tc.argsJson) === skillName);
  const tc = byName ?? candidates[0];
  if (!tc?.resultJson) return null;

  return stripSkillContent(tc.resultJson);
}

/**
 * Skill audit 板块的 target 列表：从 session 已有的 skillEvents 派生本 session 所有可对账的
 * 声明单元，同名合并。每行按 events 的 eventType 路由对账面：
 *
 * - skill 面：eventType=invoke/use → skill 对账（正文从 session 的 resultJson 恢复）
 * - agent 面：eventType=dispatch → agent 对账（.md 声明从 AGENTS_SCAN_ROOT 本地扫描，
 *   见 audit-agentsift 路由 + agent-md-scan.ts）
 *
 * 注：root 面（主 agent 编排 对账）**不在此派生**——主 agent 通常只 dispatch、不 invoke
 * skill，其 workflow 声明是 session 首条 user turn 的注入系统提示。root 目标由 SkillAuditTab /
 * SkillDetail 按 turn0 长度阈值合成添加（见 audit-skillsift kind=root +
 * recoverMainAgentWorkflowBody）。
 *
 * dual-nature（同既 invoke 又 dispatch，如 st-verifier）两面都返，UI 出多个条目各审各的
 * （与 sift 的 kind 分路一致——同名 skill 声明 ≠ agent 声明）。
 *
 * 输入是 page.tsx 已加载的内存数据（不查 DB），鸭子类型兼容 SkillEventForDetail。
 */
export interface AuditableTarget {
  name: string;
  kinds: ("skill" | "agent" | "root")[];
  skillEventCount: number;
}

export function deriveAuditableTargets(
  skillEvents: { skillName: string; eventType: string }[],
): AuditableTarget[] {
  const byName = new Map<string, { skillEvents: { eventType: string }[] }>();
  for (const se of skillEvents) {
    const name = se.skillName;
    if (!name) continue;
    const entry = byName.get(name) ?? { skillEvents: [] };
    entry.skillEvents.push(se);
    byName.set(name, entry);
  }
  const targets: AuditableTarget[] = [];
  for (const [name, e] of byName) {
    const kinds = auditKindsForEvents(e.skillEvents);
    if (kinds.length === 0) continue;
    targets.push({ name, kinds, skillEventCount: e.skillEvents.length });
  }
  targets.sort((a, b) => a.name.localeCompare(b.name));
  return targets;
}
