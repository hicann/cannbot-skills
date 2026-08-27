// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { PrismaClient, Prisma } from '@prisma/client';
import Database from 'better-sqlite3';
import fs from 'node:fs';
import { BRAND_SOURCE_TYPE } from '@/lib/branding';
import { getAdapter } from './adapters/index';
import { listSubagentSessionsWithDb, readSessionWithDb, readSessionMeta } from './adapters/opencode-db';
import { listSubagentSessions as listClaudeSubagentSessions, collectSubagentToolUseMappings as collectClaudeToolUseMappings, extractVersion as extractClaudeVersion } from './adapters/claude-jsonl';
import { proxySourceOf } from './adapters/claude-jsonl-full-context';
import { normalize } from './normalize';
import { splitIntoTurns, resetIdCounter } from './turn-split';
import type { TurnRow, ToolCallRow, SkillEventRow } from './turn-split';
import { dedupSession, mergeToolCalls, mergeSkillEvents, diffTurns, diffToolCalls, rankedKeyMap, keyByRankedTurn } from './merge';
import { buildBridges, resetIdCounter as resetBridgeIdCounter } from './bridge-builder';
import { splitExecutions, resetIdCounter as resetExecIdCounter } from './execution-split';
import type { ExecutionRow } from './execution-split';
import type { InteractionBridgeRow } from './bridge-builder';
import type { RawInteraction } from '../shared/types';

function toDate(v: string | null): Date | null {
  return v ? new Date(v) : null;
}

// proxy 文件的 round-pair 切分（claude-jsonl-wire）只产 user/assistant
// 两种 role 的 turn；原生切分会产 system role（system-reminder 拆分、
// 注册表等）。两种形态的 turnIndex:role 键对不上，直接 merge 会新旧
// 混杂 —— 删除旧 turns 整体重建。仅在「旧有 system turn 且 新无 system
// turn」时触发（= 切分逻辑从原生切到了 wire 切），原生→原生、wire→wire
// 的增量导入不受影响。
// 连带删除 Execution/InteractionBridge/SessionSkill/ExecutionSkill：
// 这些表的外键引用旧 turnId，reshape 后 turnId 全变 → 必须清掉防悬空。
// merge 路径不重建这些（deltaRefresh 路径会重建）—— reshape 后 session
// 暂无 execution/bridge 数据，但无悬空引用，不崩。
async function rebuildIfWireReshape(
  client: PrismaClient,
  sessionId: string,
  existingTurns: Array<{ id: string; role: string }>,
  freshTurns: TurnRow[],
): Promise<boolean> {
  if (existingTurns.length === 0) return false;
  const existingHasSystem = existingTurns.some(t => t.role === 'system');
  const freshHasSystem = freshTurns.some(t => t.role === 'system');
  if (!existingHasSystem || freshHasSystem) return false;
  await client.turn.deleteMany({ where: { sessionId } });
  await client.execution.deleteMany({ where: { sessionId } });
  await client.interactionBridge.deleteMany({ where: { sessionId } });
  await client.sessionSkill.deleteMany({ where: { sessionId } });
  return true;
}

async function batchCreateMany(
  tx: Prisma.TransactionClient,
  model: string,
  data: unknown[],
  batchSize: number = 500,
): Promise<void> {
  if (!Array.isArray(data) || data.length === 0) return;
  for (let i = 0; i < data.length; i += batchSize) {
    const batch = data.slice(i, i + batchSize);
    await (tx as any)[model].createMany({ data: batch });
  }
}

interface ExecutionSkillData {
  skillName: string;
  skillVersion: number | null;
  isPrimary: boolean;
  user: string | null;
}

function computeExecutionSkills(
  executions: ExecutionRow[],
  turns: TurnRow[],
  skillEvents: SkillEventRow[]
): Map<string, ExecutionSkillData[]> {
  const result = new Map<string, ExecutionSkillData[]>();

  for (const execution of executions) {
    const executionTurns = execution.isSubagent
      ? turns.filter(t => t.subagentSessionId === execution.agentSessionId)
      : turns.filter(t => !t.isSubagent);

    const executionTurnIds = new Set(executionTurns.map(t => t.id));
    const executionSkillEvents = skillEvents.filter(se => executionTurnIds.has(se.turnId));

    const uniqueSkillNames = [...new Set(executionSkillEvents.map(se => se.skillName))];
    const skills: ExecutionSkillData[] = uniqueSkillNames.map(skillName => {
      const loadEvent = executionSkillEvents.find(
        se => se.skillName === skillName && se.eventType === 'load'
      );
      return {
        skillName,
        skillVersion: loadEvent?.skillVersion ?? null,
        isPrimary: false,
        user: null,
      };
    });

    result.set(execution.id, skills);
  }

  return result;
}

// Execution / ExecutionSkill / InteractionBridge / SessionSkill 的重建数据。
// 三条写入路径（新建、importSession merge、deltaRefresh）共用：merge 路径
// 早期只 merge turns 不重建派生数据，subagent 有 Turn 无 Execution（Bug #5/#6）。
// turns 的 id 须已换成 DB id（已存在的行），否则聚合挂不上 toolCalls。
interface ExecutionGraphData {
  executionsData: Record<string, unknown>[];
  executionSkillsData: Array<{ executionId: string; skillName: string; skillVersion: number | null; isPrimary: boolean; user: string | null }>;
  bridgesData: Record<string, unknown>[];
  sessionSkillsData: Array<{ sessionId?: string; skillName: string; skillVersion: number | null; invocationCount: number }>;
  rootExecutionId: string | null;
}

function buildExecutionGraph(
  turns: TurnRow[],
  toolCalls: ToolCallRow[],
  skillEvents: SkillEventRow[],
  interactions: RawInteraction[],
  taskId: string,
  toolUseIdMapping?: Map<string, string>,
): ExecutionGraphData {
  const executions = splitExecutions(turns, toolCalls, skillEvents, taskId);
  const rootExecutionId = executions.find(e => !e.isSubagent)?.id ?? null;
  const bridges: InteractionBridgeRow[] = rootExecutionId
    ? buildBridges(interactions, toolCalls, turns, taskId, rootExecutionId, toolUseIdMapping)
    : [];

  const executionsData = executions.map(e => ({
    id: e.id,
    agentName: e.agentName,
    agentSessionId: e.agentSessionId,
    isSubagent: e.isSubagent,
    subagentType: e.subagentType,
    subagentName: e.subagentName,
    parentExecutionId: e.parentExecutionId,
    rootExecutionId: e.rootExecutionId,
    depth: e.depth,
    tokens: e.tokens,
    inputTokens: e.inputTokens,
    outputTokens: e.outputTokens,
    reasoningTokens: e.reasoningTokens,
    cacheReadInputTokens: e.cacheReadInputTokens,
    cacheCreationInputTokens: e.cacheCreationInputTokens,
    maxSingleCallTokens: e.maxSingleCallTokens,
    cost: e.cost,
    latencyMs: e.latencyMs,
    createdAt: new Date(e.createdAt),
    toolCallCount: e.toolCallCount,
    toolCallErrorCount: e.toolCallErrorCount,
    llmCallCount: e.llmCallCount,
    skillLoadCount: e.skillLoadCount,
    skillInvokeCount: e.skillInvokeCount,
    finalResult: e.finalResult,
    model: e.model,
  }));

  const executionSkillsMap = computeExecutionSkills(executions, turns, skillEvents);
  const executionSkillsData: ExecutionGraphData['executionSkillsData'] = [];
  for (const [execId, skills] of executionSkillsMap) {
    for (const es of skills) {
      executionSkillsData.push({
        executionId: execId,
        skillName: es.skillName,
        skillVersion: es.skillVersion,
        isPrimary: es.isPrimary,
        user: es.user,
      });
    }
  }

  const bridgesData = bridges.map(b => ({
    dispatchExecutionId: b.dispatchExecutionId,
    dispatchTurnId: b.dispatchTurnId,
    dispatchToolCallId: b.dispatchToolCallId,
    dispatchContent: b.dispatchContent,
    dispatchTimestamp: toDate(b.dispatchTimestamp),
    responseExecutionId: b.responseExecutionId,
    responseTurnId: b.responseTurnId,
    responseContent: b.responseContent,
    responseTimestamp: toDate(b.responseTimestamp),
    subagentSessionId: b.subagentSessionId,
    subagentType: b.subagentType,
    subagentName: b.subagentName,
    status: b.status,
    subagentTokens: b.subagentTokens,
    subagentLatencyMs: b.subagentLatencyMs,
  }));

  const uniqueSkillNames = [...new Set(skillEvents.map(se => se.skillName))];
  const sessionSkillsData = uniqueSkillNames.map(skillName => {
    const invocationCount = skillEvents.filter(
      se => se.skillName === skillName && (se.eventType === 'invoke' || se.eventType === 'use' || se.eventType === 'dispatch')
    ).length;
    const loadEvent = skillEvents.find(se => se.skillName === skillName && se.eventType === 'load');
    return {
      skillName,
      skillVersion: loadEvent?.skillVersion ?? null,
      invocationCount,
    };
  });

  return { executionsData, executionSkillsData, bridgesData, sessionSkillsData, rootExecutionId };
}

async function replaceExecutionGraph(
  tx: Prisma.TransactionClient,
  prismaSessionId: string,
  graph: ExecutionGraphData,
): Promise<void> {
  await tx.execution.deleteMany({ where: { sessionId: prismaSessionId } });
  await batchCreateMany(tx, 'Execution' as Prisma.ModelName,
    graph.executionsData.map(e => ({ ...e, sessionId: prismaSessionId })));

  await batchCreateMany(tx, 'ExecutionSkill' as Prisma.ModelName, graph.executionSkillsData);

  await tx.interactionBridge.deleteMany({ where: { sessionId: prismaSessionId } });
  await batchCreateMany(tx, 'InteractionBridge' as Prisma.ModelName,
    graph.bridgesData.map(b => ({ ...b, sessionId: prismaSessionId })));

  await tx.sessionSkill.deleteMany({ where: { sessionId: prismaSessionId } });
  await batchCreateMany(tx, 'SessionSkill' as Prisma.ModelName,
    graph.sessionSkillsData.map(s => ({ ...s, sessionId: prismaSessionId })));
}

export function computeSessionAggregates(
  turns: TurnRow[],
  toolCalls: ToolCallRow[],
  skillEvents: SkillEventRow[],
): {
  totalTokens: number;
  totalInputTokens: number;
  totalOutputTokens: number;
  totalReasoningTokens: number;
  totalCacheReadTokens: number;
  totalCacheWriteTokens: number;
  totalCost: number;
  totalLatencyMs: number;
  totalToolCallCount: number;
  totalLlmCallCount: number;
  totalSkillLoadCount: number;
  totalSubagentCount: number;
  startTime: Date;
  endTime: Date | null;
  model: string | null;
} {
  let totalTokens = 0;
  let totalInputTokens = 0;
  let totalOutputTokens = 0;
  let totalReasoningTokens = 0;
  let totalCacheReadTokens = 0;
  let totalCacheWriteTokens = 0;
  let totalCost = 0;
  let totalLatencyMs = 0;
  let totalLlmCallCount = 0;

  const startTime = turns.length > 0 && turns[0].createdAt_ts
    ? new Date(turns[0].createdAt_ts)
    : new Date();

  let endTime: Date | null = null;
  for (const turn of turns) {
    if (turn.completedAt) {
      const d = new Date(turn.completedAt);
      if (!endTime || d > endTime) endTime = d;
    } else if (turn.createdAt_ts) {
      const d = new Date(turn.createdAt_ts);
      if (!endTime || d > endTime) endTime = d;
    }
  }

  let model: string | null = null;
  for (const turn of turns) {
    if (turn.role === 'assistant' && turn.model) {
      model = turn.model;
      break;
    }
  }

  for (const turn of turns) {
    totalTokens += turn.totalTokens;
    totalInputTokens += turn.inputTokens;
    totalOutputTokens += turn.outputTokens;
    totalReasoningTokens += turn.reasoningTokens;
    totalCacheReadTokens += turn.cacheReadTokens;
    totalCacheWriteTokens += turn.cacheWriteTokens;
    if (turn.role === 'assistant') {
      totalLatencyMs += turn.latencyMs;
    }
    if (turn.role === 'assistant' && turn.totalTokens > 0) {
      totalLlmCallCount++;
      totalCost += turn.cost;
    }
  }

  const uniqueSubagentIds = new Set<string>();
  for (const turn of turns) {
    if (turn.isSubagent && turn.subagentSessionId) {
      uniqueSubagentIds.add(turn.subagentSessionId);
    }
  }

  const totalToolCallCount = toolCalls.length;
  const totalSkillLoadCount = skillEvents.length;
  const totalSubagentCount = uniqueSubagentIds.size;

  return {
    totalTokens,
    totalInputTokens,
    totalOutputTokens,
    totalReasoningTokens,
    totalCacheReadTokens,
    totalCacheWriteTokens,
    totalCost,
    totalLatencyMs,
    totalToolCallCount,
    totalLlmCallCount,
    totalSkillLoadCount,
    totalSubagentCount,
    startTime,
    endTime,
    model,
  };
}

export async function importSession(
  dbPath: string,
  sessionId: string,
  prisma?: PrismaClient,
  sourcePath?: string,
  sourceType?: string
): Promise<{ sessionId: string; imported: boolean; query: string | null }> {
  const client = prisma ?? new PrismaClient();
  const srcType = sourceType ?? 'opencode-db';
  const t0 = Date.now();

  // For opencode-db source, open DB once and reuse the connection
  let sharedDb: Database.Database | null = null;
  if (srcType === 'opencode-db') {
    try {
      sharedDb = new Database(dbPath, { readonly: true });
    } catch {
      return { sessionId, imported: false, query: null };
    }
  }

  try {
    const t1 = Date.now();
    const adapter = getAdapter(srcType);
    if (!adapter) {
      throw new Error(`No adapter available for source type: "${srcType}"`);
    }

    const rawInteractions = srcType === 'opencode-db' && sharedDb
      ? readSessionWithDb(sharedDb, sessionId)
      : adapter.readSession(dbPath, sessionId);
    console.log(`[import] readSession: ${Date.now() - t1}ms, ${rawInteractions.length} interactions`);

    let sessionMeta: { parentId: string | null; version: string | null; directory: string | null; summaryAdditions: number; summaryDeletions: number; summaryFiles: number } = { parentId: null, version: null, directory: null, summaryAdditions: 0, summaryDeletions: 0, summaryFiles: 0 };
    // proxy 来源标记单独留存：merge 路径的 version 回填只用它（原生文件
    // merge 行为保持不变 —— 不回填 version），新 session 路径仍用完整 version
    let proxyVersionMarker: string | null = null;
    if (srcType === 'opencode-db' && sharedDb) {
      sessionMeta = readSessionMeta(sharedDb, sessionId);
    } else if (srcType === 'claude-jsonl') {
      // cpx 捕获文件每行带 source 标记（claude-proxy / opencode-proxy）：
      // 来源随文件走（改名/移动不失效）。标记优先于 jsonl 里的 version 字段
      // 存入 Session.version，列表据此显示 proxy 徽标。
      proxyVersionMarker = proxySourceOf(dbPath);
      // 版本号与来源标记都保留：'<agent版本>-<marker>'（如 2.1.234.467-claude-proxy）。
      // endsWith('-proxy') 的判定不受影响；显示层剥后缀取回版本号。
      const agentVersion = extractClaudeVersion(dbPath);
      sessionMeta.version = proxyVersionMarker
        ? (agentVersion ? `${agentVersion}-${proxyVersionMarker}` : proxyVersionMarker)
        : agentVersion;
    }
    // framework 是 agent 归属：opencode 捕获（opencode-proxy 标记）归
    // opencode，与捕获文件格式（claude-jsonl）正交 —— 列表/详情据此显示
    // OpenCode 而非 Claude。
    const frameworkForCapture = srcType === 'claude-jsonl' && (proxyVersionMarker?.endsWith('opencode-proxy') ?? false)
      ? 'opencode'
      : srcType === 'claude-jsonl' ? 'claude-code'
      : srcType;

    if (rawInteractions.length === 0) {
      if (sharedDb) sharedDb.close();
      // Empty claude capture file: the proxy touches it at cpx startup, so
      // importing right after launch (or while waiting for the first user
      // input) reads zero records. Create a PLACEHOLDER session so it's
      // visible in the list; the merge path (refresh-session / re-import)
      // fills in turns as the file grows.
      if (srcType === 'claude-jsonl' && fs.existsSync(dbPath)) {
        const existing = await client.session.findFirst({ where: { taskId: sessionId }, select: { id: true, query: true } });
        if (existing) return { sessionId, imported: false, query: existing.query ?? null };
        await client.session.create({
          data: {
            taskId: sessionId,
            label: '（捕获中）',
            framework: frameworkForCapture,
            startTime: new Date(),
            sourcePath,
          },
        });
        return { sessionId, imported: true, query: null };
      }
      return { sessionId, imported: false, query: null };
    }

    const t2 = Date.now();
    const allRawInteractions = [...rawInteractions];
    let toolUseIdMapping: Map<string, string> | undefined = undefined;

    if (srcType === 'opencode-db' && sharedDb) {
      const subagentSessionIds = listSubagentSessionsWithDb(sharedDb, sessionId);
      for (const subId of subagentSessionIds) {
        const subInteractions = readSessionWithDb(sharedDb, subId);
        allRawInteractions.push(...subInteractions);
      }
    } else if (srcType === 'claude-jsonl') {
      const subagentFiles = listClaudeSubagentSessions(dbPath, sessionId);
      toolUseIdMapping = collectClaudeToolUseMappings(dbPath, sessionId);
      for (const sub of subagentFiles) {
        const metaPath = sub.filePath.replace('.jsonl', '.meta.json');
        let subName: string | null = null;
        let subType: string | null = null;
        try {
          if (fs.existsSync(metaPath)) {
            const meta = JSON.parse(fs.readFileSync(metaPath, 'utf-8'));
            subName = meta.name || meta.agentType || meta.description || null;
            subType = meta.agentType || null;
          }
        } catch {}

        const subInteractions = adapter.readSession(sub.filePath, sub.id);
        for (const interaction of subInteractions) {
          interaction.subagent_session_id = sub.id;
          if (subName) interaction.subagent_name = subName;
          if (subType) interaction.subagent_type = subType;
        }
        allRawInteractions.push(...subInteractions);
      }
    }
    console.log(`[import] subagents: ${Date.now() - t2}ms, ${allRawInteractions.length} total interactions`);

    const t3 = Date.now();
    const normalized = normalize(allRawInteractions, srcType);
    resetIdCounter();
    resetBridgeIdCounter();
    resetExecIdCounter();

    const { turns, toolCalls, skillEvents } = splitIntoTurns(normalized, sessionId);
    console.log(`[import] normalize+split: ${Date.now() - t3}ms, ${turns.length} turns, ${toolCalls.length} toolCalls, ${skillEvents.length} skillEvents`);

    const t4 = Date.now();
    // claude-jsonl（含两种 proxy 捕获）按 taskId 找已存 session，不按
    // framework 过滤 —— 同一捕获文件的 framework 归属曾从 claude-code 修正
    // 为 opencode，按 framework 过滤会让老数据 miss 掉去新建重复行。
    const existingSession = await client.session.findFirst({
      where: {
        taskId: sessionId,
        ...(srcType === 'claude-jsonl' ? {} : {
          framework: srcType === 'opencode-db' ? 'opencode' : srcType,
        }),
      },
    });

    const dedupResult = dedupSession(existingSession?.id ?? null, sessionId);
    console.log(`[import] dedup check: ${Date.now() - t4}ms, shouldImport=${dedupResult.shouldImport}`);

    // ── 增量导入路径（dedup/merge）──
    if (!dedupResult.shouldImport && dedupResult.existingSessionId) {
      const t5 = Date.now();
      let existingTurns = await client.turn.findMany({
        where: { sessionId: dedupResult.existingSessionId },
      });
      if (await rebuildIfWireReshape(client, dedupResult.existingSessionId, existingTurns, turns)) {
        existingTurns = [];
      }
      const existingToolCalls = await client.toolCall.findMany({
        where: { turnId: { in: existingTurns.map(t => t.id) } },
      });
      const existingSkillEvents = await client.skillEvent.findMany({
        where: { turnId: { in: existingTurns.map(t => t.id) } },
      });

      const existingSessionPrismaId = dedupResult.existingSessionId!;

      const existingTurnByRankedKey = keyByRankedTurn(existingTurns);
      const incomingRankedKeys = rankedKeyMap(turns);

      const newTurns = turns
        .filter(t => !existingTurnByRankedKey.has(incomingRankedKeys.get(t)!))
        .map(t => ({ ...t, sessionId: existingSessionPrismaId }));

      const turnIdRemap = new Map<string, string>();
      for (const turn of turns) {
        const existingDbId = existingTurnByRankedKey.get(incomingRankedKeys.get(turn)!)?.id;
        if (existingDbId) {
          turnIdRemap.set(turn.id, existingDbId);
        }
      }

      const newTurnsData = newTurns.map(t => {
        const { cost: _turnCost, subagentType: _subagentType, ...rest } = t;
        return { ...rest, createdAt_ts: toDate(rest.createdAt_ts), completedAt: toDate(rest.completedAt) };
      });

      const remapTurnId = (id: string): string => turnIdRemap.get(id) ?? id;

      const remappedToolCalls = toolCalls.map(tc => ({ ...tc, turnId: remapTurnId(tc.turnId) }));
      const remappedSkillEvents = skillEvents.map(se => ({ ...se, turnId: remapTurnId(se.turnId) }));
      const turnsForGraph = turns.map(t => {
        const dbId = turnIdRemap.get(t.id);
        return dbId ? { ...t, id: dbId } : t;
      });
      const executionGraph = buildExecutionGraph(
        turnsForGraph, remappedToolCalls, remappedSkillEvents,
        normalized as unknown as RawInteraction[], sessionId, toolUseIdMapping,
      );

      const mergedToolCallRows = mergeToolCalls(
        existingToolCalls.map(tc => ({
          id: tc.id,
          turnId: tc.turnId,
          toolCallId: tc.toolCallId,
          toolName: tc.toolName,
          argsJson: tc.argsJson,
          resultJson: tc.resultJson,
          state: tc.state,
          errorType: tc.errorType,
          errorMessage: tc.errorMessage,
          startedAt: tc.startedAt?.toISOString() ?? null,
          completedAt: tc.completedAt?.toISOString() ?? null,
          durationMs: tc.durationMs,
          dispatchBridgeId: tc.dispatchBridgeId,
          isSkillRelated: tc.isSkillRelated,
        })),
        remappedToolCalls
      );

      const newToolCalls = mergedToolCallRows.filter(
        mc => !existingToolCalls.some(ec => ec.toolCallId === mc.toolCallId)
      ).map(mc => ({ ...mc, turnId: remapTurnId(mc.turnId) }));

      const newToolCallsData = newToolCalls.map(tc => ({
        ...tc,
        startedAt: toDate(tc.startedAt),
        completedAt: toDate(tc.completedAt),
      }));

      const mergedSkillEventRows = mergeSkillEvents(
        existingSkillEvents.map(se => ({
          id: se.id,
          turnId: se.turnId,
          skillName: se.skillName,
          skillVersion: se.skillVersion,
          eventType: se.eventType,
          success: se.success,
          errorMessage: se.errorMessage,
          argsJson: se.argsJson,
          startedAt: se.startedAt?.toISOString() ?? null,
          completedAt: se.completedAt?.toISOString() ?? null,
          durationMs: se.durationMs,
        })),
        remappedSkillEvents
      );

      const newSkillEvents = mergedSkillEventRows.filter(
        ms => !existingSkillEvents.some(es =>
          es.turnId === ms.turnId && es.skillName === ms.skillName && es.eventType === ms.eventType
        )
      ).map(ms => ({ ...ms, turnId: remapTurnId(ms.turnId) }));

      const newSkillEventsData = newSkillEvents.map(se => ({
        ...se,
        startedAt: toDate(se.startedAt),
        completedAt: toDate(se.completedAt),
      }));

      await client.$transaction(async (tx) => {
        await batchCreateMany(tx, 'Turn' as Prisma.ModelName, newTurnsData);
        await batchCreateMany(tx, 'ToolCall' as Prisma.ModelName, newToolCallsData);
        await batchCreateMany(tx, 'SkillEvent' as Prisma.ModelName, newSkillEventsData);
        await replaceExecutionGraph(tx, existingSessionPrismaId, executionGraph);
      }, { maxWait: 30000, timeout: 60000 });

      // Update session aggregates from all turns (old + new)
      const allTurnsAfterMerge = await client.turn.findMany({ where: { sessionId: existingSessionPrismaId } });
      const allToolCallsAfterMerge = await client.toolCall.findMany({
        where: { turnId: { in: allTurnsAfterMerge.map(t => t.id) } },
      });
      const allSkillEventsAfterMerge = await client.skillEvent.findMany({
        where: { turnId: { in: allTurnsAfterMerge.map(t => t.id) } },
      });

      const turnRowsForAgg = allTurnsAfterMerge.map(t => ({
        id: t.id,
        sessionId: t.sessionId,
        turnIndex: t.turnIndex,
        role: t.role,
        totalTokens: t.totalTokens,
        inputTokens: t.inputTokens,
        outputTokens: t.outputTokens,
        reasoningTokens: t.reasoningTokens,
        cacheReadTokens: t.cacheReadTokens,
        cacheWriteTokens: t.cacheWriteTokens,
        cost: 0,
        isSubagent: t.isSubagent,
        subagentSessionId: t.subagentSessionId ?? null,
        createdAt_ts: t.createdAt_ts ? t.createdAt_ts.toISOString() : null,
        completedAt: t.completedAt ? t.completedAt.toISOString() : null,
      }));
      const updatedAggregates = computeSessionAggregates(
        turnRowsForAgg,
        allToolCallsAfterMerge as unknown as ToolCallRow[],
        allSkillEventsAfterMerge as unknown as SkillEventRow[],
      );

      const query = allTurnsAfterMerge.find(t => t.role === 'user')?.content?.substring(0, 200) ?? null;
      const safeNum = (n: number) => (typeof n === 'number' && !isNaN(n)) ? n : 0;
      const safeDate = (d: Date | null) => d instanceof Date && !isNaN(d.getTime()) ? d : null;
      const updateData = {
        totalTokens: safeNum(updatedAggregates.totalTokens),
        totalInputTokens: safeNum(updatedAggregates.totalInputTokens),
        totalOutputTokens: safeNum(updatedAggregates.totalOutputTokens),
        totalReasoningTokens: safeNum(updatedAggregates.totalReasoningTokens),
        totalCacheReadTokens: safeNum(updatedAggregates.totalCacheReadTokens),
        totalCacheWriteTokens: safeNum(updatedAggregates.totalCacheWriteTokens),
        totalCost: safeNum(updatedAggregates.totalCost),
        totalLatencyMs: safeNum(updatedAggregates.totalLatencyMs),
        totalToolCallCount: safeNum(updatedAggregates.totalToolCallCount),
        totalLlmCallCount: safeNum(updatedAggregates.totalLlmCallCount),
        totalSkillLoadCount: safeNum(updatedAggregates.totalSkillLoadCount),
        totalSubagentCount: safeNum(updatedAggregates.totalSubagentCount),
        endTime: safeDate(updatedAggregates.endTime),
        rootExecutionId: executionGraph.rootExecutionId,
        ...(query ? { query } : {}),
        ...(updatedAggregates.model ? { model: updatedAggregates.model } : {}),
        // 占位 session（空捕获文件导入时创建）没有 version；merge 时补上
        // proxy 来源标记。仅 proxy：原生文件 merge 不回填 version（行为不变）
        ...((!existingSession.version && proxyVersionMarker) ? { version: proxyVersionMarker } : {}),
        // 占位 label（捕获中）在文件填满后换成真实首问，不再永久显示占位文案
        ...((query && (existingSession.label === '（捕获中）' || existingSession.label === '（捕获中，暂无内容）'))
          ? { label: query.substring(0, 100) }
          : {}),
      };
      await client.session.update({
        where: { id: existingSessionPrismaId },
        data: updateData,
      });

      console.log(`[import] merge path total: ${Date.now() - t5}ms, overall: ${Date.now() - t0}ms`);
      return { sessionId: dedupResult.existingSessionId, imported: false, query: query };
    }

    // ── 新建 session 路径 ──
    const t6 = Date.now();
    const executionGraph = buildExecutionGraph(
      turns, toolCalls, skillEvents,
      normalized as unknown as RawInteraction[], sessionId, toolUseIdMapping,
    );
    const rootExecutionId = executionGraph.rootExecutionId;

    const aggregates = computeSessionAggregates(turns, toolCalls, skillEvents);

    const createdSessionId = await client.$transaction(async (tx) => {
      const ts0 = Date.now();
      const sessionRow = await tx.session.create({
        data: {
          taskId: sessionId,
          label: rawInteractions.find(i => i.role === 'user')?.content?.substring(0, 100)
            ?? rawInteractions[0]?.content?.substring(0, 100) ?? null,
          query: rawInteractions.find(i => i.role === 'user')?.content?.substring(0, 200) ?? null,
          framework: srcType === 'claude-jsonl' ? frameworkForCapture : srcType === BRAND_SOURCE_TYPE ? srcType : 'opencode',
          model: aggregates.model,
          startTime: aggregates.startTime,
          endTime: aggregates.endTime,
          totalTokens: aggregates.totalTokens,
          totalInputTokens: aggregates.totalInputTokens,
          totalOutputTokens: aggregates.totalOutputTokens,
          totalReasoningTokens: aggregates.totalReasoningTokens,
          totalCacheReadTokens: aggregates.totalCacheReadTokens,
          totalCacheWriteTokens: aggregates.totalCacheWriteTokens,
          totalCost: aggregates.totalCost,
          totalLatencyMs: aggregates.totalLatencyMs,
          totalToolCallCount: aggregates.totalToolCallCount,
          totalLlmCallCount: aggregates.totalLlmCallCount,
          totalSkillLoadCount: aggregates.totalSkillLoadCount,
          totalSubagentCount: aggregates.totalSubagentCount,
          rootExecutionId: rootExecutionId,
          sourcePath: sourcePath ?? dbPath,
          parentId: sessionMeta.parentId,
          version: sessionMeta.version,
          directory: sessionMeta.directory,
          summaryAdditions: sessionMeta.summaryAdditions,
          summaryDeletions: sessionMeta.summaryDeletions,
          summaryFiles: sessionMeta.summaryFiles,
        },
      });
      const sid = sessionRow.id;
      console.log(`[import] session.create: ${Date.now() - ts0}ms`);

      const turnsData = turns.map(t => {
        const { cost: _turnCost, subagentType: _subagentType, ...rest } = t;
        return { ...rest, sessionId: sid, createdAt_ts: toDate(rest.createdAt_ts), completedAt: toDate(rest.completedAt) };
      });

      const toolCallsData = toolCalls.map(tc => ({
        ...tc,
        startedAt: toDate(tc.startedAt),
        completedAt: toDate(tc.completedAt),
      }));

      const skillEventsData = skillEvents.map(se => ({
        ...se,
        startedAt: toDate(se.startedAt),
        completedAt: toDate(se.completedAt),
      }));

      const ts1 = Date.now();
      await batchCreateMany(tx, 'Turn' as Prisma.ModelName, turnsData);
      await batchCreateMany(tx, 'ToolCall' as Prisma.ModelName, toolCallsData);
      await batchCreateMany(tx, 'SkillEvent' as Prisma.ModelName, skillEventsData);
      await replaceExecutionGraph(tx, sid, executionGraph);
      console.log(`[import] sessionSkill.createMany: ${executionGraph.sessionSkillsData.length} rows`);

      return sid;
    }, { maxWait: 30000, timeout: 60000 });

    console.log(`[import] new path total: ${Date.now() - t6}ms, overall: ${Date.now() - t0}ms`);
    return { sessionId: createdSessionId, imported: true, query: rawInteractions.find(i => i.role === 'user')?.content?.substring(0, 200) ?? null };
  } finally {
    if (sharedDb) sharedDb.close();
    if (!prisma) await client.$disconnect();
  }
}

export async function deltaRefreshSession(
  sessionId: string,
  prisma: PrismaClient,
): Promise<{ addedTurns: number; updatedTurns: number; addedToolCalls: number; updatedToolCalls: number; addedSkillEvents: number }> {
  const session = await prisma.session.findFirst({ where: { taskId: sessionId } });
  if (!session || !session.sourcePath) {
    throw new Error(`Session not found or no sourcePath: "${sessionId}"`);
  }

  const sourceType = session.framework === 'opencode' && session.version?.endsWith('-proxy')
    ? 'claude-jsonl'
    : session.framework === 'opencode' ? 'opencode-db'
    : session.framework === 'claude-code' ? 'claude-jsonl'
    : session.framework;

  const srcType = sourceType === 'opencode-db' ? 'opencode-db'
    : sourceType === 'claude-jsonl' ? 'claude-jsonl'
    : sourceType;

  let sharedDb: Database.Database | null = null;
  if (srcType === 'opencode-db') {
    sharedDb = new Database(session.sourcePath, { readonly: true });
  }

  const oldTurnCount = await prisma.turn.count({ where: { sessionId: session.id } });

  try {
    const adapter = getAdapter(srcType);
    if (!adapter) throw new Error(`No adapter for: "${srcType}"`);

    const rawInteractions = srcType === 'opencode-db' && sharedDb
      ? readSessionWithDb(sharedDb, sessionId)
      : adapter.readSession(session.sourcePath, sessionId);

    const allRawInteractions = [...rawInteractions];
    let toolUseIdMapping: Map<string, string> | undefined = undefined;

    if (srcType === 'opencode-db' && sharedDb) {
      const subIds = listSubagentSessionsWithDb(sharedDb, sessionId);
      for (const subId of subIds) {
        allRawInteractions.push(...readSessionWithDb(sharedDb, subId));
      }
    } else if (srcType === 'claude-jsonl') {
      const subagentFiles = listClaudeSubagentSessions(session.sourcePath, sessionId);
      toolUseIdMapping = collectClaudeToolUseMappings(session.sourcePath, sessionId);
      for (const sub of subagentFiles) {
        const subInteractions = adapter.readSession(sub.filePath, sub.id);
        for (const interaction of subInteractions) {
          interaction.subagent_session_id = sub.id;
        }
        allRawInteractions.push(...subInteractions);
      }
    }

    if (allRawInteractions.length === 0) {
      return { addedTurns: 0, updatedTurns: 0, addedToolCalls: 0, updatedToolCalls: 0, addedSkillEvents: 0 };
    }

    const normalized = normalize(allRawInteractions, srcType);
    resetIdCounter();
    resetBridgeIdCounter();
    resetExecIdCounter();

    const { turns, toolCalls, skillEvents } = splitIntoTurns(normalized, sessionId);

    let existingTurns = await prisma.turn.findMany({ where: { sessionId: session.id } });
    if (await rebuildIfWireReshape(prisma, session.id, existingTurns, turns)) {
      existingTurns = [];
    }
    const existingToolCalls = await prisma.toolCall.findMany({
      where: { turnId: { in: existingTurns.map(t => t.id) } },
    });
    const existingSkillEvents = await prisma.skillEvent.findMany({
      where: { turnId: { in: existingTurns.map(t => t.id) } },
    });

    const existingTurnRows: TurnRow[] = existingTurns.map(et => ({
        id: et.id,
        sessionId: et.sessionId,
        turnIndex: et.turnIndex,
        role: et.role,
        content: et.content,
        contentJson: et.contentJson,
        contentSummary: et.contentSummary,
        inputMessagesJson: et.inputMessagesJson,
        inputMessagesCount: et.inputMessagesCount,
        inputMessagesTokens: et.inputMessagesTokens,
        contextWindowPct: et.contextWindowPct,
        agentName: et.agentName,
        subagentName: et.subagentName,
        subagentSessionId: et.subagentSessionId,
        subagentType: null,
        totalTokens: et.totalTokens,
        inputTokens: et.inputTokens,
        outputTokens: et.outputTokens,
        reasoningTokens: et.reasoningTokens,
        cacheReadTokens: et.cacheReadTokens,
        cacheWriteTokens: et.cacheWriteTokens,
        cost: 0,
        createdAt_ts: et.createdAt_ts?.toISOString() ?? null,
        completedAt: et.completedAt?.toISOString() ?? null,
        latencyMs: et.latencyMs,
        ttftMs: et.ttftMs,
        model: et.model,
        modelId: et.modelId,
        providerId: et.providerId,
        temperature: et.temperature,
        maxTokens: et.maxTokens,
        finishReason: et.finishReason,
        isSubagent: et.isSubagent,
        parentExecutionId: et.parentExecutionId,
    }));
    const existingTurnByRankedKey = keyByRankedTurn(existingTurnRows);

    const existingTcByToolCallId = new Map<string, ToolCallRow>();
    for (const tc of existingToolCalls) {
      existingTcByToolCallId.set(tc.toolCallId, {
        id: tc.id,
        turnId: tc.turnId,
        toolCallId: tc.toolCallId,
        toolName: tc.toolName,
        argsJson: tc.argsJson,
        resultJson: tc.resultJson,
        state: tc.state,
        errorType: tc.errorType,
        errorMessage: tc.errorMessage,
        startedAt: tc.startedAt?.toISOString() ?? null,
        completedAt: tc.completedAt?.toISOString() ?? null,
        durationMs: tc.durationMs,
        dispatchBridgeId: tc.dispatchBridgeId,
        isSkillRelated: tc.isSkillRelated,
      });
    }

    const incomingRankedKeys = rankedKeyMap(turns);
    const turnIdRemap = new Map<string, string>();
    for (const turn of turns) {
      const existingDbId = existingTurnByRankedKey.get(incomingRankedKeys.get(turn)!)?.id;
      if (existingDbId) {
        turnIdRemap.set(turn.id, existingDbId);
      }
    }

    const remapTurnId = (id: string): string => turnIdRemap.get(id) ?? id;

    const { toInsert: newTurns, toUpdate: turnUpdates } = diffTurns(existingTurnRows, turns);

    const remappedToolCalls = toolCalls.map(tc => ({ ...tc, turnId: remapTurnId(tc.turnId) }));
    const remappedSkillEvents = skillEvents.map(se => ({ ...se, turnId: remapTurnId(se.turnId) }));

    const { toInsert: newToolCalls, toUpdate: tcUpdates } = diffToolCalls(existingTcByToolCallId, remappedToolCalls);

    const existingSeKeys = new Set(
      existingSkillEvents.map(se => `${se.turnId}:${se.skillName}:${se.eventType}`)
    );
    const newSkillEvents = remappedSkillEvents.filter(se => !existingSeKeys.has(`${se.turnId}:${se.skillName}:${se.eventType}`));

    const newTurnsData = newTurns.map(t => {
      const { cost: _turnCost, subagentType: _subagentType, ...rest } = t;
      return { ...rest, sessionId: session.id, createdAt_ts: toDate(rest.createdAt_ts), completedAt: toDate(rest.completedAt) };
    });

    const newToolCallsData = newToolCalls.map(tc => ({
      ...tc,
      startedAt: toDate(tc.startedAt),
      completedAt: toDate(tc.completedAt),
    }));

    const newSkillEventsData = newSkillEvents.map(se => ({
      ...se,
      startedAt: toDate(se.startedAt),
      completedAt: toDate(se.completedAt),
    }));

    const appliedTurnUpdates = turnUpdates.map(u => {
      const mapped: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(u.data)) {
        if (k === 'completedAt' || k === 'createdAt_ts') {
          mapped[k] = toDate(v as string | null);
        } else {
          mapped[k] = v;
        }
      }
      return { dbId: u.dbId, data: mapped };
    });

    const volatileFields: (keyof TurnRow)[] = [
      'inputMessagesCount', 'inputMessagesTokens', 'contextWindowPct',
      'inputMessagesJson', 'contentJson',
    ];
    const matchedTurnVolatileUpdates: Array<{ dbId: string; data: Record<string, unknown> }> = [];
    for (const turn of turns) {
      const existingId = existingTurnByRankedKey.get(incomingRankedKeys.get(turn)!)?.id;
      if (!existingId) continue;
      const alreadyInDiffUpdate = appliedTurnUpdates.some(u => u.dbId === existingId);
      if (alreadyInDiffUpdate) {
        for (const f of volatileFields) {
          appliedTurnUpdates.find(u => u.dbId === existingId)!.data[f] = turn[f];
        }
      } else {
        const volatileData: Record<string, unknown> = {};
        let hasVolatileChange = false;
        const existingTurn = existingTurnByRankedKey.get(incomingRankedKeys.get(turn)!)!;
        for (const f of volatileFields) {
          if (turn[f] !== existingTurn[f]) {
            volatileData[f] = turn[f];
            hasVolatileChange = true;
          }
        }
        if (hasVolatileChange) {
          matchedTurnVolatileUpdates.push({ dbId: existingId, data: volatileData });
        }
      }
    }

    const appliedTcUpdates = tcUpdates.map(u => {
      const mapped: Record<string, unknown> = {};
      for (const [k, v] of Object.entries(u.data)) {
        if (k === 'completedAt' || k === 'startedAt') {
          mapped[k] = toDate(v as string | null);
        } else {
          mapped[k] = v;
        }
      }
      return { dbId: u.dbId, data: mapped };
    });

    const turnsForGraph = turns.map(t => {
      const dbId = turnIdRemap.get(t.id);
      return dbId ? { ...t, id: dbId } : t;
    });
    const executionGraph = buildExecutionGraph(
      turnsForGraph, remappedToolCalls, remappedSkillEvents,
      normalized as unknown as RawInteraction[], sessionId, toolUseIdMapping,
    );
    const rootExecutionId = executionGraph.rootExecutionId;

    await prisma.$transaction(async (tx) => {
      await batchCreateMany(tx, 'Turn' as Prisma.ModelName, newTurnsData);
      for (const u of appliedTurnUpdates) {
        await tx.turn.update({ where: { id: u.dbId }, data: u.data });
      }
      for (const u of matchedTurnVolatileUpdates) {
        await tx.turn.update({ where: { id: u.dbId }, data: u.data });
      }

      await batchCreateMany(tx, 'ToolCall' as Prisma.ModelName, newToolCallsData);
      for (const u of appliedTcUpdates) {
        await tx.toolCall.update({ where: { id: u.dbId }, data: u.data });
      }

      await batchCreateMany(tx, 'SkillEvent' as Prisma.ModelName, newSkillEventsData);

      await replaceExecutionGraph(tx, session.id, executionGraph);
    }, { maxWait: 30000, timeout: 60000 });

    const allTurnsAfterMerge = await prisma.turn.findMany({ where: { sessionId: session.id } });
    const allToolCallsAfterMerge = await prisma.toolCall.findMany({
      where: { turnId: { in: allTurnsAfterMerge.map(t => t.id) } },
    });
    const allSkillEventsAfterMerge = await prisma.skillEvent.findMany({
      where: { turnId: { in: allTurnsAfterMerge.map(t => t.id) } },
    });

    const turnRowsForAgg = allTurnsAfterMerge.map(t => ({
      id: t.id,
      sessionId: t.sessionId,
      turnIndex: t.turnIndex,
      role: t.role,
      totalTokens: t.totalTokens,
      inputTokens: t.inputTokens,
      outputTokens: t.outputTokens,
      reasoningTokens: t.reasoningTokens,
      cacheReadTokens: t.cacheReadTokens,
      cacheWriteTokens: t.cacheWriteTokens,
      cost: 0,
      isSubagent: t.isSubagent,
      subagentSessionId: t.subagentSessionId ?? null,
      createdAt_ts: t.createdAt_ts ? t.createdAt_ts.toISOString() : null,
      completedAt: t.completedAt ? t.completedAt.toISOString() : null,
    }));
    const updatedAggregates = computeSessionAggregates(
      turnRowsForAgg,
      allToolCallsAfterMerge as unknown as ToolCallRow[],
      allSkillEventsAfterMerge as unknown as SkillEventRow[],
    );

    const safeNum = (n: number) => (typeof n === 'number' && !isNaN(n)) ? n : 0;
    const safeDate = (d: Date | null) => d instanceof Date && !isNaN(d.getTime()) ? d : null;
    const query = allTurnsAfterMerge.find(t => t.role === 'user')?.content?.substring(0, 200) ?? null;

    await prisma.session.update({
      where: { id: session.id },
      data: {
        totalTokens: safeNum(updatedAggregates.totalTokens),
        totalInputTokens: safeNum(updatedAggregates.totalInputTokens),
        totalOutputTokens: safeNum(updatedAggregates.totalOutputTokens),
        totalReasoningTokens: safeNum(updatedAggregates.totalReasoningTokens),
        totalCacheReadTokens: safeNum(updatedAggregates.totalCacheReadTokens),
        totalCacheWriteTokens: safeNum(updatedAggregates.totalCacheWriteTokens),
        totalCost: safeNum(updatedAggregates.totalCost),
        totalLatencyMs: safeNum(updatedAggregates.totalLatencyMs),
        totalToolCallCount: safeNum(updatedAggregates.totalToolCallCount),
        totalLlmCallCount: safeNum(updatedAggregates.totalLlmCallCount),
        totalSkillLoadCount: safeNum(updatedAggregates.totalSkillLoadCount),
        totalSubagentCount: safeNum(updatedAggregates.totalSubagentCount),
        endTime: safeDate(updatedAggregates.endTime),
        rootExecutionId,
        ...(query ? { query } : {}),
        ...(updatedAggregates.model ? { model: updatedAggregates.model } : {}),
        // 占位 session 补来源标记（空捕获导入时创建的 session 无 version）
        ...((srcType === 'claude-jsonl' && !session.version && session.sourcePath)
          ? { version: proxySourceOf(session.sourcePath) ?? undefined }
          : {}),
      },
    });

    const newTurnCount = allTurnsAfterMerge.length;
    return {
      addedTurns: Math.max(0, newTurnCount - oldTurnCount),
      updatedTurns: appliedTurnUpdates.length,
      addedToolCalls: newToolCallsData.length,
      updatedToolCalls: appliedTcUpdates.length,
      addedSkillEvents: newSkillEventsData.length,
    };
  } finally {
    if (sharedDb) sharedDb.close();
  }
}
