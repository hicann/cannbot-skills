// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { WorkflowTree, WorkflowPhaseNode, WorkflowStepNode, WorkflowCheckpointNode, WorkflowParallelGroupNode, WorkflowSummary } from '@/lib/ingest/phase-split';

export interface AIProviderConfig {
  baseUrl: string;
  apiKey: string;
  model: string;
}

interface AIPhaseAssignment {
  phaseIndex: number;
  phaseName: string;
  turnRanges: Array<{
    startTurnIndex: number;
    endTurnIndex: number;
    subPhaseName?: string;
    iterationIndex?: number;
    parallelGroupId?: string;
    checkpoint?: {
      checkpointType: "block" | "info";
      checkpointLabel: string;
      requestTurnIndex: number;
      approveTurnIndex?: number;
    };
  }>;
}

interface AIPhaseAssignmentResult {
  phases: AIPhaseAssignment[];
}

const SYSTEM_PROMPT = `你是一个 AI Agent 工作流阶段划分专家。你的唯一任务是根据 Coding Agent 的执行对话，准确划分工作流的阶段和子阶段。

你将收到精简的 Turn 列表（仅包含 root agent 的 assistant 和 user 对话）和 Subagent dispatch 记录。请分析这些数据，输出阶段划分结果。

输出 JSON 格式：
{
  "phases": [
    {
      "phaseIndex": 1,
      "phaseName": "阶段名称（中文）",
      "turnRanges": [
        {
          "startTurnIndex": 0,
          "endTurnIndex": 5,
          "subPhaseName": "子阶段/步骤名称（中文，可选）",
          "iterationIndex": null,
          "parallelGroupId": null,
          "checkpoint": null
        },
        {
          "startTurnIndex": 3,
          "endTurnIndex": 3,
          "checkpoint": {
            "checkpointType": "block",
            "checkpointLabel": "CP1 用户审批",
            "requestTurnIndex": 3,
            "approveTurnIndex": 4
          }
        }
      ]
    }
  ]
}

划分要求：
1. 所有名称用中文
2. 根据语义内容划分阶段，不要依赖关键词标记
3. 每个阶段覆盖一段连续的 turnIndex 范围（可以不覆盖所有 turn，未覆盖的视为过渡）
4. 同一阶段内可包含多个子步骤（subPhaseName）
5. 如果发现迭代循环（反复尝试、逐步修正），标注 iterationIndex
6. 如果发现并行执行，标注 parallelGroupId（如 "pg-1"）
7. 如果发现检查点（用户审批、确认步骤），标注 checkpoint
8. turnRanges 的 startTurnIndex ≤ endTurnIndex，范围不能重叠（checkpoint 例外）
9. 阶段之间 turn 范围可以不连续
10. 严格输出 JSON，不要包含 markdown 代码块标记`;

const MAX_INPUT_CHARS = 30000;

function buildTurnDigest(turns: Array<{
  turnIndex: number;
  role: string;
  agentName: string | null;
  isSubagent: boolean;
  subagentName: string | null;
  contentSummary: string | null;
  createdAt_ts: string | null;
  totalTokens: number;
  inputTokens: number;
  outputTokens: number;
  latencyMs: number;
  model: string | null;
  toolCalls: Array<{ toolName: string; state: string }>;
}>): string {
  const filtered = turns.filter(t =>
    !t.isSubagent && (t.role === "assistant" || t.role === "user" || t.role === "system")
  );

  const baseTime = filtered.find(t => t.createdAt_ts)?.createdAt_ts;
  const baseMs = baseTime ? new Date(baseTime).getTime() : 0;

  const lines = filtered.map(t => {
    const relMin = baseMs > 0 && t.createdAt_ts
      ? `${((new Date(t.createdAt_ts).getTime() - baseMs) / 60000).toFixed(1)}min`
      : "-";
    const content = (t.contentSummary ?? "").substring(0, 80);
    const tools = t.toolCalls.length > 0
      ? t.toolCalls.map(tc => `${tc.toolName}(${tc.state})`).join(",")
      : "";
    const roleTag = t.role === "user" ? "👤" : t.role === "system" ? "⚙" : "🤖";
    const tok = t.totalTokens > 0 ? `tok:${t.totalTokens}` : "";
    const lat = t.latencyMs > 100 ? `lat:${(t.latencyMs / 1000).toFixed(1)}s` : "";

    return `#${t.turnIndex} ${roleTag} +${relMin} ${tok} ${lat} ${tools ? `tools:[${tools}]` : ""} "${content}"`;
  });

  let digest = lines.join("\n");
  if (digest.length > MAX_INPUT_CHARS) {
    const kept = [];
    let totalLen = 0;
    for (const line of lines) {
      if (totalLen + line.length > MAX_INPUT_CHARS) break;
      kept.push(line);
      totalLen += line.length;
    }
    const lastTurn = filtered[kept.length - 1];
    digest = kept.join("\n") + `\n... (截断, 仅显示到 turn #${lastTurn?.turnIndex ?? "?"})`;
  }

  return digest;
}

function buildBridgeDigest(bridges: Array<{
  id: string;
  dispatchContent: string | null;
  dispatchTimestamp: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  subagentTokens: number;
  subagentLatencyMs: number;
  dispatchTurnId: string | null;
}>): string {
  return bridges.map((b, i) => {
    const dispatch = (b.dispatchContent ?? "").substring(0, 80);
    const time = b.dispatchTimestamp
      ? new Date(b.dispatchTimestamp).toLocaleTimeString("en", { hour: "2-digit", minute: "2-digit" })
      : "-";
    return `dispatch#${i} [${time}] [${b.subagentType ?? "-"}] [${b.subagentName ?? "-"}] ` +
      `"${dispatch}" → ${b.status} tok:${b.subagentTokens}`;
  }).join("\n");
}

function buildUserPrompt(
  turnDigest: string,
  bridgeDigest: string,
  turnsCount: number,
  bridgesCount: number,
  totalTurnsCount: number,
): string {
  const turnsNote = turnsCount < totalTurnsCount
    ? `(精简为 ${turnsCount} 条，仅 root agent 的 assistant/user/system)`
    : `(${turnsCount} 条)`;

  return `## Turn 列表 ${turnsNote}
每条格式: #序号 role标记 +相对时间 tokens latency tools "摘要"
${turnDigest}

## Subagent Dispatch 记录 (${bridgesCount} 条)
每条格式: dispatch#序号 [时间] [type] [name] "任务" → status tokens
${bridgeDigest}

请划分此 session 的工作流阶段，输出 JSON 结果。`;
}

export async function analyzeWorkflow(
  taskId: string,
  provider: AIProviderConfig,
  prisma: import("@prisma/client").PrismaClient,
): Promise<WorkflowTree> {
  const session = await prisma.session.findFirst({ where: { taskId } });
  if (!session) throw new Error(`Session not found: ${taskId}`);

  const turns = await prisma.turn.findMany({
    where: { sessionId: session.id },
    orderBy: [{ turnIndex: "asc" }],
    include: {
      toolCalls: {
        select: { toolName: true, state: true },
      },
    },
  });

  const bridges = await prisma.interactionBridge.findMany({
    where: { sessionId: session.id },
  });

  const turnDigest = buildTurnDigest(turns.map(t => ({
    turnIndex: t.turnIndex,
    role: t.role,
    agentName: t.agentName,
    isSubagent: t.isSubagent,
    subagentName: t.subagentName,
    contentSummary: t.contentSummary ?? t.content?.substring(0, 80) ?? null,
    createdAt_ts: t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString(),
    totalTokens: t.totalTokens,
    inputTokens: t.inputTokens,
    outputTokens: t.outputTokens,
    latencyMs: t.latencyMs,
    model: t.model,
    toolCalls: t.toolCalls.map(tc => ({ toolName: tc.toolName, state: tc.state })),
  })));

  const filteredTurnsCount = turns.filter(t =>
    !t.isSubagent && (t.role === "assistant" || t.role === "user" || t.role === "system")
  ).length;

  const bridgeDigest = buildBridgeDigest(bridges.map(b => ({
    id: b.id,
    dispatchContent: b.dispatchContent,
    dispatchTimestamp: b.dispatchTimestamp?.toISOString() ?? null,
    subagentType: b.subagentType,
    subagentName: b.subagentName,
    status: b.status,
    subagentTokens: b.subagentTokens,
    subagentLatencyMs: b.subagentLatencyMs,
    dispatchTurnId: b.dispatchTurnId,
  })));

  const userPrompt = buildUserPrompt(turnDigest, bridgeDigest, filteredTurnsCount, bridges.length, turns.length);

  const apiBase = provider.baseUrl.replace(/\/+$/, "");
  const chatUrl = apiBase.endsWith("/v1") || apiBase.includes("/v1/")
    ? `${apiBase}/chat/completions`
    : `${apiBase}/v1/chat/completions`;

  const response = await fetch(chatUrl, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${provider.apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: provider.model,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
      temperature: 0.3,
      max_tokens: 4096,
      response_format: { type: "json_object" },
    }),
    signal: AbortSignal.timeout(180000),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`LLM API error: ${response.status} ${errText}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;
  if (!content) throw new Error("LLM API returned empty content");

  const assignment: AIPhaseAssignmentResult = JSON.parse(content);

  const turnById = new Map(turns.map(t => [t.id, t]));
  const turnByIndex = new Map(turns.map(t => [t.turnIndex, t]));
  const bridgeByDispatchTurnId = new Map<string, typeof bridges>();
  for (const b of bridges) {
    if (b.dispatchTurnId) {
      const arr = bridgeByDispatchTurnId.get(b.dispatchTurnId) ?? [];
      arr.push(b);
      bridgeByDispatchTurnId.set(b.dispatchTurnId, arr);
    }
  }

  const phases: WorkflowPhaseNode[] = [];
  let globalStepCounter = 0;
  let globalCheckpointCounter = 0;
  let globalParallelGroupCounter = 0;

  for (const aiPhase of assignment.phases) {
    const children: Array<WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode> = [];
    const phaseTurns: typeof turns = [];
    let phaseStartTime: string | null = null;
    let phaseEndTime: string | null = null;
    let phaseWaitTimeMs = 0;

    const parallelGroups = new Map<string, WorkflowParallelGroupNode>();

    for (const range of aiPhase.turnRanges) {
      if (range.checkpoint) {
        globalCheckpointCounter++;
        const cp = range.checkpoint;
        const requestTurn = turnByIndex.get(cp.requestTurnIndex);
        const approveTurn = cp.approveTurnIndex ? turnByIndex.get(cp.approveTurnIndex) : null;
        const requestedAt = requestTurn?.createdAt_ts?.toISOString() ?? requestTurn?.createdAt.toISOString() ?? null;
        const approvedAt = approveTurn?.createdAt_ts?.toISOString() ?? approveTurn?.createdAt.toISOString() ?? null;
        let waitTimeMs = 0;
        if (requestedAt && approvedAt) {
          waitTimeMs = new Date(approvedAt).getTime() - new Date(requestedAt).getTime();
        }
        const checkpointNode: WorkflowCheckpointNode = {
          type: 'checkpoint',
          checkpointIndex: globalCheckpointCounter,
          checkpointType: cp.checkpointType,
          checkpointLabel: cp.checkpointLabel,
          requestedAt,
          approvedAt,
          waitTimeMs,
          triggerTurnId: requestTurn?.id ?? null,
          responseTurnId: approveTurn?.id ?? null,
        };
        children.push(checkpointNode);
        phaseWaitTimeMs += waitTimeMs;
        continue;
      }

      for (let ti = range.startTurnIndex; ti <= range.endTurnIndex; ti++) {
        const turn = turnByIndex.get(ti);
        if (!turn) continue;
        phaseTurns.push(turn);
      }

      const turnBridgesInRange: typeof bridges = [];
      for (let ti = range.startTurnIndex; ti <= range.endTurnIndex; ti++) {
        const turn = turnByIndex.get(ti);
        if (!turn) continue;
        const bArr = bridgeByDispatchTurnId.get(turn.id) ?? [];
        turnBridgesInRange.push(...bArr);
      }

      for (const bridge of turnBridgesInRange) {
        globalStepCounter++;
        const dispatchTurn = bridge.dispatchTurnId ? turnById.get(bridge.dispatchTurnId) : null;
        const stepName = bridge.dispatchContent ?? `Step ${globalStepCounter}`;
        const step: WorkflowStepNode = {
          type: 'step',
          stepIndex: globalStepCounter,
          stepName,
          stepLabel: range.subPhaseName ? `${range.subPhaseName}: ${stepName}` : stepName,
          iterationIndex: range.iterationIndex ?? null,
          iterationName: range.iterationIndex ? `迭代${range.iterationIndex}` : null,
          startTime: dispatchTurn?.createdAt_ts?.toISOString() ?? dispatchTurn?.createdAt.toISOString() ?? bridge.dispatchTimestamp?.toISOString() ?? null,
          endTime: bridge.responseTimestamp?.toISOString() ?? null,
          durationMs: bridge.subagentLatencyMs,
          totalTokens: bridge.subagentTokens,
          totalCost: 0,
          toolCallCount: 0,
          bridgeId: bridge.id,
          subagentSessionId: bridge.subagentSessionId,
          subagentType: bridge.subagentType,
          subagentName: bridge.subagentName,
          status: bridge.status,
          parallelGroupId: range.parallelGroupId ?? null,
          triggerTurnId: dispatchTurn?.id ?? null,
        };

        if (range.parallelGroupId) {
          const pgId = range.parallelGroupId;
          let group = parallelGroups.get(pgId);
          if (!group) {
            globalParallelGroupCounter++;
            group = {
              type: 'parallel-group',
              groupId: pgId,
              label: `并行组${globalParallelGroupCounter}`,
              steps: [],
              totalDurationMs: 0,
              totalTokens: 0,
            };
            parallelGroups.set(pgId, group);
            children.push(group);
          }
          group.steps.push(step);
          group.totalTokens += step.totalTokens;
          if (step.durationMs > group.totalDurationMs) {
            group.totalDurationMs = step.durationMs;
          }
        } else {
          children.push(step);
        }
      }
    }

    const allPhaseSteps = collectSteps(children);
    const allPhaseCheckpoints = children.filter(c => c.type === 'checkpoint') as WorkflowCheckpointNode[];

    const stepTokens = allPhaseSteps.reduce((s, st) => s + st.totalTokens, 0);
    const rootTokens = phaseTurns.filter(t => !t.isSubagent).reduce((s, t) => s + t.totalTokens, 0);
    const totalTokens = stepTokens + rootTokens;
    const toolCallCount = allPhaseSteps.reduce((s, st) => s + st.toolCallCount, 0);

    const times = [
      ...allPhaseSteps.map(s => s.startTime).filter((t): t is string => t !== null),
      ...allPhaseSteps.map(s => s.endTime).filter((t): t is string => t !== null),
      ...allPhaseCheckpoints.map(c => c.requestedAt).filter((t): t is string => t !== null),
      ...allPhaseCheckpoints.map(c => c.approvedAt).filter((t): t is string => t !== null),
      ...phaseTurns.map(t => t.createdAt_ts?.toISOString() ?? t.createdAt.toISOString()),
    ];

    let durationMs = 0;
    if (times.length > 0) {
      const sortedMs = times.map(t => new Date(t).getTime()).sort((a, b) => a - b);
      phaseStartTime = new Date(sortedMs[0]).toISOString();
      phaseEndTime = new Date(sortedMs[sortedMs.length - 1]).toISOString();
      durationMs = sortedMs[sortedMs.length - 1] - sortedMs[0];
    }

    const activeTimeMs = Math.max(0, durationMs - phaseWaitTimeMs);

    const phaseNode: WorkflowPhaseNode = {
      phaseIndex: aiPhase.phaseIndex,
      phaseSequence: aiPhase.phaseIndex,
      phaseName: aiPhase.phaseName,
      fullLabel: `阶段${aiPhase.phaseIndex}：${aiPhase.phaseName}`,
      startTime: phaseStartTime,
      endTime: phaseEndTime,
      durationMs,
      activeTimeMs,
      waitTimeMs: phaseWaitTimeMs,
      totalTokens,
      totalCost: 0,
      toolCallCount,
      subagentCount: allPhaseSteps.filter(st => st.bridgeId !== null).length,
      triggerTurnId: phaseTurns[0]?.id ?? null,
      turnIndexStart: null,
      turnIndexEnd: null,
      children,
    };

    phases.push(phaseNode);
  }

  const validPhases = phases.filter(p => p.children.length > 0 || p.totalTokens > 0);
  const allSteps = collectSteps(validPhases.flatMap(p => p.children) as Array<WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode>);
  const allCheckpoints = validPhases.flatMap(p => p.children.filter(c => c.type === 'checkpoint') as WorkflowCheckpointNode[]);
  const totalDurationMs = validPhases.reduce((s, p) => s + p.durationMs, 0);
  const totalWaitTimeMs = validPhases.reduce((s, p) => s + p.waitTimeMs, 0);
  const totalActiveTimeMs = Math.max(0, totalDurationMs - totalWaitTimeMs);
  const iterations = new Set(allSteps.filter(s => s.iterationIndex !== null).map(s => s.iterationIndex!)).size;

  return {
    phases: validPhases,
    summary: {
      totalPhases: validPhases.length,
      totalSteps: allSteps.length,
      totalCheckpoints: allCheckpoints.length,
      totalActiveTimeMs,
      totalWaitTimeMs,
      activeTimePct: totalDurationMs > 0 ? Math.round((totalActiveTimeMs / totalDurationMs) * 100) : 0,
      iterations,
    },
  };
}

function collectSteps(children: Array<WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode>): WorkflowStepNode[] {
  const steps: WorkflowStepNode[] = [];
  for (const child of children) {
    if (child.type === 'step') steps.push(child);
    if (child.type === 'parallel-group') steps.push(...child.steps);
  }
  return steps;
}

export { buildTurnDigest, buildBridgeDigest, SYSTEM_PROMPT, buildUserPrompt };
