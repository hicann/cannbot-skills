// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { TurnRow, SkillEventRow } from './turn-split';
import type { InteractionBridgeRow } from './bridge-builder';
import { getFamilyKey, getFamilyLabel, getPhaseGapTurns, getPhaseInfo, type PhaseInfo } from '@/lib/skill-family-config';

export interface WorkflowTree {
  phases: WorkflowPhaseNode[];
  summary: WorkflowSummary;
}

export interface WorkflowSummary {
  totalPhases: number;
  totalSteps: number;
  totalCheckpoints: number;
  totalActiveTimeMs: number;
  totalWaitTimeMs: number;
  activeTimePct: number;
  iterations: number;
}

export interface WorkflowTurnNode {
  type: 'turn';
  turnId: string;
  turnIndex: number;
  role: string;
  contentSummary: string | null;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  totalTokens: number;
}

export interface WorkflowPhaseNode {
  phaseIndex: number;
  phaseSequence: number;
  phaseName: string;
  fullLabel: string;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  activeTimeMs: number;
  waitTimeMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  subagentCount: number;
  triggerTurnId: string | null;
  turnIndexStart: number | null;
  turnIndexEnd: number | null;
  children: Array<WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode | WorkflowTurnNode>;
}

export interface WorkflowStepNode {
  type: 'step';
  stepIndex: number;
  stepName: string;
  stepLabel: string;
  iterationIndex: number | null;
  iterationName: string | null;
  startTime: string | null;
  endTime: string | null;
  durationMs: number;
  totalTokens: number;
  totalCost: number;
  toolCallCount: number;
  bridgeId: string | null;
  subagentSessionId: string | null;
  subagentType: string | null;
  subagentName: string | null;
  status: string;
  parallelGroupId: string | null;
  triggerTurnId: string | null;
}

export interface WorkflowCheckpointNode {
  type: 'checkpoint';
  checkpointIndex: number;
  checkpointType: 'block' | 'info';
  checkpointLabel: string;
  requestedAt: string | null;
  approvedAt: string | null;
  waitTimeMs: number;
  triggerTurnId: string | null;
  responseTurnId: string | null;
}

export interface WorkflowParallelGroupNode {
  type: 'parallel-group';
  groupId: string;
  label: string;
  steps: WorkflowStepNode[];
  totalDurationMs: number;
  totalTokens: number;
}

interface MappedEvent {
  se: SkillEventRow;
  skillName: string;
  bridge: InteractionBridgeRow | null;
  rootTriggerTurnId: string | null;
  rootAnchorTurnIndex: number | null;
  rootAnchorTime: string | null;
}

interface OrderedSkillEvent {
  se: SkillEventRow;
  turnId: string;
  turnIndex: number;
  skillName: string;
  family: string;
}

const PHASE_NUM_MAP: Record<string, number> = { 一: 1, 二: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10 };
const PHASE_RE = /阶段([一二三四五六七八九十1-9]\d?)/g;
const FUTURE_RE = /后续|继续|可以|可继续|将|准备|未到|暂|若|可能|即将|预|下一步/;

function parsePhaseNum(s: string): number | null {
  if (/^\d+$/.test(s)) return parseInt(s, 10);
  return PHASE_NUM_MAP[s] ?? null;
}

function stripThinking(text: string | null): string {
  if (!text) return '';
  return text.replace(/<thinking>[\s\S]*?<\/thinking>/g, '');
}

interface PhaseStart { key: string; turn: number; info: PhaseInfo }

function detectPhaseStarts(rootAssistantTurns: TurnRow[]): PhaseStart[] | null {
  let hasAnyMarker = false;
  const reached = new Map<string, number>();
  for (const turn of rootAssistantTurns) {
    const nt = stripThinking(turn.content);
    if (!nt) continue;
    let m: RegExpExecArray | null;
    PHASE_RE.lastIndex = 0;
    while ((m = PHASE_RE.exec(nt)) !== null) {
      hasAnyMarker = true;
      const num = parsePhaseNum(m[1]);
      if (num === null || num === 1) continue;
      const prev = nt.substring(Math.max(0, m.index - 12), m.index);
      if (FUTURE_RE.test(prev)) continue;
      if (!reached.has(m[1])) reached.set(m[1], turn.turnIndex);
    }
  }
  if (!hasAnyMarker) return null;

  const starts: PhaseStart[] = [{ key: '一', turn: 0, info: getPhaseInfo('一') ?? { key: '一', label: '阶段一', order: 1 } }];
  for (const [key, turn] of reached) {
    const info = getPhaseInfo(key) ?? { key, label: `阶段${key}`, order: parsePhaseNum(key) ?? 99 };
    starts.push({ key, turn, info });
  }
  starts.sort((a, b) => a.turn - b.turn);
  return starts;
}

export function splitWorkflow(
  turns: TurnRow[],
  bridges: InteractionBridgeRow[],
  skillEvents: SkillEventRow[],
  sessionId: string,
): WorkflowTree {
  if (skillEvents.length === 0) {
    return buildFallbackTree(turns, bridges, sessionId);
  }

  const turnById = new Map(turns.map(t => [t.id, t]));
  const bridgeByDispatchTurnId = new Map<string, InteractionBridgeRow[]>();
  const bridgeBySubagentSessionId = new Map<string, InteractionBridgeRow[]>();
  for (const b of bridges) {
    if (b.dispatchTurnId) {
      const a1 = bridgeByDispatchTurnId.get(b.dispatchTurnId) ?? [];
      a1.push(b);
      bridgeByDispatchTurnId.set(b.dispatchTurnId, a1);
    }
    if (b.subagentSessionId) {
      const a2 = bridgeBySubagentSessionId.get(b.subagentSessionId) ?? [];
      a2.push(b);
      bridgeBySubagentSessionId.set(b.subagentSessionId, a2);
    }
  }

  const rootAssistantTurns = turns
    .filter(t => !t.isSubagent && t.role === 'assistant')
    .sort((a, b) => a.turnIndex - b.turnIndex);

  const mapped: MappedEvent[] = [];
  for (const se of skillEvents) {
    const turn = turnById.get(se.turnId);
    if (!turn) continue;

    let bridge: InteractionBridgeRow | null = null;
    let rootTriggerTurnId: string | null = null;
    let rootAnchorTurnIndex: number | null = null;
    let rootAnchorTime: string | null = null;

    if (turn.isSubagent) {
      const cands = (turn.subagentSessionId ? bridgeBySubagentSessionId.get(turn.subagentSessionId) : undefined) ?? [];
      bridge = cands.find(b => b.subagentType === se.skillName || b.subagentName === se.skillName) ?? cands[0] ?? null;
      const rootTurn = bridge?.dispatchTurnId ? turnById.get(bridge.dispatchTurnId) : undefined;
      rootTriggerTurnId = bridge?.dispatchTurnId ?? null;
      rootAnchorTurnIndex = rootTurn?.turnIndex ?? null;
      rootAnchorTime = bridge?.dispatchTimestamp ?? rootTurn?.createdAt_ts ?? turn.createdAt_ts ?? null;
    } else {
      const cands = bridgeByDispatchTurnId.get(turn.id) ?? [];
      bridge = cands.find(b => b.subagentType === se.skillName || b.subagentName === se.skillName) ?? null;
      rootTriggerTurnId = turn.id;
      rootAnchorTurnIndex = turn.turnIndex;
      rootAnchorTime = turn.createdAt_ts ?? null;
    }

    mapped.push({ se, skillName: se.skillName, bridge, rootTriggerTurnId, rootAnchorTurnIndex, rootAnchorTime });
  }

  if (mapped.length === 0) {
    return buildFallbackTree(turns, bridges, sessionId);
  }

  const phaseStarts = detectPhaseStarts(rootAssistantTurns);
  if (phaseStarts) {
    return splitByMarkers(mapped, phaseStarts, turns);
  }
  return splitBySkillFamily(turns, bridges, skillEvents, sessionId, turnById, bridgeByDispatchTurnId);
}

function splitByMarkers(
  events: MappedEvent[],
  phaseStarts: PhaseStart[],
  turns: TurnRow[],
): WorkflowTree {
  const maxRootTurnIndex = turns
    .filter(t => !t.isSubagent)
    .reduce((m, t) => Math.max(m, t.turnIndex), -1);

  const ranges = phaseStarts.map((s, i) => ({
    start: s,
    from: s.turn,
    to: i + 1 < phaseStarts.length ? phaseStarts[i + 1].turn - 1 : maxRootTurnIndex,
  }));

  const assignPhase = (anchorIdx: number | null): number => {
    if (anchorIdx === null) return ranges.length - 1;
    for (let i = 0; i < ranges.length; i++) {
      if (anchorIdx >= ranges[i].from && anchorIdx <= ranges[i].to) return i;
    }
    return anchorIdx < ranges[0].from ? 0 : ranges.length - 1;
  };

  const phaseEvents: MappedEvent[][] = ranges.map(() => []);
  for (const e of events) {
    phaseEvents[assignPhase(e.rootAnchorTurnIndex)].push(e);
  }

  const phases: WorkflowPhaseNode[] = [];
  let stepCounter = 0;
  for (let pi = 0; pi < ranges.length; pi++) {
    const evs = phaseEvents[pi]
      .slice()
      .sort((a, b) =>
        (a.rootAnchorTurnIndex ?? 999999) - (b.rootAnchorTurnIndex ?? 999999)
        || (a.rootAnchorTime ?? '').localeCompare(b.rootAnchorTime ?? ''),
      );
    if (evs.length === 0) continue;

    const byTrigger = new Map<string, MappedEvent[]>();
    for (const e of evs) {
      const key = e.rootTriggerTurnId ?? `__orphan_${e.se.id}`;
      const arr = byTrigger.get(key) ?? [];
      arr.push(e);
      byTrigger.set(key, arr);
    }

    const children: Array<WorkflowStepNode | WorkflowParallelGroupNode> = [];
    for (const [triggerId, groupEvents] of byTrigger) {
      const isParallel = groupEvents.length > 1;
      const pgId = isParallel ? `pg-${ranges[pi].start.key}-${triggerId}` : null;
      const steps: WorkflowStepNode[] = groupEvents.map(e => {
        stepCounter++;
        const br = e.bridge;
        const startTime = br?.dispatchTimestamp ?? e.rootAnchorTime ?? null;
        const endTime = br?.responseTimestamp ?? startTime;
        return {
          type: 'step',
          stepIndex: stepCounter,
          stepName: e.skillName,
          stepLabel: e.skillName,
          iterationIndex: null,
          iterationName: null,
          startTime,
          endTime,
          durationMs: (e.se.durationMs || br?.subagentLatencyMs) ?? 0,
          totalTokens: br?.subagentTokens ?? 0,
          totalCost: 0,
          toolCallCount: 0,
          bridgeId: br?.id ?? null,
          subagentSessionId: br?.subagentSessionId ?? null,
          subagentType: br?.subagentType ?? null,
          subagentName: br?.subagentName ?? null,
          status: br?.status ?? (e.se.success ? 'ok' : 'error'),
          parallelGroupId: pgId,
          triggerTurnId: e.rootTriggerTurnId,
        };
      });

      if (isParallel) {
        children.push({
          type: 'parallel-group',
          groupId: pgId!,
          label: ranges[pi].start.info.label,
          steps,
          totalDurationMs: Math.max(...steps.map(s => s.durationMs)),
          totalTokens: steps.reduce((s, x) => s + x.totalTokens, 0),
        });
      } else {
        children.push(...steps);
      }
    }

    const anchorIdxs = evs.map(e => e.rootAnchorTurnIndex).filter((x): x is number => x !== null);
    const turnIndexStart = anchorIdxs.length > 0 ? Math.min(...anchorIdxs) : null;
    const turnIndexEnd = anchorIdxs.length > 0 ? Math.max(...anchorIdxs) : null;

    const times: string[] = [];
    for (const e of evs) {
      if (e.bridge?.dispatchTimestamp) times.push(e.bridge.dispatchTimestamp);
      if (e.bridge?.responseTimestamp) times.push(e.bridge.responseTimestamp);
      if (e.rootAnchorTime) times.push(e.rootAnchorTime);
    }
    let startTime: string | null = null;
    let endTime: string | null = null;
    let durationMs = 0;
    if (times.length > 0) {
      const ms = times.map(t => new Date(t).getTime()).sort((a, b) => a - b);
      startTime = new Date(ms[0]).toISOString();
      endTime = new Date(ms[ms.length - 1]).toISOString();
      durationMs = ms[ms.length - 1] - ms[0];
    }

    const allSteps = collectAllSteps([{ children } as unknown as WorkflowPhaseNode]);
    const info = ranges[pi].start.info;
    phases.push({
      phaseIndex: info.order,
      phaseSequence: pi,
      phaseName: info.label,
      fullLabel: info.label,
      startTime,
      endTime,
      durationMs,
      activeTimeMs: durationMs,
      waitTimeMs: 0,
      totalTokens: allSteps.reduce((s, st) => s + st.totalTokens, 0),
      totalCost: 0,
      toolCallCount: 0,
      subagentCount: allSteps.filter(st => st.bridgeId !== null).length,
      triggerTurnId: evs[0]?.rootTriggerTurnId ?? null,
      turnIndexStart,
      turnIndexEnd,
      children,
    });
  }

  const validPhases = phases.filter(p => p.children.length > 0);
  const allSteps = collectAllSteps(validPhases);
  const totalDurationMs = validPhases.reduce((s, p) => s + p.durationMs, 0);
  const totalActiveTimeMs = validPhases.reduce((s, p) => s + p.activeTimeMs, 0);

  return {
    phases: validPhases,
    summary: {
      totalPhases: validPhases.length,
      totalSteps: allSteps.length,
      totalCheckpoints: 0,
      totalActiveTimeMs,
      totalWaitTimeMs: 0,
      activeTimePct: totalDurationMs > 0 ? Math.round((totalActiveTimeMs / totalDurationMs) * 100) : 0,
      iterations: 0,
    },
  };
}

function splitBySkillFamily(
  turns: TurnRow[],
  bridges: InteractionBridgeRow[],
  skillEvents: SkillEventRow[],
  _sessionId: string,
  turnById: Map<string, TurnRow>,
  bridgeByDispatchTurnId: Map<string, InteractionBridgeRow[]>,
): WorkflowTree {
  void _sessionId;
  const rootTurns = turns.filter(t => !t.isSubagent).sort((a, b) => a.turnIndex - b.turnIndex);
  const maxTurnIndex = turns.length > 0 ? turns.reduce((m, t) => Math.max(m, t.turnIndex), -1) : -1;

  const ordered: OrderedSkillEvent[] = [];
  for (const se of skillEvents) {
    const t = turnById.get(se.turnId);
    if (!t) continue;
    ordered.push({
      se,
      turnId: se.turnId,
      turnIndex: t.turnIndex,
      skillName: se.skillName,
      family: getFamilyKey(se.skillName),
    });
  }
  ordered.sort((a, b) =>
    a.turnIndex - b.turnIndex
    || (a.se.startedAt ?? '').localeCompare(b.se.startedAt ?? '')
    || a.skillName.localeCompare(b.skillName),
  );

  if (ordered.length === 0) {
    return buildFallbackTree(turns, bridges, _sessionId);
  }

  const gap = getPhaseGapTurns();
  const groups: OrderedSkillEvent[][] = [];
  let cur: OrderedSkillEvent[] = [];
  let prevTurn = -1;
  let prevWasOrch = false;

  for (const e of ordered) {
    const isOrch = e.family === '__orchestrator__';
    const turnGap = prevTurn >= 0 ? e.turnIndex - prevTurn : 0;
    if (cur.length === 0) {
      cur.push(e);
    } else if (isOrch || prevWasOrch || turnGap > gap) {
      groups.push(cur);
      cur = [e];
    } else {
      cur.push(e);
    }
    prevTurn = e.turnIndex;
    prevWasOrch = isOrch;
  }
  if (cur.length > 0) groups.push(cur);

  const phases: WorkflowPhaseNode[] = [];
  for (let i = 0; i < groups.length; i++) {
    const nextStart = i + 1 < groups.length ? Math.min(...groups[i + 1].map(e => e.turnIndex)) : null;
    phases.push(buildPhase(groups[i], nextStart, rootTurns, bridgeByDispatchTurnId, i, maxTurnIndex));
  }

  const validPhases = phases.filter(p => p.children.length > 0 || p.totalTokens > 0);
  const allSteps = collectAllSteps(validPhases);
  const allCheckpoints = collectAllCheckpoints(validPhases);
  const totalWaitTimeMs = validPhases.reduce((s, p) => s + p.waitTimeMs, 0);
  const totalDurationMs = validPhases.reduce((s, p) => s + p.durationMs, 0);
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

function buildPhase(
  group: OrderedSkillEvent[],
  nextStartTurnIndex: number | null,
  rootTurns: TurnRow[],
  bridgeByDispatchTurnId: Map<string, InteractionBridgeRow[]>,
  phaseSequence: number,
  maxTurnIndex: number,
): WorkflowPhaseNode {
  const turnIndexStart = Math.min(...group.map(e => e.turnIndex));
  const turnIndexEnd = nextStartTurnIndex != null
    ? nextStartTurnIndex - 1
    : (maxTurnIndex >= 0 ? maxTurnIndex : turnIndexStart);

  const famCounts = new Map<string, number>();
  for (const e of group) famCounts.set(e.family, (famCounts.get(e.family) ?? 0) + 1);
  let domFam = '';
  let domCnt = -1;
  for (const [f, c] of famCounts) {
    const score = f === '__orchestrator__' ? c - 0.5 : c;
    if (score > domCnt) { domCnt = score; domFam = f; }
  }
  const phaseName = getFamilyLabel(domFam);

  const byTurn = new Map<string, OrderedSkillEvent[]>();
  for (const e of group) {
    const arr = byTurn.get(e.turnId) ?? [];
    arr.push(e);
    byTurn.set(e.turnId, arr);
  }

  const children: Array<WorkflowStepNode | WorkflowParallelGroupNode | WorkflowTurnNode> = [];
  let stepCounter = 0;

  for (const [turnId, evs] of byTurn) {
    const turn = rootTurns.find(t => t.id === turnId);
    const dispatchBridges = bridgeByDispatchTurnId.get(turnId) ?? [];
    const isParallel = evs.length > 1;
    const pgId = isParallel ? `pg-${turnId}` : null;

    const steps: WorkflowStepNode[] = evs.map(e => {
      stepCounter++;
      const br = dispatchBridges.find(b => b.subagentType === e.skillName || b.subagentName === e.skillName) ?? null;
      const startTime = e.se.startedAt ?? turn?.createdAt_ts ?? null;
      const skillEndTime = e.se.completedAt ?? null;
      const endTime = br ? (br.responseTimestamp ?? skillEndTime) : skillEndTime;
      return {
        type: 'step',
        stepIndex: stepCounter,
        stepName: e.skillName,
        stepLabel: e.skillName,
        iterationIndex: null,
        iterationName: null,
        startTime,
        endTime,
        durationMs: (e.se.durationMs || br?.subagentLatencyMs) ?? 0,
        totalTokens: br?.subagentTokens ?? 0,
        totalCost: 0,
        toolCallCount: 0,
        bridgeId: br?.id ?? null,
        subagentSessionId: br?.subagentSessionId ?? null,
        subagentType: br?.subagentType ?? null,
        subagentName: br?.subagentName ?? null,
        status: br?.status ?? (e.se.success ? 'ok' : 'error'),
        parallelGroupId: pgId,
        triggerTurnId: turnId,
      };
    });

    if (isParallel) {
      children.push({
        type: 'parallel-group',
        groupId: pgId!,
        label: turn?.contentSummary ?? '并行',
        steps,
        totalDurationMs: Math.max(...steps.map(s => s.durationMs)),
        totalTokens: steps.reduce((s, x) => s + x.totalTokens, 0),
      });
    } else {
      children.push(...steps);
    }
  }

  const stepTurnIds = new Set(byTurn.keys());
  for (const t of rootTurns) {
    const inRange = t.turnIndex >= turnIndexStart && t.turnIndex <= turnIndexEnd;
    if (inRange && !stepTurnIds.has(t.id) && !bridgeByDispatchTurnId.has(t.id)) {
      children.push({
        type: 'turn',
        turnId: t.id,
        turnIndex: t.turnIndex,
        role: t.role,
        contentSummary: (t.content ?? '').substring(0, 200),
        startTime: t.createdAt_ts ?? null,
        endTime: t.completedAt ?? null,
        durationMs: t.latencyMs,
        totalTokens: t.totalTokens,
      });
    }
  }

  const eventTurnIndex = new Map(group.map(e => [e.turnId, e.turnIndex]));
  children.sort((a, b) => childSortKey(a, eventTurnIndex) - childSortKey(b, eventTurnIndex));

  const triggerTurnId = group[0]?.turnId ?? null;
  const phase: WorkflowPhaseNode = {
    phaseIndex: phaseSequence,
    phaseSequence,
    phaseName,
    fullLabel: phaseName,
    startTime: null,
    endTime: null,
    durationMs: 0,
    activeTimeMs: 0,
    waitTimeMs: 0,
    totalTokens: 0,
    totalCost: 0,
    toolCallCount: 0,
    subagentCount: 0,
    triggerTurnId,
    turnIndexStart,
    turnIndexEnd,
    children,
  };

  finalizePhase(phase, rootTurns);
  return phase;
}

function childSortKey(
  c: WorkflowStepNode | WorkflowParallelGroupNode | WorkflowTurnNode | WorkflowCheckpointNode,
  eventTurnIndex: Map<string, number>,
): number {
  if (c.type === 'turn') return c.turnIndex;
  if (c.type === 'checkpoint') return 0;
  const turnId = c.type === 'step' ? c.triggerTurnId : c.steps[0]?.triggerTurnId ?? null;
  if (!turnId) return 0;
  return eventTurnIndex.get(turnId) ?? 0;
}

function finalizePhase(phase: WorkflowPhaseNode, rootTurns: TurnRow[]): void {
  const allSteps = collectAllSteps([phase]);

  phase.totalCost = allSteps.reduce((s, st) => s + st.totalCost, 0);
  phase.toolCallCount = allSteps.reduce((s, st) => s + st.toolCallCount, 0);
  phase.subagentCount = allSteps.filter(st => st.bridgeId !== null).length;

  if (phase.turnIndexStart !== null && phase.turnIndexEnd !== null) {
    const stepTokens = allSteps.reduce((s, st) => s + st.totalTokens, 0);
    const rootTokensInRange = rootTurns
      .filter(t => t.turnIndex >= phase.turnIndexStart! && t.turnIndex <= phase.turnIndexEnd!)
      .reduce((s, t) => s + t.totalTokens, 0);
    phase.totalTokens = stepTokens + rootTokensInRange;
  }

  const times: (string | null)[] = [
    ...allSteps.map(s => s.startTime),
    ...allSteps.map(s => s.endTime),
  ];
  if (phase.turnIndexStart !== null && phase.turnIndexEnd !== null) {
    for (const t of rootTurns) {
      if (t.turnIndex >= phase.turnIndexStart! && t.turnIndex <= phase.turnIndexEnd!) {
        if (t.createdAt_ts) times.push(t.createdAt_ts);
        if (t.completedAt) times.push(t.completedAt);
      }
    }
  }
  const valid = times.filter((t): t is string => t !== null);
  if (valid.length > 0) {
    const sorted = valid.map(t => new Date(t).getTime()).sort((a, b) => a - b);
    phase.startTime = new Date(sorted[0]).toISOString();
    phase.endTime = new Date(sorted[sorted.length - 1]).toISOString();
    phase.durationMs = sorted[sorted.length - 1] - sorted[0];
  }

  phase.activeTimeMs = Math.max(0, phase.durationMs - phase.waitTimeMs);
}

function collectAllSteps(phases: WorkflowPhaseNode[]): WorkflowStepNode[] {
  const steps: WorkflowStepNode[] = [];
  for (const phase of phases) {
    for (const child of phase.children) {
      if (child.type === 'step') steps.push(child);
      if (child.type === 'parallel-group') steps.push(...child.steps);
    }
  }
  return steps;
}

function collectAllCheckpoints(phases: WorkflowPhaseNode[]): WorkflowCheckpointNode[] {
  return phases.flatMap(p => p.children.filter(c => c.type === 'checkpoint') as WorkflowCheckpointNode[]);
}

function buildFallbackTree(
  turns: TurnRow[],
  bridges: InteractionBridgeRow[],
  sessionId: string,
): WorkflowTree {
  void sessionId;
  const sortedBridges = [...bridges].sort((a, b) => {
    const aTime = a.dispatchTimestamp ? new Date(a.dispatchTimestamp).getTime() : 0;
    const bTime = b.dispatchTimestamp ? new Date(b.dispatchTimestamp).getTime() : 0;
    return aTime - bTime;
  });

  const rootTurns = turns.filter(t => !t.isSubagent).sort((a, b) => a.turnIndex - b.turnIndex);
  const startTime = rootTurns[0]?.createdAt_ts ?? null;
  const endTime = rootTurns.length > 0 ? rootTurns[rootTurns.length - 1]?.createdAt_ts ?? null : null;
  const durationMs = startTime && endTime ? new Date(endTime).getTime() - new Date(startTime).getTime() : 0;

  const children: Array<WorkflowStepNode | WorkflowCheckpointNode | WorkflowParallelGroupNode> = [];
  for (let i = 0; i < sortedBridges.length; i++) {
    const b = sortedBridges[i];
    children.push({
      type: 'step',
      stepIndex: i + 1,
      stepName: b.dispatchContent ?? `Step ${i + 1}`,
      stepLabel: `${i + 1} ${b.dispatchContent ?? `Step ${i + 1}`}`,
      iterationIndex: null,
      iterationName: null,
      startTime: b.dispatchTimestamp ?? null,
      endTime: b.responseTimestamp ?? null,
      durationMs: b.subagentLatencyMs,
      totalTokens: b.subagentTokens,
      totalCost: 0,
      toolCallCount: 0,
      bridgeId: b.id,
      subagentSessionId: b.subagentSessionId,
      subagentType: b.subagentType,
      subagentName: b.subagentName,
      status: b.status,
      parallelGroupId: null,
      triggerTurnId: b.dispatchTurnId,
    });
  }

  const stepTokens = children.reduce((s, c) => s + (c.type === 'step' ? c.totalTokens : 0), 0);
  const rootTokens = rootTurns.reduce((s, t) => s + t.totalTokens, 0);
  const phase: WorkflowPhaseNode = {
    phaseIndex: 1,
    phaseSequence: 0,
    phaseName: 'Main',
    fullLabel: 'Main Workflow',
    startTime,
    endTime,
    durationMs,
    activeTimeMs: durationMs,
    waitTimeMs: 0,
    totalTokens: stepTokens + rootTokens,
    totalCost: 0,
    toolCallCount: 0,
    subagentCount: sortedBridges.length,
    triggerTurnId: null,
    turnIndexStart: null,
    turnIndexEnd: null,
    children,
  };

  const allSteps = children.filter(c => c.type === 'step') as WorkflowStepNode[];
  return {
    phases: [phase],
    summary: {
      totalPhases: 1,
      totalSteps: allSteps.length,
      totalCheckpoints: 0,
      totalActiveTimeMs: durationMs,
      totalWaitTimeMs: 0,
      activeTimePct: 100,
      iterations: 0,
    },
  };
}
