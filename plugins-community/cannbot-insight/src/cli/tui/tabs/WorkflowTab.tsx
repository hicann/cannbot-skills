// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useMemo, useCallback } from 'react';
import { AsciiBar } from '@/cli/tui/components/AsciiBar';
import { getContextWindowLimit } from '@/lib/context-window-config';
import { formatDuration, formatTokens, formatCost, formatPercent } from '@/cli/utils/format';
import type { WorkflowTree, WorkflowPhaseNode, WorkflowStepNode, WorkflowCheckpointNode, WorkflowParallelGroupNode, ApiTurnItem } from '@/cli/types';

type NodeType = 'phase-header' | 'step' | 'subagent-turn' | 'checkpoint' | 'parallel-header' | 'parallel-step'

interface TreeNode {
  type: NodeType
  depth: number
  label: string
  detail?: string
  color?: string
  phaseIndex?: number
  stepKey?: string
  turnId?: string
  turnData?: ApiTurnItem
  subagentSessionId?: string
}

function buildFlatTree(
  phases: WorkflowPhaseNode[],
  allTurns: ApiTurnItem[],
  expandedPhases: Set<number>,
  expandedSteps: Set<string>,
): TreeNode[] {
  const nodes: TreeNode[] = []

  for (const phase of phases) {
    const isExpanded = expandedPhases.has(phase.phaseIndex)
    const icon = isExpanded ? '▼' : '▶'

    nodes.push({
      type: 'phase-header',
      depth: 0,
      label: `${icon} Phase ${phase.phaseIndex}: ${phase.fullLabel}`,
      detail: `${formatDuration(phase.durationMs)} │ ${formatTokens(phase.totalTokens)} │ ${formatCost(phase.totalCost)} │ ${phase.subagentCount} sub`,
      color: 'cyan',
      phaseIndex: phase.phaseIndex,
    })

    if (!isExpanded) continue

    for (const child of phase.children) {
      if (child.type === 'step') {
        const step = child
        const stepKey = `p${phase.phaseIndex}-s${step.stepIndex}`
        const isStepExpanded = expandedSteps.has(stepKey)
        const statusIcon = step.status === 'completed' ? '✓' : step.status === 'failed' ? '✗' : '◐'
        const expandHint = isStepExpanded ? '▼' : '▶'
        const agentLabel = step.subagentName ?? step.subagentType ?? 'sub'

        nodes.push({
          type: 'step',
          depth: 1,
          label: `${statusIcon} Step ${step.stepIndex}: ${step.stepLabel} ${expandHint}`,
          detail: `${formatDuration(step.durationMs)} │ ${formatTokens(step.totalTokens)} │ ${step.toolCallCount} tools │ ${agentLabel}`,
          color: step.status === 'failed' ? 'red' : 'green',
          stepKey,
          subagentSessionId: step.subagentSessionId ?? undefined,
        })

        if (isStepExpanded && step.subagentSessionId) {
          const subTurns = allTurns.filter(t => t.subagentSessionId === step.subagentSessionId)
          for (const t of subTurns) {
            nodes.push({
              type: 'subagent-turn',
              depth: 2,
              label: `#${t.turnIndex} ${t.role} │ ${formatTokens(t.inputTokens)} │ ${formatPercent(t.contextWindowPct)}`,
              detail: t.contentSummary ? truncateStr(t.contentSummary, 60) : undefined,
              color: t.role === 'assistant' ? 'green' : t.role === 'user' ? 'blue' : 'gray',
              turnId: t.turnId,
              turnData: t,
            })
          }
        }
      } else if (child.type === 'checkpoint') {
        const cp = child
        const icon = cp.checkpointType === 'block' ? '🛑' : 'ℹ'
        nodes.push({
          type: 'checkpoint',
          depth: 1,
          label: `${icon} ${cp.checkpointLabel}`,
          detail: `Wait: ${formatDuration(cp.waitTimeMs)}`,
          color: 'yellow',
        })
      } else if (child.type === 'parallel-group') {
        const pg = child
        nodes.push({
          type: 'parallel-header',
          depth: 1,
          label: `══ Parallel: ${pg.label}`,
          detail: `${formatDuration(pg.totalDurationMs)} │ ${formatTokens(pg.totalTokens)}`,
          color: 'blue',
        })
        for (const step of pg.steps) {
          const stepKey = `p${phase.phaseIndex}-pg${step.stepIndex}`
          const statusIcon = step.status === 'completed' ? '✓' : step.status === 'failed' ? '✗' : '◐'
          const agentLabel = step.subagentName ?? step.subagentType ?? 'sub'

          nodes.push({
            type: 'parallel-step',
            depth: 2,
            label: `${statusIcon} ${step.stepLabel}`,
            detail: `${formatDuration(step.durationMs)} │ ${formatTokens(step.totalTokens)} │ ${agentLabel}`,
            color: step.status === 'failed' ? 'red' : 'green',
            stepKey,
            subagentSessionId: step.subagentSessionId ?? undefined,
          })
        }
      }
    }
  }

  return nodes
}

function truncateStr(str: string, maxLen: number): string {
  if (str.length <= maxLen) return str
  return str.substring(0, maxLen) + '…'
}

export function useWorkflowInteraction(workflow: WorkflowTree, allTurns: ApiTurnItem[]) {
  const [cursorIndex, setCursorIndex] = useState(0)
  const [expandedPhases, setExpandedPhases] = useState<Set<number>>(new Set([0]))
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set())
  const [selectedTurnId, setSelectedTurnId] = useState<string | null>(null)

  const flatNodes = useMemo(
    () => buildFlatTree(workflow.phases, allTurns, expandedPhases, expandedSteps),
    [workflow.phases, allTurns, expandedPhases, expandedSteps]
  )

  const handleUp = useCallback(() => {
    setCursorIndex(i => Math.max(0, i - 1))
  }, [])

  const handleDown = useCallback(() => {
    setCursorIndex(i => Math.min(Math.max(flatNodes.length - 1, 0), i + 1))
  }, [flatNodes.length])

  const handleEnter = useCallback(() => {
    if (cursorIndex >= flatNodes.length) return
    const node = flatNodes[cursorIndex]

    if (node.type === 'phase-header' && node.phaseIndex != null) {
      setExpandedPhases(prev => {
        const next = new Set(prev)
        if (next.has(node.phaseIndex!)) next.delete(node.phaseIndex!)
        else next.add(node.phaseIndex!)
        return next
      })
    } else if ((node.type === 'step' || node.type === 'parallel-step') && node.stepKey) {
      setExpandedSteps(prev => {
        const next = new Set(prev)
        if (next.has(node.stepKey!)) next.delete(node.stepKey!)
        else next.add(node.stepKey!)
        return next
      })
    } else if (node.type === 'subagent-turn' && node.turnId) {
      setSelectedTurnId(node.turnId)
    }
  }, [cursorIndex, flatNodes])

  return {
    cursorIndex,
    expandedPhases,
    expandedSteps,
    selectedTurnId,
    flatNodes,
    handleUp,
    handleDown,
    handleEnter,
  }
}

interface WorkflowTabProps {
  workflow: WorkflowTree;
  allTurns: ApiTurnItem[];
  sessionModel: string | null;
  flatNodes: TreeNode[];
  cursorIndex: number;
  selectedTurnId: string | null;
}

export function WorkflowTab({ workflow, allTurns, sessionModel, flatNodes, cursorIndex, selectedTurnId }: WorkflowTabProps) {
  const contextWindow = getContextWindowLimit(sessionModel)
  const selectedTurn = selectedTurnId ? allTurns.find(t => t.turnId === selectedTurnId) : null

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Workflow ✦</Text>
      <Text color="gray">
        {workflow.summary.totalPhases} phases │ {workflow.summary.totalSteps} steps │ ↑↓ navigate │ Enter expand/select
      </Text>

      {flatNodes.map((node, i) => {
        const isCursor = i === cursorIndex
        const indent = '  '.repeat(node.depth)
        const cursorMarker = isCursor ? ' ←' : ''
        const nodeColor = node.color ?? 'white'

        return (
          <Box key={`node-${i}`} flexDirection="column">
            <Text color={isCursor ? 'white' : nodeColor} backgroundColor={isCursor ? 'blue' : undefined}>
              {indent}{node.label}{cursorMarker}
            </Text>
            {node.detail && (
              <Text color="gray" backgroundColor={isCursor ? 'blue' : undefined}>
                {indent}│ {node.detail}
              </Text>
            )}
          </Box>
        )
      })}

      {selectedTurn && (
        <Box flexDirection="column" marginTop={1} borderStyle="single" paddingX={1}>
          <Text bold color="cyan">── Context: #{selectedTurn.turnIndex} {selectedTurn.role} │ {selectedTurn.agentName ?? '?'} ──</Text>

          {(() => {
            const isSub = selectedTurn.isSubagent
            const rootTurn = isSub ? allTurns.filter(t => !t.isSubagent).reduce((best, t) => {
              if (!t.createdAt || !selectedTurn.createdAt) return best
              if (new Date(t.createdAt).getTime() <= new Date(selectedTurn.createdAt).getTime() && (!best || new Date(best.createdAt).getTime() < new Date(t.createdAt).getTime())) return t
              return best
            }, null as ApiTurnItem | null) : selectedTurn

            return (
              <AsciiBar
                label={`Root (${rootTurn?.agentName ?? 'root'})`}
                value={rootTurn?.inputTokens ?? 0}
                max={contextWindow}
                width={20}
                warningThreshold={0.7}
                criticalThreshold={0.9}
              />
            )
          })()}

          {selectedTurn.isSubagent && (
            <AsciiBar
              label={`Sub (${selectedTurn.subagentName ?? selectedTurn.agentName ?? '?'})`}
              value={selectedTurn.inputTokens ?? 0}
              max={contextWindow}
              width={20}
              warningThreshold={0.7}
              criticalThreshold={0.9}
            />
          )}

          {(() => {
            const cacheRead = selectedTurn.cacheReadTokens ?? 0
            const input = selectedTurn.inputTokens ?? 0
            const rate = input + cacheRead > 0 ? cacheRead / (input + cacheRead) : 0
            return (
              <Text color="gray">Cache: {rate > 0 ? `${(rate * 100).toFixed(0)}%` : '—'} │ Input: {formatTokens(input)} │ Cached: {formatTokens(cacheRead)}</Text>
            )
          })()}
        </Box>
      )}
    </Box>
  )
}
