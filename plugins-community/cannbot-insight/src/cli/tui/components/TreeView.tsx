// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import type { WorkflowPhaseNode, WorkflowChildNode, WorkflowStepNode, WorkflowCheckpointNode, WorkflowParallelGroupNode } from '@/cli/types';
import { formatDuration, formatCost, formatTokens } from '@/cli/utils/format';

interface TreeViewProps {
  phases: WorkflowPhaseNode[];
}

function renderStep(step: WorkflowStepNode, indent: string): React.ReactNode[] {
  const statusIcon = step.status === 'completed' ? '✓' : step.status === 'failed' ? '✗' : '◐';
  const line = `${indent}├─ ${statusIcon} Step ${step.stepIndex}: ${step.stepLabel}`;
  const details = `${indent}│  ${formatDuration(step.durationMs)} │ ${formatTokens(step.totalTokens)} │ ${formatCost(step.totalCost)} │ ${step.toolCallCount} tools`;
  return [
    <Text key={`${step.stepIndex}-line`} color={step.status === 'failed' ? 'red' : 'green'}>{line}</Text>,
    <Text key={`${step.stepIndex}-detail`} color="gray">{details}</Text>,
  ];
}

function renderCheckpoint(cp: WorkflowCheckpointNode, indent: string): React.ReactNode[] {
  const icon = cp.checkpointType === 'block' ? '🛑' : 'ℹ';
  return [
    <Text key={`${cp.checkpointIndex}-line`} color="yellow">{indent}├─ ${icon} Checkpoint: ${cp.checkpointLabel}</Text>,
    <Text key={`${cp.checkpointIndex}-detail`} color="gray">{indent}│  Wait: ${formatDuration(cp.waitTimeMs)}</Text>,
  ];
}

function renderParallel(pg: WorkflowParallelGroupNode, indent: string): React.ReactNode[] {
  const lines: React.ReactNode[] = [
    <Text key={`${pg.groupId}-header`} color="blue">{indent}├─ ══ Parallel: ${pg.label}</Text>,
  ];
  for (const step of pg.steps) {
    lines.push(...renderStep(step, indent + '│  '));
  }
  return lines;
}

function renderChild(child: WorkflowChildNode, indent: string): React.ReactNode[] {
  if (child.type === 'step') return renderStep(child, indent);
  if (child.type === 'checkpoint') return renderCheckpoint(child, indent);
  if (child.type === 'parallel-group') return renderParallel(child, indent);
  return [];
}

function renderPhase(phase: WorkflowPhaseNode): React.ReactNode[] {
  const lines: React.ReactNode[] = [
    <Text key={`phase-${phase.phaseIndex}`} bold color="cyan">
      Phase {phase.phaseIndex}: {phase.fullLabel} │ {formatDuration(phase.durationMs)} │ {formatTokens(phase.totalTokens)} │ {formatCost(phase.totalCost)}
    </Text>,
  ];
  for (const child of phase.children) {
    lines.push(...renderChild(child, '  '));
  }
  return lines;
}

export function TreeView({ phases }: TreeViewProps) {
  return (
    <Box flexDirection="column">
      {phases.map(phase => renderPhase(phase))}
    </Box>
  );
}
