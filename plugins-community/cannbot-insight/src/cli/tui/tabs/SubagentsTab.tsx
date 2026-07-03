// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import type { ApiExecutionItem, ApiBridgeItem } from '@/cli/types';
import { formatTokens, formatCost, formatDuration } from '@/cli/utils/format';

interface SubagentsTabProps {
  subagents: ApiExecutionItem[];
  bridges: ApiBridgeItem[];
  selectedIndex: number;
}

interface TreeNode {
  agent: ApiExecutionItem;
  children: TreeNode[];
}

function buildTree(subagents: ApiExecutionItem[]): TreeNode[] {
  const byParentId = new Map<string, TreeNode[]>();
  const nodes: TreeNode[] = [];

  for (const sub of subagents) {
    const node: TreeNode = { agent: sub, children: [] };
    nodes.push(node);
    const parentId = sub.parentExecutionId ?? 'root';
    if (!byParentId.has(parentId)) byParentId.set(parentId, []);
    byParentId.get(parentId)!.push(node);
  }

  for (const node of nodes) {
    node.children = byParentId.get(node.agent.executionId) ?? [];
  }

  const roots = byParentId.get('root') ?? [];
  return roots.length > 0 ? roots : nodes;
}

function formatTimeRange(createdAt: string, latencyMs: number): string {
  try {
    const start = new Date(createdAt);
    const end = new Date(start.getTime() + latencyMs);
    const fmt = (d: Date) => `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`;
    return `${fmt(start)}→${fmt(end)}`;
  } catch {
    return '—';
  }
}

const SUBAGENT_COLUMNS: DataTableColumn<ApiExecutionItem>[] = [
  { key: 'agentName', label: 'Name', width: 20 },
  { key: 'subagentType', label: 'Type', width: 12 },
  { key: 'model', label: 'Model', width: 18 },
  { key: 'parentExecutionId', label: 'Parent', width: 12, render: (row) => row.parentExecutionId ? row.parentExecutionId.substring(0, 8) : 'root' },
  { key: 'tokens', label: 'Tokens', width: 8, render: (row) => formatTokens(row.tokens ?? 0) },
  { key: 'cost', label: 'Cost', width: 8, render: (row) => formatCost(row.cost ?? 0) },
  { key: 'latencyMs', label: 'Latency', width: 10, render: (row) => formatDuration(row.latencyMs ?? 0) },
  { key: 'toolCallCount', label: 'Tools', width: 6 },
];

export function SubagentsTab({ subagents, bridges, selectedIndex }: SubagentsTabProps) {
  const tree = buildTree(subagents);

  const top3LatencyIds = new Set(
    [...subagents].sort((a, b) => (b.latencyMs ?? 0) - (a.latencyMs ?? 0)).slice(0, 3).map(s => s.executionId)
  );
  const top3TokenIds = new Set(
    [...subagents].sort((a, b) => (b.tokens ?? 0) - (a.tokens ?? 0)).slice(0, 3).map(s => s.executionId)
  );

  function renderHierarchyNode(node: TreeNode, depth: number): React.ReactNode[] {
    const sub = node.agent;
    const indent = '  '.repeat(depth);
    const icon = depth === 0 ? '●' : '↳';
    const name = sub.agentName ?? sub.subagentName ?? sub.subagentType ?? 'sub';
    const bridge = bridges.find(b => b.subagentSessionId === sub.agentSessionId || b.responseExecutionId === sub.executionId);
    const statusIcon = bridge?.status === 'completed' ? '✓' : bridge?.status === 'failed' ? '✗' : '◐';
    const isTop3Latency = top3LatencyIds.has(sub.executionId);
    const isTop3Token = top3TokenIds.has(sub.executionId);
    const badges = [
      isTop3Latency ? '⏱' : '',
      isTop3Token ? '🔥' : '',
    ].filter(Boolean).join('');

    const lines: React.ReactNode[] = [];

    lines.push(
      <Text key={`agent-${sub.executionId}`}>
        {indent}{icon} {name} │ {formatTokens(sub.tokens ?? 0)} │ {formatCost(sub.cost ?? 0)} │ {formatDuration(sub.latencyMs ?? 0)} │ {statusIcon} {bridge?.status ?? 'dispatched'} │ {formatTimeRange(sub.createdAt, sub.latencyMs ?? 0)}{badges ? ` │ ${badges}` : ''}
      </Text>
    );

    if (bridge?.dispatchContent) {
      const truncated = bridge.dispatchContent.length > 60 ? bridge.dispatchContent.substring(0, 60) + '…' : bridge.dispatchContent;
      lines.push(
        <Text key={`bridge-${sub.executionId}`} color="gray">
          {indent}│ dispatch: "{truncated}"
        </Text>
      );
    }

    for (const child of node.children) {
      lines.push(...renderHierarchyNode(child, depth + 1));
    }

    return lines;
  }

  return (
    <Box flexDirection="column" gap={1}>
      {subagents.length > 0 ? (
        <Box flexDirection="column">
          <Text bold color="cyan">Hierarchy ({subagents.length} subagents)</Text>
          <Text color="gray">⏱ Top3 latency │ 🔥 Top3 tokens</Text>
          {tree.flatMap(node => renderHierarchyNode(node, 0))}
        </Box>
      ) : (
        <Text color="gray">No subagents</Text>
      )}

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="cyan">Subagents Detail ({subagents.length})</Text>
        <DataTable
          columns={SUBAGENT_COLUMNS}
          data={subagents}
          selectedIndex={selectedIndex}
        />
      </Box>
    </Box>
  );
}

export { SUBAGENT_COLUMNS };
