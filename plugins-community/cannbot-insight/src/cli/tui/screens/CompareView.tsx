// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useCallback, useState } from 'react';
import { Spinner } from '@/cli/tui/components/Spinner';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import { useApi } from '@/cli/hooks/useApi';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import type { InsightClient } from '@/cli/client';
import type { ApiSessionDetailResponse } from '@/cli/types';
import { formatTokens, formatCost, formatDuration } from '@/cli/utils/format';

interface CompareViewProps {
  client: InsightClient;
  taskId1: string;
  taskId2: string;
  onBack: () => void;
}

interface CompareRow {
  metric: string;
  s1: string;
  s2: string;
  diff: string;
}

const COMPARE_COLUMNS: DataTableColumn<CompareRow>[] = [
  { key: 'metric', label: 'Metric', width: 20 },
  { key: 's1', label: 'Session 1', width: 15 },
  { key: 's2', label: 'Session 2', width: 15 },
  { key: 'diff', label: 'Diff', width: 12 },
];

export function CompareView({ client, taskId1, taskId2, onBack }: CompareViewProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);

  const fetch1 = useCallback(() => client.getSession(taskId1), [client, taskId1]);
  const fetch2 = useCallback(() => client.getSession(taskId2), [client, taskId2]);

  const { data: s1, loading: l1 } = useApi(fetch1, [client, taskId1]);
  const { data: s2, loading: l2 } = useApi(fetch2, [client, taskId2]);

  useKeyboard({
    onEscape: onBack,
    onNavigateUp: () => setSelectedIndex(i => Math.max(0, i - 1)),
    onNavigateDown: () => setSelectedIndex(i => i + 1),
  });

  if ((l1 && !s1) || (l2 && !s2)) return <Spinner label="Loading sessions..." />;
  if (!s1 || !s2) return <Text color="red">Failed to load sessions</Text>;

  const rows = buildCompareRows(s1, s2);

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">Compare: {taskId1} vs {taskId2}</Text>
      <DataTable
        columns={COMPARE_COLUMNS}
        data={rows}
        selectedIndex={selectedIndex}
      />
    </Box>
  );
}

function buildCompareRows(s1: ApiSessionDetailResponse, s2: ApiSessionDetailResponse): CompareRow[] {
  const metrics: Array<{ key: string; label: string; v1: number | null; v2: number | null; fmt: (v: number | null) => string }> = [
    { key: 'tokens', label: 'Tokens', v1: s1.totalTokens, v2: s2.totalTokens, fmt: formatTokens },
    { key: 'inputTokens', label: 'Input Tokens', v1: s1.totalInputTokens, v2: s2.totalInputTokens, fmt: formatTokens },
    { key: 'outputTokens', label: 'Output Tokens', v1: s1.totalOutputTokens, v2: s2.totalOutputTokens, fmt: formatTokens },
    { key: 'reasoningTokens', label: 'Reasoning', v1: s1.totalReasoningTokens, v2: s2.totalReasoningTokens, fmt: formatTokens },
    { key: 'cost', label: 'Cost', v1: s1.totalCost, v2: s2.totalCost, fmt: formatCost },
    { key: 'duration', label: 'Duration', v1: s1.totalLatencyMs, v2: s2.totalLatencyMs, fmt: formatDuration },
    { key: 'tools', label: 'Tool Calls', v1: s1.totalToolCallCount, v2: s2.totalToolCallCount, fmt: String },
    { key: 'skills', label: 'Skill Loads', v1: s1.totalSkillLoadCount, v2: s2.totalSkillLoadCount, fmt: String },
    { key: 'subagents', label: 'Subagents', v1: s1.totalSubagentCount, v2: s2.totalSubagentCount, fmt: String },
    { key: 'llmCalls', label: 'LLM Calls', v1: s1.totalLlmCallCount, v2: s2.totalLlmCallCount, fmt: String },
  ];

  return metrics.map(m => {
    const v1 = m.v1 ?? 0;
    const v2 = m.v2 ?? 0;
    const diff = v2 - v1;
    const diffStr = diff === 0 ? '=' : diff > 0 ? `+${m.fmt(diff)}` : `-${m.fmt(Math.abs(diff))}`;
    return {
      metric: m.label,
      s1: m.fmt(v1),
      s2: m.fmt(v2),
      diff: diffStr,
    };
  });
}
