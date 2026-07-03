// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { MetricCards } from '@/cli/tui/components/MetricCards';
import type { ApiSessionDetailResponse, ApiTurnItem } from '@/cli/types';
import { formatTokens, formatCost, formatDuration, formatDate } from '@/cli/utils/format';

interface OverviewTabProps {
  session: ApiSessionDetailResponse;
  turns: ApiTurnItem[];
}

export function OverviewTab({ session, turns }: OverviewTabProps) {
  const toolCallStats = (() => {
    const allToolCalls = turns.flatMap(t => t.toolCalls ?? []);
    if (allToolCalls.length === 0) return [];
    const grouped = new Map<string, { count: number; totalDuration: number; errorCount: number }>();
    for (const tc of allToolCalls) {
      const name = tc.toolName ?? '?';
      const existing = grouped.get(name) ?? { count: 0, totalDuration: 0, errorCount: 0 };
      existing.count++;
      existing.totalDuration += tc.durationMs ?? 0;
      if (tc.state === 'error') existing.errorCount++;
      grouped.set(name, existing);
    }
    return [...grouped.entries()]
      .sort((a, b) => b[1].count - a[1].count)
      .slice(0, 5);
  })();

  return (
    <Box flexDirection="column" gap={1}>
      <Box flexDirection="column">
        <Text bold color="cyan">Session Info</Text>
        <Text>Task: {session.taskId}</Text>
        <Text>Query: {session.query ?? '—'}</Text>
        <Text>Label: {session.label ?? '—'}</Text>
        <Text>Model: {session.model ?? '—'}</Text>
        <Text>User: {session.user ?? '—'}</Text>
        <Text>Start: {formatDate(session.startTime)}</Text>
        <Text>End: {formatDate(session.endTime)}</Text>
      </Box>

      <MetricCards
        cards={[
          { label: 'Tokens', value: formatTokens(session.totalTokens), color: 'cyan' },
          { label: 'Cache Read', value: formatTokens(session.totalCacheReadTokens), color: 'yellow' },
          { label: 'Cost', value: formatCost(session.totalCost), color: 'yellow' },
          { label: 'Duration', value: formatDuration(session.totalLatencyMs), color: 'green' },
          { label: 'LLM Calls', value: String(session.totalLlmCallCount ?? 0), color: 'blue' },
          { label: 'Tools', value: String(session.totalToolCallCount ?? 0), color: 'magenta' },
          { label: 'Skills', value: String(session.totalSkillLoadCount ?? 0), color: 'green' },
          { label: 'Subagents', value: String(session.totalSubagentCount ?? 0), color: 'orange' },
        ]}
      />

      <Box flexDirection="column">
        <Text bold color="cyan">Token Breakdown</Text>
        <Text>Input:      {formatTokens(session.totalInputTokens)}</Text>
        <Text>Output:     {formatTokens(session.totalOutputTokens)}</Text>
        <Text>Reasoning:  {formatTokens(session.totalReasoningTokens)}</Text>
        <Text>Cache Read: {formatTokens(session.totalCacheReadTokens)}</Text>
        <Text>Cache Write: {formatTokens(session.totalCacheWriteTokens)}</Text>
      </Box>

      {toolCallStats.length > 0 && (
        <Box flexDirection="column">
          <Text bold color="cyan">Tool Calls ({turns.flatMap(t => t.toolCalls ?? []).length})</Text>
          {toolCallStats.map(([name, stats]) => (
            <Text key={name}>
              {name} │ {stats.count}x │ avg {formatDuration(Math.round(stats.totalDuration / stats.count))}{stats.errorCount > 0 ? ` │ ${stats.errorCount} err` : ''}
            </Text>
          ))}
        </Box>
      )}

      {session.agents.length > 0 ? (
        <Box flexDirection="column">
          <Text bold color="cyan">Agents ({session.agents.length})</Text>
          {session.agents.map((agent, i) => (
            <Text key={i}>
              {agent.isSubagent ? '↳' : '●'} {agent.agentName ?? '—'} │ {formatTokens(agent.tokens)} │ {formatCost(agent.cost)} │ {agent.model ?? '—'}
            </Text>
          ))}
        </Box>
      ) : null}

      {session.skills.length > 0 ? (
        <Box flexDirection="column">
          <Text bold color="cyan">Skills ({session.skills.length})</Text>
          {session.skills.map((skill, i) => (
            <Text key={i}>
              {skill.skillName} (v{skill.version ?? '?'}) ×{skill.invocationCount}
            </Text>
          ))}
        </Box>
      ) : null}
    </Box>
  );
}
