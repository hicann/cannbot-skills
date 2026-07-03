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
import { MetricCards } from '@/cli/tui/components/MetricCards';
import { AsciiBar } from '@/cli/tui/components/AsciiBar';
import { useApi } from '@/cli/hooks/useApi';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import type { InsightClient } from '@/cli/client';
import { formatTokens, formatCost, formatDuration, formatDate } from '@/cli/utils/format';

interface TurnDetailProps {
  client: InsightClient;
  turnId: string;
  onBack: () => void;
}

export function TurnDetail({ client, turnId, onBack }: TurnDetailProps) {
  const [contentScroll, setContentScroll] = useState(0);
  const CONTENT_PAGE_SIZE = 20;

  const fetchTurn = useCallback(() => client.getTurnDetail(turnId), [client, turnId]);
  const { data: turn, loading, error, refresh } = useApi(fetchTurn, [client, turnId]);

  useKeyboard({
    onEscape: onBack,
    onRefresh: refresh,
    onNavigateUp: () => setContentScroll(s => Math.max(0, s - CONTENT_PAGE_SIZE)),
    onNavigateDown: () => setContentScroll(s => s + CONTENT_PAGE_SIZE),
  });

  if (loading && !turn) return <Spinner label="Loading turn..." />;
  if (error) return <Text color="red">Error: {error.message}</Text>;
  if (!turn) return <Text>No turn data</Text>;

  const contentLines = (turn.content ?? '').split('\n');
  const visibleContent = contentLines.slice(contentScroll, contentScroll + CONTENT_PAGE_SIZE);

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Turn #{turn.turnIndex} — {turn.role}</Text>

      <MetricCards
        cards={[
          { label: 'Tokens', value: formatTokens(turn.totalTokens), color: 'cyan' },
          { label: 'Input', value: formatTokens(turn.inputTokens), color: 'blue' },
          { label: 'Output', value: formatTokens(turn.outputTokens), color: 'green' },
          { label: 'Cost', value: formatCost(0), color: 'yellow' },
          { label: 'Latency', value: formatDuration(turn.latencyMs), color: 'magenta' },
        ]}
      />

      <Box flexDirection="column">
        <Text color="gray">Agent: {turn.agentName ?? '—'} │ Model: {turn.model ?? '—'} │ Finish: {turn.finishReason ?? '—'}</Text>
        <Text color="gray">Created: {formatDate(turn.createdAt)} │ TTFT: {formatDuration(turn.ttftMs)}</Text>
      </Box>

      {turn.contextWindowPct !== null ? (
        <AsciiBar
          label="Context"
          value={Math.round(turn.contextWindowPct ?? 0)}
          max={100}
          width={20}
          warningThreshold={0.7}
          criticalThreshold={0.9}
          unit="%"
        />
      ) : null}

      <Box flexDirection="column">
        <Text bold color="cyan">Content</Text>
        {visibleContent.map((line, i) => (
          <Text key={i}>{line}</Text>
        ))}
        {contentLines.length > contentScroll + CONTENT_PAGE_SIZE ? (
          <Text color="gray">... (↑↓ scroll, showing lines {contentScroll}-{contentScroll + visibleContent.length} of {contentLines.length})</Text>
        ) : null}
      </Box>

      {turn.toolCalls.length > 0 ? (
        <Box flexDirection="column">
          <Text bold color="cyan">Tool Calls ({turn.toolCalls.length})</Text>
          {turn.toolCalls.map((tc, i) => (
            <Text key={i}>
              {tc.toolName ?? '—'} │ {tc.state ?? '—'} │ {formatDuration(tc.durationMs)}
              {tc.errorMessage ? <Text color="red"> │ Error: {truncate(tc.errorMessage, 40)}</Text> : null}
            </Text>
          ))}
        </Box>
      ) : null}

      {turn.skillEvents.length > 0 ? (
        <Box flexDirection="column">
          <Text bold color="cyan">Skill Events ({turn.skillEvents.length})</Text>
          {turn.skillEvents.map((se, i) => (
            <Text key={i}>
              {se.skillName ?? '—'} │ {se.eventType ?? '—'} │ {se.success ? '✓' : '✗'} │ {formatDuration(se.durationMs)}
            </Text>
          ))}
        </Box>
      ) : null}
    </Box>
  );
}

function truncate(str: string, max: number): string {
  if (str.length <= max) return str;
  return str.slice(0, max - 1) + '…';
}
