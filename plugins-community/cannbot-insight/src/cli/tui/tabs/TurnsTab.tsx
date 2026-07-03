// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import { AsciiBar } from '@/cli/tui/components/AsciiBar';
import { getContextWindowLimit } from '@/lib/context-window-config';
import type { ApiTurnItem, ApiTurnDetailResponse } from '@/cli/types';
import { formatTokens, formatCost, formatDuration, formatPercent, truncate } from '@/cli/utils/format';

interface TurnsTabProps {
  turns: ApiTurnItem[];
  selectedIndex: number;
  turnDetail: ApiTurnDetailResponse | null;
}

const TURN_COLUMNS: DataTableColumn<ApiTurnItem>[] = [
  { key: 'turnIndex', label: '#', width: 4 },
  { key: 'role', label: 'Role', width: 10 },
  { key: 'agentName', label: 'Agent', width: 15 },
  { key: 'contentSummary', label: 'Summary', width: 40 },
  { key: 'totalTokens', label: 'Tokens', width: 8 },
  { key: 'latencyMs', label: 'Latency', width: 8 },
  { key: 'createdAt', label: 'Time', width: 16 },
];

export function TurnsTab({ turns, selectedIndex, turnDetail }: TurnsTabProps) {
  const selectedTurn = turnDetail;
  const contextWindow = getContextWindowLimit(selectedTurn?.model ?? null);

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Turns ({turns.length})</Text>
      <DataTable
        columns={TURN_COLUMNS}
        data={turns}
        selectedIndex={selectedIndex}
      />

      {selectedTurn ? (
        <Box flexDirection="column" marginTop={1} borderStyle="single" paddingX={1}>
          <Text bold color="cyan">
            ── Turn #{selectedTurn.turnIndex} {selectedTurn.role} │ {selectedTurn.agentName ?? selectedTurn.subagentName ?? '?'} ──
          </Text>

          <Box flexDirection="column">
            <Text>
              Model: {selectedTurn.model ?? '—'} │ Latency: {formatDuration(selectedTurn.latencyMs)} │ TTFT: {selectedTurn.ttftMs != null ? formatDuration(selectedTurn.ttftMs) : '—'}
            </Text>
            <Text>
              Tokens: {formatTokens(selectedTurn.inputTokens)} in │ {formatTokens(selectedTurn.outputTokens)} out │ {formatTokens(selectedTurn.reasoningTokens)} reason │ {formatTokens(selectedTurn.cacheReadTokens)} cache
            </Text>
            <Text>
              Finish: {selectedTurn.finishReason ?? '—'} │ Cost: {formatCost(selectedTurn.inputTokens != null && selectedTurn.outputTokens != null ? (selectedTurn.inputTokens + selectedTurn.outputTokens) * 0.00001 : 0)}
            </Text>
          </Box>

          {selectedTurn.contextWindowPct != null && (
            <AsciiBar
              label={selectedTurn.isSubagent ? `Sub (${selectedTurn.subagentName ?? '?'})` : `Root (${selectedTurn.agentName ?? 'root'})`}
              value={selectedTurn.inputTokens ?? 0}
              max={contextWindow}
              width={20}
              warningThreshold={0.7}
              criticalThreshold={0.9}
            />
          )}

          {(() => {
            const cacheRead = selectedTurn.cacheReadTokens ?? 0;
            const input = selectedTurn.inputTokens ?? 0;
            const rate = input + cacheRead > 0 ? cacheRead / (input + cacheRead) : 0;
            return <Text color="gray">Cache: {rate > 0 ? `${(rate * 100).toFixed(0)}%` : '—'} │ Input: {formatTokens(input)} │ Cached: {formatTokens(cacheRead)}</Text>;
          })()}

          {selectedTurn.contentSummary ? (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="cyan">Summary</Text>
              <Text>{truncate(selectedTurn.contentSummary, 120)}</Text>
            </Box>
          ) : null}

          {selectedTurn.content ? (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="cyan">Content</Text>
              {(() => {
                const contentStr = selectedTurn.content;
                const truncated = contentStr.length > 300 ? contentStr.substring(0, 300) + '…' : contentStr;
                const lines = truncated.split('\n').slice(0, 10);
                return lines.map((line, j) => {
                  const isThinking = line.includes('<thinking>') || line.includes('</thinking>');
                  return (
                    <Text key={j} color={isThinking ? 'yellow' : undefined}>
                      {truncate(line, 120)}
                    </Text>
                  );
                });
              })()}
            </Box>
          ) : null}

          {selectedTurn.toolCalls.length > 0 ? (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="cyan">Tool Calls ({selectedTurn.toolCalls.length})</Text>
              {selectedTurn.toolCalls.map((tc, i) => (
                <Text key={i}>
                  {tc.state === 'error' ? '✗' : '✓'} {tc.toolName ?? '?'} │ {formatDuration(tc.durationMs)}
                  {tc.errorType ? ` │ err: ${tc.errorType}` : ''}
                </Text>
              ))}
            </Box>
          ) : null}

          {selectedTurn.skillEvents.length > 0 ? (
            <Box flexDirection="column" marginTop={1}>
              <Text bold color="cyan">Skill Events ({selectedTurn.skillEvents.length})</Text>
              {selectedTurn.skillEvents.map((se, i) => (
                <Text key={i}>
                  {se.success ? '✓' : '✗'} {se.skillName ?? '?'} │ {se.eventType ?? '?'}
                  {se.errorMessage ? ` │ ${truncate(se.errorMessage, 40)}` : ''}
                </Text>
              ))}
            </Box>
          ) : null}
        </Box>
      ) : (
        <Box flexDirection="column" marginTop={1}>
          <Text color="gray">Press Enter to view turn details</Text>
        </Box>
      )}
    </Box>
  );
}

export { TURN_COLUMNS };
