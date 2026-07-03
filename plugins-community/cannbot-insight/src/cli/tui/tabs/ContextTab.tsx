// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { AsciiBar } from '@/cli/tui/components/AsciiBar';
import { getContextWindowLimit } from '@/lib/context-window-config';
import type { ApiTurnItem } from '@/cli/types';
import { formatTokens } from '@/cli/utils/format';

const BLOCK_CHARS = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

function pctToBlock(pct: number): string {
  const idx = Math.min(Math.floor(pct / 100 * BLOCK_CHARS.length), BLOCK_CHARS.length - 1);
  return BLOCK_CHARS[idx];
}

interface ContextTabProps {
  turns: ApiTurnItem[];
  model: string | null;
}

export function ContextTab({ turns, model }: ContextTabProps) {
  const contextWindow = getContextWindowLimit(model);
  const contextTurns = turns.filter(t => t.contextWindowPct !== null && t.contextWindowPct !== undefined);

  const maxPct = contextTurns.length > 0
    ? Math.max(...contextTurns.map(t => t.contextWindowPct ?? 0))
    : 0;

  const rootContextTurns = contextTurns.filter(t => !t.isSubagent);
  const subContextTurns = contextTurns.filter(t => t.isSubagent);

  const rootTrend = rootContextTurns.map(t => pctToBlock(t.contextWindowPct ?? 0)).join('');
  const subTrend = subContextTurns.map(t => pctToBlock(t.contextWindowPct ?? 0)).join('');

  const cacheTurns = turns.filter(t => t.cacheReadTokens !== null && t.inputTokens !== null);
  const cacheTrend = cacheTurns.map(t => {
    const input = t.inputTokens ?? 0;
    const cache = t.cacheReadTokens ?? 0;
    const total = input + cache;
    if (total === 0) return BLOCK_CHARS[0];
    const pct = cache / total * 100;
    return pctToBlock(pct);
  }).join('');

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Context Usage</Text>
      <Text color="gray">Model: {model ?? '—'} │ Context Window: {contextWindow.toLocaleString()}</Text>

      {contextTurns.length > 0 ? (
        <Box flexDirection="column">
          <AsciiBar
            label="Peak Usage"
            value={Math.round(maxPct)}
            max={100}
            width={30}
            warningThreshold={0.7}
            criticalThreshold={0.9}
            unit="%"
          />
        </Box>
      ) : (
        <Text color="gray">No context usage data available</Text>
      )}

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="cyan">Context Trend</Text>
        {rootTrend.length > 0 ? (
          <Box flexDirection="column">
            <Box flexDirection="row">
              <Text color="green">Root  </Text>
              <Text color="green">{rootTrend}</Text>
              <Text color="gray"> #{rootContextTurns[0]?.turnIndex ?? 0}→{rootContextTurns[rootContextTurns.length - 1]?.turnIndex ?? 0}</Text>
            </Box>
            {rootContextTurns.length > 0 && (
              <Box flexDirection="row">
                <Text color="gray">      </Text>
                <Text color="gray">min {Math.round(Math.min(...rootContextTurns.map(t => t.contextWindowPct ?? 0)))}% │ max {Math.round(maxPct)}% │ avg {Math.round(rootContextTurns.reduce((s, t) => s + (t.contextWindowPct ?? 0), 0) / rootContextTurns.length)}%</Text>
              </Box>
            )}
          </Box>
        ) : null}

        {subTrend.length > 0 ? (
          <Box flexDirection="column">
            <Box flexDirection="row">
              <Text color="orange">Sub   </Text>
              <Text color="orange">{subTrend}</Text>
              <Text color="gray"> {subContextTurns.length} turns</Text>
            </Box>
            {(() => {
              const subNames = [...new Set(subContextTurns.map(t => t.subagentName ?? t.agentName ?? 'sub'))];
              return subNames.map(name => {
                const nameTurns = subContextTurns.filter(t => (t.subagentName ?? t.agentName ?? 'sub') === name);
                const nameTrend = nameTurns.map(t => pctToBlock(t.contextWindowPct ?? 0)).join('');
                return (
                  <Box flexDirection="row" key={name}>
                    <Text color="yellow">      </Text>
                    <Text color="yellow">{name} </Text>
                    <Text color="yellow">{nameTrend}</Text>
                    <Text color="gray"> {nameTurns.length}t</Text>
                  </Box>
                );
              });
            })()}
          </Box>
        ) : null}

        {cacheTrend.length > 0 ? (
          <Box flexDirection="column">
            <Box flexDirection="row">
              <Text color="yellow">Cache </Text>
              <Text color="yellow">{cacheTrend}</Text>
              <Text color="gray"> {cacheTurns.length} turns</Text>
            </Box>
          </Box>
        ) : null}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="cyan">Recent Turns</Text>
        {contextTurns.slice(-10).map((t, i) => (
          <AsciiBar
            key={i}
            label={`Turn ${t.turnIndex}`}
            value={Math.round(t.contextWindowPct ?? 0)}
            max={100}
            width={20}
            warningThreshold={0.7}
            criticalThreshold={0.9}
            unit="%"
          />
        ))}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="cyan">Input Messages Trend</Text>
        {turns.filter(t => t.inputMessagesTokens !== null).slice(-10).map((t, i) => (
          <AsciiBar
            key={i}
            label={`Turn ${t.turnIndex}`}
            value={t.inputMessagesTokens ?? 0}
            max={contextWindow}
            width={20}
            warningThreshold={0.7}
            criticalThreshold={0.9}
          />
        ))}
      </Box>
    </Box>
  );
}
