// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useMemo, useCallback } from 'react';
import { formatTokens, formatCost, formatDuration, truncate } from '@/cli/utils/format';
import type { ApiBridgeItem } from '@/cli/types';

const STATUS_ICON: Record<string, { icon: string; color: string }> = {
  completed: { icon: '✓', color: 'green' },
  failed: { icon: '✗', color: 'red' },
  running: { icon: '◐', color: 'orange' },
  dispatched: { icon: '→', color: 'gray' },
  timeout: { icon: '⚠', color: 'orange' },
};

export function useInteractionInteraction(bridges: ApiBridgeItem[]) {
  const [cursorIndex, setCursorIndex] = useState(0);
  const [expandedBridges, setExpandedBridges] = useState<Set<string>>(new Set());

  const sortedBridges = useMemo(() =>
    [...bridges].sort((a, b) => {
      const aTime = a.dispatchTimestamp ? new Date(a.dispatchTimestamp).getTime() : 0;
      const bTime = b.dispatchTimestamp ? new Date(b.dispatchTimestamp).getTime() : 0;
      return aTime - bTime;
    }),
  [bridges]);

  const handleUp = useCallback(() => {
    setCursorIndex(i => Math.max(0, i - 1));
  }, []);

  const handleDown = useCallback(() => {
    setCursorIndex(i => Math.min(Math.max(sortedBridges.length - 1, 0), i + 1));
  }, [sortedBridges.length]);

  const handleEnter = useCallback(() => {
    if (cursorIndex >= sortedBridges.length) return;
    const bridgeId = sortedBridges[cursorIndex].bridgeId;
    setExpandedBridges(prev => {
      const next = new Set(prev);
      if (next.has(bridgeId)) next.delete(bridgeId);
      else next.add(bridgeId);
      return next;
    });
  }, [cursorIndex, sortedBridges]);

  return {
    cursorIndex,
    expandedBridges,
    sortedBridges,
    handleUp,
    handleDown,
    handleEnter,
  };
}

interface InteractionsTabProps {
  bridges: ApiBridgeItem[];
  rootAgentName: string | null;
  sessionStartTime: string | null;
  sessionLatencyMs: number | null;
  sortedBridges: ApiBridgeItem[];
  cursorIndex: number;
  expandedBridges: Set<string>;
}

export function InteractionsTab({ bridges, rootAgentName, sessionStartTime, sessionLatencyMs, sortedBridges, cursorIndex, expandedBridges }: InteractionsTabProps) {
  const totalTokens = bridges.reduce((s, b) => s + b.subagentTokens, 0);
  const parentName = rootAgentName ?? 'root';

  const timelineBar = useMemo(() => {
    if (!sessionStartTime || !sessionLatencyMs) return null;
    const sessionStartMs = new Date(sessionStartTime).getTime();
    const totalDuration = sessionLatencyMs || 1;

    const lines: React.ReactNode[] = [];
    const tickPcts = [0, 0.25, 0.5, 0.75, 1];
    const tickLabels = tickPcts.map(pct => {
      const ms = totalDuration * pct;
      const d = new Date(sessionStartMs + ms);
      return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
    });

    lines.push(
      <Text key="timeline-header" color="gray">
        {tickLabels.join(' │ ')}
      </Text>
    );

    for (const bridge of sortedBridges) {
      if (!bridge.dispatchTimestamp) continue;
      const startMs = new Date(bridge.dispatchTimestamp).getTime() - sessionStartMs;
      const startPct = Math.max(0, Math.min(startMs / totalDuration, 1));
      const widthPct = Math.max(0.02, Math.min(bridge.subagentLatencyMs / totalDuration, 1));
      const barWidth = 60;
      const startPos = Math.round(startPct * barWidth);
      const widthLen = Math.max(3, Math.round(widthPct * barWidth));
      const before = '░'.repeat(startPos);
      const name = bridge.subagentName ?? bridge.subagentType ?? 'sub';
      const barLabel = name.length > widthLen - 2 ? name.substring(0, widthLen - 2) : name;
      const after = '░'.repeat(barWidth - startPos - widthLen);
      const stCfg = STATUS_ICON[bridge.status] ?? STATUS_ICON.dispatched;

      lines.push(
        <Text key={`bar-${bridge.bridgeId}`}>
          {before}<Text color={stCfg.color}>{'█'.repeat(widthLen)}</Text>{after} {stCfg.icon}{bridge.status} │ {barLabel}
        </Text>
      );
    }

    return lines;
  }, [sortedBridges, sessionStartTime, sessionLatencyMs]);

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Interactions 🔗</Text>
      <Text color="gray">
        {bridges.length} dispatches │ {formatDuration(sessionLatencyMs ?? 0)} total │ {formatTokens(totalTokens)} tok │ {parentName} → subagents
      </Text>

      {timelineBar && timelineBar.length > 1 && (
        <Box flexDirection="column">
          <Text bold color="cyan">── Timeline ──</Text>
          {timelineBar}
        </Box>
      )}

      <Box flexDirection="column" marginTop={1}>
        <Text bold color="cyan">── Bridges ──</Text>
        {sortedBridges.map((bridge, i) => {
          const isCursor = i === cursorIndex;
          const isExpanded = expandedBridges.has(bridge.bridgeId);
          const stCfg = STATUS_ICON[bridge.status] ?? STATUS_ICON.dispatched;
          const childName = bridge.subagentName ?? bridge.subagentType ?? 'sub';
          const cursorMarker = isCursor ? ' ←' : '';
          const dispatch = bridge.dispatchContent ?? '—';

          return (
            <Box key={`bridge-${i}`} flexDirection="column">
              <Text
                color={isCursor ? 'white' : stCfg.color}
                backgroundColor={isCursor ? 'blue' : undefined}
              >
                {isCursor ? '▸' : ' '} #{i + 1} {childName} │ {stCfg.icon}{bridge.status} │ {formatTokens(bridge.subagentTokens)} │ {formatDuration(bridge.subagentLatencyMs)} │ "{truncate(dispatch, 50)}"{cursorMarker}
              </Text>

              {isExpanded && (
                <Box flexDirection="column">
                  {bridge.dispatchContent && (
                    <Text color="gray">  dispatch: "{truncate(bridge.dispatchContent, 80)}"</Text>
                  )}
                  {bridge.responseContent && (
                    <Text color="gray">  response: "{truncate(bridge.responseContent, 80)}"</Text>
                  )}
                  {bridge.dispatchTimestamp && (
                    <Text color="gray">  dispatched: {bridge.dispatchTimestamp}</Text>
                  )}
                  {bridge.responseTimestamp && (
                    <Text color="gray">  responded: {bridge.responseTimestamp}</Text>
                  )}
                </Box>
              )}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
