// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useMemo, useCallback } from 'react';
import { formatTokens, formatDuration, truncate } from '@/cli/utils/format';
import type { ApiTurnItem, ApiBridgeItem } from '@/cli/types';

type SourceType = 'user_input' | 'model_output' | 'tool_output' | 'root_agent_dispatch' | 'subagent_output' | 'bridge_response'
type PropagationMedium = 'model_reasoning' | 'task_dispatch' | 'subagent_return' | 'tool_output' | 'user_input'

const SOURCE_CONFIG: Record<SourceType, { icon: string; label: string; color: string }> = {
  user_input: { icon: '👤', label: 'user', color: 'blue' },
  model_output: { icon: '🤖', label: 'model', color: 'green' },
  tool_output: { icon: '🔧', label: 'tool', color: 'magenta' },
  root_agent_dispatch: { icon: '📤', label: 'dispatch', color: 'orange' },
  subagent_output: { icon: '🤖', label: 'sub', color: 'green' },
  bridge_response: { icon: '📥', label: 'return', color: 'gray' },
};

const MEDIUM_CONFIG: Record<PropagationMedium, { icon: string; label: string }> = {
  model_reasoning: { icon: '↓', label: 'model_reasoning' },
  task_dispatch: { icon: '↓', label: 'task_dispatch' },
  subagent_return: { icon: '↓', label: 'subagent_return' },
  tool_output: { icon: '↓', label: 'tool_output' },
  user_input: { icon: '↓', label: 'user_input' },
};

function classifySource(turn: ApiTurnItem, bridges: ApiBridgeItem[]): SourceType {
  if (turn.role === 'user') return 'user_input';
  if (turn.role === 'tool_result') return 'tool_output';
  if (turn.role === 'assistant' && turn.isSubagent) {
    const isBridgeResponse = bridges.some(b => b.responseTurnId === turn.turnId);
    return isBridgeResponse ? 'bridge_response' : 'subagent_output';
  }
  if (turn.role === 'assistant') {
    const hasDispatch = bridges.some(b => b.dispatchTurnId === turn.turnId);
    return hasDispatch ? 'root_agent_dispatch' : 'model_output';
  }
  return 'model_output';
}

function inferMedium(prevNode: TraceNode, currNode: TraceNode): PropagationMedium | null {
  if (currNode.sourceType === 'user_input') return 'user_input';
  if (currNode.sourceType === 'tool_output') return 'tool_output';
  if (currNode.sourceType === 'root_agent_dispatch') return 'task_dispatch';
  if (currNode.sourceType === 'bridge_response') return 'subagent_return';
  return 'model_reasoning';
}

interface TraceNode {
  turn: ApiTurnItem;
  sourceType: SourceType;
  depth: number;
  isBridgeDispatch: boolean;
  bridgeSubagentName: string | null;
  bridgeDispatchContent: string | null;
  medium: PropagationMedium | null;
}

function buildTraceNodes(turns: ApiTurnItem[], bridges: ApiBridgeItem[]): TraceNode[] {
  const dispatchBridgeMap = new Map<string, ApiBridgeItem>();
  for (const b of bridges) {
    if (b.dispatchTurnId) dispatchBridgeMap.set(b.dispatchTurnId, b);
  }

  const nodes: TraceNode[] = [];

  for (const turn of turns) {
    const source = classifySource(turn, bridges);
    const dispatchBridge = dispatchBridgeMap.get(turn.turnId);
    const isDispatch = dispatchBridge != null;
    const depth = turn.isSubagent ? 1 : 0;

    nodes.push({
      turn,
      sourceType: source,
      depth,
      isBridgeDispatch: isDispatch,
      bridgeSubagentName: dispatchBridge?.subagentName ?? dispatchBridge?.subagentType ?? null,
      bridgeDispatchContent: dispatchBridge?.dispatchContent ?? null,
      medium: null,
    });
  }

  for (let i = 1; i < nodes.length; i++) {
    nodes[i].medium = inferMedium(nodes[i - 1], nodes[i]);
  }

  return nodes;
}

export function useTraceInteraction(turns: ApiTurnItem[], bridges: ApiBridgeItem[]) {
  const [cursorIndex, setCursorIndex] = useState(0);
  const [expandedTurns, setExpandedTurns] = useState<Set<string>>(new Set());
  const [selectedTurnDetail, setSelectedTurnDetail] = useState<ApiTurnItem | null>(null);

  const flatNodes = useMemo(
    () => buildTraceNodes(turns, bridges),
    [turns, bridges]
  );

  const handleUp = useCallback(() => {
    setCursorIndex(i => Math.max(0, i - 1));
  }, []);

  const handleDown = useCallback(() => {
    setCursorIndex(i => Math.min(Math.max(flatNodes.length - 1, 0), i + 1));
  }, [flatNodes.length]);

  const handleEnter = useCallback(() => {
    if (cursorIndex >= flatNodes.length) return;
    const node = flatNodes[cursorIndex];
    const turnId = node.turn.turnId;

    if (expandedTurns.has(turnId)) {
      setExpandedTurns(prev => {
        const next = new Set(prev);
        next.delete(turnId);
        return next;
      });
      setSelectedTurnDetail(null);
    } else {
      setExpandedTurns(prev => {
        const next = new Set(prev);
        next.add(turnId);
        return next;
      });
      setSelectedTurnDetail(node.turn);
    }
  }, [cursorIndex, flatNodes, expandedTurns]);

  return {
    cursorIndex,
    expandedTurns,
    selectedTurnDetail,
    flatNodes,
    handleUp,
    handleDown,
    handleEnter,
  };
}

interface TraceTabProps {
  turns: ApiTurnItem[];
  bridges: ApiBridgeItem[];
  flatNodes: TraceNode[];
  cursorIndex: number;
  expandedTurns: Set<string>;
  selectedTurnDetail: ApiTurnItem | null;
}

export function TraceTab({ turns, bridges, flatNodes, cursorIndex, expandedTurns, selectedTurnDetail }: TraceTabProps) {
  const sourceStats = useMemo(() => {
    const counts: Record<SourceType, number> = {
      user_input: 0, model_output: 0, tool_output: 0,
      root_agent_dispatch: 0, subagent_output: 0, bridge_response: 0,
    };
    for (const node of flatNodes) {
      counts[node.sourceType]++;
    }
    return counts;
  }, [flatNodes]);

  return (
    <Box flexDirection="column" gap={1}>
      <Text bold color="cyan">Trace 🔍</Text>
      <Text color="gray">
        {turns.length} turns │ {bridges.length} bridges │ ↑↓ navigate │ Enter expand
      </Text>

      <Box flexDirection="row" gap={1}>
        {Object.entries(sourceStats).filter(([, c]) => c > 0).map(([type, c]) => {
          const cfg = SOURCE_CONFIG[type as SourceType];
          return (
            <Text key={type} color={cfg.color}>
              {cfg.icon}{cfg.label}:{c}
            </Text>
          );
        })}
      </Box>

      {flatNodes.map((node, i) => {
        const isCursor = i === cursorIndex;
        const indent = '  '.repeat(node.depth);
        const cfg = SOURCE_CONFIG[node.sourceType];
        const isExpanded = expandedTurns.has(node.turn.turnId);
        const cursorMarker = isCursor ? ' ←' : '';

        const mediumCfg = node.medium ? MEDIUM_CONFIG[node.medium] : null;

        return (
          <Box key={`trace-${i}`} flexDirection="column">
            {mediumCfg && i > 0 && (
              <Text color="gray">
                {indent}{mediumCfg.icon} {mediumCfg.label}
              </Text>
            )}
            <Text
              color={isCursor ? 'white' : cfg.color}
              backgroundColor={isCursor ? 'blue' : undefined}
            >
              {indent}{cfg.icon} #{node.turn.turnIndex} {node.turn.role} │ {node.turn.agentName ?? '?'} │ {formatTokens(node.turn.totalTokens)} │ {cfg.label}{cursorMarker}
            </Text>

            {node.isBridgeDispatch && (
              <Text
                color={isCursor ? 'white' : 'orange'}
                backgroundColor={isCursor ? 'blue' : undefined}
              >
                {indent}  → spawn: {node.bridgeSubagentName ?? 'sub'}{node.bridgeDispatchContent ? ` │ "${truncate(node.bridgeDispatchContent, 40)}"` : ''}
              </Text>
            )}

            {isExpanded && node.turn.contentSummary && (
              <Text
                color="gray"
                backgroundColor={isCursor ? 'blue' : undefined}
              >
                {indent}  │ {truncate(node.turn.contentSummary, 80)}
              </Text>
            )}

            {isExpanded && node.turn.toolCalls.length > 0 && (
              <Box flexDirection="column">
                {node.turn.toolCalls.map((tc, j) => (
                  <Text
                    key={j}
                    color="gray"
                    backgroundColor={isCursor ? 'blue' : undefined}
                  >
                    {indent}  │ {tc.state === 'error' ? '✗' : '✓'} {tc.toolName ?? '?'} │ {formatDuration(tc.durationMs)}
                  </Text>
                ))}
              </Box>
            )}
          </Box>
        );
      })}

      {selectedTurnDetail && (
        <Box flexDirection="column" marginTop={1} borderStyle="single" paddingX={1}>
          <Text bold color="cyan">
            ── #{selectedTurnDetail.turnIndex} {selectedTurnDetail.role} │ {selectedTurnDetail.agentName ?? '?'} ──
          </Text>
          <Text>
            {formatTokens(selectedTurnDetail.inputTokens)} in │ {formatTokens(selectedTurnDetail.outputTokens)} out │ {formatTokens(selectedTurnDetail.cacheReadTokens)} cache │ {formatDuration(selectedTurnDetail.latencyMs)}
          </Text>
          {selectedTurnDetail.contextWindowPct != null && (
            <Text>
              Context: {selectedTurnDetail.contextWindowPct.toFixed(1)}%
            </Text>
          )}
        </Box>
      )}
    </Box>
  );
}
