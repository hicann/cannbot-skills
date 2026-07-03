// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useInput } from 'ink';
import { useState, useCallback } from 'react';
import { Spinner } from '@/cli/tui/components/Spinner';
import { TabBar } from '@/cli/tui/components/TabBar';
import { useApi } from '@/cli/hooks/useApi';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import { OverviewTab } from '@/cli/tui/tabs/OverviewTab';
import { TurnsTab } from '@/cli/tui/tabs/TurnsTab';
import { WorkflowTab, useWorkflowInteraction } from '@/cli/tui/tabs/WorkflowTab';
import { TraceTab, useTraceInteraction } from '@/cli/tui/tabs/TraceTab';
import { SubagentsTab } from '@/cli/tui/tabs/SubagentsTab';
import { SkillsTab } from '@/cli/tui/tabs/SkillsTab';
import { BridgesTab } from '@/cli/tui/tabs/BridgesTab';
import { ContextTab } from '@/cli/tui/tabs/ContextTab';
import { InteractionsTab, useInteractionInteraction } from '@/cli/tui/tabs/InteractionsTab';
import type { InsightClient } from '@/cli/client';
import type { ApiTurnDetailResponse, ApiTurnItem } from '@/cli/types';

const TABS = [
  { key: 'overview', label: 'Overview', icon: '📋' },
  { key: 'turns', label: 'Turns', icon: '🔄' },
  { key: 'workflow', label: 'Workflow', icon: '✦' },
  { key: 'trace', label: 'Trace', icon: '🔍' },
  { key: 'interactions', label: 'Interactions', icon: '🔗' },
  { key: 'subagents', label: 'Subagents', icon: '🤖' },
  { key: 'skills', label: 'Skills', icon: '🔧' },
  { key: 'bridges', label: 'Bridges', icon: '🔗' },
  { key: 'context', label: 'Context', icon: '📊' },
];

interface SessionDetailProps {
  client: InsightClient;
  taskId: string;
  onBack: () => void;
  onSelectTurn: (turnId: string) => void;
}

export function SessionDetail({ client, taskId, onBack, onSelectTurn }: SessionDetailProps) {
  const [activeTab, setActiveTab] = useState('overview');
  const [turnSelectedIndex, setTurnSelectedIndex] = useState(0);
  const [subagentSelectedIndex, setSubagentSelectedIndex] = useState(0);
  const [bridgeSelectedIndex, setBridgeSelectedIndex] = useState(0);
  const [skillSelectedIndex, setSkillSelectedIndex] = useState(0);
  const [turnDetail, setTurnDetail] = useState<ApiTurnDetailResponse | null>(null);
  const [turnDetailLoading, setTurnDetailLoading] = useState(false);
  const [searchMode, setSearchMode] = useState(false);
  const [searchKeyword, setSearchKeyword] = useState('');
  const [searchResults, setSearchResults] = useState<ApiTurnItem[] | null>(null);

  const fetchSession = useCallback(() => client.getSession(taskId), [client, taskId]);
  const fetchTurns = useCallback(() => client.getTurns(taskId), [client, taskId]);
  const fetchWorkflow = useCallback(() => client.getWorkflow(taskId), [client, taskId]);
  const fetchExecutions = useCallback(() => client.getExecutions(taskId), [client, taskId]);
  const fetchBridges = useCallback(() => client.getBridges(taskId), [client, taskId]);

  const { data: session, loading: sessionLoading, refresh: refreshSession } = useApi(fetchSession, [client, taskId]);
  const { data: turnsData, refresh: refreshTurns } = useApi(fetchTurns, [client, taskId]);
  const { data: workflow, refresh: refreshWorkflow } = useApi(fetchWorkflow, [client, taskId]);
  const { data: executions, refresh: refreshExec } = useApi(fetchExecutions, [client, taskId]);
  const { data: bridgesData, refresh: refreshBridges } = useApi(fetchBridges, [client, taskId]);

  const workflowInteraction = useWorkflowInteraction(
    workflow ?? { phases: [], summary: { totalPhases: 0, totalSteps: 0, totalCheckpoints: 0, totalActiveTimeMs: 0, totalWaitTimeMs: 0, activeTimePct: 0, iterations: 0 } },
    turnsData?.items ?? []
  )

  const traceInteraction = useTraceInteraction(
    searchResults ?? turnsData?.items ?? [],
    bridgesData?.items ?? []
  )

  const interactionInteraction = useInteractionInteraction(
    bridgesData?.items ?? []
  )

  const fetchTurnDetail = useCallback(async (turnId: string) => {
    setTurnDetailLoading(true);
    try {
      const detail = await client.getTurnDetail(turnId);
      setTurnDetail(detail);
    } catch {
      setTurnDetail(null);
    }
    setTurnDetailLoading(false);
  }, [client]);

  const executeSearch = useCallback(async (keyword: string) => {
    try {
      const result = await client.searchTurns(taskId, keyword);
      setSearchResults(result.items as unknown as ApiTurnItem[]);
    } catch {
      setSearchResults(null);
    }
  }, [client, taskId]);

  const refreshAll = useCallback(async () => {
    await Promise.all([refreshSession(), refreshTurns(), refreshWorkflow(), refreshExec(), refreshBridges()]);
  }, [refreshSession, refreshTurns, refreshWorkflow, refreshExec, refreshBridges]);

  const isSearchableTab = activeTab === 'turns' || activeTab === 'trace';

  useKeyboard({
    onTab: () => {
      if (searchMode) return;
      const currentIdx = TABS.findIndex(t => t.key === activeTab);
      const nextIdx = (currentIdx + 1) % TABS.length;
      setActiveTab(TABS[nextIdx].key);
    },
    onEscape: () => {
      if (searchMode) {
        setSearchMode(false);
        setSearchKeyword('');
        setSearchResults(null);
        return;
      }
      onBack();
    },
    onRefresh: searchMode ? undefined : refreshAll,
    onNavigateUp: () => {
      if (searchMode) return;
      if (activeTab === 'workflow') {
        workflowInteraction.handleUp();
      } else if (activeTab === 'trace') {
        traceInteraction.handleUp();
      } else if (activeTab === 'interactions') {
        interactionInteraction.handleUp();
      } else if (activeTab === 'turns') setTurnSelectedIndex(i => Math.max(0, i - 1));
      else if (activeTab === 'subagents') setSubagentSelectedIndex(i => Math.max(0, i - 1));
      else if (activeTab === 'bridges') setBridgeSelectedIndex(i => Math.max(0, i - 1));
      else if (activeTab === 'skills') setSkillSelectedIndex(i => Math.max(0, i - 1));
    },
    onNavigateDown: () => {
      if (searchMode) return;
      if (activeTab === 'workflow') {
        workflowInteraction.handleDown();
      } else if (activeTab === 'trace') {
        traceInteraction.handleDown();
      } else if (activeTab === 'interactions') {
        interactionInteraction.handleDown();
      } else if (activeTab === 'turns') {
        const max = searchResults?.length ?? turnsData?.items.length ?? 0;
        setTurnSelectedIndex(i => Math.min(max - 1, i + 1));
      } else if (activeTab === 'subagents') {
        const max = executions?.subagents.length ?? 0;
        setSubagentSelectedIndex(i => Math.min(max - 1, i + 1));
      } else if (activeTab === 'bridges') {
        const max = bridgesData?.items.length ?? 0;
        setBridgeSelectedIndex(i => Math.min(max - 1, i + 1));
      } else if (activeTab === 'skills') {
        const max = session?.skills.length ?? 0;
        setSkillSelectedIndex(i => Math.min(max - 1, i + 1));
      }
    },
    onEnter: () => {
      if (searchMode) {
        if (searchKeyword.trim()) {
          executeSearch(searchKeyword.trim());
        }
        return;
      }
      if (activeTab === 'workflow') {
        workflowInteraction.handleEnter();
      } else if (activeTab === 'trace') {
        traceInteraction.handleEnter();
      } else if (activeTab === 'interactions') {
        interactionInteraction.handleEnter();
      } else if (activeTab === 'turns') {
        const turns = searchResults ?? turnsData?.items ?? [];
        const selectedTurn = turns[turnSelectedIndex];
        if (selectedTurn) fetchTurnDetail(selectedTurn.turnId);
      }
    },
    onSearch: () => {
      if (!isSearchableTab) return;
      setSearchMode(true);
      setSearchKeyword('');
      setSearchResults(null);
    },
  }, !searchMode);

  useInput((input, key) => {
    if (!searchMode) return;
    if (key.escape) {
      setSearchMode(false);
      setSearchKeyword('');
      setSearchResults(null);
      return;
    }
    if (key.return) {
      if (searchKeyword.trim()) {
        executeSearch(searchKeyword.trim());
      }
      return;
    }
    if (key.backspace || key.delete) {
      setSearchKeyword(prev => prev.slice(0, -1));
      return;
    }
    if (input.length === 1 && !key.ctrl && !key.meta) {
      setSearchKeyword(prev => prev + input);
    }
  }, { isActive: searchMode });

  if (sessionLoading && !session) return <Spinner label="Loading session..." />;

  const effectiveTurns = searchResults ?? turnsData?.items ?? [];

  return (
    <Box flexDirection="column">
      <TabBar tabs={TABS} activeKey={activeTab} />
      {searchMode && (
        <Box flexDirection="column" marginTop={1}>
          <Text bold color="yellow">── Search Mode ──</Text>
          <Text color="gray">Type keyword, Enter to search, Escape to cancel</Text>
          <Text>
            Keyword: <Text bold color="cyan">{searchKeyword}</Text>│
            {searchKeyword ? ' Enter=search │ Esc=cancel' : ' type keyword...'}
          </Text>
          {searchResults && (
            <Text color="green">{searchResults.length} results found</Text>
          )}
        </Box>
      )}
      <Box flexDirection="column" marginTop={1}>
        {activeTab === 'overview' && session && turnsData ? (
          <OverviewTab session={session} turns={turnsData.items} />
        ) : activeTab === 'turns' && turnsData ? (
          <TurnsTab turns={effectiveTurns} selectedIndex={turnSelectedIndex} turnDetail={turnDetail} />
        ) : activeTab === 'workflow' && workflow && turnsData ? (
          <WorkflowTab
            workflow={workflow}
            allTurns={turnsData.items}
            sessionModel={session?.model ?? null}
            flatNodes={workflowInteraction.flatNodes}
            cursorIndex={workflowInteraction.cursorIndex}
            selectedTurnId={workflowInteraction.selectedTurnId}
          />
        ) : activeTab === 'trace' && turnsData && bridgesData ? (
          <TraceTab
            turns={effectiveTurns}
            bridges={bridgesData.items}
            flatNodes={traceInteraction.flatNodes}
            cursorIndex={traceInteraction.cursorIndex}
            expandedTurns={traceInteraction.expandedTurns}
            selectedTurnDetail={traceInteraction.selectedTurnDetail}
          />
        ) : activeTab === 'interactions' && bridgesData && session ? (
          <InteractionsTab
            bridges={bridgesData.items}
            rootAgentName={session.agents.find(a => !a.isSubagent)?.agentName ?? null}
            sessionStartTime={session.startTime}
            sessionLatencyMs={session.totalLatencyMs}
            sortedBridges={interactionInteraction.sortedBridges}
            cursorIndex={interactionInteraction.cursorIndex}
            expandedBridges={interactionInteraction.expandedBridges}
          />
        ) : activeTab === 'subagents' && executions ? (
          <SubagentsTab subagents={executions.subagents} bridges={bridgesData?.items ?? []} selectedIndex={subagentSelectedIndex} />
        ) : activeTab === 'skills' && session && turnsData ? (
          <SkillsTab
            skills={session.skills}
            turns={turnsData.items}
            selectedIndex={skillSelectedIndex}
          />
        ) : activeTab === 'bridges' && bridgesData ? (
          <BridgesTab bridges={bridgesData.items} selectedIndex={bridgeSelectedIndex} />
        ) : activeTab === 'context' && turnsData && session ? (
          <ContextTab turns={turnsData.items} model={session.model} />
        ) : (
          <Spinner label="Loading..." />
        )}
      </Box>
    </Box>
  );
}
