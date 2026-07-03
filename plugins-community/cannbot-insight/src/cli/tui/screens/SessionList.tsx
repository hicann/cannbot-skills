// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useCallback } from 'react';
import { Spinner } from '@/cli/tui/components/Spinner';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import { TextInput } from '@/cli/tui/components/TextInput';
import { useApi } from '@/cli/hooks/useApi';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import type { InsightClient } from '@/cli/client';
import type { ApiSessionListItem } from '@/cli/types';
import { formatTokens, formatCost, formatDuration, formatDate, truncate } from '@/cli/utils/format';

interface SessionListProps {
  client: InsightClient;
  onSelect: (taskId: string) => void;
  onCompare?: (taskId1: string, taskId2: string) => void;
  onImport?: () => void;
  onDelete?: (taskId: string) => void;
  onUpload?: (taskId: string) => void;
}

const PAGE_SIZE = 20;

const SESSION_COLUMNS: DataTableColumn<ApiSessionListItem>[] = [
  { key: 'taskId', label: 'Task ID', width: 8 },
  { key: 'query', label: 'Query', width: 35, render: (row) => truncate(row.query ?? '—', 35) },
  { key: 'model', label: 'Model', width: 16 },
  { key: 'totalTokens', label: 'Tokens', width: 10, render: (row) => formatTokens(row.totalTokens) },
  { key: 'totalCost', label: 'Cost', width: 8, render: (row) => formatCost(row.totalCost) },
  { key: 'totalLatencyMs', label: 'Duration', width: 10, render: (row) => formatDuration(row.totalLatencyMs) },
  { key: 'startTime', label: 'Start', width: 16, render: (row) => formatDate(row.startTime) },
  { key: 'user', label: 'User', width: 12 },
];

export function SessionList({ client, onSelect, onCompare, onImport, onDelete, onUpload }: SessionListProps) {
  const [page, setPage] = useState(1);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [filterText, setFilterText] = useState('');
  const [searchMode, setSearchMode] = useState(false);
  const [compareIds, setCompareIds] = useState<string[]>([]);

  const fetchSessions = useCallback(
    () => client.listSessions({ page, pageSize: PAGE_SIZE }),
    [client, page],
  );

  const { data, loading, error, refresh } = useApi(fetchSessions, [client, page]);

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  const filteredItems = filterText
    ? items.filter(item => {
        const q = filterText.toLowerCase();
        return (item.taskId?.toLowerCase().includes(q) ?? false)
          || (item.query?.toLowerCase().includes(q) ?? false)
          || (item.model?.toLowerCase().includes(q) ?? false)
          || (item.user?.toLowerCase().includes(q) ?? false);
      })
    : items;

  useKeyboard({
    onNavigateUp: () => { if (!searchMode) setSelectedIndex(i => Math.max(0, i - 1)); },
    onNavigateDown: () => { if (!searchMode) setSelectedIndex(i => Math.min(filteredItems.length - 1, i + 1)); },
    onEnter: () => {
      if (searchMode) {
        setSearchMode(false);
        return;
      }
      const selected = filteredItems[selectedIndex];
      if (selected) onSelect(selected.taskId);
    },
    onRefresh: () => refresh(),
    onSearch: () => setSearchMode(m => !m),
    onPageNext: () => {
      if (page < totalPages) {
        setPage(p => p + 1);
        setSelectedIndex(0);
      }
    },
    onPagePrev: () => {
      if (page > 1) {
        setPage(p => p - 1);
        setSelectedIndex(0);
      }
    },
    onEscape: () => {
      if (searchMode) {
        setSearchMode(false);
        setFilterText('');
      }
    },
    custom: searchMode ? {} : {
      ' ': () => {
        const selected = filteredItems[selectedIndex];
        if (selected) {
          setCompareIds(ids => {
            if (ids.includes(selected.taskId)) return ids.filter(id => id !== selected.taskId);
            if (ids.length >= 2) return [selected.taskId];
            return [...ids, selected.taskId];
          });
        }
      },
      'c': () => {
        if (compareIds.length === 2 && onCompare) {
          onCompare(compareIds[0], compareIds[1]);
          setCompareIds([]);
        }
      },
      'i': () => onImport?.(),
      'u': () => {
        const selected = filteredItems[selectedIndex];
        if (selected) onUpload?.(selected.taskId);
      },
      'd': () => {
        const selected = filteredItems[selectedIndex];
        if (selected) onDelete?.(selected.taskId);
      },
    },
  });

  if (loading && !data) return <Spinner label="Loading sessions..." />;
  if (error) return <Text color="red">Error: {error.message}</Text>;

  return (
    <Box flexDirection="column">
      {searchMode ? (
        <Box marginBottom={1}>
          <Text color="cyan">Search: </Text>
          <TextInput
            value={filterText}
            onChange={setFilterText}
            onSubmit={() => setSearchMode(false)}
            placeholder="Type to filter..."
            focus={true}
          />
        </Box>
      ) : null}
      {compareIds.length > 0 && !searchMode ? (
        <Text color="yellow">Compare: [{compareIds.join(', ')}] │ Space: toggle │ c: compare</Text>
      ) : null}
      <DataTable
        columns={SESSION_COLUMNS}
        data={filteredItems}
        selectedIndex={selectedIndex}
        pageSize={PAGE_SIZE}
        page={page}
        totalPages={totalPages}
        markedIds={compareIds}
        idKey="taskId"
      />
      <Box marginTop={1}><Text color="gray">
        {total} sessions │ Page {page}/{totalPages} │ n:Next p:Prev /:Search r:Refresh Space:Compare │ i:Import u:Upload d:Delete Enter:Select
      </Text></Box>
    </Box>
  );
}
