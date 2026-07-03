// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useCallback } from 'react';
import fs from 'node:fs';
import { Spinner } from '@/cli/tui/components/Spinner';
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';
import { ConfirmDialog } from '@/cli/tui/components/ConfirmDialog';
import { TextInput } from '@/cli/tui/components/TextInput';
import { FilePicker } from '@/cli/tui/components/FilePicker';
import { useApi } from '@/cli/hooks/useApi';
import { useKeyboard } from '@/cli/hooks/useKeyboard';
import type { InsightClient } from '@/cli/client';
import type { ApiImportableSession } from '@/cli/types';
import { formatDate, truncate } from '@/cli/utils/format';

interface ImportPanelProps {
  client: InsightClient;
  onBack: () => void;
}

const IMPORT_COLUMNS: DataTableColumn<ApiImportableSession>[] = [
  { key: 'id', label: 'ID', width: 10 },
  { key: 'firstQuery', label: 'Query', width: 35, render: (row) => truncate(row.firstQuery ?? '—', 35) },
  { key: 'turnCount', label: 'Turns', width: 8 },
  { key: 'model', label: 'Model', width: 16 },
  { key: 'createdAt', label: 'Date', width: 16, render: (row) => formatDate(row.createdAt) },
];

type ImportFocus = 'filePath' | 'source' | 'browser' | 'list';

const SOURCE_TYPES = ['opencode-db', 'claude-jsonl'];

function isDirectory(p: string): boolean {
  try { return fs.existsSync(p) && fs.statSync(p).isDirectory(); } catch { return false; }
}

export function ImportPanel({ client, onBack }: ImportPanelProps) {
  const [filePath, setFilePath] = useState('');
  const [source, setSource] = useState('opencode-db');
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [focus, setFocus] = useState<ImportFocus>('filePath');
  const [confirmImport, setConfirmImport] = useState<ApiImportableSession | null>(null);
  const [sourceIndex, setSourceIndex] = useState(0);

  const isDir = filePath && isDirectory(filePath);

  const fetchImportable = useCallback(
    () => (filePath && !isDir) ? client.listImportableSessions(source, filePath) : Promise.resolve({ sessions: [] }),
    [client, source, filePath, isDir],
  );

  const { data, loading, refresh } = useApi(fetchImportable, [client, source, filePath, isDir], {
    cacheKey: (filePath && !isDir) ? `import:${source}:${filePath}` : undefined,
  });

  useKeyboard({
    onEscape: () => {
      if (confirmImport) {
        setConfirmImport(null);
      } else if (focus === 'browser') {
        setFocus('filePath');
      } else if (focus !== 'filePath') {
        setFocus('filePath');
      } else {
        onBack();
      }
    },
    onNavigateUp: () => {
      if (focus === 'list') setSelectedIndex(i => Math.max(0, i - 1));
      else if (focus === 'source') setSourceIndex(i => Math.max(0, i - 1));
    },
    onNavigateDown: () => {
      if (focus === 'list') setSelectedIndex(i => Math.min((data?.sessions.length ?? 0) - 1, i + 1));
      else if (focus === 'source') setSourceIndex(i => Math.min(SOURCE_TYPES.length - 1, i + 1));
    },
    onEnter: () => {
      if (focus === 'filePath') {
        if (!filePath) return;
        if (isDir) setFocus('browser');
        else setFocus('list');
      } else if (focus === 'source') {
        setSource(SOURCE_TYPES[sourceIndex]);
        setFocus('filePath');
      } else if (focus === 'list') {
        const sessions = data?.sessions ?? [];
        const selected = sessions[selectedIndex];
        if (selected) setConfirmImport(selected);
      }
    },
    onTab: () => {
      if (focus === 'filePath') setFocus('source');
      else if (focus === 'source') setFocus('filePath');
      else setFocus('filePath');
    },
    custom: {
      's': () => setFocus('source'),
      'f': () => setFocus('filePath'),
      'b': () => { if (isDir) setFocus('browser'); },
      'l': () => { if (filePath && !isDir) setFocus('list'); },
    },
  });

  const handleBrowserSelect = useCallback((selectedPath: string) => {
    if (!selectedPath) {
      setFocus('filePath');
      return;
    }
    setFilePath(selectedPath);
    if (isDirectory(selectedPath)) {
      setFocus('browser');
    } else {
      setFocus('list');
    }
  }, []);

  if (confirmImport) {
    return (
      <ConfirmDialog
        message={`Import session "${confirmImport.firstQuery ?? confirmImport.id}"?`}
        onConfirm={async () => {
          await client.importSession(source, filePath, confirmImport.id);
          setConfirmImport(null);
          refresh();
        }}
        onCancel={() => setConfirmImport(null)}
      />
    );
  }

  const sessions = data?.sessions ?? [];

  return (
    <Box flexDirection="column">
      <Text bold color="cyan">Import Sessions</Text>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color={focus === 'source' ? 'yellow' : 'gray'}>
          Source: {focus === 'source' ? '' : source}
        </Text>
        {focus === 'source' ? (
          <Box flexDirection="column">
            {SOURCE_TYPES.map((s, i) => (
              <Text key={s} color={i === sourceIndex ? 'cyan' : 'gray'} bold={i === sourceIndex}>
                {i === sourceIndex ? '▸ ' : '  '}{s}
              </Text>
            ))}
            <Text color="gray">↑↓ select │ Enter confirm │ Esc back</Text>
          </Box>
        ) : null}
      </Box>

      <Box flexDirection="column" marginTop={1}>
        <Text bold color={focus === 'filePath' ? 'yellow' : 'gray'}>
          File path: {focus !== 'filePath' ? (filePath || '(not set)') : ''}
        </Text>
        {focus === 'filePath' ? (
          <TextInput
            value={filePath}
            onChange={setFilePath}
            onSubmit={() => {
              if (!filePath) return;
              if (isDirectory(filePath)) setFocus('browser');
              else setFocus('list');
            }}
            placeholder="Enter file or directory path..."
            focus={true}
          />
        ) : null}
      </Box>

      {isDir && focus === 'browser' && filePath ? (
        <FilePicker
          basePath={filePath}
          onSelect={handleBrowserSelect}
          focus={focus === 'browser'}
        />
      ) : null}

      {isDir && focus === 'filePath' && filePath ? (
        <Box marginTop={1}><Text color="gray">Directory detected &mdash; press Enter to browse, or b to open browser.</Text></Box>
      ) : null}

      {loading && !data && !isDir ? <Spinner label="Loading importable sessions..." /> : null}

      {focus === 'list' && filePath && !isDir ? (
        sessions.length > 0 ? (
          <Box flexDirection="column" marginTop={1}>
            <Text bold color="cyan">Available Sessions</Text>
            <DataTable
              columns={IMPORT_COLUMNS}
              data={sessions}
              selectedIndex={selectedIndex}
            />
            <Text color="gray">↑↓ select │ Enter import │ Esc back to input</Text>
          </Box>
        ) : (
          <Box marginTop={1}><Text color="gray">No importable sessions found for this file.</Text></Box>
        )
      ) : null}

      {focus === 'filePath' && !filePath ? (
        <Box marginTop={1}><Text color="gray">Enter a file or directory path. Directories show a file browser.</Text></Box>
      ) : null}
    </Box>
  );
}
