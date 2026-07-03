// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import { useState, useMemo } from 'react';
import { useInput } from 'ink';
import { truncateVisual } from '@/cli/utils/table';
import fs from 'node:fs';
import path from 'node:path';

interface DirEntry {
  name: string;
  fullPath: string;
  isDir: boolean;
  size: number;
}

interface FilePickerProps {
  basePath: string;
  onSelect: (fullPath: string) => void;
  focus: boolean;
}

const PAGE_SIZE = 15;

function readDirEntries(basePath: string): { entries: DirEntry[]; error: string | null } {
  try {
    const stats = fs.statSync(basePath);
    if (!stats.isDirectory()) {
      return { entries: [], error: null };
    }
    const names = fs.readdirSync(basePath).sort((a, b) => {
      const aIsDir = fs.statSync(path.join(basePath, a)).isDirectory();
      const bIsDir = fs.statSync(path.join(basePath, b)).isDirectory();
      if (aIsDir !== bIsDir) return aIsDir ? -1 : 1;
      return a.localeCompare(b);
    });
    const dirEntries: DirEntry[] = [
      { name: '.. (parent)', fullPath: path.resolve(basePath, '..'), isDir: true, size: 0 },
      ...names.map(name => {
        const full = path.join(basePath, name);
        try {
          const s = fs.statSync(full);
          return { name, fullPath: full, isDir: s.isDirectory(), size: s.isFile() ? s.size : 0 };
        } catch {
          return { name, fullPath: full, isDir: false, size: 0 };
        }
      }),
    ];
    return { entries: dirEntries, error: null };
  } catch (e) {
    return { entries: [], error: e instanceof Error ? e.message : 'Cannot read directory' };
  }
}

export function FilePicker({ basePath, onSelect, focus }: FilePickerProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [page, setPage] = useState(0);

  const { entries, error } = useMemo(() => readDirEntries(basePath), [basePath]);
  const totalPages = Math.ceil(entries.length / PAGE_SIZE);
  const pagedEntries = entries.slice(page * PAGE_SIZE, (page + 1) * PAGE_SIZE);

  useInput((input, key) => {
    if (!focus) return;
    if (key.upArrow) {
      if (selectedIndex > 0) {
        setSelectedIndex(selectedIndex - 1);
        if (selectedIndex - 1 < page * PAGE_SIZE) setPage(page - 1);
      }
    } else if (key.downArrow) {
      if (selectedIndex < entries.length - 1) {
        setSelectedIndex(selectedIndex + 1);
        if (selectedIndex + 1 >= (page + 1) * PAGE_SIZE) setPage(page + 1);
      }
    } else if (key.return) {
      const entry = entries[selectedIndex];
      if (entry) onSelect(entry.fullPath);
    } else if (key.escape) {
      onSelect('');
    } else if (input === 'n') {
      if (page < totalPages - 1) { setPage(page + 1); setSelectedIndex((page + 1) * PAGE_SIZE); }
    } else if (input === 'p') {
      if (page > 0) { setPage(page - 1); setSelectedIndex(page * PAGE_SIZE); }
    }
  }, { isActive: focus });

  if (error) {
    return <Text color="red">{error}</Text>;
  }

  if (entries.length === 0) {
    return <Text color="gray">Path is a file (not a directory). Press Enter to use it, Esc to go back.</Text>;
  }

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text bold color="cyan">{basePath}</Text>
      {pagedEntries.map((entry, i) => {
        const globalIndex = page * PAGE_SIZE + i;
        const selected = globalIndex === selectedIndex;
        const icon = entry.isDir ? '\u{1F4C1}' : '\u{1F4C4}';
        const suffix = entry.isDir ? '/' : '';
        let sizeStr = '';
        if (!entry.isDir && entry.size > 0) {
          if (entry.size < 1024) sizeStr = ` (${entry.size}B)`;
          else if (entry.size < 1048576) sizeStr = ` (${(entry.size / 1024).toFixed(1)}K)`;
          else sizeStr = ` (${(entry.size / 1048576).toFixed(1)}M)`;
        }
        const displayName = truncateVisual(`${icon} ${entry.name}${suffix}${sizeStr}`, 60);
        return (
          <Text key={globalIndex} color={selected ? 'cyan' : entry.isDir ? 'blue' : 'gray'} bold={selected}>
            {selected ? '▸ ' : '  '}{displayName}
          </Text>
        );
      })}
      <Text color="gray">↑↓ navigate │ Enter: select/enter dir │ Esc: cancel │ n/p: page │ Total: {entries.length}</Text>
    </Box>
  );
}
