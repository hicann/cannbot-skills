// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Box, Text } from 'ink';
import stringWidth from 'string-width';
import { truncateVisual } from '@/cli/utils/table';

export interface DataTableColumn<T> {
  key: string;
  label: string;
  width: number;
  render?: (row: T, selected: boolean) => string;
}

interface DataTableProps<T> {
  columns: DataTableColumn<T>[];
  data: T[];
  selectedIndex: number;
  pageSize?: number;
  page?: number;
  totalPages?: number;
  markedIds?: string[];
  idKey?: string;
}

function padEndVisual(str: string, width: number): string {
  const visualWidth = stringWidth(str);
  if (visualWidth >= width) return str;
  return str + ' '.repeat(width - visualWidth);
}

export function DataTable<T>({ columns, data, selectedIndex, pageSize, page, totalPages, markedIds, idKey }: DataTableProps<T>) {
  const headerRow = columns.map(col => padEndVisual(col.label, col.width)).join(' │ ');
  const separator = columns.map(col => '─'.repeat(col.width)).join('─┼─');

  return (
    <Box flexDirection="column">
      <Text bold>{headerRow}</Text>
      <Text color="gray">{separator}</Text>
      {data.map((row, i) => {
        const selected = i === selectedIndex;
        const rowId = idKey ? String((row as Record<string, unknown>)[idKey] ?? '') : '';
        const marked = markedIds?.includes(rowId) ?? false;
        const cells = columns.map(col => {
          const val = col.render
            ? col.render(row, selected)
            : String((row as Record<string, unknown>)[col.key] ?? '—');
          return padEndVisual(truncateVisual(val, col.width), col.width);
        });
        const line = cells.join(' │ ');
        const prefix = marked ? '◉' : selected ? '▸' : ' ';
        return (
          <Text key={i} color={selected ? 'cyan' : marked ? 'yellow' : undefined} bold={selected || marked}>
            {prefix} {line}
          </Text>
        );
      })}
      {pageSize && totalPages ? (
        <Text color="gray">
          Page {page ?? 1}/{totalPages} │ n:Next p:Prev │ Total items: {data.length}
        </Text>
      ) : null}
    </Box>
  );
}
