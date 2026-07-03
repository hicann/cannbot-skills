// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import React from 'react';
import { renderTui } from '../../helpers/render-tui';
import { useTable } from '@/cli/hooks/useTable';
import type { DataTableColumn } from '@/cli/tui/components/DataTable';
import { Box, Text } from 'ink';

interface TestRow {
  id: string;
  name: string;
  value: number;
}

const columns: DataTableColumn<TestRow>[] = [
  { key: 'id', label: 'ID', width: 10 },
  { key: 'name', label: 'Name', width: 20 },
  { key: 'value', label: 'Value', width: 8 },
];

const data: TestRow[] = [
  { id: '001', name: 'Alice', value: 42 },
  { id: '002', name: 'Bob', value: 99 },
  { id: '003', name: 'Carol', value: 7 },
  { id: '004', name: 'Dave', value: 55 },
];

function TestTableComponent({ d, cols, ps }: { d: TestRow[]; cols: DataTableColumn<TestRow>[]; ps: number }) {
  const table = useTable(d, cols, ps);
  return (
    <Box flexDirection="column">
      <Text>idx:{table.selectedIndex} rows:{table.visibleData.length} page:{table.page} tp:{table.totalPages}</Text>
      <Text>sort:{table.sortKey ?? 'none'} dir:{table.sortDir} filter:{table.filterText}</Text>
      <Text>selected:{table.selectedRow?.id ?? 'none'}</Text>
    </Box>
  );
}

describe('useTable', () => {
  it('returns initial state', () => {
    const { getPlainText } = renderTui(<TestTableComponent d={data} cols={columns} ps={20} />);
    const output = getPlainText();
    expect(output).toContain('idx:0');
    expect(output).toContain('rows:4');
    expect(output).toContain('page:1');
    expect(output).toContain('sort:none');
    expect(output).toContain('filter:');
    expect(output).toContain('selected:001');
  });

  it('calculates totalPages correctly', () => {
    const bigData = Array.from({ length: 50 }, (_, i) => ({ id: String(i), name: `Item ${i}`, value: i }));
    const { getPlainText } = renderTui(<TestTableComponent d={bigData} cols={columns} ps={20} />);
    const output = getPlainText();
    expect(output).toContain('tp:3');
  });

  it('single page for small data', () => {
    const { getPlainText } = renderTui(<TestTableComponent d={data} cols={columns} ps={20} />);
    const output = getPlainText();
    expect(output).toContain('tp:1');
  });
});
