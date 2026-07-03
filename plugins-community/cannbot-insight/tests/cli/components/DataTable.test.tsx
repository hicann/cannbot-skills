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
import { DataTable, DataTableColumn } from '@/cli/tui/components/DataTable';

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
  { id: '003', name: '中文测试', value: 7 },
];

describe('DataTable', () => {
  it('renders header row', () => {
    const { getPlainText } = renderTui(<DataTable columns={columns} data={data} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('ID');
    expect(output).toContain('Name');
    expect(output).toContain('Value');
  });

  it('renders separator line', () => {
    const { getPlainText } = renderTui(<DataTable columns={columns} data={data} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('─');
  });

  it('renders data rows', () => {
    const { getPlainText } = renderTui(<DataTable columns={columns} data={data} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('001');
    expect(output).toContain('Alice');
    expect(output).toContain('002');
    expect(output).toContain('Bob');
  });

  it('renders CJK characters', () => {
    const { getPlainText } = renderTui(<DataTable columns={columns} data={data} selectedIndex={2} />);
    const output = getPlainText();
    expect(output).toContain('中文测试');
  });

  it('shows selection marker for selected row', () => {
    const { getPlainText } = renderTui(<DataTable columns={columns} data={data} selectedIndex={1} />);
    const output = getPlainText();
    expect(output).toContain('▸');
  });

  it('renders pagination info when provided', () => {
    const { getPlainText } = renderTui(
      <DataTable columns={columns} data={data} selectedIndex={0} pageSize={20} page={1} totalPages={3} />,
    );
    const output = getPlainText();
    expect(output).toContain('Page 1/3');
  });

  it('uses custom render function', () => {
    const customCols: DataTableColumn<TestRow>[] = [
      { key: 'id', label: 'ID', width: 10 },
      { key: 'value', label: 'Value', width: 15, render: (row) => `val=${row.value}` },
    ];
    const { getPlainText } = renderTui(<DataTable columns={customCols} data={data} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('val=42');
  });

  it('shows dash for null values', () => {
    const nullableData: Array<{ id: string; name: string | null; value: number | null }> = [{ id: '004', name: null, value: null }];
    const nullableCols: DataTableColumn<{ id: string; name: string | null; value: number | null }>[] = [
      { key: 'id', label: 'ID', width: 10 },
      { key: 'name', label: 'Name', width: 20 },
      { key: 'value', label: 'Value', width: 8 },
    ];
    const { getPlainText } = renderTui(<DataTable columns={nullableCols} data={nullableData} selectedIndex={0} />);
    const output = getPlainText();
    expect(output).toContain('—');
  });
});
