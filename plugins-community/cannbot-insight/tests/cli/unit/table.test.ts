// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { padEndVisual, truncateVisual, renderTable } from '@/cli/utils/table';
import type { TableColumn } from '@/cli/utils/table';

describe('padEndVisual', () => {
  it('pads ASCII string to target width', () => {
    expect(padEndVisual('hello', 10)).toBe('hello     ');
  });

  it('does not pad if already at width', () => {
    expect(padEndVisual('hello', 5)).toBe('hello');
  });

  it('does not truncate if over width', () => {
    expect(padEndVisual('hello world', 5)).toBe('hello world');
  });

  it('pads CJK string correctly (2 visual columns per char)', () => {
    expect(padEndVisual('你好', 8)).toBe('你好    ');
  });

  it('handles mixed ASCII and CJK', () => {
    expect(padEndVisual('a你好', 8)).toBe('a你好   ');
  });
});

describe('truncateVisual', () => {
  it('returns original string if within width', () => {
    expect(truncateVisual('hello', 10)).toBe('hello');
  });

  it('truncates and adds ellipsis', () => {
    expect(truncateVisual('hello world', 8)).toBe('hello w…');
  });

  it('handles CJK characters (2 visual width each)', () => {
    expect(truncateVisual('你好世界测试', 7)).toBe('你好世…');
  });

  it('handles mixed content', () => {
    expect(truncateVisual('a你好世界', 7)).toBe('a你好…');
  });
});

describe('renderTable', () => {
  const columns: TableColumn[] = [
    { key: 'name', label: 'Name', width: 10 },
    { key: 'value', label: 'Value', width: 8 },
  ];

  const data = [
    { name: 'Alice', value: 100 },
    { name: 'Bob你好', value: 200 },
  ];

  it('renders header row', () => {
    const output = renderTable(columns, []);
    expect(output).toContain('Name');
    expect(output).toContain('Value');
  });

  it('renders separator', () => {
    const output = renderTable(columns, []);
    expect(output).toContain('─┼─');
  });

  it('renders data rows', () => {
    const output = renderTable(columns, data);
    expect(output).toContain('Alice');
    expect(output).toContain('100');
  });

  it('handles CJK alignment in data', () => {
    const output = renderTable(columns, data);
    expect(output).toContain('Bob你好');
  });

  it('uses custom renderCell', () => {
    const output = renderTable(columns, data, (row, key) => {
      if (key === 'value') return `$${row.value}`;
      return String(row[key] ?? '');
    });
    expect(output).toContain('$100');
  });
});
