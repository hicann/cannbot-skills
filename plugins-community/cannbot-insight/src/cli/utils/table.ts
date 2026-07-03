// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import stringWidth from 'string-width';

export function padEndVisual(str: string, width: number): string {
  const visualWidth = stringWidth(str);
  if (visualWidth >= width) return str;
  return str + ' '.repeat(width - visualWidth);
}

export function truncateVisual(str: string, maxWidth: number): string {
  if (stringWidth(str) <= maxWidth) return str;
  let result = '';
  let width = 0;
  for (const char of str) {
    const charWidth = stringWidth(char);
    if (width + charWidth > maxWidth - 1) break;
    result += char;
    width += charWidth;
  }
  return result + '…';
}

export interface TableColumn {
  key: string;
  label: string;
  width: number;
}

export function renderTable<T extends Record<string, unknown>>(
  columns: TableColumn[],
  data: T[],
  renderCell?: (row: T, key: string) => string,
): string {
  const lines: string[] = [];

  const header = columns.map(col => padEndVisual(col.label, col.width)).join(' │ ');
  lines.push(header);

  const separator = columns.map(col => '─'.repeat(col.width)).join('─┼─');
  lines.push(separator);

  for (const row of data) {
    const cells = columns.map(col => {
      const raw = renderCell ? renderCell(row, col.key) : String(row[col.key] ?? '—');
      return padEndVisual(truncateVisual(raw, col.width), col.width);
    });
    lines.push(cells.join(' │ '));
  }

  return lines.join('\n');
}
