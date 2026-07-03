// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useState, useCallback, useMemo } from 'react';
import type { DataTableColumn } from '@/cli/tui/components/DataTable';

export interface UseTableResult<T> {
  selectedIndex: number;
  selectedRow: T | null;
  sortKey: string | null;
  sortDir: 'asc' | 'desc';
  filterText: string;
  page: number;
  visibleData: T[];
  totalPages: number;
  selectUp: () => void;
  selectDown: () => void;
  setSelectedIndex: (i: number) => void;
  setSort: (key: string, dir?: 'asc' | 'desc') => void;
  setFilterText: (text: string) => void;
  setPage: (page: number) => void;
  pageNext: () => void;
  pagePrev: () => void;
}

export function useTable<T>(
  data: T[],
  columns: DataTableColumn<T>[],
  pageSize = 20,
): UseTableResult<T> {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [sortKey, setSortKeyState] = useState<string | null>(null);
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('asc');
  const [filterText, setFilterTextState] = useState('');
  const [page, setPageState] = useState(1);

  const visibleData = useMemo(() => {
    let filtered = data;
    if (filterText) {
      filtered = data.filter(row =>
        columns.some(col => {
          const val = col.render
            ? col.render(row, false)
            : String((row as Record<string, unknown>)[col.key] ?? '');
          return val.toLowerCase().includes(filterText.toLowerCase());
        }),
      );
    }

    if (sortKey) {
      filtered = [...filtered].sort((a, b) => {
        const aVal = (a as Record<string, unknown>)[sortKey];
        const bVal = (b as Record<string, unknown>)[sortKey];
        const aNum = typeof aVal === 'number' ? aVal : null;
        const bNum = typeof bVal === 'number' ? bVal : null;
        if (aNum !== null && bNum !== null) {
          return sortDir === 'asc' ? aNum - bNum : bNum - aNum;
        }
        const aStr = String(aVal ?? '');
        const bStr = String(bVal ?? '');
        return sortDir === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
      });
    }

    const start = (page - 1) * pageSize;
    return filtered.slice(start, start + pageSize);
  }, [data, columns, filterText, sortKey, sortDir, page, pageSize]);

  const totalFiltered = filterText
    ? data.filter(row =>
        columns.some(col => {
          const val = col.render
            ? col.render(row, false)
            : String((row as Record<string, unknown>)[col.key] ?? '');
          return val.toLowerCase().includes(filterText.toLowerCase());
        }),
      ).length
    : data.length;

  const totalPages = Math.max(1, Math.ceil(totalFiltered / pageSize));

  const selectedRow = visibleData[selectedIndex] ?? null;

  const selectUp = useCallback(() => {
    setSelectedIndex(i => Math.max(0, i - 1));
  }, []);

  const selectDown = useCallback(() => {
    setSelectedIndex(i => Math.min(visibleData.length - 1, i + 1));
  }, [visibleData.length]);

  const setSort = useCallback((key: string, dir?: 'asc' | 'desc') => {
    setSortKeyState(key);
    if (dir) setSortDir(dir);
  }, []);

  const setFilterText = useCallback((text: string) => {
    setFilterTextState(text);
    setPageState(1);
    setSelectedIndex(0);
  }, []);

  const pageNext = useCallback(() => {
    setPageState(p => Math.min(totalPages, p + 1));
    setSelectedIndex(0);
  }, [totalPages]);

  const pagePrev = useCallback(() => {
    setPageState(p => Math.max(1, p - 1));
    setSelectedIndex(0);
  }, []);

  return {
    selectedIndex,
    selectedRow,
    sortKey,
    sortDir,
    filterText,
    page,
    visibleData,
    totalPages,
    selectUp,
    selectDown,
    setSelectedIndex,
    setSort,
    setFilterText,
    setPage: setPageState,
    pageNext,
    pagePrev,
  };
}
