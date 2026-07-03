// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import {
  parseReadArgs,
  isDirectoryRead,
  computeRange,
  rangesOverlap,
  mergeIntervals,
  analyzeReads,
} from '../src/lib/file-reads.ts';
import type { ReadEntry, ReadRange } from '../src/lib/file-reads.ts';

describe('file-reads', () => {
  describe('parseReadArgs', () => {
    it('parses valid JSON with filePath', () => {
      const result = parseReadArgs('{"filePath":"/src/foo.ts","offset":10,"limit":20}');
      expect(result).toEqual({ filePath: '/src/foo.ts', offset: 10, limit: 20 });
    });

    it('parses JSON with summary field', () => {
      const result = parseReadArgs('{"filePath":"/src/foo.ts","summary":"src/foo.ts"}');
      expect(result).toEqual({ filePath: '/src/foo.ts', summary: 'src/foo.ts' });
    });

    it('returns null for null input', () => {
      expect(parseReadArgs(null)).toBeNull();
    });

    it('returns null for invalid JSON', () => {
      expect(parseReadArgs('not json')).toBeNull();
    });
  });

  describe('isDirectoryRead', () => {
    it('detects directory by trailing slash in filePath', () => {
      expect(isDirectoryRead('/src/components/', null)).toBe(true);
    });

    it('detects directory by resultJson content', () => {
      expect(isDirectoryRead('/src/components', '<type>directory</type>')).toBe(true);
    });

    it('returns false for file paths', () => {
      expect(isDirectoryRead('/src/foo.ts', null)).toBe(false);
    });

    it('returns false for file with non-directory result', () => {
      expect(isDirectoryRead('/src/foo.ts', 'file content here')).toBe(false);
    });
  });

  describe('computeRange', () => {
    it('full read with no offset or limit', () => {
      const range = computeRange({ filePath: '/foo.ts' });
      expect(range).toEqual({ type: 'full', start: 0, end: null });
    });

    it('full read from offset (no limit)', () => {
      const range = computeRange({ filePath: '/foo.ts', offset: 50 });
      expect(range).toEqual({ type: 'full', start: 50, end: null });
    });

    it('partial read with limit only', () => {
      const range = computeRange({ filePath: '/foo.ts', limit: 30 });
      expect(range).toEqual({ type: 'partial', start: 0, end: 30 });
    });

    it('partial read with offset and limit', () => {
      const range = computeRange({ filePath: '/foo.ts', offset: 50, limit: 30 });
      expect(range).toEqual({ type: 'partial', start: 50, end: 80 });
    });
  });

  describe('rangesOverlap', () => {
    it('detects overlapping partial ranges', () => {
      const a: ReadRange = { type: 'partial', start: 20, end: 50 };
      const b: ReadRange = { type: 'partial', start: 40, end: 90 };
      expect(rangesOverlap(a, b)).toBe(true);
    });

    it('detects non-overlapping partial ranges', () => {
      const a: ReadRange = { type: 'partial', start: 0, end: 20 };
      const b: ReadRange = { type: 'partial', start: 50, end: 80 };
      expect(rangesOverlap(a, b)).toBe(false);
    });

    it('adjacent ranges do not overlap', () => {
      const a: ReadRange = { type: 'partial', start: 0, end: 20 };
      const b: ReadRange = { type: 'partial', start: 20, end: 40 };
      expect(rangesOverlap(a, b)).toBe(false);
    });

    it('full read overlaps with any other read', () => {
      const full: ReadRange = { type: 'full', start: 0, end: null };
      const partial: ReadRange = { type: 'partial', start: 50, end: 80 };
      expect(rangesOverlap(full, partial)).toBe(true);
    });

    it('two full reads overlap', () => {
      const a: ReadRange = { type: 'full', start: 0, end: null };
      const b: ReadRange = { type: 'full', start: 50, end: null };
      expect(rangesOverlap(a, b)).toBe(true);
    });

    it('contained range overlaps', () => {
      const outer: ReadRange = { type: 'partial', start: 0, end: 100 };
      const inner: ReadRange = { type: 'partial', start: 20, end: 50 };
      expect(rangesOverlap(outer, inner)).toBe(true);
    });
  });

  describe('mergeIntervals', () => {
    it('merges overlapping intervals', () => {
      const result = mergeIntervals([[20, 50], [40, 90]]);
      expect(result).toEqual([[20, 90]]);
    });

    it('keeps non-overlapping intervals separate', () => {
      const result = mergeIntervals([[0, 20], [50, 80]]);
      expect(result).toEqual([[0, 20], [50, 80]]);
    });

    it('merges adjacent intervals', () => {
      const result = mergeIntervals([[0, 20], [20, 40]]);
      expect(result).toEqual([[0, 40]]);
    });

    it('handles empty input', () => {
      expect(mergeIntervals([])).toEqual([]);
    });

    it('merges multiple overlapping intervals', () => {
      const result = mergeIntervals([[0, 30], [20, 50], [40, 70]]);
      expect(result).toEqual([[0, 70]]);
    });

    it('handles unsorted input', () => {
      const result = mergeIntervals([[40, 90], [0, 20], [15, 50]]);
      expect(result).toEqual([[0, 90]]);
    });
  });

  describe('analyzeReads', () => {
    it('overlapping ranges: [20,50) and [40,90)', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 20, end: 50 } },
        { turnId: 't2', turnIndex: 1, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 40, end: 90 } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(2);
      expect(result.totalLinesRead).toBe(80);
      expect(result.uniqueLinesRead).toBe(70);
      expect(result.redundancyRate).toBeCloseTo(1 - 70 / 80);
    });

    it('non-overlapping ranges: [0,20) and [50,80)', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 0, end: 20 } },
        { turnId: 't2', turnIndex: 1, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 50, end: 80 } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(0);
      expect(result.totalLinesRead).toBe(50);
      expect(result.uniqueLinesRead).toBe(50);
      expect(result.redundancyRate).toBe(0);
    });

    it('full read with partial reads', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'full', start: 0, end: null } },
        { turnId: 't2', turnIndex: 1, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 20, end: 50 } },
        { turnId: 't3', turnIndex: 2, agent: 'sub', prompt: null, subagentSessionId: 'sub-1', llmOutput: null, range: { type: 'partial', start: 40, end: 90 } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(3);
      expect(result.totalLinesRead).toBe(80);
      expect(result.uniqueLinesRead).toBe(70);
      expect(result.redundancyRate).toBeCloseTo(1 - 70 / 80);
    });

    it('all full reads', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'full', start: 0, end: null } },
        { turnId: 't2', turnIndex: 1, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'full', start: 0, end: null } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(2);
      expect(result.totalLinesRead).toBe(0);
      expect(result.uniqueLinesRead).toBe(0);
      expect(result.redundancyRate).toBe(0);
    });

    it('single read has no overlap', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 0, end: 100 } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(0);
      expect(result.totalLinesRead).toBe(100);
      expect(result.uniqueLinesRead).toBe(100);
    });

    it('empty reads', () => {
      const result = analyzeReads([]);
      expect(result.overlappingReads).toBe(0);
      expect(result.totalLinesRead).toBe(0);
      expect(result.uniqueLinesRead).toBe(0);
      expect(result.redundancyRate).toBe(0);
    });

    it('mixed full and partial with spec scenario: 1 full + [20,50) + [40,90)', () => {
      const reads: ReadEntry[] = [
        { turnId: 't1', turnIndex: 0, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'full', start: 0, end: null } },
        { turnId: 't2', turnIndex: 1, agent: 'root', prompt: null, subagentSessionId: null, llmOutput: null, range: { type: 'partial', start: 20, end: 50 } },
        { turnId: 't3', turnIndex: 2, agent: 'sub', prompt: null, subagentSessionId: 'sub-1', llmOutput: null, range: { type: 'partial', start: 40, end: 90 } },
      ];
      const result = analyzeReads(reads);
      expect(result.overlappingReads).toBe(3);
      expect(result.totalLinesRead).toBe(80);
      expect(result.uniqueLinesRead).toBe(70);
      expect(result.redundancyRate).toBeCloseTo(0.125);
    });
  });
});
