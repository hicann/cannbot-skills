// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface ReadArgs {
  filePath?: string;
  file_path?: string;
  offset?: number;
  limit?: number;
  summary?: string;
}

export interface ReadRange {
  type: 'full' | 'partial';
  start: number;
  end: number | null;
}

export interface ReadEntry {
  turnId: string;
  turnIndex: number;
  agent: string;
  prompt: string | null;
  subagentSessionId: string | null;
  llmOutput: string | null;
  range: ReadRange;
}

export function parseReadArgs(argsJson: string | null): ReadArgs | null {
  if (!argsJson) return null;
  try {
    const parsed = JSON.parse(argsJson)
    // Normalize file_path → filePath for Claude Code format
    if (parsed.file_path && !parsed.filePath) parsed.filePath = parsed.file_path
    if (parsed.offset && !parsed.start) parsed.offset = parsed.offset ?? parsed.start
    if (parsed.limit && !parsed.end) parsed.limit = parsed.limit ?? parsed.end
    return parsed as ReadArgs;
  } catch {
    return null;
  }
}

export function isDirectoryRead(filePath: string, resultJson: string | null): boolean {
  if (filePath.endsWith('/')) return true;
  if (resultJson && resultJson.includes('<type>directory</type>')) return true;
  return false;
}

export function computeRange(args: ReadArgs): ReadRange {
  const hasOffset = args.offset !== undefined && args.offset !== null;
  const hasLimit = args.limit !== undefined && args.limit !== null;

  if (!hasOffset && !hasLimit) {
    return { type: 'full', start: 0, end: null };
  }
  if (hasOffset && !hasLimit) {
    return { type: 'full', start: args.offset!, end: null };
  }
  if (!hasOffset && hasLimit) {
    return { type: 'partial', start: 0, end: args.limit! };
  }
  return { type: 'partial', start: args.offset!, end: args.offset! + args.limit! };
}

export function rangesOverlap(a: ReadRange, b: ReadRange): boolean {
  if (a.type === 'full' || b.type === 'full') return true;
  return a.start < b.end! && b.start < a.end!;
}

export function mergeIntervals(intervals: Array<[number, number]>): Array<[number, number]> {
  if (intervals.length === 0) return [];
  const sorted = [...intervals].sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [sorted[0]];
  for (let i = 1; i < sorted.length; i++) {
    const last = merged[merged.length - 1];
    const curr = sorted[i];
    if (curr[0] <= last[1]) {
      last[1] = Math.max(last[1], curr[1]);
    } else {
      merged.push(curr);
    }
  }
  return merged;
}

export function analyzeReads(reads: ReadEntry[]): {
  overlappingReads: number;
  totalLinesRead: number;
  uniqueLinesRead: number;
  redundancyRate: number;
} {
  let overlappingReads = 0;
  for (let i = 0; i < reads.length; i++) {
    let overlaps = false;
    for (let j = 0; j < reads.length; j++) {
      if (i !== j && rangesOverlap(reads[i].range, reads[j].range)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) overlappingReads++;
  }

  const partialRanges = reads
    .filter(r => r.range.type === 'partial')
    .map(r => [r.range.start, r.range.end!] as [number, number]);

  const totalLinesRead = partialRanges.reduce((sum, [s, e]) => sum + (e - s), 0);
  const merged = mergeIntervals(partialRanges);
  const uniqueLinesRead = merged.reduce((sum, [s, e]) => sum + (e - s), 0);
  const redundancyRate = totalLinesRead > 0 ? 1 - uniqueLinesRead / totalLinesRead : 0;

  return { overlappingReads, totalLinesRead, uniqueLinesRead, redundancyRate };
}
