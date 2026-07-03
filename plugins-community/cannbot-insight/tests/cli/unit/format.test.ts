// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { formatTokens, formatCost, formatDuration, formatDate, formatPercent, truncate } from '@/cli/utils/format';

describe('format utils', () => {
  describe('formatTokens', () => {
    it('returns — for null', () => {
      expect(formatTokens(null)).toBe('—');
    });

    it('returns — for undefined', () => {
      expect(formatTokens(undefined)).toBe('—');
    });

    it('returns raw number for small values', () => {
      expect(formatTokens(42)).toBe('42');
    });

    it('formats thousands with K suffix', () => {
      expect(formatTokens(1500)).toBe('1.5K');
    });

    it('formats millions with M suffix', () => {
      expect(formatTokens(1_500_000)).toBe('1.5M');
    });
  });

  describe('formatCost', () => {
    it('returns — for null', () => {
      expect(formatCost(null)).toBe('—');
    });

    it('returns $0 for zero', () => {
      expect(formatCost(0)).toBe('$0');
    });

    it('formats small costs with 4 decimals', () => {
      expect(formatCost(0.005)).toBe('$0.0050');
    });

    it('formats normal costs with 2 decimals', () => {
      expect(formatCost(1.23)).toBe('$1.23');
    });
  });

  describe('formatDuration', () => {
    it('returns — for null', () => {
      expect(formatDuration(null)).toBe('—');
    });

    it('formats milliseconds', () => {
      expect(formatDuration(500)).toBe('500ms');
    });

    it('formats seconds', () => {
      expect(formatDuration(2500)).toBe('2.5s');
    });

    it('formats minutes and seconds', () => {
      expect(formatDuration(125_000)).toBe('2m5s');
    });
  });

  describe('formatDate', () => {
    it('returns — for null', () => {
      expect(formatDate(null)).toBe('—');
    });

    it('returns — for empty string', () => {
      expect(formatDate('')).toBe('—');
    });

    it('formats ISO date string', () => {
      const result = formatDate('2026-06-14T12:30:00Z');
      expect(result).toContain('06');
      expect(result).toContain('14');
      expect(result.length).toBeGreaterThan(0);
    });
  });

  describe('formatPercent', () => {
    it('returns — for null', () => {
      expect(formatPercent(null)).toBe('—');
    });

    it('formats percentage with 1 decimal', () => {
      expect(formatPercent(45.6)).toBe('45.6%');
    });
  });

  describe('truncate', () => {
    it('returns original string if within width', () => {
      expect(truncate('hello', 10)).toBe('hello');
    });

    it('truncates and adds ellipsis', () => {
      expect(truncate('hello world', 8)).toBe('hello w…');
    });

    it('handles CJK characters', () => {
      expect(truncate('你好世界测试', 7)).toBe('你好世…');
    });
  });
});
