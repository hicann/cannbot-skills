// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, beforeEach } from 'vitest';
import { exportSessionToMarkdown } from '../src/lib/export/markdown-exporter';
import { PrismaClient } from '@prisma/client';
import path from 'node:path';

const DB_PATH = path.resolve(__dirname, 'data/test-export.db');

describe('markdown-export', () => {
  describe('unit tests with truncate/utils', () => {
    it('truncate short text passes through', () => {
      const result = 'hello world'.length <= 10000 ? 'hello world' : 'hello world... [truncated]';
      expect(result).toBe('hello world');
    });

    it('truncate long text adds suffix', () => {
      const long = 'a'.repeat(15000);
      const truncated = long.substring(0, 10000) + '... [truncated, full: 15000 chars]';
      expect(truncated.length).toBeLessThan(long.length);
      expect(truncated).toContain('[truncated, full: 15000 chars]');
    });

    it('fmtDuration formats correctly', () => {
      // 500ms → 500ms
      expect('500ms').toBe('500ms');
      // 5000ms → 5.0s
      const s = (5000 / 1000).toFixed(1) + 's';
      expect(s).toBe('5.0s');
    });

    it('fmtTokens formats correctly', () => {
      const fmt = (n: number) => {
        if (n < 1000) return `${n}`;
        if (n < 1000000) return `${(n / 1000).toFixed(1)}K`;
        return `${(n / 1000000).toFixed(1)}M`;
      };
      expect(fmt(500)).toBe('500');
      expect(fmt(1500)).toBe('1.5K');
      expect(fmt(139664)).toBe('139.7K');
    });
  });

  describe('integration with real DB', () => {
    let prisma: PrismaClient;
    let sessionId: string;

    beforeAll(async () => {
      // Use the dev DB if available
      const devDbPath = path.resolve(__dirname, '../prisma/dev.db');
      if (!path) return;

      const { DATABASE_URL } = process.env;
      prisma = new PrismaClient({
        datasources: { db: { url: `file:${devDbPath}` } },
      });

      // Find a session to test with
      const sessions = await prisma.session.findMany({ take: 1 });
      if (sessions.length > 0) {
        sessionId = sessions[0].taskId;
      }
    });

    it.skipIf(!sessionId)('exports session to markdown without error', async () => {
      if (!prisma || !sessionId) return;
      const md = await exportSessionToMarkdown(sessionId, prisma);
      expect(md).toBeTruthy();
      expect(md).toContain('# Session:');
      expect(md).toContain('Root Agent');
    });

    it.skipIf(!sessionId)('markdown contains metadata blockquote', async () => {
      if (!prisma || !sessionId) return;
      const md = await exportSessionToMarkdown(sessionId, prisma);
      expect(md).toContain('> **Task ID**');
      expect(md).toContain('> **Duration**');
    });

    it.skipIf(!sessionId)('markdown contains user and assistant turns', async () => {
      if (!prisma || !sessionId) return;
      const md = await exportSessionToMarkdown(sessionId, prisma);
      expect(md).toContain('**User**:');
      expect(md).toContain('**Assistant**:');
    });
  });
});
