// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { normalize } from '../src/lib/ingest/normalize.ts';
import type { RawInteraction } from '../src/lib/shared/types.ts';
import { getAdapter } from '../src/lib/ingest/adapters/index.ts';
import fs from 'node:fs';
import path from 'node:path';

const REAL_DB_PATH = path.resolve(__dirname, 'data/opencode-sessions.db');
const SYNTHETIC_PATH = path.resolve(__dirname, 'data/synthetic-opencode.json');
const hasRealDB = fs.existsSync(REAL_DB_PATH);

function loadSynthetic(): RawInteraction[] {
  return JSON.parse(fs.readFileSync(SYNTHETIC_PATH, 'utf-8')) as RawInteraction[];
}

describe('normalize', () => {
  describe('opencode-db source', () => {
    it('normalizes synthetic data: field names mapped, missing fields defaulted', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      expect(result.length).toBe(synthetic.length);
      for (const item of result) {
        expect(typeof item.role).toBe('string');
        expect(typeof item.timestamp).toBe('string');
        expect(item.timeInfo).not.toBeNull();
        expect(item.timeInfo!.created).toBeTypeOf('number');
      }
    });

    it('defaults role to "unknown" when null or empty', () => {
      const raw: RawInteraction[] = [
        {
          role: '',
          content: null,
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1 },
          agent: null,
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: null,
          usage: null,
          model: null,
          modelID: null,
          providerID: null,
          latency: null,
          finish_reason: null,
        },
      ];
      const result = normalize(raw, 'opencode-db');
      expect(result[0].role).toBe('unknown');
    });

    it('validates timestamps as ISO format', () => {
      const raw: RawInteraction[] = [
        {
          role: 'user',
          content: 'test',
          timestamp: 'not-a-date',
          timeInfo: { created: 1000 },
          agent: null,
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: null,
          usage: null,
          model: null,
          modelID: null,
          providerID: null,
          latency: null,
          finish_reason: null,
        },
      ];
      const result = normalize(raw, 'opencode-db');
      expect(result[0].timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    });

    it('preserves valid ISO timestamps', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      for (const item of result) {
        expect(item.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
      }
    });

    it('preserves content as-is', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      const userMsg = result.find(i => i.role === 'user');
      expect(userMsg).toBeDefined();
      expect(userMsg!.content).toBe(synthetic.find(i => i.role === 'user')!.content);
    });

    it('preserves tool_calls as-is', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      const withTools = result.find(i => i.tool_calls && i.tool_calls!.length > 0);
      expect(withTools).toBeDefined();
      const orig = synthetic.find(i => i.tool_calls && i.tool_calls!.length > 0);
      expect(withTools!.tool_calls!.length).toBe(orig!.tool_calls!.length);
    });

    it('normalizes token usage fields correctly', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      const withUsage = result.find(i => i.usage !== null);
      expect(withUsage).toBeDefined();
      const u = withUsage!.usage!;
      expect(typeof u.total).toBe('number');
      expect(typeof u.input).toBe('number');
      expect(typeof u.output).toBe('number');
      expect(typeof u.reasoning).toBe('number');
      expect(typeof u.cacheRead).toBe('number');
      expect(typeof u.cacheWrite).toBe('number');
      expect(typeof u.cost).toBe('number');
    });

    it('defaults missing usage fields to 0', () => {
      const raw: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'test',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: { created: 1 },
          agent: null,
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: null,
          usage: { total: 100, input: 50, output: 50 } as any,
          model: null,
          modelID: null,
          providerID: null,
          latency: null,
          finish_reason: null,
        },
      ];
      const result = normalize(raw, 'opencode-db');
      expect(result[0].usage!.total).toBe(100);
      expect(result[0].usage!.input).toBe(50);
      expect(result[0].usage!.output).toBe(50);
      expect(result[0].usage!.reasoning).toBe(0);
      expect(result[0].usage!.cacheRead).toBe(0);
      expect(result[0].usage!.cacheWrite).toBe(0);
      expect(result[0].usage!.cost).toBe(0);
    });

    it('preserves subagent fields', () => {
      const synthetic = loadSynthetic();
      const result = normalize(synthetic, 'opencode-db');
      const subagent = result.find(i => i.subagent_name !== null);
      expect(subagent).toBeDefined();
      expect(subagent!.subagent_name).toBe('Kuafu');
      expect(subagent!.subagent_session_id).toBe('ses_synthetic_subagent_001');
    });

    it('returns empty result for empty input', () => {
      const result = normalize([], 'opencode-db');
      expect(result).toEqual([]);
    });

    it('defaults null timeInfo to { created: 0 }', () => {
      const raw: RawInteraction[] = [
        {
          role: 'user',
          content: 'test',
          timestamp: '2026-01-01T00:00:00.000Z',
          timeInfo: null,
          agent: null,
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: null,
          usage: null,
          model: null,
          modelID: null,
          providerID: null,
          latency: null,
          finish_reason: null,
        },
      ];
      const result = normalize(raw, 'opencode-db');
      expect(result[0].timeInfo).toEqual({ created: 0 });
    });

    it.skipIf(!hasRealDB)('works with real DB data via opencode-db adapter', () => {
      const adapter = getAdapter('opencode-db')!;
      const sessions = adapter.listSessions(REAL_DB_PATH);
      const interactions = adapter.readSession(REAL_DB_PATH, sessions[0].id);
      const result = normalize(interactions, 'opencode-db');
      expect(result.length).toBe(interactions.length);
      for (const item of result) {
        expect(typeof item.role).toBe('string');
        expect(item.timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T/);
      }
    });
  });

  describe('claude-jsonl source', () => {
    it('normalizes empty input without error', () => {
      const result = normalize([], 'claude-jsonl');
      expect(result).toEqual([]);
    });

    it('normalizes claude-jsonl interactions with correct defaults', () => {
      const raw: RawInteraction[] = [
        {
          role: 'assistant',
          content: 'test',
          timestamp: '2025-01-01T00:00:00.000Z',
          timeInfo: { created: 1704067200000 },
          agent: null,
          subagent_name: null,
          subagent_session_id: null,
          tool_calls: null,
          usage: {
            total: 300,
            input: 150,
            output: 150,
            reasoning: 0,
            cacheRead: 50,
            cacheWrite: 30,
            cost: 0,
          },
          model: 'claude-sonnet-4-20250514',
          modelID: null,
          providerID: null,
          latency: 5000,
          finish_reason: 'success',
        },
      ];
      const result = normalize(raw, 'claude-jsonl');
      expect(result.length).toBe(1);
      expect(result[0].role).toBe('assistant');
      expect(result[0].content).toBe('test');
      expect(result[0].model).toBe('claude-sonnet-4-20250514');
      expect(result[0].usage?.total).toBe(300);
      expect(result[0].usage?.reasoning).toBe(0);
      expect(result[0].usage?.cacheRead).toBe(50);
    });
  });

  describe('unknown source', () => {
    it('throws descriptive error', () => {
      expect(() => normalize([], 'unknown')).toThrow('Unknown source type');
    });
  });
});
