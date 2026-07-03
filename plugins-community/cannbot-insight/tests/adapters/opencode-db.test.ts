// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { listSessions, readSession } from '../../src/lib/ingest/adapters/opencode-db.ts';
import path from 'node:path';
import fs from 'node:fs';

const REAL_DB_PATH = path.resolve(__dirname, '../data/opencode-sessions.db');
const MISSING_DB_PATH = path.resolve(__dirname, '../data/nonexistent.db');
const hasRealDB = fs.existsSync(REAL_DB_PATH);

describe.skipIf(!hasRealDB)('opencode-db adapter (real DB)', () => {
  describe('listSessions', () => {
    it('returns session list from real DB', () => {
      const sessions = listSessions(REAL_DB_PATH);
      expect(sessions.length).toBeGreaterThan(0);
      expect(sessions.length).toBeLessThanOrEqual(39);

      const first = sessions[0];
      expect(first).toHaveProperty('id');
      expect(first).toHaveProperty('createdAt');
      expect(first).toHaveProperty('firstQuery');
      expect(first).toHaveProperty('turnCount');
      expect(first).toHaveProperty('modelName');

      expect(typeof first.id).toBe('string');
      expect(typeof first.createdAt).toBe('string');
      expect(typeof first.turnCount).toBe('number');
      expect(first.createdAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);

      console.log('\n=== listSessions() sample output (first 5) ===');
      for (const s of sessions.slice(0, 5)) {
        console.log(JSON.stringify(s, null, 2));
      }
      console.log(`Total sessions returned: ${sessions.length}`);
    });

    it('returns root sessions only (not subagent sessions)', () => {
      const sessions = listSessions(REAL_DB_PATH);
      for (const s of sessions) {
        expect(s.id).not.toContain('subagent');
      }
    });

    it('handles missing DB gracefully', () => {
      const sessions = listSessions(MISSING_DB_PATH);
      expect(sessions).toEqual([]);
    });

    it('handles empty path gracefully', () => {
      const sessions = listSessions('');
      expect(sessions).toEqual([]);
    });
  });

  describe('readSession', () => {
    it('returns RawInteraction[] for a real session', () => {
      const sessions = listSessions(REAL_DB_PATH);
      const sessionId = sessions[0].id;
      const interactions = readSession(REAL_DB_PATH, sessionId);

      expect(interactions.length).toBeGreaterThan(0);
      expect(interactions[0]).toHaveProperty('role');
      expect(interactions[0]).toHaveProperty('content');
      expect(interactions[0]).toHaveProperty('timestamp');
      expect(interactions[0]).toHaveProperty('timeInfo');
      expect(interactions[0]).toHaveProperty('agent');
      expect(interactions[0]).toHaveProperty('subagent_name');
      expect(interactions[0]).toHaveProperty('subagent_session_id');
      expect(interactions[0]).toHaveProperty('tool_calls');
      expect(interactions[0]).toHaveProperty('usage');
      expect(interactions[0]).toHaveProperty('model');
      expect(interactions[0]).toHaveProperty('modelID');
      expect(interactions[0]).toHaveProperty('providerID');
      expect(interactions[0]).toHaveProperty('latency');
      expect(interactions[0]).toHaveProperty('finish_reason');

      console.log('\n=== readSession() sample output (first 3 interactions) ===');
      for (const i of interactions.slice(0, 3)) {
        console.log(JSON.stringify(i, null, 2));
      }
      console.log(`Total interactions: ${interactions.length}`);
    });

    it('returns correct roles from DB', () => {
      const sessions = listSessions(REAL_DB_PATH);
      const sessionId = sessions.find(s => s.turnCount > 5)?.id || sessions[0].id;
      const interactions = readSession(REAL_DB_PATH, sessionId);

      const roles = interactions.map(i => i.role);
      expect(roles).toContain('user');
      expect(roles).toContain('assistant');
    });

    it('handles non-existent sessionId gracefully', () => {
      const interactions = readSession(REAL_DB_PATH, 'fake_session_id_12345');
      expect(interactions).toEqual([]);
    });

    it('handles missing DB gracefully', () => {
      const interactions = readSession(MISSING_DB_PATH, 'any_id');
      expect(interactions).toEqual([]);
    });

    it('extracts tool_calls from assistant messages', () => {
      const sessions = listSessions(REAL_DB_PATH);
      const sessionId = sessions.find(s => s.turnCount > 10)?.id || sessions[0].id;
      const interactions = readSession(REAL_DB_PATH, sessionId);

      const assistantWithTools = interactions.find(
        i => i.role === 'assistant' && i.tool_calls && i.tool_calls!.length > 0
      );
      if (assistantWithTools) {
        expect(assistantWithTools.tool_calls![0]).toHaveProperty('toolCallId');
        expect(assistantWithTools.tool_calls![0]).toHaveProperty('toolName');
        expect(assistantWithTools.tool_calls![0]).toHaveProperty('state');
      }
    });

    it('extracts token usage from assistant messages', () => {
      const sessions = listSessions(REAL_DB_PATH);
      const sessionId = sessions.find(s => s.turnCount > 5)?.id || sessions[0].id;
      const interactions = readSession(REAL_DB_PATH, sessionId);

      const assistantWithUsage = interactions.find(
        i => i.role === 'assistant' && i.usage
      );
      if (assistantWithUsage) {
        expect(assistantWithUsage.usage!.input).toBeGreaterThan(0);
        expect(typeof assistantWithUsage.usage!.cost).toBe('number');
      }
    });

    it('returns subagent info for subagent sessions', () => {
      const subagentSessionId = 'ses_203f114dfffeXGLOGG5RKmEDJ3';
      const interactions = readSession(REAL_DB_PATH, subagentSessionId);

      if (interactions.length > 0) {
        const nonUser = interactions.find(i => i.role !== 'user');
        if (nonUser) {
          expect(nonUser.subagent_name).not.toBeNull();
          expect(nonUser.subagent_session_id).toBe(subagentSessionId);
        }
      }

      console.log('\n=== Subagent session sample ===');
      for (const i of interactions.slice(0, 2)) {
        console.log(JSON.stringify(i, null, 2));
      }
    });

    it('has null subagent info for root sessions', () => {
      const sessions = listSessions(REAL_DB_PATH);
      const sessionId = sessions[0].id;
      const interactions = readSession(REAL_DB_PATH, sessionId);

      for (const i of interactions) {
        expect(i.subagent_name).toBeNull();
        expect(i.subagent_session_id).toBeNull();
      }
    });
  });
});
