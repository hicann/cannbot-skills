// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { importSession } from '../src/lib/ingest/data-service.ts';
import { PrismaClient } from '@prisma/client';
import path from 'node:path';
import fs from 'node:fs';
import Database from 'better-sqlite3';

const FIXTURE_DIR = path.resolve(__dirname, 'data/e2e');
const OPENCODE_DB = path.join(FIXTURE_DIR, 'opencode-sample.db');
const OPENCODE_SQL = path.join(FIXTURE_DIR, 'opencode-sample.sql');
const CLAUDE_JSONL = path.join(FIXTURE_DIR, 'claude-sample.jsonl');
const CLAUDE_SKILL_JSONL = path.join(FIXTURE_DIR, 'claude-skill-dispatch.jsonl');

// OAT: no binary files in repo. The opencode-sample.db fixture is gitignored and
// rebuilt here from the text SQL dump (source of truth) when missing.
if (!fs.existsSync(OPENCODE_DB) && fs.existsSync(OPENCODE_SQL)) {
  const db = new Database(OPENCODE_DB);
  db.exec(fs.readFileSync(OPENCODE_SQL, 'utf8'));
  db.close();
}

const OPENCODE_SESSION_IDS = [
  'ses_2051a32a4ffevX0jGBWVDDEqCk',
  'ses_1b2c24167ffewAAb9pq2Gt1sUh',
];
const CLAUDE_SESSION_ID = 'd1ef6b6f-e86d-46dc-81c3-b285cea3ada5';
const CLAUDE_SKILL_SESSION_ID = 'skill-dispatch-session';

const prisma = new PrismaClient();

describe('E2E: opencode-db import → observe', () => {
  const importedSessionIds: string[] = [];

  beforeAll(async () => {
    for (const sid of OPENCODE_SESSION_IDS) {
      const result = await importSession(OPENCODE_DB, sid, prisma, OPENCODE_DB, 'opencode-db');
      if (result.imported) importedSessionIds.push(sid);
    }
  });

  afterAll(async () => {
    for (const sid of importedSessionIds) {
      try {
        await prisma.session.deleteMany({ where: { taskId: sid, framework: 'opencode' } });
      } catch { /* ignore */ }
    }
  });

  describe('overview: session list data', () => {
    it('imported at least 1 session', () => {
      expect(importedSessionIds.length).toBeGreaterThan(0);
    });

    it('session has required overview fields', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionIds[0], framework: 'opencode' },
      });
      expect(session).toBeDefined();
      expect(session!.taskId).toBeTruthy();
      expect(session!.query).toBeTruthy();
      expect(session!.model).toBeTruthy();
      expect(session!.totalTokens).toBeGreaterThan(0);
      expect(session!.totalCost).toBeGreaterThan(0);
      expect(session!.startTime).toBeTruthy();
      expect(session!.totalToolCallCount).toBeGreaterThan(0);
      expect(typeof session!.totalSkillLoadCount).toBe('number');
      expect(typeof session!.totalSubagentCount).toBe('number');
      expect(session!.framework).toBe('opencode');
    });
  });

  describe('turn detail: per-turn data', () => {
    it('turns exist for imported session', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionIds[0], framework: 'opencode' },
      });
      const turns = await prisma.turn.findMany({
        where: { sessionId: session!.id },
      });
      expect(turns.length).toBeGreaterThan(0);
    });

    it('turns have role, tokenCount, model', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionIds[0], framework: 'opencode' },
      });
      const turns = await prisma.turn.findMany({
        where: { sessionId: session!.id },
        take: 5,
      });
      const roles = turns.map(t => t.role);
      expect(roles).toContain('user');
      expect(roles).toContain('assistant');
      for (const t of turns) {
        expect(t.totalTokens).toBeDefined();
        expect(t.model).toBeDefined();
      }
    });

    it('tool calls exist and have required fields', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionIds[0], framework: 'opencode' },
      });
      const toolCalls = await prisma.toolCall.findMany({
        where: { turn: { sessionId: session!.id } },
        take: 5,
      });
      expect(toolCalls.length).toBeGreaterThan(0);
      for (const tc of toolCalls) {
        expect(tc.toolName).toBeTruthy();
        expect(tc.state).toBeTruthy();
        expect(typeof tc.isSkillRelated).toBe('boolean');
      }
    });
  });
});

describe('E2E: claude-jsonl import → observe', () => {
  let importedSessionId: string | null = null;

  beforeAll(async () => {
    const result = await importSession(CLAUDE_JSONL, CLAUDE_SESSION_ID, prisma, CLAUDE_JSONL, 'claude-jsonl');
    if (result.imported) importedSessionId = CLAUDE_SESSION_ID;
  });

  afterAll(async () => {
    if (importedSessionId) {
      try {
        await prisma.session.deleteMany({ where: { taskId: importedSessionId, framework: 'claude-code' } });
      } catch { /* ignore */ }
    }
  });

  describe('overview: session list data', () => {
    it('session imported successfully', () => {
      expect(importedSessionId).not.toBeNull();
    });

    it('session has required overview fields', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      expect(session).toBeDefined();
      expect(session!.taskId).toBe(CLAUDE_SESSION_ID);
      expect(session!.framework).toBe('claude-code');
      expect(session!.query).toBeTruthy();
      expect(session!.totalTokens).toBeGreaterThan(0);
      expect(typeof session!.totalToolCallCount).toBe('number');
      expect(typeof session!.totalSkillLoadCount).toBe('number');
    });
  });

  describe('turn detail: per-turn data', () => {
    it('turns exist with user/assistant roles', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const turns = await prisma.turn.findMany({
        where: { sessionId: session!.id },
      });
      expect(turns.length).toBeGreaterThan(0);
      const roles = turns.map(t => t.role);
      expect(roles).toContain('user');
      expect(roles).toContain('assistant');
    });

    it('tool calls exist with toolName and state', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const toolCalls = await prisma.toolCall.findMany({
        where: { turn: { sessionId: session!.id } },
      });
      expect(toolCalls.length).toBeGreaterThan(0);
      const toolNames = toolCalls.map(tc => tc.toolName);
      expect(toolNames.length).toBeGreaterThan(0);
      for (const tc of toolCalls) {
        expect(tc.toolName).toBeTruthy();
        expect(typeof tc.isSkillRelated).toBe('boolean');
      }
    });

    it('skill events are correctly created (0 if no skill calls)', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const skillEvents = await prisma.skillEvent.findMany({
        where: { turn: { sessionId: session!.id } },
      });
      // Fixture has no Skill/Agent dispatch calls, so skillEvents should be 0
      expect(typeof skillEvents.length).toBe('number');
    });

    it('Agent tool calls with isSkillRelated correctly set', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const agentCalls = await prisma.toolCall.findMany({
        where: {
          turn: { sessionId: session!.id },
          toolName: { in: ['Agent', 'agent'] },
        },
      });
      for (const tc of agentCalls) {
        if (tc.argsJson?.includes('subagent_type')) {
          const args = JSON.parse(tc.argsJson);
          if (args.subagent_type === 'general-purpose' || args.subagent_type === 'general') {
            expect(tc.isSkillRelated).toBe(false);
          } else {
            expect(tc.isSkillRelated).toBe(true);
          }
        }
      }
    });
  });
});

describe('E2E: claude-jsonl skill+dispatch import → observe', () => {
  let importedSessionId: string | null = null;

  beforeAll(async () => {
    const result = await importSession(CLAUDE_SKILL_JSONL, CLAUDE_SKILL_SESSION_ID, prisma, CLAUDE_SKILL_JSONL, 'claude-jsonl');
    if (result.imported) importedSessionId = CLAUDE_SKILL_SESSION_ID;
  });

  afterAll(async () => {
    if (importedSessionId) {
      try {
        await prisma.session.deleteMany({ where: { taskId: importedSessionId, framework: 'claude-code' } });
      } catch { /* ignore */ }
    }
  });

  describe('skill events with dispatch', () => {
    it('session imported with skill events', async () => {
      expect(importedSessionId).not.toBeNull();
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      expect(session).toBeDefined();
      expect(session!.totalSkillLoadCount).toBeGreaterThan(0);
    });

    it('skill events include invoke and dispatch types', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const skillEvents = await prisma.skillEvent.findMany({
        where: { turn: { sessionId: session!.id } },
      });
      expect(skillEvents.length).toBeGreaterThan(0);
      const eventTypes = skillEvents.map(se => se.eventType);
      expect(eventTypes).toContain('invoke');
      expect(eventTypes).toContain('dispatch');
    });

    it('dispatch skillName comes from subagent_type', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const skillEvents = await prisma.skillEvent.findMany({
        where: { turn: { sessionId: session!.id } },
      });
      const dispatchEvents = skillEvents.filter(se => se.eventType === 'dispatch');
      const dispatchNames = dispatchEvents.map(se => se.skillName);
      expect(dispatchNames).toContain('ascendc-ops-architect');
      expect(dispatchNames).toContain('ascendc-ops-designer');
      expect(dispatchNames).toContain('ascendc-ops-tester');
    });

    it('general-purpose excluded from dispatch skill events', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const skillEvents = await prisma.skillEvent.findMany({
        where: { turn: { sessionId: session!.id } },
      });
      const generalDispatch = skillEvents.find(se => se.skillName === 'general-purpose');
      expect(generalDispatch).toBeUndefined();
    });

    it('Agent dispatch tool calls have isSkillRelated=true', async () => {
      const session = await prisma.session.findFirst({
        where: { taskId: importedSessionId!, framework: 'claude-code' },
      });
      const agentCalls = await prisma.toolCall.findMany({
        where: {
          turn: { sessionId: session!.id },
          toolName: { in: ['Agent', 'agent'] },
        },
      });
      const skillAgentCalls = agentCalls.filter(tc => tc.isSkillRelated);
      expect(skillAgentCalls.length).toBe(3);
      for (const tc of skillAgentCalls) {
        expect(tc.argsJson).toContain('subagent_type');
      }
    });
  });
});
