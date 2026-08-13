// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { listSessions } from '../../src/lib/ingest/adapters/cannbot-insight.ts';
import Database from 'better-sqlite3';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

let tmpFile: string;

function buildDb(dbPath: string) {
  const db = new Database(dbPath);
  db.exec(`
    CREATE TABLE "Session" (
      id TEXT PRIMARY KEY,
      taskId TEXT NOT NULL,
      query TEXT,
      model TEXT,
      startTime TEXT NOT NULL,
      endTime TEXT,
      totalLlmCallCount INTEGER NOT NULL DEFAULT 0,
      totalTokens INTEGER NOT NULL DEFAULT 0,
      version TEXT
    );
  `);
  const start = '2026-08-04T10:00:00.000Z';
  const end = '2026-08-04T10:42:00.000Z';
  db.prepare(
    `INSERT INTO "Session" (id, taskId, query, model, startTime, endTime, totalLlmCallCount, totalTokens, version)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run('s1', 'T-001', 'how to fuse matmul', 'qwen3.7-max', start, null, 12, 1234, '1.15.7');
  db.prepare(
    `INSERT INTO "Session" (id, taskId, query, model, startTime, endTime, totalLlmCallCount, totalTokens, version)
     VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`
  ).run('s2', 'T-002', 'debug npu error', 'glm-5.2', end, end, 30, 0, '1.14.30');
  db.close();
}

describe('cannbot-insight adapter listSessions', () => {
  beforeEach(() => {
    tmpFile = path.join(fs.mkdtempSync(path.join(os.tmpdir(), 'cannbot-insight-')), 'sample.db');
    buildDb(tmpFile);
  });

  afterEach(() => {
    fs.rmSync(path.dirname(tmpFile), { recursive: true, force: true });
  });

  it('exposes endedAt and totalTokens from the Session table', () => {
    const sessions = listSessions(tmpFile);
    expect(sessions.length).toBe(2);

    const t2 = sessions.find(s => s.id === 'T-002');
    expect(t2).toBeDefined();
    expect(t2!.createdAt).toBe(new Date('2026-08-04T10:42:00.000Z').toISOString());
    expect(t2!.endedAt).toBe(new Date('2026-08-04T10:42:00.000Z').toISOString());
    expect(t2!.totalTokens).toBe(0);

    const t1 = sessions.find(s => s.id === 'T-001');
    expect(t1).toBeDefined();
    expect(t1!.endedAt).toBeNull();
    expect(t1!.totalTokens).toBe(1234);
    expect(t1!.version).toBe('1.15.7');
  });

  it('returns empty list for nonexistent path', () => {
    expect(listSessions('/nonexistent/path/nope.db')).toEqual([]);
  });
});
