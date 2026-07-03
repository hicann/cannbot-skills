// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { getAdapter } from '../../src/lib/ingest/adapters/index.ts';
import type { Adapter } from '../../src/lib/ingest/adapters/index.ts';
import path from 'node:path';
import fs from 'node:fs';

const REAL_DB_PATH = path.resolve(__dirname, '../data/opencode-sessions.db');
const CLAUDE_DIR = path.resolve(__dirname, '../data/claude-sessions');
const CLAUDE_FILE = path.join(CLAUDE_DIR, 'abc123.jsonl');
const hasRealDB = fs.existsSync(REAL_DB_PATH);

describe('adapter registry', () => {
  it('getAdapter("opencode-db") returns adapter with listSessions and readSession', () => {
    const adapter = getAdapter('opencode-db') as Adapter;
    expect(adapter).not.toBeNull();
    expect(typeof adapter.listSessions).toBe('function');
    expect(typeof adapter.readSession).toBe('function');
  });

  it('getAdapter("claude-jsonl") returns adapter with listSessions and readSession', () => {
    const adapter = getAdapter('claude-jsonl') as Adapter;
    expect(adapter).not.toBeNull();
    expect(typeof adapter.listSessions).toBe('function');
    expect(typeof adapter.readSession).toBe('function');
  });

  it('getAdapter("unknown") throws descriptive error', () => {
    expect(() => getAdapter('unknown')).toThrow('Unknown source type');
  });

  it.skipIf(!hasRealDB)('listSessions works through registry with real DB', () => {
    const adapter = getAdapter('opencode-db') as Adapter;
    const sessions = adapter.listSessions(REAL_DB_PATH);
    expect(sessions.length).toBeGreaterThan(0);
    expect(sessions[0]).toHaveProperty('id');
    expect(sessions[0]).toHaveProperty('createdAt');
  });

  it.skipIf(!hasRealDB)('readSession works through registry with real DB', () => {
    const adapter = getAdapter('opencode-db') as Adapter;
    const sessions = adapter.listSessions(REAL_DB_PATH);
    const interactions = adapter.readSession(REAL_DB_PATH, sessions[0].id);
    expect(interactions.length).toBeGreaterThan(0);
    expect(interactions[0]).toHaveProperty('role');
    expect(interactions[0]).toHaveProperty('timestamp');
  });

  it('listSessions works through registry with claude-jsonl', () => {
    const adapter = getAdapter('claude-jsonl') as Adapter;
    const sessions = adapter.listSessions(CLAUDE_DIR);
    expect(sessions.length).toBeGreaterThan(0);
    expect(sessions[0]).toHaveProperty('id');
    expect(sessions[0]).toHaveProperty('createdAt');
  });

  it('readSession works through registry with claude-jsonl', () => {
    const adapter = getAdapter('claude-jsonl') as Adapter;
    const interactions = adapter.readSession(CLAUDE_FILE, 'abc123');
    expect(interactions.length).toBeGreaterThan(0);
    expect(interactions[0]).toHaveProperty('role');
    expect(interactions[0]).toHaveProperty('timestamp');
  });
});
