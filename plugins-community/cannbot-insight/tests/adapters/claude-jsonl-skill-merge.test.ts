// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { readSession } from '../../src/lib/ingest/adapters/claude-jsonl.ts';
import path from 'node:path';

const FIXTURE = path.resolve(__dirname, '../data/claude-sessions/consecutive-skill-injection.jsonl');
const SESSION_ID = 'consecutive-skill-injection';

// Regression: when multiple skill-injection user lines (reclassified to role:'system')
// appear consecutively at the end of a transcript, the consecutive-merge post-process
// splices several elements out of `result` while iterating downward with a for-loop
// whose condition only checks `i >= 0`. After splicing, `i` exceeds the new
// `result.length`, so `result[i]` is undefined and `r.role` throws
// "Cannot read properties of undefined (reading 'role')".
// This manifested on real sessions with many subagents (e.g. f3bb027a-...).
describe('claude-jsonl adapter: consecutive skill-injection merge', () => {
  it('does not throw when consecutive skill injections sit at the end', () => {
    expect(() => readSession(FIXTURE, SESSION_ID)).not.toThrow();
  });

  it('merges consecutive system skill-injection turns into one', () => {
    const rows = readSession(FIXTURE, SESSION_ID);
    const systemTurns = rows.filter(r => r.role === 'system');
    expect(systemTurns.length).toBe(1);
    // Merged content must contain all three skill bodies, joined by the merge separator.
    const merged = systemTurns[0].content ?? '';
    expect(merged).toContain('AAA alpha content');
    expect(merged).toContain('BBB beta content');
    expect(merged).toContain('CCC gamma content');
    expect(merged).toContain('---');
  });

  it('keeps the preceding real user prompt as a separate user turn', () => {
    const rows = readSession(FIXTURE, SESSION_ID);
    const userTurns = rows.filter(r => r.role === 'user');
    expect(userTurns.length).toBe(1);
    expect(userTurns[0].content).toContain('please help me with the ops skill');
  });
});
