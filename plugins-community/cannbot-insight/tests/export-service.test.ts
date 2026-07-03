// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import Database from 'better-sqlite3';

const EXPORT_SERVICE_PATH = path.resolve(__dirname, '../src/lib/ingest/export-service.ts');

function extractDDL(source: string): string {
  const match = source.match(/const DDL = `([\s\S]*?)`;/);
  if (!match) throw new Error('DDL not found in export-service.ts');
  return match[1];
}

function extractInsertStatements(source: string): Array<{ table: string; columns: string[]; placeholders: number }> {
  const regex = /db\.prepare\(`\s*INSERT INTO "(\w+)" \(\s*([\s\S]*?)\s*\) VALUES \(([\s\S]*?)\)\s*`\)/g;
  const results: Array<{ table: string; columns: string[]; placeholders: number }> = [];
  let match: RegExpExecArray | null;
  while ((match = regex.exec(source)) !== null) {
    const table = match[1];
    const colsRaw = match[2];
    const valsRaw = match[3];
    const columns = colsRaw
      .split(',')
      .map(c => c.trim().replace(/"/g, '').replace(/\n/g, ''))
      .filter(c => c.length > 0);
    const placeholders = valsRaw
      .split(',')
      .map(v => v.trim())
      .filter(v => v === '?')
      .length;
    results.push({ table, columns, placeholders });
  }
  return results;
}

function extractRunArgCounts(source: string): Record<string, number> {
  const result: Record<string, number> = {};
  const regex = /(\w+)\.run\(\s*([\s\S]*?)\s*\);/g;
  let match: RegExpExecArray | null;
  while ((match = regex.exec(source)) !== null) {
    const varName = match[1];
    if (varName.startsWith('insert')) {
      const args = match[2]
        .split(',')
        .map(a => a.trim())
        .filter(a => a.length > 0 && !a.startsWith('//'));
      result[varName] = args.length;
    }
  }
  return result;
}

describe('export-service INSERT column/placeholder parity', () => {
  const source = fs.readFileSync(EXPORT_SERVICE_PATH, 'utf-8');
  const ddl = extractDDL(source);
  const inserts = extractInsertStatements(source);

  it('DDL should be extractable from source', () => {
    expect(ddl.length).toBeGreaterThan(0);
  });

  it('all INSERT prepared statements should have equal column count and placeholder count', () => {
    expect(inserts.length).toBe(8);
    for (const { table, columns, placeholders } of inserts) {
      expect(columns.length, `${table}: column count (${columns.length}) must match ? count (${placeholders})`).toBe(placeholders);
    }
  });

  it('DDL should parse correctly with better-sqlite3 (all tables and indexes created)', () => {
    const db = new Database(':memory:');
    db.exec(ddl);

    const tables = db.prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").all().map(r => (r as any).name);
    const expectedTables = ['Execution', 'ExecutionSkill', 'InteractionBridge', 'Session', 'SessionSkill', 'SkillEvent', 'ToolCall', 'Turn'];
    expect(tables).toEqual(expectedTables);

    const indexes = db.prepare("SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%' ORDER BY name").all().map(r => (r as any).name);
    expect(indexes.length).toBeGreaterThan(0);

    db.close();
  });

  it('each INSERT prepared statement should bind without column mismatch error', () => {
    const db = new Database(':memory:');
    db.exec(ddl);

    for (const { table, columns, placeholders } of inserts) {
      const placeholdersStr = Array.from({ length: placeholders }, () => '?').join(', ');
      const colsStr = columns.map(c => `"${c}"`).join(', ');
      const sql = `INSERT INTO "${table}" (${colsStr}) VALUES (${placeholdersStr})`;
      const stmt = db.prepare(sql);
      expect(stmt, `${table}: prepared statement should compile without error`).toBeDefined();
    }

    db.close();
  });
});

describe('export-service run() arg count matches VALUES placeholder count', () => {
  const source = fs.readFileSync(EXPORT_SERVICE_PATH, 'utf-8');
  const inserts = extractInsertStatements(source);
  const runArgs = extractRunArgCounts(source);

  it('insertSession.run() args should match VALUES placeholder count', () => {
    const sessionInsert = inserts.find(i => i.table === 'Session');
    expect(sessionInsert).toBeDefined();
    expect(runArgs['insertSession'], `insertSession.run() has ${runArgs['insertSession']} args, VALUES has ${sessionInsert!.placeholders} ?`).toBe(sessionInsert!.placeholders);
  });

  it('insertTurn.run() args should match VALUES placeholder count', () => {
    const turnInsert = inserts.find(i => i.table === 'Turn');
    expect(turnInsert).toBeDefined();
    expect(runArgs['insertTurn']).toBe(turnInsert!.placeholders);
  });

  it('insertToolCall.run() args should match VALUES placeholder count', () => {
    const tcInsert = inserts.find(i => i.table === 'ToolCall');
    expect(tcInsert).toBeDefined();
    expect(runArgs['insertToolCall']).toBe(tcInsert!.placeholders);
  });

  it('insertExecution.run() args should match VALUES placeholder count', () => {
    const exInsert = inserts.find(i => i.table === 'Execution');
    expect(exInsert).toBeDefined();
    expect(runArgs['insertExecution']).toBe(exInsert!.placeholders);
  });

  it('insertBridge.run() args should match VALUES placeholder count', () => {
    const brInsert = inserts.find(i => i.table === 'InteractionBridge');
    expect(brInsert).toBeDefined();
    expect(runArgs['insertBridge']).toBe(brInsert!.placeholders);
  });
});

describe('regression guard: Session 30-vs-31 bug', () => {
  const source = fs.readFileSync(EXPORT_SERVICE_PATH, 'utf-8');
  const inserts = extractInsertStatements(source);
  const runArgs = extractRunArgCounts(source);

  it('Session INSERT must have 31 columns, 31 placeholders, and 31 run() args', () => {
    const sessionInsert = inserts.find(i => i.table === 'Session');
    expect(sessionInsert).toBeDefined();
    expect(sessionInsert!.columns.length).toBe(31);
    expect(sessionInsert!.placeholders).toBe(31);
    expect(runArgs['insertSession']).toBe(31);
  });
});
