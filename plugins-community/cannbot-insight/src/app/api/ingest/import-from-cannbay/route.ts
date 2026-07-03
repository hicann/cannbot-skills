// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { importSession } from '@/lib/ingest/data-service';
import { BRAND_SOURCE_TYPE } from '@/lib/branding';
import { prisma } from '@/lib/db';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import Database from 'better-sqlite3';
import { execSync } from 'node:child_process';

const CANNBAY_PULL_URL = 'https://gitcode.com/guanxinghua/CANNBay.git';

function runGit(cmd: string, cwd: string) {
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: 120_000 });
}

export const maxDuration = 300;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    const randomSuffix = crypto.randomBytes(4).toString('hex');
    const cloneDir = path.join(os.tmpdir(), `cannbay_import_${randomSuffix}`);

    try {
      runGit(`git clone --depth 1 "${CANNBAY_PULL_URL}" "${cloneDir}"`, os.tmpdir());
    } catch (cloneErr) {
      const msg = cloneErr instanceof Error ? cloneErr.message : 'Clone failed';
      return NextResponse.json({ error: `Failed to clone CANNBay: ${msg}` }, { status: 500 });
    }

    try {
      if (action === 'list') {
        const files = fs.readdirSync(cloneDir).filter(f => f.endsWith('.db'));

        const sessions: Array<{
          filename: string;
          taskId: string;
          query: string | null;
          model: string | null;
          startTime: string | null;
          totalTokens: number;
          turnCount: number;
          size: number;
        }> = [];

        for (const f of files) {
          const fullPath = path.join(cloneDir, f);
          const stat = fs.statSync(fullPath);

          const taskIdMatch = f.match(/^cannbot_db_session_(.+?)_/);
          const taskId = taskIdMatch ? taskIdMatch[1] : f.replace(/\.db$/, '');

          let query: string | null = null;
          let model: string | null = null;
          let startTime: string | null = null;
          let totalTokens = 0;
          let turnCount = 0;

          try {
            const db = new Database(fullPath, { readonly: true });
            const row = db.prepare(
              'SELECT query, model, startTime, totalTokens, totalLlmCallCount FROM "Session" LIMIT 1'
            ).get() as { query: string | null; model: string | null; startTime: string; totalTokens: number; totalLlmCallCount: number } | undefined;
            if (row) {
              query = row.query;
              model = row.model;
              startTime = row.startTime;
              totalTokens = row.totalTokens;
              turnCount = row.totalLlmCallCount;
            }
            db.close();
          } catch {}

          sessions.push({ filename: f, taskId, query, model, startTime, totalTokens, turnCount, size: stat.size });
        }

        try { fs.rmSync(cloneDir, { recursive: true, force: true }); } catch {}
        return NextResponse.json({ sessions });
      }

      if (action === 'import') {
        const { filenames } = body as { filenames: string[] };
        if (!filenames || filenames.length === 0) {
          return NextResponse.json({ error: 'Missing filenames' }, { status: 400 });
        }

        const results: Array<{
          filename: string;
          taskId: string;
          imported: boolean;
          query: string | null;
          error?: string;
        }> = [];

        for (const f of filenames) {
          const fullPath = path.join(cloneDir, f);
          if (!fs.existsSync(fullPath)) {
            results.push({ filename: f, taskId: '', imported: false, query: null, error: 'File not found in CANNBay' });
            continue;
          }

          let sessionId = '';
          try {
            const db = new Database(fullPath, { readonly: true });
            const row = db.prepare('SELECT taskId FROM "Session" LIMIT 1').get() as { taskId: string } | undefined;
            sessionId = row?.taskId ?? f.replace(/^cannbot_db_session_/, '').replace(/_\w+\.db$/, '');
            db.close();
          } catch {
            sessionId = f.replace(/^cannbot_db_session_/, '').replace(/_\w+\.db$/, '');
          }

          try {
            const result = await importSession(fullPath, sessionId, prisma, fullPath, BRAND_SOURCE_TYPE);
            results.push({ filename: f, taskId: result.sessionId, imported: result.imported, query: result.query ?? null });
          } catch (err) {
            results.push({ filename: f, taskId: sessionId, imported: false, query: null, error: err instanceof Error ? err.message : 'Import failed' });
          }
        }

        try { fs.rmSync(cloneDir, { recursive: true, force: true }); } catch {}
        return NextResponse.json({ results });
      }

      try { fs.rmSync(cloneDir, { recursive: true, force: true }); } catch {}
      return NextResponse.json({ error: `Unknown action: "${action}". Supported: list, import` }, { status: 400 });
    } catch (err) {
      try { fs.rmSync(cloneDir, { recursive: true, force: true }); } catch {}
      const message = err instanceof Error ? err.message : 'Unknown error';
      return NextResponse.json({ error: message }, { status: 500 });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
