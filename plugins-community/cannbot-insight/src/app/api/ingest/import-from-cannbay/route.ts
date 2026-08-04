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
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import Database from 'better-sqlite3';
import { execSync } from 'node:child_process';

const CANNBAY_PULL_URL = 'https://gitcode.com/guanxinghua/CANNBay.git';

function runGit(cmd: string, cwd: string) {
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: 120_000 });
}

// 解析 cannbay 提交约定：`提交人:<姓名> 内容描述:<描述>`（冒号兼容半角/全角，
// 描述词兼容"内容描述"/"描述"，提交人可为空）。
// 失败时 submitter 回退 git author，description 回退完整 subject。
function parseCommitSubject(subject: string, author: string): { submitter: string; description: string } {
  let submitter = '';
  let description = '';
  if (subject) {
    const subMatch = subject.match(/提交人\s*[:：]\s*/);
    const descMatch = subject.match(/(?:内容)?描述\s*[:：]\s*/);
    const subStart = subMatch ? (subMatch.index ?? 0) + subMatch[0].length : -1;
    const descIdx = descMatch ? (descMatch.index ?? 0) : -1;
    if (subStart >= 0) {
      // 提交人取到 描述标记 之前；描述标记不存在则取到行尾
      // (>= 而非 >：当提交人为空、描述标记紧随其后时，didx==subStart，应切出空名→回退 author)
      const end = descIdx >= subStart ? descIdx : subject.length;
      submitter = subject.slice(subStart, end).trim();
    }
    if (descIdx >= 0) {
      description = subject.slice(descIdx + (descMatch![0].length)).trim();
    }
  }
  if (!submitter) submitter = author;
  if (!description) description = subject;
  return { submitter, description };
}

export const maxDuration = 300;

// 持久化本地镜像：首次全量 clone，之后每次只 fetch+reset 增量（.db 加一次不改，增量极小）。
// 避免每次 list/import 都全量下载 ~300MB。
const CANNBAY_CACHE_DIR = process.env.CANNBAY_CACHE_DIR
  || path.join(process.cwd(), 'tmp', 'cannbay-cache');

let cloneLock: Promise<string> | null = null;

function ensureCannbayClone(): Promise<string> {
  if (cloneLock) return cloneLock;
  cloneLock = (async () => {
    const cacheDir = CANNBAY_CACHE_DIR;
    fs.mkdirSync(path.dirname(cacheDir), { recursive: true });
    if (fs.existsSync(path.join(cacheDir, '.git'))) {
      try {
        // 增量更新：只拉取 master 新 commit，硬重置工作树
        runGit('git fetch --quiet origin master', cacheDir);
        runGit('git reset --hard --quiet origin/master', cacheDir);
        runGit('git clean -fdq', cacheDir);
        return cacheDir;
      } catch {
        // 镜像损坏或历史被强推 → 删除重建
        fs.rmSync(cacheDir, { recursive: true, force: true });
      }
    }
    fs.rmSync(cacheDir, { recursive: true, force: true });
    runGit(`git clone --quiet "${CANNBAY_PULL_URL}" "${cacheDir}"`, os.tmpdir());
    return cacheDir;
  })().finally(() => { cloneLock = null; });
  return cloneLock;
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action } = body;

    // 持久化镜像（增量更新），不再每次全量 clone 到临时目录
    let cloneDir: string;
    try {
      cloneDir = await ensureCannbayClone();
    } catch (cloneErr) {
      const msg = cloneErr instanceof Error ? cloneErr.message : 'Clone failed';
      return NextResponse.json({ error: `Failed to clone CANNBay: ${msg}` }, { status: 500 });
    }

    try {
      if (action === 'list') {
        // 扫描根目录 .db 文件 + 一层子目录里的 .db 文件（兼容 session 文件夹上传）
        const files: string[] = [];
        const rootEntries = fs.readdirSync(cloneDir, { withFileTypes: true });
        for (const entry of rootEntries) {
          if (entry.isDirectory() && entry.name !== '.git') {
            try {
              const subEntries = fs.readdirSync(path.join(cloneDir, entry.name), { withFileTypes: true });
              for (const sub of subEntries) {
                if (sub.isFile() && sub.name.endsWith('.db')) {
                  files.push(path.join(entry.name, sub.name));
                }
              }
            } catch {}
          } else if (entry.isFile() && entry.name.endsWith('.db')) {
            files.push(entry.name);
          }
        }

        const sessions: Array<{
          filename: string;
          taskId: string;
          query: string | null;
          author: string;
          submitter: string;
          commitTime: string;
          commitMessage: string;
          size: number;
        }> = [];

        for (const f of files) {
          const fullPath = path.join(cloneDir, f);
          const stat = fs.statSync(fullPath);

          const taskIdMatch = path.basename(f).match(/^cannbot_db_session_(.+?)_/);
          const taskId = taskIdMatch ? taskIdMatch[1] : path.basename(f).replace(/\.db$/, '');

          // 每个文件的最后改动 commit：作者 / 时间 / subject（需要完整历史，故非 --depth 1）
          let author = '';
          let commitTime = '';
          let commitSubject = '';
          try {
            const out = runGit(
              `git log -1 --pretty=format:%an%x1f%ad%x1f%s --date=iso-strict -- "${f}"`,
              cloneDir,
            ).toString().trim();
            const parts = out.split('\x1f');
            author = parts[0] ?? '';
            commitTime = parts[1] ?? '';
            commitSubject = parts.slice(2).join('\x1f');
          } catch {}

          const { submitter, description } = parseCommitSubject(commitSubject, author);

          let query: string | null = null;
          try {
            const db = new Database(fullPath, { readonly: true });
            const row = db.prepare(
              'SELECT query FROM "Session" LIMIT 1'
            ).get() as { query: string | null } | undefined;
            if (row) query = row.query;
            db.close();
          } catch {}

          sessions.push({ filename: f, taskId, query, author, submitter, commitTime, commitMessage: description, size: stat.size });
        }

        // 按提交时间降序（最新上传在最前）
        sessions.sort((a, b) => (b.commitTime || '').localeCompare(a.commitTime || ''));

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
            sessionId = row?.taskId ?? path.basename(f).replace(/^cannbot_db_session_/, '').replace(/_\w+\.db$/, '');
            db.close();
          } catch {
            sessionId = path.basename(f).replace(/^cannbot_db_session_/, '').replace(/_\w+\.db$/, '');
          }

          try {
            const result = await importSession(fullPath, sessionId, prisma, fullPath, BRAND_SOURCE_TYPE);
            results.push({ filename: f, taskId: result.sessionId, imported: result.imported, query: result.query ?? null });
          } catch (err) {
            results.push({ filename: f, taskId: sessionId, imported: false, query: null, error: err instanceof Error ? err.message : 'Import failed' });
          }
        }

        return NextResponse.json({ results });
      }

      return NextResponse.json({ error: `Unknown action: "${action}". Supported: list, import` }, { status: 400 });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      return NextResponse.json({ error: message }, { status: 500 });
    }
  } catch (err) {
    const message = err instanceof Error ? err.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
