// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { NextRequest, NextResponse } from 'next/server';
import { exportSession } from '@/lib/ingest/export-service';
import { prisma } from '@/lib/db';
import crypto from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { execSync } from 'node:child_process';
import { planArchive, MAX_MASTER_SESSIONS, type FileWithDate } from '@/lib/cannbay-archive';

const CANNBAY_PULL_URL = 'https://gitcode.com/guanxinghua/CANNBay.git';
const CANNBAY_PUSH_URL = Buffer.from('aHR0cHM6Ly9ndWFueGluZ2h1YTpwc3F5WXAyYnpFRkI0eDVQRlVTV0dMS3lAZ2l0Y29kZS5jb20vZ3VhbnhpbmdodWEvQ0FOTkJheS5naXQ=', 'base64').toString();

function runGit(cmd: string, cwd: string) {
  // First-ever clone of ~1.6GB needs many minutes; subsequent uploads only
  // fetch the delta (seconds). 60s is far too short for the first clone.
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: 1_800_000 });
}

// Commit date (ms) of the commit that first added `filename` to master.
// Falls back to -1 (treated as oldest) when history can't resolve it.
function fileAddedAtMs(filename: string, cwd: string): number {
  try {
    const out = runGit(`git log -1 --diff-filter=A --format=%aI -- "${filename}"`, cwd).toString().trim();
    const ms = out ? Date.parse(out) : NaN;
    return Number.isFinite(ms) ? ms : -1;
  } catch {
    return -1;
  }
}

// List session .db files currently on master with their add-commit dates.
function listMasterSessions(cwd: string): FileWithDate[] {
  return fs.readdirSync(cwd)
    .filter((f) => f.endsWith('.db'))
    .map((f) => ({ filename: f, addedAtMs: fileAddedAtMs(f, cwd) }));
}

// Persistent local mirror of CANNBay: full-clone ONCE, then each upload only
// `git fetch`es the delta (one new commit + ~one .db blob). Re-cloning ~1.6GB
// of history every upload was the real perf bottleneck — the 20-cap shrinks
// the working tree, but archived files remain in history, so a fresh clone
// still pulls everything. `origin` is the public pull URL (safe to persist);
// pushes use the explicit credentialed push URL so creds never land in config.
const UPLOAD_CACHE_DIR = process.env.CANNBAY_UPLOAD_CACHE_DIR
  || path.join(process.cwd(), 'tmp', 'cannbay-upload-cache');

let uploadMirrorLock: Promise<string> | null = null;
function ensureUploadMirror(): Promise<string> {
  if (uploadMirrorLock) return uploadMirrorLock;
  uploadMirrorLock = (async () => {
    const dir = UPLOAD_CACHE_DIR;
    fs.mkdirSync(path.dirname(dir), { recursive: true });
    if (fs.existsSync(path.join(dir, '.git'))) {
      try {
        runGit('git fetch --quiet origin master', dir);
        return dir;
      } catch {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    }
    fs.rmSync(dir, { recursive: true, force: true });
    runGit(`git clone --quiet "${CANNBAY_PULL_URL}" "${dir}"`, os.tmpdir());
    return dir;
  })().finally(() => { uploadMirrorLock = null; });
  return uploadMirrorLock;
}

// When master exceeds the cap, move the oldest .db files off it onto a
// date-bucketed archive branch. Archive is append-only; master stays a rolling
// window of the newest N. Pushes use the explicit credentialed `pushUrl`.
//
// Sequence: stash to-archive files in a sibling temp dir → create/advance the
// archive branch (fetch remote if it exists) → drop the files in + commit +
// push → return to master, `git rm` the rotated files + commit. Push of
// master happens after, in the caller.
export function rotateArchiveIfNeeded(cloneDir: string, runMs: number, pushUrl: string): string | null {
  const plan = planArchive(listMasterSessions(cloneDir), MAX_MASTER_SESSIONS, runMs);
  if (plan.toArchive.length === 0) return null;

  const staging = path.join(os.tmpdir(), `cannbay_arch_${crypto.randomBytes(4).toString('hex')}`);
  fs.mkdirSync(staging, { recursive: true });
  try {
    for (const name of plan.toArchive) {
      fs.copyFileSync(path.join(cloneDir, name), path.join(staging, name));
    }

    const branch = plan.archiveBranch;
    try {
      runGit(`git fetch origin "${branch}"`, cloneDir);
      runGit(`git checkout -B "${branch}" "origin/${branch}"`, cloneDir);
    } catch {
      runGit(`git checkout -B "${branch}" master`, cloneDir);
    }
    for (const name of plan.toArchive) {
      fs.copyFileSync(path.join(staging, name), path.join(cloneDir, name));
    }
    runGit('git add .', cloneDir);
    try {
      runGit(`git commit -m "archive: rotate ${plan.toArchive.length} oldest sessions to ${branch}"`, cloneDir);
    } catch { /* nothing to commit — already archived */ }
    runGit(`git push "${pushUrl}" "${branch}"`, cloneDir);

    runGit('git checkout master', cloneDir);
    const rmArgs = plan.toArchive.map((n) => `"${n}"`).join(' ');
    runGit(`git rm --quiet ${rmArgs}`, cloneDir);
    runGit(`git commit -m "archive: rotate out ${plan.toArchive.length} oldest sessions (-> ${branch})"`, cloneDir);

    return branch;
  } finally {
    try { fs.rmSync(staging, { recursive: true, force: true }); } catch {}
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { taskId, framework, description } = body;

    if (!taskId) {
      return NextResponse.json({ error: 'Missing taskId' }, { status: 400 });
    }

    const randomSuffix = crypto.randomBytes(4).toString('hex');
    const filename = `cannbot_db_session_${taskId}_${randomSuffix}.db`;
    const tmpDir = os.tmpdir();
    const dbPath = path.join(tmpDir, `upload_${randomSuffix}.db`);

    let mirrorDir: string;
    try {
      mirrorDir = await ensureUploadMirror();
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Clone failed';
      return NextResponse.json({ error: `Failed to clone CANNBay mirror: ${msg}` }, { status: 500 });
    }

    try {
      const filePath = await exportSession(taskId, dbPath, prisma, framework);

      // Mirror already fetched origin/master in ensureUploadMirror — sync the
      // working tree to that tip (discards stale local state from a previous
      // failed upload; last success was pushed, so remote has it).
      try {
        runGit('git checkout -B master origin/master', mirrorDir);
      } catch {
        runGit('git checkout -B master', mirrorDir);
      }
      runGit('git clean -fdq', mirrorDir);

      fs.copyFileSync(filePath, path.join(mirrorDir, filename));
      runGit('git add .', mirrorDir);
      const commitMsg = description ?? `Add session ${taskId}`;
      const msgFile = path.join(mirrorDir, '_commit_msg.txt');
      fs.writeFileSync(msgFile, commitMsg);
      runGit(`git commit -F "${msgFile}"`, mirrorDir);
      try { fs.unlinkSync(msgFile); } catch {}

      // Integrate any concurrent remote pushes so our push is fast-forward.
      try {
        runGit('git fetch --quiet origin master', mirrorDir);
        runGit('git merge --allow-unrelated-histories --no-edit origin/master', mirrorDir);
      } catch {
        // Remote has no master yet (first-ever push) — nothing to integrate.
      }

      // Retention: cap master at the newest MAX_MASTER_SESSIONS .db files;
      // rotate older ones onto a date-bucketed archive branch before pushing.
      let archivedBranch: string | null = null;
      try {
        archivedBranch = rotateArchiveIfNeeded(mirrorDir, Date.now(), CANNBAY_PUSH_URL);
      } catch {
        // Archiving is best-effort: never let it block the upload.
      }

      runGit(`git push "${CANNBAY_PUSH_URL}" master`, mirrorDir);

      return NextResponse.json({ success: true, filename, archivedBranch });
    } finally {
      try { fs.unlinkSync(dbPath); } catch {}
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
