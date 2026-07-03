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

const CANNBAY_PULL_URL = 'https://gitcode.com/guanxinghua/CANNBay.git';
const CANNBAY_PUSH_URL = Buffer.from('aHR0cHM6Ly9ndWFueGluZ2h1YTpwc3F5WXAyYnpFRkI0eDVQRlVTV0dMS3lAZ2l0Y29kZS5jb20vZ3VhbnhpbmdodWEvQ0FOTkJheS5naXQ=', 'base64').toString();

function runGit(cmd: string, cwd: string) {
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: 60_000 });
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
    const cloneDir = path.join(tmpDir, `cannbay_${randomSuffix}`);

    try {
      const filePath = await exportSession(taskId, dbPath, prisma, framework);

      try {
        runGit(`git clone "${CANNBAY_PULL_URL}" "${cloneDir}"`, tmpDir);
      } catch {
        fs.mkdirSync(cloneDir, { recursive: true });
        runGit('git init', cloneDir);
        runGit(`git remote add origin "${CANNBAY_PULL_URL}"`, cloneDir);
      }

      fs.copyFileSync(filePath, path.join(cloneDir, filename));

      runGit('git add .', cloneDir);
      const commitMsg = description ?? `Add session ${taskId}`;
      const msgFile = path.join(cloneDir, '_commit_msg.txt');
      fs.writeFileSync(msgFile, commitMsg);
      runGit(`git commit -F "${msgFile}"`, cloneDir);
      try { fs.unlinkSync(msgFile); } catch {}

      // Set push URL with credentials and force branch name
      runGit(`git remote set-url origin "${CANNBAY_PUSH_URL}"`, cloneDir);
      runGit('git branch -M master', cloneDir);
      runGit('git push -u origin master', cloneDir);

      return NextResponse.json({ success: true, filename });
    } finally {
      try { fs.unlinkSync(dbPath); } catch {}
      try { fs.rmSync(cloneDir, { recursive: true, force: true }); } catch {}
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : 'Unknown error';
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
