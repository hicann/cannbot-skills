// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbay2 git 镜像层：partial clone（--filter=blob:none）单镜像上传下载共用。
// 列表 = ls-tree + 一次 git log walk（零 blob 下载）；导入 = 按会话 cat-file
// 物化（重排成 proxy 布局）；上传 = narrow sparse-checkout + scoped add +
// commit + push main。写操作进程内 mutex 串行。测试通过
// CANNBAY2_REMOTE_URL / CANNBAY2_PUSH_URL / CANNBAY2_CACHE_DIR 指向本地 bare 仓。
import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';

export const CANNBAY2_PULL_URL = process.env.CANNBAY2_REMOTE_URL
  ?? 'https://atomgit.com/guanxinghua/cannbay2.git';
// 密文常量（与 v1 upload-session/route.ts 同方式），解码 = https://<user>:<token>@atomgit.com/…
const CANNBAY2_PUSH_URL_ENCODED = 'aHR0cHM6Ly9ndWFueGluZ2h1YTpwc3F5WXAyYnpFRkI0eDVQRlVTV0dMS3lAZ2l0Y29kZS5jb20vZ3VhbnhpbmdodWEvY2FubmJheTIuZ2l0';
export const CANNBAY2_PUSH_URL = process.env.CANNBAY2_PUSH_URL
  ?? Buffer.from(CANNBAY2_PUSH_URL_ENCODED, 'base64').toString();
export const CANNBAY2_BRANCH = process.env.CANNBAY2_BRANCH ?? 'main';

export function cannbay2CacheDir(): string {
  return process.env.CANNBAY2_CACHE_DIR
    ?? path.join(process.cwd(), 'tmp', 'cannbay2-cache');
}

export function runGit(cmd: string, cwd: string, timeoutMs = 120_000): Buffer {
  return execSync(cmd, { cwd, stdio: 'pipe', timeout: timeoutMs, maxBuffer: 256 * 1024 * 1024 });
}

function runGitText(cmd: string, cwd: string): string {
  return runGit(cmd, cwd).toString();
}

// ── 镜像生命周期 ───────────────────────────────────────────────────────

function cloneMirror(dir: string): void {
  fs.rmSync(dir, { recursive: true, force: true });
  fs.mkdirSync(path.dirname(dir), { recursive: true });
  runGit(`git clone --quiet --filter=blob:none --no-checkout "${CANNBAY2_PULL_URL}" "${dir}"`, path.dirname(dir), 600_000);
}

/** 确保 partial clone 镜像存在且 origin/main 元数据最新（不下载 blob）。全同步，单进程内无重入。 */
export function ensureMirror(): string {
  const dir = cannbay2CacheDir();
  if (fs.existsSync(path.join(dir, '.git'))) {
    try {
      runGit(`git fetch --quiet origin ${CANNBAY2_BRANCH}`, dir);
      return dir;
    } catch {
      fs.rmSync(dir, { recursive: true, force: true });
    }
  }
  cloneMirror(dir);
  return dir;
}

export function revParseMain(dir: string): string | null {
  try {
    return runGitText(`git rev-parse origin/${CANNBAY2_BRANCH}`, dir).trim();
  } catch {
    return null;
  }
}

// ── 列表（零 blob） ────────────────────────────────────────────────────

export interface Cannbay2SessionEntry {
  sid: string;
  fileCount: number;
  author: string;
  submitter: string;
  description: string;
  commitTime: string;
}

function parseCommitSubject(subject: string, author: string): { submitter: string; description: string } {
  let submitter = '';
  let description = '';
  const subMatch = subject.match(/提交人\s*[:：]\s*/);
  const descMatch = subject.match(/(?:内容)?描述\s*[:：]\s*/);
  const subStart = subMatch ? (subMatch.index ?? 0) + subMatch[0].length : -1;
  const descIdx = descMatch ? (descMatch.index ?? 0) : -1;
  if (subStart >= 0) {
    const end = descIdx >= subStart ? descIdx : subject.length;
    submitter = subject.slice(subStart, end).trim();
  }
  if (descIdx >= 0) description = subject.slice(descIdx + (descMatch![0].length)).trim().split('\n')[0];
  if (!submitter) submitter = author;
  if (!description) description = subject.split('\n')[0];
  return { submitter, description };
}

/** ls-tree + 一次 git log walk → 全部会话文件夹及提交元数据。 */
export function listSessions(): Cannbay2SessionEntry[] {
  const dir = ensureMirror();
  const mainRev = revParseMain(dir);
  if (!mainRev) return [];

  const treeOut = runGitText(`git ls-tree -r --name-only origin/${CANNBAY2_BRANCH} -- sessions/`, dir);
  const fileCount = new Map<string, number>();
  for (const line of treeOut.split('\n')) {
    const m = line.match(/^sessions\/([^/]+)\//);
    if (m) fileCount.set(m[1], (fileCount.get(m[1]) ?? 0) + 1);
  }

  // 一次 log walk：文件 → 最后触碰 commit（log 新→旧，首个命中即最新）
  const fileCommit = new Map<string, { author: string; time: string; subject: string }>();
  try {
    const logOut = runGitText(
      `git log --pretty=format:%x00%an%x1f%aI%x1f%s --name-only origin/${CANNBAY2_BRANCH} -- sessions/`,
      dir,
    );
    for (const record of logOut.split('\x00')) {
      const lines = record.split('\n').filter(l => l.trim());
      if (lines.length === 0) continue;
      const [author, time, ...subjectParts] = lines[0].split('\x1f');
      const subject = subjectParts.join('\x1f');
      for (const file of lines.slice(1)) {
        if (!fileCommit.has(file)) fileCommit.set(file, { author, time, subject });
      }
    }
  } catch { /* 无历史时列表退化为纯文件名 */ }

  const entries: Cannbay2SessionEntry[] = [];
  for (const [sid, count] of fileCount) {
    const newest = [...fileCommit.entries()]
      .filter(([f]) => f.startsWith(`sessions/${sid}/`))
      .map(([, c]) => c)
      .sort((a, b) => (b.time || '').localeCompare(a.time || ''))[0];
    const author = newest?.author ?? '';
    const { submitter, description } = parseCommitSubject(newest?.subject ?? '', author);
    entries.push({ sid, fileCount: count, author, submitter, description, commitTime: newest?.time ?? '' });
  }
  entries.sort((a, b) => (b.commitTime || '').localeCompare(a.commitTime || ''));
  return entries;
}

// ── 导入物化（cat-file，按需下载 blob） ─────────────────────────────────

/**
 * 把 sessions/<sid>/ 物化到 <cache>/materialized/，重排成 proxy 布局
 * （<sid>.jsonl 与 <sid>/subagents/ 并列 —— claude-jsonl adapter 的约定）。
 * 返回主 jsonl 路径。
 */
export function materializeSession(sid: string): string {
  const dir = ensureMirror();
  if (!revParseMain(dir)) throw new Error('cannbay2 mirror has no main branch');
  if (!/^[\w.-]+$/.test(sid)) throw new Error(`Invalid session id: "${sid}"`);

  const treeOut = runGitText(`git ls-tree -r origin/${CANNBAY2_BRANCH} -- sessions/${sid}/`, dir);
  const entries = treeOut.split('\n').filter(l => l.trim()).map(line => {
    const m = line.match(/^(\d+)\s+(\w+)\s+([0-9a-f]+)\t(.+)$/);
    if (!m) throw new Error(`Unexpected ls-tree line: ${line.slice(0, 60)}`);
    return { mode: m[1], type: m[2], oid: m[3], repoPath: m[4] };
  });
  if (entries.length === 0) throw new Error(`Session folder not found in cannbay2: "${sid}"`);

  const matDir = path.join(dir, 'materialized', sid);
  fs.rmSync(matDir, { recursive: true, force: true });
  let mainJsonl: string | null = null;
  for (const e of entries) {
    const rel = e.repoPath.slice(`sessions/${sid}/`.length);
    // sessions/<sid>/<sid>.jsonl → materialized/<sid>.jsonl
    // sessions/<sid>/subagents/* → materialized/<sid>/subagents/*
    const targetRel = rel === `${sid}.jsonl` ? `${sid}.jsonl` : `${sid}/${rel}`;
    const target = path.join(dir, 'materialized', targetRel);
    fs.mkdirSync(path.dirname(target), { recursive: true });
    fs.writeFileSync(target, runGit(`git cat-file -p ${e.oid}`, dir));
    if (targetRel === `${sid}.jsonl`) mainJsonl = target;
  }
  if (!mainJsonl) throw new Error(`Main jsonl missing for session "${sid}"`);
  return mainJsonl;
}

// ── 上传（sparse 物化 + scoped add + commit + push） ────────────────────

let writeLock: Promise<unknown> | null = null;
async function withWriteLock<T>(fn: () => Promise<T>): Promise<T> {
  while (writeLock) await writeLock;
  let release: () => void = () => {};
  writeLock = new Promise(r => { release = r; });
  try {
    return await fn();
  } finally {
    writeLock = null;
    release();
  }
}

export interface UploadResult {
  folder: string;
  unchanged: boolean;
  gitattributesFixed: boolean;
}

/**
 * 把 stagingDir/sessions/<sid>/ 提交推送到远端 main。
 * - narrow sparse-checkout：只物化该会话文件夹（+ .gitattributes）
 * - .gitattributes 若含 *.jsonl LFS 行 → 同 commit 剔除（防装了 git-lfs 的机器误传指针）
 * - push 失败（并发推进）→ fetch+merge 后重试一次
 */
export async function uploadFolder(stagingDir: string, sid: string, commitMessage: string): Promise<UploadResult> {
  return withWriteLock(async () => {
    const dir = ensureMirror();
    if (!revParseMain(dir)) throw new Error('cannbay2 mirror has no main branch');

    runGit('git sparse-checkout init --no-cone', dir);
    runGit(`git sparse-checkout set "/sessions/${sid}/**" "/.gitattributes"`, dir);
    runGit(`git checkout -B ${CANNBAY2_BRANCH} origin/${CANNBAY2_BRANCH}`, dir);
    runGit('git reset --quiet', dir); // 丢弃上次失败上传可能残留的 staged 状态

    const targetDir = path.join(dir, 'sessions', sid);
    fs.mkdirSync(targetDir, { recursive: true });
    fs.cpSync(path.join(stagingDir, 'sessions', sid), targetDir, { recursive: true });

    let gitattributesFixed = false;
    const gaPath = path.join(dir, '.gitattributes');
    if (fs.existsSync(gaPath)) {
      const ga = fs.readFileSync(gaPath, 'utf8');
      const fixed = ga.split('\n').filter(l => !(l.trim().startsWith('*.jsonl') && l.includes('filter=lfs'))).join('\n');
      if (fixed !== ga) {
        fs.writeFileSync(gaPath, fixed);
        runGit('git add .gitattributes', dir);
        gitattributesFixed = true;
      }
    }

    runGit(`git add sessions/${sid}`, dir);
    const msgFile = path.join(dir, '_commit_msg.txt');
    fs.writeFileSync(msgFile, commitMessage);
    let unchanged = false;
    try {
      runGit(`git -c user.name=cannbot-insight -c user.email=insight@localhost commit -F "${msgFile}"`, dir);
    } catch {
      unchanged = true; // 内容无变化（重复上传同内容）
    }
    try { fs.unlinkSync(msgFile); } catch { /* ignore */ }

    if (!unchanged) {
      const push = () => { runGit(`git push "${CANNBAY2_PUSH_URL}" ${CANNBAY2_BRANCH}`, dir, 300_000); };
      try {
        push();
      } catch {
        runGit(`git fetch --quiet origin ${CANNBAY2_BRANCH}`, dir);
        runGit(`git -c user.name=cannbot-insight -c user.email=insight@localhost merge --allow-unrelated-histories --no-edit origin/${CANNBAY2_BRANCH}`, dir);
        push();
      }
    }
    return { folder: `sessions/${sid}/`, unchanged, gitattributesFixed };
  });
}
