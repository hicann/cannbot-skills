// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// cannbay2 编排层（route 唯一入口）：list / import / upload。
// 上传数据流：源（proxy 捕获 / native claude jsonl 原文件 / opencode DB 导出）
// → staging → 治理（清洗 + 复检熔断，governance.ts）→ mirror.uploadFolder。
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { listSessions, materializeSession, uploadFolder, type Cannbay2SessionEntry, type UploadResult } from './mirror';
import { governText } from './governance';
import { exportSessionToClaudeJsonl } from './export';
import { importSession } from '@/lib/ingest/data-service';
import type { PrismaClient } from '@prisma/client';

export type { Cannbay2SessionEntry };

const MAX_SESSION_BYTES = 100 * 1024 * 1024; // atomgit 单文件上限兜底

export class GovernanceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'GovernanceError';
  }
}

type PrismaLike = PrismaClient;

export function listCannbay2Sessions(): Cannbay2SessionEntry[] {
  return listSessions();
}

export async function importCannbay2Session(sid: string, prisma: PrismaLike) {
  const mainJsonl = materializeSession(sid);
  return importSession(mainJsonl, sid, prisma, mainJsonl, 'claude-jsonl');
}

interface SessionRow {
  id: string;
  taskId: string;
  framework: string;
  sourcePath: string | null;
}

// 收集待传文件到 staging 的 sessions/<sid>/ 布局，返回 staging 根目录。
// - proxy 捕获 / native claude：sourcePath 指向主 jsonl，subagents 在
//   <dirname>/<fileSid>/subagents/（proxy 落盘约定）
// - opencode：无捕获文件，从 DB 导出
async function stageSessionFiles(session: SessionRow, prisma: PrismaLike): Promise<{ stagingDir: string; sid: string }> {
  const sid = session.taskId;
  // staging 必须在镜像缓存目录之外：首次 clone / 镜像自愈会 rmSync 整个缓存目录
  const stagingDir = fs.mkdtempSync(path.join(os.tmpdir(), `cannbay2-staging-${sid}-`));
  const sessionDir = path.join(stagingDir, 'sessions', sid);
  fs.rmSync(stagingDir, { recursive: true, force: true });
  fs.mkdirSync(sessionDir, { recursive: true });

  if (session.framework !== 'opencode' && session.sourcePath && session.sourcePath.endsWith('.jsonl') && fs.existsSync(session.sourcePath)) {
    fs.copyFileSync(session.sourcePath, path.join(sessionDir, `${sid}.jsonl`));
    const fileSid = path.basename(session.sourcePath, '.jsonl');
    const subagentsDir = path.join(path.dirname(session.sourcePath), fileSid, 'subagents');
    if (fs.existsSync(subagentsDir)) {
      fs.cpSync(subagentsDir, path.join(sessionDir, 'subagents'), { recursive: true });
    }
  } else if (session.framework === 'opencode') {
    await exportSessionToClaudeJsonl(prisma, session.taskId, stagingDir);
  } else {
    throw new Error(`Session "${sid}" has no uploadable jsonl source (sourcePath missing or not .jsonl)`);
  }
  return { stagingDir, sid };
}

// 治理 staging 内 sessions/<sid>/ 全部文件：清洗写回 + 复检。
// 治理只碰 staging 副本，永不改写源捕获文件；复检残留 → 熔断。
function governStagedSession(stagingDir: string, sid: string): void {
  const sessionDir = path.join(stagingDir, 'sessions', sid);
  const files = fs.readdirSync(sessionDir, { recursive: true }).filter(f => f.endsWith('.jsonl') || f.endsWith('.json')) as string[];
  const residues: string[] = [];
  let totalBytes = 0;
  for (const rel of files) {
    const file = path.join(sessionDir, rel);
    const raw = fs.readFileSync(file, 'utf8');
    totalBytes += Buffer.byteLength(raw);
    const { output, residue } = governText(raw);
    if (residue.length > 0) residues.push(...residue.map(r => `${rel}: ${r}`));
    fs.writeFileSync(file, output);
  }
  if (totalBytes > MAX_SESSION_BYTES) {
    throw new GovernanceError(`Session "${sid}" exceeds single-file size limit (${(totalBytes / 1024 / 1024).toFixed(1)}MB > 100MB)`);
  }
  if (residues.length > 0) {
    throw new GovernanceError(`上传熔断：清洗后仍检出疑似密钥残留，已拒绝上传。检出项（值已截断）：\n${residues.slice(0, 10).join('\n')}`);
  }
}

export async function uploadCannbay2Session(
  prisma: PrismaLike,
  taskId: string,
  description?: string,
): Promise<UploadResult & { sid: string }> {
  const session = await prisma.session.findFirst({ where: { taskId } }) as SessionRow | null;
  if (!session) throw new Error(`Session not found: "${taskId}"`);

  const { stagingDir, sid } = await stageSessionFiles(session, prisma);
  try {
    governStagedSession(stagingDir, sid);
    const commitMessage = description?.trim() || `Add session ${sid}`;
    const result = await uploadFolder(stagingDir, sid, commitMessage);
    return { ...result, sid };
  } finally {
    fs.rmSync(stagingDir, { recursive: true, force: true });
  }
}
