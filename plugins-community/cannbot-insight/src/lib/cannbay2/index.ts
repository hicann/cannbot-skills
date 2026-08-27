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
import { isMetaPayload } from '../ingest/adapters/claude-jsonl';
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
  const result = await importSession(mainJsonl, sid, prisma, mainJsonl, 'claude-jsonl');
  // x_cannbay 补列（docs/cannbay-schema-spec.md）：核心管线无通道、只存在于
  // 声明里的列（Turn.ttftMs/temperature/maxTokens/modelId/providerId、
  // ToolCall.errorType/errorMessage/durationMs/startedAt、Session.framework/
  // version）在 cannbay2 编排层恢复 —— 核心红线不动。
  await patchXCannbayColumns(mainJsonl, sid, prisma).catch(() => { /* 补列失败不阻塞导入 */ });
  return result;
}

interface DbTurnPayload {
  ttftMs?: number;
  temperature?: number;
  maxTokens?: number;
  modelId?: string;
  providerId?: string;
  toolCalls?: Array<{ toolUseId: string; state?: string; errorType?: string; errorMessage?: string; durationMs?: number; startedAt?: string }>;
}

// 按声明补列。匹配按文件进行：主文件声明 ↔ 非 subagent assistant 轮，
// subagents/<subId>.jsonl 声明 ↔ 对应 subagent 轮。安全性规则：单文件
// assistant 轮数 ≠ 声明数 → 该文件放弃（形状不符时不猜，宁缺勿错）；
// 单条 toolUseId 匹配不到 → 跳过该条。
async function patchXCannbayColumns(mainJsonl: string, taskId: string, prisma: PrismaLike): Promise<void> {
  const sid = path.basename(mainJsonl, '.jsonl');
  const fileGroups: Array<{ subId: string | null; payloads: DbTurnPayload[] }> = [];
  const collect = (file: string, subId: string | null): void => {
    if (!fs.existsSync(file)) return;
    const payloads: DbTurnPayload[] = [];
    for (const ln of fs.readFileSync(file, 'utf8').split('\n')) {
      if (!ln.trim()) continue;
      try {
        const xb = (JSON.parse(ln) as { x_cannbay?: { schema?: string; data?: DbTurnPayload } }).x_cannbay;
        if (xb?.schema === 'cc-db-turn' && xb.data) payloads.push(xb.data);
      } catch { /* 截断行跳过 */ }
    }
    if (payloads.length > 0) fileGroups.push({ subId, payloads });
  };
  collect(mainJsonl, null);
  const subagentsDir = path.join(path.dirname(mainJsonl), sid, 'subagents');
  if (fs.existsSync(subagentsDir)) {
    for (const entry of fs.readdirSync(subagentsDir, { withFileTypes: true })) {
      if (entry.isFile() && entry.name.endsWith('.jsonl')) {
        collect(path.join(subagentsDir, entry.name), entry.name.replace(/\.jsonl$/, ''));
      }
    }
  }

  const session = await prisma.session.findFirst({ where: { taskId } });
  if (!session) return;

  for (const { subId, payloads } of fileGroups) {
    const turns = await prisma.turn.findMany({
      where: {
        sessionId: session.id, role: 'assistant', isSubagent: subId != null,
        ...(subId != null ? { subagentSessionId: subId } : {}),
      },
      orderBy: { turnIndex: 'asc' },
    });
    if (turns.length !== payloads.length) continue;
    for (let i = 0; i < turns.length; i++) {
      const d = payloads[i];
      const turnPatch: Record<string, unknown> = {};
      if (d.ttftMs != null) turnPatch.ttftMs = d.ttftMs;
      if (d.temperature != null) turnPatch.temperature = d.temperature;
      if (d.maxTokens != null) turnPatch.maxTokens = d.maxTokens;
      if (d.modelId != null) turnPatch.modelId = d.modelId;
      if (d.providerId != null) turnPatch.providerId = d.providerId;
      if (Object.keys(turnPatch).length > 0) {
        await prisma.turn.update({ where: { id: turns[i].id }, data: turnPatch });
      }
      for (const tc of d.toolCalls ?? []) {
        if (!tc?.toolUseId) continue;
        const tcPatch: Record<string, unknown> = {};
        if (tc.state != null) tcPatch.state = tc.state;
        if (tc.errorType != null) tcPatch.errorType = tc.errorType;
        if (tc.errorMessage != null) tcPatch.errorMessage = tc.errorMessage;
        if (tc.durationMs != null) tcPatch.durationMs = tc.durationMs;
        if (tc.startedAt != null) tcPatch.startedAt = new Date(tc.startedAt);
        if (Object.keys(tcPatch).length > 0) {
          await prisma.toolCall.updateMany({
            where: { turnId: turns[i].id, toolCallId: tc.toolUseId },
            data: tcPatch,
          });
        }
      }
    }
  }

  // cc-session-meta：framework/version 恢复（重导入不再错变 claude-code）。
  // 仅 insight-export 生效：cpx 捕获的 version/framework 由导入管线按行
  // 标记计算 —— version 形如 '2.1.234.467-claude-proxy'（徽标靠 -proxy
  // 后缀），meta 里的纯 ccVersion 会把徽标覆盖掉。
  const metaPath = mainJsonl.replace(/\.jsonl$/, '.meta.json');
  if (fs.existsSync(metaPath)) {
    try {
      const metaJson = JSON.parse(fs.readFileSync(metaPath, 'utf8')) as { x_cannbay?: { schema?: string; version?: number; data?: { producer?: string; framework?: string; ccVersion?: string } } };
      const xb = metaJson.x_cannbay;
      const d = xb && isMetaPayload(xb, 'cc-session-meta') ? xb.data : null;
      if (d?.producer === 'insight-export' && d.framework) {
        const sessionPatch: Record<string, unknown> = { framework: d.framework };
        if (d.ccVersion) sessionPatch.version = d.ccVersion;
        await prisma.session.update({ where: { id: session.id }, data: sessionPatch });
      }
    } catch { /* meta 损坏 → 不 patch */ }
  }
}

interface SessionRow {
  id: string;
  taskId: string;
  framework: string;
  sourcePath: string | null;
}

// 收集待传文件到 staging 的 sessions/<sid>/ 布局，返回 staging 根目录。
// - sourcePath 是本机存在的 jsonl（proxy 捕获 / native claude）：原文件直传，
//   subagents 在 <dirname>/<fileSid>/subagents/（proxy 落盘约定）
// - 其余一律 DB 导出兜底：opencode（无捕获文件）、v1 CANNBay 下载的 .db 快照
//   会话（sourcePath 是 .db）、源文件已被删除/移动的会话。导出件不含
//   Full Context（system/tools），管线数据齐全。
async function stageSessionFiles(session: SessionRow, prisma: PrismaLike): Promise<{ stagingDir: string; sid: string }> {
  const sid = session.taskId;
  // staging 必须在镜像缓存目录之外：首次 clone / 镜像自愈会 rmSync 整个缓存目录
  const stagingDir = fs.mkdtempSync(path.join(os.tmpdir(), `cannbay2-staging-${sid}-`));
  const sessionDir = path.join(stagingDir, 'sessions', sid);
  fs.rmSync(stagingDir, { recursive: true, force: true });
  fs.mkdirSync(sessionDir, { recursive: true });

  const hasJsonlSource = session.framework !== 'opencode'
    && !!session.sourcePath
    && session.sourcePath.endsWith('.jsonl')
    && fs.existsSync(session.sourcePath);
  if (hasJsonlSource) {
    fs.copyFileSync(session.sourcePath, path.join(sessionDir, `${sid}.jsonl`));
    const fileSid = path.basename(session.sourcePath!, '.jsonl');
    // 主会话 meta（cc-session-meta，若存在）随行上传 —— 文件级 producer/framework 声明
    const siblingMeta = path.join(path.dirname(session.sourcePath!), `${fileSid}.meta.json`);
    if (fs.existsSync(siblingMeta)) {
      fs.copyFileSync(siblingMeta, path.join(sessionDir, `${sid}.meta.json`));
    }
    const subagentsDir = path.join(path.dirname(session.sourcePath!), fileSid, 'subagents');
    if (fs.existsSync(subagentsDir)) {
      // 白名单拷贝：只带 .jsonl/.json —— 捕获目录里的杂散文件（.txt/.log/.env…）
      // 不清洗也不该出公开仓
      const dest = path.join(sessionDir, 'subagents');
      fs.mkdirSync(dest, { recursive: true });
      for (const entry of fs.readdirSync(subagentsDir, { withFileTypes: true })) {
        if (entry.isFile() && (entry.name.endsWith('.jsonl') || entry.name.endsWith('.json'))) {
          fs.copyFileSync(path.join(subagentsDir, entry.name), path.join(dest, entry.name));
        }
      }
    }
  } else {
    const written = await exportSessionToClaudeJsonl(prisma, session.taskId, stagingDir);
    const mainFile = path.join(stagingDir, 'sessions', sid, `${sid}.jsonl`);
    if (!written.includes(path.posix.join('sessions', sid, `${sid}.jsonl`)) || fs.statSync(mainFile).size === 0) {
      throw new Error(`Session "${sid}" has no uploadable content (no local jsonl source, DB export empty)`);
    }
  }
  return { stagingDir, sid };
}

// 治理 staging 内 sessions/<sid>/ 全部文件：清洗写回 + 复检。
// 治理只碰 staging 副本，永不改写源捕获文件；白名单外文件 / 复检残留 → 熔断。
function governStagedSession(stagingDir: string, sid: string): void {
  const sessionDir = path.join(stagingDir, 'sessions', sid);
  const all = fs.readdirSync(sessionDir, { recursive: true, withFileTypes: true });
  const files: string[] = [];
  for (const d of all) {
    if (!d.isDirectory()) files.push(path.relative(sessionDir, path.join(d.parentPath, d.name)));
  }
  const unexpected = files.filter(f => !f.endsWith('.jsonl') && !f.endsWith('.json'));
  if (unexpected.length > 0) {
    throw new GovernanceError(`上传熔断：staging 存在白名单外文件（仅允许 .jsonl/.json）：${unexpected.join(', ')}`);
  }
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
  if (!/^[\w.-]+$/.test(taskId)) throw new Error(`Invalid session id: "${taskId}"`);
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
