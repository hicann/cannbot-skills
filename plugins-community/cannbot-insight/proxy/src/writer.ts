// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';

export const PROXY_DIR = process.env.CANNBOT_PROXY_DIR
  ?? path.join(os.homedir(), '.cannbot-insight', 'proxy');

// Lazy read of the proxy dir so tests can redirect via CANNBOT_PROXY_DIR after
// module load (the const above is captured at server-startup time only).
function dir(): string {
  return process.env.CANNBOT_PROXY_DIR ?? PROXY_DIR;
}

// Top-level capture files carry a cpx- prefix: the stem is claude's REAL
// session id (identical to claude's native jsonl filename), so the prefix
// keeps proxy captures collision-free if the capture and native projects
// directories are ever archived together. Insight never parses the prefix —
// session identity there is the unprefixed sid (import param / Session.taskId);
// only proxy-side code goes through this helper.
export function captureStem(sid: string): string {
  return `cpx-${sid}`;
}

export function sessionFilePath(sid: string): string {
  return path.join(dir(), `${captureStem(sid)}.jsonl`);
}

// Main-session meta（cc-session-meta，schema 见 docs/cannbay-schema-spec.md）：
// producer/framework/protocol/ccVersion 等文件级声明，紧邻主捕获文件。
export function sessionMetaPath(sid: string): string {
  return path.join(dir(), `${captureStem(sid)}.meta.json`);
}

// Per-run manifest: the proxy appends every sid it emitted a record for.
// With header-based routing the capture sid is claude's REAL session id
// (differs from cpx's pinned sid; /resume can add more mid-run), so cpx reads
// this manifest at exit to know exactly which capture files this run produced.
export function sidsFilePath(pinnedSid: string): string {
  return path.join(dir(), `${captureStem(pinnedSid)}.sids`);
}

export function appendSid(pinnedSid: string, sid: string): void {
  fs.appendFileSync(sidsFilePath(pinnedSid), sid + '\n');
}

// Subagent capture files live under <PROXY_DIR>/<sid>/subagents/<subId>.jsonl
// (claude's convention, read by cannbot-insight's listSubagentSessions).
export function subagentFilePath(sid: string, subId: string): string {
  return path.join(dir(), sid, 'subagents', `${subId}.jsonl`);
}

export function subagentMetaPath(sid: string, subId: string): string {
  return subagentFilePath(sid, subId).replace(/\.jsonl$/, '.meta.json');
}

export function ensureProxyDir(): void {
  fs.mkdirSync(PROXY_DIR, { recursive: true });
}

// Append one claude-format JSON line to a capture file (creating parent dirs).
export function appendClaudeLine(file: string, line: object): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.appendFileSync(file, JSON.stringify(line) + '\n');
}

// Write the subagent meta.json (toolUseId + name + agentType) once per subagent.
export function writeMeta(metaPath: string, meta: object): void {
  fs.mkdirSync(path.dirname(metaPath), { recursive: true });
  fs.writeFileSync(metaPath, JSON.stringify(meta, null, 2));
}
