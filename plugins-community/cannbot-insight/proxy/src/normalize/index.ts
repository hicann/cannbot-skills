// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// normalize 层（布局归一）：读取捕获层的 verbatim jsonl，在 norm/ 下产出
// 导入视图 —— verbatim 行纯拷贝 + subagents 目录镜像。独立于捕获路径
// （emitter 不感知本层），幂等可重跑。
//
// 内容 interpret（task-notification 摘要、skill 注入分类等框架行为解释）
// 一律在 insight 的 claude-jsonl adapter —— 它对 verbatim 与 norm 两种输入
// 形状都正确，本层不再重复（双份逻辑会漂移）。
//
// 架构分层（proxy 内）：
//   捕获层  server/emitters/writer —— verbatim 落盘，不感知 insight
//   规范层  normalize/*            —— 布局归一（norm/ 导入视图）
//   消费层  cannbot-insight        —— 导入 + 框架行为解释 + 渲染

import fs from 'node:fs';
import path from 'node:path';
import type { JsonlLine } from './line-transforms';
import {
  normMainFile,
  normMainMetaFile,
  normSubagentFile,
  normSubagentDir,
  captureSubagentDir,
  sidOfCapture,
} from './layout';
import { PROXY_DIR } from '../writer';

export interface NormalizeResult {
  sid: string;
  mainFile: string;
  subagentFiles: string[];
}

function readLines(file: string): JsonlLine[] {
  const lines: JsonlLine[] = [];
  for (const ln of fs.readFileSync(file, 'utf-8').split('\n')) {
    const t = ln.trim();
    if (!t) continue;
    try {
      lines.push(JSON.parse(t) as JsonlLine);
    } catch {
      // 截断的最后一行（会话进行中）—— 跳过，重跑时补齐
    }
  }
  return lines;
}

function writeLines(file: string, lines: JsonlLine[]): void {
  fs.mkdirSync(path.dirname(file), { recursive: true });
  fs.writeFileSync(file, lines.map(l => JSON.stringify(l)).join('\n') + '\n');
}

export function normalizeSession(captureFile: string): NormalizeResult | null {
  if (!fs.existsSync(captureFile)) return null;
  const sid = sidOfCapture(captureFile);
  const lines = readLines(captureFile);
  const mainFile = normMainFile(sid);
  writeLines(mainFile, lines);

  // 主会话 meta（cc-session-meta）随行镜像到 norm/ —— cpx- 前缀剥掉，
  // 与主文件 stem 对齐，insight 侧按 <dirname>/<sid>.meta.json 发现。
  const captureMeta = captureFile.replace(/\.jsonl$/, '.meta.json');
  if (fs.existsSync(captureMeta)) {
    fs.copyFileSync(captureMeta, normMainMetaFile(sid));
  }

  const subagentFiles: string[] = [];
  const subDir = captureSubagentDir(sid);
  if (fs.existsSync(subDir)) {
    for (const entry of fs.readdirSync(subDir, { withFileTypes: true })) {
      if (!entry.isFile()) continue;
      const src = path.join(subDir, entry.name);
      if (entry.name.endsWith('.meta.json')) {
        const dst = path.join(normSubagentDir(sid), entry.name);
        fs.mkdirSync(path.dirname(dst), { recursive: true });
        fs.copyFileSync(src, dst);
      } else if (entry.name.endsWith('.jsonl')) {
        const subId = entry.name.replace(/\.jsonl$/, '');
        const dst = normSubagentFile(sid, subId);
        writeLines(dst, readLines(src));
        subagentFiles.push(dst);
      }
    }
  }
  return { sid, mainFile, subagentFiles };
}

export function normalizeAll(): NormalizeResult[] {
  const dir = process.env.CANNBOT_PROXY_DIR ?? PROXY_DIR;
  if (!fs.existsSync(dir)) return [];
  const results: NormalizeResult[] = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (!entry.isFile() || !entry.name.endsWith('.jsonl')) continue;
    const r = normalizeSession(path.join(dir, entry.name));
    if (r) results.push(r);
  }
  return results;
}
