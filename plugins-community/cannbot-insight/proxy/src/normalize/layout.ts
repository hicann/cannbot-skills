// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// normalize 层的输出布局。insight 原生 claude-jsonl 导入按
// <parentDir>/<sessionId>/subagents/ 约定发现子代理文件，因此规范化产物
// 放在 PROXY_DIR/norm/ 下、镜像该约定：
//
//   PROXY_DIR/
//   ├── <sid>.jsonl                  verbatim 捕获（single source of truth）
//   ├── <sid>/subagents/*.jsonl      verbatim 子代理捕获
//   └── norm/
//       ├── <sid>.jsonl              规范化主文件（insight 导入这个）
//       └── <sid>/subagents/         规范化子代理 + meta.json 副本
//
// 导入 norm/<sid>.jsonl 时 sessionId 取文件名 stem = <sid>，子代理发现
// 解析为 dirname(norm/<sid>.jsonl)/<sid>/subagents = norm/<sid>/subagents ✓。

import path from 'node:path';
import { PROXY_DIR } from '../writer';

function dir(): string {
  return process.env.CANNBOT_PROXY_DIR ?? PROXY_DIR;
}

export function normDir(): string {
  return path.join(dir(), 'norm');
}

export function normMainFile(sid: string): string {
  return path.join(normDir(), `${sid}.jsonl`);
}

export function normMainMetaFile(sid: string): string {
  return path.join(normDir(), `${sid}.meta.json`);
}

export function normSubagentDir(sid: string): string {
  return path.join(normDir(), sid, 'subagents');
}

export function normSubagentFile(sid: string, subId: string): string {
  return path.join(normSubagentDir(sid), `${subId}.jsonl`);
}

export function captureSubagentDir(sid: string): string {
  return path.join(dir(), sid, 'subagents');
}

// 捕获文件名 stem 即 session id；顶层捕获带 cpx- 前缀（writer.captureStem，
// 防与原生 claude jsonl 归档混放时撞名），剥掉后 subagents 目录（无前缀，
// writer.subagentFilePath 的约定）才能对上。
export function sidOfCapture(captureFile: string): string {
  return path.basename(captureFile).replace(/\.jsonl$/, '').replace(/^cpx-/, '');
}
