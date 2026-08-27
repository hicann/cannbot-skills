// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 集成测试：verbatim 捕获 fixture → normalize → norm/ 产物断言。
// normalize 只做布局归一（纯拷贝），内容解释（task-notification 摘要等）
// 在 insight adapter —— 所以这里断言的核心是：
//   - 行内容逐字节等价（含 task-notification 原始 XML 原样保留）
//   - norm 布局镜像 insight 的 <parentDir>/<sessionId>/subagents/ 发现约定
//   - cpx- 前缀剥离：norm 文件名/subagents 目录用无前缀 sid

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import { normalizeSession, normalizeAll } from '../src/normalize/index';
import { sidOfCapture } from '../src/normalize/layout';

let tmpDir: string;

beforeAll(() => {
  tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cpx-norm-'));
  process.env.CANNBOT_PROXY_DIR = tmpDir;
});

afterAll(() => {
  delete process.env.CANNBOT_PROXY_DIR;
  fs.rmSync(tmpDir, { recursive: true, force: true });
});

const NOTIF_XML = `<task-notification>
<task-id>a3c210037e742e104</task-id>
<status>completed</status>
<summary>Agent "创建算子开发目录" finished</summary>
<result>DONE</result>
</task-notification>`;

function writeCapture(fileName: string, lines: object[]): string {
  const file = path.join(tmpDir, fileName);
  fs.writeFileSync(file, lines.map(l => JSON.stringify(l)).join('\n') + '\n');
  return file;
}

describe('layout: cpx- 前缀剥离', () => {
  it('cpx-<sid>.jsonl → sid（subagents 目录无前缀）', () => {
    expect(sidOfCapture('/x/cpx-abc-123.jsonl')).toBe('abc-123');
  });
  it('无前缀文件不受影响', () => {
    expect(sidOfCapture('/x/abc-123.jsonl')).toBe('abc-123');
  });
});

describe('normalizeSession（纯拷贝 + 布局）', () => {
  it('norm 行与 verbatim 逐行等价 —— task-notification 原样保留（insight 负责解释）', () => {
    const lines = [
      { type: 'user', message: { role: 'user', content: NOTIF_XML }, timestamp: '2026-08-18T07:46:31.000Z', source: 'claude-proxy' },
      { type: 'user', message: { role: 'user', content: '真实输入' }, source: 'claude-proxy' },
      { type: 'assistant', message: { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Agent', input: { prompt: 'p' } }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, system: [{ type: 'text', text: 'sys' }], tools: [{ name: 'Agent' }], source: 'claude-proxy' },
    ];
    const f = writeCapture('s-copy.jsonl', lines);
    const r = normalizeSession(f)!;
    const out = fs.readFileSync(r.mainFile, 'utf-8').trim().split('\n').map(l => JSON.parse(l));
    expect(out).toEqual(lines);
  });

  it('cpx- 前缀捕获 → norm/<sid>.jsonl + norm/<sid>/subagents/', () => {
    const lines = [
      { type: 'user', message: { role: 'user', content: '开始' }, source: 'claude-proxy' },
    ];
    const f = writeCapture('cpx-s-pfx.jsonl', lines);
    // subagents 目录约定是无前缀 <sid>/subagents/
    const subDir = path.join(tmpDir, 's-pfx', 'subagents');
    fs.mkdirSync(subDir, { recursive: true });
    fs.writeFileSync(path.join(subDir, 'sub-a.jsonl'), JSON.stringify({ type: 'user', message: { role: 'user', content: '子任务' }, source: 'claude-proxy' }) + '\n');
    fs.writeFileSync(path.join(subDir, 'sub-a.meta.json'), JSON.stringify({ toolUseId: 't1', name: '任务', agentType: 'developer' }));

    const r = normalizeSession(f)!;
    expect(r.sid).toBe('s-pfx');
    expect(r.mainFile).toBe(path.join(tmpDir, 'norm', 's-pfx.jsonl'));
    expect(fs.readFileSync(r.mainFile, 'utf-8')).toContain('开始');
    expect(r.subagentFiles).toHaveLength(1);
    expect(r.subagentFiles[0]).toBe(path.join(tmpDir, 'norm', 's-pfx', 'subagents', 'sub-a.jsonl'));
    expect(fs.readFileSync(r.subagentFiles[0], 'utf-8')).toContain('子任务');
    const metaCopy = path.join(path.dirname(r.subagentFiles[0]), 'sub-a.meta.json');
    expect(JSON.parse(fs.readFileSync(metaCopy, 'utf-8'))).toEqual({ toolUseId: 't1', name: '任务', agentType: 'developer' });
  });

  it('幂等：重跑产物逐字节一致', () => {
    const f = writeCapture('s-idem.jsonl', [
      { type: 'user', message: { role: 'user', content: NOTIF_XML }, source: 'claude-proxy' },
    ]);
    const a = normalizeSession(f)!;
    const first = fs.readFileSync(a.mainFile, 'utf-8');
    normalizeSession(f);
    expect(fs.readFileSync(a.mainFile, 'utf-8')).toBe(first);
  });

  it('截断的最后一行（会话进行中）被跳过', () => {
    const file = path.join(tmpDir, 's-trunc.jsonl');
    const good = JSON.stringify({ type: 'user', message: { role: 'user', content: '完整行' }, source: 'claude-proxy' });
    fs.writeFileSync(file, good + '\n{"type":"user","mess');
    const r = normalizeSession(file)!;
    const out = fs.readFileSync(r.mainFile, 'utf-8').trim().split('\n');
    expect(out).toHaveLength(1);
    expect(JSON.parse(out[0]).message.content).toBe('完整行');
  });
});

describe('normalizeAll', () => {
  it('扫描目录内全部捕获（含 cpx- 前缀文件）', () => {
    writeCapture('s-all1.jsonl', [{ type: 'user', message: { role: 'user', content: 'a' } }]);
    writeCapture('cpx-s-all2.jsonl', [{ type: 'user', message: { role: 'user', content: 'b' } }]);
    const rs = normalizeAll().map(r => r.sid);
    expect(rs).toContain('s-all1');
    expect(rs).toContain('s-all2'); // cpx- 前缀已剥
  });
});
