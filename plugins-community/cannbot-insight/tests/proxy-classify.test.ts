// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software; you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// 统一 proxy 分类器测试：覆盖产物代际 Gen1（wire 指纹）→ Gen2（行级 source）
// → Gen4（meta.json 权威）→ Gen5（codex），以及信号优先级与 native 排除。
import { describe, it, expect, afterEach } from 'vitest';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { classifyProxyCapture } from '../src/lib/ingest/proxy-classify';

const tmpFiles: string[] = [];

function tmpCapture(lines: object[], meta?: object): string {
  const tmp = path.join(os.tmpdir(), `pclassify-${Date.now()}-${Math.random().toString(36).slice(2, 8)}.jsonl`);
  fs.writeFileSync(tmp, lines.map(l => JSON.stringify(l)).join('\n') + '\n');
  tmpFiles.push(tmp);
  if (meta) {
    const metaPath = tmp.replace(/\.jsonl$/, '.meta.json');
    fs.writeFileSync(metaPath, JSON.stringify(meta));
    tmpFiles.push(metaPath);
  }
  return tmp;
}

afterEach(() => {
  for (const f of tmpFiles.splice(0)) {
    try { fs.unlinkSync(f); } catch { /* ignore */ }
  }
});

describe('classifyProxyCapture 代际信号', () => {
  it('Gen1 早期无标记 opencode 捕获：wire 指纹（system/tools）+ "You are opencode" 推断归属', () => {
    const f = tmpCapture([
      { type: 'user', message: { role: 'user', content: 'go' } },
      { type: 'assistant', system: 'You are opencode, an interactive CLI tool.', tools: [{ name: 'bash' }], message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } },
    ]);
    const c = classifyProxyCapture(f);
    expect(c).toEqual({ isProxy: true, marker: 'opencode-proxy', framework: 'opencode', via: 'wire-fingerprint' });
  });

  it('Gen1 早期无标记 claude 捕获：wire 指纹 → claude-code', () => {
    const f = tmpCapture([
      { type: 'assistant', system: 'You are Claude Code, Anthropics official CLI for Claude.', tools: [{ name: 'Bash' }], message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] } },
    ]);
    const c = classifyProxyCapture(f);
    expect(c.framework).toBe('claude-code');
    expect(c.marker).toBe('claude-proxy');
    expect(c.via).toBe('wire-fingerprint');
  });

  it('Gen2 行级 source 标记：claude-proxy / opencode-proxy / codex-proxy 各归其属', () => {
    const mk = (source: string) => tmpCapture([
      { type: 'user', message: { role: 'user', content: 'go' }, source, tools: [{ name: 'bash' }] },
    ]);
    expect(classifyProxyCapture(mk('claude-proxy'))).toMatchObject({ framework: 'claude-code', via: 'source-line' });
    expect(classifyProxyCapture(mk('opencode-proxy'))).toMatchObject({ framework: 'opencode', via: 'source-line' });
    expect(classifyProxyCapture(mk('codex-proxy'))).toMatchObject({ framework: 'codex', via: 'source-line' });
  });

  it('历史污染导出件（裸 source:claude-proxy 无 wire/x_cannbay）不算 proxy 捕获（a25ebdfd 前上传件回归）', () => {
    const f = tmpCapture([
      { type: 'user', message: { role: 'user', content: 'go' }, source: 'claude-proxy', duration_ms: 1200 },
      { type: 'assistant', message: { role: 'assistant', id: 'a1', content: [{ type: 'text', text: 'ok' }], model: 'm', usage: { input_tokens: 1, output_tokens: 1 } }, source: 'claude-proxy', duration_ms: 3400 },
    ]);
    expect(classifyProxyCapture(f)).toEqual({ isProxy: false, marker: null, framework: null, via: null });
  });

  it('行级标记 + x_cannbay 声明（无 legacy system/tools）仍是 proxy', () => {
    const f = tmpCapture([
      { type: 'assistant', message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] }, source: 'opencode-proxy', x_cannbay: { schema: 'cc-wire-round', version: 1, data: { roundIndex: 0 } } },
    ]);
    expect(classifyProxyCapture(f)).toMatchObject({ isProxy: true, framework: 'opencode', via: 'source-line' });
  });

  it('Gen4 meta.json cc-session-meta 是文件级权威源，优先于行级标记', () => {
    const f = tmpCapture(
      [{ type: 'user', message: { role: 'user', content: 'go' }, source: 'claude-proxy' }],
      { x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'cpx', framework: 'codex', protocol: 'responses', sid: 'x' } } }
    );
    const c = classifyProxyCapture(f);
    expect(c).toEqual({ isProxy: true, marker: 'codex-proxy', framework: 'codex', via: 'meta' });
  });

  it('meta framework 值归一化：claude → claude-code / claude-code 直存', () => {
    const mk = (fw: string) => tmpCapture(
      [{ type: 'user', message: { role: 'user', content: 'go' } }],
      { x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'cpx', framework: fw } } }
    );
    expect(classifyProxyCapture(mk('claude')).framework).toBe('claude-code');
    expect(classifyProxyCapture(mk('claude-code')).framework).toBe('claude-code');
    expect(classifyProxyCapture(mk('opencode')).framework).toBe('opencode');
  });

  it('insight-export 的 meta（producer 门禁）不算 proxy 捕获', () => {
    const f = tmpCapture(
      [{ type: 'user', message: { role: 'user', content: 'go' } }],
      { x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'insight-export', framework: 'opencode' } } }
    );
    const c = classifyProxyCapture(f);
    expect(c.isProxy).toBe(false);
    expect(c.via).toBeNull();
  });

  it('native claude jsonl（无 source 标记、无 system/tools 顶层字段）不是 proxy', () => {
    const f = tmpCapture([
      { type: 'user', message: { role: 'user', content: 'go' }, parentUuid: null, uuid: 'u1', sessionId: 's1', cwd: '/x', version: '1.0.0', gitBranch: 'main' },
      { type: 'assistant', message: { role: 'assistant', content: [{ type: 'text', text: 'ok' }] }, uuid: 'a1' },
    ]);
    expect(classifyProxyCapture(f)).toEqual({ isProxy: false, marker: null, framework: null, via: null });
  });

  it('未知 meta framework / 破损 meta 回退下一级信号', () => {
    const f = tmpCapture(
      [{ type: 'user', message: { role: 'user', content: 'go' }, source: 'opencode-proxy', tools: [{ name: 'bash' }] }],
      { x_cannbay: { schema: 'cc-session-meta', version: 1, data: { producer: 'cpx', framework: 'future-agent' } } }
    );
    expect(classifyProxyCapture(f)).toMatchObject({ framework: 'opencode', via: 'source-line' });
  });
});

describe('isClaudeFormatSession 扩展到全 proxy 捕获', () => {
  it('codex-proxy 会话是 claude 格式（之前误判 false → wire/full-context/刷新全失效）', async () => {
    const { isClaudeFormatSession } = await import('../src/lib/shared/session-format');
    expect(isClaudeFormatSession('codex', 'gpt-5-codex-proxy')).toBe(true);
    expect(isClaudeFormatSession('opencode', '1.17.9-opencode-proxy')).toBe(true);
    expect(isClaudeFormatSession('claude-code', null)).toBe(true);
    expect(isClaudeFormatSession('opencode', null)).toBe(false);
    expect(isClaudeFormatSession('codex', null)).toBe(false);
  });
});
