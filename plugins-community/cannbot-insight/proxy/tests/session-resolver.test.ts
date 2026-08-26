// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// Header-based sid routing: claude-code declares its real session id via
// x-claude-code-session-id on every request (verified against cc 2.1.234:
// main, subagent, and title-gen requests all carry the MAIN session id).
import { describe, it, expect } from 'vitest';
import { resolveSession } from '../src/session-resolver.ts';

const hdr = (sid: string) => (_name: string) => sid;

describe('resolveSession: x-claude-code-session-id routing', () => {
  it('anthropic request with the header routes to claude\'s real sid (beats pinned default)', () => {
    const r = resolveSession('/v1/messages', hdr('335a8fa9-1111-2222-3333-444455556666'), null, 'cpx-pinned-sid');
    expect(r.sid).toBe('335a8fa9-1111-2222-3333-444455556666');
    expect(r.protocol).toBe('anthropic');
  });

  it('unsafe header values are ignored — falls back to the pinned sid', () => {
    for (const bad of ['../../etc', 'a/b', '.hidden', '']) {
      expect(resolveSession('/v1/messages', hdr(bad), null, 'cpx-pinned-sid').sid).toBe('cpx-pinned-sid');
    }
  });

  it('openai-protocol requests ignore the claude header (opencode x-session-id path unaffected)', () => {
    const headerGet = (name: string) => name === 'x-claude-code-session-id' ? '335a8fa9-x' : null;
    const r = resolveSession('/v1/chat/completions', headerGet, null, 'cpx-pinned-sid');
    expect(r.sid).toBe('cpx-pinned-sid');
    expect(r.protocol).toBe('openai');
  });

  it('openai request with x-session-id routes to it (beats pinned default) — opencode resume-safe naming', () => {
    const headerGet = (name: string) => name === 'x-session-id' ? 'ses_fec213167ffe31P4l' : null;
    const r = resolveSession('/v1/chat/completions', headerGet, null, 'cpx-pinned-sid');
    expect(r.sid).toBe('ses_fec213167ffe31P4l');
    expect(r.protocol).toBe('openai');
  });

  it('openai request with UNSAFE x-session-id falls back to the pinned sid', () => {
    const headerGet = (name: string) => name === 'x-session-id' ? '../escape' : null;
    expect(resolveSession('/v1/chat/completions', headerGet, null, 'cpx-pinned-sid').sid).toBe('cpx-pinned-sid');
  });

  it('no header → pinned default (legacy behavior)', () => {
    expect(resolveSession('/v1/messages', () => null, null, 'cpx-pinned-sid').sid).toBe('cpx-pinned-sid');
  });

  it('no header, no pinned default → fingerprint fallback intact', () => {
    const body = { messages: [{ role: 'user', content: 'hello fingerprint' }] };
    const r = resolveSession('/v1/messages', () => null, body);
    expect(r.sid.startsWith('fp-')).toBe(true);
  });
});
