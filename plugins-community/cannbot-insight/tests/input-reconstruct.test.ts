// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { selectInputContextTurns, isLocalCommandNoise, type ContextTurn } from '../src/lib/ingest/input-reconstruct';

function ct(turnIndex: number, role: string, content: string, agentName?: string | null): ContextTurn {
  return { id: `t${turnIndex}`, turnIndex, role, content, agentName: agentName ?? null, isSubagent: false, subagentSessionId: null };
}

const CONT = 'This session is being continued from a previous conversation that ran out of context.';
// A summary that QUOTES command tags mid-text (it summarizes a session that used /compact).
const CONT_WITH_COMMAND_TAGS = CONT + ' The /compact command produces `<command-name>/compact</command-name>`, `<local-command-caveat>`, `<local-command-stdout>` turns.';

describe('input-reconstruct — compact-aware LLM input window', () => {
  it('excludes pre-compact conversation; window starts at the continuation summary', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', 'old question 1'),
      ct(1, 'assistant', 'old answer 1'),
      ct(2, 'assistant', 'old answer 2'),
      ct(3, 'user', CONT),                       // compact boundary 1
      ct(4, 'user', '<local-command-stdout>Compacted</local-command-stdout>'), // noise, skipped
      ct(5, 'user', 'new question after compact'),
      ct(6, 'assistant', 'response'),            // target
    ];
    const win = selectInputContextTurns(turns, 6);
    expect(win.map(t => t.turnIndex)).toEqual([3, 5]);
    expect(win[0].content).toContain('This session is being continued');
    // pre-compact turns excluded
    expect(win.find(t => t.turnIndex === 0)).toBeUndefined();
    expect(win.find(t => t.turnIndex === 1)).toBeUndefined();
  });

  it('does not drop the summary when the summary text contains command tags', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', 'old'),
      ct(1, 'assistant', 'old answer'),
      ct(2, 'user', CONT_WITH_COMMAND_TAGS),     // summary quoting command tags
      ct(3, 'user', 'new question'),
      ct(4, 'assistant', 'response'),            // target
    ];
    const win = selectInputContextTurns(turns, 4);
    expect(win.map(t => t.turnIndex)).toEqual([2, 3]);
    expect(win[0].content).toContain('This session is being continued');
  });

  it('does not treat an assistant turn that QUOTES the marker as a compact boundary', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', CONT),                       // real compact boundary
      ct(1, 'user', 'new question'),
      ct(2, 'assistant', 'the marker is: "This session is being continued from a previous conversation" — note this'), // quotes marker
      ct(3, 'assistant', 'response'),            // target
    ];
    const win = selectInputContextTurns(turns, 3);
    // boundary stays at 0, not the quoted-marker assistant at 2
    expect(win.map(t => t.turnIndex)).toEqual([0, 1, 2]);
  });

  it('handles multiple compactions: each continuation truncates the window again', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', 'q1'),
      ct(1, 'assistant', 'a1'),
      ct(2, 'user', CONT),                       // compact 1
      ct(3, 'user', 'q2'),
      ct(4, 'assistant', 'a2'),
      ct(5, 'user', CONT),                       // compact 2
      ct(6, 'user', 'q3'),
      ct(7, 'assistant', 'a3'),                  // target
    ];
    const win = selectInputContextTurns(turns, 7);
    expect(win.map(t => t.turnIndex)).toEqual([5, 6]);
  });

  it('skips local command noise (command/caveat/stdout) but keeps real user turns', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', CONT),
      ct(1, 'user', '<command-name>/compact</command-name>'),
      ct(2, 'user', '<local-command-caveat>Caveat</local-command-caveat>'),
      ct(3, 'user', '<local-command-stdout>done</local-command-stdout>'),
      ct(4, 'user', 'real question'),
      ct(5, 'assistant', 'response'),            // target
    ];
    const win = selectInputContextTurns(turns, 5);
    expect(win.map(t => t.turnIndex)).toEqual([0, 4]);
  });

  it('without any compact, includes all prior eligible turns', () => {
    const turns: ContextTurn[] = [
      ct(0, 'user', 'q1'),
      ct(1, 'assistant', 'a1'),
      ct(2, 'user', 'q2'),
      ct(3, 'assistant', 'a2'),                  // target
    ];
    const win = selectInputContextTurns(turns, 3);
    expect(win.map(t => t.turnIndex)).toEqual([0, 1, 2]);
  });

  it('isLocalCommandNoise is start-anchored', () => {
    expect(isLocalCommandNoise('<command-name>/x</command-name>')).toBe(true);
    expect(isLocalCommandNoise('<local-command-caveat>x</local-command-caveat>')).toBe(true);
    expect(isLocalCommandNoise('<local-command-stdout>x</local-command-stdout>')).toBe(true);
    // mid-text occurrence (e.g. inside a summary) is NOT noise
    expect(isLocalCommandNoise('summary mentions <command-name>/x</command-name> in passing')).toBe(false);
    expect(isLocalCommandNoise(null)).toBe(false);
    expect(isLocalCommandNoise('normal user text')).toBe(false);
  });

  describe('opencode compaction: agent-based detection', () => {
    it('excludes compaction-boundary and starts window at compaction summary', () => {
      const turns: ContextTurn[] = [
        ct(0, 'system', 'system prompt'),
        ct(1, 'user', 'old question', null),
        ct(2, 'assistant', 'old answer', null),
        ct(3, 'assistant', 'old answer 2', null),
        ct(4, 'user', '/compact', 'compaction-boundary'),
        ct(5, 'assistant', '## Goal\nCompact summary of conversation', 'compaction'),
        ct(6, 'user', 'new question after compact', null),
        ct(7, 'assistant', 'response', null),
      ];
      const win = selectInputContextTurns(turns, 7);
      expect(win.map(t => t.turnIndex)).toEqual([5, 6]);
      expect(win.find(t => t.agentName === 'compaction-boundary')).toBeUndefined();
      expect(win.find(t => t.turnIndex === 0)).toBeUndefined();
      expect(win.find(t => t.turnIndex === 1)).toBeUndefined();
      expect(win.find(t => t.turnIndex === 2)).toBeUndefined();
    });

    it('compaction summary turn has no reconstructable prior context', () => {
      const turns: ContextTurn[] = [
        ct(0, 'system', 'system prompt'),
        ct(1, 'user', 'old question', null),
        ct(2, 'assistant', 'old answer', null),
        ct(3, 'user', '/compact', 'compaction-boundary'),
        ct(4, 'assistant', '## Goal\nCompact summary', 'compaction'),
      ];
      const win = selectInputContextTurns(turns, 4);
      expect(win).toEqual([]);
    });

    it('handles multiple compactions with agent-based boundaries', () => {
      const turns: ContextTurn[] = [
        ct(0, 'user', 'q1', null),
        ct(1, 'assistant', 'a1', null),
        ct(2, 'user', '/compact', 'compaction-boundary'),
        ct(3, 'assistant', 'summary 1', 'compaction'),
        ct(4, 'user', 'q2', null),
        ct(5, 'assistant', 'a2', null),
        ct(6, 'user', '/compact (auto)', 'compaction-boundary'),
        ct(7, 'assistant', 'summary 2', 'compaction'),
        ct(8, 'user', 'q3', null),
        ct(9, 'assistant', 'a3', null),
      ];
      const win = selectInputContextTurns(turns, 9);
      expect(win.map(t => t.turnIndex)).toEqual([7, 8]);
      expect(win.find(t => t.agentName === 'compaction-boundary')).toBeUndefined();
    });

    it('mixed format: old continuation + new compaction-boundary', () => {
      const turns: ContextTurn[] = [
        ct(0, 'user', 'q1', null),
        ct(1, 'assistant', 'a1', null),
        ct(2, 'user', CONT, null),
        ct(3, 'assistant', 'a2 after old compact', null),
        ct(4, 'user', '/compact', 'compaction-boundary'),
        ct(5, 'assistant', 'new format summary', 'compaction'),
        ct(6, 'user', 'q3', null),
        ct(7, 'assistant', 'a3', null),
      ];
      const win = selectInputContextTurns(turns, 7);
      expect(win.map(t => t.turnIndex)).toEqual([5, 6]);
    });
  });
});
