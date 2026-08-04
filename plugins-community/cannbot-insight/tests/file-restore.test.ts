// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import {
  extractFilePath,
  extractOffset,
  parseReadLines,
  parseWriteLines,
  restoreFile,
  renderRestoredText,
  type RestoreOp,
} from '../src/lib/file-restore.ts';

const t = (n: number) => new Date(n);

describe('file-restore', () => {
  describe('extractFilePath / extractOffset', () => {
    it('normalizes file_path → filePath', () => {
      expect(extractFilePath('{"file_path":"/a/b.ts"}')).toBe('/a/b.ts');
      expect(extractFilePath('{"filePath":"/a/b.ts"}')).toBe('/a/b.ts');
    });
    it('returns null on missing/invalid', () => {
      expect(extractFilePath(null)).toBeNull();
      expect(extractFilePath('{"x":1}')).toBeNull();
      expect(extractFilePath('nope')).toBeNull();
    });
    it('extracts offset', () => {
      expect(extractOffset('{"offset":50}')).toBe(50);
      expect(extractOffset('{"offset":0}')).toBe(0);
      expect(extractOffset('{}')).toBeUndefined();
    });
  });

  describe('parseReadLines', () => {
    it('parses claude cat -n numbered format', () => {
      const out = parseReadLines('1\talpha\n2\tbeta\n3\tgamma');
      expect(out.get(1)).toBe('alpha');
      expect(out.get(2)).toBe('beta');
      expect(out.get(3)).toBe('gamma');
      expect(out.size).toBe(3);
    });
    it('handles padded line numbers', () => {
      const out = parseReadLines('   1\talpha\n  10\tbeta');
      expect(out.get(1)).toBe('alpha');
      expect(out.get(10)).toBe('beta');
    });
    it('keeps content with tabs after the first', () => {
      const out = parseReadLines('1\ta\tb\tc');
      expect(out.get(1)).toBe('a\tb\tc');
    });
    it('falls back to offset-based numbering when no prefix', () => {
      const out = parseReadLines('foo\nbar\nbaz', 5);
      expect(out.get(5)).toBe('foo');
      expect(out.get(6)).toBe('bar');
      expect(out.get(7)).toBe('baz');
    });
    it('falls back to numbering from 1 with no offset', () => {
      const out = parseReadLines('foo\nbar');
      expect(out.get(1)).toBe('foo');
      expect(out.get(2)).toBe('bar');
    });
    it('handles null/empty', () => {
      expect(parseReadLines(null).size).toBe(0);
      expect(parseReadLines('').size).toBe(0);
    });

    it('parses opencode envelope: strips <path>/<type>/<content> and reads N: colon lines', () => {
      const opencodeResult = [
        '<path>/docs/STATE.md</path>',
        '<type>file</type>',
        '<content>',
        '1: # Title',
        '2: ',
        '3: ## Section',
        '4: hello',
        '</content>',
      ].join('\n');
      const out = parseReadLines(opencodeResult);
      expect(out.get(1)).toBe('# Title');
      expect(out.get(2)).toBe('');
      expect(out.get(3)).toBe('## Section');
      expect(out.get(4)).toBe('hello');
      expect(out.size).toBe(4);
      expect(out.has(5)).toBe(false);
    });

    it('opencode colon format preserves legitimate leading indentation', () => {
      const r = ['<content>', '1:     indented', '2: x', '</content>'].join('\n');
      const out = parseReadLines(r);
      expect(out.get(1)).toBe('    indented');
      expect(out.get(2)).toBe('x');
    });

    it('claude tab format with envelope is also stripped at <content> marker', () => {
      const r = ['1\t<path>/a</path>', '2\t<type>file</type>', '3\t<content>', '4\treal1', '5\treal2', '6\t</content>'].join('\n');
      const out = parseReadLines(r);
      expect(out.get(4)).toBe('real1');
      expect(out.get(5)).toBe('real2');
      expect(out.size).toBe(2);
    });

    it('strips envelope but keeps blank content lines', () => {
      const r = ['<content>', '1: a', '2: ', '3: b', '</content>'].join('\n');
      const out = parseReadLines(r);
      expect(out.get(2)).toBe('');
      expect(out.size).toBe(3);
    });
  });

  describe('parseWriteLines', () => {
    it('splits content into numbered lines, dropping one trailing newline', () => {
      const out = parseWriteLines(JSON.stringify({ content: 'a\nb\nc\n' }));
      expect(out!.get(1)).toBe('a');
      expect(out!.get(2)).toBe('b');
      expect(out!.get(3)).toBe('c');
      expect(out!.size).toBe(3);
    });
    it('content without trailing newline', () => {
      const out = parseWriteLines(JSON.stringify({ content: 'x\ny' }));
      expect(out!.get(1)).toBe('x');
      expect(out!.get(2)).toBe('y');
    });
    it('returns null when content missing or not a string', () => {
      expect(parseWriteLines('{"file_path":"/a"}')).toBeNull();
      expect(parseWriteLines('{"content":123}')).toBeNull();
      expect(parseWriteLines(null)).toBeNull();
      expect(parseWriteLines('not json')).toBeNull();
    });
  });

  describe('restoreFile', () => {
    it('empty ops → no lines', () => {
      const r = restoreFile([]);
      expect(r.maxLine).toBe(0);
      expect(r.lines).toEqual([]);
      expect(r.opsUsed).toBe(0);
    });

    it('single read → continuous lines, no gaps', () => {
      const ops: RestoreOp[] = [
        { kind: 'read', argsJson: '{"offset":1}', resultJson: '1\ta\n2\tb\n3\tc', ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.maxLine).toBe(3);
      expect(r.lines.map(l => l.content)).toEqual(['a', 'b', 'c']);
      expect(r.lines.every(l => l.source === 'read')).toBe(true);
    });

    it('marks uncovered lines as gaps (per-line, not merged)', () => {
      const ops: RestoreOp[] = [
        { kind: 'read', argsJson: null, resultJson: '1\ta\n5\tb', ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.maxLine).toBe(5);
      expect(r.lines[0]).toMatchObject({ n: 1, content: 'a', source: 'read' });
      expect(r.lines[1]).toMatchObject({ n: 2, content: null, source: 'gap' });
      expect(r.lines[2]).toMatchObject({ n: 3, content: null, source: 'gap' });
      expect(r.lines[3]).toMatchObject({ n: 4, content: null, source: 'gap' });
      expect(r.lines[4]).toMatchObject({ n: 5, content: 'b', source: 'read' });
    });

    it('last write wins: read(t1) then write(t2) → write content on overlapping lines', () => {
      const ops: RestoreOp[] = [
        { kind: 'read', argsJson: null, resultJson: '1\told1\n2\told2', ts: t(1) },
        { kind: 'write', argsJson: JSON.stringify({ content: 'new1\nnew2' }), resultJson: null, ts: t(2) },
      ];
      const r = restoreFile(ops);
      expect(r.lines[0]).toMatchObject({ n: 1, content: 'new1', source: 'write' });
      expect(r.lines[1]).toMatchObject({ n: 2, content: 'new2', source: 'write' });
    });

    it('read after write: later read wins on its lines, write keeps its own', () => {
      const ops: RestoreOp[] = [
        { kind: 'write', argsJson: JSON.stringify({ content: 'w1\nw2\nw3' }), resultJson: null, ts: t(1) },
        { kind: 'read', argsJson: null, resultJson: '1\tr1', ts: t(2) },
      ];
      const r = restoreFile(ops);
      expect(r.lines[0]).toMatchObject({ n: 1, content: 'r1', source: 'read' });
      expect(r.lines[1]).toMatchObject({ n: 2, content: 'w2', source: 'write' });
      expect(r.lines[2]).toMatchObject({ n: 3, content: 'w3', source: 'write' });
    });

    it('sorts by ts regardless of input order', () => {
      const ops: RestoreOp[] = [
        { kind: 'write', argsJson: JSON.stringify({ content: 'late' }), resultJson: null, ts: t(5) },
        { kind: 'read', argsJson: null, resultJson: '1\tearly', ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.lines[0].content).toBe('late');
      expect(r.lines[0].source).toBe('write');
    });

    it('write trailing newline does not create extra gap line', () => {
      const ops: RestoreOp[] = [
        { kind: 'write', argsJson: JSON.stringify({ content: 'a\nb\n' }), resultJson: null, ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.maxLine).toBe(2);
      expect(r.lines.map(l => l.content)).toEqual(['a', 'b']);
    });

    it('skips ops that yield no lines (empty result)', () => {
      const ops: RestoreOp[] = [
        { kind: 'read', argsJson: null, resultJson: '', ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.maxLine).toBe(0);
      expect(r.opsUsed).toBe(0);
    });

    it('null ts treated as earliest (overwritten by any later op)', () => {
      const ops: RestoreOp[] = [
        { kind: 'read', argsJson: null, resultJson: '1\tnull-ts', ts: null },
        { kind: 'write', argsJson: JSON.stringify({ content: 'real' }), resultJson: null, ts: t(1) },
      ];
      const r = restoreFile(ops);
      expect(r.lines[0].content).toBe('real');
    });
  });

  describe('renderRestoredText', () => {
    it('renders content lines and gap markers joined by newline', () => {
      const text = renderRestoredText([
        { n: 1, content: 'a', source: 'read' },
        { n: 2, content: null, source: 'gap' },
        { n: 3, content: 'c', source: 'write' },
      ]);
      expect(text).toBe('a\n--line 2 not found --\nc');
    });
  });
});
