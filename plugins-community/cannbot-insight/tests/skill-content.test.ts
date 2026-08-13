// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import {
  isSkillToolCall,
  extractSkillName,
  extractSkillNameFromReadPath,
  stripSkillPreamble,
  selectSkillContent,
  type SkillToolCall,
} from '../src/lib/skill-content.ts';

const t = (n: number) => new Date(n);

describe('skill-content', () => {
  describe('isSkillToolCall', () => {
    it('matches Skill/skill/load_skill/skill/*', () => {
      expect(isSkillToolCall('Skill')).toBe(true);
      expect(isSkillToolCall('skill')).toBe(true);
      expect(isSkillToolCall('load_skill')).toBe(true);
      expect(isSkillToolCall('skill/load_skill')).toBe(true);
      expect(isSkillToolCall('skill/invoke')).toBe(true);
      expect(isSkillToolCall('skill/foo')).toBe(true);
    });
    it('rejects non-skill tools', () => {
      expect(isSkillToolCall('Read')).toBe(false);
      expect(isSkillToolCall('Write')).toBe(false);
      expect(isSkillToolCall('Bash')).toBe(false);
    });
  });

  describe('extractSkillName', () => {
    it('reads args.skill', () => {
      expect(extractSkillName('Skill', '{"skill":"a2h-spec"}')).toBe('a2h-spec');
    });
    it('falls back to skill_name / name', () => {
      expect(extractSkillName('skill', '{"skill_name":"x"}')).toBe('x');
      expect(extractSkillName('skill', '{"name":"y"}')).toBe('y');
    });
    it('falls back to toolName strip when no args', () => {
      expect(extractSkillName('skill/ops-spec-gen', null)).toBe('ops-spec-gen');
    });
    it('falls back to toolName strip on bad json', () => {
      expect(extractSkillName('skill/foo', 'not json')).toBe('foo');
    });
  });

  describe('extractSkillNameFromReadPath', () => {
    it('extracts parent dir of SKILL.md (forward slash)', () => {
      expect(extractSkillNameFromReadPath('/a/b/ops-spec-gen/SKILL.md')).toBe('ops-spec-gen');
    });
    it('normalizes backslashes (windows)', () => {
      expect(extractSkillNameFromReadPath('C:\\Users\\me\\skills\\a2h\\SKILL.md')).toBe('a2h');
    });
    it('returns null for non-SKILL.md paths', () => {
      expect(extractSkillNameFromReadPath('/a/b/foo.md')).toBeNull();
      expect(extractSkillNameFromReadPath('')).toBeNull();
    });
  });

  describe('stripSkillPreamble', () => {
    it('strips "Base directory for this skill:" first line', () => {
      const c = 'Base directory for this skill: /x/y/.claude/skills/foo\n\n# Foo\n\nbody';
      expect(stripSkillPreamble(c)).toBe('# Foo\n\nbody');
    });
    it('leaves content without preamble untouched', () => {
      expect(stripSkillPreamble('# Title\nbody')).toBe('# Title\nbody');
    });
    it('handles preamble-only content', () => {
      expect(stripSkillPreamble('Base directory for this skill: /x')).toBe('');
    });
  });

  describe('selectSkillContent', () => {
    it('prefers native Skill tool result (strips preamble)', () => {
      const ops: SkillToolCall[] = [
        {
          toolName: 'Skill',
          argsJson: '{"skill":"foo"}',
          resultJson: 'Base directory for this skill: /x/foo\n\n# Foo\n\nreal content',
          startedAt: t(1),
        },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r).not.toBeNull();
      expect(r!.source).toBe('skill-tool');
      expect(r!.content).toBe('# Foo\n\nreal content');
      expect(r!.fullRead).toBe(true);
      expect(r!.maxLine).toBeNull();
    });

    it('falls back to Read of SKILL.md when no Skill tool', () => {
      const ops: SkillToolCall[] = [
        {
          toolName: 'Read',
          argsJson: '{"file_path":"/a/b/foo/SKILL.md"}',
          resultJson: '1\t# Foo\n2\t\n3\tbody',
          startedAt: t(1),
        },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r).not.toBeNull();
      expect(r!.source).toBe('read');
      expect(r!.content).toBe('# Foo\n\nbody');
      expect(r!.fullRead).toBe(true);
      expect(r!.maxLine).toBe(3);
    });

    it('marks Read with offset/limit as partial (not full)', () => {
      const ops: SkillToolCall[] = [
        {
          toolName: 'Read',
          argsJson: '{"file_path":"/a/b/foo/SKILL.md","offset":10,"limit":5}',
          resultJson: '10\t# Foo\n11\t\n12\tbody',
          startedAt: t(1),
        },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r!.source).toBe('read');
      expect(r!.fullRead).toBe(false);
      expect(r!.maxLine).toBe(12);
    });

    it('marks Read with truncation marker as partial', () => {
      const ops: SkillToolCall[] = [
        {
          toolName: 'Read',
          argsJson: '{"file_path":"/a/b/foo/SKILL.md"}',
          resultJson: '1\t# Foo\n2\tbody\n<system-reminder>File is large; only showing first 2 lines.</system-reminder>',
          startedAt: t(1),
        },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r!.source).toBe('read');
      expect(r!.fullRead).toBe(false);
    });

    it('picks the longest among multiple Skill tool candidates', () => {
      const ops: SkillToolCall[] = [
        { toolName: 'Skill', argsJson: '{"skill":"foo"}', resultJson: 'Base directory for this skill: /x\n\nshort', startedAt: t(1) },
        { toolName: 'Skill', argsJson: '{"skill":"foo"}', resultJson: 'Base directory for this skill: /x\n\nlonger content here', startedAt: t(2) },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r!.content).toBe('longer content here');
    });

    it('matches skill via skill/<name> synthetic toolName', () => {
      const ops: SkillToolCall[] = [
        { toolName: 'skill/foo', argsJson: '{"skill":"foo","file_path":"/a/foo/SKILL.md"}', resultJson: 'Base directory for this skill: /x\n\n# Foo', startedAt: t(1) },
      ];
      const r = selectSkillContent(ops, 'foo');
      expect(r!.source).toBe('skill-tool');
    });

    it('returns null when skillName does not match', () => {
      const ops: SkillToolCall[] = [
        { toolName: 'Skill', argsJson: '{"skill":"other"}', resultJson: 'Base directory for this skill: /x\n\n# Other', startedAt: t(1) },
      ];
      expect(selectSkillContent(ops, 'foo')).toBeNull();
    });

    it('returns null when no resultJson', () => {
      const ops: SkillToolCall[] = [
        { toolName: 'Skill', argsJson: '{"skill":"foo"}', resultJson: null, startedAt: t(1) },
      ];
      expect(selectSkillContent(ops, 'foo')).toBeNull();
    });

    it('ignores Read of non-SKILL.md files', () => {
      const ops: SkillToolCall[] = [
        { toolName: 'Read', argsJson: '{"file_path":"/a/b/foo/README.md"}', resultJson: '1\tstuff', startedAt: t(1) },
      ];
      expect(selectSkillContent(ops, 'foo')).toBeNull();
    });

    it('opencode envelope read result is handled by parseReadLines', () => {
      const r = [
        '<path>/a/foo/SKILL.md</path>',
        '<type>file</type>',
        '<content>',
        '1: # Foo',
        '2: body',
        '</content>',
      ].join('\n');
      const ops: SkillToolCall[] = [
        { toolName: 'Read', argsJson: '{"file_path":"/a/foo/SKILL.md"}', resultJson: r, startedAt: t(1) },
      ];
      const out = selectSkillContent(ops, 'foo');
      expect(out!.source).toBe('read');
      expect(out!.content).toBe('# Foo\nbody');
    });
  });
});
