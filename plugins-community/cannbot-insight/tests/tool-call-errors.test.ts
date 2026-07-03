// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { describe, it, expect } from 'vitest';
import { summarizeToolCallErrors } from '../src/lib/tool-call-errors.ts';

describe('tool-call-errors: summarizeToolCallErrors', () => {
  describe('cancelled detection', () => {
    it('detects <tool_use_error>Cancelled from resultJson', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: '<tool_use_error>Cancelled: parallel tool call Bash errored</tool_use_error>', state: 'completed', errorType: null, errorMessage: null },
      ])
      expect(result.cancelled).toBe(1)
      expect(result.total).toBe(1)
      expect(result.details[0].type).toBe('cancelled')
    })

    it('detects multiple cancelled calls', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: '<tool_use_error>Cancelled: parallel tool call Bash(SPEC…) errored</tool_use_error>', state: 'completed', errorType: null, errorMessage: null },
        { toolName: 'Read', resultJson: '<tool_use_error>Cancelled: parallel tool call Bash(SPEC…) errored</tool_use_error>', state: 'completed', errorType: null, errorMessage: null },
      ])
      expect(result.cancelled).toBe(2)
      expect(result.total).toBe(2)
    })
  })

  describe('failed detection', () => {
    it('detects Exit code from resultJson', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: 'Exit code 1\nTraceback (most recent call last):', state: 'completed', errorType: null, errorMessage: null },
      ])
      expect(result.failed).toBe(1)
      expect(result.total).toBe(1)
      expect(result.details[0].type).toBe('failed')
    })

    it('detects error state', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: null, state: 'error', errorType: null, errorMessage: null },
      ])
      expect(result.failed).toBe(1)
    })

    it('detects failed state', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: null, state: 'failed', errorType: null, errorMessage: null },
      ])
      expect(result.failed).toBe(1)
    })

    it('detects errorType field', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: null, state: 'completed', errorType: 'permission', errorMessage: 'Permission denied' },
      ])
      expect(result.failed).toBe(1)
    })
  })

  describe('skill fail detection', () => {
    it('detects SkillEvent success=false with errorMessage', () => {
      const result = summarizeToolCallErrors([], [
        { skillName: 'ops-registry-invoke-workflow', eventType: 'invoke', success: false, errorMessage: 'Skill load failed' },
      ])
      expect(result.skillFail).toBe(1)
      expect(result.total).toBe(1)
      expect(result.details[0].type).toBe('skill_fail')
    })

    it('ignores SkillEvent success=true', () => {
      const result = summarizeToolCallErrors([], [
        { skillName: 'ops-registry-invoke-workflow', eventType: 'invoke', success: true, errorMessage: null },
      ])
      expect(result.skillFail).toBe(0)
      expect(result.total).toBe(0)
    })
  })

  describe('mixed errors', () => {
    it('counts all error types together', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: 'Exit code 1\nPermissionError', state: 'completed', errorType: null, errorMessage: null },
        { toolName: 'Bash', resultJson: '<tool_use_error>Cancelled: parallel tool call Bash errored</tool_use_error>', state: 'completed', errorType: null, errorMessage: null },
        { toolName: 'Read', resultJson: '<tool_use_error>Cancelled: parallel tool call Bash errored</tool_use_error>', state: 'completed', errorType: null, errorMessage: null },
      ], [
        { skillName: 'my-skill', eventType: 'invoke', success: false, errorMessage: 'failed' },
      ])
      expect(result.failed).toBe(1)
      expect(result.cancelled).toBe(2)
      expect(result.skillFail).toBe(1)
      expect(result.total).toBe(4)
    })

    it('returns 0 for no errors', () => {
      const result = summarizeToolCallErrors([
        { toolName: 'Bash', resultJson: 'OK', state: 'completed', errorType: null, errorMessage: null },
      ])
      expect(result.total).toBe(0)
      expect(result.cancelled).toBe(0)
      expect(result.failed).toBe(0)
      expect(result.skillFail).toBe(0)
    })
  })
})
