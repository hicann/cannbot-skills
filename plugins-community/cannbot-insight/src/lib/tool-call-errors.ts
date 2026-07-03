// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface ToolCallErrorItem {
  toolName: string
  resultJson?: string | null | undefined
  state: string
  errorType?: string | null | undefined
  errorMessage?: string | null | undefined
  turn?: { sessionId: string; turnIndex: number }
}

export interface SkillEventErrorItem {
  skillName: string
  eventType: string
  success: boolean
  errorMessage?: string | null | undefined
  turn?: { sessionId: string; turnIndex: number }
}

export interface ErrorSummary {
  total: number
  cancelled: number
  failed: number
  skillFail: number
  details: Array<{ toolName: string; type: string }>
}

export function summarizeToolCallErrors(
  toolCalls: ToolCallErrorItem[],
  skillEvents?: SkillEventErrorItem[]
): ErrorSummary {
  let failed = 0
  let cancelledCount = 0
  const details: Array<{ toolName: string; type: string }> = []

  for (const tc of toolCalls) {
    const r = tc.resultJson ?? ''

    if (r.includes('<tool_use_error>') && r.includes('Cancelled')) {
      cancelledCount++
      details.push({ toolName: tc.toolName, type: 'cancelled' })
    } else if (r.includes('Exit code') || tc.state === 'error' || tc.state === 'failed' || tc.errorType) {
      failed++
      details.push({ toolName: tc.toolName, type: 'failed' })
    }
  }

  let skillFail = 0
  if (skillEvents) {
    for (const se of skillEvents) {
      if (!se.success && se.errorMessage) {
        skillFail++
        details.push({ toolName: se.skillName, type: 'skill_fail' })
      }
    }
  }

  return {
    total: cancelledCount + failed + skillFail,
    cancelled: cancelledCount,
    failed,
    skillFail,
    details,
  }
}

export function countToolCallErrors(
  toolCalls: ToolCallErrorItem[],
  skillEvents?: SkillEventErrorItem[]
): Map<string, { count: number; firstTurnIndex: number | null }> {
  const map = new Map<string, { count: number; firstTurnIndex: number | null }>()

  for (const tc of toolCalls) {
    const r = tc.resultJson ?? ''
    let isError = false
    if (r.includes('<tool_use_error>') && r.includes('Cancelled')) isError = true
    else if (r.includes('Exit code') || tc.state === 'error' || tc.state === 'failed' || tc.errorType) isError = true

    if (isError && tc.turn?.sessionId) {
      const sid = tc.turn.sessionId
      const entry = map.get(sid) ?? { count: 0, firstTurnIndex: null }
      entry.count++
      if (entry.firstTurnIndex === null || tc.turn.turnIndex < entry.firstTurnIndex) {
        entry.firstTurnIndex = tc.turn.turnIndex
      }
      map.set(sid, entry)
    }
  }

  if (skillEvents) {
    for (const se of skillEvents) {
      if (!se.success && se.errorMessage && se.turn?.sessionId) {
        const sid = se.turn.sessionId
        const entry = map.get(sid) ?? { count: 0, firstTurnIndex: null }
        entry.count++
        if (entry.firstTurnIndex === null || se.turn.turnIndex < entry.firstTurnIndex) {
          entry.firstTurnIndex = se.turn.turnIndex
        }
        map.set(sid, entry)
      }
    }
  }

  return map
}
