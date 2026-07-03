// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface SkillEventItem {
  id: string
  skillName: string
  skillVersion: number | null
  eventType: string
  success: boolean
  errorMessage: string | null
  argsJson: string | null
  durationMs: number
}

export interface SkillToolCallItem {
  id: string
  toolCallId: string
  toolName: string
  argsJson: string | null
  resultJson: string | null
  state: string
  durationMs: number
}

export interface SkillGroup {
  skillName: string
  skillVersion: number | null
  loadEvent: SkillEventItem | null
  invokeEvent: SkillEventItem | null
  dispatchEvent: SkillEventItem | null
  otherEvents: SkillEventItem[]
  allSuccess: boolean
  compositeId: string
  loadArgsJson: string | null
  invokeArgsJson: string | null
  invokeResultJson: string | null
  loadResultJson: string | null
  dispatchArgsJson: string | null
  dispatchResultJson: string | null
}

export function extractSkillNameFromArgs(argsJson: string | null): string | null {
  if (!argsJson) return null
  try {
    const args = JSON.parse(argsJson)
    return args.skill ?? args.skill_name ?? args.name ?? args.subagent_type ?? args.subagent_name ?? null
  } catch { return null }
}

export function groupSkillEvents(events: SkillEventItem[], toolCalls?: SkillToolCallItem[]): SkillGroup[] {
  const byName = new Map<string, SkillEventItem[]>()
  for (const se of events) {
    const arr = byName.get(se.skillName) ?? []
    arr.push(se)
    byName.set(se.skillName, arr)
  }

  const tcByName = new Map<string, SkillToolCallItem[]>()
  if (toolCalls) {
    for (const tc of toolCalls) {
      const skillName = extractSkillNameFromArgs(tc.argsJson)
      if (skillName) {
        const arr = tcByName.get(skillName) ?? []
        arr.push(tc)
        tcByName.set(skillName, arr)
      }
    }
  }

  const groups: SkillGroup[] = []
  for (const [skillName, skillEvents] of byName) {
    const loadEvent = skillEvents.find(se => se.eventType === 'load') ?? null
    const invokeEvent = skillEvents.find(se => se.eventType === 'invoke' || se.eventType === 'use') ?? null
    const dispatchEvent = skillEvents.find(se => se.eventType === 'dispatch') ?? null
    const otherEvents = skillEvents.filter(se => se.eventType !== 'load' && se.eventType !== 'invoke' && se.eventType !== 'use' && se.eventType !== 'dispatch')

    const allSuccess = skillEvents.every(se => se.success)
    const version = loadEvent?.skillVersion ?? invokeEvent?.skillVersion ?? skillEvents[0]?.skillVersion ?? null
    const compositeId = skillEvents.map(se => se.id).join('-')

    const matchingTcs = tcByName.get(skillName) ?? []
    const loadTc = loadEvent ? matchingTcs.find(tc => tc.toolName.toLowerCase() === 'skill/load_skill' || tc.toolName.toLowerCase() === 'load_skill') : null
    const invokeTc = invokeEvent ? matchingTcs.find(tc => tc.toolName.toLowerCase() === 'skill/invoke' || tc.toolName.toLowerCase() === 'skill' || tc.toolName.toLowerCase() === 'skill/use') : null
    const dispatchTc = dispatchEvent ? matchingTcs.find(tc => tc.toolName.toLowerCase() === 'agent' || tc.toolName.toLowerCase() === 'task') : null

    groups.push({
      skillName,
      skillVersion: version,
      loadEvent,
      invokeEvent,
      dispatchEvent,
      otherEvents,
      allSuccess,
      compositeId,
      loadArgsJson: loadTc?.argsJson ?? loadEvent?.argsJson ?? null,
      invokeArgsJson: invokeTc?.argsJson ?? invokeEvent?.argsJson ?? null,
      invokeResultJson: invokeTc?.resultJson ?? null,
      loadResultJson: loadTc?.resultJson ?? null,
      dispatchArgsJson: dispatchTc?.argsJson ?? dispatchEvent?.argsJson ?? null,
      dispatchResultJson: dispatchTc?.resultJson ?? null,
    })
  }

  return groups
}
