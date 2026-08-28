// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software: you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export interface AvailableSkill {
  name: string
  description: string
  origin: string | null
}

export interface UsedSkillAggregate {
  skillName: string
  invokeCount: number
  loadCount: number
  dispatchCount: number
  success: boolean
}

export type CoverageStatus = 'invoked' | 'loaded' | 'dispatched' | 'unused' | 'extra'

export interface CoverageItem {
  name: string
  description: string
  origin: string | null
  status: CoverageStatus
  invokeCount: number
  loadCount: number
  dispatchCount: number
}

export interface CoverageStats {
  availableTotal: number
  invoked: number
  loaded: number
  dispatched: number
  unused: number
  extra: number
}

export interface SkillCoverage {
  items: CoverageItem[]
  stats: CoverageStats
}

const CLAUDE_SKILLS_MARKER = 'The following skills are available'
const OPENCODE_SKILLS_OPEN = '<available_skills>'

export function parseClaudeAvailableSkills(text: string): AvailableSkill[] {
  const markerIdx = text.indexOf(CLAUDE_SKILLS_MARKER)
  if (markerIdx < 0) return []
  const after = text.slice(markerIdx)
  const lines = after.split('\n')
  const out: AvailableSkill[] = []
  const seen = new Set<string>()
  for (const raw of lines.slice(1)) {
    const line = raw.trim()
    if (line === '') {
      if (out.length > 0) break
      continue
    }
    if (!line.startsWith('- ')) {
      if (out.length > 0) break
      continue
    }
    const entry = line.slice(2)
    const colonIdx = entry.indexOf(': ')
    if (colonIdx <= 0) continue
    const name = entry.slice(0, colonIdx).trim()
    let description = entry.slice(colonIdx + 2).trim()
    let origin: string | null = null
    const fromMatch = description.match(/\s\(from (.+?)\)\.?$/)
    if (fromMatch) {
      origin = fromMatch[1]
      description = description.slice(0, fromMatch.index ?? 0).trim()
    }
    if (!name || seen.has(name)) continue
    seen.add(name)
    out.push({ name, description, origin })
  }
  return out
}

export function parseOpencodeAvailableSkills(text: string): AvailableSkill[] {
  const openIdx = text.indexOf(OPENCODE_SKILLS_OPEN)
  if (openIdx < 0) return []
  const closeIdx = text.indexOf('</available_skills>', openIdx)
  const block = closeIdx >= 0 ? text.slice(openIdx, closeIdx) : text.slice(openIdx)
  const out: AvailableSkill[] = []
  const seen = new Set<string>()
  const re = /<skill>([\s\S]*?)<\/skill>/g
  let m: RegExpExecArray | null
  while ((m = re.exec(block)) !== null) {
    const body = m[1]
    const name = body.match(/<name>([\s\S]*?)<\/name>/)?.[1]?.trim() ?? ''
    if (!name || seen.has(name)) continue
    seen.add(name)
    const description = body.match(/<description>([\s\S]*?)<\/description>/)?.[1]?.trim() ?? ''
    const origin = body.match(/<location>([\s\S]*?)<\/location>/)?.[1]?.trim() ?? null
    out.push({ name, description, origin })
  }
  return out
}

export function parseAvailableSkills(text: string): AvailableSkill[] {
  const xml = parseOpencodeAvailableSkills(text)
  if (xml.length > 0) return xml
  return parseClaudeAvailableSkills(text)
}

export function aggregateUsedSkills(
  events: Array<{ skillName: string; eventType: string; success: boolean }>
): UsedSkillAggregate[] {
  const byName = new Map<string, UsedSkillAggregate>()
  for (const e of events) {
    let agg = byName.get(e.skillName)
    if (!agg) {
      agg = { skillName: e.skillName, invokeCount: 0, loadCount: 0, dispatchCount: 0, success: true }
      byName.set(e.skillName, agg)
    }
    if (e.eventType === 'invoke' || e.eventType === 'use') agg.invokeCount++
    else if (e.eventType === 'load') agg.loadCount++
    else if (e.eventType === 'dispatch') agg.dispatchCount++
    if (!e.success) agg.success = false
  }
  return Array.from(byName.values())
}

export function buildCoverage(
  available: AvailableSkill[] | null,
  used: UsedSkillAggregate[]
): SkillCoverage {
  const items: CoverageItem[] = []
  const usedByName = new Map(used.map(u => [u.skillName, u]))
  const consumedUsed = new Set<string>()

  if (available) {
    for (const av of available) {
      const u = usedByName.get(av.name)
      if (u) consumedUsed.add(av.name)
      let status: CoverageStatus = 'unused'
      if (u && u.invokeCount > 0) status = 'invoked'
      else if (u && u.loadCount > 0) status = 'loaded'
      else if (u && u.dispatchCount > 0) status = 'dispatched'
      items.push({
        name: av.name,
        description: av.description,
        origin: av.origin,
        status,
        invokeCount: u?.invokeCount ?? 0,
        loadCount: u?.loadCount ?? 0,
        dispatchCount: u?.dispatchCount ?? 0,
      })
    }
    for (const u of used) {
      if (consumedUsed.has(u.skillName)) continue
      items.push({
        name: u.skillName,
        description: '',
        origin: null,
        status: 'extra',
        invokeCount: u.invokeCount,
        loadCount: u.loadCount,
        dispatchCount: u.dispatchCount,
      })
    }
  } else {
    for (const u of used) {
      const status: CoverageStatus =
        u.invokeCount > 0 ? 'invoked' : u.loadCount > 0 ? 'loaded' : 'dispatched'
      items.push({
        name: u.skillName,
        description: '',
        origin: null,
        status,
        invokeCount: u.invokeCount,
        loadCount: u.loadCount,
        dispatchCount: u.dispatchCount,
      })
    }
  }

  // 未使用置顶：覆盖度看板的核心问题是"哪些没用上"
  const statusOrder: Record<CoverageStatus, number> = {
    unused: 0, invoked: 1, loaded: 2, dispatched: 3, extra: 4,
  }
  items.sort((a, b) => statusOrder[a.status] - statusOrder[b.status] || a.name.localeCompare(b.name))

  const stats: CoverageStats = {
    availableTotal: available ? available.length : 0,
    invoked: items.filter(i => i.status === 'invoked').length,
    loaded: items.filter(i => i.status === 'loaded').length,
    dispatched: items.filter(i => i.status === 'dispatched').length,
    unused: items.filter(i => i.status === 'unused').length,
    extra: items.filter(i => i.status === 'extra').length,
  }
  return { items, stats }
}
