// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import fs from "node:fs"
import path from "node:path"

export interface HandoffSessionRecord {
  sessionId: string
  from: string | null
  to: string[]
  birthHandoffDoc: string | null
  handoffDoc: string | null
  status?: string
}

export function readHandoffRegistry(projectPath: string): HandoffSessionRecord[] {
  const regPath = path.join(projectPath, ".opencode-handoff-registry.json")
  if (!fs.existsSync(regPath)) return []
  try { return JSON.parse(fs.readFileSync(regPath, "utf-8")) } catch { return [] }
}

export function writeHandoffRegistry(projectPath: string, registry: HandoffSessionRecord[]): void {
  const regPath = path.join(projectPath, ".opencode-handoff-registry.json")
  fs.writeFileSync(regPath, JSON.stringify(registry, null, 2), "utf-8")
}

export function findSessionRecord(registry: HandoffSessionRecord[], sessionId: string): HandoffSessionRecord | null {
  return registry.find(r => r.sessionId === sessionId) ?? null
}

export function addOrUpdateSessionRecord(
  registry: HandoffSessionRecord[],
  record: HandoffSessionRecord
): HandoffSessionRecord[] {
  const existing = registry.findIndex(r => r.sessionId === record.sessionId)
  if (existing >= 0) {
    const updated = [...registry]
    updated[existing] = record
    return updated
  }
  return [...registry, record]
}

export function updateSessionStatus(
  registry: HandoffSessionRecord[],
  sessionId: string,
  status: string
): HandoffSessionRecord[] {
  const index = registry.findIndex(r => r.sessionId === sessionId)
  if (index < 0) return registry
  const updated = [...registry]
  updated[index] = { ...updated[index], status }
  return updated
}

export function addChildToSession(
  registry: HandoffSessionRecord[],
  parentId: string,
  childId: string
): HandoffSessionRecord[] {
  const parentIndex = registry.findIndex(r => r.sessionId === parentId)
  if (parentIndex < 0) return registry
  const updated = [...registry]
  updated[parentIndex] = {
    ...updated[parentIndex],
    to: [...updated[parentIndex].to, childId]
  }
  return updated
}

export function extractOperatorNameFromPath(projectPath: string): string | null {
  const operatorsDir = path.join(projectPath, "operators")
  if (!fs.existsSync(operatorsDir)) return null
  for (const entry of fs.readdirSync(operatorsDir)) {
    const docsDir = path.join(operatorsDir, entry, "docs")
    if (fs.existsSync(docsDir)) return entry
  }
  return null
}

export interface HandoffLinks {
  parent: { id: string; title: string | null } | null
  children: Array<{ id: string; title: string | null }>
}

export function findOriginalSession(registry: HandoffSessionRecord[], sessionId: string): string | null {
  let current = sessionId
  const visited = new Set<string>()
  while (!visited.has(current)) {
    visited.add(current)
    const record = findSessionRecord(registry, current)
    if (!record || !record.from) break
    current = record.from
  }
  return current === sessionId ? null : current
}

export function computeSessionIdShort(sessionId: string): string {
  return sessionId.slice(0, 12)
}

export function computeHandoffNum(registry: HandoffSessionRecord[], sessionId: string): number {
  const record = findSessionRecord(registry, sessionId)
  return (record?.to.length ?? 0) + 1
}

export function computeHandoffDocName(sessionIdShort: string, handoffNum: number): string {
  return `SESSION-HANDOFF-${sessionIdShort}-${handoffNum}.md`
}

export function computeHandoffSessionTitle(sessionIdShort: string, handoffNum: number): string {
  return `${sessionIdShort}-handoff-${handoffNum}`
}

export function computeContinuationTitle(sessionIdShort: string, handoffNum: number): string {
  return `${sessionIdShort}-continue-${handoffNum}`
}

export function computeDoneFlagPath(projectPath: string, operatorName: string | null, sessionIdShort: string, handoffNum: number): string {
  const docName = `SESSION-HANDOFF-${sessionIdShort}-${handoffNum}.done`
  return operatorName
    ? path.join(projectPath, "operators", operatorName, "docs", docName)
    : path.join(projectPath, docName)
}

export function extractHandoffNumFromBirthDoc(birthHandoffDoc: string): number {
  const match = birthHandoffDoc.match(/-(\d+)\.md$/)
  return match ? parseInt(match[1], 10) : 1
}

export function extractSourceShortFromDocName(docName: string): string {
  const match = docName.match(/^SESSION-HANDOFF-(.+)-\d+\.md$/)
  return match?.[1] ?? ""
}

export function findBirthHandoffDoc(registry: HandoffSessionRecord[], sessionId: string): string | null {
  const record = findSessionRecord(registry, sessionId)
  return record?.birthHandoffDoc ?? null
}

export function buildHandoffLinks(
  registry: HandoffSessionRecord[],
  currentSessionId: string,
  titleMap: Map<string, string | null>
): HandoffLinks {
  const currentRecord = findSessionRecord(registry, currentSessionId)

  let parent: { id: string; title: string | null } | null = null
  if (currentRecord?.from) {
    const parentRecord = findSessionRecord(registry, currentRecord.from)
    const titleFromMap = titleMap.get(currentRecord.from)
    if (titleFromMap) {
      parent = { id: currentRecord.from, title: titleFromMap }
    } else if (parentRecord?.birthHandoffDoc) {
      const sourceShort = extractSourceShortFromDocName(parentRecord.birthHandoffDoc)
      const handoffNum = extractHandoffNumFromBirthDoc(parentRecord.birthHandoffDoc)
      parent = { id: currentRecord.from, title: computeContinuationTitle(sourceShort, handoffNum) }
    } else if (parentRecord?.handoffDoc) {
      const sourceShort = extractSourceShortFromDocName(parentRecord.handoffDoc)
      const handoffNum = extractHandoffNumFromBirthDoc(parentRecord.handoffDoc)
      parent = { id: currentRecord.from, title: computeHandoffSessionTitle(sourceShort, handoffNum) }
    } else {
      parent = { id: currentRecord.from, title: null }
    }
  }

  const children: Array<{ id: string; title: string | null }> = []
  for (const childId of (currentRecord?.to ?? [])) {
    const childRecord = findSessionRecord(registry, childId)
    if (childRecord?.birthHandoffDoc) {
      const sourceShort = extractSourceShortFromDocName(childRecord.birthHandoffDoc)
      const handoffNum = extractHandoffNumFromBirthDoc(childRecord.birthHandoffDoc)
      children.push({ id: childId, title: computeContinuationTitle(sourceShort, handoffNum) })
    } else if (childRecord?.handoffDoc) {
      const sourceShort = extractSourceShortFromDocName(childRecord.handoffDoc)
      const handoffNum = extractHandoffNumFromBirthDoc(childRecord.handoffDoc)
      children.push({ id: childId, title: computeHandoffSessionTitle(sourceShort, handoffNum) })
    } else {
      children.push({ id: childId, title: null })
    }
  }

  return { parent, children }
}
