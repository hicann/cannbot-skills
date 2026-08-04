// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

export const TRAJECTORY_PATTERNS = {
  turn: /^## §(\d+) /,
  skill_call: /^\*Skill: (\S+) \((\w+)\) (✅|❌)/,
  tool_skill: /^\*\*Tool: skill\*\*/,
  tool_task: /^\*\*Tool: task\*\*/,
  skill_content_open: /<skill_content name="([^"]+)">/,
  skill_content_close: /<\/skill_content>/,
  pass: /^(PASS)\s*$/,
  fail: /^(FAILED|FAIL)\s*$/,
  skill_err: /^\*Skill:.*❌/,
  tool_err: /^\*Error:/,
  duration: /\*\*Duration:\*\*\s*(\S+)/,
  tokens: /\*\*Tokens:\*\*\s*([^|]+)/,
  stats_row: /\|\s*(\w+)\s*\|\s*[^|]+\|\s*[^|]+\|\s*([^|]+?)\s*\|/,
  section_h: /^### \*\*§(\d+\.\d+)\*\*/,
} as const

export interface SkeletonEntry {
  turn: number
  skill: string
  type: string
  status: "ok" | "fail"
  line: number
}

export interface SkeletonResult {
  skeleton: SkeletonEntry[]
  occurrences: Record<string, number>
  turnCount: number
}

export function extractSkeleton(text: string): SkeletonResult {
  const lines = text.split("\n")
  const skeleton: SkeletonEntry[] = []
  const occurrences: Record<string, number> = {}
  const turnSeen = new Set<number>()

  let currentTurn = 0
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const tm = line.match(TRAJECTORY_PATTERNS.turn)
    if (tm) {
      currentTurn = parseInt(tm[1], 10)
      turnSeen.add(currentTurn)
      continue
    }
    const sm = line.match(TRAJECTORY_PATTERNS.skill_call)
    if (sm) {
      const [, skill, type, mark] = sm
      const entry: SkeletonEntry = {
        turn: currentTurn,
        skill,
        type,
        status: mark === "✅" ? "ok" : "fail",
        line: i + 1,
      }
      skeleton.push(entry)
      occurrences[skill] = (occurrences[skill] ?? 0) + 1
    }
  }

  return { skeleton, occurrences, turnCount: turnSeen.size }
}

export interface SkillContentResult {
  skillMd: string
  skillName: string
}

export function extractSkillContent(text: string): SkillContentResult {
  const lines = text.split("\n")
  const invokeSkillLines: number[] = []
  for (let i = 0; i < lines.length; i++) {
    const sm = lines[i].match(TRAJECTORY_PATTERNS.skill_call)
    if (sm && sm[2] === "invoke") {
      invokeSkillLines.push(i)
    }
  }

  for (const startLineIdx of invokeSkillLines) {
    for (let j = startLineIdx; j < Math.min(lines.length, startLineIdx + 500); j++) {
      const openM = lines[j].match(TRAJECTORY_PATTERNS.skill_content_open)
      if (openM) {
        const skillName = openM[1]
        const closeIdx = findCloseTag(lines, j + 1)
        if (closeIdx === -1) break
        return { skillMd: lines.slice(j, closeIdx + 1).join("\n"), skillName }
      }
    }
  }

  return { skillMd: "", skillName: "" }
}

function findCloseTag(lines: string[], from: number): number {
  for (let i = from; i < lines.length; i++) {
    if (TRAJECTORY_PATTERNS.skill_content_close.test(lines[i])) return i
  }
  return -1
}

export interface StatsResult {
  duration: string
  tokens: string
  turns: string
  subagents: string
  cost: string
}

export function parseStats(text: string): StatsResult {
  const durationM = text.match(TRAJECTORY_PATTERNS.duration)
  const tokensM = text.match(TRAJECTORY_PATTERNS.tokens)

  let turns = ""
  let subagents = ""
  let cost = ""

  const lines = text.split("\n")
  for (const line of lines) {
    const rm = line.match(TRAJECTORY_PATTERNS.stats_row)
    if (rm) {
      const [, metric, total] = rm
      if (metric === "Turns") turns = total.trim()
      else if (metric === "Subagents") subagents = total.trim()
      else if (metric === "Cost") cost = total.trim()
    }
  }

  return {
    duration: durationM?.[1] ?? "",
    tokens: (tokensM?.[1] ?? "").trim(),
    turns,
    subagents,
    cost,
  }
}

export interface GateEntry {
  turn: number
  result: "PASS" | "FAIL"
  line: number
  snippet: string
}

export function extractGates(text: string): GateEntry[] {
  const lines = text.split("\n")
  const gates: GateEntry[] = []
  let currentTurn = 0
  let inDetails = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (/^\s*<details/.test(line)) inDetails++
    if (/^\s*<\/details>/.test(line)) inDetails = Math.max(0, inDetails - 1)

    const tm = line.match(TRAJECTORY_PATTERNS.turn)
    if (tm) {
      currentTurn = parseInt(tm[1], 10)
      continue
    }

    if (inDetails > 0) continue

    if (TRAJECTORY_PATTERNS.pass.test(line)) {
      gates.push({
        turn: currentTurn,
        result: "PASS",
        line: i + 1,
        snippet: line,
      })
    } else if (TRAJECTORY_PATTERNS.fail.test(line)) {
      gates.push({
        turn: currentTurn,
        result: "FAIL",
        line: i + 1,
        snippet: line,
      })
    }
  }

  return gates
}

export interface ErrorEntry {
  turn: number
  lineRange: [number, number]
  snippet: string
}

export function extractErrors(text: string, contextLines = 15): ErrorEntry[] {
  const lines = text.split("\n")
  const errors: ErrorEntry[] = []
  let currentTurn = 0

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    const tm = line.match(TRAJECTORY_PATTERNS.turn)
    if (tm) {
      currentTurn = parseInt(tm[1], 10)
      continue
    }

    const isSkillErr = TRAJECTORY_PATTERNS.skill_err.test(line)
    const isToolErr = TRAJECTORY_PATTERNS.tool_err.test(line)
    if (!isSkillErr && !isToolErr) continue

    const from = Math.max(0, i - contextLines)
    const to = Math.min(lines.length - 1, i + contextLines)
    errors.push({
      turn: currentTurn,
      lineRange: [from + 1, to + 1],
      snippet: lines.slice(from, to + 1).join("\n"),
    })
  }

  return errors
}

export type ReadRequest =
  | { lines: [number, number] }
  | { section: string }

export function readSection(text: string, req: ReadRequest): string {
  const lines = text.split("\n")

  if ("lines" in req) {
    const [from, to] = req.lines
    const start = Math.max(0, from - 1)
    const end = Math.min(lines.length, to)
    return lines.slice(start, end).join("\n")
  }

  if ("section" in req) {
    const target = req.section.replace(/^§/, "")
    let startIdx = -1
    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(TRAJECTORY_PATTERNS.section_h)
      if (m && m[1] === target) {
        startIdx = i
        break
      }
    }
    if (startIdx === -1) return ""

    let endIdx = lines.length
    for (let i = startIdx + 1; i < lines.length; i++) {
      const m = lines[i].match(TRAJECTORY_PATTERNS.section_h)
      if (m) {
        endIdx = i
        break
      }
    }
    return lines.slice(startIdx, endIdx).join("\n")
  }

  return ""
}
