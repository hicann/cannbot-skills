// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * Detect and parse Claude Code CLI command turns and continuation (compact summary) turns.
 *
 * Commands like /compact produce 3 consecutive user turns:
 *   1. <command-name>/xxx</command-name><command-message>xxx</command-message><command-args>...</command-args>
 *   2. <local-command-caveat>...</local-command-caveat>
 *   3. <local-command-stdout>...</local-command-stdout>
 *
 * Continuation turns (compact summary) are injected after /compact:
 *   "This session is being continued from a previous conversation..."
 */

const COMMAND_PATTERNS = {
  commandName: /<command-name>([\s\S]*?)<\/command-name>/,
  commandArgs: /<command-args>([\s\S]*?)<\/command-args>/,
  localCaveat: /<local-command-caveat>/,
  localStdout: /<local-command-stdout>([\s\S]*?)<\/local-command-stdout>/,
}

const CONTINUATION_MARKER = "This session is being continued from a previous conversation"

export interface CommandInfo {
  name: string
  args: string
  output: string | null
}

export interface ContinuationInfo {
  /** First line of the summary, e.g. "Primary Request and Intent: ..." */
  summaryLine: string | null
  /** Number of numbered sections in the summary */
  sectionCount: number
  /** Full summary text */
  fullSummary: string
}

export function isCommandTurn(text: string): boolean {
  return COMMAND_PATTERNS.commandName.test(text)
}

export function isCommandCaveat(text: string): boolean {
  return COMMAND_PATTERNS.localCaveat.test(text)
}

export function isCommandStdout(text: string): boolean {
  return COMMAND_PATTERNS.localStdout.test(text)
}

export function isAnyCommandRelated(text: string): boolean {
  return isCommandTurn(text) || isCommandCaveat(text) || isCommandStdout(text)
}

export function isContinuationTurn(text: string): boolean {
  // A real /compact continuation summary STARTS with the marker. Using
  // includes() would falsely match any turn that QUOTES the marker (e.g. an
  // assistant explaining the compact feature), mis-detecting it as a compact
  // boundary and truncating the reconstructed input at the wrong point.
  return text.startsWith(CONTINUATION_MARKER)
}

export function parseCommandTurns(texts: string[]): CommandInfo {
  let name = "unknown"
  let args = ""
  let output: string | null = null

  for (const text of texts) {
    const nameMatch = COMMAND_PATTERNS.commandName.exec(text)
    if (nameMatch) name = nameMatch[1].trim()

    const argsMatch = COMMAND_PATTERNS.commandArgs.exec(text)
    if (argsMatch) args = argsMatch[1].trim()

    const stdoutMatch = COMMAND_PATTERNS.localStdout.exec(text)
    if (stdoutMatch) {
      // Strip ANSI escape sequences
      output = stdoutMatch[1].replace(/\x1b\[[0-9;]*m/g, "").trim()
    }
  }

  return { name, args, output }
}

export function formatCommandDisplay(info: CommandInfo): string {
  return info.args ? `${info.name} ${info.args}` : info.name
}

export function parseContinuationTurn(text: string): ContinuationInfo {
  // Extract the first substantive line after the marker
  const lines = text.split("\n")
  const afterMarker = lines.slice(1).filter(l => l.trim().length > 0 && !l.startsWith("Summary:") && !l.startsWith(CONTINUATION_MARKER))
  const summaryLine = afterMarker.length > 0 ? afterMarker[0].trim() : null

  // Count numbered sections (e.g. "1.", "2.", "3.")
  const sectionLines = text.match(/^\d+\.\s/m)
  const sectionCount = sectionLines ? sectionLines.length : 0

  return {
    summaryLine: summaryLine ? summaryLine.substring(0, 120) : null,
    sectionCount,
    fullSummary: text,
  }
}
