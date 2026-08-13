// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS FILE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import type { ReactNode } from "react"

export interface TurnHighlight {
  keyword: string
  matchField?: "content" | "contentSummary" | "toolResult" | "toolError" | "toolArgs"
  toolName?: string
}

/**
 * Split text around keyword occurrences and wrap matches in <mark>.
 * Case-insensitive. Returns the original text unchanged when keyword is empty.
 */
export function highlightKeyword(text: string, keyword: string | undefined | null): ReactNode {
  if (!keyword || !keyword.trim()) return text
  const lowerText = text.toLowerCase()
  const lowerKeyword = keyword.trim().toLowerCase()
  const kwLen = keyword.trim().length
  const parts: Array<{ text: string; isKeyword: boolean }> = []
  let lastIndex = 0
  let idx = lowerText.indexOf(lowerKeyword)
  while (idx !== -1) {
    if (idx > lastIndex) {
      parts.push({ text: text.substring(lastIndex, idx), isKeyword: false })
    }
    parts.push({ text: text.substring(idx, idx + kwLen), isKeyword: true })
    lastIndex = idx + kwLen
    idx = lowerText.indexOf(lowerKeyword, lastIndex)
  }
  if (lastIndex < text.length) {
    parts.push({ text: text.substring(lastIndex), isKeyword: false })
  }
  return parts.map((p, i) =>
    p.isKeyword
      ? <mark key={i} className="bg-yellow-200 dark:bg-yellow-500/30 text-foreground rounded px-0.5">{p.text}</mark>
      : p.text
  )
}
