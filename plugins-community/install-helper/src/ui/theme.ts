// ----------------------------------------------------------------------------------------------------------
// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// ----------------------------------------------------------------------------------------------------------

import chalk from "chalk";

export const selectTheme = {
  helpMode: "never" as const,
};

export function isCJK(code: number): boolean {
  return (
    (code >= 0x2E80 && code <= 0x9FFF) ||
    (code >= 0xF900 && code <= 0xFAFF) ||
    (code >= 0xFE30 && code <= 0xFE4F) ||
    (code >= 0xFF00 && code <= 0xFFEF) ||
    (code >= 0x3400 && code <= 0x4DBF) ||
    (code >= 0x20000 && code <= 0x2A6DF) ||
    (code >= 0x2A700 && code <= 0x2B73F) ||
    (code >= 0x2B740 && code <= 0x2B81F) ||
    (code >= 0x2B820 && code <= 0x2CEAF)
  );
}

function isEmoji(code: number): boolean {
  return (
    (code >= 0x1F300 && code <= 0x1FAFF) ||
    (code >= 0x2600 && code <= 0x27BF)
  );
}

export function getDisplayWidth(str: string): number {
  let width = 0;
  for (const ch of str) {
    const code = ch.codePointAt(0)!;
    if (isCJK(code) || isEmoji(code)) {
      width += 2;
    } else {
      width += 1;
    }
  }
  return width;
}

function wrapDescription(text: string): string {
  const prefix = "  💡 ";
  const indent = "      ";
  const cols = process.stdout.columns || 120;
  const maxLen = Math.max(20, cols - getDisplayWidth(prefix) - 2);

  const tokens: { text: string; width: number }[] = [];
  let i = 0;
  while (i < text.length) {
    const code = text[i].codePointAt(0)!;
    if (isCJK(code)) {
      tokens.push({ text: text[i], width: 2 });
      i++;
    } else {
      let j = i;
      while (j < text.length) {
        const c = text[j].codePointAt(0)!;
        if (isCJK(c)) break;
        j++;
      }
      const word = text.slice(i, j);
      tokens.push({ text: word, width: getDisplayWidth(word) });
      i = j;
    }
  }

  const lines: string[] = [];
  let line = "";
  let lineWidth = 0;
  for (const token of tokens) {
    if (lineWidth + token.width > maxLen && line) {
      lines.push(line.trimEnd());
      line = token.text;
      lineWidth = token.width;
    } else {
      line += token.text;
      lineWidth += token.width;
    }
  }
  if (line) lines.push(line.trimEnd());

  return lines
    .map((l, i) => chalk.cyan(i === 0 ? prefix + l : indent + l))
    .join("\n");
}

export const checkboxTheme = {
  icon: {
    checked: "\x1b[32m☑\x1b[22m",
    unchecked: "☐",
    cursor: "❯",
  },
  helpMode: "never" as const,
  style: {
    description: (text: string) => wrapDescription(text),
  },
};
