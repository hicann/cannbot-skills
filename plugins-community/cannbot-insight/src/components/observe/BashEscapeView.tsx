"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// claude-code REPL 的 `!` shell 转义：CLI 把命令与输出包成伪 XML 标签
// （<bash-input>/<bash-stdout>/<bash-stderr>）塞进下一条 user 消息。标签
// 本身就是模型读到的内容，这里只做显示糖（终端样式），不改变数据。
import { Badge } from "@/components/ui/badge"

export interface BashEscape {
  command: string
  stdout: string
  stderr: string
}

// 仅当整条消息就是一次 shell 转义时才启用显示糖；混有其他内容的消息
// 按普通文本渲染，避免美化吞掉上下文。
const BASH_ESCAPE_RE =
  /^\s*<bash-input>([\s\S]*?)<\/bash-input>(?:\s*<bash-stdout>([\s\S]*?)<\/bash-stdout>)?(?:\s*<bash-stderr>([\s\S]*?)<\/bash-stderr>)?\s*$/

export function parseBashEscape(text: string): BashEscape | null {
  const m = text.match(BASH_ESCAPE_RE)
  if (!m) return null
  return { command: m[1], stdout: m[2] ?? "", stderr: m[3] ?? "" }
}

export function BashEscapeView({ esc }: { esc: BashEscape }) {
  return (
    <div className="rounded-md border bg-muted/40 px-2 py-1.5 font-mono text-[11px] leading-relaxed overflow-x-auto">
      <div className="flex items-baseline gap-1.5">
        <Badge variant="gray" className="text-[9px] shrink-0">shell</Badge>
        <span className="text-emerald-600 dark:text-emerald-400 shrink-0">$</span>
        <span className="font-semibold text-foreground break-all">{esc.command}</span>
      </div>
      {esc.stdout && (
        <div className="whitespace-pre-wrap break-all text-foreground/75 mt-1">{esc.stdout}</div>
      )}
      {esc.stderr && (
        <div className="whitespace-pre-wrap break-all text-red-600 dark:text-red-400 mt-1">{esc.stderr}</div>
      )}
    </div>
  )
}
