"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Badge } from "@/components/ui/badge"
import type { AuditFinding } from "@/lib/skill-eval-audit-types"
import {
  CATEGORY_LABEL,
  METHOD_TAG,
  NOISE_LABELS,
  VERDICT_DISPLAY,
} from "./verdictConfig"

/**
 * 单条对账结果卡片。移植自 audit-report.html 的 renderFinding + findingRow：
 * verdict 徽章 + №seq + category + source(s) + derived↺ + related（跳 №N）+ noise +
 * method + 指令 + 证据 + transcript。外层带 verdict 边框色，id=finding-<seq> 供 related 锚点跳转。
 *
 * 符号约定：# 仅用于 session 对话 turn（turn_refs 跳转），№ 用于 finding 序号（seq / related）。
 *
 * №seq 跳转：finding 有 turn_refs（仅 prohibition/step 类 FAIL/INDETERMINATE 由 LLM 填的
 * turn 序号）时，№seq 渲染为可点击 button → onJumpToTurn(turn_refs[0])，切到 Turns tab。
 * conditional/output 类的 FAIL 是段级缺席（trace 全程未调用 X），turn_refs 恒空 → №seq 不可点击。
 */
const VERDICT_BORDER: Record<string, string> = {
  PASS: "border-l-emerald-500",
  FAIL: "border-l-red-500",
  "N/A": "border-l-orange-500",
  UNRESOLVED: "border-l-muted-foreground/40 border-l-2 border-dashed",
  INDETERMINATE: "border-l-red-500 border-l-2 border-dashed",
}

export function FindingCard({
  f,
  onJumpToSeq,
  onJumpToTurn,
}: {
  f: AuditFinding
  onJumpToSeq?: (seq: number) => void
  onJumpToTurn?: (turn: number) => void
}) {
  const vd = VERDICT_DISPLAY[f.verdict] ?? {
    variant: "gray" as const,
    icon: "?",
    label: f.verdict || "?",
    hint: "",
  }
  const srcs = f.sources && f.sources.length > 0 ? f.sources : [f.source]
  const noise = f.verdict === "FAIL" && f.noise_label ? NOISE_LABELS[f.noise_label] : undefined
  const turnRefs = f.turn_refs ?? []
  const canJumpTurn = turnRefs.length > 0 && !!onJumpToTurn
  const jumpTitle = turnRefs.length === 1
    ? `跳转到 turn #${turnRefs[0]}`
    : `跳转到 turn #${turnRefs[0]}（共 ${turnRefs.length} 个: ${turnRefs.map(t => `#${t}`).join(" ")}）`
  // 指令行三层回退 + 判定锚点:与 audit-report.html 的 renderInstruction 对齐——
  // 只要显示的主标题(headline>excerpt>instruction_text)与 judge 实际判定的 instruction_text
  // 不同,就把 instruction_text 作小字「判定锚点」附下供核对。早先用 source_excerpt 判定会漏掉
  // headline 命中但 source_excerpt 为空的指令(prose 派生的 "Use reference file" 类),造成
  // 有的显示有的不显示。
  const mainInstruction = f.headline || f.source_excerpt || f.instruction_text || ""
  const showAnchor = mainInstruction.trim() !== (f.instruction_text || "").trim()

  return (
    <div
      id={f.seq ? `finding-${f.seq}` : undefined}
      className={`rounded border border-border bg-card px-3 py-2 text-xs ${VERDICT_BORDER[f.verdict] ?? "border-l-2 border-l-muted"}`}
    >
      <div className="flex flex-wrap items-center gap-1.5">
        <Badge variant={vd.variant} className="h-5 gap-0.5 px-1.5 font-mono" title={vd.hint}>
          <span>{vd.icon}</span>
          <span className="font-sans">{vd.label}</span>
        </Badge>
        {f.verdict === "N/A" && (
          <span className="text-[10px] text-muted-foreground" title="规则存在但本次 transcript 场景不触发">
            (场景不触发)
          </span>
        )}
        {f.seq ? (
          canJumpTurn ? (
            <button
              type="button"
              onClick={() => onJumpToTurn?.(turnRefs[0])}
              className="font-mono text-[10px] font-semibold text-teal-600 underline-offset-2 hover:underline dark:text-teal-400 cursor-pointer"
              title={jumpTitle}
            >
              №{f.seq}
            </button>
          ) : (
            <span
              className="font-mono text-[10px] font-semibold text-muted-foreground"
              title={turnRefs.length === 0 ? "段级缺席（无具体 turn 可跳转）" : undefined}
            >
              №{f.seq}
            </span>
          )
        ) : null}
        <Badge variant="gray" className="h-5 px-1 text-[10px]">
          {CATEGORY_LABEL[f.category] ?? f.category}
        </Badge>
        <Badge variant="outline" className="h-5 px-1 text-[10px] uppercase" title="指令的规则类型">
          {srcs.join(" + ")}
        </Badge>
        {f.derived_from_transcript && (
          <span className="text-[10px] text-muted-foreground" title="由 transcript 反推出的指令">
            ↺ 反推
          </span>
        )}
        {f.related && f.related.length > 0 && (
          <span className="flex items-center gap-0.5 text-[10px] text-muted-foreground">
            ↔ 同源
            {f.related.map((r) => {
              const n = String(r).replace("#", "")
              const num = parseInt(n, 10)
              return (
                <button
                  key={r}
                  type="button"
                  className="text-blue-600 underline-offset-2 hover:underline dark:text-blue-400"
                  onClick={() => num >= 0 && onJumpToSeq?.(num)}
                >
                  №{n}
                </button>
              )
            })}
          </span>
        )}
        {noise && (
          <span
            className={`rounded border px-1 text-[10px] ${noise.cls}`}
            title={(f.noise_reasons ?? []).join("; ")}
          >
            {noise.label}
          </span>
        )}
        <span
          className={`ml-auto text-[10px] ${METHOD_TAG[f.method]?.cls ?? "text-muted-foreground"}`}
          title={METHOD_TAG[f.method]?.hint ?? ""}
        >
          {METHOD_TAG[f.method]?.label ?? "—"}
        </span>
      </div>
      <div className="mt-1 whitespace-pre-line font-medium leading-snug">
        {mainInstruction}
      </div>
      {showAnchor && (
        <div className="mt-0.5 text-[10px] text-muted-foreground/70 leading-snug">
          判定锚点:{f.instruction_text}
        </div>
      )}
      <div className="mt-0.5 text-muted-foreground leading-snug">{f.evidence}</div>
      <div className="mt-0.5 text-[10px] text-muted-foreground/80">@ {f.transcript}</div>
    </div>
  )
}
