"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import type { InstructionAggregate } from "@/lib/sift-audit-types"
import { CATEGORY_LABEL, NOISE_LABELS } from "./verdictConfig"

/**
 * 指令级聚合概要表（移植自 audit-report.html ③ 区块，单/多 transcript 都显）。FAIL 与 N/A
 * 语义不同，拆两张表。多 transcript 带跨会话列（fail% / fail/total / in transcripts）；
 * 单 transcript 无跨会话可比，精简成「明细 № / 指令 / 类别（+ FAIL 噪声判别）」。
 * 全无 fail → 不列违规表；全无 na → 不列不适用表。
 */
function NoiseCell({ a }: { a: InstructionAggregate }) {
  if (!a.noise_label) return <span className="text-muted-foreground">—</span>
  const nl = NOISE_LABELS[a.noise_label]
  if (!nl) return <span className="text-muted-foreground">{a.noise_label}</span>
  return (
    <span className={`rounded border px-1 text-[10px] ${nl.cls}`} title={(a.noise_reasons ?? []).join("; ")}>
      {nl.label}
    </span>
  )
}

/** 明细 #N 链接簇:点 #N 跳到对应 finding(复用 onJumpToSeq,与 FindingCard related 同机制)。 */
function SeqLinks({ seqs, onJumpToSeq }: { seqs: number[]; onJumpToSeq?: (seq: number) => void }) {
  if (!seqs.length) return <span className="text-muted-foreground">—</span>
  return (
    <>
      {seqs.map((s, i) => (
        <span key={s}>
          {i > 0 && ", "}
          <button
            type="button"
            className="font-mono text-blue-600 underline-offset-2 hover:underline dark:text-blue-400"
            onClick={() => onJumpToSeq?.(s)}
          >
            #{s}
          </button>
        </span>
      ))}
    </>
  )
}

export function ByInstructionTable({
  byInstruction,
  nTrans,
  onJumpToSeq,
}: {
  byInstruction: InstructionAggregate[]
  nTrans: number
  onJumpToSeq?: (seq: number) => void
}) {
  const multi = nTrans > 1
  const failRows = byInstruction.filter((a) => a.fail_count > 0)
  const naRows = byInstruction
    .filter((a) => a.na_count > 0)
    .sort((x, y) => (y.na_count ?? 0) - (x.na_count ?? 0))

  return (
    <div className="space-y-4">
      {failRows.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-sm font-semibold">
            {multi ? `违规 FAIL（跨 ${nTrans} transcript）` : "违规 FAIL"}
          </h3>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>明细 №</TableHead>
                {multi && <TableHead>fail%</TableHead>}
                <TableHead>指令</TableHead>
                <TableHead>类别</TableHead>
                {multi && <TableHead>fail/total</TableHead>}
                <TableHead>噪声判别</TableHead>
                {multi && <TableHead>fail in transcripts</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {failRows.map((a, i) => {
                const pct = Math.round((a.fail_rate ?? 0) * 100)
                const cls = (a.fail_rate ?? 0) >= 0.5 ? "text-red-600 dark:text-red-400 font-medium" : "text-orange-700 dark:text-orange-400"
                return (
                  <TableRow key={i}>
                    <TableCell className="text-[10px]">
                      <SeqLinks seqs={a.fail_seqs ?? []} onJumpToSeq={onJumpToSeq} />
                    </TableCell>
                    {multi && <TableCell className={cls + " tabular-nums"}>{pct}%</TableCell>}
                    <TableCell className="whitespace-normal" title={a.instruction_text}>
                      {a.headline || a.source_excerpt || a.instruction_text}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{CATEGORY_LABEL[a.category] ?? a.category}</TableCell>
                    {multi && <TableCell className="tabular-nums">{a.fail_count}/{a.total}</TableCell>}
                    <TableCell><NoiseCell a={a} /></TableCell>
                    {multi && <TableCell className="text-[10px] text-muted-foreground">{(a.fail_in_transcripts ?? []).join(", ")}</TableCell>}
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      )}

      {naRows.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-sm font-semibold">
            {multi ? `场景不适用 N/A（跨 ${nTrans} transcript）` : "场景不适用 N/A"}
          </h3>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>明细 №</TableHead>
                {multi && <TableHead>na/total</TableHead>}
                <TableHead>指令</TableHead>
                <TableHead>类别</TableHead>
                {multi && <TableHead>na in transcripts</TableHead>}
              </TableRow>
            </TableHeader>
            <TableBody>
              {naRows.map((a, i) => {
                return (
                  <TableRow key={i}>
                    <TableCell className="text-[10px]">
                      <SeqLinks seqs={a.na_seqs ?? []} onJumpToSeq={onJumpToSeq} />
                    </TableCell>
                    {multi && <TableCell className="tabular-nums text-orange-700 dark:text-orange-400 font-medium">{a.na_count}/{a.total}</TableCell>}
                    <TableCell className="whitespace-normal" title={a.instruction_text}>
                      {a.headline || a.source_excerpt || a.instruction_text}
                    </TableCell>
                    <TableCell className="text-muted-foreground">{CATEGORY_LABEL[a.category] ?? a.category}</TableCell>
                    {multi && <TableCell className="text-[10px] text-muted-foreground">{(a.na_in_transcripts ?? []).join(", ")}</TableCell>}
                  </TableRow>
                )
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  )
}
