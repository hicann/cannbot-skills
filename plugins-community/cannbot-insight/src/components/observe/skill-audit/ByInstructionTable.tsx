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
import type { InstructionAggregate } from "@/lib/skill-eval-audit-types"
import { CATEGORY_LABEL, NOISE_LABELS } from "./verdictConfig"

/**
 * 多 transcript 的指令级聚合表（移植自 audit-report.html）。FAIL 与 N/A 语义不同，
 * 拆两张表：违规表看 fail% / fail/total + 噪声判别；不适用表看 na/total + 哪几段不适用。
 * 全 PASS 不列。cannbot-insight 实际单 transcript 多数无 by_instruction，此时整块不显。
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

export function ByInstructionTable({
  byInstruction,
  nTrans,
}: {
  byInstruction: InstructionAggregate[]
  nTrans: number
}) {
  const failRows = byInstruction.filter((a) => a.fail_count > 0)
  const naRows = byInstruction
    .filter((a) => a.na_count > 0)
    .sort((x, y) => (y.na_count ?? 0) - (x.na_count ?? 0))

  return (
    <div className="space-y-4">
      {failRows.length > 0 && (
        <div>
          <h3 className="mb-1.5 text-sm font-semibold">
            违规 FAIL（跨 {nTrans} transcript）
          </h3>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>明细 #</TableHead>
                <TableHead>fail%</TableHead>
                <TableHead>指令</TableHead>
                <TableHead>类别</TableHead>
                <TableHead>fail/total</TableHead>
                <TableHead>噪声判别</TableHead>
                <TableHead>fail in transcripts</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {failRows.map((a, i) => {
                const pct = Math.round((a.fail_rate ?? 0) * 100)
                const cls = (a.fail_rate ?? 0) >= 0.5 ? "text-red-600 dark:text-red-400 font-medium" : "text-orange-700 dark:text-orange-400"
                const seqs = (a.fail_seqs ?? []).map((s) => `#${s}`).join(", ")
                return (
                  <TableRow key={i}>
                    <TableCell className="font-mono text-[10px] text-muted-foreground">{seqs}</TableCell>
                    <TableCell className={cls + " tabular-nums"}>{pct}%</TableCell>
                    <TableCell className="whitespace-normal">{a.instruction_text}</TableCell>
                    <TableCell className="text-muted-foreground">{CATEGORY_LABEL[a.category] ?? a.category}</TableCell>
                    <TableCell className="tabular-nums">{a.fail_count}/{a.total}</TableCell>
                    <TableCell><NoiseCell a={a} /></TableCell>
                    <TableCell className="text-[10px] text-muted-foreground">{(a.fail_in_transcripts ?? []).join(", ")}</TableCell>
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
            场景不适用 N/A（跨 {nTrans} transcript）
          </h3>
          <Table className="text-xs">
            <TableHeader>
              <TableRow>
                <TableHead>明细 #</TableHead>
                <TableHead>na/total</TableHead>
                <TableHead>指令</TableHead>
                <TableHead>类别</TableHead>
                <TableHead>na in transcripts</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {naRows.map((a, i) => {
                const seqs = (a.na_seqs ?? []).map((s) => `#${s}`).join(", ")
                return (
                  <TableRow key={i}>
                    <TableCell className="font-mono text-[10px] text-muted-foreground">{seqs}</TableCell>
                    <TableCell className="tabular-nums text-orange-700 dark:text-orange-400 font-medium">{a.na_count}/{a.total}</TableCell>
                    <TableCell className="whitespace-normal">{a.instruction_text}</TableCell>
                    <TableCell className="text-muted-foreground">{CATEGORY_LABEL[a.category] ?? a.category}</TableCell>
                    <TableCell className="text-[10px] text-muted-foreground">{(a.na_in_transcripts ?? []).join(", ")}</TableCell>
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
