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
import { cn } from "@/lib/utils"
import type { StageBreakdownRow } from "@/lib/skill-eval-audit-types"

/**
 * 阶段耗时 / token 表（移植自 skill-eval audit-report.html 的 stage-breakdown）。把 audit()
 * 的 6 阶段（extract/refine/headline/rebuild/judge/postprocess）各自墙钟 + token 列出，诊断
 * 慢在哪。类型列标 LLM（refine/headline/judge，橙）vs 程序（其余，灰）——程序阶段 calls/tokens
 * 恒 0，标出来免误读成「漏跑」。refine/headline 在 calls=0 时 note 标「缓存命中/未启用」。
 * 末行合计 = 各列求和（tokens/calls 干净可加；耗时是各阶段顺序非重叠之和 ≈ duration_seconds）。
 */
const STAGE_LABEL: Record<string, string> = {
  extract: "抽取",
  refine: "精筛",
  headline: "标题",
  rebuild: "切片",
  judge: "判定",
  postprocess: "聚合",
}
const STAGE_KIND: Record<string, "LLM" | "程序"> = {
  extract: "程序",
  refine: "LLM",
  headline: "LLM",
  rebuild: "程序",
  judge: "LLM",
  postprocess: "程序",
}

// 各阶段 hover 说明（与 skill-eval audit-report.html 的 STAGE_DESC 对齐；postprocess=聚合阶段直说）。
const STAGE_DESC: Record<string, string> = {
  extract: "从声明(SKILL.md / agent .md)静态正则抽全部候选指令",
  refine: "LLM 精筛:砍 extractor 误抽的示例/占位/描述性伪指令(per-SKILL 缓存)",
  headline: "LLM 把指令改写成人话标题(仅显示、判定不读;per-声明缓存)",
  rebuild: "把 transcript 重建为 SessionDigest + 按作用域切执行段",
  judge: "三类 LLM 判官(条件性/禁令/步骤)逐批判定指令遵循性",
  postprocess: "判定后汇总:跨 transcript 按指令聚合 findings + 噪声判别标注",
}

function fmtTok(i: number, o: number, cr?: number): string {
  return i || o ? `${i} / ${o}${cr ? ` · cache ${cr}` : ""}` : "—"
}

export function StageBreakdownTable({
  stages,
  durationSeconds,
}: {
  stages: StageBreakdownRow[]
  durationSeconds?: number
}) {
  if (!stages.length) return null
  const dur = durationSeconds ?? 0
  const sum = stages.reduce(
    (a, s) => ({
      sec: a.sec + (s.seconds ?? 0),
      in: a.in + (s.input_tokens ?? 0),
      out: a.out + (s.output_tokens ?? 0),
      cr: a.cr + (s.cache_read_input_tokens ?? 0),
      calls: a.calls + (s.calls ?? 0),
    }),
    { sec: 0, in: 0, out: 0, cr: 0, calls: 0 },
  )

  return (
    <details open className="text-[11px]">
      <summary className="cursor-pointer select-none w-fit text-muted-foreground">
        阶段耗时 / token（诊断慢在哪）
      </summary>
      <Table className="mt-1">
        <TableHeader>
          <TableRow className="bg-muted/40 hover:bg-muted/40">
            <TableHead>阶段</TableHead>
            <TableHead>类型</TableHead>
            <TableHead>耗时（占总）</TableHead>
            <TableHead>tokens in/out</TableHead>
            <TableHead>calls</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {stages.map((s) => {
            const sec = s.seconds ?? 0
            const pct = dur && sec ? ` (${Math.round((sec / dur) * 100)}%)` : ""
            const kind = STAGE_KIND[s.stage]
            return (
              <TableRow key={s.stage}>
                <TableCell title={STAGE_DESC[s.stage]}>{STAGE_LABEL[s.stage] ?? s.stage}</TableCell>
                <TableCell
                  className={cn(
                    kind === "LLM"
                      ? "text-orange-700 dark:text-orange-400 font-medium"
                      : "text-muted-foreground",
                  )}
                >
                  {kind ?? ""}
                </TableCell>
                <TableCell className="tabular-nums">
                  {sec.toFixed(1)}s{pct}
                </TableCell>
                <TableCell className="tabular-nums">
                  {fmtTok(s.input_tokens ?? 0, s.output_tokens ?? 0, s.cache_read_input_tokens)}
                </TableCell>
                <TableCell className="tabular-nums">
                  {s.calls ?? 0}
                  {s.note ? (
                    <span className="ml-1 text-[10px] text-muted-foreground">（{s.note}）</span>
                  ) : null}
                </TableCell>
              </TableRow>
            )
          })}
          <TableRow className="border-t-2 font-medium">
            <TableCell>合计</TableCell>
            <TableCell className="text-muted-foreground">—</TableCell>
            <TableCell className="tabular-nums">{sum.sec.toFixed(1)}s</TableCell>
            <TableCell className="tabular-nums">{fmtTok(sum.in, sum.out, sum.cr || undefined)}</TableCell>
            <TableCell className="tabular-nums">{sum.calls}</TableCell>
          </TableRow>
        </TableBody>
      </Table>
    </details>
  )
}
