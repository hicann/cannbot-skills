"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { useMemo, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import type { SkillAuditResult } from "@/lib/skill-audit-job"
import {
  EMPTY_FILTER,
  filterFindings,
  groupFindingsByInstruction,
  hasNonPass,
  isMultiTranscript,
  sortFindings,
  VERDICT_DISPLAY,
  type FindingFilter,
} from "./verdictConfig"
import { FindingFilters, verdictCounts } from "./FindingFilters"
import { FindingCard } from "./FindingCard"
import { ByInstructionTable } from "./ByInstructionTable"

/**
 * 对账报告原生视图（替代 iframe 嵌 skill-eval HTML）。数据 = SkillAuditResult
 * （AuditReport + 可选 _html 逃生口）。移植 audit-report.html 的全部区块：header →
 * warnings → by_instruction（多 transcript）→ 过滤 → findings（分组/扁平）→ _html 逃生口。
 */
export function AuditReportView({
  result,
  onRerun,
}: {
  result: SkillAuditResult
  onRerun?: () => void
}) {
  const { _html, ...report } = result
  const [filter, setFilter] = useState<FindingFilter>(EMPTY_FILTER)

  const findings = useMemo(() => report.findings ?? [], [report.findings])
  const multi = isMultiTranscript(report.transcripts ?? [])
  const counts = useMemo(() => verdictCounts(findings), [findings])
  const derivedCount = useMemo(() => findings.filter((f) => f.derived_from_transcript).length, [findings])
  const refineDropped = report.refine_dropped_count ?? 0
  const filtered = useMemo(() => {
    const f = filterFindings(findings, filter)
    return multi ? groupFindingsByInstruction(f) : null
  }, [findings, filter, multi])
  const flat = useMemo(() => (multi ? null : sortFindings(filterFindings(findings, filter))), [findings, filter, multi])

  // transcript 名 → 来源（cc/opencode/…），同序 zip
  const srcOf = useMemo(() => {
    const m = new Map<string, string>()
    ;(report.transcripts ?? []).forEach((n, i) => m.set(n, (report.transcript_sources ?? [])[i] ?? ""))
    return m
  }, [report.transcripts, report.transcript_sources])

  const jumpToSeq = (seq: number) => {
    if (typeof document === "undefined") return
    document.getElementById(`finding-${seq}`)?.scrollIntoView({ behavior: "smooth", block: "center" })
  }

  return (
    <div className="space-y-3">
      {/* ① header */}
      <div className="space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="text-base font-semibold">{report.skill_name}</h2>
          {onRerun && (
            <Button size="xs" variant="outline" onClick={onRerun}>重跑</Button>
          )}
        </div>
        {(report.transcripts ?? []).length > 0 && (
          <div className="text-[11px] text-muted-foreground break-all">
            {(report.transcripts ?? []).map((n, i) => (
              <span key={i}>
                {i > 0 && " · "}
                {n}
                {srcOf.get(n) && (
                  <Badge variant="outline" className="ml-1 h-3.5 px-1 text-[9px] uppercase">{srcOf.get(n)}</Badge>
                )}
              </span>
            ))}
          </div>
        )}
        <details className="text-[11px] text-muted-foreground">
          <summary className="cursor-pointer select-none w-fit">判定图例</summary>
          <div className="mt-1 space-y-0.5 pl-3">
            {(["PASS", "FAIL", "N/A", "UNRESOLVED", "INDETERMINATE"] as const).map((v) => (
              <div key={v}>
                <span className="font-medium">{VERDICT_DISPLAY[v].icon} {VERDICT_DISPLAY[v].label}</span>
                ：{VERDICT_DISPLAY[v].hint}
              </div>
            ))}
            <div className="mt-0.5"><span className="font-medium">程序</span>：程序对账（tool_trace 命中/子串匹配），不调 LLM</div>
            <div><span className="font-medium">LLM</span>：LLM 给出结论（含未判定/存疑）</div>
          </div>
        </details>
      </div>

      {/* ② warnings */}
      {(report.warnings ?? []).length > 0 && (
        <div className="rounded-lg border border-red-500/40 bg-red-500/5 p-2.5">
          <div className="text-sm font-semibold text-red-700 dark:text-red-400">⚠️ 警告</div>
          <ul className="mt-1 list-disc space-y-0.5 pl-5 text-xs">
            {(report.warnings ?? []).map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}

      {/* ③ by_instruction（多 transcript 才显） */}
      {multi && (report.by_instruction ?? []).length > 0 && (
        <ByInstructionTable byInstruction={report.by_instruction ?? []} nTrans={(report.transcripts ?? []).length} />
      )}

      {/* ④ 过滤 + findings */}
      {findings.length > 0 ? (
        <>
          <FindingFilters filter={filter} counts={counts} onChange={setFilter} />
          <div className="text-[11px] text-muted-foreground">
            显示 <span className="tabular-nums text-foreground">{multi ? (filtered ?? []).reduce((n, g) => n + g.findings.length, 0) : (flat ?? []).length}</span> / {findings.length} 条
            {derivedCount > 0 && <span className="ml-1" title="由 transcript 反推出的指令">· ↺ {derivedCount} 反推</span>}
            {refineDropped > 0 && (
              <span className="ml-1 text-orange-700 dark:text-orange-400" title="LLM 精筛砍掉 extractor 误抽的示例/占位伪指令">
                · refine 砍 {refineDropped} 伪指令
              </span>
            )}
          </div>
          {multi ? (
            <div className="space-y-1.5">
              {(filtered ?? []).map((g) => (
                <details key={g.key} open={hasNonPass(g.findings)} className="rounded border border-border bg-card/50">
                  <summary className="cursor-pointer select-none px-2.5 py-1.5 text-xs font-medium">
                    {g.key} <span className="text-muted-foreground">({g.findings.length})</span>
                  </summary>
                  <div className="space-y-1 px-2 pb-2">
                    {g.findings.map((f) => <FindingCard key={f.seq ?? f.instruction_text + f.transcript} f={f} onJumpToSeq={jumpToSeq} />)}
                  </div>
                </details>
              ))}
              {(filtered ?? []).length === 0 && <EmptyFilter />}
            </div>
          ) : (
            <div className="space-y-1">
              {(flat ?? []).map((f) => <FindingCard key={f.seq ?? f.instruction_text + f.transcript} f={f} onJumpToSeq={jumpToSeq} />)}
              {(flat ?? []).length === 0 && <EmptyFilter />}
            </div>
          )}
        </>
      ) : (
        <div className="text-xs text-muted-foreground">无 findings。</div>
      )}

      {/* ⑤ _html 逃生口 */}
      {_html ? (
        <div className="border-t border-border pt-2">
          <HtmlEscapeHatch html={_html} />
        </div>
      ) : null}
    </div>
  )
}

function EmptyFilter() {
  return <div className="py-3 text-center text-xs text-muted-foreground">当前过滤无匹配结果。</div>
}

/** 把 _html 物化成 blob URL，供"在新页打开 skill-eval 原始 HTML"逃生口。SSR / 无 blob 支持时返回 null。 */
function HtmlEscapeHatch({ html }: { html: string }) {
  const url = useMemo(() => {
    if (typeof window === "undefined" || typeof URL?.createObjectURL !== "function") return null
    try {
      return URL.createObjectURL(new Blob([html], { type: "text/html" }))
    } catch {
      return null
    }
  }, [html])
  if (!url) return null
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 hover:underline dark:text-blue-400">
      在新页打开 skill-eval 原始 HTML ↗
    </a>
  )
}
