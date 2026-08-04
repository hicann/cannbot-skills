"use client"
// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { cn } from "@/lib/utils"
import type { AuditCategory, AuditFinding, AuditMethod, AuditVerdict } from "@/lib/skill-eval-audit-types"
import { ALL_VERDICTS, CATEGORY_LABEL, METHOD_TAG, VERDICT_DISPLAY, type FindingFilter } from "./verdictConfig"

const ALL_CATEGORIES: AuditCategory[] = ["conditional", "prohibition", "step", "output"]
const METHODS: Array<{ value: AuditMethod | undefined; label: string }> = [
  { value: undefined, label: "全部" },
  { value: "PROGRAMMATIC", label: METHOD_TAG.PROGRAMMATIC.label },
  { value: "LLM", label: METHOD_TAG.LLM.label },
]

function toggleInSet<T>(set: Set<T>, v: T): Set<T> {
  const next = new Set(set)
  if (next.has(v)) next.delete(v)
  else next.add(v)
  return next
}

/** 统计每个 verdict 的 finding 数（供 toggle 上显示分布）。 */
export function verdictCounts(findings: AuditFinding[]): Record<AuditVerdict, number> {
  const c = { PASS: 0, FAIL: 0, "N/A": 0, UNRESOLVED: 0, INDETERMINATE: 0 } as Record<AuditVerdict, number>
  for (const f of findings) c[f.verdict] = (c[f.verdict] ?? 0) + 1
  return c
}

export function FindingFilters({
  filter,
  counts,
  onChange,
}: {
  filter: FindingFilter
  counts: Record<AuditVerdict, number>
  onChange: (f: FindingFilter) => void
}) {
  return (
    <div className="space-y-2 rounded-lg border border-border bg-muted/30 p-2.5 text-xs">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="text-muted-foreground">结果</span>
        {ALL_VERDICTS.map((v) => {
          const vd = VERDICT_DISPLAY[v]
          const active = filter.verdicts.size === 0 || filter.verdicts.has(v)
          return (
            <label key={v} className="flex items-center gap-1 cursor-pointer select-none">
              <Checkbox
                checked={active}
                onCheckedChange={() => {
                  // 空集合 = 全显；点掉一个 → 该 verdict 进排除集；再点全显时复位到空集合
                  if (filter.verdicts.size === 0) {
                    onChange({ ...filter, verdicts: new Set(ALL_VERDICTS.filter((x) => x !== v)) })
                  } else {
                    onChange({ ...filter, verdicts: toggleInSet(filter.verdicts, v) })
                  }
                }}
              />
              <span className={cn("font-medium", active && "text-foreground")} style={{}}>
                <span className={cn(vd.variant === "green" && "text-emerald-600 dark:text-emerald-400", vd.variant === "red" && "text-red-600 dark:text-red-400", vd.variant === "orange" && "text-orange-700 dark:text-orange-400", vd.variant === "gray" && "text-muted-foreground")}>
                  {vd.icon} {vd.label}
                </span>
                <span className="ml-0.5 tabular-nums text-muted-foreground">{counts[v] ?? 0}</span>
              </span>
            </label>
          )
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="text-muted-foreground">类别</span>
        {ALL_CATEGORIES.map((c) => {
          const active = filter.categories.size === 0 || filter.categories.has(c)
          return (
            <label key={c} className="flex items-center gap-1 cursor-pointer select-none">
              <Checkbox
                checked={active}
                onCheckedChange={() => {
                  if (filter.categories.size === 0) {
                    onChange({ ...filter, categories: new Set(ALL_CATEGORIES.filter((x) => x !== c)) })
                  } else {
                    onChange({ ...filter, categories: toggleInSet(filter.categories, c) })
                  }
                }}
              />
              <span className={cn("font-medium", active ? "text-foreground" : "text-muted-foreground")}>
                {CATEGORY_LABEL[c]}
              </span>
            </label>
          )
        })}
      </div>

      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5">
        <span className="text-muted-foreground">方法</span>
        {METHODS.map((m) => {
          const active = filter.method === m.value
          return (
            <button
              key={m.label}
              type="button"
              onClick={() => onChange({ ...filter, method: m.value })}
              className={cn(
                "rounded border px-1.5 py-0.5 transition-colors",
                active
                  ? "border-primary bg-primary/10 text-foreground"
                  : "border-border text-muted-foreground hover:bg-muted",
              )}
            >
              {m.label}
            </button>
          )
        })}
        <Input
          placeholder="搜指令 / 证据…"
          value={filter.text}
          onChange={(e) => onChange({ ...filter, text: (e.target as HTMLInputElement).value })}
          className="ml-auto h-6 w-48 text-xs"
        />
      </div>
    </div>
  )
}
