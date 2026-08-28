// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * 对账报告的展示映射 + 纯函数（排序 / 过滤 / 分组），移植自 sift 的
 * reporting/templates/audit-report.html 的 JS 渲染逻辑，确保原生页与原 HTML
 * 行为一致。纯函数独立可测（无 React 依赖）。
 */
import type {
  AuditCategory,
  AuditFinding,
  AuditMethod,
  AuditVerdict,
} from "@/lib/sift-audit-types"

/** Badge variant + 图标 + 中文标签，五态 verdict 一致表（汇总徽章 + 明细共用）。 */
export const VERDICT_DISPLAY: Record<
  AuditVerdict,
  { variant: "green" | "red" | "orange" | "gray"; icon: string; label: string; hint: string }
> = {
  PASS: { variant: "green", icon: "✓", label: "通过", hint: "已判明通过" },
  FAIL: { variant: "red", icon: "✗", label: "失败", hint: "违规" },
  "N/A": { variant: "orange", icon: "⊘", label: "不适用", hint: "规则存在但本次 transcript 场景不触发" },
  UNRESOLVED: { variant: "gray", icon: "○", label: "未判定", hint: "LLM 调用或解析失败/未提及，可重跑救回" },
  INDETERMINATE: { variant: "red", icon: "?", label: "存疑", hint: "证据相互矛盾或关键缺失，重跑无益" },
}

export const ALL_VERDICTS: AuditVerdict[] = ["PASS", "FAIL", "N/A", "UNRESOLVED", "INDETERMINATE"]

/** method（判定方法）→ 标签 + 颜色 class。与 verdict 正交：回答"怎么查的"。 */
export const METHOD_TAG: Record<AuditMethod, { label: string; cls: string; hint: string }> = {
  PROGRAMMATIC: { label: "程序", cls: "text-emerald-600 dark:text-emerald-400", hint: "程序对账（tool_trace 命中 / 子串匹配），不调 LLM" },
  LLM: { label: "LLM", cls: "text-muted-foreground", hint: "LLM 给出结论" },
}

/** FAIL 噪声判别徽章（仅 FAIL 标）。reasons 进 title hover。 */
export const NOISE_LABELS: Record<string, { label: string; cls: string; hint: string }> = {
  tool_noise: { label: "疑似工具噪声", cls: "text-muted-foreground border-muted-foreground/40", hint: "工具结果噪声导致的误判" },
  skill_issue: { label: "疑似 skill 问题", cls: "text-red-600 dark:text-red-400 border-red-500/40", hint: "疑似 skill 本身的问题" },
  uncertain: { label: "待确认", cls: "text-orange-700 dark:text-orange-400 border-orange-500/40", hint: "待人工确认" },
}

/** category → 中文标签。 */
export const CATEGORY_LABEL: Record<AuditCategory, string> = {
  conditional: "条件性",
  prohibition: "禁令",
  step: "步骤",
  output: "输出",
}

/**
 * findings 排序权重：FAIL → UNRESOLVED → INDETERMINATE → N/A → PASS。
 * 与 audit-report.html 的 ORDER 一致（FAIL 优先，问题在前）。
 */
const VERDICT_ORDER: Record<AuditVerdict, number> = {
  FAIL: 0,
  UNRESOLVED: 1,
  INDETERMINATE: 2,
  "N/A": 3,
  PASS: 4,
}

/** 按 verdict 优先级排序（FAIL 在前），稳定排序保留原 seq 顺序。 */
export function sortFindings(findings: AuditFinding[]): AuditFinding[] {
  return findings
    .slice()
    .sort((a, b) => (VERDICT_ORDER[a.verdict] ?? 9) - (VERDICT_ORDER[b.verdict] ?? 9))
}

export interface FindingFilter {
  /** 选中的 verdict 集合；空集合 = 不过滤 verdict（全显）。 */
  verdicts: Set<AuditVerdict>
  /** 选中的 category 集合；空 = 全显。 */
  categories: Set<AuditCategory>
  /** method 过滤：undefined = 全显。 */
  method: AuditMethod | undefined
  /** 在 instruction_text + evidence 里搜（不区分大小写）。空 = 不过滤。 */
  text: string
}

export const EMPTY_FILTER: FindingFilter = {
  verdicts: new Set(),
  categories: new Set(),
  method: undefined,
  text: "",
}

/** 按 filter 过滤 findings。空集合 / 空串 = 该维度不过滤。 */
export function filterFindings(findings: AuditFinding[], f: FindingFilter): AuditFinding[] {
  const q = f.text.trim().toLowerCase()
  return findings.filter((fd) => {
    if (f.verdicts.size > 0 && !f.verdicts.has(fd.verdict)) return false
    if (f.categories.size > 0 && !f.categories.has(fd.category)) return false
    if (f.method !== undefined && fd.method !== f.method) return false
    if (q) {
      const hay = (fd.instruction_text + " " + fd.evidence).toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
}

/**
 * 多 transcript 按指令分组（同 instruction_text 聚到一组），保留原顺序。
 * 返回值每项 = 该指令下所有 findings（已按 verdict 排序）。
 */
export function groupFindingsByInstruction(
  findings: AuditFinding[],
): Array<{ key: string; findings: AuditFinding[] }> {
  const order: string[] = []
  const map = new Map<string, AuditFinding[]>()
  for (const fd of findings) {
    const key = fd.instruction_text || "(空)"
    if (!map.has(key)) {
      map.set(key, [])
      order.push(key)
    }
    map.get(key)!.push(fd)
  }
  return order.map((key) => ({ key, findings: sortFindings(map.get(key)!) }))
}

/**
 * 单 transcript 按 verdict 分组(顺序 FAIL → UNRESOLVED → INDETERMINATE → N/A → PASS,
 * 与 VERDICT_ORDER 一致;未知 verdict 排末尾),跳过空组。每组 findings 已按 verdict 排序。
 * 用于单 transcript 明细:违规/未判定/存疑/不适用/通过 分节前置,替代扁平列表——
 * 让"哪几条违规、哪几条不适用"一眼可见,不必滚扁平列表找。
 */
export function groupFindingsByVerdict(
  findings: AuditFinding[],
): Array<{ verdict: AuditVerdict; findings: AuditFinding[] }> {
  const map = new Map<AuditVerdict, AuditFinding[]>()
  for (const fd of findings) {
    const arr = map.get(fd.verdict) ?? []
    arr.push(fd)
    map.set(fd.verdict, arr)
  }
  const verdicts = [...map.keys()].sort(
    (a, b) => (VERDICT_ORDER[a] ?? 9) - (VERDICT_ORDER[b] ?? 9),
  )
  return verdicts.map((v) => ({ verdict: v, findings: sortFindings(map.get(v)!) }))
}

/** 是否多 transcript（决定 by_instruction 表 + 分组显示）。 */
export function isMultiTranscript(transcripts: string[]): boolean {
  return transcripts.length > 1
}

/** findings 里是否有任何非 PASS（决定分组默认展开）。 */
export function hasNonPass(findings: AuditFinding[]): boolean {
  return findings.some((f) => f.verdict !== "PASS")
}
