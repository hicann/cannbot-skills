// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

// @vitest-environment happy-dom

/**
 * AuditReportView 测试。无 @testing-library/react：纯函数（sort/filter/group/counts）
 * 直接测；叶子组件（Badge-only，不依赖 base-ui）用 renderToStaticMarkup 断言关键文本。
 * AuditReportView 组合由 typecheck + 手动验（同 compare-perf/render-smoke 模式）。
 */
import { describe, it, expect } from "vitest"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import sample from "./data/audit-report-sample.json"
import multi from "./data/audit-report-multi.json"
import type { AuditFinding, AuditReport, SkillAuditStoredResult } from "@/lib/sift-audit-types"
import {
  ALL_VERDICTS,
  CATEGORY_LABEL,
  filterFindings,
  groupFindingsByInstruction,
  groupFindingsByVerdict,
  hasNonPass,
  isMultiTranscript,
  METHOD_TAG,
  sortFindings,
  VERDICT_DISPLAY,
  EMPTY_FILTER,
} from "@/components/observe/skill-audit/verdictConfig"
import { verdictCounts } from "@/components/observe/skill-audit/FindingFilters"
import { FindingCard } from "@/components/observe/skill-audit/FindingCard"

const SAMPLE = sample as unknown as AuditReport
const MULTI = multi as unknown as AuditReport
const FAIL = SAMPLE.findings[0]
const PASS = SAMPLE.findings[2]

describe("verdictConfig 纯函数", () => {
  it("VERDICT_DISPLAY/METHOD_TAG/CATEGORY_LABEL 覆盖全枚举值", () => {
    for (const v of ALL_VERDICTS) expect(VERDICT_DISPLAY[v].label).toBeTruthy()
    expect(METHOD_TAG.PROGRAMMATIC.label).toBe("程序")
    expect(METHOD_TAG.LLM.label).toBe("LLM")
    expect(CATEGORY_LABEL.step).toBe("步骤")
    expect(CATEGORY_LABEL.output).toBe("输出")
  })

  it("sortFindings: FAIL 最前，PASS 最后（稳定）", () => {
    const shuffled = [PASS, FAIL, SAMPLE.findings[4], SAMPLE.findings[1]]
    const sorted = sortFindings(shuffled)
    expect(sorted[0].verdict).toBe("FAIL")
    expect(sorted[sorted.length - 1].verdict).toBe("PASS")
  })

  it("filterFindings: 空 filter 全显", () => {
    expect(filterFindings(SAMPLE.findings, EMPTY_FILTER).length).toBe(SAMPLE.findings.length)
  })

  it("filterFindings: 按 verdict 集合过滤（点掉 PASS → 只留非 PASS）", () => {
    const onlyFail = filterFindings(SAMPLE.findings, {
      ...EMPTY_FILTER,
      verdicts: new Set(["FAIL", "UNRESOLVED", "N/A"]),
    })
    expect(onlyFail.every((f) => f.verdict !== "PASS")).toBe(true)
    expect(onlyFail.length).toBe(4)
  })

  it("filterFindings: 按 category 过滤", () => {
    const steps = filterFindings(SAMPLE.findings, { ...EMPTY_FILTER, categories: new Set(["step"]) })
    expect(steps.every((f) => f.category === "step")).toBe(true)
    expect(steps.length).toBe(2)
  })

  it("filterFindings: 按 method 过滤（只程序对账）", () => {
    const prog = filterFindings(SAMPLE.findings, { ...EMPTY_FILTER, method: "PROGRAMMATIC" })
    expect(prog.every((f) => f.method === "PROGRAMMATIC")).toBe(true)
    expect(prog.length).toBe(2)
  })

  it("filterFindings: 文本搜指令/证据（不区分大小写）", () => {
    const r = filterFindings(SAMPLE.findings, { ...EMPTY_FILTER, text: "tool_trace" })
    expect(r.length).toBe(1)
    expect(r[0].seq).toBe(3)
  })

  it("filterFindings: 多维度叠加", () => {
    const r = filterFindings(SAMPLE.findings, {
      ...EMPTY_FILTER,
      verdicts: new Set(["FAIL"]),
      categories: new Set(["prohibition"]),
    })
    expect(r.length).toBe(1)
    expect(r[0].seq).toBe(2)
  })

  it("verdictCounts: 全枚举键 + 计数正确", () => {
    const c = verdictCounts(SAMPLE.findings as AuditFinding[])
    expect(c.FAIL).toBe(2)
    expect(c.PASS).toBe(2)
    expect(c["N/A"]).toBe(1)
    expect(c.UNRESOLVED).toBe(1)
    expect(c.INDETERMINATE).toBe(0)
  })

  it("groupFindingsByInstruction: 同指令聚合 + 组内 FAIL 优先", () => {
    const groups = groupFindingsByInstruction(MULTI.findings as AuditFinding[])
    expect(groups.length).toBe(2)
    const g = groups[0]
    expect(g.findings.length).toBe(2)
    expect(g.findings[0].verdict).toBe("FAIL")
    expect(g.findings[1].verdict).toBe("PASS")
  })

  it("groupFindingsByVerdict: 按 verdict 分组(FAIL 在前 / PASS 在末),跳空组,总数守恒", () => {
    const groups = groupFindingsByVerdict(SAMPLE.findings as AuditFinding[])
    expect(groups.length).toBeGreaterThan(1)
    // 无空组
    expect(groups.every((g) => g.findings.length > 0)).toBe(true)
    // 顺序由 VERDICT_ORDER 决定:首组 FAIL(问题在前)、末组 PASS
    expect(groups[0].verdict).toBe("FAIL")
    expect(groups[groups.length - 1].verdict).toBe("PASS")
    // 组内 verdict 一致
    expect(groups.every((g) => g.findings.every((f) => f.verdict === g.verdict))).toBe(true)
    // 总数守恒(所有 finding 都进组,无丢)
    expect(groups.reduce((n, g) => n + g.findings.length, 0)).toBe(SAMPLE.findings.length)
  })

  it("isMultiTranscript / hasNonPass", () => {
    expect(isMultiTranscript(MULTI.transcripts)).toBe(true)
    expect(isMultiTranscript(SAMPLE.transcripts)).toBe(false)
    expect(hasNonPass(MULTI.findings as AuditFinding[])).toBe(true)
    expect(hasNonPass([{ ...PASS } as AuditFinding])).toBe(false)
  })
})

describe("FindingCard render smoke", () => {
  it("FAIL finding 渲染 verdict 徽章 + №seq + 指令 + 证据 + transcript + noise + related 跳转按钮", () => {
    const html = renderToStaticMarkup(createElement(FindingCard, { f: FAIL as AuditFinding }))
    expect(html).toContain("✗")
    expect(html).toContain("失败")
    expect(html).toContain("№1")
    expect(html).toContain("改动前必须先读相关源码")
    expect(html).toContain("未先 Read")
    expect(html).toContain("ses_cuid_abc")
    expect(html).toContain("疑似工具噪声")
    expect(html).toContain("↔ 同源")
    expect(html).toContain("№9") // related 按钮（finding 序号，用 № 不用 #）
    expect(html).toContain("LLM") // method
  })

  it("有 turn_refs + onJumpToTurn 时 №seq 渲染为可点击 button（跳 turn_refs[0]）", () => {
    // FAIL finding (seq=1, step) has turn_refs: [2]
    const html = renderToStaticMarkup(
      createElement(FindingCard, { f: FAIL as AuditFinding, onJumpToTurn: () => {} }),
    )
    // №seq 是 button（含 onClick），title 提示跳转（turn 用 #）
    expect(html).toContain("№1")
    expect(html).toContain("跳转到 turn #2")
    // 不再有独立 turn chips 段
    expect(html).not.toContain(">turn<")
  })

  it("turn_refs 多个时 title 列出全部 turn（seq=2 有 turn_refs: [5, 6]）", () => {
    const f2 = SAMPLE.findings[1] as AuditFinding
    const html = renderToStaticMarkup(
      createElement(FindingCard, { f: f2, onJumpToTurn: () => {} }),
    )
    expect(html).toContain("№2")
    expect(html).toContain("共 2 个")
    expect(html).toContain("#5") // turn 序号用 #
    expect(html).toContain("#6")
  })

  it("有 turn_refs 但无 onJumpToTurn 时 №seq 不可点击（灰显 span）", () => {
    const html = renderToStaticMarkup(createElement(FindingCard, { f: FAIL as AuditFinding }))
    expect(html).toContain("№1")
    // 是 span 不是 button
    expect(html).not.toContain("跳转到 turn")
  })

  it("无 turn_refs 的 finding №seq 不可点击（conditional PASS）", () => {
    const html = renderToStaticMarkup(
      createElement(FindingCard, { f: PASS as AuditFinding, onJumpToTurn: () => {} }),
    )
    expect(html).toContain("№3")
    expect(html).not.toContain("跳转到 turn")
  })

  it("N/A finding 渲染场景不触发 hint", () => {
    const na = SAMPLE.findings[4] as AuditFinding
    const html = renderToStaticMarkup(createElement(FindingCard, { f: na }))
    expect(html).toContain("不适用")
    expect(html).toContain("场景不触发")
  })

  it("PASS finding 渲染程序 method 标签", () => {
    const html = renderToStaticMarkup(createElement(FindingCard, { f: PASS as AuditFinding }))
    expect(html).toContain("✓")
    expect(html).toContain("通过")
    expect(html).toContain("程序")
  })
})

describe("AuditReportView render smoke (base-ui 组合)", () => {
  // base-ui Button/Checkbox/Input 在 renderToStaticMarkup 下应产出 markup（SSR 兼容）。
  // verdict 计数已并入 FindingFilters 过滤条（每 toggle 带计数），不再有独立 AuditSummary 行。
  it("渲染 skill_name + warnings + findings + 图例 + 计数行（含 refine 折叠）", async () => {
    const { AuditReportView } = await import("@/components/observe/skill-audit/AuditReportView")
    const result = { ...SAMPLE, findings: SAMPLE.findings as AuditFinding[] } as SkillAuditStoredResult
    const html = renderToStaticMarkup(createElement(AuditReportView, { result }))
    expect(html).toContain("opdef-developer")
    expect(html).toContain("缺少 assistant 收尾 turn")
    expect(html).toContain("改动前必须先读相关源码")
    expect(html).toContain("判定图例")
    expect(html).toContain("显示")
    expect(html).toContain("refine 砍 3 伪指令")
  })

  it("无 _html 时不渲染逃生口链接", async () => {
    const { AuditReportView } = await import("@/components/observe/skill-audit/AuditReportView")
    const result = { ...SAMPLE, findings: SAMPLE.findings as AuditFinding[] } as SkillAuditStoredResult
    const html = renderToStaticMarkup(createElement(AuditReportView, { result }))
    expect(html).not.toContain("原始 HTML")
  })

  it("多 transcript 渲染 by_instruction 表 + 按指令分组", async () => {
    const { AuditReportView } = await import("@/components/observe/skill-audit/AuditReportView")
    const result = { ...MULTI, findings: MULTI.findings as AuditFinding[] } as SkillAuditStoredResult
    const html = renderToStaticMarkup(createElement(AuditReportView, { result }))
    expect(html).toContain("违规 FAIL")
    expect(html).toContain("跨 2 transcript")
    expect(html).toContain("fail%")
    expect(html).toContain("待确认")
  })

  it("单 transcript 也渲染 by_instruction 概要表(去跨会话列 fail%/in transcripts,标题不带「跨」)", async () => {
    const { AuditReportView } = await import("@/components/observe/skill-audit/AuditReportView")
    // SAMPLE 单 transcript、原 by_instruction 空 → 注入 1 fail 行 + 1 na 行
    const result = {
      ...SAMPLE,
      findings: SAMPLE.findings as AuditFinding[],
      by_instruction: [
        { instruction_text: "改动前必须先读相关源码", source: "CONSTRAINT", category: "prohibition",
          pass_count: 0, fail_count: 1, na_count: 0, total: 1, fail_rate: 1.0,
          fail_in_transcripts: ["ses_cuid_abc"], fail_seqs: [1],
          noise_label: "uncertain", noise_reasons: ["r"] },
        { instruction_text: "Use ref: x.md", source: "REFERENCE_USAGE", category: "conditional",
          pass_count: 0, fail_count: 0, na_count: 1, total: 1, fail_rate: 0.0,
          fail_in_transcripts: [], na_in_transcripts: ["ses_cuid_abc"], na_seqs: [4] },
      ],
    } as SkillAuditStoredResult
    const html = renderToStaticMarkup(createElement(AuditReportView, { result }))
    expect(html).toContain("违规 FAIL")
    expect(html).toContain("场景不适用 N/A")
    // 单 transcript 精简:标题不带「跨 N transcript」、无跨会话列
    expect(html).not.toContain("（跨 ")
    expect(html).not.toContain("fail%")
    expect(html).not.toContain("fail/total")
    expect(html).not.toContain("in transcripts")
  })
})
