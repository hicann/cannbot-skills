// Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

/**
 * AuditReport 的 TS 类型固化，对应 skill-eval 的
 * `skill_eval/models/audit_models.py`（AuditReport / AuditFinding /
 * InstructionAggregate）。后端 spawn skill-eval 跑出的 audit-report.json
 * 即此 shape，经 NDJSON result 事件原样回传前端（见 skill-eval-runner.ts）。
 *
 * 判定模型两个正交维度：
 * - verdict（结果）五态：PASS / FAIL / N/A / UNRESOLVED（未判定，可重跑）/
 *   INDETERMINATE（存疑，重跑无益）。
 * - method（方法）两态：PROGRAMMATIC（程序对账，不调 LLM）/ LLM。
 */

export type AuditVerdict = "PASS" | "FAIL" | "N/A" | "UNRESOLVED" | "INDETERMINATE"

export type AuditMethod = "PROGRAMMATIC" | "LLM"

export type AuditCategory = "conditional" | "prohibition" | "step" | "output"

/** 单条指令的对账结果（对应 pydantic AuditFinding）。 */
export interface AuditFinding {
  instruction_text: string
  /** 声明原文里所属 bullet 的完整文本（含被切的主语上下文）；显示用、判定不读。可缺省。 */
  source_excerpt?: string
  /** 原文 bullet 的人话标题（去噪+补主语+压一句）；显示优先用它，判定不读。可缺省。 */
  headline?: string
  /** 指令的规则类型，如 "CONSTRAINT" / "STEP"（InstructionSource 的值，非 method）。 */
  source: string
  verdict: AuditVerdict
  method: AuditMethod
  category: AuditCategory
  /** 中文，为何这个 verdict（INDETERMINATE 时即 indeterminate_reason）。 */
  evidence: string
  /** 该 finding 对应的 transcript 文件名。 */
  transcript: string
  derived_from_transcript?: boolean
  /** 去重后编排的序号 #1..#N（related 引用 + 报告定位用）。 */
  seq?: number
  /** 同文去重后合并的所有来源；单条 = [source]。 */
  sources?: string[]
  /** 同行为跨类（Case B）关联的其它 finding 序号，如 ["#9"]。 */
  related?: string[]
  /** FAIL 噪声判别标签（仅 FAIL 有）；非 FAIL 为 null。 */
  noise_label?: string | null
  noise_reasons?: string[]
  /** 该 finding 证据涉及的对话 turn 序号（1-based turnIndex），点击可跳转到 Turns tab 对应 turn。 */
  turn_refs?: number[]
}

/** 单条指令跨 transcript 的对账聚合（对应 InstructionAggregate）。 */
export interface InstructionAggregate {
  instruction_text: string
  /** 所属 bullet 完整原文（含主语）；显示用、判定不读。可缺省。 */
  source_excerpt?: string
  /** 人话标题；显示优先用它，判定不读。可缺省。 */
  headline?: string
  source: string
  category: AuditCategory
  pass_count: number
  fail_count: number
  na_count: number
  unresolved_count?: number
  indeterminate_count?: number
  total: number
  fail_rate: number
  fail_in_transcripts: string[]
  na_in_transcripts?: string[]
  fail_seqs?: number[]
  na_seqs?: number[]
  noise_label?: string | null
  noise_reasons?: string[]
}

/** audit 各阶段（extract/refine/headline/rebuild/judge/postprocess）的耗时 + token（对应 pydantic stage_breakdown）。 */
export interface StageBreakdownRow {
  stage: string
  seconds: number
  input_tokens: number
  output_tokens: number
  /** cache 命中读取的 token（仅 LLM 阶段 refine/headline/judge 才有；对齐 skill-eval stage_breakdown 透传）。 */
  cache_read_input_tokens?: number
  calls: number
  /** refine/headline 在 calls=0 时："缓存命中"（开关开、跑了秒回）/ "未启用"（开关关）；其余阶段空。 */
  note?: string
}

/** 一个 skill 一次 audit 的完整报告（对应 pydantic AuditReport）。 */
export interface AuditReport {
  skill_name: string
  transcripts: string[]
  transcript_sources?: string[]
  /** {total, pass, fail, na, unresolved, indeterminate}。 */
  summary: {
    total: number
    pass: number
    fail: number
    na: number
    unresolved?: number
    indeterminate?: number
  }
  findings: AuditFinding[]
  by_instruction?: InstructionAggregate[]
  warnings?: string[]
  /** refine（LLM 精筛伪指令）砍掉的条数。 */
  refine_dropped_count?: number
  /** 本次 audit 所有 LLM 判定调用的 token 用量合计（input/output/cache/cost/calls）。 */
  token_usage?: Record<string, number>
  /** audit 墙钟（秒）；stage_breakdown 各阶段秒数之和 ≈ 此值。 */
  duration_seconds?: number
  /** 各阶段耗时 + token，诊断慢在哪；refine/headline calls=0 时 note 标缓存命中/未启用。 */
  stage_breakdown?: StageBreakdownRow[]
}

/** store 侧物化的 result：report + 可选原始 HTML（逃生口用）。 */
export type SkillAuditStoredResult = AuditReport & { _html?: string }
