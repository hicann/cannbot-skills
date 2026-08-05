---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bf16 Adds with a zero scalar may hit PB-4"
description: "`static_cast<bfloat16_t>(0)` triggers PB-4 (bisheng bf16 scalar cast bug), so Adds does not actually add 0. **Safe alternatives**: (1) fp32 round-trip: `Cast(fp32, bf16, CAST_NONE) → modify → Cast(bf1"
phenomenon: precision_issue
signal:
  - "wanting to use `Adds(dst, src, static_cast<bfloat16_t>(0), count)` for bf16 data movement (bridge / no-op pass-through)"
confidence: single_run
original_id: OL-74
timestamp_inferred: true
tags: [ascendc, precision, ol-74]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
wanting to use `Adds(dst, src, static_cast<bfloat16_t>(0), count)` for bf16 data movement (bridge / no-op pass-through)

## 教训 / 根因
`static_cast<bfloat16_t>(0)` triggers PB-4 (bisheng bf16 scalar cast bug), so Adds does not actually add 0. **Safe alternatives**: (1) fp32 round-trip: `Cast(fp32, bf16, CAST_NONE) → modify → Cast(bf16, fp32, CAST_ROUND)`; (2) for identity copy, use OL-73 pybind clone(); (3) use `Mul(dst, src, one_buf, count)` where one_buf is pre-filled with 1.0; (4) use `Duplicate(dst, src_value, count)` combined with GetValue for scalar extraction (still beware bf16 Cast). **Best practice**: never use bf16 scalar literals in arithmetic; convert to fp32 first for all bf16 math.

## 证据
29_TanhGatedResidualAddBackward first tried `Adds(dst, src, bf16(0))` for copy — compiled but numerically wrong at runtime. Switching to fp32 round-trip + clone() → 50/50 PASS. Related to PB-4. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-74（category=precision，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
