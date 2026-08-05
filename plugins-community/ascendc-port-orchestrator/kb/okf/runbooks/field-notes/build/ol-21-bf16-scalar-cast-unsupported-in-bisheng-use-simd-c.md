---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "bf16 scalar cast unsupported in bisheng — use SIMD Cast + GetValue"
description: "bisheng compiler (CANN 9.0.0) does not support `static_cast<float>(bfloat16_t)` or reverse. Error: \"not support bf16 type cast\". Half (fp16) scalar cast works fine. The correct approach: use SIMD `Cas"
phenomenon: build_failure
signal:
  - "bfloat16_t type in kernel code, bf16 GetValue, static_cast<float>(bf16)"
confidence: single_run
original_id: OL-21
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-21]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
bfloat16_t type in kernel code, bf16 GetValue, static_cast<float>(bf16)

## 教训 / 根因
bisheng compiler (CANN 9.0.0) does not support `static_cast<float>(bfloat16_t)` or reverse. Error: "not support bf16 type cast". Half (fp16) scalar cast works fine. The correct approach: use SIMD `Cast(bf16→float)` vector intrinsic on a buffer, then `GetValue(i)` to read float scalar. See P-P27 pattern.

## 证据
tests/repro/bf16_cast_repro.cpp (6 cases), reg_convert.h API inventory

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-21（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
