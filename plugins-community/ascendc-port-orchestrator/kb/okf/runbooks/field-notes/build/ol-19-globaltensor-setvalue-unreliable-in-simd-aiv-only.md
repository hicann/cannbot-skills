---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "GlobalTensor::SetValue() unreliable in SIMD AIV-only mode"
description: "`GlobalTensor<float>::SetValue(index, value)` in AIV-only SIMD mode silently drops ~80% of writes. Discovered in E11: SG sorted backward wrote grad_weight[edge_id] via SetValue, 1633/2048 values were"
phenomenon: build_failure
signal:
  - "writing scalar values to GM in SIMD (KERNEL_TYPE_AIV_ONLY) kernel"
confidence: single_run
original_id: OL-19
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-19]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
writing scalar values to GM in SIMD (KERNEL_TYPE_AIV_ONLY) kernel

## 教训 / 根因
`GlobalTensor<float>::SetValue(index, value)` in AIV-only SIMD mode silently drops ~80% of writes. Discovered in E11: SG sorted backward wrote grad_weight[edge_id] via SetValue, 1633/2048 values were zero while the kernel logic was proven correct (grad_in via DataCopy was PASS). Root cause unknown — may be CANN 9.0.T501 or bisheng AIV scalar GM write bug.

## 证据
E11 debugging session and on-device ablation

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-19（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
