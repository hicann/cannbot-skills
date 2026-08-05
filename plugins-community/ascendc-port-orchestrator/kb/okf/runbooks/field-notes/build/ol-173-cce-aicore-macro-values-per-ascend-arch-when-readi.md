---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`__CCE_AICORE__` macro values per Ascend arch (when reading CANN source)"
description: "`__CCE_AICORE__` is the bisheng compile-time arch macro. Common values seen in CANN source: - `100` = Ascend310 / 310p (older inference cards) - `200` = Ascend910 (original training) - `220` = **Ascen"
phenomenon: build_failure
signal:
  - "encountering `#if __CCE_AICORE__ == NNN` in CANN kernel source"
confidence: single_run
original_id: OL-173
timestamp_inferred: true
tags: [__cce_aicore__, ascendc, arch_compat, ol-173]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
encountering `#if __CCE_AICORE__ == NNN` in CANN kernel source

## 教训 / 根因
`__CCE_AICORE__` is the bisheng compile-time arch macro. Common values seen in CANN source:
  - `100` = Ascend310 / 310p (older inference cards)
  - `200` = Ascend910 (original training)
  - `220` = **Ascend910b / Ascend910_93** (Atlas A2 / A3 — most CANN ops gated here)
  - `300` = (reserved / not seen in our context)
  - `350` = **Ascend950PR / Ascend950DT** (arch35 — our target A5)

## 证据
2026-04-24 op#3 — `cann/ops-nn/optim/advance_step/op_kernel/advance_step.cpp` has `#if __CCE_AICORE__ == 220` gate; no arch35 variant. The kernel uses PB-20 broken pattern → CANN explicitly excluded A5 from binary distribution. See OL-68 Case B.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-173（category=arch_compat，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
