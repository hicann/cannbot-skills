---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Histc bin assignment needs double precision — A5 has no double"
description: "torch.histc uses double precision internally for bin boundary computation. A5 AIV cores only support float32/fp16/bf16. Float kernel gets 1-2 bins wrong at boundaries due to rounding. CPU double fallb"
phenomenon: build_failure
signal:
  - "when implementing histogram / binning operations"
confidence: single_run
original_id: OL-33
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-33]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when implementing histogram / binning operations

## 教训 / 根因
torch.histc uses double precision internally for bin boundary computation. A5 AIV cores only support float32/fp16/bf16. Float kernel gets 1-2 bins wrong at boundaries due to rounding. CPU double fallback is NOT acceptable — need float kernel with adjusted bin formula.

## 证据
Histc V1 (2026-04-09), float kernel had 2% mismatch, CPU fallback got 100% match but is cheating

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-33（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
