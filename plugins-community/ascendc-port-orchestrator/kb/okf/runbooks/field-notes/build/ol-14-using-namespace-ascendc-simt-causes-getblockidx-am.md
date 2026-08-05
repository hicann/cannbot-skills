---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`using namespace AscendC::Simt;` causes GetBlockIdx ambiguity"
description: "CANN has TWO `GetBlockIdx()` — `AscendC::Simt::GetBlockIdx()` (int32_t) and basic_api `GetBlockIdx()` (int64_t). Using `using namespace AscendC::Simt;` brings both into scope → compile error \"call to"
phenomenon: build_failure
signal:
  - "when writing AscendC SIMT kernel headers"
confidence: single_run
original_id: OL-14
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-14]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when writing AscendC SIMT kernel headers

## 教训 / 根因
CANN has TWO `GetBlockIdx()` — `AscendC::Simt::GetBlockIdx()` (int32_t) and basic_api `GetBlockIdx()` (int64_t). Using `using namespace AscendC::Simt;` brings both into scope → compile error "call to GetBlockIdx is ambiguous". Fix: use ONLY `using namespace AscendC;` (without `::Simt`). Dispatchers already use `Simt::VF_CALL` with qualified prefix.

## 证据
SG forward generated kernel, 20 ambiguity errors. Pooling (only `using namespace AscendC;`) compiled fine.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-14（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
