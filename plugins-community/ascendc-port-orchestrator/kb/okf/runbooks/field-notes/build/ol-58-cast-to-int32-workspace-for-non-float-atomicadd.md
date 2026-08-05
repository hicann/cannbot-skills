---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Cast-to-Int32 workspace for non-float atomicAdd"
description: "CANN's scatter_add uses a three-phase scheme for fp16/bf16/int8: Phase 1 cast output tensor to int32 into a workspace; Phase 2 atomicAdd on the int32 workspace; Phase 3 cast back from workspace to the"
phenomenon: build_failure
signal:
  - "scatter-add kernel uses fp16/bf16/int8 or other non-native atomicAdd types"
confidence: single_run
original_id: OL-58
timestamp_inferred: true
tags: [ascendc, platform_constraint, ol-58]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
scatter-add kernel uses fp16/bf16/int8 or other non-native atomicAdd types

## 教训 / 根因
CANN's scatter_add uses a three-phase scheme for fp16/bf16/int8: Phase 1 cast output tensor to int32 into a workspace; Phase 2 atomicAdd on the int32 workspace; Phase 3 cast back from workspace to the original dtype. int32 atomicAdd is natively supported on all Ascend hardware. Between phases, use SyncAll() for full-core sync + pipe_.Reset() to reclaim UB.

## 证据
CANN scatter_add_simd.h:365-389, scatter_add_common.h:72-93. E1 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-58（category=platform_constraint，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
