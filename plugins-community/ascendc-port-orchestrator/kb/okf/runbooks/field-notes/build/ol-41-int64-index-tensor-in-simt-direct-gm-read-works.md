---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "int64 index tensor in SIMT — direct __gm__ read works"
description: "Reading int64_t from GM via `__gm__ int64_t* idx; int64_t val = idx[i];` works correctly in SIMT. No special handling needed for 8-byte types. This avoids the int64→int32 conversion overhead in pybind"
phenomenon: build_failure
signal:
  - "kernels reading int64 values from GM in SIMT mode"
confidence: single_run
original_id: OL-41
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-41]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
kernels reading int64 values from GM in SIMT mode

## 教训 / 根因
Reading int64_t from GM via `__gm__ int64_t* idx; int64_t val = idx[i];` works correctly in SIMT. No special handling needed for 8-byte types. This avoids the int64→int32 conversion overhead in pybind11.

## 证据
Gather V2 (2026-04-10): 47/47 PASS with direct int64 index reads

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-41（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
