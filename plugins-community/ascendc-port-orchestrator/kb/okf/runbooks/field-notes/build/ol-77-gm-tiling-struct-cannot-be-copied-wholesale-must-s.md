---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "GM tiling struct cannot be copied wholesale — must scalar-read each field"
description: "AscendC does not support `auto tmp = *tilingGmPtr` (struct-wide copy from GM to a local variable). It errors as address-space mismatch. **Correct approach**: cast the tiling struct GM pointer to `__gm"
phenomenon: build_failure
signal:
  - "kernel uses a `__gm__` tiling struct to pass parameters (batch_size, hidden_size, etc. packed into a struct)"
confidence: single_run
original_id: OL-77
timestamp_inferred: true
tags: [ascendc, platform_compat, ol-77]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
kernel uses a `__gm__` tiling struct to pass parameters (batch_size, hidden_size, etc. packed into a struct)

## 教训 / 根因
AscendC does not support `auto tmp = *tilingGmPtr` (struct-wide copy from GM to a local variable). It errors as address-space mismatch. **Correct approach**: cast the tiling struct GM pointer to `__gm__ int32_t*` and read field by field: `batchSize_ = *(tilingGmInt + 0); seqLen_ = *(tilingGmInt + 1);`. Or pass each parameter as an independent kernel argument in the pybind layer (avoid struct).

## 证据
30_TimeDecayExponentialStabilization V1: `auto tiling = *tilingData` failed to compile. Switching to scalar reads compiled. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-77（category=platform_compat，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
