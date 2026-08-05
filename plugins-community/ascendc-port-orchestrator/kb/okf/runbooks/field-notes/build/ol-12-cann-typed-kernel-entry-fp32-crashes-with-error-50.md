---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CANN typed kernel entry _fp32 crashes with error 507035"
description: "CANN 9.0.T501's auto-generated typed kernel entry points (e.g., `aclrtlaunch_pooling_forward_kernel_fp32`) crash at launch time with error code 507035. The legacy untyped entry points (e.g., `aclrtlau"
phenomenon: build_failure
signal:
  - "when registering kernel entry points for fp32 kernels on CANN 9.0.T501"
confidence: single_run
original_id: OL-12
timestamp_inferred: true
tags: [507035, aclrtlaunch_pooling_forward_kernel_fp32, aclrtlaunch_pooling_forward_kernel, _fp16, _bf16, _fp32, ascendc, platform_bug, ol-12]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when registering kernel entry points for fp32 kernels on CANN 9.0.T501

## 教训 / 根因
CANN 9.0.T501's auto-generated typed kernel entry points (e.g., `aclrtlaunch_pooling_forward_kernel_fp32`) crash at launch time with error code 507035. The legacy untyped entry points (e.g., `aclrtlaunch_pooling_forward_kernel`) work correctly for the same kernel. The `_fp16` and `_bf16` typed entries are unaffected. This is a CANN runtime bug specific to the `_fp32` suffix. Workaround: use legacy (untyped) entry points for all fp32 kernels. May be fixed in CANN 9.0.0.beta.1.

## 证据
docs/guides/A5_CONTAINER_SETUP.md ("Known Issues"), tests/npu/npu_prod_benchmark.cpp lines 41-42, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md line 248

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-12（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
