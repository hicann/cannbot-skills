---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMT 1024 threads crashes with 507035 on Ascend950PR"
description: "Raising LAUNCH_BOUND from 512 to 1024 caused a 507035 vector core exception. Not every kernel can run 1024 threads — kernel complexity (int64 division loops etc.) may exceed register-file capacity. Sa"
phenomenon: build_failure
signal:
  - "SIMT kernel attempts to raise thread count to 1024"
confidence: single_run
original_id: OL-56
timestamp_inferred: true
tags: [507035, ascendc, platform_constraint, ol-56]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
SIMT kernel attempts to raise thread count to 1024

## 教训 / 根因
Raising LAUNCH_BOUND from 512 to 1024 caused a 507035 vector core exception. Not every kernel can run 1024 threads — kernel complexity (int64 division loops etc.) may exceed register-file capacity. Safe practice: compute-heavy SIMT kernels keep 512 threads.

## 证据
Permute kernel int64 division loop + 1024 threads → 507035 crash

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-56（category=platform_constraint，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
