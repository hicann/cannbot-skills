---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Use task-owned GM canaries to detect out-of-bounds writes"
description: "Use task-owned GM canaries to detect out-of-bounds writes."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R1-gm-canary
timestamp_inferred: true
tags: [npu-validation, memory, canary]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Place distinct guard regions around the declared task-owned output or workspace. Verify aligned, tail, empty, and minimum shapes after synchronization. Padding may protect a transfer but must not change the logical contract.
