---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Keep validation task-owned and avoid target overlays"
description: "Keep validation task-owned and avoid target overlays."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#gotchas
timestamp_inferred: true
tags: [npu-validation, safety, environment]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Use task-owned build directories, allocations, containers, and generated entries. Do not copy into an installed operator tree or derive truth from it. Keep credentials, host addresses, private mounts, and machine-specific package paths in secured runtime configuration.
