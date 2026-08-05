---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Pair every device-exception trigger with a valid control"
description: "Pair every device-exception trigger with a valid control."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R2-device-exception-probe
timestamp_inferred: true
tags: [npu-validation, exception, control]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Run a valid control and a contract-invalid trigger through the same current generated entry with task-owned allocations. Record launch and synchronize status separately. The result is admissible only when the control succeeds and provenance proves the current generated binary ran.
