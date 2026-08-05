---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Confirm every source-derived generated dispatch branch"
description: "Confirm every source-derived generated dispatch branch."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R4-dispatch-confirmation
timestamp_inferred: true
tags: [npu-validation, dispatch, coverage]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Create at least one case per selected-source dtype, shape, layout, attribute, optional-input, and backward branch. Instrument only current generated code with task-owned counters or sentinels, then remove instrumentation for the final clean build.
