---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Run NPU validation gates in provenance-first order"
description: "Run NPU validation gates in provenance-first order."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#decision-order-for-verifying
timestamp_inferred: true
tags: [npu-validation, workflow, provenance]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Validate truth provenance, clean build and loaded digest, minimal launch, memory controls, source-derived dispatch, numerical truth, option behavior, then determinism and performance. Do not optimize or reinterpret results before these gates pass.
