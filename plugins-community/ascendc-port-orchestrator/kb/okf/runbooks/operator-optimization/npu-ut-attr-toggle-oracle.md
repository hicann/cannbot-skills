---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Validate one attribute or optional input at a time"
description: "Validate one attribute or optional input at a time."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R5-attr-toggle-oracle
timestamp_inferred: true
tags: [npu-validation, attributes, contract]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Hold tensors fixed and vary one declared option. Predict behavior from the contract and CPU or source-arch oracle, then confirm the generated output. An unchanged result is valid only when the declared semantics make that option inert.
