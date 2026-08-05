---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Run a minimal runtime health probe before kernel diagnosis"
description: "Run a minimal runtime health probe before kernel diagnosis."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#gotcha-aclinit-500000
timestamp_inferred: true
tags: [npu-validation, runtime, environment]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Before changing generated kernel code, confirm the authorized lane can initialize its runtime, allocate, launch a minimal public-API kernel, and synchronize. Record the ABI stack. Classify failures before operator execution as environment failures.
