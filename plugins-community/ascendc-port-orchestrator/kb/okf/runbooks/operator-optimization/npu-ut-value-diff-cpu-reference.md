---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Compare generated outputs with admissible migration or backward truth"
description: "Compare generated outputs with admissible migration or backward truth."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R3-value-diff-vs-cpu
timestamp_inferred: true
tags: [npu-validation, precision, backward]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Migration truth is the declared contract plus the selected independent source: preferred `npubench` uses an immutable task/sidecar bundle; only explicit `a3_live` uses the source capture. For `a3_live`, Migration truth is the declared contract plus current selected-arch22 source NPU capture. Backward truth is CPU fp64 autograd plus gradient equations and saved-tensor contract. Normalize to declared output dtype and cover near-zero, overflow, reduction, optional, and non-contiguous cases.
