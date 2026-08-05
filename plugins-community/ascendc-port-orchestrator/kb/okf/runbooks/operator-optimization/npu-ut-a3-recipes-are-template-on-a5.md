---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Re-prove NPU validation gates on each architecture"
description: "Re-prove NPU validation gates on each architecture."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#scope-a3-validated-a5-unverified
timestamp_inferred: true
tags: [npu-validation, migration, architecture]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Treat a gate proven on one architecture as methodology evidence only. Re-run build, launch, memory, dispatch, dtype, and option probes on each authorized source and target lane. Target execution validates the generated candidate; it does not provide semantic truth.
