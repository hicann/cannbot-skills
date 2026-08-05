---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Prove the current generated source produced the loaded binary"
description: "Prove the current generated source produced the loaded binary."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#R0-source-rebuild-overlay
timestamp_inferred: true
tags: [npu-validation, provenance, build]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Build the current generated source in an empty task-owned directory. Use a temporary sentinel change in task-owned probe code to prove the loaded binary changes, then restore and clean-build. Target archives and installed implementations may be inspected as prior-art, and isolated overlays may be used for research, but the final validation gate must load the task-owned clean build and must not treat target behavior as truth.
