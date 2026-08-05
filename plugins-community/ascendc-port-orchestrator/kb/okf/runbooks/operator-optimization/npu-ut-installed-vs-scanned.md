---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Match current generated source digests to the tested binary"
description: "Match current generated source digests to the tested binary."
confidence: single_run
original_id: NPU_UT_VALIDATION.md#cardinal-rule-installed-vs-scanned
timestamp_inferred: true
tags: [npu-validation, provenance, digest]
created_at: 2026-07-29T00:00:00Z
updated_at: 2026-07-29T00:00:00Z
---
## 验证规则

Record digests for selected arch22 source, schema, generated source, build manifest, object, and loaded extension. If uncertain, delete task-owned build outputs and rebuild. Installed target sources, objects, metadata, and dispatch records may be consulted as prior-art, but they are neither proof of the loaded generated binary nor migration/backward truth.
