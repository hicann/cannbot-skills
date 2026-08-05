---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch::zeros on NPU is NOT stream-ordered with custom kernels"
description: "The zero-fill of `torch::zeros` is not necessarily executed on the same stream as `aclrtlaunch_*`. A custom kernel may begin before zero-fill finishes, reading garbage data. **Fix**: use `torch::empty"
phenomenon: build_failure
signal:
  - "custom AscendC kernel uses a torch::zeros tensor as output"
confidence: single_run
original_id: OL-66
timestamp_inferred: true
tags: [ascendc, platform_bug, ol-66]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
custom AscendC kernel uses a torch::zeros tensor as output

## 教训 / 根因
The zero-fill of `torch::zeros` is not necessarily executed on the same stream as `aclrtlaunch_*`. A custom kernel may begin before zero-fill finishes, reading garbage data. **Fix**: use `torch::empty` + `aclrtMemsetAsync(stream)` instead of `torch::zeros`. Or use two kernel launches (the second implicitly waits for the first).

## 证据
Histc debugging 2026-04-14, 5/5 failures without fix, 5/5 passes with fix. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-66（category=platform_bug，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
