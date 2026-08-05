---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "nblk=1 test is the ultimate diagnostic tool for multi-core contention"
description: "When precision fails at nblk=56 but passes at nblk=1, the root cause is always a multi-core parallel issue (DataCopy alignment overrun EC-22, buffer overrun, write-write race). Strong diagnostic tool:"
phenomenon: perf_regression
signal:
  - "precision test fails at nblk>1"
confidence: single_run
original_id: OL-43
timestamp_inferred: true
tags: [ascendc, measurement, ol-43]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
precision test fails at nblk>1

## 教训 / 根因
When precision fails at nblk=56 but passes at nblk=1, the root cause is always a multi-core parallel issue (DataCopy alignment overrun EC-22, buffer overrun, write-write race). Strong diagnostic tool: change nblk to 1 in pybind11.cpp to confirm.

## 证据
Pad V5 (2026-04-10): nblk=1 → 51/51, nblk=56 → 28/51

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-43（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
