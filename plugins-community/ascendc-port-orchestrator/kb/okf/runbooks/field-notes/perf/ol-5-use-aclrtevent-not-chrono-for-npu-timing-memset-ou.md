---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Use aclrtEvent not chrono for NPU timing; memset outside event"
description: "Two measurement errors compounded in early benchmarks. (1) `std::chrono` measures wall-clock time including host synchronization overhead, not device execution time -- use `aclrtCreateEvent` / `aclrtR"
phenomenon: perf_regression
signal:
  - "when writing or reviewing NPU benchmark code"
confidence: single_run
original_id: OL-5
timestamp_inferred: true
tags: [aclrtcreateevent, aclrtrecordevent, aclrteventelapsedtime, aclrtmemset, ascendc, measurement, ol-5]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
when writing or reviewing NPU benchmark code

## 教训 / 根因
Two measurement errors compounded in early benchmarks. (1) `std::chrono` can
capture host launch/synchronization effects instead of device execution; use
`aclrtCreateEvent` / `aclrtRecordEvent` / `aclrtEventElapsedTime`. (2)
`aclrtMemset` was inside the timed region and inflated backward latency. Put
initialization outside the event window and discard conclusions produced by the
legacy timing path.

## 证据
BENCHMARK_METHODOLOGY.md Section 1, output/docs/archive/BENCHMARK_RESULTS_legacy_timing.md ("旧计时方法" / legacy timing method)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-5（category=measurement，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
