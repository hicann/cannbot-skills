---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Recurrent ops are super-speedup targets for AscendC"
description: "The reference for Python-looped ops (RWKV WKV, recurrent state update, sequential scan) has one torch-op launch overhead per step (~100us/step). At seq_len=100 that is ~10ms of pure launch overhead. A"
phenomenon: perf_regression
signal:
  - "reference implementation has a Python `for t in range(seq_len)` loop, with each step calling an independent torch op"
confidence: single_run
original_id: OL-76
timestamp_inferred: true
tags: [ascendc, performance, ol-76]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 现象 / 触发
reference implementation has a Python `for t in range(seq_len)` loop, with each step calling an independent torch op

## 教训 / 根因
The reference for Python-looped ops (RWKV WKV, recurrent state update, sequential scan) has one torch-op launch overhead per step (~100us/step). At seq_len=100 that is ~10ms of pure launch overhead. An AscendC single-kernel internal loop eliminates **all** per-step launch overhead, keeping state in UB without GM round-trip. Speedup ≈ O(seq_len × launch_overhead / kernel_compute_time). **This is not compute optimization — it is launch elimination.** Even with no VEC tuning inside AscendC, replacing the Python loop with an in-kernel C++ loop yields 10-50x.

## 证据
30_TimeDecayExponentialStabilization (RWKV): Python ref mean 4.56ms (mix of seq=1..100), AscendC mean 0.13ms → **29.7x mean** (independent verification). Worst 4.3x (seq=1, almost no loop), best 49.8x (seq=100, loop dominates). 50/50 ≥ 1.0x. E3 level.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-76（category=performance，convert_ol_to_okf.py，M1）。confidence/severity/reproduce_count 未升格。 -->
