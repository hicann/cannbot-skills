---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "int64 fixes cause ~6% performance regression on AscendC SIMT"
description: "Changing inner loop counters from `int` to `int64_t` (e.g., `for(int j)` → `for(int64_t j)`) increases register pressure on Ascend950PR. The AIV general-purpose registers are 32-bit; int64 occupies 2"
phenomenon: perf_regression
signal:
  - "when applying int64 preservation fixes to AscendC SIMT kernels"
confidence: inferred
classified_by: llm-assisted
original_id: OL-16
timestamp_inferred: true
tags: [int, int64_t, ascendc, ol-16, llm-classified]
created_at: 2026-07-10T16:00:00Z
updated_at: 2026-07-10T16:00:00Z
---
## 现象 / 触发
when applying int64 preservation fixes to AscendC SIMT kernels

## 教训 / 根因
Changing inner loop counters from `int` to `int64_t` (e.g., `for(int j)` → `for(int64_t j)`) increases register pressure on Ascend950PR. The AIV general-purpose registers are 32-bit; int64 occupies 2 registers. Measured: Pooling D variant regressed 24.55ms → 25.94ms (~6%). The int64 interface guarantee (CLAUDE.md) is correct and must not be violated, but the regression is expected and must be reported honestly. Do NOT claim "no perf regression" without benchmark verification.

## 证据
E9-1 investigation (2026-03-29), commit 475e83c claimed "no perf regression" without running benchmark

<!-- LLM-辅助分类迁移(convert 二次 pass, migrate_ol_dispositions.py)。confidence=inferred,待人工确认。原 OL-16。 -->
