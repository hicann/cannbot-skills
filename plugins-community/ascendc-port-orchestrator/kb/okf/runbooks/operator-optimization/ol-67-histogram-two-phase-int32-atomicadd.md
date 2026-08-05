---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Histogram two-phase int32 atomicAdd pattern"
description: "Do not use float atomicAdd for counts (output may read all-zeros before the kernel completes); use phase-1 int32 atomicAdd then a phase-2 kernel to convert dtype, whose launch is the implicit sync point."
confidence: single_run
original_id: OL-67
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-67, histogram, atomicadd, two-phase]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: histogram / count-class ops (atomicAdd on counters). Loaded by Generator.

Do **not** use float `atomicAdd` for counts. When a float atomicAdd writes directly to the
output buffer, and there is no subsequent kernel launch to act as an implicit sync point, the
output may be read before the kernel completes — yielding all zeros.

**Correct pattern (two-phase):**
1. Phase 1 — int32 `atomicAdd` to accumulate counts.
2. Phase 2 — a separate kernel converts int32 → target dtype. That second kernel launch is
   itself the implicit synchronization point that guarantees Phase 1 has finished.

**Evidence**: Histc case[0] (128 elements): float atomicAdd produced all zeros; the int32
two-phase pattern gave 15/15 PASS. E3 level (measured).
