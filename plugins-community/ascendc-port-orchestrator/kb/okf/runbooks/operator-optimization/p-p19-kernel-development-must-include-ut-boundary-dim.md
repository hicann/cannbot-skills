---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Kernel development must include UT (boundary dim + sorted/original consistency)"
description: "Every new kernel or kernel variant must have a corresponding CPU reference test covering: 1. Large-dim boundary: dim=512, 1024, 4096 (validates BRE=512 path + accum fallback) 2. Sorted/unsorted consis"
severity: high
confidence: single_run
original_id: P-P19
timestamp_inferred: true
tags: [kernel_launch, optimization, p-p19, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

Every new kernel or kernel variant must have a corresponding CPU reference test covering:
1. **Large-dim boundary**: dim=512, 1024, 4096 (validates BRE=512 path + accum fallback)
2. **Sorted/unsorted consistency**: element-wise comparison of sorted output vs original output
3. **Production-grade stress test**: edges > 10K, dim > 256
4. **Boundary values**: edges=0, edges=1, dim=1

**Anti-pattern**: Considering a kernel correct just because it "runs through" production data — the large-dim path has never been tested.

**Correct pattern**: First write the CPU reference implementation + UT -> compile and run in CPU mode -> only deploy to NPU after passing.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P19，convert_patterns_to_okf.py）。confidence 未升格。 -->
