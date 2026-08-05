---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Sort Algorithm Selection Decision Tree"
description: "Problem: Different N/dtype/batch combinations need different sort algorithms. No single algorithm covers all scenarios. Decision tree (check in priority order): 1. N ≤ 4096 (fp32) or N ≤ 1024 (fp16/bf"
confidence: single_run
original_id: P-P43
timestamp_inferred: true
tags: [sort, optimization, p-p43, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: Different N/dtype/batch combinations need different sort algorithms. No single algorithm covers all scenarios.

**Decision tree** (check in priority order):

```
1. N ≤ 4096 (fp32) or N ≤ 1024 (fp16/bf16)?
   YES → Hardware Merge Sort (Concat+Sort+Extract, P-P42)
   NO → continue

2. All data (value + index + tmp) fits in UB?
   YES → Single-core Radix Sort (AscendC Sort API with RADIX_SORT)
   NO → continue

3. B==1 AND 4096 < N ≤ 32768 AND fp32?
   YES → Multi-core Merge Sort (per-core local sort + 4-way hardware MrgSort merge)
   NO → continue

4. B ≥ core count AND N > 4096 AND blocksPerRow ≤ 256?
   YES → Big Batch Merge Sort (each core processes a whole row independently)
   NO → continue

5. Default → Multi-core Radix Sort (decoupled lookback prefix scan)
```

**Generator instructions**: When generating a Sort kernel:
1. Analyze the benchmark shape → determine N and dtype
2. For N ≤ 4096 (fp32): use P-P42 hardware sort directly — simplest and fastest
3. For N > 4096: consider radix sort, but implementation is complex. You can first use P-P42 on UB-sized chunks and then manually merge.

**Evidence**: CANN sort_tiling_arch35.cpp:746-761. E1 level.

**Stop condition**: When N < 32 the setup overhead dominates for all algorithms; a simple insertion sort may be faster.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/sort.md（P-P43，convert_patterns_to_okf.py）。confidence 未升格。 -->
