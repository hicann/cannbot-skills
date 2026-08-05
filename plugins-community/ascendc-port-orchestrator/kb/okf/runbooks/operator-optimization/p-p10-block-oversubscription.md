---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Block Oversubscription"
description: "nblk > physical AIV core count (56) → disperses atomicAdd contention. Applicability conditions (key): Only effective when the kernel has atomicAdd contention. - Unsorted backward (has atomicAdd conten"
confidence: single_run
original_id: P-P10
timestamp_inferred: true
tags: [scatter_add, optimization, p-p10, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

nblk > physical AIV core count (56) → disperses atomicAdd contention.

**Applicability conditions (key)**: Only effective when the kernel has atomicAdd contention.
- **Unsorted backward** (has atomicAdd contention): nblk=224 → bwd 2.0x speedup ✅
- **Sorted backward** (register accumulation, no atomicAdd): nblk=56 is strictly optimal. Oversubscription monotonically gets worse: 112 (+3.8%), 224 (+9.8%), 448 (+20.8%) ❌
- **Forward**: nblk=56 is always optimal (pipe-bound; oversubscription gives no benefit)

**Root cause**: Sorted + register accumulation eliminates atomicAdd contention → the benefit of dispersing contention via oversubscription disappears → only the register-pressure cost remains. Multiple blocks compete for the same AIV core's register file, and accumulator variables get spilled to HBM.

**E9-2 measurement confirmed** (2026-03-30, 61 clusters, NPU idle): see E10 exploration results.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P10，convert_patterns_to_okf.py）。confidence 未升格。 -->
