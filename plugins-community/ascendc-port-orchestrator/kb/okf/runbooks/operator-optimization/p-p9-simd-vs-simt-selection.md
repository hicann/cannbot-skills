---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "SIMD vs SIMT selection"
description: "Core rule: Prefer SIMD unless scatter-write (atomicAdd) is required Scenario Choice Reason ------------------ scatter-write (atomicAdd to random addresses) SIMT SIMD's SetAtomicAdd requires alignment"
severity: high
confidence: single_run
original_id: P-P9
timestamp_inferred: true
tags: [kernel_launch, optimization, p-p9, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Core rule**: **Prefer SIMD unless scatter-write (atomicAdd) is required**

| Scenario | Choice | Reason |
|------|------|------|
| scatter-write (atomicAdd to random addresses) | **SIMT** | SIMD's SetAtomicAdd requires alignment and is functionally limited |
| indirect-read + weighted sum (e.g., SG Forward) | **SIMD** | DataCopy block transfer + 4-pipeline parallelism |
| Contiguous aligned read/write | **SIMD** | Natural pipeline-overlap advantage |
| scatter-read + scatter-write mixed | **SIMT** | SIMD cannot orchestrate when both ends are irregular |

**~~Old rule~~ (deprecated)**: ~~"indirect indexing / random access -> SIMT"~~

**Reason for deprecation (msprof evidence)**: SG Forward has indirect addressing (each token reads a different expert), but the expert rows themselves are contiguous in memory. SIMD DataCopy batch-transports expert rows to UB (via MTE2), which is 2-7x faster than SIMT thread-scalar scattered reads. SIMT's VEC pipe gets stalled by GM read latency (vec=0.95+, mte2=0.000); SIMD runs 4 pipes simultaneously (vec+scl+mte2+mte3 each at 30-90%).

**Key insight**: "Indirect addressing" must distinguish the addressing layer from the data layer. SG Forward's addressing layer is indirect (which expert), but the data layer is contiguous (expert rows are contiguous memory). SIMD's DataCopy handles the data layer while scalar computes the addressing. indirect-read != must-be-SIMT.

**SIMT architecture limitation**: In SIMT mode, GM access only uses the VEC pipe's load/store units; the MTE2/MTE3 DMA engines do not participate. 4-pipeline parallelism is unachievable — this is a hardware limitation, not a code problem.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/kernel_launch.md（P-P9，convert_patterns_to_okf.py）。confidence 未升格。 -->
