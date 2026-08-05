---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Sorted-Edge First-Occurrence Dedup (atomicCAS-Free)"
description: "Problem: generate_assign_edges-class kernels use atomicCAS to detect first occurrence — one atomic per edge. When edge count is large, atomicCAS serialization becomes the bottleneck. Precondition: edg"
severity: high
confidence: single_run
original_id: P-P32
timestamp_inferred: true
tags: [scatter_add, optimization, generate_assign_edges, atomiccas, p-p32, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: `generate_assign_edges`-class kernels use `atomicCAS` to detect first occurrence — one atomic per edge. When edge count is large, atomicCAS serialization becomes the bottleneck.

**Precondition**: edges are sorted by target (coordinate with P-P21 sort preprocessing).

**Anti-pattern** (atomicCAS per edge):
```cpp
// One atomicCAS per edge — checks whether this is the first write to the target
int old = Simt::AtomicCas(&assign_edges[target], INVALID, source);
if (old == INVALID) {
    // first occurrence
}
```

**Correct pattern** (adjacent comparison):
```cpp
// Precondition: edges sorted by edge_out
// Compare adjacent edges' targets — a change of target marks "first occurrence"
int prev_target = (tid > 0) ? edge_out[tid - 1] : -1;
int cur_target = edge_out[tid];
if (cur_target != prev_target) {
    // first occurrence of this target — no atomic needed!
    assign_edges[cur_target] = edge_in[tid];
}
```

**Advantages**:
- Zero atomics (pure load/store, no CAS contention)
- O(1) per element (vs atomicCAS's O(contention))
- Handles block boundaries: the first thread needs to compare with the previous block's last element (cross-block boundary)

**Measurement**: Pooling assign_edges: 259ms → 10ms (**25.6x**), combined with the overall sorted pipeline.

**Trigger condition**: any first-occurrence / dedup operation where the input is already sorted or can be pre-sorted.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P32，convert_patterns_to_okf.py）。confidence 未升格。 -->
