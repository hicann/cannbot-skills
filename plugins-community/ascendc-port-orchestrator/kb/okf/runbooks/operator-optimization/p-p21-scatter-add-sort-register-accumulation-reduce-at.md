---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Scatter-add Sort + Register Accumulation (reduce atomicAdd count)"
description: "Applicability: any scatter-add kernel (multiple inputs write the same output address) Problem: In scatter-add every edge does one atomicAdd. When many edges target the same output (high fan-in), atomi"
confidence: single_run
original_id: P-P21
timestamp_inferred: true
tags: [scatter_add, optimization, edge_length, unique_targets, atomicadd, p-p21, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Applicability**: any scatter-add kernel (multiple inputs write the same output address)

**Problem**: In scatter-add every edge does one atomicAdd. When many edges target the same output (high fan-in), atomicAdd serialization becomes the bottleneck. msprof signature: `aiv_vec_ratio=1.0` but HBM bandwidth utilisation < 1% (pipe is jammed by atomicAdd).

**Optimization idea**: Pre-sort edges so writes for the same target are contiguous → register accumulate → single atomicAdd.

**Anti-pattern** (one atomicAdd per edge):
```cpp
for (int i = 0; i < edge_length; i++) {
    atomicAdd(&output[edge_out[i] * dim + d], input[edge_in[i] * dim + d]);
    // 100,000 edges pointing to the same target → 100,000 atomicAdd ops queued up
}
```

**Correct pattern** (register accumulation after sort):
```cpp
// Precondition: edges are sorted by edge_out
float accum = 0;
int prev_target = -1;
for (int i = 0; i < edge_length; i++) {
    int target = edge_out[i];
    if (target != prev_target) {
        if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);  // write only when target changes
        accum = input[edge_in[i] * dim + d];
        prev_target = target;
    } else {
        accum += input[edge_in[i] * dim + d];  // register accumulation, no atomicAdd
    }
}
// flush final
if (prev_target >= 0) atomicAdd(&output[prev_target * dim + d], accum);
```

**Effect**: atomicAdd count drops from `edge_length` to `unique_targets`. With average fan-in=100, this is a 100x reduction.

**A5 measurement**: Pooling backward 100.53ms → 14.53ms (**-86%**), forward 15.66ms → 11.29ms (**-28%**). Bwd improvement is larger because atomicAdd contention is more severe (scatter-write vs scatter-read). Fwd improvement comes from register accum after sort reducing atomicAdd count (from `edge_length` to `unique_targets`).

**Sort overhead**: host `std::sort` 1147ms, NPU counting sort 405ms. Sort is a one-shot preprocessing step (no re-sort needed while the graph structure is unchanged).

**Trigger condition (generator must check)**:
- See `atomicAdd` inside a loop → check if it is a scatter-add pattern (multiple inputs write the same output)
- msprof shows `HBM_util < 1%` but `aiv_vec_ratio = 1.0` → atomicAdd bottleneck confirmed
- **The generation stage should already hint "this kernel has scatter-add; recommend providing a sorted variant"** — do not wait until the optimization stage.

**Notes**:
- ~~Sort only benefits backward~~ **Correction (E9-1 measurement, 61 clusters)**: forward benefits too. D_Fwd (sorted + BRE=emb_dim + register accum) is **1.39x** faster than B_Fwd (unsorted) overall (15.66ms → 11.29ms). For small dim (≤33), **1.45x**; for dim=1 + high fan-in, **7x** (cluster 5: 0.507 → 0.072ms). But for dim > 128, D actually becomes slower (BRE=emb_dim leaves too few index_threads); fall back to C (template sorted, BRE=32) in that case.
- Must be combined with P-P20 (BRE=emb_dim) so iter_emb=1 and a single scalar accum is sufficient.
- Precision: the order of float register accumulation differs from per-edge atomicAdd, but differences remain within atol (61/61 PASS).
- **dim cutoff**: dim ≤ ~128 use D variant (BRE=emb_dim); dim > 128 use C variant (BRE=32, template sorted).

#### Multi-Dim Accumulator Array (iter_emb > 1)

When BRE < emb_dim (large dims where BRE=emb_dim is not feasible), `iter_emb > 1` and a single scalar `accum` is insufficient. Use an accumulator array:

```cpp
constexpr int MAX_ACCUM = 16;  // max dim positions per thread
float accum[MAX_ACCUM] = {0};
int prev_target = -1;
int my_emb_lane = threadIdx.x % BRE;  // cache once

// Loop nesting: INDEX OUTER, EMBEDDING INNER (CRITICAL — never invert!)
for (int i = iter_start; i < iter_end; i++) {
    int target = edge_out[sorted_idx[i]];
    if (target != prev_target) {
        // Flush all accumulators for previous target
        if (prev_target >= 0) {
            for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
                if (accum[j] != 0.0f)
                    atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
                accum[j] = 0.0f;
            }
        }
        prev_target = target;
    }
    // Accumulate ALL embedding positions for this edge
    for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
        int64_t src_offset = static_cast<int64_t>(edge_in[sorted_idx[i]]) * dim + my_emb_lane + j * BRE;
        accum[j] += simt_to_float_generic<DATA_TYPE>(input[src_offset]);
    }
}
// Final flush
for (int j = 0; j < iter_emb && j < MAX_ACCUM; j++) {
    if (prev_target >= 0 && accum[j] != 0.0f)
        atomicAdd(&output[prev_target * dim + my_emb_lane + j * BRE], accum[j]);
}

// FALLBACK: when iter_emb > MAX_ACCUM, fall back to per-element atomicAdd
if (iter_emb > MAX_ACCUM) {
    // Use baseline atomicAdd path for the overflow portion
}
```

**CRITICAL loop nesting rule**: Index MUST be the outer loop, embedding inner. Inverting causes:
1. `prev_target` state reset between embedding iterations → redundant atomicAdd
2. iter_emb × redundant GM reads on edge_in/edge_out arrays
3. Defeats the entire purpose of sorted accumulation for multi-dim

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P21，convert_patterns_to_okf.py）。confidence 未升格。 -->
