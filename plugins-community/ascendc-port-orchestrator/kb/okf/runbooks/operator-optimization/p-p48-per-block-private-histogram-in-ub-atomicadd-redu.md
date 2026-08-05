---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Per-Block Private Histogram in UB (atomicAdd Reduction)"
description: "Problem: N threads do GM atomicAdd against B bins (N >> B); atomicAdd serialization becomes the bottleneck. 28672 threads hitting 50-256 bins spend > 99% of time queueing. Pattern: Each block maintain"
confidence: single_run
original_id: P-P48
timestamp_inferred: true
tags: [scatter_add, optimization, __ubuf__, total_elems, p-p48, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Problem**: N threads do GM atomicAdd against B bins (N >> B); atomicAdd serialization becomes the bottleneck. 28672 threads hitting 50-256 bins spend > 99% of time queueing.

**Pattern**: Each block maintains a private histogram in `__ubuf__`. Threads do UB atomicAdd (much faster than GM). After a ThreadBarrier, merge to GM via SIMT per-thread atomicAdd:

**Pattern** (Histc, NPUKernelBench):
```cpp
__ubuf__ int32_t local_hist[MAX_BINS];
// clear
for (uint32_t b = threadIdx.x; b < bins; b += THREAD_NUM) local_hist[b] = 0;
Simt::ThreadBarrier();
// local accumulation
for (...) atomicAdd(&local_hist[bin_idx], 1);
Simt::ThreadBarrier();
// merge to GM — SIMT per-thread atomicAdd
for (uint32_t b = threadIdx.x; b < bins; b += THREAD_NUM) {
    if (local_hist[b] > 0) atomicAdd(&gm_counts[b], local_hist[b]);
}
```

**Effect**: GM atomicAdd count drops from `total_elems` to `bins × nblk`. 65536 elements, 128 bins, 56 blocks: ~33K → 7168 (4.6x reduction).

**UB requirement**: MAX_BINS × 4 bytes = 1024 bytes (negligible).

**Trigger condition**: any kernel with `atomicAdd(&output[computed_index], ...)` where output is much smaller than input (histogram, voting, binning, etc.).

**Evidence**:
- Histc (2026-04-14): 0.28x → 0.53x. E3 level.
- histogram_v2 (2026-06-22): 0.44x → 0.95x overall wallclock ratio (2.15× improvement). Kernel time for 1M elements: 894μs → 5.8–28.6μs (14–35× profiler speedup). 31/31 precision PASS (30/31 bit-exact). SIMT UB atomicAdd with per-block histogram. E4 level — validated on 2 independent ops (Histc + histogram_v2) with strong quantitative evidence.

**Stop condition**: When bins > UB available space / sizeof(int32) (theoretically 256KB / 4 = 64K bins, far larger than anything realistic).

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/scatter_add.md（P-P48，convert_patterns_to_okf.py）。confidence 未升格。 -->
