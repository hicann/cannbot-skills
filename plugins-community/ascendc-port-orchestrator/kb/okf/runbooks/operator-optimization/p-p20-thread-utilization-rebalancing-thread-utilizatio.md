---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Thread utilization rebalancing (Thread Utilization Analysis)"
description: "General strategy: SKILLS_DESIGN.md §6.2 — can be auto-discovered by static analysis; no msprof needed Core principle: In multi-dimensional work decomposition, the number of threads assigned to each di"
confidence: single_run
original_id: P-P20
timestamp_inferred: true
tags: [thread_utilization, optimization, _rt_vf, p-p20, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**General strategy**: SKILLS_DESIGN.md §6.2 — can be auto-discovered by static analysis; no msprof needed

**Core principle**: In multi-dimensional work decomposition, the number of threads assigned to each dimension should match the actual work. Rebalance when `utilization = min(actual_work, threads) / threads < 50%`.

**Concrete instance (Pooling BRE=emb_dim)**: Set BLOCK_READ_EMB to emb_dim (instead of a fixed 32 or 512) so that iter_emb=1 and the number of index threads is maximized.

**Anti-pattern** (fixed BRE=32):
```cpp
// At dim=9: 512/32=16 index threads; 23/32 emb threads wasted
constexpr int BRE = 32;
// At dim=1: 16 index threads handle 16 edge/tile (severely inefficient)
```

**Correct pattern** (BRE=emb_dim, runtime parameter):
```cpp
int BRE = (emb_dim <= 512) ? static_cast<int>(emb_dim) : 512;
int block_read_indices = THREAD_NUM / BRE;
// At dim=9: 512/9=56 index threads (3.5x more edge/tile)
// At dim=1: 512/1=512 index threads (32x more edge/tile)
```

**A5 measured effect** (61 clusters, fp32, sorted edges):

| dim range | BRE=32 Fwd | BRE=dim Fwd | Speedup |
|---------|:---:|:---:|:---:|
| dim=1~9 (clusters 0-6) | 0.64ms | 0.39ms | **1.65x** |
| dim=33 (clusters 13-16) | 16.51ms | 11.38ms | **1.45x** |
| dim>256 (clusters 52+) | roughly even | roughly even | 1.0x |
| **Total** | 15.43ms | 10.70ms | **1.44x** |

**Notes**:
- BRE=emb_dim requires turning the template parameter into a runtime parameter (cannot instantiate a template for every dim value).
- **Replaces P-P11's BRE selection strategy**: dim <= 512 uses BRE=emb_dim; dim > 512 uses BRE=512.
- Compatible with sorted-edge register accumulation: when iter_emb=1, accum degenerates to a single scalar.

**Migration-time trigger rule (generator MUST check)**:
- For arch22→arch35 migration, check every fixed work-decomposition constant against
  the target AIV count, UB budget, and runtime shape.
- A fixed `BRE=32` wastes work on small embedding dimensions; derive it from
  `emb_dim` and emit runtime dispatch when the shape is dynamic.

#### Runtime-BRE Variant (_rt_vf) implementation pattern

When BRE=emb_dim, the BRE value is unknown at compile time (different callers use different emb_dim). A separate `_rt_vf` variant must be generated:

**Naming convention**:
```
pool_{op}_{dir}_kernel_vf<T, BRE, TI>        — template BRE/TI (compile-time)
pool_{op}_{dir}_sorted_kernel_vf<T, BRE, TI> — sorted variant (template BRE/TI)
pool_{op}_{dir}_sorted_rt_vf<T>              — sorted + runtime BRE/TI
```

**_rt_vf function signature** (BRE/TI + pre-computed params all passed as runtime parameters):
```cpp
template <typename DATA_TYPE>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void pool_{op}_{dir}_sorted_rt_vf(
    GM_ADDR ...,
    int BRE, int TI, int block_read_indices,
    int iter_indices_block, int iter_indices_thread, int iter_emb,
    uint32_t block_index, uint32_t total_block_num) {
  // BRE/TI etc. are pre-computed in the host dispatcher and passed in
  // Combine with P-P21 sorted so that iter_emb=1 degenerates to a single scalar accum
}
```

**Host dispatcher pre-computation** (in `{op}_launch_config.h`):
```cpp
inline void compute_pooling_params(int emb_dim, int thread_num,
    int& BRE, int& TI, int& bri, int& iib, int& iit, int& ie) {
  BRE = (emb_dim <= 512) ? emb_dim : 512;
  bri = thread_num / BRE;  // block_read_indices
  TI = bri;                // tile indices = index threads per block
  iib = (work_items + 56 * bri - 1) / (56 * bri);  // iter per block
  iit = 1;                                           // iter per thread (simplified)
  ie = (emb_dim + BRE - 1) / BRE;                   // iter_emb
}
```

**When to use template vs runtime**:
- dim is known and fixed (e.g., SG hidden_dim=256) -> template BRE/TI + `#pragma unroll`
- dim varies at runtime (e.g., Pooling 61 clusters dim 1~512) -> runtime `_rt_vf`
- **Generate both**: template for benchmark/specialization, runtime for production dispatch.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/thread_utilization.md（P-P20，convert_patterns_to_okf.py）。confidence 未升格。 -->
