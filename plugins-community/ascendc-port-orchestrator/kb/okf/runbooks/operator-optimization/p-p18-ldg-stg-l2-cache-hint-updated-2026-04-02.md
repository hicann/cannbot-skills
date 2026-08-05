---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "`__ldg`/`__stg` L2 Cache Hint (updated 2026-04-02)"
description: "API: AscendC provides templated __ldg/__stg to control L2 cache and L1/dcache behaviour: cpp #include <kernel_operator.h> // LD_L2CacheType, ST_L2CacheType, L1CacheType // Read: controls L2 allocation"
severity: high
confidence: single_run
original_id: P-P18
timestamp_inferred: true
tags: [platform_compat, optimization, __ldg, __stg, l2_cache_hint_normal, p-p18, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**API**: AscendC provides templated `__ldg`/`__stg` to control L2 cache and L1/dcache behaviour:
```cpp
#include <kernel_operator.h>  // LD_L2CacheType, ST_L2CacheType, L1CacheType

// Read: controls L2 allocation policy + L1/dcache caching
T val = __ldg<LD_L2CacheType::hint, L1CacheType::hint>(ptr);

// Write: controls L2 write-back policy + L1 caching
__stg<ST_L2CacheType::hint, L1CacheType::hint>(ptr, val);
```

**Available hint values** (source: HKV expert code + manual HA.FS007):

| Read (LD_L2CacheType) | Meaning |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL` | Normal caching (default; equivalent to no-arg `__ldg`) |
| `L2_CACHE_HINT_NOTALLOC_CLEAN` | Do not occupy an L2 slot after read; prevents large-range scans from polluting the cache |

| Write (ST_L2CacheType) | Meaning |
|---------------------|------|
| `L2_CACHE_HINT_NORMAL_FV` | Normal write-back to L2 |

| L1/dcache (L1CacheType) | Meaning |
|-------------------------|------|
| `CACHEABLE` | Cache through L1/dcache |
| `NON_CACHEABLE` | Bypass L1/dcache |

**Hint selection by access pattern**:

```cpp
// 1. Data read repeatedly by multiple cores/tokens (expert rows, embedding table)
//    -> Keep in L2 + dcache: maximize hit rate
val = __ldg<L2_CACHE_HINT_NORMAL, L1CacheType::CACHEABLE>(expert_ptr);

// 2. Sequential scan, read once (edge index, weight array)
//    -> L2 no-alloc: prevent cache pollution, leaving room for hot data
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(index_ptr);

// 3. Output write (one-shot write, no subsequent read)
//    -> L1 not cached: do not waste dcache space
__stg<ST_L2CacheType::L2_CACHE_HINT_NORMAL_FV, L1CacheType::NON_CACHEABLE>(out_ptr, val);

// 4. HKV in-bucket random lookup (small chunk scanned repeatedly)
//    -> L2 no-alloc + L1 cached: in-bucket data takes the dcache fast path
val = __ldg<L2_CACHE_HINT_NOTALLOC_CLEAN, L1CacheType::CACHEABLE>(bucket_ptr);
```

**History**: Earlier tests of no-template-arg `__ldg` (OL-18, 2026-03-26) showed no effect — because the default is `L2_CACHE_HINT_NORMAL` + default L1 policy, indistinguishable from a plain read on wide sequential scan. The hinted version can differentiate hot data (keep in L2) from cold data (do not allocate L2) — that is the correct usage.

**Experiment result (Batch 14-5)**: On SIMT persistent SG forward, tested `NOTALLOC_CLEAN` (index/weight) + `NORMAL_PERS` (expert). **No positive effect** — dim=64 24% slower (instruction overhead); others unchanged. dcache was already caching effectively. The value of L2 hints lies in cross-core sharing scenarios (e.g., HKV), not in SIMT persistent sequential traversal.

**A5 measured data** (56 blocks x 32 threads, stride-scan, aclrtEvent timing):

| Data size | Plain-read BW | `__ldg` BW | Difference |
|---------|----------|-----------|------|
| 4 MB | 49.5 GB/s | 49.7 GB/s | +0.3% |
| 16 MB | 54.3 GB/s | 54.4 GB/s | +0.2% |
| 64 MB | 43.5 GB/s | 43.6 GB/s | +0.2% |
| 256 MB | 22.8 GB/s | 22.8 GB/s | -0.1% |

**Decision rule**:
- Dataset >> L2 cache -> do not use `__ldg` (pooling, SG, large-scale reduction)
- Dataset <= L2 cache and repeatedly accessed -> use `__ldg` (hash-bucket scan, small matmul)
- Uncertain -> do not add (zero benefit, added code complexity)

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/platform_compat.md（P-P18，convert_patterns_to_okf.py）。confidence 未升格。 -->
