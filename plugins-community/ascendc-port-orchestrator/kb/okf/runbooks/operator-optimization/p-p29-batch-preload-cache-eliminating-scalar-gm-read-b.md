---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Batch Preload Cache — eliminating scalar GM read bottleneck"
description: "Scenario: SIMD kernel loop needs to read a small amount of scalar data (index/weight); each GetValue() reads from GM at ~100 cycles. msprof evidence: scalar=42% (SG backward); most of it is GetValue G"
severity: critical
confidence: single_run
original_id: P-P29
timestamp_inferred: true
tags: [memory_access, optimization, p-p29, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点

**Scenario**: SIMD kernel loop needs to read a small amount of scalar data (index/weight); each `GetValue()` reads from GM at ~100 cycles.

**msprof evidence**: scalar=42% (SG backward); most of it is GetValue GM scalar reads.

**Anti-pattern** (per-element GM read):
```cpp
for (int i = 0; i < actual_k; i++) {
  local_index[i] = indexGm_.GetValue(index_base + i);   // ~100 cycle per read
  local_weight[i] = weightGm_.GetValue(index_base + i); // ~100 cycle per read
}
```

**Correct pattern** (batch preload into UB cache):
```cpp
// Init: allocate cache buffers
static constexpr uint32_t CACHE_SIZE = 1024;
pipe_.InitBuffer(idxCacheBuf_, CACHE_SIZE * sizeof(int32_t));
pipe_.InitBuffer(wtCacheBuf_, CACHE_SIZE * sizeof(float));

// Cache accessor: batch load on cache miss
__aicore__ inline int32_t GetIndexCached(int64_t idx, int64_t endIdx) {
  if (idx >= idxCacheBase_ + idxCacheLen_) {
    uint32_t copyLen = min(endIdx - idx, CACHE_SIZE);
    DataCopyPad(cache, indexGm_[idx], {1, copyLen * sizeof(int32_t), 0, 0}, padNone);
    idxCacheBase_ = idx; idxCacheLen_ = copyLen;
    // MTE2→Scalar sync: GetValue only after DataCopyPad completes
    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(ev);
    WaitFlag<HardEvent::MTE2_S>(ev);
  }
  return cache.GetValue(idx - idxCacheBase_);  // ~1 cycle from UB
}
```

**Principle**: one GM DMA loading 1024 elements (amortized ~0.1 cycle/element) replaces 1024 scalar GM reads (~100 cycle/each). Hit rate depends on whether consecutively accessed indices in the loop fall in the same cache block.

**Applicability**:
- GM scalars read sequentially in a loop (index, weight, offset, etc.)
- When total access > cache size, chunked load; accesses must be increasing
- Non-increasing accesses (e.g. indirect indexing) do not fit this pattern

**Combine with P-P28 (Ping-Pong)**: P-P29 eliminates the scalar read bottleneck (scalar pipe); P-P28 overlaps MTE2/VEC. They are orthogonal.

**Cache size selection**: 1024 is an expert empirical value. Too large wastes UB; too small increases miss frequency. Adjust based on UB headroom and top_k.

<!-- 迁移自 porter kb/target/ascendc/patterns/domains/memory_access.md（P-P29，convert_patterns_to_okf.py）。confidence 未升格。 -->
