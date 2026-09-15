---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-core outer_blocks partitioning template (elementwise/fused)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; op_class=elementwise,fused-elementwise verified_on: soc=Ascend950PR_957b; cann=9.0.0 status: canonical Template for partitioning independent outer iterations a"
confidence: single_run
original_id: P-P114
timestamp_inferred: true
tags: [patterns-index, optimization, p-p114, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; op_class=elementwise,fused-elementwise`
`verified_on: soc=Ascend950PR_957b; cann=9.0.0`
`status: canonical`

Template for partitioning independent outer iterations across multiple AI cores
via `GetBlockIdx()`. Each core processes a disjoint range of outer_blocks.

**When to use**: Any elementwise or fused-elementwise op where:
  1. Each outer_block is independent (no cross-block data dependency)
  2. outer_blocks >= 2
  3. The op has no atomic/scatter operations

**Template — kernel.h Init()**:

```cpp
__aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint64_t N, uint64_t half,
                             uint64_t stride, uint32_t num_cores) {
    uint64_t raw_block = half * stride;
    uint64_t outer_blocks = (stride > 0 && half > 0) ? (N / (2 * raw_block)) : 1;
    num_cores_ = (num_cores > 0) ? num_cores : 1;
    uint64_t core_idx = GetBlockIdx();
    uint64_t base = outer_blocks / num_cores_;
    uint64_t rem  = outer_blocks % num_cores_;
    start_ob_ = core_idx * base + (core_idx < rem ? core_idx : rem);
    count_ob_ = base + (core_idx < rem ? 1 : 0);
    block_size_ = raw_block;
    tileLen_ = MAX_TILE;
}
```

**Template — kernel.h Process()**:

```cpp
__aicore__ inline void Process() {
    for (uint64_t k = 0; k < count_ob_; ++k) {
        uint64_t ob = start_ob_ + k;
        uint64_t xb = ob * 2 * block_size_;
        for (uint64_t o = 0; o < block_size_; o += tileLen_) {
            uint64_t c = (o + tileLen_ > block_size_) ? (block_size_ - o) : tileLen_;
            // ... per-tile compute ...
        }
    }
}
```

**Template — pybind11.cpp launch**:

```cpp
static uint32_t compute_nblk(uint64_t outer_blocks) {
    if (outer_blocks <= 1) return 1;
    return static_cast<uint32_t>(std::min<uint64_t>(outer_blocks, 32));
}
```

**Cross-ref**: OL-254 (decision rule), EC-78 (NBLK=1 diagnostic fallback), P-P115 (combine with zero-copy strided access for split-input ops).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P114，convert_patterns_to_okf.py）。confidence 未升格。 -->
