---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Multi-phase op via inheritance chain — phases share state via class-hierarchy, dispatch by TILING_KEY in same .so"
description: "> Source: cann-learner CAND-A3A5-7 (promoted 2026-05-12, Mode 5 batch 2). C36 lift applied — op-class generalized from index_put_with_sort to multi-phase-scatter-gather. applies_to: soc=all; cann=9.0.0; bisheng=15.0"
confidence: single_run
original_id: P-P122
timestamp_inferred: true
tags: [patterns-index, optimization, index_put_with_sort, multi-phase-scatter-gather, inheritance, tiling_key, p-p122, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

> **ID rename note (2026-08-31)**: this entry was originally `P-P92` in `patterns/PATTERN_INDEX.md:256`. Renamed to **P-P122** during OKF migration to resolve collision with the pre-existing `memory_access.md` P-P92 ("V220 PipeBarrier<PIPE_V> coalescing", card `p-p92-v220-pipebarrier-pipe-v-coalescing-remove-redund.md`).

> **Source**: cann-learner CAND-A3A5-7 (promoted 2026-05-12, Mode 5 batch 2). C36 lift applied — op-class generalized from "index_put_with_sort" → "multi-phase-scatter-gather" (covers scatter-with-pre-sort, gather-then-scatter, segmented-reduction with intermediate workspace, etc.).

`applies_to: soc=all; cann=9.0.0; bisheng=15.0.5; op_class=multi-phase-scatter-gather`
`verified_on: soc=Ascend950PR (index_put_with_sort/op_kernel/arch35/)`
`unverified_on: soc=Ascend910_V220 — pattern observed in 1 op's A5 port; V220 master uses different organization (may or may not benefit from this convention)`

**Principle**: when an op decomposes into 2-3 distinct algorithmic phases (e.g. sort-then-scatter, gather-then-process, segmented-reduce-then-finalize) that need to share intermediate state, organize the kernels as a **class inheritance chain in the SAME `.so`** dispatched by TILING_KEY values 0/1/2. State flows between phases via class member fields (the derived class sees the base class's accumulators directly), NOT via cross-kernel GM-workspace passing.

This is preferable to a multi-kernel approach (separate `.so` files for each phase, GM workspace handoff) because:
1. **No cross-kernel synchronization** — phases run in the same kernel-launch context with implicit ordering by TILING_KEY dispatch.
2. **Shared UB state** — intermediate buffers persist across phases without writeback-then-reload.
3. **Smaller binary footprint** — one `.so`, one launch.
4. **Simpler debugging** — single stack frame; inheritance chain is greppable.

**Concrete anchor** (from `index_put_with_sort/op_kernel/arch35/`):
```cpp
// base.h — common state + helpers
class IndexPutWithSortBase {
protected:
    GlobalTensor<int32_t> sortedIndicesGm;  // shared across phases
    LocalTensor<half> workBuffer;
    // ... init common state
};

// gather_data.h — inherits base, implements gather phase
class GatherDataOp : public ScatterDataInKernelOp {  // chains to scatter
    void Process() { /* gather, leaves results in base member */ }
};

// scatter_data.h — inherits base, implements scatter phase
class ScatterDataInKernelOp : public IndexPutWithSortBase {
    void Process() { /* scatter using sortedIndicesGm from base */ }
};

// In <op>.cpp dispatcher:
if (TILING_KEY_IS(0))      { IndexPutWithSortBase op; op.Init(...); op.Process(); }
else if (TILING_KEY_IS(1)) { GatherDataOp op; op.Init(...); op.Process(); }
else if (TILING_KEY_IS(2)) { ScatterDataInKernelOp op; op.Init(...); op.Process(); }
```

**Anti-pattern (DO NOT)**:
- Two separate `.so` files (one per phase) communicating via GM workspace — adds a kernel-launch boundary + double-write of intermediate state.
- Single mega-template kernel with `if constexpr` per phase — defeats the per-variant TILING_KEY dispatch.
- Using virtual functions in the inheritance chain — A5 kernel objects must be POD/concrete.

**Other instances (predicted)**: segmented-reduction ops (e.g. `MoeFinalizeRouting` could use this), sparse-to-dense ops with sorting prologue, any op with a "preprocess → main-compute" split where preprocess outputs structure the main-compute access pattern.

**Cross-ref**: P-P91 variant-split (this pattern is the inheritance-based variant of variant-split — use P-P122 when phases share state, P-P91 when variants are independent dtype/layout dispatches); patterns/domains/scatter_add.md (scatter-add primitives that the scatter phase invokes).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（原 P-P92，撞号改名为 P-P122，convert_patterns_to_okf.py 曾 skip:card-exists 误核销；2026-08-31 手工补卡，format 对齐该脚本输出）。confidence 未升格。 -->
