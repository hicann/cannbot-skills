---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Input-arity-specialized kernels for variable-input-count ops"
description: "verified_on: cv-agent concat_dv2 4-kernel architecture (dim0_1/2/3/4 inputs) with shared common base class Pattern: For ops accepting a variable number of input tensors (concat, stack, elementwise wit"
phenomenon: build_failure
signal:
  - "verified_on: cv-agent concat_dv2 4-kernel architecture (dim0_1/2/3/4 inputs) with shared common base class"
confidence: inferred
status: stub
original_id: CAND-FA-CV-12
timestamp_inferred: true
tags: [candidate, inferred, concat_dim0_kernel_common.h, _1.cpp, _2.cpp, _3.cpp, _4.cpp, cand-fa-cv-12]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`verified_on: cv-agent concat_dv2 4-kernel architecture (dim0_1/2/3/4 inputs) with shared common base class`

**Pattern**: For ops accepting a variable number of input tensors (concat, stack, elementwise with N inputs), emit one specialized kernel per supported input count rather than a single generic kernel with dynamic buffer allocation. Each per-arity kernel has optimal buffer sizing — no over-allocation for the common small-N case, no under-allocation for the large-N case.

**Why not one generic kernel**: Dynamic buffer allocation based on input count means either (a) allocating for max possible inputs (wasteful for common 2-input case), or (b) allocating per-call based on runtime count (complex, error-prone, hard to verify at build time). Per-arity kernels make buffer sizes compile-time constants → verifiable by prebuild check.

**Shared base class pattern**: All per-arity kernels inherit from a common base (`concat_dim0_kernel_common.h`) holding shared logic (tiling struct, DataCopy primitives, row iteration). Per-arity kernel only customizes: buffer count, DataCopy loop count, output offset computation.

**When to use**: Input count variadicity is bounded and known at kernel compile time (not runtime-dynamic):
- concat: 1-4 inputs → 4 kernels
- stack: 2-4 inputs → 3 kernels
- Elementwise with N operands: N ∈ {2,3,4} → 3 kernels

**When NOT to use**: Input count is truly runtime-dynamic (≥5, unbounded, or determined by host-side logic after build). Then fall back to max-allocation single kernel.

**Detection**: count kernel .cpp files with input-count in filename (`_1.cpp`, `_2.cpp`, `_3.cpp`, `_4.cpp`). Single generic kernel handling N inputs → gap for bounded-N ops.

**Evidence**: cv-agent `concat_dv2/kernel/` has `concat_dim0_1_kernel.h` through `concat_dim0_4_kernel.h`, all inheriting from `concat_dim0_kernel_common.h` with shared `CopyTiling`, `SetGlobalBuffer` logic.

**Cross-ref**: CAND-FA-CV-7 (multi-strategy dispatch — same "specialize per parameter-class" principle, different dispatch axis: input count vs shape regime), CAND-FA-CV-4 (cube/vec class separation — common base + per-variant subclass is the same architectural pattern)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-CV-12，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
