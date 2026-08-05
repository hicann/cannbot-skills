---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Default to GetBlockIdx() multi-core outer_blocks partitioning for partitionable elementwise/fused ops, not NBLK=1"
description: "For elementwise/fused ops with no cross-element reduction, atomics, or scatter, partition outer_blocks across cores via GetBlockIdx() (nblk=min(outer_blocks,32)); swi_glu 8.5× vs single-core."
confidence: single_run
original_id: OL-254
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-254, multi-core, outer-blocks, elementwise]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**Category:** algorithm_selection. Loaded by the Generator (elementwise ops) at Phase B code generation.

**Decision rule — use multi-core outer_blocks partitioning by default:**

1. **Partitionability check** (all must be YES):
   - NO cross-element reduction (no sum/mean across the split dimension)
   - NO atomic operations (no AtomicAdd / AtomicExchange)
   - NO indirect/scatter writes (no `arr[index[i]] = val`)
   - Independent outer_blocks (each block reads its own input slice, writes its own output slice)
   → all 4 pass ⇒ the op IS partitionable ⇒ GENERATE MULTI-CORE.

2. **Compute outer_blocks** (count of independent work units):
   - split-input ops: `outer_blocks = N / (2 × half × stride)`
   - flat elementwise ops: `outer_blocks = total_elements / MAX_TILE`
   - batched ops: `outer_blocks = batch_count`

3. **IF outer_blocks ≥ 2:** generate a multi-core kernel with `GetBlockIdx()` partitioning; `nblk = min(outer_blocks, 32)` (32 = safe cap for A5; 40 cores total, leave margin). Template (P-P114):
   ```cpp
   uint64_t core_idx = GetBlockIdx();
   uint64_t base = outer_blocks / num_cores;
   uint64_t rem  = outer_blocks % num_cores;
   start_ob_ = core_idx * base + (core_idx < rem ? core_idx : rem);
   count_ob_ = base + (core_idx < rem ? 1 : 0);
   for (uint64_t k = 0; k < count_ob_; ++k) {
       uint64_t ob = start_ob_ + k;
       // ... per-block tile loop ...
   }
   ```

4. **IF outer_blocks == 1:** single-core (nblk=1) is correct — document "outer_blocks=1, no parallelism" in a kernel comment.

5. **Anti-patterns:**
   - Do NOT default to NBLK=1 as a "safe" choice (see EC-78).
   - Do NOT use a data-parallel `blockDim=N` launch — use `GetBlockIdx()` partition with per-core distinct work ranges.
   - Do NOT use EC-17's nblk=1 workaround for alignment issues — use kernel-side `DataCopyPad` or mask instead.

**Edge cases where multi-core may regress:**
- Very small outer_blocks (< 4): launch overhead may exceed benefit; if outer_blocks ≤ 2, single-core is acceptable.
- Non-determinism: if multi-core introduces non-deterministic output (rare for partitionable ops), fall back to NBLK=1 and document.

### Evidence
swi_glu V5 (Ascend950PR_957b, CANN 9.0.0, 2026-06-24): 32-core outer_blocks partitioning on 50 cases — 41/50 faster than PyTorch NPU ref (vs 19/50 for single-core V1). Large-shape geo_mean 1.35× (vs V1 0.16× = 8.5× improvement). Precision 50/50 bit-exact; determinism 50/50 identical across runs. Unverified on Ascend910_V220 (A3 — outer_blocks GetBlockIdx partition should also work; verify on first A3 cold-start).

### Related
- EC-78 (NBLK=1 diagnostic), OL-255 (zero-copy strided split-input — apply together), OL-63 (tile-first UB allocation).
