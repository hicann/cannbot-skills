---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Column-major intermediate output + pybind transpose to eliminate per-element scalar pack in predicate-driven N-D coord emit"
description: "When the V4 GatherMask emit pattern (P-P80) is bottlenecked on the per-element scalar pack from column-major coord UB → row-major output GM (per-element interleave at ~10–15 scalar cycles per int64 di"
severity: high
confidence: single_run
original_id: P-P83
timestamp_inferred: true
tags: [scatter_add, optimization, out_ub, p-p83, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

When the V4 GatherMask emit pattern (P-P80) is bottlenecked on the per-element scalar pack from column-major coord UB → row-major output GM (per-element interleave at ~10–15 scalar cycles per int64 dimension), the **column-major intermediate output composite** eliminates the scalar pack entirely. **Kernel side**: allocate output GM as `[ndim, numel]` column-major instead of `[numel, ndim]` row-major; after SIMD coord decode produces `coord64[d * CHUNK_ROWS + r]` per dim, emit each dim's contiguous chunk via aligned `DataCopy(outputGm[d * numel + my_offset + emitted], coord64[d * CHUNK_ROWS], chunk_aligned)` (chunk_aligned = `(chunk/4)*4` for int64 32B alignment; tail <4 elements via uint32 split-AtomicAdd on pre-zeroed slots). No `out_ub` row-major staging buffer needed — saves UB and eliminates per-element scalar copy. **Pybind side**: allocate `torch::empty({ndim, numel}, kInt64, kNPU)`, kernel writes column-major, then `output_cm.slice(1, 0, K).transpose(0, 1).contiguous()` returns `[K, ndim]` row-major to caller. Pybind transpose is a single GM→GM strided copy of `K * ndim * 8` bytes by torch (~400 GB/s HBM → ~2.5 ms for 67M-row dense). **Determinism (A-P61)**: each block writes disjoint `[d * numel + my_offset, ... + my_count)` per dim — no race; within block, GatherMask emits in increasing pos order, SIMD decode is order-fixed, DataCopy is bulk MTE3 deterministic; pybind transpose deterministic per torch contract. **When to use**: predicate-driven emit kernels with N-D coord decode (N > 1) where the V4 P-P80 pattern is profile-confirmed bottlenecked on the scalar interleave. **When NOT useful**: ndim==1 (no interleave to avoid); kernel-natural layout already matches output contract (rare for index ops); stretch perf demand requires fused single-pass transpose-fused kernel (different architecture). Validated op#22 22_Nonzero V5 kw-2 (2026-05-03 Ascend950PR_9579): Pass A 50/50 + Pass B 10/10 bit-exact, Det 50/50, perf overall 0.3563× (median 0.2832×) vs V4 baseline 0.1603× (median 0.0368×) → **2.22× cumulative speedup**, 7.7× median improvement. Some sparse cases now beat CANN (max 1.79×). **Generalizes** to any N-D index-emit kernel sharing the column-major-coord-UB → row-major-GM scalar-pack bottleneck (where, masked_select, sparse coalesce, scatter-with-mask, sparse-COO build). **Cross-ref**: P-P80 (the V4 pattern this composite improves on — same kernel surface), PB-23 (SIMD int32 reject — int64 indices required throughout), determinism.md A-P61 (block-disjoint write ordering preserved).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P83，convert_patterns_to_okf.py）。confidence 未升格。 -->
