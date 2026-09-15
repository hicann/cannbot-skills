---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Vectorized index-compression emit via GatherMask for predicate-driven kernels"
description: "For kernels of the shape \"emit positions where predicate is true\" (e.g. nonzero, where, scatter-with-mask, sparse compress), the canonical multi-core SIMD V4 pattern is: (1) CompareScalar over fp32-pr"
severity: high
confidence: single_run
original_id: P-P80
timestamp_inferred: true
tags: [scatter_add, optimization, comparescalar, rsvdcnt, p-p80, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

For kernels of the shape "emit positions where predicate is true" (e.g. nonzero, where, scatter-with-mask, sparse compress), the canonical multi-core SIMD V4 pattern is: (1) `CompareScalar` over fp32-promoted source → packed bitmask UB (one bit per element, LSB-first); (2) `ArithProgression<int32>(pos_local, 0, 1, tile_aligned)` to materialize local positions [0, tile); (3) `GatherMask<int32, uint32>(pos_compressed, pos_local, ReinterpretCast<uint32_t>(mask_packed), reduceMode=true, mask=tile_size, params={1,1,0,0}, rsvdCnt)` — compresses positions where bitmask is set into a dense prefix; `rsvdCnt` returns count by reference; (4) `Cast<int64,int32>(flat64, pos_compressed)` + `Adds<int64>(flat64, base_offset)` to convert to global flat index; (5) per N-D dim, `Divs<int64>` / `Muls<int64>` / `Sub<int64>` chunked at 32B-aligned CHUNK_ROWS=256 for UB budget; (6) scalar pack into row-major out_ub + DataCopy bulk emit to GM. **Determinism**: GatherMask emits in increasing input-position order → A-P61 fixed-order multi-core merge ALLOWED when each block owns disjoint input/output ranges. **Caveats**: (a) ALL SIMD binary-scalar ops MUST use int64 (not int32) per PB-23; (b) chunk-base addresses MUST be 32B-aligned via fixed CHUNK_ROWS stride, NOT partial-chunk size (causes runtime error 340 unaligned access); (c) dense-case GM write bandwidth dominates — for sparse (<25% true) reaches ~0.5-0.9× CANN, but for dense workloads a different design (dim-major direct stream, no UB staging) is needed. Validated op#22 22_Nonzero V4 kw-5 (2026-04-30): probe-first verified GatherMask API contract (`probes/gathermask_probe/RESULTS.md`), then V4 kernel landed 50/50 + 10/10 + det 50/50 PASS, median 0.0108×, p90 0.5319×, max 0.9487× (case 19). **Generalizes to**: any "compress positions matching a predicate" kernel — torch.where, torch.masked_select, sparse coalesce, scatter-with-mask. **Cross-ref**: PB-23 (SIMD int32 reject), PB-20 (GM write workarounds for SetValue), P-P57 (SIMD ReduceMax for top-k), determinism.md A-P61 (fixed-order multi-core merge ALLOWED).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 路由表行（无 domains 正文）（P-P80，convert_patterns_to_okf.py）。confidence 未升格。 -->
