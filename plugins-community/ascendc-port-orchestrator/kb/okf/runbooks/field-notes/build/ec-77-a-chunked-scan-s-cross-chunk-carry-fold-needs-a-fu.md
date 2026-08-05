---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A chunked scan's cross-chunk carry fold needs a FULL `PipeBarrier<PIPE_ALL>` (not `PIPE_V`) before the wide intra-chunk scan reads it — at large state width the carry write has not drained"
description: "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=scan"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=scan"
confidence: single_run
original_id: EC-77
timestamp_inferred: true
tags: [pipe_v, ascendc, ec-77]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=scan`

When an L-chunked scan carries recurrence state across chunk boundaries by folding the incoming carry into position 0 of the chunk buffer (`Add(Bscan[0:N], dA[0:N]*xst)`) and then runs a WIDE parallel (Hillis-Steele) scan that reads that buffer, a `PipeBarrier<PIPE_V>` between the fold and the scan is **insufficient at large state width N** → deterministic carry corruption. Distinctive signature: **chunk 0 is correct** (its carry-in is zero, the fold is a no-op) while **chunk 1+ are wrong**; the corruption is deterministic and N-dependent (masked at small N where the working set drains within pipeline time); and a pure-fp64 simulation of the exact algorithm is CORRECT (so it is an on-device execution/fence issue, not an algorithm bug). Fix = `PipeBarrier<PIPE_ALL>` at the fold.

**Method note (saves a dead end)**: a sub-granule VEC-offset-alignment hypothesis (HS `off = stride*N` non-64-aligned at N>16) was A/B-REFUTED by a contiguous control; the real cause was the RAW fence. When chunk0-right/chunk1-wrong + algorithm-numpy-correct + N-only, suspect the carry-path fence before suspecting alignment.

**Evidence**: selective_scan fwd-SIMD L-chunk (2026-06-24, PR #52). N=32 multi-chunk (L≥257) deterministic-wrong with `PIPE_V`; `PIPE_ALL` at the fold → N∈{32,64} multi-chunk all 0-wrong. Sibling of the cross-row / cross-iteration V→MTE2 fences (PB-47, PB-49).

**Other instances (predicted)**: any chunked/tiled scan or recurrence that folds a cross-tile carry into a buffer immediately consumed by a wide multi-pass vector op — prefix-sum tiling, segmented scan, attention online-softmax running-stat carry.

<!-- 迁移自 porter kb/target/ascendc/（EC-77，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
