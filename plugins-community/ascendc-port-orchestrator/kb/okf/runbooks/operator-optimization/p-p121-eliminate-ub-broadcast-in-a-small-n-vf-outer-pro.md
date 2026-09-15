---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Eliminate UB `Broadcast` in a small-N VF outer-product build via in-register `Gather` — value-identical (precision-neutral), wins on large-row shapes where the broadcast cost bites"
description: "applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.1.x; op_class=SIMD/VF vec builds with a small broadcast dim (N); kernel_type=ascendc __simd_vf__ verified_on: soc=Ascend950PR_957b; cann=9.1.T500 (s"
confidence: single_run
original_id: P-P121
timestamp_inferred: true
tags: [patterns-index, optimization, broadcast, gather, arange, p-p121, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR (a5, arch 351x); cann=9.1.x; op_class=SIMD/VF vec builds with a small broadcast dim (N); kernel_type=ascendc __simd_vf__`
`verified_on: soc=Ascend950PR_957b; cann=9.1.T500 (selective_scan_fwd_simd R2, 2026-07-24, .171 card2)`

**Pattern**: an outer-product-style VF build that materializes `[L,N]` from `[L]`- and `[N]`-shaped operands typically UB-`Broadcast`s each operand up to `[L,N]` before the elementwise chain (3 `Broadcast<float,2,axis>` + their `PipeBarrier<PIPE_V>` per chunk). For **small N** (N=16 = 4 rows per 64-lane fp32 VL) you can DELETE those UB Broadcasts and generate the broadcast **logically in-register via `Gather`**: build the lane→N index once with `Arange`→`ShiftRights`(÷N)→`ShiftLefts`→`Sub` (`nIndex`, `rowInTile`), `Gather` the tile-invariant `[N]` operand once per call (`Gather(rAf, Af, nIndex)`), and `Gather` the `[L]`-varying operands per tile by `rowIndex`. The Gather reproduces the exact broadcast layout → **value-identical** (fp16/bf16 bit-identical, fp32 unchanged at its floor) = precision-neutral. Fold R1's real `Exp` into the same VF.

**Scope / no-regression**: gate on `if (N==16)` on the bf16/fp16 fast path only; keep the membase `Broadcast`+build path for general-N and the fp32 `softTrans` path (they are UNTOUCHED — no regression).

**Perf (why it wins only on large rows)**: broadcast-elimination bites where the per-row work is large. selective_scan_fwd_simd customer **L=5000 N=16 bf16**: baseline `2467.9µs` → R2 `2214.2µs` = **−10.3% (1.115×)** device-time, precision bit-identical (same-session back-to-back npu.Event A/B, median+min agree 0.1%). Small stock shapes (L≤768) are launch-bound/noisy — the reliable signal is the large customer row (≈80k elem/row vs ≤12k).

**Method note**: this op is custom-`ACLRT`-launch → `torch_npu.profiler` exports empty (DEBT-149) and in-container msprof analyzer is EPERM — measure device-time with `torch.npu.Event`, same session, back-to-back baseline-vs-opt on the SAME card (per the "same-condition A/B" rule). Reconcile the rig's built kernel to the current production md5 FIRST (OL-283) or the baseline is wrong.

**Cross-ref**: P-P106 (the L-chunk + Hillis-Steele scan build this optimizes), OL-245 (regbase amortization boundary — the Gather-build is the amortized-WIN case: one wide gather-fed chain, not high-freq tiny VF), OL-231 (A5 small-N issue-bound ceiling — this shaves a real per-chunk term under it), DEBT-149 (profiler-empty on custom-ACLRT → npu.Event), OL-283 (reconcile rig kernel to production before A/B). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P121，convert_patterns_to_okf.py）。confidence 未升格。 -->
