---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "SIMD-over-rows / row-packing for a small-N scan LOSES at sub-granule N — the mandatory sub-granule strided transpose tax dwarfs the scan it accelerates"
description: "applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32 Source: selective_scan_source_a5 fwd-SIMD perf-loop"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32"
confidence: inferred
status: stub
original_id: CAND-SIMD-OVER-ROWS-SUBGRANULE-REPACK-TAX
timestamp_inferred: true
tags: [candidate, inferred, cand-simd-over-rows-subgranule-repack-tax]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.T500; bisheng=AIV; op_class=scan/SSM (small-N state-space, e.g. Mamba selective_scan); dtype=fp32`
**Source**: selective_scan_source_a5 fwd-SIMD perf-loop iter-3 (2026-06-23, A5/Ascend950PR_957b) | **Validation status**: anti-pattern, whitebox-measured, NOT pursued

**Concept (the instinct)**: a small-N serial-in-L scan under-fills the vector lanes (N=16 on a 64-fp32-lane unit = ¼ lane utilization). The instinct is to pack R rows' N-state side-by-side so R×N fills the 128-lane width, run the per-step combine over the packed vector, and amortize the scan over R rows at once.

**Why it LOSES (the sub-granule repack tax)**: packing R rows requires interleaving the rows' N-state at N=16 element granularity, but **N=16 (64 B fp32) < the A5 fp32 vector granule (64 fp32 = 256 B)**. So the pack/unpack is a MANDATORY sub-granule strided transpose — the hardware cannot gather/scatter at 16-element stride within a granule cheaply, it must do a strided element-shuffle. Measured: the strided transpose runs **3× per chunk** (pack-in, the carry layout, unpack-out) and each costs ~10× the scan it is meant to accelerate.

**Measured (A5/Ascend950PR_957b, 2026-06-23, msprof device-time)**:
- Isolated R8 packed-serial scan ALONE: **0.68×** (i.e. faster — the packed scan itself is a win in isolation, confirming the lane-fill instinct is directionally real).
- BUT + the mandatory repack (the 3×/chunk sub-granule transpose): **7.39× SLOWER** end-to-end. The transpose tax is ~10× the scan, so the net is a large regression.
- Also a **correctness hazard**: the sub-granule VEC offset is exactly the OL-221 / EC-22 sub-granule-VEC-offset trap (a packed layout that reads/writes at a non-granule-aligned offset silently corrupts).

**Boundary (where it WOULD work)**: realizable only for **dstate (N) ≥ 64** — at full-granule N the packing needs NO sub-granule transpose (rows already align to the granule), so the tax vanishes and the lane-fill win can land. For N=16 (single-head Mamba scan) it is a net loss.

**Promote when**: a 2nd small-N (N≤32) serial-L scan/SSM reproduces the "packed-scan-fast-in-isolation but repack-tax-net-negative" measurement, OR an N≥64 variant lands the packing win (confirming the boundary). Cross-ref: OL-231 (the issue-bound architecture-floor anchor — this is one of the 8 levers that fail on a small-N serial-L SSM), OL-221 / EC-22 (sub-granule VEC-offset correctness trap this would trip), OL-245 (regbase amortization boundary — same "MEASURE the per-call work before rewriting" discipline), P-P106 (the scan structure). Whitebox trace: `workspace/ss_perf_loop/whitebox_log.md` (iter-3). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SIMD-OVER-ROWS-SUBGRANULE-REPACK-TAX，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
