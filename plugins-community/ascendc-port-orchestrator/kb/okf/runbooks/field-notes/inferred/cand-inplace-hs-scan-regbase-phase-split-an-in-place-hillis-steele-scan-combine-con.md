---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "an in-place Hillis-Steele scan combine converted to VL-tiled regbase MUST split phase1(pre-cache sources)/phase2(compute+write dst) — a single pass corrupts the scan via cross-tile RAW"
description: "applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0; bisheng=AIV; op_class=scan/SSM (Hillis-Steele affine-prefix, in-place); dtype=fp32/fp16/bf16 Source: selective_scan_full_grad bwd loc1/loc2 regba"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0; bisheng=AIV; op_class=scan/SSM (Hillis-Steele affine-prefix, in-place); dtype=fp32/fp16/bf16"
confidence: inferred
status: stub
original_id: CAND-INPLACE-HS-SCAN-REGBASE-PHASE-SPLIT
timestamp_inferred: true
tags: [candidate, inferred, mul, add, count, prod, updatemask, cand-inplace-hs-scan-regbase-phase-split]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0; bisheng=AIV; op_class=scan/SSM (Hillis-Steele affine-prefix, in-place); dtype=fp32/fp16/bf16`
**Source**: selective_scan_full_grad bwd loc1/loc2 regbase conversion (2026-07-01, A5/Ascend950PR_9579, branch archived/perf/ss-bwd-regbase-loc12) | **Validation status**: whitebox-measured, behavior-neutral (grads BIT-IDENTICAL to Membase baseline), device-time verified

**The trap**: a Membase in-place Hillis-Steele scan pass — `for (stride...) { B[i+stride] += A[i+stride]·B[i]; A[i+stride] *= A[i]; }` over `count=(cl-stride)*N` elements — reads `B[i]` (a SOURCE at the low index) and writes `B[i+stride]` (a DST at the high index). In Membase this is safe: each `Mul`/`Add` processes the WHOLE `count` in one op and `PipeBarrier<PIPE_V>` orders the ops. Naively lowering it to a SINGLE regbase VF loop over VL=64 tiles BREAKS it: the loop writes `B[i+stride]` in an early tile, and a LATER tile reads `B[i']` where `i'=i+stride` (the same element, now a source) → it reads the ALREADY-OVERWRITTEN value → **cross-tile RAW hazard → corrupt scan** (silent wrong grads, not a crash).

**The fix — phase1/phase2 split (two VL-tiled VFs, shared scratch)**:
- **phase1** (read sources, write SCRATCH only): `prod[off+i] = A[off+i]·B[i]; tA[off+i] = A[i];` — reads all sources, touches no dst → no hazard.
- **phase2** (read scratch + same-index dst, write dst): `B[off+i] += prod[off+i]; A[off+i] *= tA[off+i];` — the low-index source `B[i]` was fully consumed in phase1, so phase2 only does same-index read-modify-write → no cross-tile hazard.
```
__simd_vf__ HSScanPhase1VF(A,B,prod,tA,count){ __VEC_SCOPE__{ for(i<nt){ adr=CreateAddrReg(i,VL);
  LoadAlign(a,A+off,adr);LoadAlign(b,B,adr);Mul(p,a,b,m);StoreAlign(prod+off,p,adr,m);
  LoadAlign(ai,A,adr);StoreAlign(tA+off,ai,adr,m);}}}   // sources→scratch only
__simd_vf__ HSScanPhase2VF(A,B,prod,tA,count){ ... B[off]+=prod[off]; A[off]*=tA[off]; }  // scratch→dst
```
The existing Membase already keeps `tA`/`prod` scratch (for the barrier chain) — reuse them; you are only re-partitioning the SAME data-flow into two hazard-free passes.

**General principle**: any IN-PLACE strided combine (scan / cumulative / recurrence) whose write-index overlaps a later tile's read-index cannot be a single register-tiled pass — the tile granularity re-orders the reads/writes that the Membase whole-vector op + barrier implicitly serialized. Split into a source-cache phase and a compute-write phase. Distinct from OL-245 (WHETHER regbase pays off) — this is a CORRECTNESS precondition for regbasing an in-place scan at all. The reverse-scan direction adds a companion tail-mask correctness rule (offset-0 dst → `MaskPattern::ALL` over-write corrupts the preserved tail scan-state → use `UpdateMask` remaining-count tail mask; see the UpdateMask KB entry).

**Promote when**: a 2nd in-place strided-combine regbase conversion (another scan/cumsum/recurrence) reproduces "single-pass corrupts, phase1/phase2 split restores bit-identical". Cross-ref: OL-245 (regbase amortization — the orthogonal WHETHER-it-pays question; its ~300us-est→~80us-real evidence is this same op), P-P106 (the HS affine-prefix scan structure), the UpdateMask tail-mask entry (companion reverse-scan correctness). Evidence: loc1(fwd HS)+loc2(rev HS), grads bit-identical to baseline 3 dtypes 3+20-chunk, det 5/5, ~1% device-time (`output/selective_scan_source_a5/src/kernels/selective_scan_bwd_simd/perf_evidence_regbase_loc12/`). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-INPLACE-HS-SCAN-REGBASE-PHASE-SPLIT，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
