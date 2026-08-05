---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "scan/SSM backward — cooperative parallel-prefix SIMT is the right SIMT mapping (beats naive SIMT ~165× and UN-optimized SIMD ~1.6×), BUT a fully-optimized vectorized SIMD ties/beats it — fully optimize BOTH architectures before declaring a winner"
description: "applies_to: soc=Ascend950PR; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN); scope=architecture-selection+perf; kernel_type=SIMT-cooperative verified_o"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN); scope=architecture-selection+perf;"
confidence: inferred
status: stub
original_id: CAND-SSM-BWD-COOPSIMT-PERF
timestamp_inferred: true
tags: [candidate, inferred, sum, cand-ssm-bwd-coopsimt-perf]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN); scope=architecture-selection+perf; kernel_type=SIMT-cooperative`
`verified_on: selective_scan_full_grad backward, a5 Ascend950PR_957b cann-9.1.T500 device1, 30/30 PASS all dtypes, device-time self-measured + __file__-proven (2026-06-18)`

**⚠ UPDATE 2026-06-19 — the original "coop is THE fastest" held only vs UN-optimized SIMD; after BOTH were fully optimized to the A5 vector roofline, SIMD wins.** Optimized device-time (msprof, NPU2, both 30/30): SIMD **1074µs large / 151µs small** vs coop **1082µs large / 306µs small** → tie on large, **SIMD ~2× faster on small**. The earlier "coop 1.61× faster" was real only because SIMD was still scalar-pipe-bound; once SIMD is vectorized both converge near the A5 vector roofline (~1080µs large, vec_ratio ≈ 0.99). **Production = optimized SIMD; coop kept as A/B evidence** (`output/selective_scan_source_a5/src/kernels/selective_scan_bwd_coopsimt/`).

**Optimization levers (each precision-clean, 30/30 held throughout):**
- **SIMD → 2.47×**: replace the per-l `V→S→V ReduceSum` round-trips (scalar-pipe-bound, vec_ratio 0.63) with a batched high-level `Sum<float>` + vectorized combine (vec_ratio 0.98). The `Sum` primitive runs on c310 despite the arch-guard — see **OL-230**.
- **coop → 1.54×**: `nblk = min(row_groups, 168)` oversubscription (**P-P10**, disperses cross-d atomic contention + un-caps the small case) + **Brent-Kung** work-efficient case-split scans for large + **WarpReduceAddSync** hybrid grad_A/D/db reductions (**P-P2**).
- Residual to the *true* A5 roofline (~37µs, ~29× further) needs a different execution model (vectorize over N=16 / use cube) — structural + precision-risky; both architectures plateau at vec_ratio ≈ 0.99 (vector-instruction-throughput saturated).

**Refines SIMT_VS_SIMD_DECISION/P-P9 for the scan-class**: the coarse table picks SIMD for "continuous-read + vector compute", but for scan/SSM that auto-pick is WRONG by tens-of-× *when SIMD is naive*. The forward proved SIMT wins; for the BACKWARD, the cooperative SIMT mapping (not the naive one) beats un-optimized SIMD — but a *fully-vectorized* SIMD (batched Sum) ties/wins, so don't stop at one architecture.

**Three architectures, measured (a5 device1, large fp32 (2,512,256,N16), torch.npu.Event, __file__-proven distinct .so, all 30/30 precision)**:
- naive-SIMT (1 thread = 1 whole row, serial in-thread fwd+reverse scan): **~270 ms — LOSES to SIMD ~80–165×** (single-issue scalar vs 16-lane vec; per-element dcache vs bulk DataCopy; per-element atomics).
- SIMD (regbase vector, ko'd): 2.70 ms.
- **cooperative parallel-prefix SIMT: 1.67 ms — BEATS SIMD 1.61×** (and naive-SIMT ~165×).

**Winning pattern**: a block (THREAD_NUM=512) cooperates on ONE row's L-axis, doing BOTH the forward state scan `x[l]=dA[l]·x[l-1]+dBu[l]` and the reverse adjoint scan `dx[l]=gs[l]·C[l]+dA[l+1]·dx[l+1]` as O(log L) affine-prefix trees (Hillis-Steele; associative op `(a1,b1)∘(a2,b2)=(a1·a2, a2·b1+b2)`). Mirror the forward SIMT kernel `output/selective_scan_source_a5/src/kernels/selective_scan_fwd_simt/`.

**Breakthrough gated on the atomicAdd structure, NOT the scan** (profiling-first atomics-off diagnostic): grad_A's per-(b,l) atomicAdd on the same [d,n] cell was 92% of device-time; replacing with a **cross-l block tree-reduction** (accumulate-then-ONE-atomic: N atomics/group vs L×N per row) dropped fp32-large 14.6ms→1.64ms, precision held 30/30. Project lesson "atomicAdd serialization, not thread occupancy". Scan itself (atomics-off floor 1.42ms) already < SIMD 2.70ms → cooperative-SIMT is intrinsically faster for this op-class.

**Anti-pattern**: concluding "SIMT loses / use SIMD" from a NAIVE 1-thread-per-row SIMT backward — that's the wrong mapping (loses ~165×) and does NOT mean SIMD is best. Always build+measure the COOPERATIVE parallel-prefix SIMT before deciding the scan-class architecture.

**Status 2026-06-19**: architecture comparison RESOLVED on selective_scan_full_grad — fully-optimized vectorized SIMD ≥ cooperative-SIMT (tie large / SIMD 2× small), both at the A5 vector roofline. Production ships the optimized SIMD; coop archived as A/B evidence. **Promote-to-canonical when** a second scan/SSM backward (GDN/mamba2) reproduces the nuance: cooperative-SIMT is the right SIMT mapping, but a fully-vectorized SIMD ties/beats it — so the actionable rule is "fully optimize BOTH (SIMD batched-Sum + coop cooperative-prefix) before picking", not "coop is fastest".

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SSM-BWD-COOPSIMT-PERF，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
