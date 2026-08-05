---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Offloading a scan's outer-product build or contraction to the IDLE cube unit LOSES at small contraction dim (~N=16) — the 16^3 min tile cannot be filled and the per-IterateAll fixed floor dwarfs the vec cost"
description: "A scan's contraction is a batched dot / diagonal-of-GEMM (256x tile waste) and the build is a rank-1 outer product (k=1); at N=16 cube throughput is ~0.002 TFLOPS with a ~44-69us per-IterateAll floor, so cube is >=65x slower than the vec reduce. Cube only helps with a genuine tile-filling contraction (N>=128) and cube-resident operands."
original_id: OL-246
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [scan, ssm, cube, ol-246, small-contraction, tile-underfill, selective-scan, optimization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** a SIMD scan / SSM kernel (small-N state-space, e.g. Mamba selective_scan) is vec-bound and the cube unit is idle; you are tempted to offload the matmul-LOOKING parts. `applies_to: soc=Ascend950PR (V351/arch35); cann=9.1.0.B060; bisheng=AIV+AIC; op_class=scan/SSM (small-N); dtype=fp16/bf16/fp32`. `verified_on: Ascend950PR_957b; cann=9.1.0.B060; op=selective_scan_source_a5 fwd_simd (perf-loop iter-1)`. (`unverified_on: Ascend910_V220 (A3) — cube tile granularity + KFC round-trip cost differ by arch`.)

### Principle

When a SIMD scan kernel is vec-bound and the cube unit is idle, the instinct is to offload the per-step outer-product "build" and the `Sigma_n` "reduce" to cube. This LOSES whenever the contraction/feature dim is small (~N=16), for two structural reasons the shapes decide BEFORE any prototype:

1. **The ops are not GEMM-shaped.** A scan's contraction `y[l]=Sigma_n x[l,n]*C[l,n]` is a BATCHED DOT / the DIAGONAL of a GEMM (`Y[l,l']` keep only `l==l'`) — as a full GEMM it is 256x tile-waste (compute 65536 outputs, keep 256); as per-`l` tiny mmads it is 256x tile-UNDER-fill (each fills 1 row x 1 col of the 16x16 tile). The per-`l` weight `C[l,:]` is NOT shareable, so there is no operand to amortize. The build `dA[l,n]=delta[l]*A[n]` is a rank-1 OUTER PRODUCT with k=1 (fp16/bf16 native cube k-tile=16 -> 16x k-waste); the companion `dBu=(delta*u)*B` is an elementwise Hadamard product the cube CANNOT express at all.

2. **Fixed per-call floor + small-tile throughput catastrophe.** A cube `mmad.IterateAll` has a ~44-69us PER-CALL fixed floor (KFC AIC<->AIV request/response round-trip + SetOrgShape/SetTensor + L1/L0 load + L0C->GM fixpipe), INVARIANT to K below K~=128 (M256/K16/N16 = M256/K1/N16 ~= 69us). At K=N=16 single-core throughput is ~0.002 TFLOPS (~0.0005% of the 373-TFLOPS fp16 peak) — the 16x16x16 tile is the minimum granularity and N=16 = exactly one tile-column, so there is no n-width to fill.

### Quantified verdict (the load-bearing number)

Even granting impossible-best assumptions (shareable weight + PERFECT 28-AIC linear scaling), the cube `Sigma_n` for the scan's 7.86M l-positions projects to ~32ms vs the **489us** actual VEC y-reduce -> cube is >=65x SLOWER. With the real per-l-distinct-C diagonal-of-GEMM (256x waste) + the UB->GM->cube->GM->UB round-trip on top of the vec build that already produced the operands in UB, the real gap is far worse. (Precision is NOT the issue — the cube matmul is bit-accurate, maxerr 1e-4 fp16; this is a pure perf loss.)

### Positive boundary — where cube WOULD help

Cube needs a genuine contraction that FILLS the m x k x n tile and AMORTIZES the per-call floor. For this op family that means a FUSED-multi-head variant with **N>=128** (contraction dim fills >=8 cube k-tiles) AND cube-RESIDENT operands (a shared weight reused across many `l`, so the per-call setup amortizes over real work).

## 证据
- selective_scan_source_a5 fwd_simd perf-loop iter-1 (2026-06-2x, A5 Ascend950PR_957b, CANN 9.1.0.B060): cube `Sigma_n` projected ~32ms vs 489us actual VEC y-reduce (>=65x slower) even under best-case assumptions; per-IterateAll floor ~44-69us invariant below K~=128; N=16 throughput ~0.002 TFLOPS. Perf loss only (bit-accurate, maxerr 1e-4 fp16).
