---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "On A5/arch35, SIMD compute CHAINS default to REGBASE, not Membase/LocalTensor — but regbase wins require large per-VF-call amortization (measure, don't blind-rewrite)"
description: "Membase round-trips every intermediate through UB (store+reload + PipeBarrier per op); regbase (MicroAPI RegTensor) loads once at the chain head and stores once at the tail. Regbase wins only when its ~0.42us/call VF entry/exit overhead amortizes over a deep chain and/or many tiles — high-frequency small-tile ops go SLOWER."
original_id: OL-245
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [simd, regbase, membase, ol-245, arch35, micro-api, vec-utilization, optimization]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**When to consult:** authoring or optimizing a SIMD compute chain (>=2 dependent vector ops) on A5 / arch35. `applies_to: soc=Ascend950PR; cann=9.1.T500; bisheng=AIV; op_class=all (SIMD)`. `verified_on: Ascend950PR_957b; cann=9.1.T500; op=selective_scan_source_a5 (fwd_simd + bwd_simd)`. (`unverified_on: Ascend910_V220 (A3) — MicroAPI/regbase availability differs by arch; A3 has low-level intrinsics, not the arch35 MicroAPI RegTensor surface`.)

### Principle

On A5 / arch35 (Ascend950PR), a SIMD **compute chain** (>=2 dependent vector ops) DEFAULTS to **regbase** (MicroAPI: `__VEC_SCOPE__` + `AscendC::MicroAPI::RegTensor<float>` + `LoadAlign` head / reg->reg `Mul`/`Add`/... / `StoreAlign` tail), NOT Membase/LocalTensor.

Membase (the LocalTensor `Add(dst,a,b,count)` / `Mul` / `Cast` style) READS operands from UB, WRITES each intermediate back to UB, and RELOADS it from UB for the next op — a redundant UB store+reload **plus a `PipeBarrier<PIPE_V>` per dependent step**. For an N-op chain that is ~N redundant UB round-trips + N barriers of pure overhead that scales with chain length. Regbase loads UB->RegTensor ONCE at the chain head, keeps the whole chain in registers, and stores back to UB ONCE at the tail — the intermediate round-trips and their per-step barriers vanish. Membase stays acceptable ONLY for trivial single-op (no dependent successor) or bitwise/type-punning cases where the LocalTensor high-level API is the documented A5 path.

### BOUNDARY — regbase wins require LARGE per-VF-call amortization (decision rule, measured 2026-06-23)

Regbase is the default to CONSIDER, NOT a blind rewrite. The regbase win comes WITH a fixed cost: each `__VEC_SCOPE__` / `__simd_vf__` invocation pays a per-call entry/exit overhead measured at **~0.42us/call** on A5 (Ascend950PR_957b, CANN 9.1.0.B060). The regbase advantage holds only when that fixed per-call cost is AMORTIZED over LARGE per-call work — a deep dependent chain AND/OR many VL-tiles processed inside ONE VF call. The kw MUST check the per-call work before rewriting:
- **Amortized (regbase WINS)**: one VF call does a deep chain over many tiles. Measured on an isolated 8-op fp32 chain (CN=4096, ONE VF call): regbase **1.26x faster** (depth-2: 2.4x), bit-identical output. The mechanism is reduced per-op UB-operand-staging cost (scalar_ratio 2.9% -> 0.1% + vec_time -16%) — NOT barrier elimination (explicitly REFUTED: membase-no-barrier measured the same 7940us as membase-with-barrier).
- **High-frequency small-tile (regbase LOSES — MEASURE, don't blind-rewrite)**: an op that invokes the VF on MANY small/shallow chunks pays the 0.42us/call overhead repeatedly and it EXCEEDS the UB-round-trip saving. Measured: the selective_scan fwd-SIMD build (~30720 VF calls on 64-tile, 3-op-shallow chunks) went **1.27x SLOWER** with regbase (40282us -> ~51xxx us).

## 证据
- selective_scan_source_a5 fwd_simd + bwd_simd (2026-06-23, A5 Ascend950PR_957b, CANN 9.1.0.B060): isolated 8-op fp32 chain (CN=4096, one VF call) regbase 1.26x faster / depth-2 2.4x, bit-identical; fwd-SIMD ~30720 shallow VF calls regbase 1.27x SLOWER — per-call amortization is the discriminator.
