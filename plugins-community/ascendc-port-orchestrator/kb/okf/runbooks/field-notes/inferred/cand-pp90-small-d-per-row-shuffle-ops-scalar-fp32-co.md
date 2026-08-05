---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Small-D per-row shuffle ops — scalar fp32 compute beats over-engineered SIMD plumbing"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=elementwise-with-permute (interleave / odd-even / shuffle / per-row-permute small-D) verified_on: soc=Ascend950PR; cann=9.0"
phenomenon: build_failure
signal:
  - "elementwise op with per-row shuffle / permute / FMA where the row is small (D ≤ 128) AND the permutation pattern is non-contiguous (interleave, even-odd split,"
confidence: inferred
status: stub
original_id: CAND-PP90
timestamp_inferred: true
tags: [candidate, inferred, gather, torch_npu.npu_interleave_rope, cand-pp90]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=elementwise-with-permute (interleave / odd-even / shuffle / per-row-permute small-D)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (A3 V220 — scalar GetValue/SetValue cost may differ relative to VEC dispatch; needs A3 probe before generalizing)`

**Trigger**: elementwise op with per-row shuffle / permute / FMA where the row is small (`D ≤ 128`) AND the permutation pattern is non-contiguous (interleave, even-odd split, butterfly within row). Reference algorithm reads scalar-style: `for i in range(D): out[i] = formula(x[π(i)], y[π'(i)], ...)`.

**Recommendation**: scalar fp32 compute loop after SIMD `Cast<fp32, T>(...)` is a legitimate, maintainable implementation. Don't reach for VEC `Mul/Add` + `Gather` + scratch buffers + PipeBarriers when:

1. The natural reference is per-element scalar (not block-vectorized).
2. The permutation pattern requires `Gather` (or scalar scatter writes) anyway.
3. Total scalar ops per tile is small (D=64 → 256 scalar/row, 2048-4096/tile — sub-millisecond on AIV).

**Anti-pattern avoided** (the specific over-engineering this entry points away from): "build gather-style contiguous sub-tiles → apply VEC Mul/Add → scatter writes back" needs O(tile) scalar scatter writes anyway, plus PipeBarriers, plus scratch buffers, AND ends up matching the reference algorithm 1-to-1 only after per-element fixups. The "pure SIMD" version is more code, more buffers, more risk — and not faster than scalar at small-D.

**Trade-off (transparent)**: scalar path will not match a CANN fused op that uses specialized ISA (e.g. RoPE-specific instructions). Expect 0.5×–0.8× ratio vs `torch_npu.npu_<fused>` — acceptable when (a) the fused op exists for the reference but the port was via the generic docstring, (b) precision matters more than perf, OR (c) baseline pass is the priority and a vectorized rewrite is a follow-up optimization.

**Decision rule** (when in doubt):
- If reference docstring is a `for i in range(D)` per-element loop → scalar path is the literal translation, use it (Iron-law §5).
- If `D ≤ 128` AND permutation is non-contiguous → scalar path FIRST.
- If `D > 128` OR permutation is contiguous (concatenation, slice-and-shift) → VEC path likely wins.

**Evidence**: op#13 13_InterleaveRope (2026-04-30). Reference: `torch_npu.npu_interleave_rope` per-row interleave-then-FMA, `D = 64` fixed. Scalar fp32 loop after SIMD `Cast<fp32, T>` → 50/50 PASS bit-exact (both fp16 + bf16 paths), perf ratio 0.72× vs `torch_npu.npu_interleave_rope`. CANN fused op presumably uses specialized ISA for this exact shape — 0.72× is the cost of generic-codegen scalar vs specialized-instruction. Maintainability + correctness > 30% perf gap when the reference algorithm IS scalar.

**Promote when**: a second small-D per-row shuffle op (e.g. odd-even shuffle, butterfly-within-D layouts in attention variants) confirms the same scalar-beats-SIMD-plumbing decision. Likely co-promotes with P-P9 (SIMD vs SIMT decision framework) into an OL with explicit small-D scope clause.

**Source**: op#13 13_InterleaveRope kw-1 (2026-04-30). 1-op evidence; author's own promotion gate cited in `output/npukernelbench/src/kernels/13_InterleaveRope/knowledge_update.md`.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP90，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
