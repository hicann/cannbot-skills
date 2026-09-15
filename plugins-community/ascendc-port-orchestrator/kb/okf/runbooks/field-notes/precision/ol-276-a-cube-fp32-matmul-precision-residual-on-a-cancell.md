---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "A Cube fp32 matmul precision residual on a cancellation case is `baseK`-tiling-INVARIANT — it is reference-ub (accumulation order vs the vendor Mmad), not a tunable, so do not sweep tiles to fix it"
description: "applies_to: soc=Ascend910_9382 (V220); cann=9.1.0; bisheng=n/a; op_class=matmul; dtype=float32"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend910_9382 (V220); cann=9.1.0; bisheng=n/a; op_class=matmul; dtype=float32"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-276
timestamp_inferred: true
tags: [basek, ascendc, ol-276]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220); cann=9.1.0; bisheng=n/a; op_class=matmul; dtype=float32`
`verified_on: soc=Ascend910_9382; cann=9.1.0 (1_BatchMatmul DEBT-206 2026-07-13; independently reproduced by the same-chip a3-ds sibling variant-sweep)`
`unverified_on: soc=Ascend950PR (V351 / arch35 — the L0C-accumulator invariance is expected to transfer but re-verify before relying on it)`

**Principle**: On a Cube (AIC) fp32 matmul the L0C accumulator sums each `baseK` partial-product chunk into the SAME fp32 accumulator regardless of how K is tiled, so the output is **byte-identical across `baseK` choices** (64 vs 128 = 4×64 vs 2×128 partials → bit-for-bit identical result). Therefore, when an fp32 matmul misses a near-zero (cancellation-region) case by a hair versus CANN, retiling `baseK` is **NOT a lever** — the residual is the Mmad accumulation ORDER differing from the vendor's Mmad, i.e. reference-ub (a representation-floor artifact), neither a bug nor a tunable. Do not burn optimizer/probe iters sweeping tiles; go straight to an fp64-truth triage (is `|ours − fp64_truth| ≤ |CANN − fp64_truth|`?) to classify it, and report `PARTIAL_PASS_WITHIN_TOLERANCE` if ours is at/under the vendor's distance to truth. This is the matmul-specific companion to the general fp32-cancellation-is-a-range-edge rule.

**Concrete anchor**: fp32 `[1,128,256]×[1,256,128]`, 2 output elements in the cancellation region — `baseK` 64→128 gave byte-identical `ours_mere` 6.317e-6; the fp64-truth triage showed `|ours−true|` mean 1.611e-6 ≤ `|CANN−true|` 1.631e-6 (ours marginally CLOSER to truth) → reference-ub confirmed, kernel byte-identical, no tile change helps.

**Evidence**: 1_BatchMatmul (DEBT-206, 2026-07-13, Ascend910_9382 / CANN 9.1.0): `baseK` sweep byte-invariant on the failing case; fp64-truth triage ours ≤ CANN vs truth; independently reproduced by the same-chip a3-ds sibling's variant-sweep ("no tiling fix improves case[4]").

**Other instances (predicted)**: any AIC fp32 GEMM/BMM/conv-as-GEMM with a near-zero (cancellation) golden element; fp32 accumulation in `MatmulImpl<>` / `matmul::Matmul<>` regardless of K-tiling. Does NOT apply to fp16/bf16 accumulation (different accumulator width/rounding) nor to a genuine formula bug (which DOES move under a correct-formula edit).

**Cross-ref**: OL-112 (fp32 op-order is load-bearing — vec chains), OL-191 / §5.4 (fp32 near-ceiling summation-order cancellation = dtype-range edge → `PASS_WITHIN_TOLERANCE`), OL-275 (the cube that produced this — `AIC_ONLY` single-cube matmul), EC-59 (declare an INCLUSIVE `pass_a.status` so a T2-promoted fp32 matmul does not O5-rollback).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-276（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
