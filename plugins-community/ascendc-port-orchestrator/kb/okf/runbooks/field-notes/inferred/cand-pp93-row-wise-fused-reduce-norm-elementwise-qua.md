---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Row-wise fused `<reduce-norm> + <elementwise> + <quant>` via 2-pass single-tile pattern when row fits one tile"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=fused-norm-quant (row-reduction + per-element scaling + final integer-quant cast chain) verified_on: soc=Ascend950PR; cann="
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=fused-norm-quant (row-reduction + per-element scaling + final integer-quant cast ch"
confidence: inferred
status: stub
original_id: CAND-PP93
timestamp_inferred: true
tags: [candidate, inferred, inv_rms, gamma, beta, scale, cand-pp93]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5+2026-04-13; op_class=fused-norm-quant (row-reduction + per-element scaling + final integer-quant cast chain)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (V220 register-file & TBuf budget differ; pattern likely transfers but not probed); D > TILE_FP16 (multi-tile row chunking — different pipeline structure required); bf16 + per-channel scale combo (rms_norm_quant covered only bf16 per-tensor)`

**Principle**: when a fused op decomposes as `row_reduce_then_normalize(x) → per_element_affine(gamma, beta) → quant_cast_to_int8(scale, offset)` AND every row fits in one UB tile (D ≤ TILE_FP16 = 4096 for fp16/bf16), the simplest correct pattern is a 2-pass row-resident loop:
- **Pass 1**: load row x → cast fp32 → square → reduction-sum (BinaryFoldReduceSum, P-P47) → derive scalar `inv_rms` from the per-row sum
- **Pass 2**: reload row x → multiply by `inv_rms` → multiply by gamma → add beta → divide by scale → add offset → CAST_RINT to int32 → SetDeqScale + CAST_NONE to fp16 → CAST_TRUNC to int8 → store

No tile-inner loop within a row, no MTE2/VEC inter-tile pipeline, no flash-attention-style chunked accumulation. The pattern is a clean composition of three existing primitives — **P-P45/47** (row-reduction primitive), **P-P46** (quant cast chain fp32→int8 via the RINT/SetDeqScale/TRUNC sequence; see OL-81 for `CAST_RINT = IEEE RNE` correctness), and the row-resident structure proven in `18_FusedAddRmsnorm`.

**Concrete anchor** (rms_norm_quant kw-1 kernel skeleton, 2026-05-13):
```cpp
// Pass 1: row reduce
DataCopy(xLocal, gmX[row * D], D);                  // VECIN
Cast(xF32, xLocal, RoundMode::CAST_NONE, D);        // fp16→fp32 or bf16→fp32
Mul(sqLocal, xF32, xF32, D);
BinaryFoldReduceSum(sumLocal, sqLocal, D);          // P-P47 half-interval tree
float inv_rms = 1.0f / sqrtf(sumLocal.GetValue(0) / D + epsilon);

// Pass 2: row apply + quant cast
DataCopy(xLocal2, gmX[row * D], D);
Cast(xF32_2, xLocal2, CAST_NONE, D);
Muls(yF32, xF32_2, inv_rms, D);
Mul(yF32, yF32, gammaF32, D);
Add(yF32, yF32, betaF32, D);
Div(yF32, yF32, scaleF32, D);                       // OL-82: keep literal Div, don't pre-compute 1/scale
Add(yF32, yF32, offsetF32, D);                      // offsetF32 is host-side widened (OL-137)
Mins(yF32, yF32, 127.0f, D); Maxs(yF32, yF32, -128.0f, D);
Cast(i32, yF32, RoundMode::CAST_RINT, D);           // OL-81
SetDeqScale(static_cast<half>(1.0f));
Cast(fp16Tmp, i32, RoundMode::CAST_NONE, D);
Cast(out_int8, fp16Tmp, RoundMode::CAST_TRUNC, D);
DataCopy(gmY[row * D], out_int8, D);
```

**Host-side responsibilities** (template):
- Align32 D for kernel UB alignment
- Broadcast scalar/short scale/offset to length D when reference uses per-tensor variant (single-code-path kernel)
- Int8 offset → fp32 widen at pybind (OL-137) to avoid in-kernel Cast(fp32, int8)
- Persistent TBuf<VECCALC> for `gamma`, `beta`, `scale`, `offset_f32` loaded once per kernel launch
- TQue<VECIN, depth=2> for `x` to pipeline Pass 1 / Pass 2 reloads (OL-63)

**Evidence**:
- rms_norm_quant kw-1 (2026-05-13): 8/8 Pass A + 8/8 Pass B PASS; perf 6.60× over Path-A reference (CPU-truth Model.forward decomposition path executing ~10 PyTorch primitives sequentially). First-try success with no compile-fix or precision-fix iter.

**Other instances (predicted)**: LayerNormQuant variants (replace RmsNorm step with mean-subtract+var-divide reduce), GroupNormQuant variants with per-group reduction (group_size ≤ TILE_FP16 case), AddRmsNormQuant (extra Add op fused into Pass 2 prologue), per-token quant variants of any row-wise norm.

**Promotion gate**: requires ≥2 independent ops in this family verified PASS. rms_norm_quant is instance #1. Candidate ops for second-instance evidence: AddRmsNormQuant, GroupNormSiluQuant (norm+activation+quant; OL-69 covers the activation-cost framing). Promote to `patterns/domains/quant.md` or `patterns/domains/fused_norm.md` as a P-P entry once second-op evidence is logged.

**Cross-ref**:
- P-P45 (single-pass UB-resident dynamic quantization — this is the 2-pass row-resident cousin)
- P-P46 (quantize cast chain fp32→int8)
- P-P47 (half-interval tree reduction via BinaryFoldReduceSum)
- P-P51 / P-P52 (fp32 promotion in compute path; output cast back to int8 only at the very end)
- OL-63 (TQue depth for pipeline overlap)
- OL-69 (norm + activation fusion cost analysis — generalizes to norm + quant: activation/quant tail adds only a handful of VEC instructions, end-to-end gain comes from eliminating intermediate GM round-trip)
- OL-81 (CAST_RINT = IEEE RNE — bit-exact match to `torch.round` chain)
- OL-82 (no math-equivalent rewrites without minimal repro — kept literal `Div(scale)`, not `Muls(1/scale)`)
- OL-127 (no single-thread SIMT as final state — kernel uses nblk=56 row partitioning)
- OL-137 (host-side dtype-widening for int8 offset)
- PB-9 (DataCopy UB→UB unsafe — used `Adds(dst, src, 0.0f, count)` bridge)

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP93，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
