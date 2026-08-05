---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Interleaved-pair RoPE via GatherMask even/odd split + symmetric Mul(cos)+Mul(sin) (single-pass, no transpose)"
description: "applies_to: any soc with public AscendC GatherMask + Mul + Add + BinaryRepeatParams; cann=9.0.0+; op_class=rotary_position_embedding / pairwise_rotation_on_interleaved_layout derived-from: cann-source"
phenomenon: build_failure
signal:
  - "Op needs RoPE rotation on a tensor laid out as [row, col] where each row's col elements are interleaved pairs (x0, x1, x2, x3, ...) with the rotation defined as"
confidence: inferred
status: stub
original_id: CAND-MLA-4
timestamp_inferred: true
tags: [candidate, inferred, col, mul, cand-mla-4]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with public AscendC GatherMask + Mul + Add + BinaryRepeatParams; cann=9.0.0+; op_class=rotary_position_embedding / pairwise_rotation_on_interleaved_layout`
`derived-from: cann-source (mla-class prolog, 2026-05-10 multicann)`
`verified_on: cann ops-transformer attention/mla_prolog/op_kernel/service_rope.h (RotaryPosEmb, ~75 lines, public-API only — GatherMask, Mul, Add, BinaryRepeatParams, GatherMaskParams), attention/mla_prolog/docs/aclnnMlaProlog.md (formula block: q^R = ROPE(c^Q · W_QR) on a 64-wide head dim, sin/cos shape (B,S,Dr))`
`unverified_on: a5_ops (a5_ops has 1_RotaryMul which uses a different concrete shape — verify alignment before promoting)`

**Trigger**: Op needs RoPE rotation on a tensor laid out as `[row, col]` where each row's `col` elements are interleaved pairs `(x0, x1, x2, x3, ...)` with the rotation defined as `(x_2i, x_2i+1) -> (x_2i*cos - x_2i+1*sin, x_2i*sin + x_2i+1*cos)`. The sin/cos coefficient tensors are pre-broadcast to length `col` per row with the convention that `cos[:col/2]` and `cos[col/2:]` are the two halves used by the two `Mul` calls (sin/cos are NOT per-pair scalars; they are per-half vectors). `col` is small (e.g. 64 for MLA — `Dr`-dimension rope), small enough to fit several rows in UB. `col` must be a multiple of `ALIGN_BLOCK_SIZE / sizeof(C)` (32B / element-size).

**Recommendation**: Implement RoPE in a single vec pass with NO explicit transpose / reshape — use `GatherMask` with patterns `1` (odd-indexed, picking `x_0, x_2, ...`) and `2` (even-indexed, picking `x_1, x_3, ...`) to materialize the two half-vectors in scratch UB, then issue four `Mul` calls and one `Add`:

  - `evenHalf = GatherMask(input, mask=1)`  — selects `x_0, x_2, x_4, ...`
  - `oddHalf  = GatherMask(input, mask=2)`  — selects `x_1, x_3, x_5, ...`
  - `tmp0 = evenHalf * cos[lower half]`  — strided mul, blockNumPerRowHalf source stride
  - `tmp0_high = oddHalf * cos[upper half]` — written to `outputLocal[col/2 :]`
  - `tmp1 = oddHalf * sin[lower half]` — written to sin-scratch
  - `tmp1_high = evenHalf * sin[upper half]` — written to sin-scratch[col/2 :]
  - `out = tmp0 + tmp1` (single Add over the full `row*col` count — order-irrelevant per element)

The `BinaryRepeatParams` use `src0BlkStrideIn=1, src1BlkStrideIn=1, dstBlkStrideIn=1` with `src0RepStrideIn = blockNumPerRowHalf` (the gathered half-vector's per-row block count) and `src1RepStrideIn = blockNumSinCosRepStride` (the sin/cos tensor's per-row block count). This lets a SINGLE `Mul` call cover multiple rows in one vec instruction — the inner repeat axis is rows, not columns. Critical for amortizing the vec-instruction overhead when `col` is small (`Dr=64` means `col/2 = 32` elements per Mul, which is one repeat-iteration; rows are the parallelism axis).

The `shareTmpUb` buffer holds the two reinterpret-cast scratches: `reArrLocal[2*row*col/2]` for the gathered halves and `outputLocalSinTmp[row*col]` for the sin-side Mul outputs. Sized `2 * row * col * sizeof(C)` bytes total. The output can alias the input — the gather has already materialized the rearrangement, so overwriting input mid-pass is safe.

**Concrete anchor** (verified pattern; assumes `col` is a multiple of 32B/element, `sinCosRepStride` is the per-row stride of the sin/cos arrays in elements):
```cpp
// rsvdCnt is the GatherMask output count for residue tracking
uint64_t cnt = row * col;
uint64_t rsvdCnt = 0;
LocalTensor<C> reArr   = sharedTmp.ReinterpretCast<C>();
LocalTensor<C> sinTmp  = sharedTmp.ReinterpretCast<C>()[cnt];
GatherMaskParams gp { 1, 1, 0, 0 };  // 1 repeat, src0 stride 1, no rep strides

// Materialize the two halves of every row in one scratch
GatherMask(reArr,          input, /*mask=*/1, true, cnt, gp, rsvdCnt);
GatherMask(reArr[cnt >> 1], input, /*mask=*/2, true, cnt, gp, rsvdCnt);
AscendC::PipeBarrier<PIPE_V>();

uint8_t bpr     = col / (32 / sizeof(C));   // blocks per row
uint8_t bprHalf = bpr >> 1;
uint8_t bprSinCos = sinCosRepStride / (32 / sizeof(C));
BinaryRepeatParams mp { 1, 1, 1, bpr, bprHalf, bprSinCos };

// rows are the outer repeat, cos/sin sit at strided per-row offsets in their tensor
Mul(output,             reArr,             cos,             col >> 1, row, mp);
Mul(output[col >> 1],   reArr[cnt >> 1],   cos[col >> 1],   col >> 1, row, mp);
Mul(sinTmp,             reArr[cnt >> 1],   sin,             col >> 1, row, mp);
Mul(sinTmp[col >> 1],   reArr,             sin[col >> 1],   col >> 1, row, mp);
AscendC::PipeBarrier<PIPE_V>();
Add(output, output, sinTmp, cnt);
```

**Why it works**:
- The two `GatherMask` calls with masks `1` and `2` decompose the interleaved layout into two contiguous half-tensors with no explicit transpose / no DataCopy — `GatherMask` is a single vec instruction that materializes the gathered output by stride-selection, far cheaper than a Transpose op or a strided DataCopy.
- The four `Mul` calls each use the SAME `BinaryRepeatParams` with `repeatTimes=row` — the cost per row is one `Mul` instruction over `col/2` elements, with the row dimension absorbed by the repeat. For small `col` (e.g. Dr=64), this is ~16x more efficient than four vec-instruction-per-row.
- The `(cos, sin)` per-half encoding (cos lower half = cos for x_{2i}, cos upper half = cos for x_{2i+1}) sidesteps the negation that a naive `(x_2i*cos - x_2i+1*sin)` formula would need — the negation is absorbed into the sin tensor's pre-broadcast layout (sin upper half is the negated counterpart). The four-Mul-plus-Add shape is symmetric and uses NO negate primitive.
- Output aliasing input is safe because both `GatherMask` calls retire before any `Mul` reads `reArr`, and the `Add` reads `output` and `sinTmp`, neither of which is `input`. The `PipeBarrier<PIPE_V>` between gather and mul (and between mul and add) is the only required intra-core ordering.

**Determinism**: Each output element is `out[i] = a*cos_i + b*sin_i` with one Mul-and-one-Mul-and-one-Add per element — no reduction across elements, deterministic by construction. The four `Mul` calls touch disjoint regions of `output` and `sinTmp` and may be issued in any order; the `Add` is per-element and order-irrelevant.

**Other instances predicted**:
- MLA-prolog rope head (this verified instance) — Dr=64 interleaved-pair rope
- Standard LLaMA / Qwen / DeepSeek rope heads — same interleaved-pair convention, typically Dh=64 or 128
- Any pairwise-rotation pattern (complex multiplication on a real-valued interleaved layout): `(a + b*i) * (c + d*i) = (ac - bd) + (ad + bc)*i`
- Spherical / hyperspherical rotations expressed in pair-of-coordinates form
- Audio / vision positional encodings that use sinusoidal pairwise mixing on small head dims

**Risks before promotion**:
- `col` must be a multiple of `32B / sizeof(C)` (2 for fp32, 16 for bf16). For Dr=64 in bf16, `col/2 = 32` elements = 64B = 2 blocks — works cleanly; for non-aligned `Dr`, the gather pattern needs padding (out-of-scope here).
- The "interleaved-pair" layout convention `(x0,x1,x2,x3,...)` differs from the "half-half" convention `(x0,x1,...,xn/2, xn/2+1, ..., xn-1)` used by some HuggingFace rope variants. Verify the input layout convention against the reference before promoting — the gather masks 1/2 are correct ONLY for interleaved-pair.
- `sinCosRepStride` is the per-row stride of the cos/sin tensor IN ELEMENTS, not bytes; mis-specifying it produces all rows reading the same sin/cos row (silent staleness).
- The `BinaryRepeatParams` strides assume the sin/cos tensor has a separate row-stride from the gathered half. If sin/cos are broadcast (same row used for all `row` repeats), set `bprSinCos = 0`. The MLA reference uses non-zero stride; the broadcast case is a simplification that needs its own verification.
- The four `Mul` calls share the same `mp` — if rows have different per-row sin/cos (i.e. sin/cos are per-row from upstream gather), this is correct; if sin/cos are batch-broadcast, the rep stride configuration changes. Verify upstream sin/cos shape.
- a5_ops 1_RotaryMul exists but uses a different concrete shape (a5_ops is single-row, non-strided); the candidate is structurally similar but the strided multi-row form is not yet exercised.

**Cross-reference**:
- CAND-MLA-1 (latent-prolog skeleton) — the rope head is one of the two consumers of the fused up-projection in the skeleton; this candidate is the rope-head's concrete implementation
- CAND-MLA-2 (paged scatter cache) — the rope output is one of the two tensors scattered into the cache via CAND-MLA-2
- a5_ops 1_RotaryMul — closest existing benchmark; a port that adopted the gather-mask-1/2 + four-Mul shape would be the promotion vehicle for this candidate
- a5_ops 12_KvRmsnormRopeCache — the fused norm+rope+cache shape that combines CAND-MLA-1+3+4+2; if it ever lands as a real port, all four candidates promote together

**Promote when**: an a5_ops rope-shape op ships using this exact gather-mask-1/2 + four-Mul + Add shape, AND vec-instruction count vs an explicit `Transpose + per-row Mul` baseline is measured (the candidate claims ~2-4× vec-instruction reduction for small Dr; verify on a target shape), AND the sin/cos layout convention (interleaved-pair, per-half encoding) is verified against the upstream model's rope convention.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-MLA-4，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
