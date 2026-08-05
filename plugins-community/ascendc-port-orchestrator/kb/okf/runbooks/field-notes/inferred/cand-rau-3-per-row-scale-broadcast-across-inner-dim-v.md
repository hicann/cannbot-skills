---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Per-row scale broadcast across inner dim via non-zero src1Stride in BinaryRepeatParams — Brcb-free broadcast for `[R, inner]` *= `[R, 1]` shape"
description: "applies_to: any SoC with public AscendC VEC Mul/Add and BinaryRepeatParams; cann=9.0.0+; op_class=row_broadcast_apply / online_softmax_output_rescale / attention_output_combine derived-from: cann-sour"
phenomenon: build_failure
signal:
  - "Apply a per-row scalar scale (scale[R, softmax_tail] with softmax_tail ≤ 1 fp32 block = 8) to a per-row vector (data[R, inner] with inner ≥ several fp32 blocks)"
confidence: inferred
status: stub
original_id: CAND-RAU-3
timestamp_inferred: true
tags: [candidate, inferred, inner, head_dim, embed_dim, softmaxtempbuf, data, cand-rau-3]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with public AscendC VEC Mul/Add and BinaryRepeatParams; cann=9.0.0+; op_class=row_broadcast_apply / online_softmax_output_rescale / attention_output_combine`
`derived-from: cann-source (ring-attn-class update, 2026-05-10 multicann)`
`verified_on: cann ops-transformer/attention/ring_attention_update.h AttnCompute body (source-structure-only; no a5_ops measurement)`
`unverified_on: a5_ops`

**Trigger**: Apply a per-row scalar scale (`scale[R, softmax_tail]` with `softmax_tail ≤ 1 fp32 block = 8`) to a per-row vector (`data[R, inner]` with `inner` ≥ several fp32 blocks). The classic shape arising in online-softmax output rescale: every row has one (or up to 8) scale values, every row also has `head_dim` (or `embed_dim`) data values, and we must compute `data[r, j] *= scale[r, jj_mod_softmax_tail]` for `r ∈ [0, R)`, `j ∈ [0, inner)`.

**Why an alternative to Brcb (P-P62) is useful**: Brcb requires `R % 8 == 0` and a separate scratch tensor of size `R * 8 * inner_blocks` worth of broadcast results. For attention-class kernels where `inner = head_dim ∈ {64, 128, 256}` is large and `R = seqNumLoop` (the row count per AIV tile) is naturally a multiple of 8, the BinaryRepeatParams stride-based form sidesteps the Brcb entirely and reuses `softmaxTempBuf` (8 elements/row) in place.

**Shape**:
- `data` layout: `[R, inner]`, fp32, contiguous in `inner`. So row `r` occupies `data[r * inner .. (r+1) * inner)`.
- `scale` layout: `[R, softmax_tail]`, fp32, contiguous. Row `r` occupies `scale[r * softmax_tail .. (r+1) * softmax_tail)`. Typical `softmax_tail = 8` (one fp32 block).
- After broadcast, `data[r, j] *= scale[r, 0]` (scalar per row — or potentially per fp32-block, depending on softmax_tail).

**Pattern**: Issue Mul in `inner / 64`-step chunks across `inner` (a fp32 repeat is 64 elements). For each chunk, the BinaryRepeatParams.src1RepStride is set to `softmax_tail / 8` (in fp32 blocks), causing each successive repeat (= each successive row) to advance src1 by one row's worth of scale. The dst/src0 strides advance by `inner / 8` (one row of data). The mask covers 64 elements per repeat. Loop `inner / 64` times to cover the full inner dim.

**Concrete anchor** (public-API; worker-local names):
```cpp
// data, scale are LocalTensor<float>. R = rows, inner = inner-dim element count (mult of 64).
// softmax_tail typically 8 (1 fp32 block). repeatNumB32 = 64. blockNumB32 = 8.
constexpr uint64_t mask[2] = {UINT64_MAX, 0};
AscendC::BinaryRepeatParams rp = {
    /*dstBlkStride=*/1, /*src0BlkStride=*/1, /*src1BlkStride=*/0,
    /*dstRepStride=*/(uint8_t)(inner / 8),
    /*src0RepStride=*/(uint8_t)(inner / 8),
    /*src1RepStride=*/(uint8_t)(softmax_tail / 8)  // KEY: non-zero, walks across rows of scale
};
for (int64_t c = 0; c < inner / 64; c++) {
    AscendC::Mul(data[c * 64], data[c * 64], scale,
                 mask, /*repeatTimes=*/R, rp);
}
AscendC::PipeBarrier<PIPE_V>();
```

**Relationship to P-P62 (Brcb shape)**:
- P-P62: Brcb `scale [R]` → `bcastScale [R, 8]`, then per-block Sub/Mul against `data` with all strides = inner. Requires `R >= 8`. Costs one Brcb + one full broadcast tensor in UB.
- This pattern: Skip Brcb. Issue Mul `inner / 64` times with src1RepStride walking the scale tensor row-by-row. Costs `inner / 64` extra Mul calls (cheap; same compute either way) and zero extra UB scratch.
- Both produce identical results. Use this when `softmax_tail` is already > 0 (you have a real per-row scale buffer, not a single scalar) and inner-dim is large enough that the `inner / 64` Mul calls amortize well. Use P-P62 when you only have a scalar-per-row in a `[R]` shape and need to manifest the broadcast tensor.

**Numerics**: Identical to per-element Mul; no order effects. Deterministic.

**Hard do-not-apply**:
- Do NOT use when `inner % 64 != 0`: the trailing partial chunk needs a separate tail handler with adjusted mask (not shown in anchor — worker must add).
- Do NOT use when `R` exceeds 255 (RepeatTimes is uint8): split the outer R loop.
- Do NOT use when `softmax_tail / 8 > 255`: src1RepStride is uint8.
- Do NOT confuse with `src1BlkStride = 0` (which broadcasts within a single repeat); the KEY field here is `src1RepStride = softmax_tail / 8` (non-zero, walks across repeats).

**Other instances predicted**:
- Attention output combine (`out *= scale_per_row`) — the canonical case.
- LayerNorm / RMSNorm output scaling (gamma is per-feature not per-row, so opposite axis — but the same stride trick applies).
- Per-row gain/bias application in any normalized output (group-norm, batch-norm fold).
- Per-token weighting in MoE finalize (token-scale broadcast across hidden_dim).

**Risks before promotion**:
- Tail handling: `inner` not multiple of 64 needs a tail Mul with reduced mask. Worker must verify the op's actual head_dim values.
- The BinaryRepeatParams shape is brittle to read; document inline. Mis-setting `src1RepStride = 0` silently broadcasts the FIRST row's scale to all rows (catastrophic numerical bug, but cases would still "look reasonable" because outputs are not NaN). Add an assertion in dev builds.
- Source-structure verification only — promotion to P-P requires an a5_ops kernel benchmarking this against P-P62 Brcb form to confirm the trade-off.

**Cross-reference**: This is a sibling to P-P62 (different shape, same semantic). C37 dedup should treat them as alternative implementations under the same `applies_to` umbrella, NOT merge them.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-RAU-3，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
