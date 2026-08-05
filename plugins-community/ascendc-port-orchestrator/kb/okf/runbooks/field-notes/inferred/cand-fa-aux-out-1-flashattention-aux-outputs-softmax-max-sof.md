---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "FlashAttention aux outputs (softmax_max, softmax_sum) emit via Brcb broadcast + single tile-wise DataCopy (or strided DataCopy with dstStride for BNGS1 interleave), NOT per-row scalar SetValue"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_aux_softmax_outputs / FlashAttention_forward_train / online_softmax_with_external_max_sum_emit derived-from: ca"
phenomenon: build_failure
signal:
  - "Op emits per-row scalar fp32 auxiliary outputs (e.g. softmax_max[B, N2G, S1], softmax_sum[B, N2G, S1], or any \"one fp32 per softmax row\") at the END of each Q-t"
confidence: inferred
status: stub
original_id: CAND-FA-AUX-OUT-1
timestamp_inferred: true
tags: [candidate, inferred, s1_real, brcb, datacopyparams, datacopypad, cand-fa-aux-out-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_aux_softmax_outputs / FlashAttention_forward_train / online_softmax_with_external_max_sum_emit`
`derived-from: cann-source (FA reference V220 s1s2_bn2gs1 + s1_bn2gs1 + bn2gs1s2_b epilogues, 2026-05-24 cl-fa-diff)`
`evidence_family: FA-AUX-OUT`
`verified_on: cann ops-transformer FA reference V220 aux-output emit pattern (kernel-structural evidence; V351 epilogue is in BaseApi base class, out of scope for this extraction)`

**Trigger**: Op emits per-row scalar fp32 auxiliary outputs (e.g. `softmax_max[B, N2*G, S1]`, `softmax_sum[B, N2*G, S1]`, or any "one fp32 per softmax row") at the END of each Q-tile, in addition to the main attention output tensor. The aux output layout is typically (B, N2*G, S1) — interleaved per (n2, g) head — meaning aux for head_i and aux for head_(i+1) for the same s1 row are NOT contiguous in GM.

**Why "candidate"**: derived from CANN FA reference's aux-emit pattern. Verified-on is structural only — need a port-side measurement showing the Brcb+DataCopy combo outperforms per-row SetValue (or a different L4 op showing the same aux-emit problem benefits from this pattern).

**Recommendation**:
1. Per-Q-tile, AFTER softmax reduction (max/sum is computed in UB as `LocalTensor<float>` of length `s1_real`), use `Brcb` to broadcast each fp32 scalar to 8 fp32 lanes:
   ```cpp
   Brcb(broadcast_buf, max_per_row, (s1_real + 7) / 8, {1, 8});
   // broadcast_buf now has s1_real * 8 fp32 elements
   ```
   The "×8" matches the SoftmaxFlashV2 / canonical AscendC online-softmax convention where each row's scalar reduction is broadcast to 8 lanes for downstream Mul-by-reciprocal arithmetic.
2. Sync the V→MTE3 transition explicitly:
   ```cpp
   event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
   SetFlag<HardEvent::V_MTE3>(e_v_mte3);
   WaitFlag<HardEvent::V_MTE3>(e_v_mte3);
   ```
3. **Contiguous BS layout** (single output stream — aux outputs for one head, no interleave): single DataCopy:
   ```cpp
   DataCopy(softmax_max_gm[max_off], broadcast_buf, s1_real * 8);  // fp32 count
   ```
4. **Strided BNGS1 layout** (aux outputs interleaved across N2*G heads at the s1 axis — typical for FA aux outputs feeding back into a backward pass): single STRIDED DataCopy via `DataCopyParams`:
   ```cpp
   DataCopy(softmax_max_gm[max_off], broadcast_buf,
            DataCopyParams{
                /*blockCount=*/ static_cast<uint16_t>(s1_real),
                /*blockLen=*/   1,        // 1 unit = 8 fp32 = 32 B
                /*srcStride=*/  0,        // packed source
                /*dstStride=*/  static_cast<uint16_t>(n2_g - 1)  // skip (N2*G - 1) units between rows
            });
   ```
   The strided destination writes row 0's aux at offset 0, row 1's aux at offset N2*G*32 B, etc. — exactly the BNGS1 layout external callers expect, with no extra layout-conversion pass.

**Concrete anchor** (public-API end-to-end pattern; placeholder names):
```cpp
LocalTensor<float> max_row    = max_buf.Get<float>();   // [s1_real] from softmax reduction
LocalTensor<float> sum_row    = sum_buf.Get<float>();
LocalTensor<float> broadcast  = aux_emit_buf.Get<float>();  // [s1_real * 8]

PipeBarrier<PIPE_V>();
Brcb(broadcast, max_row, (s1_real + 7) / 8, {1, 8});
event_t e_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
SetFlag<HardEvent::V_MTE3>(e_v_mte3);
WaitFlag<HardEvent::V_MTE3>(e_v_mte3);

if (layout_is_strided) {
    DataCopy(softmax_max_gm[max_off], broadcast,
             {static_cast<uint16_t>(s1_real), 1, 0,
              static_cast<uint16_t>(n2_g - 1)});
} else {
    DataCopy(softmax_max_gm[max_off], broadcast, s1_real * 8);
}
// Repeat for sum_row → softmax_sum_gm.
```

**Reject_cond**: do NOT use this pattern when:
- Aux output is per-tile (NOT per-row) — e.g. an op emitting a single fp32 reduction per (B, N1, S1_tile) chunk; in that case a plain single-element DataCopy is the right shape.
- The aux output layout is (B, S1, N2*G) with N2*G as the innermost dim — that's contiguous across heads-per-row and benefits from `DataCopyPad` not Brcb+strided-DataCopy.
- S1_real ≤ 8 — the 8-lane Brcb broadcast is wasteful at this size; per-row SetValue may be cheaper.

**Symptom anchor**: DEBT-FA-AUX (FA aux output write currently unblocked path). Hypothesis-link: the V220 row-tiled fp16 kernel's current aux-emit implementation is unknown but if it's per-row SetValue, it will be 8× slower than the Brcb+DataCopy shape. Need port-side measurement to validate.

**Other-instances-predicted**: any fused op with per-row scalar aux outputs (BatchNorm aux mean/var, GroupNorm aux mean/var, online-LayerNorm aux, RMS-Norm aux). Same Brcb-to-8-lane + (optional strided) DataCopy pattern applies.

**Promote when**: measured perf delta vs per-row SetValue on independent prototype fa_v220 aux-output emit AND one other aux-emit op (BatchNorm or GroupNorm forward) shows the Brcb+DataCopy shape is ≥ 2× faster.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA-AUX-OUT-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
