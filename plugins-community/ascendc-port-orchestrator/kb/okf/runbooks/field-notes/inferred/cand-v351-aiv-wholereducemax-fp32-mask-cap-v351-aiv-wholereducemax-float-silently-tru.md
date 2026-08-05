---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V351 AIV `WholeReduceMax<float>` silently truncates per-repeat when `mask > 64` — split into chunked-reduce + scalar combine"
description: "applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=fused-norm fused-quant attention-softmax-denom any-per-row-fp32-reduction verified_on: grouped_matmul_swiglu_quant_v2 2026"
phenomenon: build_failure
signal:
  - "per-row absmax / max / sum computed across N>64 fp32 elements returns a value that is the max/sum of only the first 64 elements of the row. Downstream quant sca"
confidence: inferred
status: stub
original_id: CAND-V351-AIV-WholeReduceMax-fp32-mask-cap
timestamp_inferred: true
tags: [candidate, inferred, reducemaxtemplate, max, add, verified_on, cand-v351-aiv-wholereducemax-fp32-mask-cap]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR_9579 (V351); cann=9.0.0; bisheng=15.0.5; op_class=fused-norm | fused-quant | attention-softmax-denom | any-per-row-fp32-reduction`
`verified_on: grouped_matmul_swiglu_quant_v2 2026-05-24 (pp-5 fix landed case 4 → 8/8 PASS_WITHIN_TOLERANCE @ commit f8fecd70)`
`unverified_on: V220 / Ascend910 (A3 chip family — empirical evidence is V351-only; A3 may have different per-repeat mask limit, needs probe)`

**Principle**: On V351 AIV, `WholeReduceMax(dst, src, mask, repeat, ...)` and the WholeReduce* family have a **hardware-level per-repeat fp32 mask cap of 64 elements** (one fp32 vector unit = 64-element lane). Passing `mask > 64` (e.g. mask=128, mask=256) **silently completes** the call but reduces over only the first 64 fp32 elements per repeat — no compile error, no runtime warning, no bit-pattern indicator. The remaining elements beyond mask=64 are ignored. Same caveat applies to other WholeReduce* primitives (WholeReduceMin, WholeReduceSum, possibly WholeReduceAdd).

**Symptom**: per-row absmax / max / sum computed across N>64 fp32 elements returns a value that is the max/sum of only the first 64 elements of the row. Downstream quant scale or softmax denominator is wrong; precision FAIL on cases where the missed elements would have dominated.

**Mitigation (concrete anchor — code from GMSQ_v2 utils.h `ReduceMaxTemplate`)**:
```cpp
constexpr uint32_t FP32_LEN_64_REPEAT = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_ELE = 64;
constexpr uint32_t VEC_LEN_ONCE_REPEAT_BLOCK = 8;

if (count <= FP32_LEN_64_REPEAT) {
    // Small case — single per-repeat WholeReduceMax is safe.
    WholeReduceMax(dst, src, count, 1, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK,
                   ReduceOrder::ORDER_ONLY_VALUE);
} else {
    // Large case (count > 64) — first 64 via per-repeat block-reduce, tail
    // via small-case ReduceMaxSmall, scalar Max combine.
    BlockReduceMax(workLocal, src, /*repeat=*/REPEAT_64,
                   /*mask=*/VEC_LEN_ONCE_REPEAT_ELE, 1, 1, VEC_LEN_ONCE_REPEAT_BLOCK);
    PipeBarrier<PIPE_V>();
    WholeReduceMax(resTmpLocal, workLocal, VEC_LEN_ONCE_REPEAT_ELE, 1, 1, 1,
                   VEC_LEN_ONCE_REPEAT_BLOCK, ReduceOrder::ORDER_ONLY_VALUE);
    PipeBarrier<PIPE_V>();
    ReduceMaxSmall(dst, workLocal, src + FP32_LEN_64_REPEAT,
                   count - FP32_LEN_64_REPEAT);  // tail (≤64 elements)
    PipeBarrier<PIPE_V>();
    const BinaryRepeatParams repeatParams = {1, 1, 1, NUM_8, NUM_8, NUM_8};
    Max(dst, dst, resTmpLocal, 1, 1, repeatParams);  // scalar combine head + tail
}
```

The mitigation generalizes: split N-element reduction into `ceil(N/64)` per-repeat reductions (mask=64 each), each producing one partial result, then combine via `Max` (or `Add` for sum) operating on the partial-result vector. Final reduce-of-partials uses single-repeat `WholeReduce*` over ≤64 partials.

**Reject_cond** — do NOT apply when:
- The reduction target is fp16/bf16 (mask cap is 128 for half-precision, not 64). Verify per-dtype before applying.
- The op is V220-only (verified_on is V351; V220 may have different limit per `verified_on` line).
- The reduction count is provably ≤64 at compile time (single-repeat call is correct and faster).

**Symptom anchor**: GMSQ_v2 case 4 originally failed with off-by-up-to-50% scale value when per-row silu*gate absmax was computed across 128 fp32 elements (count=N/2=128). Worker initially wrote `WholeReduceMax(dst, src, /*mask=*/128, repeat=1, ...)` — compiled clean, ran clean, returned absmax over first 64 elements only. pp-5 root-cause diagnosis: split into BlockReduceMax(64) + WholeReduceMax(64) + ReduceMaxSmall(tail) + scalar Max combine → case 4 PASS_WITHIN_TOLERANCE.

**Other instances (predicted)**:
- Fused-quant absmax over inner-D dimensions > 64 (e.g. group_norm_silu_quant inner_dim > 64, rms_norm_quant with hidden_size > 64)
- Attention softmax-max (per-row max over sequence-length > 64)
- Reduction-sum patterns where the same hardware lane structure applies (WholeReduceSum / Add over > 64 fp32 elements)
- Any V351 op whose Tier-2 N axis falls in (64, 8192] range AND reduces along that axis

**Verdict mapping (honest per-output disclosure, post independent cross-review 2026-05-24)**:
The chunked-reduce + scalar `Max`/`Add` combine introduces an fp32 **rounding-order difference** vs the (silently-truncated) single-call baseline. Per-output verdict impact:
- **Quantized outputs** (e.g. `y_int8` in GMSQ_v2): stay `T1_BIT_EXACT` — the quant step (Cast/Round) truncates the sub-ULP scale difference below the int8 quantization threshold.
- **fp32 reduction outputs** (e.g. `y_scale` in GMSQ_v2, or per-row max/sum): land in `T2_WITHIN_FP32_FLOOR` band on cases where the chunked-reduce activates (count > 64). Within fp32 ULP, but NOT strict bit-exact vs reference.

This is the **correct** outcome (better than the wrong silently-truncated answer), not a hardware-floor cheat. Customers applying this pattern should set tolerance gates accordingly: expect strict T1 on downstream quantized/integer outputs, T2_WITHIN_FP32_FLOOR on direct fp32 reduction outputs that flow through `ceil(N/64)` chunks.

GMSQ_v2 case 4 evidence (anchor): post-pp-5 verification.json `per_case_summary_pp5`:
```
case 4: verdict=PASS_WITHIN_TOLERANCE, y_int8=T1_BIT_EXACT, y_scale=T2_WITHIN_FP32_FLOOR
case 6: verdict=PASS_WITHIN_TOLERANCE, y_int8=T1_BIT_EXACT, y_scale=T2_WITHIN_FP32_FLOOR
(cases 1,2,3,5,7,8: y_scale=T1_FP32_NEAR_BIT_EXACT — count ≤ 64 path, single-call safe)
```

**Promote when**:
1. A second V351 op (e.g. group_norm_silu_quant, rms_norm_quant, or attention-softmax-fwd) independently hits the silent-truncation symptom AND applies the same chunked-reduce mitigation successfully (pass rate improves on at-risk cases).
2. Hardware-engineering team confirms the per-repeat fp32 mask=64 is a documented spec (not just empirical), and equivalent caps for fp16/bf16/int32 reductions are catalogued.

**Cross-link**: kernel anchor on origin/main (commit f8fecd70):
`output/a3_to_a5_port/src/kernels/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_utils.h::ReduceMaxTemplate` — customer can grep this function directly post-fresh-clone for the reference template. verification.json `precision.pass_a.per_case_summary_pp5` cases 4 + 6 are the y_scale T2_WITHIN_FP32_FLOOR evidence anchors.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-V351-AIV-WholeReduceMax-fp32-mask-cap，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
