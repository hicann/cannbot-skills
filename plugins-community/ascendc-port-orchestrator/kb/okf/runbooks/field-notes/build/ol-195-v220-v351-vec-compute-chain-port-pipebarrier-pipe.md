---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220→V351 vec compute-chain port — `PipeBarrier<PIPE_V>` + V→S sync MUST surround every reduce/elementwise/scalar-GetValue tuple; `WholeReduceMax` default `ORDER_VALUE_INDEX` packs index bits into the value array"
description: "applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5 (vec kernels that compose softmax / online-softmax / row-reductions from raw AscendC primitives, NOT high-level Soft"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5 (vec kernels that compose softmax / online-softmax / row-reductions from raw"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-195
timestamp_inferred: true
tags: [wholereducemax, order_value_index, ascendc, ol-195]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=port_a3_to_a5 (vec kernels that compose softmax / online-softmax / row-reductions from raw AscendC primitives, NOT high-level SoftmaxFlashV2)`
`verified_on: soc=Ascend950PR (V351, Ascend950PR_9579); cann=9.0.0; op=flash_attention_score IL emit failure 2026-05-28 (designer/translator emit 0 PipeBarrier, kernel softmax_max/sum ~1e-12 vs truth ~0.x = 12-14 OOM)`
`unverified_on: soc=Ascend910_9382 (V220); cv-agent FA V220 reference passes 16/16 WITH 8 explicit PipeBarriers in ComputeVec1 — without them V220 was empirically working only by implicit-sync luck per OL-193 same class`

**Principle**: When a port_a3_to_a5 kernel decomposes online-softmax into raw primitives — `WholeReduceMax(row_max, scores) → Max(m_new, m_i, row_max) → Sub/Exp(alpha) → GetValue(m_new[r]) loop → Adds(scores, scores, -mv) → Exp(scores) → WholeReduceSum(row_sum, scores)` — TWO classes of cross-pipe synchronization are LOAD-BEARING on V351 and silently absent if the translator emits the natural sequential C++ without barrier inserts:

1. **V→V (read-after-write on UB)**: every adjacent V-op pair that reads/writes the same UB tensor needs `PipeBarrier<PIPE_V>()` between them. V351 strict pipe separation lets the dual-issue scheduler overlap dependent vector ops; without a barrier `Adds(scores, scores, …)` can write before the preceding `WholeReduceMax(row_max, scores, …)` finishes reading.
2. **V→S (UB-write then GetValue)**: a scalar `GetValue(r)` after a vector op that wrote that UB cell needs `PipeBarrier<PIPE_V>()` (or `SetWaitFlag<HardEvent::V_S>()`) BEFORE the GetValue. The scalar pipe sees pre-V-op stale UB content otherwise.

Both classes are V220-tolerant (V220 has implicit dependency tracking or benign timing) and V351-strict (per OL-193 same class). The translator's mechanical lowering MUST emit these barriers; the designer DSL contract cannot encode them via tile-level primitive selection alone.

**Concrete anchor** (correct compute chain shape for an FA-class online-softmax kv-tile body on V351):
```cpp
AscendC::WholeReduceMax<AccT>(row_max, scores,
    BLOCK_N, subTileM, /*dstRepStride=*/1, /*srcBlkStride=*/1, /*srcRepStride=*/BLOCK_N/8,
    AscendC::ReduceOrder::ORDER_ONLY_VALUE);          // (1) NOT default ORDER_VALUE_INDEX
AscendC::PipeBarrier<PIPE_V>();                       // (2) V→V before Max consumes row_max
AscendC::Max(m_new, m_i, row_max, subTileM);
AscendC::PipeBarrier<PIPE_V>();                       // V→V before Sub consumes m_new
AscendC::Sub(alpha, m_i, m_new, subTileM);
AscendC::Exp(alpha, alpha, subTileM);
AscendC::PipeBarrier<PIPE_V>();                       // (3) V→S before GetValue
for (int32_t r = 0; r < subTileM; ++r) {
    AccT mv = m_new.GetValue(r);
    AscendC::Adds(scores[r * BLOCK_N], scores[r * BLOCK_N], -mv, BLOCK_N);
}
AscendC::PipeBarrier<PIPE_V>();                       // V→V before Exp consumes scores
AscendC::Exp(scores, scores, subTileM * BLOCK_N);
AscendC::PipeBarrier<PIPE_V>();                       // V→V before WholeReduceSum consumes scores
AscendC::WholeReduceSum<AccT>(row_sum, scores,
    BLOCK_N, subTileM, 1, 1, BLOCK_N/8);
AscendC::PipeBarrier<PIPE_V>();
```

**Sub-finding (1) — ORDER_VALUE_INDEX is the WRONG default for online-softmax**: `WholeReduceMax` default `order = ReduceOrder::ORDER_VALUE_INDEX` packs `[value, index]` into the destination per repeat. With `dstRepStride=1` (the natural choice for dense `row_max[0..M-1]` consumption), half the dst lanes are `reinterpret_cast<float>(uint32_t_index)`. Downstream `Max(m_new, m_i, row_max)` reads these garbage lanes as fp32 values (typically large finite floats from index bit patterns); `m_new` becomes a large positive number; `exp(scores - m_new) ≈ 0`. Failure signature: kernel softmax_max/sum values 12-14 orders of magnitude smaller than truth, near-zero P, near-zero softmax_out. **Fix**: explicit `AscendC::ReduceOrder::ORDER_ONLY_VALUE` (V351/Atlas 350 supports per CANN 9.1.0-beta1 doc page 0079).

**Sub-finding (2) — srcRepStride unit is datablocks (32B), NOT bytes, NOT elements**: CANN 9.1.0-beta1 doc page 0079 line 81 "源操作数每次迭代跳过的datablock数目". For fp32 BLOCK_N=64 row: 256B/32 = 8 datablocks. The intuition "stride is N/8 elements" works ONLY for fp32 because `sizeof(fp32) × 8 elements = 32B = 1 datablock`. The same expression breaks for fp16 (where stride would be N/16 datablocks). Translators must compute `srcRepStride = (BLOCK_N * sizeof(T)) / 32`, not `BLOCK_N / 8`.

**Sub-finding (3) — `Adds(dst, src, scalar, count)` in-place IS supported on both V220 and V351 for the count-form**: CANN doc page 00018 line 93 — overlap restriction applies only to the high-dim slicing (repeat-stride) form. Hypothesis "V351 Adds requires src≠dst" is FALSIFIED.

**Sub-finding (4) — fp32 sentinel `-3.0e38` is in V351's validated safe range**: arch35 CANN production uses `NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF = -FLT_MAX ≈ -3.4028e38` (slightly more negative). `-3.0e38` is a valid finite normal fp32. Hypothesis "−3e38 overflows V351 reduce internals" is FALSIFIED.

**Evidence**:
- CANN install header `/data/cann_b103/cann-9.0.0/x86_64-linux/asc/include/interface/kernel_operator_vec_reduce_intf.h:133-204` — V351 WholeReduce signatures behind `__NPU_ARCH__ == 3510|5102|3003|3113` guard.
- CANN 9.1.0-beta.1 docs page 0079 (WholeReduceMax) — srcRepStride unit + ReduceOrder semantics + Atlas 350 fp32 support.
- CANN source `cann/ops-transformer/attention/common/op_kernel/arch35/flash_attention_score_common_regbase.h:36` — `NEGATIVE_MIN_VALUE_FP32 = 0xFF7FFFFF`.
- cv-agent V220 reference `flash_attention/kernel/flash_attention_vec.h` — 8 `PipeBarrier<PIPE_V>` in ComputeVec1 (grep-verified). Vs current FA-class IL emit `workspace/flash_attention_score/kernel/flash_attention_score_vec.h` — 0 PipeBarrier (grep-verified).
- pp-3 empirical probe (2026-05-28) — A5 NPU 0, edge_dataset truth, cases 0/3/4 — kernel smax/ssum 12-14 OOM below truth, `exp(scores - huge)` underflow signature.

**Other instances (predicted)**: applies to ANY port_a3_to_a5 op that composes per-row softmax (LayerNorm, RmsNorm row-softmax, attention variants without high-level SoftmaxFlashV2, MoE routing softmax, fused-norm-softmax). Lint regexes:
- `WholeReduceMax\(` without nearby `ORDER_ONLY_VALUE` → fire (Sub-finding 1).
- `WholeReduce(Max|Sum).*BLOCK_N\s*/\s*8` → fire dtype-portability warning (Sub-finding 2 — works for fp32 by coincidence).
- Adjacent `WholeReduce*` + `Adds`/`Muls` on same UB tensor without intervening `PipeBarrier<PIPE_V>` → fire (V→V sync).
- `GetValue\(` preceded by a V-pipe write to the same TBuf without `PipeBarrier<PIPE_V>` or `SetWaitFlag<HardEvent::V_S>` → fire (V→S sync).

**Cross-ref**: OL-193 (V→MTE3 sync sibling — same V220-implicit-V351-strict class), OL-190 (cross-core barrier sibling), CAND-V220-to-V351-PortPattern-CubeVecFusedOp.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-195（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
