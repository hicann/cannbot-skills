---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Collapse per-tile V→S sync into a single end-of-pass V→S sync via per-tile UB result slots"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=multi-tile-row-reduction verified_on: soc=Ascend950PR; cann=9.0.0 unverified_on: soc=Ascend910_V220 (A3 chip family — V→S sync semantics"
phenomenon: build_failure
signal:
  - "multi-tile per-row reduction (max, sum, etc.) where the natural pattern is for each tile: ReduceXxx → V→S sync → scalar combine into row-level accumulator. mspr"
confidence: inferred
status: stub
original_id: CAND-PP84
timestamp_inferred: true
tags: [candidate, inferred, tile_count, aiv_scalar_ratio, cand-pp84]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=multi-tile-row-reduction`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 chip family — V→S sync semantics likely identical, but pipe-stage scheduling differs; A3 should re-validate)`

**Trigger**: multi-tile per-row reduction (max, sum, etc.) where the natural pattern is `for each tile: ReduceXxx → V→S sync → scalar combine into row-level accumulator`. msprof shows non-trivial scalar-pipe time and V→S sync count = `tile_count` per row.

**Principle**: Each per-tile V→S sync costs scalar-pipe cycles AND prevents the V pipe from overlapping with other VEC work. For multi-tile rows (3-7 tiles in LLM-shape kernels), this is significant overhead. Replace with: write each tile's reduction result to a per-tile slot of a **scratch UB tensor** (e.g. `maxAccum[t * 8]`), do ONE `PipeBarrier<PIPE_V>` at end of pass, then ONE V→S sync, then the scalar combine loop reads from the scratch UB. Reduces V→S sync count from `tile_count` to 1 per row.

**Concrete anchor**:
```cpp
LocalTensor<float> maxAccum = scratchBuf.Get<float>();   // per-tile slots, scratch reused
for (int32_t t = 0; t < tile_count; ++t) {
    ReduceMax<float>(maxAccum[t * 8], srcLocal, ws, cnt_align, false);  // direct-write to slot
}
PipeBarrier<PIPE_V>();
event_t evVS = pipe_.FetchEventID(HardEvent::V_S);
SetFlag<HardEvent::V_S>(evVS);
WaitFlag<HardEvent::V_S>(evVS);
float row_max = 0.0f;
for (int32_t t = 0; t < tile_count; ++t)
    row_max = std::max(row_max, maxAccum.GetValue(t * 8));
```

The 8-element stride per tile slot is for fp32 datablock alignment (32 B / 4 B = 8 elements). For fp16/bf16 use stride-16 per slot.

**Quantified benefit (op#29 ko-iter4 evidence)**: DynamicQuant case 12 [4096, 11008] fp16. Pass 1 has 3 tiles per row; per-tile sync collapse reduced `aiv_scalar_ratio` from 0.282 → ~0.18; honest mean perf +30-45 % on multi-tile cases. Single-tile cases see no change (only one sync to begin with) — pattern only helps when `tile_count > 1`.

**Cost / risk**:
- Adds 1 scratch UB tensor of size `tile_count * 8 * sizeof(float)` per row (negligible — typically < 256 B).
- Increases peak VEC register pressure slightly (per-tile result lingering in scratch), but msprof confirms no spill on op#29's 11008-D path.
- Does NOT help when `tile_count == 1` (single-tile rows already do one sync).

**Promote when**: 2nd op (e.g. RmsNorm sum-of-squares pass, layernorm mean pass, softmax max pass) shows the same `tile_count > 1` × scalar-ratio reduction. The pattern complements OL-115 (manual prefetch) — both target the same thin-compute multi-tile signature.

**Source**: op#29 29_DynamicQuant ko-iter4 (2026-05-02). 1-op evidence; needs second op to promote.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP84，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
