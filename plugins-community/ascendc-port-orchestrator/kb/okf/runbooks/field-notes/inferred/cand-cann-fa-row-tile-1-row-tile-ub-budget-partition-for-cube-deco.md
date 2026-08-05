---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Row-tile UB-budget partition for cube-decomposed forward attention with 3-task pipelined carousel and matmul-library-driven cube↔vec sync"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=forward_flash_attention_with_high_level_matmul_library AND row_dim_exceeds_ub_budget derived-from: cann-source (ops-transformer flas"
phenomenon: build_failure
signal:
  - "A forward fused-attention class kernel decomposes into two cube stages (QK_dot, P_at_V) and intermediate vector stages (mask + softmax, output rescale + writeba"
confidence: inferred
status: stub
original_id: CAND-CANN-FA-ROW-TILE-1
timestamp_inferred: true
tags: [candidate, inferred, mmad, fixpipe, crosscoresetflag, bngs1s2, q_block_rows, cand-cann-fa-row-tile-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=forward_flash_attention_with_high_level_matmul_library AND row_dim_exceeds_ub_budget`
`derived-from: cann-source (ops-transformer flash_attention_score arch22 s1s2_bn2gs1 variant, 2026-05-26)`
`verified_on: cann-source (read-only structural extraction); unverified_on: a5_ops`
`local-kb-crossref: CAND-FA2 (online-softmax recurrence — composes inside vec1 phase), CAND-FA4 (block-reduce shape for rowmax/rowsum), CAND-NSA-1 (Matmul-library + local SetFlag<MTE3_MTE2> cube↔vec sync — this candidate extends NSA-1's 2-stage ping-pong to a 3-stage carousel), CAND-FA-CV-1 (WorkspaceQueue ring buffer — different abstraction layer, V220 manual CrossCore path), OL-186 (V351 forward FA cube-MatmulImpl P@V precision requirement — the cube halves this candidate orchestrates).`

**Trigger**: A forward fused-attention class kernel decomposes into two cube stages (QK_dot, P_at_V) and intermediate vector stages (mask + softmax, output rescale + writeback), AND the per-output-row (Q-seq direction) score-matrix tile `[s1_rows, kv_chunk_cols]` does not fit in a single UB buffer in one shot, AND the cube halves are driven by the high-level `matmul::Matmul<>` library client API (NOT raw `Mmad` / `Fixpipe` intrinsics with manual `CrossCoreSetFlag` — that case is the negative side of CAND-FA1).

Concretely: this is the structural skeleton for the `BNGS1S2`-style FA variant where Q-seq is parallelized across cores (each AIV/AIC core owns a contiguous block of Q-rows), KV-seq is the inner reduction axis processed in chunks, and head_dim is small enough to fit alongside the score tile in UB.

**Recommendation — two-level row partition + 3-task pipelined carousel**:

(1) **Outer row tile** chosen by host tiling: a per-task block of `Q_BLOCK_ROWS` query rows (size chosen so the running softmax state `[Q_BLOCK_ROWS, 1]` and the output accumulator `[Q_BLOCK_ROWS, head_dim_v]` together fit in their own dedicated UB buffers, leaving the score-tile UB budget free for the inner step).

(2) **Inner row sub-tile** computed at runtime each KV-chunk iteration:

```
SUB_ROWS_VEC1 = min( UB_SCORE_BUDGET_FP32 / kv_chunk_cols_aligned , Q_BLOCK_ROWS )
SPLIT_N_VEC1  = ceil_div( Q_BLOCK_ROWS , SUB_ROWS_VEC1 )

SUB_ROWS_VEC2 = (head_dim_v_aligned > 64)
                ? UB_OUT_BUDGET_FP32 / head_dim_v_aligned
                : Q_BLOCK_ROWS
SPLIT_N_VEC2  = ceil_div( Q_BLOCK_ROWS , SUB_ROWS_VEC2 )
```

`UB_SCORE_BUDGET_FP32` and `UB_OUT_BUDGET_FP32` are static UB-allocation constants chosen at `TPipe::InitBuffer` time (typically the same fp32-element count, e.g. 8192 fp32 = 32 KB, matching the score-tile and output-tile ping-pong buffers). Use the formula to size the inner row stride at each KV chunk so the per-chunk score (or output) matrix `[SUB_ROWS, col_aligned]` fits in its dedicated buffer; loop the inner step `SPLIT_N` times to cover the full block.

Why "inner step has different size per cube stage": the score matrix is `[Q_BLOCK_ROWS, kv_chunk_cols]` whereas the output matrix is `[Q_BLOCK_ROWS, head_dim_v]`. With the same UB-element budget, the sub-row count differs because the column dimension differs. The `head_dim_v_aligned > 64` gate just means "if head_dim_v is large enough to actually need sub-partition, partition; else the whole Q_BLOCK_ROWS fits in one shot for the output stage". This is the canonical shape; do NOT replicate one sub-tile size across both stages.

(3) **3-task pipelined carousel** across the KV-chunk loop, using three task descriptors indexed by `carouselId % 3`. Two GM ping-pong slots (`carouselId % 2`) hold the cube outputs (score and output) because at most two of the three in-flight tasks touch the same GM-slot kind at any moment.

Per iteration of the KV-chunk loop, six actions in this order:

```
//      slot_now  = carouselId % 3
//      slot_prev = (carouselId + 2) % 3     // = carouselId - 1
//      slot_pp   = (carouselId + 1) % 3     // = carouselId - 2

(A) if carouselId >= 1:  mmQK.WaitIterateAll(); mmQK.End();                  // retire QK_dot for slot_prev
(B) if not-tail:         mmQK.IterateAll<false>(scoreGmSlot[carouselId % 2], ...)
                                                                              // issue QK_dot for slot_now
(C) if carouselId >= 1:  VecPhase1(carousel[slot_prev]);                     // mask + softmax + write probs
                          AscendC::SetFlag<HardEvent::MTE3_MTE2>(evt)         // arm probs->mmPV sync
(D) if carouselId >= 2:  mmPV.WaitIterateAll(); mmPV.End();                  // retire P_at_V for slot_pp
(E) if carouselId >= 1:  AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evt)
                          mmPV.IterateAll<false>(outGmSlot[slot_prev % 2], ...)
                                                                              // issue P_at_V for slot_prev
(F) if carouselId >= 2:  VecPhase2(carousel[slot_pp]);                       // output rescale + writeback
carouselId += 1
```

Steady state (`carouselId >= 2`) has THREE work units overlapped:
- slot T   — QK_dot cube just issued (running on cube engine, AIV moves on)
- slot T-1 — vec1 (mask + softmax) running on AIV, P_at_V cube kicked in same iter
- slot T-2 — P_at_V cube just retired, vec2 (output rescale + GM writeback) running on AIV

This is structurally deeper than NSA-1's 2-stage cube→vec ping-pong: NSA-1 alternates one cube call + one or two vec phases per iter; this carousel keeps the cube ENGINE always busy (next QK_dot already issued before current iter's P_at_V completes) AND keeps the vec ENGINE always busy (two distinct vec phases interleaved across iters).

**Sync primitive set**:
- Cube↔vec retirement: `matmul::Matmul<>::IterateAll<false>(gmDst, ...)` to issue, `WaitIterateAll(); End();` to retire. The library owns the underlying AIC↔AIV sync — do NOT layer `CrossCoreSetFlag/WaitFlag` on top (see CAND-FA1 hard-do-not-apply). One `mmQK` instance (for QK_dot) and one `mmPV` instance (for P_at_V) are declared as class members and reused across the carousel; the library tracks per-call state across iters.
- AIV-side GM-write retirement: `AscendC::SetFlag<HardEvent::MTE3_MTE2>(evt)` after the vec1 phase finishes writing the probs matrix to GM, paired with `AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evt)` before the next `mmPV.IterateAll` reads it. `evt` is `FetchEventID(HardEvent::MTE3_MTE2)` once outside the loop.
- Intra-AIV pipe sync inside the vec1 and vec2 phases: standard `HardEvent::MTE2_V`, `V_MTE2`, `V_MTE3`, `MTE2_MTE3`, `MTE3_V` flags — see CAND-FA2 and P-P75.

**Per-row state ownership across KV chunks** (composes with CAND-FA2):
- Score-tile UB buffer: ping-pong by `carouselId % 2` to overlap stage1 of next iter with stage2 of current iter.
- Per-row online softmax state arrays (running max, running sum, rescale-delta): allocated as two-deep ping-pong slot arrays indexed by `carouselId % 2`. Size each slot = `Q_BLOCK_ROWS * FLOAT_BLOCK_SIZE * sizeof(fp32) = Q_BLOCK_ROWS * 32 bytes` (8 fp32 lanes per row for `Brcb`-compatible layout per P-P62).
- Output accumulator `O_running`: held in a stage2 UB buffer; carries across KV chunks, divided by final running_sum on the last KV chunk (CAND-FA2 §5).

**Concrete anchor** (public-API surface only; worker chooses op-local member names; UB-element budget shown as a worker-chosen constant):
```cpp
// Worker-chosen UB budget per ping-pong buffer (32 KB shown; pick based on UB layout).
constexpr uint32_t UB_SCORE_BUDGET_FP32 = 8 * 1024;
constexpr uint32_t UB_OUT_BUDGET_FP32   = 8 * 1024;

// One-time UB allocations done in InitBuffer() — sizes are static.
this->pipe->InitBuffer(this->scoreUbPing , UB_SCORE_BUDGET_FP32 * sizeof(float));
this->pipe->InitBuffer(this->scoreUbPong , UB_SCORE_BUDGET_FP32 * sizeof(float));
this->pipe->InitBuffer(this->outAccumUb  , UB_OUT_BUDGET_FP32   * sizeof(float));
this->pipe->InitBuffer(this->runSumUb[0] , Q_BLOCK_ROWS * 4 * 8);   // 8 fp32 lanes per row
this->pipe->InitBuffer(this->runSumUb[1] , Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->runMaxUb    , Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->rescaleUb[0], Q_BLOCK_ROWS * 4 * 8);
this->pipe->InitBuffer(this->rescaleUb[1], Q_BLOCK_ROWS * 4 * 8);

// Per-task descriptor (UB-resident, NOT GM); 3-slot carousel array as class member.
struct CarouselTaskDesc {
    int64_t carouselId;
    int32_t kvChunkCols;
    int32_t subRowsVec1, splitNVec1;
    int32_t subRowsVec2, splitNVec2;
    // ... per-iter offsets, mask offsets, ... (worker fills) ...
};

void ComputeInnerSubTileSplit(CarouselTaskDesc& td, int32_t headDimVal) {
    int32_t kvAligned = AlignUp(td.kvChunkCols, /*alignTo=*/8);   // fp32 8-element block align
    td.subRowsVec1 = Min<int32_t>(UB_SCORE_BUDGET_FP32 / kvAligned, Q_BLOCK_ROWS);
    td.splitNVec1  = CeilDiv<int32_t>(Q_BLOCK_ROWS, td.subRowsVec1);

    int32_t hdAligned = AlignUp(headDimVal, 8);
    td.subRowsVec2 = (hdAligned > 64)
                  ? (UB_OUT_BUDGET_FP32 / hdAligned)
                  : Q_BLOCK_ROWS;
    td.splitNVec2  = CeilDiv<int32_t>(Q_BLOCK_ROWS, td.subRowsVec2);
}

// Outer scheduling loop (sketch — 3-task pipelined carousel):
event_t evtVec1Done = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
CarouselTaskDesc carousel[3];
int64_t carouselId = 0;

for (int64_t kvChunk = 0; kvChunk <= kvChunkLimit; ++kvChunk) {
    // (A) Retire previous QK_dot cube call.
    if (carouselId >= 1)               { this->mmQK.WaitIterateAll(); this->mmQK.End(); }

    // (B) Issue current QK_dot cube call into ping-pong slot [carouselId % 2].
    if (kvChunk <= kvChunkLimit) {
        SetCarouselTaskDesc(carousel[carouselId % 3], carouselId, kvChunk);
        ComputeInnerSubTileSplit(carousel[carouselId % 3], this->headDimV);
        this->mmQK.SetTensorA(this->qGmInput[QInputOffset(carouselId)]);
        this->mmQK.SetTensorB(this->kGmInput[KInputOffset(carouselId, kvChunk)], /*transposeB=*/true);
        this->mmQK.SetTail(Q_BLOCK_ROWS, carousel[carouselId % 3].kvChunkCols, this->headDimQ);
        this->mmQK.template IterateAll<false>(this->scoreGmSlot[carouselId % 2], 0, false, true);
    }

    // (C) Vec1 for previous task: mask + softmax (CAND-FA2 + CAND-FA4); writes probs to GM.
    if (carouselId >= 1) {
        VecPhase1(carousel[(carouselId + 2) % 3]);
        AscendC::SetFlag<HardEvent::MTE3_MTE2>(evtVec1Done);
    }

    // (D) Retire previous-previous P_at_V cube call.
    if (carouselId >= 2)               { this->mmPV.WaitIterateAll(); this->mmPV.End(); }

    // (E) Issue current P_at_V cube call (for previous task).
    if (carouselId >= 1) {
        AscendC::WaitFlag<HardEvent::MTE3_MTE2>(evtVec1Done);
        this->mmPV.SetTensorA(this->probsGmSlot[(carouselId + 2) % 3 % 2]);
        this->mmPV.SetTensorB(this->vGmInput[VInputOffset(carousel[(carouselId + 2) % 3])]);
        this->mmPV.SetTail(Q_BLOCK_ROWS, this->headDimV, carousel[(carouselId + 2) % 3].kvChunkCols);
        this->mmPV.template IterateAll<false>(this->outGmSlot[(carouselId + 2) % 3 % 2], 0, false, true);
    }

    // (F) Vec2 for previous-previous task: output rescale + GM writeback.
    if (carouselId >= 2) {
        VecPhase2(carousel[(carouselId + 1) % 3]);
    }

    carouselId++;
}
```

**VecPhase1 inner sub-row loop** (the UB-budget sub-tile partition in action):
```cpp
void VecPhase1(CarouselTaskDesc& td) {
    LocalTensor<float> scoreUb = this->scoreUbPing.template Get<float>();  // or scoreUbPong by parity
    // ... DataCopy(scoreUb, scoreGmSlot[td.carouselId % 2], ...) — score tile GM→UB ...

    int32_t subRowsRun = td.subRowsVec1;
    for (int32_t splitIdx = 0; splitIdx < td.splitNVec1; ++splitIdx) {
        if (splitIdx == td.splitNVec1 - 1) {
            // Tail sub-tile shrinks to remaining rows; mandatory to avoid UB overflow.
            subRowsRun = Q_BLOCK_ROWS - splitIdx * td.subRowsVec1;
        }
        // Per sub-row chunk:
        //   - apply atten_mask via Select / Adds(-large_negative)
        //   - rowmax via CAND-FA4 block-reduce shape
        //   - online-softmax recurrence (CAND-FA2 step 3) updating runMaxUb / runSumUb / rescaleUb
        //   - rowsum via CAND-FA4 block-reduce shape
        //   - emit per-row probs sub-block to GM via DataCopy (MTE3 pipe)
    }
}
```

**Determinism**:
- Each task descriptor `carousel[carouselId % 3]` is owned by a single carousel slot; reading/writing within one carousel iter is sequential.
- GM ping-pong slots `[carouselId % 2]` for cube outputs are single-writer (the matched cube call) and single-reader (the matched vec phase); the carousel structure structurally guarantees that no slot is written while another stage reads it (the `WaitIterateAll` barriers create a strict total order between writer and reader).
- Per-row softmax state has one row per AIV core (no cross-core write).
- Det-preserving by construction when the inner row-sub-tile reduction is deterministic (CAND-FA2 + CAND-FA4 preconditions).

**Hard do-not-apply**:
- Do NOT use a 3-stage carousel when the cube engine cannot keep up with the kick rate — if cube latency for QK_dot exceeds the AIV's combined vec1+vec2 time, the carousel adds no overlap and just burns 50% more task-descriptor UB; degrade to 2-stage NSA-1 ping-pong.
- Do NOT use the `min(UB_BUDGET / col_aligned, Q_BLOCK_ROWS)` formula when the per-row UB state arrays (runMax, runSum, rescale buffers) have not been carved out of the UB budget first — the formula's `UB_SCORE_BUDGET_FP32` assumes those slots are already reserved.
- Do NOT mix this with manual `CrossCoreSetFlag/WaitFlag` for the cube↔vec handoff — that races with the Matmul library's internal AIC↔AIV flag pool (PB-34 / 507014 territory).
- Do NOT collapse `splitNVec1` and `splitNVec2` to one value — the two stages can have different sub-tile sizes when `kv_chunk_cols != head_dim_v`. Forcing equality wastes UB and may pessimize one stage.
- Do NOT use this skeleton for backward FA — the gradient ops have a different cube/vec ladder (saved-tensor restore + multi-output dispatch — see CAND-FAG-3 / CAND-FAG-4). This candidate is forward-only.

**Other instances predicted**:
- Forward FlashAttention (the canonical case the source structure derives from).
- Variable-length forward attention (TND layout) where Q_BLOCK_ROWS is per-batch-item but the inner UB-budget formula still applies.
- Forward multi-query / grouped-query attention (GQA) — Q_BLOCK_ROWS effectively scales by group factor on the score side but the sub-tile formula is unchanged (the worker chooses Q_BLOCK_ROWS post-group-expand).
- Forward MLA prefill where the latent-projected K/V dimension is small enough to fit alongside the score tile.
- Other cube-vec-cube forward ops that decompose as (matmul A → vec post-process → matmul B → vec finalize) and need to amortize cube/vec overlap across an inner reduction loop. Examples: fused GEMM + activation + GEMM (e.g. SwiGLU on top of LM head), block-sparse attention with chunked KV iteration.

NOT predicted:
- Standalone softmax / pure-VEC reductions — no cube halves, the carousel does nothing.
- Backward FA (different cube/vec ladder).
- FA variants using raw Mmad/Fixpipe without the Matmul library client API — those are CAND-FA1 territory.

**Risks before promotion**:
- The UB-budget anchor `8192 fp32 elements (= 32 KB)` is a CHOICE not a constant — it depends on how the worker partitions UB. The candidate's formula is shape-correct regardless, but workers must declare the chosen anchor explicitly in their kernel's `InitBuffer` so the inner sub-tile size is computed against the right number.
- 3-task carousel assumes a kernel structure where exactly 2 cube stages and 2 vec stages alternate; an op with 3 cube stages (e.g. fused-attention with extra side-product) needs a different scheduling skeleton.
- The pattern depends on `matmul::Matmul<>::IterateAll/WaitIterateAll/End` being available — V220 + V351 both ship this API but a future arch that drops it would invalidate the sync layer.
- Inner sub-tile formula `min(UB / col_aligned, Q_BLOCK_ROWS)` integer-rounds DOWN; the tail iter's `subRowsRun = Q_BLOCK_ROWS - splitIdx * subRows` handles the remainder. Workers MUST emit the tail-iter shrink — forgetting it underflows UB on the last sub-tile.
- Verification gap: source-structure read only; no a5_ops kernel has shipped this exact carousel + sub-tile formula yet. Promotion to P-P / OL requires a5_ops forward FA implementation that passes Pass A + Pass B + det + perf vs CANN baseline AND msprof confirms cube/vec overlap (cube utilization > 0 AND vec utilization > 0 in the same wall-clock window for >= 80% of total kernel time).

**Promote when**:
- a5_ops 3_FusionAttention forward implementation lands using this carousel + sub-tile formula, achieves >= 51/61 case PASS_T1 (closing the existing 1/61 gap), AND msprof shows cube/vec overlap >= 60% of total kernel time.
- A SECOND op outside FA (e.g. fused GEMM-activation-GEMM, MLA prefill) successfully uses the same carousel skeleton.

**Cross-reference**:
- CAND-FA2 (online-softmax recurrence — runs inside the vec1 phase).
- CAND-FA4 (block-reduce shape for rowmax/rowsum — runs inside the vec1 phase).
- CAND-NSA-1 (Matmul-library + local SetFlag<MTE3_MTE2> — the same sync primitive set; this candidate extends to 3-stage carousel).
- CAND-FA1 (manual CrossCore — explicit negative complement; do NOT mix).
- CAND-FA3 (GM workspace slot rotation modulo MAX_LAG+1 — this candidate uses MAX_LAG=1 for cube-output slots; the 3-stage carousel is decoupled from the 2-slot GM ping-pong because task descriptors are UB-resident, not GM-resident).
- CAND-FA-CV-1 (cv-agent WorkspaceQueue with prelaunch+1 ring slots — different abstraction layer; cv-agent kernels use manual CrossCore, this candidate uses Matmul library; not interchangeable).
- OL-159 (forward FA softmax tile-scheduling — algorithmic scope companion).
- OL-186 (V351 forward FA cube-MatmulImpl P_at_V precision requirement — the cube halves this candidate orchestrates).
- P-P62 (Brcb broadcast precondition for the per-row state).
- P-P75 (intra-core SetFlag/WaitFlag<HardEvent> pipe sync primitive).

**Anti-overlap-with-NSA-1 statement** (explicit for C35 self-review): NSA-1 documents the 2-stage primitive (Matmul library IterateAll + local SetFlag<MTE3_MTE2>) and a per-iter ping-pong on `iter & 1`. This candidate is structurally deeper:
- 3 task descriptors instead of 2 (carousel vs ping-pong)
- Cube ENGINE next-iter pre-kick: while slot T-1's P_at_V is computing on cube, slot T's QK_dot has already been issued (separate Matmul instance)
- Inner row sub-tile partition formula (NSA-1 does not address UB overflow on the score tile because the compressed-attention output is much smaller per iter)
- Two-level partition (outer host tiling + inner runtime UB-budget split)

If C35 flags overlap, the **delta-content** answer is the UB-budget sub-tile formula + the 3-stage carousel structure. NSA-1 should remain a separate entry covering the 2-stage primitive; this candidate is the FA-class extension.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-CANN-FA-ROW-TILE-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
