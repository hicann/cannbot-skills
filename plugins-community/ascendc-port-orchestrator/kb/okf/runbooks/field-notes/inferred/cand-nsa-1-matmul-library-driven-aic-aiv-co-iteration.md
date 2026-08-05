---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Matmul-library-driven AIC/AIV co-iteration — vector phases chained via local `SetFlag<HardEvent::MTE3_MTE2>` between iterations, while the cube side is driven by `Matmul<>::IterateAll() + WaitIterateAll()` (the **complement** to CAND-FA1's hard-do-not-apply clause)"
description: "applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_high_level_matmul matmul_lib_driven_pipeline_with_aiv_postprocess derived-from: cann-source (nsa-class compress"
phenomenon: build_failure
signal:
  - "A fused-attention-class kernel uses the high-level matmul::Matmul<> template (NOT tile-MMAD primitives) for its cube halves AND needs the AIV side to chain >=2"
confidence: inferred
status: stub
original_id: CAND-NSA-1
timestamp_inferred: true
tags: [candidate, inferred, event_t, waititerateall, mte3_mte2, taskid, cand-nsa-1]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend910_V220 / Ascend950PR; cann=9.0.0+; op_class=fused_attention_with_high_level_matmul | matmul_lib_driven_pipeline_with_aiv_postprocess`
`derived-from: cann-source (nsa-class compressed attention, 2026-05-10 multicann)`
`verified_on: nsa_compress_attention (cann ops-transformer 2026-05) — top-level kernel main loop pattern`
`unverified_on: a5_ops (3_FusionAttention currently does NOT layer multi-phase AIV postprocess on top of high-level Matmul<>)`

**Trigger**: A fused-attention-class kernel uses the high-level `matmul::Matmul<>` template (NOT tile-MMAD primitives) for its cube halves AND needs the AIV side to chain >=2 distinct vector phases per outer iteration (e.g. QK→softmax→aux-scoring→TopK) with cross-iteration overlap. CAND-FA1's flag protocol is forbidden here (its hard-do-not-apply clause names exactly this case); a different sync recipe is required.

**Recommendation**: Drive the cube side with the high-level Matmul library client API — one `IterateAll<false>(gmDst, ...)` per outer iter writes the cube result into a GM ping-pong slot; the matching `WaitIterateAll(); End();` retires the cube call from the AIV side. AIV phases within the iter chain through local `event_t` allocated from the TPipe and synchronized with `SetFlag<HardEvent::MTE3_MTE2>` / `WaitFlag<HardEvent::MTE3_MTE2>`, NOT through `CrossCoreSetFlag/WaitFlag`. The library's internal AIC↔AIV sync is hidden behind `WaitIterateAll` — adding user-owned cross-core flags on top would race with library-owned flag IDs (see CAND-FA1 hard-do-not-apply clause and `507014` evidence cited there).

Per-iter shape:

1. AIV calls `bmm.IterateAll<false>(gmDst[taskId & 1], ...)` to kick a cube call into one GM ping-pong slot.
2. AIV calls `WaitIterateAll(); End();` to block on cube retirement of the *current* iter.
3. AIV runs phase-1 vec compute consuming `gmDst[taskId & 1]`; emits its own MTE3 GM write; sets `MTE3_MTE2` flag.
4. AIV (still same iter) kicks the next cube call `bmm2.IterateAll<false>(gmDst2[taskId & 1], ...)`.
5. AIV runs phase-2/phase-3 vec compute, each gated by `WaitFlag<MTE3_MTE2>` against the prior MTE3 emission and re-arming the flag at the end.
6. Between iters, the next iter's bmm1 can be pre-kicked while the current iter's phase-3 (e.g. TopK) is still running because they bind to different GM ping-pong slots (`taskId+1` vs `taskId`).

**Concrete anchor** (3–5 line public-API shape; no internal symbols):
```cpp
event_t mte3mte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);
for (int64_t it = innerOffset; it < innerLimit; ++it) {
    // cube side via Matmul library client
    bmm1.IterateAll<false>(gQK[it & 1], /*sync=*/false, /*reuse=*/false, /*wait=*/true);
    bmm1.WaitIterateAll(); bmm1.End();

    // AIV phase 1 (softmax-class) — consumes gQK[it & 1], writes gProbs[it & 1]
    WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
    runVecPhase1(gQK[it & 1], gProbs[it & 1]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);

    // chain next cube + AIV phases on the SAME ping-pong index, then bump taskId
    WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
    bmm2.IterateAll<false>(gPV[it & 1], false, false, true);
    runVecPhase2(gProbs[it & 1], gPV[it & 1]);
    SetFlag<HardEvent::MTE3_MTE2>(mte3mte2);
}
WaitFlag<HardEvent::MTE3_MTE2>(mte3mte2);
```

**Why it works**: `Matmul<>::WaitIterateAll()` is the library-owned barrier covering the cube → GM-write retirement, so user-owned `CrossCoreSetFlag/WaitFlag` on the same MODE/pipe space is redundant and provably conflict-prone (CAND-FA1 hard-do-not-apply names this). `SetFlag<HardEvent::MTE3_MTE2>` is a per-AIV-core local pipe flag — it orders the AIV's own MTE3 emission with its next iter's MTE2 read of the same GM region, which is what's needed once the cube↔vec handoff is already library-owned. Ping-pong on `taskId & 1` keeps two GM slots live so the next iter's cube call can prefetch while the current iter's vec tail finishes.

**Determinism**: The AIV's per-iter phase order is fixed by the source structure. Each GM ping-pong slot has a single writer per iter (the matched cube call) and a single reader (the matched vec phase). `WaitIterateAll` is a strict barrier — no in-flight cube write can leak into the next iter's vec read of slot `(taskId+1) & 1` because that slot was the prior iter's read target and is now free. Det-preserving by construction.

**Hard do-not-apply**:
- Do NOT combine this pattern with user-owned `CrossCoreSetFlag/WaitFlag` on overlapping `MODE` / pipe space — the high-level `Matmul<>` library already uses CrossCore internally and flag-ID collisions can stall the iter (the same `507014`-class failure CAND-FA1 cites).
- Do NOT use `PIPE_V` for the local AIV flag — the AIV's GM write retirement is on the MTE3 pipe; releasing on `PIPE_V` would publish before the GM write drains.
- Do NOT extend the ping-pong depth beyond 2 unless the GM workspace contract (P-P89) is restructured for >2-way rotation per CAND-FA3 modulo discipline — TQue-style depth-4 (OL-63) does not apply to GM ping-pong slots.

**Other instances predicted**:
- FlashAttention-class forward where cube QK → AIV softmax → cube PV → AIV rescale, when the cube side is built on the high-level `Matmul<>` template (not the tile-MMAD path that CAND-FA1 covers).
- Fused-attention with auxiliary side-output passes (e.g. attention + per-block aux-score + top-K-indices) where AIV runs ≥3 distinct phases per cube outer iter.
- Streaming attention prefill/decode hybrids that use the public `Matmul<>` template for BMM1 + BMM2 and need AIV-side multi-phase post-processing per row block.
- MoE GEMM chains where the expert GEMM uses `Matmul<>` and the AIV side runs gather/scatter + scale between GEMMs.

**Risks before promotion**:
- a5_ops 3_FusionAttention currently uses simpler AIV postprocess; no multi-phase chain layered on `Matmul<>` is shipped yet — this candidate is unverified on a5_ops measurements.
- The `WaitIterateAll(); End();` pair MUST appear in that order per iter; reversing them is observed to silently miss-retire the cube call on some library versions.
- The local `MTE3_MTE2` event must be `FetchEventID`-acquired ONCE outside the loop and re-armed (`SetFlag`) before the loop body's first `WaitFlag`; per-iter `AllocEventID` inside the loop is permitted only when paired with `ReleaseEventID` before iter exit, otherwise event-ID pool exhausts.
- Pipe selection: producing MTE3 → consuming MTE2 on the same AIV is the safe per-iter chain. Using a `V_MTE3` event in place of `MTE3_MTE2` between iters releases before the GM-write retires (silent stale-read).

**Cross-reference**:
- CAND-FA1 (cross-core user-owned flag handoff for kernels that do NOT use `Matmul<>`) — this candidate is the complementary case named in CAND-FA1's hard-do-not-apply clause.
- CAND-FA3 (GM workspace slot rotation modulo MAX_LAG+1) — directly composes; the ping-pong is the `MAX_LAG=1` instance of FA3.
- P-P89 (GM workspace contract — public outputs vs opaque scratch) — supplies the GM layout discipline the ping-pong slots sit inside.
- P-P75 (intra-core `SetFlag/WaitFlag<HardEvent>` for SIMD pipe sync) — the local pipe-flag layer this candidate composes on top of; same primitive, applied to inter-phase chaining within one AIV across iterations.
- OL-91 (cube playbook conventions for `Matmul<>` users) — orthogonal but the same dispatch class.

**Promote when**: an a5_ops fused op (e.g. a future 3_FusionAttention + topK variant, or fused attention + auxiliary statistics) ships with `Matmul<>`-driven cube halves AND a measurable cube/vec overlap improvement vs a baseline that serializes vec phases behind `SyncAll<true>()`. Verification must include msprof showing the next-iter cube kick overlapping the current-iter vec tail.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-NSA-1，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
