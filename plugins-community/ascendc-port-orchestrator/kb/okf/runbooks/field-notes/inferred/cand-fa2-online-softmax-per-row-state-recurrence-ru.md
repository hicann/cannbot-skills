---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Online-softmax per-row state recurrence — running max, running sum, and an external accumulator rescale carried across row tiles (final divide deferred to last tile)"
description: "applies_to: any SoC with public AscendC VEC Exp/Sub/Mul/Add/Max/Div + Brcb primitives; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul derived-from: cann-source ("
phenomenon: build_failure
signal:
  - "Op needs row-softmax over a logical row whose width exceeds UB capacity AND has a streaming downstream consumer that can absorb per-tile probability numerators"
confidence: inferred
status: stub
original_id: CAND-FA2
timestamp_inferred: true
tags: [candidate, inferred, o_running, runsum, center_k, cand-fa2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any SoC with public AscendC VEC Exp/Sub/Mul/Add/Max/Div + Brcb primitives; cann=9.0.0+; op_class=online_softmax / streaming_normalization / fused_softmax_matmul`
`derived-from: cann-source (FA-class kernel structure, 2026-05-10 revise-cl3)`
`verified_on: cann ops-transformer FA-class block epilogues for online softmax + output rescale (source-structure-only; no a5_ops measurement)`

**Trigger**: Op needs row-softmax over a logical row whose width exceeds UB capacity AND has a streaming downstream consumer that can absorb per-tile probability numerators (e.g. `softmax(scores) @ V` in FlashAttention-style forward, streaming-softmax-then-weighted-sum, online-attention with KV-cache). The op MUST be able to either (a) carry an unnormalized output accumulator across tiles and divide once at the end, OR (b) buffer per-tile numerators with their center-max metadata for a deferred rescale pass.

**Key constraint — standalone softmax CANNOT use this recurrence as-is**:
The exp(scores - newMax) values produced in tile k are centered on the *running* max at step k, not the final max. They are valid numerators of an unnormalized accumulator only. To get final probabilities you need one of:
  1. **Streaming consumer with rescaled accumulator** (the FlashAttention path): maintain an unnormalized output `O_running`. Each tile, multiply `O_running` by `delta_k = exp(oldRunMax - newMax)` BEFORE adding the current tile's `probs_k @ V_k`. On the last tile, divide `O_running` by final `runSum`. The FA-class reference structure splits this across two stages: a softmax stage emitting per-tile probability numerators plus the per-tile delta state, and a downstream output-rescale stage that multiplies the running accumulator by delta each tile and performs the divide-by-runSum only on the last-tile branch.
  2. **Stored numerators with replay**: persist `numerator_k = exp(scores_k - center_k)` AND `center_k` per tile, then after the final tile compute `prob_k = numerator_k * exp(center_k - finalRunMax) / finalRunSum` in a second pass. Costs an extra full GM round-trip; only do this when there is no fused downstream consumer.

Do NOT emit per-tile probability-numerator values to GM as if they were final softmax outputs for standalone consumers — they are valid only within the rescale-aware pipeline.

**Recommendation**: Process the row in tiles of width `TILE_W`. Maintain fp32 per-row state arrays of length `R` in UB across all tile iterations of one row group. Naming follows the source structure exactly so future readers can map back to the FA layout:

- `runMax[R]` (CANN `gm` — running/global max across all processed tiles for this row)
- `runSum[R]` (CANN `gl` — running normalizer, centered on current `runMax`)
- `newMax[R]` (CANN `hm` — scratch holding max(runMax, tileMax) before commit)
- `delta[R]` (CANN `dm` — scratch holding `exp(runMax_old - newMax)`; kept around because the rescale phase consumes it)
- `tileMax[R]` (CANN `lm` — current tile's local rowmax)
- `tileSum[R]` (CANN `ll` — current tile's local rowsum after the tile's Exp)

Per tile `k`, after reading scores into UB and applying scale + mask (mask = add(-3e38) on masked positions, standard CANN form):

**1. tileMax = rowmax(scores_k)** — use the wide-row block-reduction shape from CAND-FA4.

**2. First-tile guard** (explicit branch — the first-tile-flag form used in the FA-class reference):
  - If this is the row's first tile: `runMax := tileMax`, then proceed directly to Step 3 with `runMax` as the centering value; after Step 4 set `runSum := tileSum` and (for streaming consumers) `O_running := probs_k @ V_k`. Do NOT compute `delta` on this path. Do NOT initialize with `runMax = -inf` and then evaluate `exp(runMax - newMax)` — the result is undefined (NaN) and corrupts downstream.
  - If the very first tile of the row is itself fully masked (`tileMax ≈ -3e38`), defer state init to the next non-masked tile (track a per-row `seenAnyValidTile` flag). The recurrence is undefined on an empty-evidence row; the worker MUST honor the op's reference behavior for empty / all-masked rows (often: produce zero output, skip the divide).

**3. Later-tile recurrence** (the steady-state body):
  - `newMax = max(tileMax, runMax)`
  - `delta = exp(runMax - newMax)` — `delta ∈ [0, 1]`; underflow to `0.0` is the *correct* value when the old running tail is irrelevant. Do NOT clamp to a small positive constant; the consumer's `O_running *= delta` then correctly forgets the stale accumulator.
  - `probs_k = exp(scores_k - newMax)` — broadcast `newMax` across the tile's columns (see Broadcast note below).
  - `tileSum = rowsum(probs_k)`
  - `runSum = delta * runSum + tileSum`
  - `runMax := newMax` (committed last, since the Sub above still uses old runMax)

**4. Streaming-consumer accumulator update** (carries the rescale through to the output side, mirroring the FA-class output-rescale stage):
  - `O_running := delta * O_running + probs_k @ V_k`
  - Broadcasting `delta` over the embed dimension of `O_running` uses the same P-P62 Brcb-then-Mul shape as the softmax core.

**5. Final-tile finalize** (only on the last tile of the row's tile sweep):
  - `O_final := O_running / runSum`
  - Cast to output dtype (CAST_RINT for bfloat16, CAST_NONE for fp16).

**Broadcast note (P-P62 precondition)**:
The `Brcb(...) -> Sub` shape used to broadcast `newMax` across tile columns, and the `Brcb(...) -> Mul/Div` used for `delta`/`runSum` across the embed dimension, both require the rows-axis aligned up to `FLOAT_BLOCK_SIZE = 8`. CANN uses `BrcbRepeatParams(1, 8)` with repeat count `R_round / 8`. Use this exact shape only when `R >= 8` (multi-row batched per AIV). For single-row or `R < 8` kernels, use a scalar `GetValue/SetValue` broadcast or `Sub<float, scalar>` overload — both are correct but slower. Do NOT force the Brcb shape on a `R = 1` kernel; the repeat count goes to 0 and the op silently no-ops.

**Concrete anchor** (per-tile non-first body, public-API VEC primitives — copy the *shape*, not vendor identifiers; pick worker-local LocalTensor names):
```cpp
// All buffers are AscendC::LocalTensor<float>; repeatPar1 is BinaryRepeatParams(1,1,1,8,8,8).
// runMax/runSum/newMax/delta/tileMax/tileSum each hold R fp32 elements (R rows per AIV).

// Step 3a: newMax = max(tileMax, runMax)
AscendC::Max<float, false>(newMax, tileMax, runMax, 0, 1, repeatPar1);
AscendC::PipeBarrier<PIPE_V>();

// Step 3b: delta = exp(runMax - newMax)  (uses runMax BEFORE we overwrite it)
AscendC::Sub<float, false>(delta, runMax, newMax, 0, 1, repeatPar1);
AscendC::Exp<float, false>(delta, delta, 0, 1, AscendC::UnaryRepeatParams(1,1,8,8));
AscendC::PipeBarrier<PIPE_V>();

// Step 3c: probs_k = exp(scores_k - newMax) — broadcast newMax via Brcb→Sub (P-P62 shape; requires R>=8)
AscendC::Brcb(tvScratch.ReinterpretCast<uint32_t>(),
              newMax.ReinterpretCast<uint32_t>(),
              R_round / 8, AscendC::BrcbRepeatParams(1, 8));
// then per-column Sub(scores_k, tvScratch) repeats R times across the tile width, then Exp on the tile.

// Step 3d: tileSum = rowsum(probs_k) — use CAND-FA4 block-reduce shape

// Step 3e: runSum = delta * runSum + tileSum
AscendC::Mul<float, false>(runSum, delta, runSum, 0, 1, repeatPar1);
AscendC::Add<float, false>(runSum, runSum, tileSum, 0, 1, repeatPar1);

// Step 3f: commit runMax = newMax
AscendC::DataCopy(runMax, newMax, AscendC::DataCopyParams(1, R_round / 8, 0, 0));
```

**Numerics**:
- Stable because every tile's `Exp` argument lies in `(-∞, 0]`.
- `delta ∈ [0, 1]`; underflow to exactly `0.0` is correct (the old `O_running` contribution is fully dominated). Do not clamp.
- Mask-as-`-3e38` propagates cleanly: an all-masked tile produces `tileMax ≈ -3e38`, `delta ≈ 1`, `probs_k ≈ 0`, `tileSum ≈ 0` — runSum and O_running unchanged. An all-masked row across ALL tiles is a contract violation; handle per op-spec.

**Determinism**: Deterministic when each row is single-AIV-owned (no cross-core writes participate in the state), the per-tile rowmax/rowsum reduction order is fixed (CAND-FA4 block-reduce shape is fixed), and `Exp/Sub/Mul/Add/Max/Div` are per-element. `delta` order matters only via `Mul(runSum, delta, runSum)` which is a per-element scalar product — order-independent. By construction det-preserving when those preconditions hold.

**Hard do-not-apply**:
- Do NOT use this recurrence to produce final probabilities for a *standalone* softmax output without either (a) the streaming-rescaled accumulator path or (b) the stored-numerator-then-replay path. Emitting `exp(scores_k - newMax_at_step_k)` directly is wrong because the centering changes across tiles.
- Do NOT use the `Brcb(R_round/8, BrcbRepeatParams(1,8))` shape when `R < 8`; the repeat count rounds to 0 and the broadcast silently emits nothing (P-P62 precondition violation).
- Do NOT use the unified loop body with `runMax = -inf` initialization; `exp(-inf - newMax)` is undefined for an all-masked first tile and NaN-poisons the row.
- Do NOT clamp `delta` to `epsilon` to avoid underflow; the correct semantics rely on `delta == 0` killing the stale accumulator.

**Other instances predicted**:
- FlashAttention-class forward (`QK` tile → online softmax state → `P @ V` accumulator) — the canonical case.
- Sliding-window / block-sparse attention (each window/block uses one row sweep with this recurrence).
- Streaming softmax fused with downstream reduction (e.g. cross-entropy logsumexp, attention-pooling), where the rescale folds into the consumer.
- Chunked LogSumExp: same `runMax` + `delta * runSum + tileSum` recurrence; final value `log(runSum) + runMax`.
- Online normalization where each update step is a rescaled add (e.g. running weighted mean with re-centered weights).
NOT predicted: streaming L2-normalize (the rescale identity does not factor through `sqrt(sum_sq)` cleanly without a different recurrence — kept out of scope per codex r1).

**Risks before promotion**:
- Brcb precondition R>=8: silently breaks on small-row kernels; worker MUST select a scalar-broadcast variant for `R < 8`. Add a static_assert or runtime guard.
- Mask-as-additive-`-3e38` is the *only* validated path; mask-as-multiply (`scores * mask` with `mask ∈ {0, 1}`) interacts badly with the rowmax step (a zero is not `-inf`). Worker must verify mask form before reusing this pattern.
- The pipeline depends on a separate "rescale_o" stage to consume `delta`. Worker must wire `delta` (or equivalent per-stack-tile state buffer) through to the output-accumulator stage; merging them into one body works only if `O_running` fits in UB alongside the softmax state.
- Source-structure verification only — no a5_ops kernel has yet shipped this exact recurrence. Promotion to P-P requires an a5_ops implementation that passes Pass A + Pass B + det + perf on 3_FusionAttention or a streaming-softmax op, plus a second op confirming portability.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
