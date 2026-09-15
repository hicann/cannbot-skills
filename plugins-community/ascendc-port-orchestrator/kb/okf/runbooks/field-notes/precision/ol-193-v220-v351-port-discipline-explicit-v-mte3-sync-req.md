---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220→V351 port discipline — explicit `V→MTE3` sync required between Cast/V-compute and DataCopy store; `PipeBarrier<PIPE_V>` is NOT sufficient"
description: "applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=all (any cube↔vec or vec-only kernel storing fp16/bf16 results to GM after fp32 compute)"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=all (any cube↔vec or vec-only kernel storing fp16/bf16 results to GM after fp32 compute)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-193
timestamp_inferred: true
tags: [ascendc, ol-193]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=all (any cube↔vec or vec-only kernel storing fp16/bf16 results to GM after fp32 compute)`
`verified_on: soc=Ascend950PR (V351, Ascend950PR_9579); cann=9.0.0`
`unverified_on: soc=Ascend910_9382 (V220 / A3 — same kernel source passes 16/16; V220 likely has implicit sync or benign timing)`

**Principle**: When a V220 kernel uses the pattern `Cast(fp16_buf, fp32_buf, ...); PipeBarrier<PIPE_V>(); DataCopy(gmOut, fp16_buf, count);` (V op → V-barrier → MTE3 store), it works on V220 but produces **deterministic garbage on V351**. `PipeBarrier<PIPE_V>` only synchronizes within the V pipe; the DataCopy is an MTE3 op which is NOT synced by a V-only barrier. On V220 the timing happens to be benign (or the hardware has an implicit V→MTE3 dependency); V351 enforces strict pipe separation, so MTE3 fires before V completes → MTE3 stores **stale UB buffer contents** to GM. The stale contents typically read as a deterministic non-zero pattern (e.g. all-`1.0` for fp16 UB initialized to `0x3c00`-fill or carrying prior-iteration leftover) for the first chunk of a loop where MTE3 hasn't yet been preceded by another MTE3 (which would have implicitly drained V).

**Failure signature** (from cv-agent FA cv_agent_fa_a5test on V351, 2026-05-28):
- For all-zero input `q=k=v=0`, expected output is all-zero (softmax(0)·V=0). Actual output: binary `{0.0, 1.0}` with 25 % of elements at 1.0, distributed as **periodic chunks** matching the inner loop's chunk size (here `VEC2_M_CHUNK=8` → 8 rows of `1.0` then 24 rows of `0.0`, repeating). The "8-row" chunk is the FIRST iteration of the per-AIV chunk loop in `ComputeVec2::FinalizeOutputChunk` — chunks 1-3 are correct because prior iter's MTE3 already drained V; only chunk 0 has the missing-sync exposed.
- Output is largely **input-independent** (changing Q or K barely shifts sum; only V scaling moves abs_max) because the stale UB pattern dominates the affected positions.

**Diagnostic flow** (~30 min from "wrong output" to root cause):
1. Probe input-sensitivity: run with `q=k=v=0`, then with `q=randn`, then with `v=randn`. If output is largely input-independent + has binary `{0, c}` pattern → fixed value source (not math).
2. Map binary pattern across `(B, N, S)`: dump where the `c`-valued rows are. Periodic chunk pattern matching some inner loop's `_M_CHUNK` constant points to "first iter of loop".
3. Bisect within `FinalizeOutputChunk`: comment out `RowDivsImpl` or its equivalent normalize → same pattern → not the divide. Replace `Cast(outHalf, oUb)` with `Duplicate(outHalf, sentinel)` → chunks 1-3 take the sentinel value but chunk 0 STILL has the wrong constant → confirms `outHalf` is being overwritten between the V op and the MTE3.

**Fix**: replace `PipeBarrier<PIPE_V>()` between the V op (Cast / Muls / Adds / etc.) and the GM-store DataCopy with `SetWaitFlag<HardEvent::V_MTE3>()`. This is **strictly safer on V220** (V220 was working only by luck), so the fix is portable and does not need a V220/V351 conditional.

```cpp
// V220-compatible but V351-broken (cv-agent original):
Cast(outHalf, oUb, RoundMode::CAST_ROUND, count);
PipeBarrier<PIPE_V>();                                  // V→V only
DataCopy(outGm[outBase], outHalf, count);               // MTE3 reads stale outHalf on V351

// V220+V351 correct:
Cast(outHalf, oUb, RoundMode::CAST_ROUND, count);
SetWaitFlag<HardEvent::V_MTE3>();                       // V→MTE3 explicit
DataCopy(outGm[outBase], outHalf, count);
```

**Evidence**: cv-agent FA `flash_attention_vec.h::FinalizeOutputChunk` lines 286 + 297 (cv-agent-master commit 2026-05-28 patch). After fix: 5/5 PASS across shapes `(B,N,S,D) ∈ {(2,4,64,64), (1,1,64,64), (1,2,128,64), (2,4,64,128), (1,8,256,64)}`, all `max_diff < 0.05` vs PyTorch fp32 reference. Before fix: every shape produced `out_abs_max=1.0` (fp16 saturated) `max_diff ≈ 1.03`. cv-agent on V220 retains 16/16 PASS post-fix (no regression).

**Other instances (confirmed)**:
- cv-agent FA `flash_attention_vec.h::FinalizeOutputChunk` (2026-05-28): `Cast→PipeBarrier<PIPE_V>→DataCopy` produced 25% `1.0` output. Fix: `SetWaitFlag<HardEvent::V_MTE3>()`.
- ctc_loss_v3 port_a3_to_a5 (2026-06-25): `Cast→SToMTE3Sync→DataCopyPad` in `CopyOutNegLogLikelihood` / `CopyOutAlphaTensor` produced garbage (fp16 max_abs=7.31, bf16 max_abs=4.8e+5). Fix: `PipeBarrier<PIPE_V>(); VToMTE3Sync();`. Same root class — wrong pipe sync after V-pipe Cast — but different API surface (`SToMTE3Sync` vs insufficient `PipeBarrier`). See EC-81 for the `SToMTE3Sync`-specific variant.

**Other instances (predicted)**: applies to ANY V220→V351 port that uses `Cast(low_prec, fp32_buf) → PipeBarrier<PIPE_V> → DataCopy(gm_out, low_prec)` epilogue pattern. Concretely: any port_a3_to_a5 op with vec-side fp32 → fp16/bf16 output cast (LayerNorm, RmsNorm, Softmax, fused-norm, fused-quant, attention output, MoE finalize). The lint regex to find candidates: `Cast\(.*PipeBarrier<PIPE_V>\(\).*DataCopy\(.*Gm` over `workspace/<op>/kernel/*_vec.h`. Also grep for `SToMTE3Sync` following any VEC op — the EC-81 variant.

**Cross-ref**: OL-190 (cross-core group barrier on AIV→AIC — different pipe pair, same "implicit-sync-on-V220 broken-on-V351" class), CAND-V220-to-V351-PortPattern-CubeVecFusedOp (the broader port pattern that this OL is a member of), EC-81 (`SToMTE3Sync` after V-pipe Cast — the lower-level API variant of this same class).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-193（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
