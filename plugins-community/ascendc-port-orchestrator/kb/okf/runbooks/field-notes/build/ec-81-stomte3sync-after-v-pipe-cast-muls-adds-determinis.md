---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`SToMTE3Sync()` after V-pipe Cast/Muls/Adds → deterministic garbage on V351 — must use `VToMTE3Sync()`"
description: "<!-- applies_to_backend: ascendc -->"
phenomenon: build_failure
signal:
  - "fp32 outputs are fine (bit-exact to reference), but fp16 and bf16 outputs are deterministic garbage — fp16 max_abs_error ~7.3e+0 (completely wrong), bf16 max_ab"
confidence: single_run
original_id: EC-81
timestamp_inferred: true
tags: [ascendc, ec-81]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

<!-- applies_to_backend: ascendc -->
`applies_to: soc=Ascend950PR (V351); cann=9.0.0; bisheng=n/a; op_class=all (any vec-side kernel storing fp16/bf16 results to GM after V-pipe compute)`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A2/A3 — V220 likely has implicit V→S forwarding that masks the wrong sync)`

- **Error pattern**: fp32 outputs are fine (bit-exact to reference), but fp16 and bf16 outputs are **deterministic garbage** — fp16 max_abs_error ~7.3e+0 (completely wrong), bf16 max_abs_error ~4.8e+5 (overflow-scale). The error is consistent across runs (same input → same wrong output). Only affects half-precision paths; fp32 path (which skips Cast) is unaffected.
- **Root cause**: On A5/V351, the `Cast` intrinsic writes its result to UB via the V (vector) pipe. To transfer this data to GM via MTE3 (`DataCopyPad`), the sync primitive must come from the **V pipe** (`VToMTE3Sync()`), NOT from the scalar pipe (`SToMTE3Sync()`). The `SToMTE3Sync()` synchronizes the S pipe with MTE3, but the Cast result hasn't been drained from V yet → MTE3 reads **stale UB contents** (prior-iteration leftover or uninitialized). On V220, this may have been benign due to implicit V→S forwarding inside the chip, but V351 enforces strict pipe separation.
- **Fix**:
  ```cpp
  // BEFORE (V220-compatible but V351-broken):
  Cast(halfOut, fp32Buf, RoundMode::CAST_ROUND, count);
  SToMTE3Sync();                   // Scalar→MTE3 sync — WRONG pipe
  DataCopyPad(gmOut[offset], halfOut, count);

  // AFTER (V220+V351 correct):
  Cast(halfOut, fp32Buf, RoundMode::CAST_ROUND, count);
  PipeBarrier<PIPE_V>();           // drain V pipe first
  VToMTE3Sync();                   // V→MTE3 sync — CORRECT pipe
  DataCopyPad(gmOut[offset], halfOut, count);
  ```
- **Key insight**: The sync primitive naming follows `{srcPipe}ToMTE3Sync()` — `SToMTE3Sync` syncs S→MTE3, `VToMTE3Sync` syncs V→MTE3. A `Cast` (or `Muls`/`Adds`/any VEC op) output lives on V. Using `SToMTE3Sync` to guard the MTE3 store after a VEC op is always wrong — the correct primitive is `VToMTE3Sync` + `PipeBarrier<PIPE_V>`.
- **Bisect method**: if fp32 path passes but fp16/bf16 produce garbage on V351, immediately grep for `SToMTE3Sync` at every site following a `Cast` / `Muls` / `Adds` / VEC op. The fp32 path typically skips the Cast (direct DataCopyPad), so the wrong-sync half/bf16 path is the only one with the Cast→SToMTE3Sync→DataCopyPad chain.
- **Distinction from existing Cast+V-pipe OL**: the broader OL entry (Cast→PipeBarrier<PIPE_V>→DataCopy) diagnoses `PipeBarrier<PIPE_V>` as insufficient because it doesn't sync V→MTE3. This EC-81 is the lower-level variant: `SToMTE3Sync()` is the explicitly-wrong primitive (different pipe), not just an insufficient barrier. Both produce the same class of failure (stale UB → GM), but the grep target differs: search for `SToMTE3Sync` after any VEC op, not just `PipeBarrier<PIPE_V>`.
- **Evidence**: ctc_loss_v3 port_a3_to_a5 (2026-06-25, A5/V351, CANN 9.0.0): `CopyOutNegLogLikelihood` and `CopyOutAlphaTensor` used `Cast→SToMTE3Sync→DataCopyPad` for half/bf16 paths. fp32 path (no Cast) was fine. Fix: `VToMTE3Sync+PipeBarrier<PIPE_V>` — fp16 went from max_abs=7.31 (garbage) to max_abs=0.068 (within fp16 tolerance); bf16 went from max_abs=4.8e+5 (garbage) to max_abs=0.57 (within bf16 tolerance for non-T≥100 cases). 15/15 PASS post-fix.
- **Other instances (predicted)**: ANY V220→V351 port that has `Cast(half, fp32) → SToMTE3Sync → DataCopyPad` — this is the most natural V220 code pattern (V220's S pipe might implicitly forward V results). Ports from V220 `op_kernel/*.h` where the output epilogue contains half/bf16 Cast-to-GM paths. Grep for `SToMTE3Sync` in any port_a3_to_a5 workspace kernel source.
- **Related**: unnumbered OL (Cast→PipeBarrier<PIPE_V>→DataCopy produces garbage on V351 — same root class, different API surface); OL-260 (member shadowing — the ctc_loss_v3 session also had both bugs).

<!-- 迁移自 porter kb/target/ascendc/（EC-81，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
