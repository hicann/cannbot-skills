---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "De-scalarize flash online-softmax — hand-rolled mem-based VEC online-softmax + precision-safety triad (replaces `SoftmaxFlashV2` scalar pole)"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=attention-softmax (FA forward/backward, masked-LM, any flash online-softmax) applies_to_backend: ascendc verified_on: soc=Ascend950PR;"
severity: high
confidence: single_run
original_id: P-P101
timestamp_inferred: true
tags: [precision+memory_access, optimization, softmaxflashv2, deinterleavestat8, getvalue, rowmuls, rowdivs, p-p101, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=15.0.5; op_class=attention-softmax (FA forward/backward, masked-LM, any flash online-softmax)`
`applies_to_backend: ascendc`
`verified_on: soc=Ascend950PR; cann=9.0.0 b226; bisheng=15.0.5 (FA-A5 path-B 2026-05-31, in-scope 5/5 precision, commits 5a3a1cee v2 / 697ed8c8 v5)`
`unverified_on: soc=Ascend910_V220 (arch22 — the technique is arch-independent in principle: standard LocalTensor vec ops run on V220 too; A3 precision/e2e NOT yet measured — independent prototype A3 FA cross-pollination pending, feed WITH the kernel≠wall caveat)`

**Trigger**: an attention/FA softmax stage that is scalar-bound — the `AscendC::SoftmaxFlashV2` library call (its internal max/sum reduce + `[rows,8]` stat-packing / `DeinterleaveStat8`) OR per-row `GetValue`/`RowMuls`/`RowDivs`. msprof signature: `aiv_scalar_ratio` is the dominant pipe (A5 FA: 0.27–0.31 inside SoftmaxFlashV2; A3 FA independent prototype: 0.503), the cube starved behind the vec long-pole (Amdahl). Cross-ref OL-200 (the cube/vec pipeline overlap is the complementary lever; this shortens the vec stage so the overlap pays off).

**Technique**: replace the scalar softmax with a hand-written online flash softmax using standard AscendC `LocalTensor` vector ops (NOT MicroAPI register-compute `__simd_vf__`/`RegTensor` — that path is a separate, in-FA-context runtime-UNVERIFIED concern, see CAND-FA-MICROAPI-REG-507015). Per KV-tile, m-loop over rows:
1. **row-max**: `WholeReduceMax` (replaces the SoftmaxFlashV2 internal scalar).
2. **exp(x − rowmax)**: subtract rowmax (broadcast) THEN `Exp` — never bare `Exp(x)` then divide.
3. **row-sum**: `WholeReduceSum`.
4. **online combine**: `newMax=max(runMax,tileMax)`; `corrPrev=exp(runMax-newMax)`; `corrCur=exp(tileMax-newMax)`; `newSum=runSum*corrPrev + tileSum*corrCur` (standard `Max`/`Muls`/`Add`).
5. **O rescale**: `O = O*corrPrev + Otile*corrCur` via `Muls` with the per-row corr **broadcast** across D cols.
6. **final normalize**: `O /= runSum` via vector `Div` (or `Reciprocal`+`Muls`) with runSum **broadcast** — NOT scalar per-row `RowDivs`/`GetValue`.

**Precision-safety triad (mandatory — THE inf-bug fix)**:
1. **masked positions → `minValue` (large negative FINITE, NOT −inf)** → `exp≈0` AND the exp-sum is never 0 (even a fully-masked row sums to `row_len·exp(0)`). #1 inf防线.
2. **subtract rowmax BEFORE exp** (`ExpSub` fused, or `Sub` then `Exp`) — no overflow. Never bare `Exp` then divide.
3. **normalize via vector `Div`/`Reciprocal`+`Muls`** (sum>0 guaranteed by #1) — NOT scalar `RowDivs`.

**Pitfall (cost an iter — stat-buffer aliasing)**: a V-pipe `Brcb(softmaxSumUb_)` for the broadcast-normalize that ALIASES the MTE3 `DataCopy(smSumGm_, softmaxSumUb_)` sm-emit on the SAME buffer → sm_sum corruption + softmax_out inf. Fix: a DISTINCT spare buffer for the Brcb (copy the stat first), OR a proper `SetFlag`/`WaitFlag` barrier between the MTE3 sm-emit and the V-pipe Brcb. (Same class as the tileMax/tileSum aliasing fix.)

**Perf — SCOPED (kernel≠wall; do NOT write a bare "−24%")**: removing the SoftmaxFlashV2 scalar pole gave **kernel-msprof task-duration −24%** (sum 349→264us) + `aiv_scalar` materially reduced — a **kernel-time** result. **e2e WALL was NEUTRAL** in the benchmark (independent same-card A/B, author≠measurer): host BNSD-fold + pybind/wrapper overhead dominate the wall, kernel is only ~20–40% of it, so the kernel win does NOT transmit to the benchmark e2e wall. Whether it reaches a given op's customer e2e depends on that op's bottleneck profile — MEASURE the wall decomposition (host / launch / kernel), do NOT assume. See OL-201 (the pybind-wrapper-wall-not-vendor-fair measurement caveat).

**Anti-pattern**: (a) bare `Exp(x)` then divide (ITER-9 inf path — missing the max-subtract); (b) −inf masking (exp-sum can hit 0 → inf on normalize); (c) scalar `RowDivs`/per-row `GetValue` normalize (re-introduces the scalar pole this pattern removes); (d) Brcb aliasing the sm-emit buffer (the pitfall above).

**Other instances (predicted)**: any scalar-softmax-bound attention op — FA forward/backward, GQA/MQA, masked-LM softmax, flash-decoding; more broadly any per-row reduce-then-normalize stage (the de-scalarize-via-WholeReduce + broadcast-normalize shape) where msprof shows `aiv_scalar` dominant. Detailed concrete params (tile shapes, exact reduce/broadcast op signatures, stat-buffer layout) live in `fa_class/cv_reference_concrete_params.md`.

**Cross-ref**: OL-200 (cube/vec pipeline overlap — the complementary perf lever; softmax de-scalarize shortens the vec stage so the overlap is not Amdahl-capped); OL-201 (kernel≠wall measurement caveat that scopes the −24%); OL-54 (the MicroAPI register path — explicitly NOT used here); CAND-FA-MICROAPI-REG-507015 (the register-reduction route that crashed; this mem-based path is the route-around); P-P62 (Row-Scalar VEC Multiply via Brcb — the broadcast-multiply primitive used in steps 5–6).

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P101，convert_patterns_to_okf.py）。confidence 未升格。 -->
