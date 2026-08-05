---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Advanced `Sort` 6-arg overload with `srcIndexTensor` triggers aivec 343 on Ascend950PR"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Runtime aicore exception 507015 / errcode 343 / \"Incorrectly sorted data entered by the VMS\" at the MrgSort step inside hardware radix sort, when invoked via th"
confidence: single_run
original_id: EC-33
timestamp_inferred: true
tags: [507015, sort, srcindextensor, ascendc, ec-33]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Runtime `aicore exception 507015 / errcode 343 / "Incorrectly sorted data entered by the VMS"` at the MrgSort step inside hardware radix sort, when invoked via the 6-argument `Sort` overload: `Sort<T, U, isReuse, CFG>(dstV, dstI, srcV, srcI, tmp, count)` — even when `srcIndexTensor` is correctly populated with increasing values `[0, 1, ..., count-1]`.
- **Root cause**: Unconfirmed, but reproducible on CANN 9.0.0 / Ascend950PR_9589. Likely a CANN-internal behavior of the srcIndex path in radix sort on this SOC. Same error signature (`aivec 343 "Incorrectly sorted data entered by the VMS"`) appears when `torch_npu.npu_top_k_top_p` runs under sustained calls — suggests a shared CANN-side issue with the radix sort VMS module under certain input patterns, independent of caller.
- **Fix**: Use the 5-argument overload `Sort<T, isReuse, CFG>(dstV, dstI, srcV, tmp, count)`, which auto-generates local indices `0..count-1` and avoids the srcIndex path. If you need global/external indices (e.g. chunk offsets), add the offset manually when reading `dstI` after the call:
  ```cpp
  // Chunked merge example: chunk starts at col_offset
  Sort<float, false, SORT_CFG_DESC>(svOut, siOut, svIn, stmp, count);
  for (int32_t i = 0; i < count; i++) {
      global_idx[i] = siOut.GetValue(i) + col_offset;  // manual offset
  }
  ```
- **Detection**: If you call the 6-arg Sort overload and hit `aivec 343`, switch to the 5-arg variant first before any other debugging.
- **Evidence**: 9_TopKTopP cold-run round 2, Phase C iter 2 (2026-04-18). 6-arg Sort with caller-provided srcIndex → 507015/343. Switching to 5-arg + external offset → OK. Same error signature also observed in CANN reference path (`torch_npu.npu_top_k_top_p` under performance.py sustained calls) on the same SOC — suggests shared internal cause.
- **Additional trigger (2026-04-18 V3.2 9_TopKTopP test 2)**: Even with the 5-arg Sort overload (no srcIndex) and 2-field `SortConfig`, using `SortType::RADIX_SORT` can still trigger VMS 343 at runtime on Ascend950PR under certain input patterns (exact conditions unconfirmed; may correlate with chunk count > 1 or specific value distributions). Switching to `SortType::MERGE_SORT` resolved the runtime crash in one case, and precision stayed at 50/50 PASS.
- **Mitigation — prefer MERGE_SORT by default**: Use `constexpr SortConfig CFG = {SortType::MERGE_SORT, isDescend}` as the default choice on Ascend950PR CANN 9.0.0. Only switch to `SortType::RADIX_SORT` if perf profiling demonstrates a significant improvement AND you can confirm no VMS 343 on representative inputs.
- **Status**: Reproducible on current session's A5 state; may be specific to CANN 9.0.0 + Ascend950PR_9589. Confirmation across different NPU state / CANN version desirable.
- **Benchmarking methodology for affected ops (2026-04-19 calibrated on 9_TopKTopP)**: When the reference itself (e.g. `torch_npu.npu_top_k_top_p`) is implicated in EC-33 VMS 343 crashes, the standard `utils/performance.py current_task all` (default warmup=5 repeat=10 = 750 ref calls for 50 cases) will reliably crash. Empirical threshold map on Ascend950PR CANN 9.0.0:

  | warmup × repeat | Total ref calls (50 cases) | Behavior |
  |-----------------|---------------------------|----------|
  | 0 × 1 | 50 | Never crashes; BUT cold-launch overhead dominates → inflated ratios on small-N cases (launch overhead ≈ compute time); not representative of warm production perf |
  | **1 × 2** | **150** | **Recommended default** — never crashed in 3/3 runs; warm kernel measurement; ratio represents steady-state |
  | 3 × 3 | 300 | Usually OK (1/1 observed) |
  | 5 × 5 | 500 | **Flaky** — 2/3 runs crashed in one experiment |
  | 5 × 10 (default) | 750 | Always crashes |

  **Methodology rules when hitting EC-33-affected ops**:
  1. **Default to `warmup=1 repeat=2` × 3 runs**, take median of per-run sum/median/geomean ratios. This gives warm-kernel numbers without tripping VMS 343.
  2. **Do NOT use `warmup=0 repeat=1`** as a primary measurement. Cold-launch overhead on small-N cases inflates the ratio artifact; numbers are not comparable across sessions. Acceptable only as a fallback when `warmup=1 repeat=2` also crashes.
  3. Worst-case: drop to `warmup=0 repeat=3` (150 calls but all cold) if ref is unusually fragile. Still warm-vs-warm between impls.
  4. Script ratio computation: `src/scripts/perf_ab.py` (promoted from /tmp) reads `performance.py` output files and emits sum/median/geomean ratios + distribution buckets.

  **Illustrative (9_TopKTopP R3b snapshot, 2026-04-19)**:
  - `warmup=0 repeat=1` (50 calls): sum 0.475x (cold artifact — inflated)
  - `warmup=1 repeat=2` × 3 runs (warm median): sum 0.222x (honest number)

  **Cross-session reinforcement (op#9 9_TopKTopP kw-1 + ko-1, 2026-05-02 Ascend950PR_9579)**:
  - `utils/performance.py` 50-case sweep with default warmup=5 repeat=10 → CANN reference hits VMS 343 around case 30–40 in BOTH kw-1 and ko-1 sessions on this SOC, independent of the test kernel; behavior consistent with the kw-1 measurement n=40/50 and the ko-1 measurement n=31/50 before the reference crashed
  - `warmup=1 repeat=2` wall-clock measurement completes for n=31–40 of 50 cases reliably; this remains the recommended methodology for any op whose reference is `torch_npu.npu_top_k_top_p` or another EC-33-affected fused op on Ascend950PR
  - Profiler-based timing (`utils/performance.py`) cannot complete because the profiler's overhead amplifies CANN's sustained-call instability — confirmed across 2 sessions
  - **op#9 9_TopKTopP pp-3 (2026-05-03 Ascend950PR_9579) — 4th data point**: pp-3 ran 5 sequential calls into `torch_npu.npu_top_k_top_p` (cases 8, 17, 26, 35, 44 via `Model().forward()` in a single Python session, NO profiler attached). All 5 calls completed clean. **Corroborates pp-2's "single-call or multi-call without profiler is safe" refinement** (line 685 above). EC-33 trigger appears to require profiler-induced sustained call patterns specifically, not just call count alone. Practical implication: Pass-B style harnesses that loop `Model().forward()` directly without msprof attachment can run the full 50-case sweep on EC-33-affected ops without VMS 343.

  **kw-3 RADIX_SORT retry — 2nd-data-point evidence narrows the trigger (op#9 9_TopKTopP, 2026-05-03 Ascend950PR_9579 + CANN 9.0.0 b103)**:
  - Switched `SortType::MERGE_SORT` → `SortType::RADIX_SORT` (1-line constexpr swap), re-ran 50 Pass B + 50 determinism cases. **VMS 343 did NOT trigger.** RADIX completed all 100 cases without `errcode 343 / aicore exception 507015 / "Incorrectly sorted data entered by the VMS"`.
  - Kernel uses CHUNK_LEN=2048 with up to 32 chunks per row for bf16 N=65536 — well above the originally hypothesized "chunk count > 1" trigger threshold, so the original threshold-correlation hypothesis is weakened.
  - **Perf cost of MERGE vs RADIX on op#9 is ~0% (within noise) on this SOC**, NOT the previously suspected ~38%. Median 0.385× (RADIX) vs 0.388× (MERGE) across n=49 cases. The historical R3b 0.610× archive number was likely a different harness shape mix or CANN sub-version.
  - **Implications**:
    - Trigger is narrower than originally documented — likely correlated with specific input distributions or specific CANN VMS-module state, not RADIX_SORT-on-Ascend950PR in general.
    - **MERGE_SORT remains the safer default** (defensive posture is cheap); op-level retests of RADIX_SORT under representative inputs are reasonable when perf profiling identifies a benefit.
    - When testing RADIX_SORT for a new op, a 50-case Pass B sweep + a determinism check is sufficient to confirm no VMS 343 on the chosen config.

  **Wall-clock truncation variance methodology refinement (op#9 kw-3, 2026-05-03)**:
  - Three back-to-back warmup=1 repeat=2 runs in one container session produced n=22, 40, 49 cases completed before VMS 343 — i.e. the truncation point itself is non-deterministic and depends on NPU sustained-call state.
  - **Don't average median ratios across runs** — each run is computed over a different shape subset; values are not comparable.
  - **Report the run with the largest `n`** as the representative measurement (covers the most shape diversity).
  - **Don't claim 0.6× threshold pass/fail off a single run** — re-baseline at session start, run A and B kernels in same session order with NPU re-init between, and use n=max for each.
  - Cross-session sanity: across 4 sessions of op#9 (kw-1 / ko-1 / kw-2 / kw-3), median ratios all fell in 0.27–0.42× — that range IS the structural ceiling, robust to the truncation-point variance.
  - Gap: 2.1x. The warm number is the correct one for product reporting.

<!-- 迁移自 porter kb/target/ascendc/（EC-33，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
