---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "1-ULP boundary mismatch signature — probe must be exhaustive, cannot be waived directly"
description: "Category: precision / verification"
phenomenon: precision_issue
signal:
  - "kernel precision test N-M/N PASS, with 1-2 residual cases differing by 1 bit on a single element at a rounding / reduction boundary."
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-83
timestamp_inferred: true
tags: [104308, 106182, ascendc, ol-83]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

- **Category**: precision / verification
- **Loaded by**: Worker (when 1-2 cases remain in Phase D), Probe (when classifying residual signatures)
- **Status history**: the original 2026-04-18 version was titled "torch_npu vs pytorch-native may drift at the 1-ULP boundary — not a kernel bug", with the framework premised on "reference drift is not a kernel bug, can be waived". **2026-04-22 user correction: the reference runs on the same NPU and has not drifted; the reference takes one valid fp32 path (FMA grouping order A), our AscendC kernel takes another valid fp32 path (FMA grouping order B), giving 1-ULP different results at the boundary — these are two correct-but-different implementations**. op#10 SwigluQuant's 2026-04-19 "OL-83 waiver" label nominally accepted PARTIAL, but actually assumed "unreachable without CANN source" — that assumption was falsified by op#16 (2026-04-22) (AscendC does have a ready `SinAlgo::RADIAN_REDUCTION` convention-class fix).
- **Trigger signature (unchanged)**: kernel precision test N-M/N PASS, with 1-2 residual cases differing by 1 bit on a single element at a rounding / reduction boundary.
- **New detection + fix protocol (2026-04-22)**:
  1. Official verifier fails → record failing case + unequal count + ratio
  2. Write a standalone probe: kernel vs pytorch-native CPU reference
  3. Compare: is kernel == pytorch-native bit-exact? If yes → **proceed to step 4, do not waive directly**; if no → kernel has a real bug
  4. **Systematically search AscendC available APIs** (op#16 lesson):
     - `find /data/cann_b103/cann-9.0.0 -type f -name "*.h" | xargs grep -l <op_keyword>` to find fused APIs
     - Check under `adv_api/<category>/` for HIGH_PRECISION / RADIAN_REDUCTION / strict-mode variants
     - Check AscendC API catalog §Config: common primitives have `*Algo::` enum parameters
  5. If step 4 finds a ready API → **convention-class fix, not a waiver**
  6. If step 4 is exhausted with no solution → **only then** consider marking PARTIAL; at that point probe_report.md must list:
     - paths / keywords searched
     - every candidate API tried and the result
     - why each one does not apply
     - only then may an honest PARTIAL be accepted (not a waiver — "waiver" implies "we accept the drift", actually "we did not find a fix")
- **Forbidden actions** (stricter):
  - FORBIDDEN: add a bias offset to fit the boundary (reward hacking, OL-85)
  - FORBIDDEN: reverse-engineer CANN source (NPUKernelBench scope forbidden)
  - FORBIDDEN: mark "OL-83 waiver" without doing the step 4 exhaustive search
  - FORBIDDEN: use "reference drift / unreachable" framing — both are unvalidated assumptions
- **Counter-examples (historical misses)**:
  - op#10 SwigluQuant: the original probe only did an 11-step logic diff, did not search `adv_api/quant/` for a fused-SwigluQuant. 2026-04-22 grep found `swi_glu_quant_static.h` + `swi_glu_quant.h` + `block_epilogue_dequant_swiglu.h` — **possibly a usable API, pending probe verification**. Tracked as DEBT-NEW.
  - op#16 Batched2DRopeBack: initially assumed "need Payne-Hanek 50 LoC"; probe actually found `SinAlgo::RADIAN_REDUCTION` (Cody-Waite) — **convention fix, not a requirement**
- **Evidence history**:
  - 9_TopKTopP cold-run probe iter 5 (2026-04-18) — original case; adv_api was not checked for a cumsum HIGH_PRECISION variant, should be re-probed today
  - op#10 SwigluQuant (2026-04-19) — similar "waiver" label but not exhaustively searched
  - op#16 Batched2DRopeBack (2026-04-22) — assumption falsified by probe, became the trigger event for withdrawing the OL-83 "waiver is acceptable" framework
  - **9_TopKTopP a3 V200 (2026-04-30, `topktopp-pp-1` aog-precision-probe)** — case 33 ([128, 16384] fp16) shows 1-ULP cumsum drift between our manual ascending fp32 cumsum (in scalar-S sequential add order over k survivors) and `torch_npu.npu_top_k_top_p`'s internal CANN cumsum. Single position [127, 5561] flips between -inf and 5.98 because top-p threshold lands EXACTLY on the boundary; both valid mathematically. Probe verifies ours_MERE = ours_MARE = 0.0 vs CPU truth (T1 PASS); CANN_MERE = CANN_MARE = 0.0 vs CPU truth too — both bit-exact to CPU, just disagree with each other at the 1-ULP boundary. Same exact phenomenon as a5 9_TopKTopP archive's "29/50 → 49/50, the remaining 1/50 is torch_npu vs pytorch-native 1-ULP cumsum drift, not a kernel bug" residual. Counts as architectural-equivalent residual; OL-83 step-4 search done (no HIGH_PRECISION cumsum API in AscendC), accepted as honest residual on T1-vs-CPU PASS.
  - **9_TopKTopP cluster {8,17,26,35} bf16 mantissa-collision amplification (2026-05-03, `topktopp-pp-3`)** — same 1-ULP-at-boundary mechanism, but AMPLIFIED 100× when dtype is bf16/fp16 AND N is large (≥16384). The limited mantissa quantizes ~65K random fp32 values down to ~256 distinct bit patterns → top-K boundary lands inside a 248-284-position tie cluster → our `Sort<MERGE_SORT>` keeps 121-of-248 tied positions; CANN ref keeps 74-of-248 (both valid under op spec). Verifier reports `max_abs_diff = FLT_MAX` and `mean_abs_diff = inf`, but `finite_max_diff = 0.0` (only -inf vs finite mask flips, no value disagreement). T1-vs-CPU triage (see CAND-PP80) shows `kernel_vs_cpu_truth_flips ≈ ref_vs_cpu_truth_flips` (case 17: 104308 vs 106182) — **CANN ref is NOT MORE CORRECT than our kernel** vs fp64 truth; both diverge equally. Verifier methodology refinement: should permit `n_flip_positions ≤ Σ_rows tie_count_at_kth_value` as T2-with-evidence carry-over conditional on `finite_max_diff = 0.0`. Anti-fix: do NOT attempt to reproduce CANN's specific tie-break order; that is OL-85 territory (case-specific overfitting that breaks on CANN updates).

**Case-set instability lesson (op#9 9_TopKTopP, kw-1 / kw-2 / kw-3 cross-session, 2026-04-30 to 2026-05-03 Ascend950PR_9579)** — **applies to ANY OL-83 case-set, not just op#9**:

The specific case-indices that fall on an OL-83 cumsum-boundary are **non-deterministic across sessions** because the CANN reference uses parallel reductions whose floating-point order may differ run-to-run:

| Session | OL-83 failing case-set |
|---------|------------------------|
| kw-1 (2026-05-02) | {17, 26, 35} |
| kw-2 (2026-05-03) | {17, 26, 35} |
| kw-3 RADIX iter1 (2026-05-03) | {8, 17, 26, 35} |
| kw-3 MERGE revert × 3 consecutive runs | {8, 17, 26, 35} stable |

Across the kw-3 transition, the case-set grew by 1 (case 8 [512, 65536] joined the cluster) NOT due to any kernel change — verified by 3 stable runs on the kw-2-architecture MERGE-reverted build. The drift is on the CANN reference side.

**Implications for floor definitions and worker handoffs**:
- Pass B floor gates based on case-index identity (e.g. "must achieve 47/50 with cases {17,26,35}") will randomly fail when CANN drifts. **Don't write floors this way.**
- **Correct floor definition** — pattern-level: "all cases EXCEPT those in the {bf16, N≥65536, top-p threshold-tie} boundary family must pass". A floor is breached when a NEW pattern emerges (e.g. fp32 cases failing, or non-N=65536 bf16 cases failing), not when a NEW case-index from the same pattern joins the existing cluster.
- Worker handoff between sessions: if a worker reports `Pass B 47/50, cases {A,B,C} fail OL-83 T2`, the next worker should NOT treat that exact 3-case set as a hard ceiling. They should treat the **family** (e.g. `bf16 N=65536 cumsum boundary`) as the T2 family and accept any subset of its cases failing.
- `verification.json` should store T2 cases at family granularity: `{family: "bf16-N65536-topp-boundary", indices: [list], stable: false}` not just `[17, 26, 35]`. Cross-session re-verification compares family identity, not index identity.
- **Related**:
  - OL-68 (fallback to pytorch-native when a torch_npu fused op is unavailable on Ascend950PR) — OL-68 scope: torch_npu FAILS; OL-83 scope: torch_npu and kernel both run on NPU but give different results
  - OL-85 (logic-first fix) — step 4 systematic API search is the tool layer of logic-first (first verify the algorithm, then explore APIs)
  - OL-97 covers CPU-truth verification; OL-83 covers two valid fp32 paths on the same NPU
  - P-P52 (fp32 promotion) — different problem

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-83（category=precision / verification，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
