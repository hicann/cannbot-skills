---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Path A canonical-evaluator §4.5.3 small-value fallback degenerates when the reference equals CPU truth"
description: "In port_a3_to_a5 Path A the §4.5.3 small-value fallback budget collapses to 2 absolute mismatches because cann_error_count is trivially 0 (CPU-truth vs CPU-truth) — too tight to absorb honest bf16/fp16 rounding; drive the kernel to bit-exact instead of loosening tolerance."
phenomenon: precision_issue
signal:
  - "op runs in port_a3_to_a5 Path A (CPU-truth reference) with precision_eval_two_tier.py §4.5.3 small-value fallback enabled, and small-value outliers appear (e.g. bf16 case fails with 5 outliers, budget=2)"
original_id: OL-138
confidence: single_run
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, evaluator-behavior, ol-138, port_a3_to_a5, path-a, small-value-fallback]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

**Applies to** `soc=Ascend950PR; cann=9.0.0; op_class=any port_a3_to_a5 / CPU-truth-reference op`.
Verified on Ascend950PR / cann 9.0.0.

Loaded by aog-kernel-worker (Phase D, when reasoning about §4.5.3 fallback budgets) and
aog-precision-probe (when an op runs on Path A and small-value outliers appear).

**Trigger**: an op runs in `port_a3_to_a5` Path A (CPU-truth reference, because the vendor
torch_npu API is not exposed) AND the precision evaluator is the canonical two-tier
`precision_eval_two_tier.py` with the §4.5.3 small-value fallback enabled.

Concrete signature — ada_layer_norm Path A port (2026-05-13): iter-1 kernel produced 5
small-value outliers in bf16 case 5; the Path A §4.5.3 budget was 2 → FAIL.

## 根因 / 教训

The §4.5.3 fallback rule allows `ours_error_count ≤ 2 × max(cann_error_count, 1)` for
small-value outputs where bit-exactness is implausible. It is calibrated for the standard
benchmark setup where `cann_error_count` reflects vendor-kernel small-value drift against CPU
truth — typically non-zero, giving our kernel headroom.

In Path A mode the "CANN" comparison degenerates to **CPU-truth vs CPU-truth** (no real vendor
kernel exists), so `cann_error_count = 0` trivially. The fallback budget collapses to
`2 × max(0, 1) = 2` absolute mismatches — far too tight to absorb the bf16/fp16 small-value
rounding that any honest kernel will produce. This degeneracy is **structural** (no second
reference exists to give headroom), not a bug.

### Recommended action when this signature applies

1. Treat §4.5.3 as effectively disabled in Path A — the evaluator is in
   "strict-bit-exact-or-very-close" mode.
2. Drive the kernel to literal-translation precision (P-P50, OL-112, OL-81) so element error
   counts drop to a handful at most.
3. Do NOT propose loosening tolerance or adding a Path-A-specific §4.5.3 expansion as a "fix";
   the degeneracy is structural, not repairable by widening the budget.
4. When reporting PARTIAL precision under Path A, distinguish a "true precision gap" from a
   "§4.5.3 would have rescued under Path B" case — the latter is informational, not a defense.

### Evidence

- ada_layer_norm (2026-05-13, A3→A5 kw-1..4): iter-1 had 7 cases PASS but bf16 case 5 hit
  MARE=1240 with 5 small-value outliers; §4.5.3 budget=2 left no headroom. Resolution was
  P-P50 (native-dtype post-modulation) → 8/8 PASS, 4 bit-exact. No evaluator change.

**Predicted other instances**: any future `port_a3_to_a5` op where Path A is forced (vendor
API absent on Ascend950PR); any custom-reference op where the evaluator's secondary reference
is the same as the primary CPU truth (CPU-truth-only benchmarks, hand-rolled reference ops).
(Source text truncated at this point.)
