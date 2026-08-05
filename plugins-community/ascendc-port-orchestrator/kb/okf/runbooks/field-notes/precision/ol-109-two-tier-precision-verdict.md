---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Two-tier precision verdict with fail-closed independent-truth boundary"
description: "PASS_T1 and PASS_T2 remain grounded in declared CPU/source truth; a target-only differential is diagnostic and cannot become a final pass."
phenomenon: precision_issue
signal:
  - "Any op-archive verification at Phase O5 where per-dtype Tier-1 MERE/MARE thresholds are unreachable due to a hardware/algorithm precision floor (FMA grouping, hw sigmoid polynomial, parallel-reduction reorder)"
confidence: single_run
original_id: OL-109
classified_by: llm-assisted
timestamp_inferred: true
tags: [precision, verification, ol-109, tier1-tier2, cann-baseline]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发
Any op archive verification. Applies to ALL ops, but the Tier-2 fallback only fires for cases that fail Tier 1 — typically where the strict per-dtype threshold is unreachable due to a precision floor (FMA grouping, hardware sigmoid polynomial, parallel-reduction reorder). See `docs/design/PRECISION_METRICS_CANONICAL.md` for the full 2x2 grid of T1/T2 (judging standards) x Pass A/Pass B (input sets).

## 根因 / 教训
A single strict-vs-CPU threshold rejects kernels that are at the hardware precision floor yet still at least as good as the industry baseline. The rule is a ladder:

**Tier 1 — Strict (preferred):**
```
PASS_T1  ⟺  ours_MERE < threshold(dtype)  AND  ours_MARE < 10 × threshold(dtype)
```
All MERE/MARE computed against CPU truth (`Model.forward` on CPU). Per-dtype thresholds (same as production AscendC SKILL): fp16 = 2^-10 ≈ 9.77e-4, bf16 = 2^-7 ≈ 7.81e-3, fp32 = 2^-13 ≈ 1.22e-4. Integer / bool: bit-exact required — no Tier-2 fallback for ints.

**Tier 2 — Relative (fallback when T1 fails):**
```
PASS_T2  ⟺  ours_MERE ≤ CANN_MERE  AND  ours_MARE ≤ CANN_MARE
```
Both computed against CPU truth on the same case — i.e. ours is at least as accurate as CANN under the strict CPU-truth standard. CANN is the de-facto industry baseline; T2 asserts "we are not worse than the reference".

**Per-case verdict ladder (best → worst):**
| Verdict | Meaning |
|---|---|
| PASS_T1 | Tier 1 passes outright (independent of CANN); CPU+fp64 truth available. |
| PASS_T2 | Tier 1 fails, but ours ≤ CANN MERE/MARE (parity-or-better); CPU+fp64 truth available. |
| FAIL_NO_INDEPENDENT_TRUTH | Required CPU fp64 or selected-source-arch truth is unavailable. Target-NPU behavior may be recorded as a diagnostic, never as a final pass. |
| FAIL | T1 fails AND ours is strictly worse than the declared truth on ≥1 metric. |
| EVAL_ERR | Eval crashed / output-count mismatch / unknown-dtype — judge couldn't classify. |

### Target-only differential when CPU truth is structurally undefined
A target implementation may be run as prior-art or a diagnostic. It cannot establish final truth.
Restore the selected source-arch capture for migration or a CPU fp64 autograd oracle for backward;
otherwise emit `FAIL_NO_INDEPENDENT_TRUTH`.
