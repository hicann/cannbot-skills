---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "a sequential exp-scan backward is fp32-imprecise at cancellation points (2.4–14.5× worse than CPU-fp32 autograd) — compensated (Dekker double-single / Kahan) accumulation in the scan state recovers it, but DEFER when it does not flip the terminal verdict"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=backward-gradient (sequential exp-scan, e.g. Mamba SSM) verified_on: a5_ops:selective_scan_full_grad fp32 tier 2026-07-01 (Ascend950PR_95"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=backward-gradient (sequential exp-scan, e.g. Mamba SSM)"
confidence: inferred
status: stub
original_id: CAND-SEQ-SCAN-BWD-FP32-COMPENSATED-ACCUM
timestamp_inferred: true
tags: [candidate, inferred, cand-seq-scan-bwd-fp32-compensated-accum]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=backward-gradient (sequential exp-scan, e.g. Mamba SSM)`
`verified_on: a5_ops:selective_scan_full_grad fp32 tier 2026-07-01 (Ascend950PR_9579)`

A backward kernel whose forward is a sequential exp-scan (`x[l]=dA·x[l-1]+Δ·B·u`) accumulates the reverse-scan gradient state across L steps. At **cancellation points** (grads where opposite-sign summands nearly cancel — here grad u/B/C) plain fp32 accumulation is **2.4–14.5× worse** than a CPU-fp32 autograd reference (kernel MARE ~2.1e-3 vs native 9.3e-4); CPU-fp32 passes the 生态 fp32 gate (2^-13), the kernel does not. This is a **GENUINE kernel imprecision** (the standard SHOULD catch it), distinct from the grader-degeneracy sibling.

**Fix (candidate)**: double-single (Dekker 2×fp32) / Kahan compensated accumulation of the sequential scan state → fp32 up to 16/16.

**DEFER rule (P7 — the load-bearing decision)**: when fixing fp32 alone does NOT flip the terminal verdict (op still blocked on a separate gate), a compensated-accumulation rewrite of the sequential scan is a 3–5 iter, floor-regression-risk change with NO verdict-flip value NOW → DEFER to a future kw spawn that applies it WHEN it actually contributes to the terminal PASS. (selective_scan_full_grad: fp32 12→16 does not flip terminal FAIL while DEBT-180 blocks 32 fp16/bf16 cases; Kahan deferred until after the harness truth-gen fix lands.)

**Distinct from the grader-degeneracy sibling**: this is a real kernel error that NEW correctly FAILs (it serves as a natural negative control proving the 生态 standard discriminates) — NOT a well-conditioned grad falsely-failed by the OLD ratio-gate (that is CAND-BWD-RATIO-DEGENERATE-ZERO). Same op, same fp32 tier, opposite direction.

**fp32-backward CAND family (each attacks a DIFFERENT aspect — not duplicates)**: (1) CAND-KW-FAG-2 = MARE small-value metric ARTIFACT (grading, "not a kernel bug"); (2) CAND-SSM-BWD-WEIGHTGRAD-FP32 = dtype underflow (return grads in fp32); (3) CAND-BWD-RATIO-DEGENERATE-ZERO = competitor-ratio degeneracy (grading gate); (4) THIS = a GENUINE accumulation imprecision the standard SHOULD catch (real kernel error, fix=compensated accum). #1/#3 are grading artifacts; #2/#4 are real kernel issues — mine is the only one whose fix is compensated accumulation.

**Promote when**: a 2nd sequential-scan backward reproduces "plain fp32 scan-state accum N×-worse than CPU-fp32 at cancellation, recovered by compensated accum". Cross-ref: CAND-BWD-RATIO-DEGENERATE-ZERO (grader-degeneracy sibling), CAND-KW-FAG-2 + CAND-SSM-BWD-WEIGHTGRAD-FP32 (fp32-backward family), DEBT-180 (the co-blocking harness truth-gen bug), PRECISION_METRICS_CANONICAL §0.2 (fp32 2^-13 gate). Source: selective_scan_full_grad e2e 2026-07-01 (workspace/forward_spec_grad/verification.json root_cause cause_2). backend=ascendc.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SEQ-SCAN-BWD-FP32-COMPENSATED-ACCUM，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
