---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "fp32 backward-gradient MARE fail-floor — MERE-perfect-but-MARE-over-threshold is a small-value-domain metric artifact, not a kernel bug"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 dtype tier) verified_on: NPU-independent (CPU torch_fp32-vs-fp64 triage); a5_ops:flash_attention_grad fp32 tier An fp32 bac"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 dtype tier)"
confidence: inferred
status: stub
original_id: CAND-KW-FAG-2
timestamp_inferred: true
tags: [candidate, inferred, mare_thr, requirement, cand-kw-fag-2]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 dtype tier)`
`verified_on: NPU-independent (CPU torch_fp32-vs-fp64 triage); a5_ops:flash_attention_grad fp32 tier`

An fp32 backward op showing **MERE perfect (≈0.0 — mean accuracy at or better than the fp16/bf16 path
that passes T1) but MARE > `mare_thr = 10·2^-13 ≈ 1.22e-3`** is almost certainly the small-value-domain
metric-amplification fail-floor, NOT a kernel defect. The MARE is driven by genuinely-near-zero gradient
elements (|ref| < 2^-14) where the abs-err ~1e-8 (fp32 ULP floor) becomes a large RELATIVE error.

**Discriminator (NPU-independent, ≤30s — CAND-PP80 triage specialized to the metric)**: run the verify
metric on a **same-precision CPU reference (torch_fp32 vs fp64)**. If that reference ALSO exceeds
`mare_thr` on a large fraction of records with the MARE driver in |ref| < 2^-14, the residual is a metric
artifact → classify `requirement`, ship PASS_WITHIN_TOLERANCE with Tier-2 evidence. fp32 cannot exceed
fp32 — a kernel sitting at torch_fp32 ULP parity is at the best attainable accuracy.

**Distinct from OL-110 (reduction-tree fail-floor)**: even the cancellation-FREE output (dV here — no
scatter-add, no cross-row sum) is affected → this is an **output-floor ULP property**, NOT
reduction-ordering cancellation, so there is NO compensated-summation / reduction-shape lever. Do not
sweep reduction algorithms hoping to close it.

Concrete anchor (flash_attention_grad fp32 tier, 2026-06-14): fp32 MARE ci[0.0038, 0.0065] > 1.22e-3;
full 20-draw CPU triage (300 records) — torch_fp32-vs-fp64 ALSO fails the same threshold on 254/300
(84.7%, worst 8.73e-2 ≈ 71× thr), MARE driver in the small-value domain on 270/300 (90%). Maps to
PRECISION_STANDARD_v2.1 §4.5.3 (small-value-domain, Small-Value-Threshold = 2^-14) + §4.5.1
(competitor-ratio MARE_npu/MARE_baseline ≈ 1.0 ≤ 2).

**Anti-pattern (do NOT)**: relax the verify `mare_thr` inside the op's verify to force PASS
(reward-hacking the grading contract, OL-85); add a near-floor-epsilon kernel branch to mask the elements
(OL-85 forbidden Phase-D pattern). Whether `mare_thr = 10·2^-13` (max-relative-error) is structurally too
tight for fp32 backward gradients is a HARNESS-OWNER threshold question (a same-precision reference fails
it on ~85% of records), NOT a per-op edit.

**Promote when**: a SECOND fp32 backward op (another attention / norm / gemm backward) reproduces the
MERE≈0-but-MARE-over discriminator with the CPU same-precision reference also failing — confirms the
recognition rule generalizes across backward op-classes. On promotion, move to
`patterns/domains/precision.md` (cross-ref OL-109).

**Cross-ref**: OL-83 / OL-110 (fail-floor sub-families — this is the output-floor ULP sibling), OL-109
(two-tier verdict — PASS_WITHIN_TOLERANCE for classified residuals), CAND-PP80 (the T1-vs-CPU-fp64 triage
this specializes to the MARE metric), PRECISION_STANDARD_v2.1 §4.5.1 / §4.5.3, CAND-KW-FAG-1 (the
cast-free fp32 tier whose grading then lands on this floor).

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-KW-FAG-2，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
