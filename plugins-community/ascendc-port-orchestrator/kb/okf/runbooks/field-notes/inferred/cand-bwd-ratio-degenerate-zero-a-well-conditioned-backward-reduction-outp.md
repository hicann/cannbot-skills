---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "a well-conditioned backward reduction output whose same-precision competitor hits EXACTLY-0 error makes the competitor-RATIO gate (our/0=inf) structurally unwinnable — but only under the small-N per-case-all-pass fallback, NOT the bootstrap-median path"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 reduction outputs); kernel_type=any verified_on: NPU-independent (grading-gate mechanism + competitor-provenance analysis);"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 reduction outputs); kernel_type=any"
confidence: inferred
status: stub
original_id: CAND-BWD-RATIO-DEGENERATE-ZERO
timestamp_inferred: true
tags: [candidate, inferred, per_case_all_pass, inf, phase_o25_backward, grade_batch, cand-bwd-ratio-degenerate-zero]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=backward-gradient (fp32 reduction outputs); kernel_type=any`
`verified_on: NPU-independent (grading-gate mechanism + competitor-provenance analysis); a5_ops:selective_scan_full_grad fp32 tier 2026-06-30`

A backward op exposes per-output gradients of two kinds: **direct per-element** grads and **reduction** grads (Σ over batch/seq/dim). When a reduction output is **well-conditioned** (the summation is cancellation-free, e.g. `grad_D = Σ_{b,l} gy·silu(z)·u`, or a pure bias-sum grad), a **same-precision competitor** (the fp32 forward with fp32-accumulated reductions, compared to the fp64 autograd golden) can achieve **EXACTLY 0 error** on every element. The cannbot **② competitor-ratio gate** is `our_err / competitor_err`; with `competitor_err = 0` and `our_err > 0` (our independent vector-op transcendental differs from torch's libm by ~1 ULP), the ratio = `+inf` → automatic FAIL. **Passing requires bit-exactness to torch's fp32 libm**, infeasible for any independent AscendC kernel (software-fp32 sigmoid/exp floor ≈ 3.8e-6 abs). This is a **grading-gate degeneracy, not a kernel defect**.

**Decision rule — which gate path is the limiter (the load-bearing distinction)**:
- The `per_case_all_pass` verdict basis is a **small-N circuit-breaker (小样本熔断)**, used ONLY because `N < 200`. Under it, a single `+inf`-ratio element on ONE output fails the whole case → the degenerate-zero output is unwinnable.
- The **intended** path is **bootstrap-median-CI** (valid at N≥200; L1 wants ≥700/dtype), which gates the **MEDIAN** MARE-ratio CI (≤5.0). The median is **robust to outliers** like the `inf`-ratio output. Empirically: a representative distribution (5/8 outputs ratio≈1, A≈5, B≈8, C≈10.5, D=inf, N=700/dtype) yields `ci_upper = 3.65 ≤ 5.0 → PASS`. i.e. the SAME kernel passes once the statistical path is selected.

**Provisioning fix (the actionable lever)**: `phase_o25_backward` (the backward-reference generator) MUST emit ≥ the cannbot statistical scale (**L1 ≥ 700 representative randn samples per (case,dtype)**) so `grade_batch` selects the median-CI path instead of the brittle `per_case_all_pass` fallback. A 48-sample backward-truth dataset triggers the small-N breaker and structurally cannot pass a degenerate-zero output, regardless of kernel quality.

Concrete anchor (selective_scan_full_grad / Mamba SSM FULL backward, 2026-06-30, Ascend950PR_9579/arch35): FAIL 41/48 representative; the 7 failers are ALL fp32 reduction outputs A/B/C/D. grad_D competitor MERE = 0.0 on every element (both records) → ratio +inf, unwinnable; grad_A inherits the SoftExp-vs-libm-expf summand diff (mare 5.07); grad_B/C are cross-d atomic-reduction-noise bound (~2-3.4× torch's pairwise fp32). Direct outputs (grad_u/grad_delta/grad_z) + delta_bias all PASS fp32; ALL outputs PASS fp16/bf16 (both at the low-precision noise floor, ratio≈1.0).

**Distinct from siblings**: CAND-KW-FAG-2 is the **MARE small-value-domain** amplification (single-record metric, |ref|<2^-14); THIS is the **competitor-RATIO degeneracy** (our/0=inf) compounded by **gate-PATH selection** (small-N fallback vs median-CI). CAND-SSM-BWD-WEIGHTGRAD-FP32 is a **dtype underflow** fix (return fp32). All three are backward-fp32 grading artifacts but attack different gate stages.

**Anti-pattern (do NOT)**: relax the verify ratio threshold or add a kernel branch to mask the degenerate output (OL-85 reward-hacking). The fix is in the **reference-dataset scale** (harness-side provisioning), not the kernel or the per-op verify.

**Promote when**: a 2nd backward op with a well-conditioned reduction output reproduces "competitor_err=0 → ratio inf → fails under small-N, passes under median-CI", confirming the gate-path-selection rule generalizes. Cross-ref: CAND-KW-FAG-2 (MARE sibling), CAND-SSM-BWD-WEIGHTGRAD-FP32 (dtype sibling), OL-103 (V220 transcendental floor — consumed but NOT the limiter here), OL-85 (no-reward-hack), OL-109/OL-110 (two-tier verdict / fail-floor family), PRECISION_STANDARD_v2.1 §4.5.1. Source: derived from selective_scan_full_grad (forward_spec_grad) knowledge_update.md 2026-06-30. backend=ascendc.

**Empirical confirmation (2026-07-01) — OLD-vs-NEW dual-grade on REAL fp32 kernel outputs**: the shipped `selective_scan_full_grad` kernel was rebuilt (`--clean`) on A5 (so md5 `4334544868`, Ascend950PR_9579) and its REAL fp32 grad outputs graded through the OLD adapter (git `b4535ddf`, 商用 ratio) vs the NEW adapter (HEAD, 生态 vendored compare.py) — the adapter is the ONLY variable (kernel outputs held FIXED). In the op's real **small-N regime** (16 representative = 2 records × 8 grads, the same `per_case_all_pass` regime as the real 41/48 FAIL): 2 `grad_D` cases have competitor_err EXACTLY 0 → OLD ratio=inf=unwinnable FAIL despite our abs-err ~1e-5 (accurate) → NEW correctly PASS; +4 more reduction grads flip via the finite-ratio family (ratio 1.8–5.2 penalizing an accurate kernel). NEW **discriminates** (not blanket-loosening): a negative control (perturbed grad_A rel-err 2.50) → NEW FAIL, and one `grad_C` OLD-ratio passed (1.11) → NEW FAIL (our_mare 4.6e-3 > 生态 fp32 bar). Evidence: `docs/validation/ss_bwd_grader_regrade_2026_07_01/` (whitebox_log.md + FINAL_old_vs_new.json + reproducible dual_grade scripts).

**Scope / nuance (do NOT over-claim, per 2026-07-01 framing)**: the OLD false-FAIL above is demonstrated in the op's REAL small-N regime — the FULL mechanism = ratio-gate degeneracy × small-N `per_case_all_pass` fallback. At N≥700 the OLD median-CI path **MIGHT** tolerate the inf-outlier (the `ci_upper ~3.65 ≤ 5.0` figure in the Decision-rule above is **ANALYTICALLY ESTIMATED, NOT a real 700-sample run**). The direction is **asymmetric**: if the estimate is wrong and large-N ALSO fails, it only STRENGTHENS "OLD ratio is broken", never weakens it. Either way the **load-bearing fix does NOT depend on N**: the merged 生态-absolute standard (vendored compare.py, PRECISION_METRICS_CANONICAL §0) removes the ratio path entirely → robust at any N. So the reckoning story is: OLD 商用-ratio false-FAILs well-conditioned/degenerate fp32 grads **at the op's real scale**; NEW 生态-absolute correctly passes them AND still catches real error.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-BWD-RATIO-DEGENERATE-ZERO，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
