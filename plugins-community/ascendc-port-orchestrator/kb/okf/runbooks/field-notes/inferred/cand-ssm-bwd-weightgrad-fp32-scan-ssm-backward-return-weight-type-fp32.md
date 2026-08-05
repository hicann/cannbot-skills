---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "scan/SSM backward — return weight_type=fp32 reduction grads in fp32, NOT the activation input dtype (else tiny→0 underflow / large→inf overflow)"
description: "applies_to: soc=all; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN, any op with mixed weight/activation params); scope=pybind-output-dtype + verify-tru"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN, any op with mixed weight/activation params)"
confidence: inferred
status: stub
original_id: CAND-SSM-BWD-WEIGHTGRAD-FP32
timestamp_inferred: true
tags: [candidate, inferred, float, touch, cand-ssm-bwd-weightgrad-fp32]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=all; cann=all; bisheng=n/a; op_class=scan/SSM/linear-recurrent backward (selective_scan, mamba, GDN, any op with mixed weight/activation params); scope=pybind-output-dtype + verify-truth-dtype; kernel_type=SIMD`
`verified_on: selective_scan_full_grad backward, a5 Ascend950PR_957b cann-9.1.T500, 30/30 PASS (fp32/fp16/bf16 10/10 each), run x2 stable (2026-06-18)`

**The op-class shape**: a backward that returns grads for BOTH activation-type params (input dtype = fp16/bf16: u, delta, B, C, z) AND weight-type params (framework keeps these fp32 regardless of activation dtype: A, D, delta_bias). The reference framework (PyTorch mamba) declares `weight_type = float` and returns dA/dD/ddelta_bias **in fp32, never rounded to the activation dtype**. The activation grads ARE returned at the activation dtype.

**Anti-pattern (the bug)**: pybind uniformly rounds EVERY output grad to the activation input dtype (`grad.to(input_dtype)` for all 8). For the weight grads this is wrong because:
- **tiny-magnitude profile** (inputs ~1e-3 → weight grads ~1e-8..1e-10): fp16's smallest positive denormal is **5.96e-8**; weight grads below that flush to **exactly 0.0** → relative-L2 ≈ 1.0 (looks like a catastrophic kernel bug, but the accumulator was fine). bf16 denormal floor ~9e-41 so bf16 hides it; fp16 exposes it.
- **large-magnitude profile** (inputs ~400 → weight grads ~1e11..1e12): exceed fp16 max **65504** → **inf**.
- Diagnostic tell: the "average" rel-L2 looks like a uniform mid value (e.g. 0.25) but is actually `(N_good × ~5e-4 + N_bad × ~1.0)/N` — i.e. a FEW catastrophic small/large cases, the rest fine. Always break the aggregate down per-case/per-profile before theorizing a uniform precision loss.

**Pattern (the fix)** — TWO coupled edits, both required:
1. **pybind**: return weight-type grads (dA/dD/ddelta_bias) in **fp32** (drop `.to(activation_dtype)` for those; keep it for the activation grads). The weight INPUTS are already fp32, so fp32 grads are dtype-consistent. Accumulation is (and should already be) fp32 via `SetAtomicAdd<float>()` into fp32 GM — this pattern is NOT about the accumulator, it is about the final output dtype.
2. **verify**: compare the weight grads against the fp64 truth cast to **fp32** (not the activation dtype). If the verify downcasts truth to fp16, the truth itself underflows to 0 and the bug is masked as "matches 0". Same atol/rtol — this is a truth-dtype correction, NOT a tolerance loosening.

**Residual honesty (do NOT mask)**: activation grads on the large profile genuinely overflow fp16 (true |grad| > 65504) — NO fp16 output can represent them; the reference (source mamba) returns them inf/nan too. A high-dynamic-range metric (skip non-finite truth elements, atol+rtol on significant elements) correctly passes these on the representable grads. This is a real fp16 dynamic-range edge, characterize it — don't "fix" it by changing the activation output contract.

**Refuted sibling-hypotheses (logged so a graybox doesn't chase them)**:
- "reduction grads accumulate in fp16, fix = fp32 accum" — REFUTED by reading the kernel: GM buffers and atomic-add were already `float`/fp32. The accumulator was never the problem.
- "rounding ALL grads to fp16 causes a uniform 0.25" — REFUTED by A/B: fp16-rounding a well-scaled grad (~1e3) changes rel-L2 by ~0 (~5e-4). The 0.25 was 2 underflow cases, not uniform.
- General lesson reinforced: a root-cause derived from the REFERENCE's convention is a HYPOTHESIS; the on-device A/B against OUR artifact decides. Verify the output **dtype** of the built .so (not the build exit code) — an incremental rebuild can silently keep a stale pybind .o (identical .so size tell); force `touch`+`rm .o/.so`+relink and re-check the artifact.

**Promote when**: a second mixed-weight/activation scan/SSM backward (e.g. GDN backward, mamba2) reproduces the same weight-grad-fp32 requirement, confirming it is the op-class convention and not selective_scan-specific.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-SSM-BWD-WEIGHTGRAD-FP32，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
