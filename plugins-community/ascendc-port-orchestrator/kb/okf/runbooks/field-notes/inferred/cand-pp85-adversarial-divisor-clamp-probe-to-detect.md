---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Adversarial divisor-clamp probe to detect torch_npu fused-op docstring divergence"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=dynamic-quant-fused verified_on: soc=Ascend950PR; cann=9.0.0 unverified_on: soc=Ascend910_V220 (A3 chip family — torch_npu fused-op behav"
phenomenon: build_failure
signal:
  - "porting a dynamic-quantization-class CANN fused op where the public reference docstring specifies clamp(quant_scales, min=epsilon) (typical: 1e-10) for divisor"
confidence: inferred
status: stub
original_id: CAND-PP85
timestamp_inferred: true
tags: [candidate, inferred, amax, cand-pp85]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=dynamic-quant-fused`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`unverified_on: soc=Ascend910_V220 (A3 chip family — torch_npu fused-op behavior may differ across chip families; re-probe before relying on this divergence on A3)`

**Trigger**: porting a dynamic-quantization-class CANN fused op where the public reference docstring specifies `clamp(quant_scales, min=epsilon)` (typical: 1e-10) for divisor safety, BUT the actual `torch_npu.npu_<op>` implementation may NOT apply the clamp — it uses the unclamped amax/N_levels divisor and relies on natural overflow + output clamp to produce saturation behavior.

**Principle**: when CANN fused-op docstrings describe a "clamp divisor at small ε" safety guard, the **public docstring may diverge from the actual fused-op kernel's algorithm**. A docstring-literal kernel writing `qs_div = max(out_scale, 1e-10); dyn_scale = 1/qs_div` produces all-zero output for degenerate rows (tiny magnitudes where `amax / N_levels << ε`), while the reference fused op produces non-zero saturated int8 output (because it uses the unclamped tiny `amax` directly).

This is the **same family of divergence as P-P58.X (swiglu_mode mode-flag dispatch)** — different axis (divisor-clamp vs mode dispatch), same root cause (docstring not authoritative for fused op).

**Diagnostic probe (1-shot detection)**: write a small const-magnitude probe with all inputs at very tiny magnitude (~5e-7 or smaller), where `amax / N_levels` would land below the docstring's clamp ε. Run the reference; inspect whether output has non-zero int8 values on the max positions. If yes → divergence confirmed; drop the clamp in your kernel.

**Concrete fix**:
```cpp
// docstring-literal (BROKEN on degenerate rows)
qs_div = std::max(out_scale, 1e-10f);
dyn_scale = 1.0f / qs_div;

// matches torch_npu actual behavior
if (y_max > 0.0f) dyn_scale = CLIP_MAX / y_max;
else              dyn_scale = 0.0f;
```

The `y_max > 0` guard catches the exact-zero-row degenerate case (where `stored_scale = 0` and `output = 0` is correct); for any positive y_max, use natural `127 / y_max` (no ε floor). Output clamp `[-128, 127]` handles any overflow naturally.

**Quantified evidence (op#11 v3.2 cold-restart, 2026-04-21)**: edge-dataset cases `dist_const_near_zero_seed{0,1}` (inputs ~5.95e-7) — docstring-literal kernel produced all-zero output; torch_npu reference produced `[2, 51, 26, 21, 104, 127, ...]`. Drop-clamp fix → 24/24 int8 bit-exact across full edge dataset.

**Cost / risk**:
- Removes a "safety" guard from kernel — but the actual safety is the output clamp `[-128, 127]`, not the divisor clamp.
- Risk: zero-amax row (true zero input) — handled by the explicit `if (y_max > 0)` guard.
- Pre-probe ritual takes ~5 min; saves multi-iter precision-probe loops on degenerate edge cases.

**Promote when**: 2nd CANN dynamic-quant-class op (e.g. AddRmsNormDynamicQuant, RmsNormDynamicQuant, SwigluDynamicQuant, GroupQuantize) confirms the same divergence pattern AND probe documented + drop-clamp fix applied. Will likely co-promote with P-P58.X into a unified "CANN fused-op docstring is not authoritative" pattern family.

**Anti-pattern avoided**: trusting fused-op docstrings as authoritative. Pass A (benchmark) inputs typically have normal magnitudes where divisor clamp never kicks in, so the bug is invisible without an adversarial Pass B / edge-dataset probe.

**Source**: op#11 DequantSwigluQuant v3.2 cold-restart (2026-04-21). 1-op evidence (sibling family P-P58.X). Needs second dynamic-quant-class op to promote.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP85，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
