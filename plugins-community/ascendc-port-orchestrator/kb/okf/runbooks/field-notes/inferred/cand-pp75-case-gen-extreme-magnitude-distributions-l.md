---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "case_gen extreme-magnitude distributions (`large_mag`, `denormal`, `const_near_zero`) produce overflow/underflow refs in dequant→quant fused ops"
description: "Source: pp-2 / 11_DequantSwigluQuant a3 Pass B 4 cases (2026-04-28) Validation status: 1 op confirmed; pattern likely general for any dequant→activation→quant pipeline. Symptom: Pass B (edge_dataset)"
phenomenon: build_failure
signal:
  - "Pass B (edge_dataset) sign_off-tier cases pulled from case_gen.py distributions large_mag (uniform 1e20..1e30) and const_near_zero (= eps_fp32 5 = 5.96e-7) pro"
confidence: inferred
status: stub
original_id: CAND-PP75
timestamp_inferred: true
tags: [candidate, inferred, large_mag, denormal, const_near_zero, case_gen.py, cand-pp75]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: pp-2 / 11_DequantSwigluQuant a3 Pass B 4 cases (2026-04-28)
**Validation status**: 1 op confirmed; pattern likely general for any dequant→activation→quant pipeline.

**Symptom**: Pass B (edge_dataset) sign_off-tier cases pulled from `case_gen.py` distributions `large_mag` (uniform 1e20..1e30) and `const_near_zero` (= `eps_fp32 * 5 = 5.96e-7`) produce reference outputs that contain `inf`/`nan` (from fp32 overflow on multiplication of two ≥1e20 scalars) or sub-1e-28 normal-range scales (from product underflow). A kernel implementing typical-magnitude-correct dynamic int8 quant tail (with a `clamp(min=1e-10)` div-guard floor or similar) will diverge:

- For `large_mag` cases: ref `quant_scales = [inf, inf, ..., nan]`, ref `quantized_output = [0, 0, ...]` (since dividing by inf gives 0). Kernel either matches (if it also overflows) or differs (if it implements explicit overflow guard).
- For `const_near_zero` cases: ref `quant_scales ≈ 2-3e-28` (still in fp32 normal range, but 18 orders below typical). Kernel's `div-guard floor=1e-10` clamps the scale to `1e-10`, producing different quantized output.

**Confirmed evidence** (op#11 a3 edge_dataset, 2026-04-28):
- Cases 10/11 (`dist_large_mag_seed{0,1}`): `weight_scale.amax ≈ 9.97e+29 / 9.91e+29`, `activation_scale.amax ≈ 9.32e+29 / 4.82e+29`, ref `quant_scales = [inf, inf, inf, inf]` / `[inf, inf, inf, nan]`.
- Cases 18/19 (`dist_const_near_zero_seed{0,1}`): all scales = 5.95e-7, ref `quant_scales ≈ 2.69e-28 / 2.74e-28` (vastly below kernel div-guard floor `1e-10`).

**case_gen.py distributions** (`src/scripts/reference_provider/case_gen.py`):
```
{"tag": "large_mag",      "fn": mk_uniform(1e20, 1e30)}     # produces overflow on op*op
{"tag": "small_mag",      "fn": mk_uniform(1e-30, 1e-20)}   # similar underflow risk
{"tag": "denormal",       "fn": mk_denormal()}              # explicit denormal fp32 inputs
{"tag": "const_near_zero","fn": mk_const(eps_fp32 * 5)}     # 5.96e-7
```

**Treatment / verdict policy**:
- These are `convention`-class divergences for dequant→activation→quant fused ops with multi-tensor scale products: the kernel matches typical-magnitude inputs and diverges only on stress-test extremes that produce ref `inf/nan` or sub-fp32-normal-range scales.
- Honest workflow: Pass B should NOT gate on these for ops where the chain of operand multiplications can overflow/underflow fp32. The right SCHEMA-level fix is to add a `skip_extreme_magnitudes=True` flag (or `extreme_magnitude_only_subset=True`) on per-op SCHEMA, so input_gen.py emits these cases into a separate documentation-only edge subset, not the Pass B precision gate.
- Affected ops: any fused op where ≥2 user-provided scale tensors multiply (dequant→quant pipelines, attention scaled softmax with scale param, layernorm with scaled output, MoE quant pipelines).

**Recommendation for input_gen.py / case_gen.py**:
```python
# In SCHEMA:
SCHEMA = {
    ...
    "skip_extreme_magnitudes": True,  # exclude large_mag / small_mag / const_near_zero / denormal
    # OR
    "extreme_magnitude_subset_only": True,  # emit them but tag as 'edge_documentation_only'
}
```

`case_gen.py` filters extreme distributions when the SCHEMA flag is set; alternatively keeps them in a separate `edge_dataset_extreme.pt` that does not gate Pass B but is documented in `analysis.md` for completeness.

**Related**: OL-85 (no overfit fixes), OL-91 (convention evidence bar), pp-1 §Recommendation; mirrors P-P70 dynamic-quant-tail context but at extreme operand magnitudes.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP75，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
