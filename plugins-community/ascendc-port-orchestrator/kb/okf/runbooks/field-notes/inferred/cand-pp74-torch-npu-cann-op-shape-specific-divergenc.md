---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "torch_npu CANN op shape-specific divergence from public formula"
description: "Source: pp-2 / 11_DequantSwigluQuant a3 case 32 (2026-04-28) Validation status: 1 confirmed instance (mode=1 gpt-oss SwiGLU, H=2528 N=216), evidence on disk. Symptom: a public-formula fingerprint matc"
phenomenon: build_failure
signal:
  - "a public-formula fingerprint match (P-P71-style: parameter signature like clamp_limit / glu_alpha=1.702 / glu_bias=1.0 → OpenAI gpt-oss) is bit-exact verified a"
confidence: inferred
status: stub
original_id: CAND-PP74
timestamp_inferred: true
tags: [candidate, inferred, convention, requirement, cand-pp74]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

**Source**: pp-2 / 11_DequantSwigluQuant a3 case 32 (2026-04-28)
**Validation status**: 1 confirmed instance (mode=1 gpt-oss SwiGLU, H=2528 N=216), evidence on disk.

**Symptom**: a public-formula fingerprint match (P-P71-style: parameter signature like `clamp_limit / glu_alpha=1.702 / glu_bias=1.0` → OpenAI gpt-oss) is bit-exact verified across many shapes (pp-1 confirmed 7/7), BUT a small subset of shape×N combinations diverge from the torch_npu fused-op reference. The kernel implementing the public formula matches torch_npu on most shapes and diverges on the same shapes torch_npu itself cannot be reproduced from public formula on NPU torch.

**Reproducer pattern**:
```python
# 1. Implement public-formula fingerprint by hand on NPU torch (same dtype/precision as kernel)
xf = x.float() * weight_scale.float() * activation_scale.float()
... # public formula
manual_q, manual_sc = quantize_dynamic(out)
# 2. Compare to torch_npu fused-op reference at the failing shape
ref_q, ref_sc = torch_npu.npu_dequant_swiglu_quant(...)
# 3. If manual_sc - ref_sc deviation matches kernel's deviation → upstream divergence
```

**Verified evidence** (op#11 a3 case 32, 2026-04-28):
- Manual fp32 gpt-oss formula on NPU torch ops, shape `[216, 5056]` mode=1 al=True:
  - vs torch_npu reference: `q_match=85.52%, sc_max_diff=0.1017`
- Kernel (implementing identical formula in AscendC):
  - vs torch_npu reference: `q_match=91.73%, sc_max_diff=0.241`
- Both diverge from reference at the SAME shape with similar magnitude — proving CANN-internal computation differs from publicly verifiable formula.

**Hypothesis (not verified)**: CANN may use different internal tile-boundary precision handling (possibly fp16 intermediate accumulation, different rounding at tile boundaries, or a fused MAC sequence that produces different rounding than chained `mul→add` on torch ops) at certain non-standard (N, H) combinations. Typical magnitude: 1-2 cases out of ~50 in benchmark distributions.

**Treatment / verdict policy**:
- This is a `convention` class divergence — kernel cannot bit-match torch_npu by implementing the public formula (since torch_npu itself diverges from public formula at these shapes).
- Reasonable acceptance: report PARTIAL_PASS (e.g., "49/50 with case 32 documented as upstream torch_npu CANN-internal divergence"), do NOT pursue overfit fix (OL-85).
- A `requirement` verdict is NOT warranted unless msprof reverse-engineering reveals a public-API decomposition that bit-matches the divergent torch_npu output (probe iter ≥6+ territory).

**Related**: P-P71 (public-formula fingerprint), OL-85 (no overfit fixes), OL-91 (convention vs requirement evidence bar), pp-1 §Recommendation.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-PP74，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
