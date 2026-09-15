---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Quant-op CPU reference MUST `.clamp(low, high).to(int_dtype)` to match NPU hardware clamp"
description: "applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=quant,fused-quant; phase=precision-verify verified_on: soc=Ascend950PR; cann=9.0.0 source: PR 103 references/precision-tes"
confidence: single_run
original_id: P-P93
timestamp_inferred: true
tags: [patterns-index, optimization, round, rms_norm_quant, group_norm_silu_quant, add_rms_norm_quant, p-p93, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=quant,fused-quant; phase=precision-verify`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/precision-testing/OPS_PRECISION_STANDARDS.md "迁移场景特殊考虑 / 量化算子(Quant)精度测试设计规范"`

**Pattern**: When writing CPU reference / golden for a quant op (output dtype is INT8 / UINT8 / INT4-in-INT8), the reference **MUST** clamp to the output dtype's representable range BEFORE the integer cast. NPU hardware clamps automatically; without matching clamp in the reference, out-of-range cases produce false-FAIL on the cast result.

**Concrete pattern**:

```python
# ❌ WRONG — silent saturation difference NPU vs CPU
def cpu_reference_quant_int8(x_fp32, scale):
    pre = x_fp32 / scale
    ref = torch.round(pre).to(torch.int8)   # CPU: overflow → undefined; NPU: clamps to ±127
    return ref

# ✅ RIGHT — explicit clamp matches NPU clamp
def cpu_reference_quant_int8(x_fp32, scale):
    pre = x_fp32 / scale
    ref = torch.round(pre).clamp(-128, 127).to(torch.int8)
    return ref
```

**Clamp range table by output dtype**:

| Output dtype | clamp range | code |
|---|---|---|
| INT8 | [-128, 127] | `.clamp(-128, 127).to(torch.int8)` |
| UINT8 | [0, 255] | `.clamp(0, 255).to(torch.uint8)` |
| INT4 (stored as INT8) | [-8, 7] | `.clamp(-8, 7).to(torch.int8)` |
| FP8 E4M3FN | [-448, 448] (max representable, hardware clamp at ±448 saturating) | `.clamp(-448, 448).to(<fp8>)` |
| FP8 E5M2 | [-57344, 57344] | `.clamp(-57344, 57344).to(<fp8>)` |
| HiFloat8 | n/a (built-in range-encoding, no clamp needed) | `.to(<hifloat8>)` direct |

**Core insight** (PR 103 OPS_PRECISION_STANDARDS.md "核心认知"):
1. **quantScale value need NOT be restricted** — even when quantScale is tiny and most values overflow INT8 range, BOTH NPU and CPU clamp to ±127 → saturation region results agree.
2. **CPU reference MUST add clamp** — to align with NPU behavior; otherwise PyTorch's `.to(torch.int8)` on overflow is **undefined behavior**, not equivalent to saturation.
3. **Precision diff source**: ONLY in the non-overflow region, from FP16/BF16 vs FP32 intermediate compute differences. After `round`, max 1 ULP diff per element.

**Why this matters for us**:
- Cohort 1 archived ops: `rms_norm_quant` (8/8 PASS T1) + `group_norm_silu_quant` (Pass A 8/8 T1, Pass B 2/7 T1 + 5/7 T2). If those tests used unclamped reference, the 5/7 T2 cases on `group_norm_silu_quant` Pass B may be hiding clamp-mismatch instead of genuine fp16/bf16 precision difference. **Worth re-verifying.**
- Cohort 2 quant ops pending: `add_rms_norm_quant`, `flat_quant`, `grouped_matmul_swiglu_quant`, `fused_quant_mat_mul` — all need this rule in their test harness from day 1.
- Upcoming fp8 / mxfp8 / mxfp4 kernels: clamp values per OL-144 narrow-float ranges.

**Detection signature**:

```bash
# Search workspace for quant-test CPU reference code that may be missing clamp
grep -nE "\.to\(torch\.int8|\.to\(torch\.uint8|\.to\(torch\.int4" workspace/*/run_a3_reference.py workspace/*/edge_runner.py | \
  grep -v "clamp"
# Each hit = candidate for adding clamp before .to(...)
```

**Anti-patterns**:
- Limit `quantScale` to avoid overflow — masks the real test surface; NPU handles overflow correctly, test should too
- Use `torch.clamp` AFTER `.to(int)` — UB triggered before clamp, garbage values

**Evidence**:
- PR 103 PRECISION_STANDARDS.md codifies as MANDATORY for quant op test design
- Our `group_norm_silu_quant` Pass B 5/7 T2 result (instead of T1) — possible clamp-mismatch contribution; verification.json pre-dates this rule

**Other instances (predicted)**: every quant / dynamic-quant / quantize-with-scale op in cohort 2; every fp8/mxfp8/mxfp4 kernel where output is the narrow type.

**Cross-reference**:
- OL-144 (narrow-float range table — supplies clamp bounds for FP8/HiFloat8)
- OL-146 (CastTrait `SatMode::SAT` for quant — hardware-side clamp; this P-P is the reference-side equivalent)
- P-P94 (MERE/MARE thresholds — clamp affects which test cases hit the metric ceiling)

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P93，convert_patterns_to_okf.py）。confidence 未升格。 -->
