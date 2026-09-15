---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "MERE/MARE aux precision standard (Mean/Max Relative Error per dtype Threshold) — ecosystem-blessed metric"
description: "applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all; phase=precision-verify verified_on: soc=Ascend950PR; cann=9.0.0 source: PR 103 references/precision-testing/OPS_PRECI"
confidence: single_run
original_id: P-P94
timestamp_inferred: true
tags: [patterns-index, optimization, golden, aux_metrics, p-p94, ascendc]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 优化点

`applies_to: soc=Ascend950PR,Ascend910_V220; cann=9.0.0; bisheng=all; op_class=all; phase=precision-verify`
`verified_on: soc=Ascend950PR; cann=9.0.0`
`source: PR 103 references/precision-testing/OPS_PRECISION_STANDARDS.md (生态算子开源精度标准)`

**Pattern**: When verifying an A5 kernel's precision against a CPU reference, compute and report **MERE** (Mean Relative Error) and **MARE** (Max Relative Error) as **aux metrics** alongside our existing T1/T2 tier classification. MERE/MARE is the official Ascend ecosystem standard for open-source-grade compute ops — having both metrics enables cross-comparison and lets us calibrate our T1/T2 tiers against the wider ecosystem.

**Formulas**:

```
MERE = avg( abs(actual - golden) / (abs(golden) + 1e-7) )
MARE = max( abs(actual - golden) / (abs(golden) + 1e-7) )
```

The `+1e-7` avoids div-by-zero when `golden` has zero / near-zero elements.

**PASS criterion**: `MERE < Threshold AND MARE < 10 × Threshold`.

**Threshold table by dtype**:

| dtype | Threshold | MERE limit | MARE limit (10×) |
|---|---|---|---|
| float16 | 2⁻¹⁰ ≈ 9.77e-4 | 9.77e-4 | 9.77e-3 |
| bfloat16 | 2⁻⁷ ≈ 7.81e-3 | 7.81e-3 | 7.81e-2 |
| float32 | 2⁻¹³ ≈ 1.22e-4 | 1.22e-4 | 1.22e-3 |
| HiFloat32 | 2⁻¹¹ ≈ 4.88e-4 | 4.88e-4 | 4.88e-3 |
| FP8 E4M3 | 2⁻³ = 0.125 | 0.125 | 1.25 |
| FP8 E5M2 | 2⁻² = 0.25 | 0.25 | 2.5 |
| INT8 | 0 (exact) | exact | exact |

**Relationship to our existing T1/T2 tiers** (from `ASCEND_OP_PRECISION_STANDARD_v2.1.md`):

| Tier | Our definition | MERE/MARE rough analog |
|---|---|---|
| T1 (bit-exact) | `max_abs_err == 0.0` | MARE strictly = 0 |
| T2 (compute-grade tolerance) | `atol=1e-3, rtol=1e-3` | Roughly MERE < 1e-3 for fp32 (≈ Threshold of `1.22e-4` × ~10) |

T1/T2 is **stricter** than MERE/MARE for most dtypes (we measure absolute error against an atol+rtol envelope; they measure relative error against a div-by-near-zero-protected denominator). MERE/MARE is **more permissive on small-value cases** (the +1e-7 epsilon prevents division blow-up).

**Why add this as AUX (not replacement)**:
- T1/T2 keeps producing the binary verdict (PASS / FAIL) for finalize decisions — keep it.
- MERE/MARE numerical values get exported alongside, enabling:
  - **Cross-comparison with ecosystem (Ascend Modelzoo, ops-nn upstream)** — they report MERE/MARE
  - **Calibration**: if T1/T2 marks PARTIAL but MERE/MARE shows clean PASS → indicates our tolerance is over-strict
  - **Detection of clamp-mismatch (per P-P93)** — high MARE concentrated on out-of-range cases is a clamp signature

**Implementation** (proposed `verification.json` schema addition):

```json
{
  "precision": {
    "status": "PASS",                  // existing T1/T2 verdict
    "tier": "T1",
    "max_abs_err": 0.0,
    "aux_metrics": {                   // NEW — MERE/MARE block
      "standard": "ecosystem_MERE_MARE_v1",
      "per_case": [
        {
          "case_id": 1,
          "dtype": "fp32",
          "MERE": 1.4e-5,
          "MARE": 8.2e-5,
          "threshold": 1.22e-4,
          "MERE_pass": true,
          "MARE_pass": true
        }
      ],
      "summary": {
        "n_pass": 8,
        "n_total": 8,
        "median_MERE": 1.4e-5,
        "p99_MARE": 8.2e-5
      }
    }
  }
}
```

**Helper code** (Python, to land in `aog-a3-author` Path A template):

```python
import torch

def compute_mere_mare(actual: torch.Tensor, golden: torch.Tensor) -> tuple[float, float]:
    """MERE/MARE per Ascend ecosystem precision standard."""
    rel = (actual.float() - golden.float()).abs() / (golden.float().abs() + 1e-7)
    return float(rel.mean().item()), float(rel.max().item())

DTYPE_THRESHOLD = {
    "torch.float32": 2 ** -13,
    "torch.float16": 2 ** -10,
    "torch.bfloat16": 2 ** -7,
    # narrow floats:
    "torch.float8_e4m3fn": 2 ** -3,
    "torch.float8_e5m2": 2 ** -2,
}

def passes_mere_mare(actual, golden, dtype_str: str) -> dict:
    threshold = DTYPE_THRESHOLD.get(dtype_str, None)
    if threshold is None:
        return {"verdict": "SKIP_NO_THRESHOLD"}
    mere, mare = compute_mere_mare(actual, golden)
    return {
        "MERE": mere, "MARE": mare,
        "threshold": threshold,
        "MERE_pass": mere < threshold,
        "MARE_pass": mare < 10 * threshold,
        "verdict": "PASS" if (mere < threshold and mare < 10 * threshold) else "FAIL",
    }
```

**Evidence**:
- PR 103 OPS_PRECISION_STANDARDS.md codifies as the official ecosystem standard for compute-op precision
- We have ASCEND_OP_PRECISION_STANDARD_v2.1.md (vendor v2.1 with MARE/MERE/RMSE) — MERE/MARE here is a subset of v2.1, formulas match

**Other instances (predicted)**: every op verification — adding aux fields is no-cost. Especially valuable for narrow-float ops where T1 (bit-exact) is unrealistic but MARE < `10 × 2⁻³` is meaningful.

**Cross-reference**:
- P-P93 (clamp rule — clamp mismatches show as high MARE concentration)
- `ASCEND_OP_PRECISION_STANDARD_v2.1.md` (vendor v2.1 — superset of MERE/MARE)
- Workflow Batch 4: `verification.json` schema gets `aux_metrics` field with MERE/MARE per-case

<!-- 迁移自 porter kb/target/ascendc/patterns/PATTERN_INDEX.md 全文小节（P-P94，convert_patterns_to_okf.py）。confidence 未升格。 -->
