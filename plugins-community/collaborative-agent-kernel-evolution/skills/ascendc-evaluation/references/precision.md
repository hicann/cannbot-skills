# Precision Framework Reference

## Table of Contents

- [Three-Way Comparison](#three-way-comparison)
- [Pass/Fail Judgment](#passfail-judgment)
- [L0 Ratio Acceptance Guidelines](#l0-ratio-acceptance-guidelines)
- [Per-Dtype Tolerances](#per-dtype-tolerances)
- [Float Diagnosis Patterns](#float-diagnosis-patterns)
- [Integer Diagnosis Patterns](#integer-diagnosis-patterns)
- [Precision JSON Format](#precision-json-format)
- [Backward Compatibility](#backward-compatibility)

## Three-Way Comparison

Compares custom kernel error against PyTorch's own error:

1. **Golden** (CPU fp64): Highest precision baseline
2. **Ref** (NPU, working dtype): PyTorch reference on NPU — captures PyTorch's own numerical error
3. **Ans** (NPU, working dtype): Custom kernel on NPU

The system computes errors for both `ref` and `ans` against the `golden` baseline, then forms ratios:

```
ratio = ans_error / max(ref_error, floor)
```

The `floor` prevents division-by-zero when PyTorch's error is extremely small.

## Pass/Fail Judgment

**Float types:**
```
passed = (
    NO new NaN/Inf introduced by custom kernel
    AND ans_vs_golden.mismatch_rate == 0.0   (torch.isclose with dtype-specific atol/rtol)
)
```

Ratios (ans_error / ref_error) are always computed and written to precision.json for reference, but do NOT determine pass/fail.

**Integer types:**
```
passed = (mismatch_rate == 0.0)    # exact match required
```

Integer comparison uses ae (absolute error) and mismatch_rate only — no ratio/ULP metrics.

## L0 Ratio Acceptance Guidelines

After precision test passes (mismatch_rate == 0), check ratios to decide acceptance level:

| Ratio    | Ship (accept) | NEEDS_REVIEW    | Iterate (optimize) |
|----------|---------------|-----------------|---------------------|
| max_re   | <= 10.0       | 10.0 ~ 20.0    | > 20.0              |
| mean_re  | <= 2.0        | 2.0 ~ 4.0      | > 4.0               |
| rmse     | <= 2.0        | 2.0 ~ 4.0      | > 4.0               |
| svec     | <= 2.0        | 2.0 ~ 4.0      | > 4.0               |

- All <= L0: **Ship** — precision acceptable
- Some in L0 ~ 2xL0: **NEEDS_REVIEW** — consider operator type and dtype
- Any > 2xL0: **Iterate** — kernel code needs modification

## Per-Dtype Tolerances

**Float types** — three-way comparison with full metrics:

| dtype | atol | rtol | ulp_tol | sv_th | sv_err |
|-------|------|------|---------|-------|--------|
| bfloat16 | 1e-2 | 1e-2 | 2 | 2^-8 | 2^-16 |
| float16 | 1e-3 | 1e-3 | 2 | 2^-11 | 2^-16 |
| float32 | 1e-5 | 1e-5 | 2 | 2^-14 | 2^-30 |

- `atol` / `rtol`: Absolute and relative tolerance for element-wise comparison
- `ulp_tol`: ULP (units in the last place) tolerance
- `sv_th`: Small-value threshold — values below this magnitude are treated as "small"
- `sv_err`: Small-value error threshold

**Integer/bool types** — exact match by default (atol=0, rtol=0):

`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `bool`

Integer types have no ratio/ULP metrics — `ratios` is `null` in output.

### Quantized Output Flag

The default `atol=0` requires bit-exact match, which is correct for most integer operators (bitwise, lookup, indexing). However, **quantization operators** (float→int conversion) have inherent ±1 LSB rounding error from different rounding implementations. Set `quantized_output: true` in `test_cases.py` to apply `atol=1` to all integer outputs:

```python
# test_cases.py — for quantization operators (e.g., DynamicQuant)
TEST_CASES = {
    "operator": "DynamicQuant",
    "operator_type": "unary",
    "quantized_output": True,  # ±1 LSB tolerance for integer outputs
    "cases": [...]
}
```

When `quantized_output` is `True`, integer outputs use `atol=1`; float outputs are unaffected.

**When to set `quantized_output: True`:**
- float→int quantization (DynamicQuant, StaticQuant, etc.)

**When NOT to set (keep default `False`):**
- Bitwise operations, lookup tables, indexing, sorting — must be exact

## Float Diagnosis Patterns

When a float test case fails, the system identifies the root cause:

| Pattern | Description | Likely Cause |
|---------|-------------|--------------|
| `nan_inf_introduced` | NaN/Inf in ans but not in golden/ref | Division by zero, log(0), exp overflow |
| `zero_output` | Output near-zero everywhere | Placeholder code, incomplete computation |
| `localized_outlier` | max_re high, mean_re ok | Boundary or special-value path issue |
| `systematic_drift` | mean_re exceeded | Algorithm error, dtype cast loss |
| `small_value_error` | svec exceeded | Small-value underflow |
| `uneven_distribution` | rmse exceeded, others ok | Input-dependent precision patterns |

## Integer Diagnosis Patterns

When an integer test case fails:

| Pattern | Description | Likely Cause |
|---------|-------------|--------------|
| `int_total_mismatch` | mismatch_rate > 90% | Algorithm wrong or output uninitialized |
| `int_overflow` | ae_max > 128 | Integer overflow or truncation during cast |
| `int_sign_flip` | Mean near zero, high std | Signed/unsigned mismatch or sign-extension error |
| `int_off_by_one` | ae_max == 1 | Floor/ceil rounding direction issue |
| `int_scattered_error` | Moderate mismatch, bounded ae | Partial tile or branch computation error |

## Precision JSON Format

Export with `--json-output`:

```json
{
    "schema_version": 2,
    "cases": [
        {
            "case_id": "case_0",
            "forward": {
                "output_0": {
                    "passed": true,
                    "ratios": {
                        "max_re": 1.23,
                        "mean_re": 0.95,
                        "rmse": 0.88,
                        "svec": 1.01
                    },
                    "diagnosis": null
                }
            }
        }
    ]
}
```

Key fields:
- `ratios`: Error ratios (ans_error / max(ref_error, floor)) — `null` for integer types
- `diagnosis`: Diagnosis dict if failed, `null` if passed

## Backward Compatibility

- `evaluate_correctness()` still returns `(bool, str)` — no downstream changes needed.
- If the golden model (CPU fp64) fails to compute, the system falls back to legacy two-way comparison.