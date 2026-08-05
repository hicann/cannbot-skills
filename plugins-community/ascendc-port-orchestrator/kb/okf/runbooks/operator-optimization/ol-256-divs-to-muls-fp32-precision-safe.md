---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Divs(vec,scalar) → Muls(vec,1.0f/scalar) is precision-safe in fp32 and 4-8× faster on da Vinci VEC"
description: "In fp32 a*(1/b) differs from a/b by ≤1 ULP, so replacing Divs with a precomputed-reciprocal Muls is precision-safe and 4-8× faster; keep Divs for fp16/bf16 where 1/b loses mantissa."
confidence: single_run
original_id: OL-256
classified_by: llm-assisted
timestamp_inferred: true
tags: [vec-throughput, optimization, ol-256, divs-muls, fp32-precision]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型
**When to use:** any op with a scalar-divide normalizer (RMSNorm / LayerNorm / GroupNorm / instance-norm / fused-norm-quant) whose kernel-internal compute is fp32.

**Principle.** da Vinci VEC `Divs(vec, scalar)` throughput ≈ 1 elem/cycle; `Muls(vec, scalar)` ≈ 8-16 elem/cycle. Precomputing `inv = 1.0f / scalar` (ONE scalar divide, not pipelined through VEC) and replacing `Divs(vec, scalar)` with `Muls(vec, inv)` is a **4-8× speedup on the normalization step with zero precision cost in fp32**.

**Why it's precision-safe in fp32.** In IEEE-754 fp32 both `a/b` and `a*(1/b)` are correctly rounded — the intermediate `1/b` rounds once, each `a_i * (1/b)` rounds once, and each `a_i / b` also rounds once; the two sequences differ by at most 1 ULP. This is **fundamentally different from fp16**, where `1/b` loses 13+ bits of mantissa and `a*(1/b) ≠ a/b` is a real precision hazard.

**Decision rule:**
- **fp32 internal compute: MUST use `Muls(x, 1.0f/scalar)`.**
- **fp16/bf16 internal compute: keep `Divs`.**
- When in doubt: the first iteration may use `Divs` for safety; the optimizer converts to `Muls` behind a precision gate.

### Concrete anchor
```cpp
// BEFORE (V1, geo_mean 0.47×):
float rms = sqrt(meanSq + eps);
Divs(work, work, rms, N);
PipeBarrier<PIPE_V>();

// AFTER (V2, geo_mean 0.91×):
float rms = sqrt(meanSq + eps);
float invRms = 1.0f / rms;
Muls(work, work, invRms, N);
PipeBarrier<PIPE_V>();
```

### Evidence
add_rms_norm_quant V1→V2 A/B (Ascend950PR_957b, CANN 9.0.0, NPU 0, 196 cases, 2026-06-24):
- V1 (Divs): geo_mean 0.47×, arith_mean 0.50×
- V2 (Muls): geo_mean 0.91×, arith_mean 1.04×
- Precision: V2 196/196 PASS (better than V1's 194/196); x_out max_diff ≤ fp32 1 ULP vs CPU fp64 truth.
- The Divs→Muls transform contributed ~45% of the 1.94× speedup.

### Related
- ALWAYS_LOADED_RULES §5 fp32 carve-out (normative permission), OL-257 (VEC cost model — the "why it's faster" half), OL-245 (regbase — same "prefer the efficient API" principle).
