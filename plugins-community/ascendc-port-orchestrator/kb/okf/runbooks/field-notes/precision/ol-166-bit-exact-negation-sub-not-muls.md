---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Bit-exact negation needs Sub(z, zero_buf, x); Muls(x, -1) diverges on signed zero"
description: "For y=-x, Sub(z, zero_buf, x) matches IEEE-754 negation used by CANN/PyTorch/A3, while Muls(x,-1) flips the sign of zero. They differ only on signed zero, present in every ±0.0 edge case."
phenomenon: precision_issue
signal:
  - "elementwise negation y=-x must bit-exactly match A3/CANN/PyTorch ground truth and the edge dataset includes signed zero (±0.0)"
confidence: single_run
original_id: OL-166
classified_by: llm-assisted
timestamp_inferred: true
tags: [port-a3-to-a5, negation, signed-zero, ol-166, ieee-754, foreach-neg]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 现象 / 触发

Applies to `soc=Ascend950PR, cann=9.0.0, op_class=elementwise_negation, dtype=fp32,fp16,bf16`.
Verified on `soc=Ascend950PR, op=foreach_neg`. Unverified on Ascend910_V220 (the divergence is
IEEE-754 semantics, not bisheng codegen, so it is expected to transfer, but no A3 standalone evidence
yet).

An elementwise negation `y = -x` (foreach_neg, standalone `Neg`, the negative leg of a linear
combination, or any sub-step contributing a sign-flipped term) that must bit-exactly match the
A3/CANN/PyTorch ground truth. The divergence surfaces on **signed zero** inputs — which are in the
standard "special values" case for port_a3_to_a5 ops (`±0, ±inf, NaN, big, tiny`).

## 根因 / 教训

`Sub(z, zero_buf, x)` matches the IEEE-754 negation semantics that CANN, PyTorch, and the A3
reference all use, while `Muls(z, x, -1)` is the IEEE-754 multiplication operator. They are equal on
every input EXCEPT signed zero:
- IEEE-754 add/sub (default RNE): `0 - (+0) = +0`, `0 - (-0) = +0`. Both signed-zero inputs collapse
  to `+0`.
- IEEE-754 mul: `(+0) × (-1) = -0`, `(-0) × (-1) = +0`. `Muls(x,-1)` flips the sign of zero.
- For normals, denormals, ±inf, NaN the two forms are bit-identical (sign-bit flip is exact in both
  pathways; NaN payload preservation is the same). The divergence is restricted to signed zero.

**Vendor anchor**: CANN's upstream `foreach_neg` kernel uses
`ForeachImplictOutputLevelZeroApi<T, U, Sub, 2, 1>` with `Init(..., 0)` — it materializes a
zero-init buffer and dispatches `Sub(zero, x)`. It does NOT use `Mul(x, -1)`. Aligning to CANN's
primitive choice yields bit-exact A3↔A5 ground-truth match on signed-zero inputs without any
tolerance relaxation.

**Code anchor**:
```cpp
// BAD — signed zero diverges (+0 → -0, bit pattern 0x00000000 → 0x80000000)
Muls(z, x, T(-1), count);

// GOOD — bit-exact across the full IEEE-754 domain
Duplicate(zero_buf, T(0), TILE);   // in Init(): allocate + zero-fill once
Sub(z, zero_buf, x, count);        // in Compute(): +0 → +0, -0 → +0
```
Cost: one extra TBuf of TILE elements, zero-initialized once at Init time — UB delta 16–32 KB
depending on dtype; negligible on A5's 192 KB UB.

**Decision rule**:
- Bit-exact target (selected source-arch ground-truth match or equivalent declared contract per
  OL-97): MUST use `Sub(z, zero_buf, x)`, carrying an init-time zero TBuf.
- Tolerance target only (T2 numeric comparison, no bit-exact requirement): `Muls(x, -1)` is
  acceptable; signed zero rarely surfaces in tolerance metrics.
- Integer types (int8/16/32/64): no signed-zero concept — either form works; prefer `Sub` for
  uniformity so one kernel template handles all dtypes.
