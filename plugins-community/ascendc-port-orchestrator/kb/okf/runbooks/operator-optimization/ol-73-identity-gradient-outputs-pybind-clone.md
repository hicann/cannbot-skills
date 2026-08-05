---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Identity-gradient outputs — use pybind clone() instead of a kernel"
description: "Backward ops with an identity-mapped output (grad_X = grad_output) should return grad_output.clone() in pybind11.cpp instead of launching a kernel — saves ~10us, avoids bf16 UB-copy risk, less code."
confidence: single_run
original_id: OL-73
classified_by: llm-assisted
timestamp_inferred: true
tags: [kernel-design, optimization, ol-73, backward, identity-gradient, pybind]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

When a backward op produces an output that is an identity mapping of an input (e.g. `grad_residual = grad_output`), do NOT write an AscendC kernel for it. Even a bare `DataCopy(GM→GM)` still requires a kernel launch.

Instead, return the tensor directly in `pybind11.cpp`:

```cpp
auto grad_residual = grad_output.clone();
```

`clone()` on NPU is an async device-to-device copy that is cheaper than a kernel launch.

**Why prefer clone() over a kernel:**
1. Saves one kernel launch (~10us).
2. Avoids the bf16 UB-copy hazards of PB-4 / PB-9 (a copy kernel would round-trip bf16 through UB).
3. Less code — one line instead of a kernel + host launch stub.

**When to apply:** any backward op where one or more outputs equal an input tensor unchanged (identity gradient). Handle those in pybind; reserve kernels for the outputs that actually require compute.

**Evidence:** 29_TanhGatedResidualAddBackward has `grad_residual = grad_output`; pybind uses `go.clone()` instead of a kernel. Worker tool count ~10 fewer than "write kernels for all three outputs". E3 level.
