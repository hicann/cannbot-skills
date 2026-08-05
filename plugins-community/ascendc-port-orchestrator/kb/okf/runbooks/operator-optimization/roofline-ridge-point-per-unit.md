---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Compute the ridge point per compute unit (VEC vs CUBE)"
description: "Ridge OI = Peak_FLOPS / Peak_BW, computed separately per unit; the CUBE ridge is ~6.7x the VEC ridge, so matmul/FA shapes that look compute-bound may still be memory-bound."
confidence: single_run
original_id: ROOFLINE_MODEL.md#ridge-point
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, ridge-point, cube, vec]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

The ridge point separates memory-bound from compute-bound. Compute it per unit, using that unit's peak against HBM bandwidth (1.5 TB/s):

```
Ridge OI = Peak_FLOPS / Peak_BW

VEC-bound ops (elementwise / reduction / softmax):
  fp32: 28  TFLOPS / 1.5 TB/s ≈ 18.7 FLOP/byte
  fp16: 56  TFLOPS / 1.5 TB/s ≈ 37.3 FLOP/byte

CUBE-bound ops (matmul / attention):
  fp16: 373 TFLOPS / 1.5 TB/s ≈ 248.7 FLOP/byte
  fp32: 24  TFLOPS / 1.5 TB/s ≈ 16.0  FLOP/byte
```

**Key trap:** the CUBE ridge (~248.7) is ~6.7x higher than the VEC ridge (~37.3). Many matmul/FlashAttention shapes that look "compute-bound" when checked against the VEC ridge are still well below the CUBE ridge — i.e. they are memory-bound on the cube unit. Always compare an op's OI against the ridge of the unit it actually runs on.
