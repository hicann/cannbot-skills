---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Compute Operational Intensity (OI) to classify compute- vs memory-bound"
description: "OI = FLOPs / Bytes_transferred (FLOP/byte); compare against the per-unit ridge to decide whether a kernel is compute-bound or memory-bound before optimizing."
confidence: single_run
original_id: ROOFLINE_MODEL.md#operational-intensity-calculation
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, operational-intensity, memory-bound]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Before optimizing a kernel, compute its Operational Intensity and compare to the ridge point:

```
OI = FLOPs / Bytes_transferred      (FLOP/byte)
```

If `OI < ridge` the kernel is memory-bound; if `OI > ridge` it is compute-bound (ridge is per-unit — see the ridge-point card).

**Worked example — SG Forward** (per token, top_k experts):
```
FLOPs = 2 * top_k * hdim            (multiply + add per element per expert)
Bytes = top_k * hdim * sizeof(T)    (load experts)
      + hdim * sizeof(T)            (store output)
OI    = 2 * top_k / ((top_k + 1) * sizeof(T))

fp32, K=4: OI = 8 / (5*4) = 0.4 FLOP/byte  -> memory-bound
fp16, K=4: OI = 8 / (5*2) = 0.8 FLOP/byte  -> memory-bound
```

SG-class kernels land at OI < 1, i.e. deeply memory-bound — the classification directly steers which optimizations are worth pursuing.
