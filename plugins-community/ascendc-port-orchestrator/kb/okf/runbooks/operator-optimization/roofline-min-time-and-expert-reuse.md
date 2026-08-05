---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Bound wall time by bytes/bandwidth; cut reads with expert-major reuse"
description: "For a memory-bound kernel min_time = total_bytes / bandwidth; reordering to expert-major so each expert weight is loaded once collapses the dominant read term (~40 MB to ~8.2 MB in the SG example)."
confidence: single_run
original_id: ROOFLINE_MODEL.md#theoretical-performance-bounds-sg-forward
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, min-time, data-reuse, loop-reorder]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

**Theoretical floor** for a memory-bound kernel:
```
min_time    = total_bytes / bandwidth
total_bytes = T * K * hdim * sizeof(T)   # expert reads (dominant term)
            + T * hdim * sizeof(T)        # output writes
```

**Baseline example** — prod_a (T=8192, K=4, H=256, fp32):
```
read  = 8192 * 4 * 256 * 4 = 32 MB
write = 8192 *     256 * 4 =  8 MB
total = 40 MB  ->  40 MB / 1.5 TB/s = 0.027 ms
```

**Optimization — expert-major loop reorder (data reuse):** load each of the 64 experts once instead of once per token that routes to it. The dominant read term collapses:
```
read  = 64 * 256 * 4         = 64 KB (experts)
      + 8192 * 4 * 4          = 128 KB (indices)
total ≈ 8.2 MB (output + indices + expert data)
      ->  8.2 MB / 1.5 TB/s = 0.005 ms
```

The reorder cuts total bytes ~5x (0.027 ms → 0.005 ms floor). Use the two floors to size the payoff before implementing, and to set the efficiency target against the reuse-aware floor rather than the naive one.
