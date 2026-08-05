---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Efficiency bands for setting optimization targets"
description: "efficiency = theoretical min_time / actual; use the bands (<30% big opportunity, 30-60% moderate, >60% diminishing returns, >80% algorithmic-only) to decide how hard to keep optimizing."
confidence: single_run
original_id: ROOFLINE_MODEL.md#using-the-model-setting-targets
classified_by: llm-assisted
timestamp_inferred: true
tags: [roofline, optimization, efficiency, targets]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 决策

Compute efficiency against the theoretical floor, then read the band:
```
efficiency = theoretical min_time / actual time
```

- **< 30%** — significant optimization opportunity exists.
- **30-60%** — moderate improvements possible.
- **> 60%** — close to hardware limits, diminishing returns.
- **> 80%** — near optimal; only algorithmic changes are worth exploring.

Workflow before exploration: (1) calculate OI, (2) determine compute- vs memory-bound, (3) calculate theoretical min_time, (4) compare with actual → efficiency, then pick effort by the band above. Use the reuse-aware floor when a data-reuse reorder is available so the band reflects the real ceiling.
