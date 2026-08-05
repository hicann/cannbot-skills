---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "RMSNorm needs no padding correction (unlike LayerNorm)"
description: "RMSNorm needs no padding correction — padded zeros don't affect sum(x^2)/N with N=hidden_orig — but LayerNorm/Mean must mask the padding region before reducing, since padded zeros pollute the mean."
confidence: single_run
original_id: OL-72
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm, optimization, ol-72, rmsnorm, layernorm, padding]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: implementing RMSNorm / LayerNorm / similar reduce-over-last-dim normalization where
`hidden_size` must be aligned to 8/16 elements. Loaded by Generator.

**RMSNorm**: padded zeros contribute 0 to `sum(x^2)`, so they do not affect the final
`mean = sum / N` — as long as you divide by `1/hidden_orig` (not `1/hidden_pad`). No correction
needed.

**LayerNorm / Mean**: padded zeros contribute 0 to `sum(x)`, but they still **pollute**
`mean = sum / N`. Using `N = hidden_pad` pulls the mean down; using `N = hidden_orig` computes
the mean correctly, but the sum still mistakenly includes the padding-region zeros (effectively
an unintended compression of the valid range).

**Cleanest solution for mean-style reduces**: mask out the padding region *before* reducing —
e.g. SIMD `Duplicate(invalid_region, NaN)` before writing 0 then `ReduceSum` skipping NaN, or
use `hidden_orig` and simply do not include padding elements.

**Evidence**: 18_FusedAddRmsnorm — `hidden_size_orig = 128` → `hidden_pad = 128` (already
aligned), no issue; if `hidden = 120` → `pad = 128`, `sum / 120` is directly correct because the
padding is 0. 10_LayerNorm earlier padding handling needs care. E3 level (measured).
