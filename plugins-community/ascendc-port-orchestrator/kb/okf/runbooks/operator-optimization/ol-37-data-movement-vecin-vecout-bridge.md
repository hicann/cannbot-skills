---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Data-movement ops — use the VECIN→VECOUT bridge pattern"
description: "Pure data-movement kernels need a VEC op between VECIN and VECOUT for pipeline sync; use Adds(dst,src,0.0f) as a no-op bridge (Cast->Adds->Cast for fp16/bf16). Direct UB-to-UB DataCopy is prohibited (PB-9)."
confidence: single_run
original_id: OL-37
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-37, data-movement, vecin-vecout, adds-bridge, ascendc]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**场景 / Trigger**: implementing cat, split, pad, repeat, permute, or any pure-copy op.

**Pattern (P-CAT-1)**: Pure data-movement kernels need a VEC op between VECIN (GM→UB) and VECOUT (UB→GM) to maintain pipeline sync. Use `Adds(dst, src, 0.0f, count)` as a no-op bridge. For fp16/bf16, bridge through fp32 via `Cast → Adds → Cast` (lossless roundtrip).

**Constraint / why**: Direct UB-to-UB `DataCopy` is prohibited (PB-9), so a VEC bridge is required rather than a straight copy.

**Evidence**: Cat V2 kernel (2026-04-09), P-CAT-1 pattern. See OL-49 for the more efficient TQueBind alternative to the Adds bridge.
