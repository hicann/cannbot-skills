---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "GatherElements — compile-time axis specialization + axis merging"
description: "GatherElements uses a compile-time AXIS template param, magic-number division for flat→multi-dim index decomposition, and MergeAxis to fold adjacent non-axis dims (7D→3D), cutting UintDiv calls."
confidence: single_run
original_id: OL-62
classified_by: llm-assisted
timestamp_inferred: true
tags: [algorithm-selection, optimization, ol-62, gather-elements, axis-specialization, merge-axis]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Trigger**: generating GatherElements / gather-class ops. Loaded by Generator.

Three optimization techniques from CANN GatherElements:

1. **Compile-time axis specialization** — pass the gather axis as a `constexpr AXIS` template
   parameter so runtime branches in the inner loop are eliminated (the compiler specializes
   per axis).
2. **Magic-number division** — high-dim gather decomposes a flat index into a multi-dim index;
   the divisions are replaced by pre-computed magic-number division (magic multiplier + shift),
   turning `/` into a multiply + shift.
3. **MergeAxis** — adjacent non-axis dimensions with identical `x` and `index` shapes are
   merged, reducing the number of `UintDiv` calls. A 7D case can collapse to 3D.

Thread count scales with rank: dim1-3 = 2048, dim4-5 = 1024, dim6-7 = 512.

**Evidence**: CANN `gather_elements.h:66-300`, `gather_elements_tiling_arch35.h:110-111`.
E1 level (source analysis).
