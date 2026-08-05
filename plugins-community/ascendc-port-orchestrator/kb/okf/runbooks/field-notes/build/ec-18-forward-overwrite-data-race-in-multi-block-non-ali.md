---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Forward-overwrite data race in multi-block non-aligned DMA"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail)."
confidence: single_run
original_id: EC-18
timestamp_inferred: true
tags: [ascendc, ec-18]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures in multi-block kernels where non-aligned DataCopy uses forward-overwrite technique (write ALIGN elements, let next iteration overwrite tail). When multiple blocks process different rows in parallel, the overwrites from different blocks race.
- **Root cause**: Block K-1's tail overwrite extends into Block K's write region. Without ordering, Block K may read before K-1's overwrite completes, or K's write may be overwritten by K-1's stale data.
- **Fix**: Two approaches:
  1. **Per-row overlap** (chunk >= ALIGN): re-copy last ALIGN elements from `chunk - ALIGN` offset. No cross-row overwrite. Safe for multi-block.
  2. **nblk=1 + padded alloc** (chunk < ALIGN): serialize to one block. Over-allocate output with ALIGN padding, narrow() after kernel.
- **Evidence**: Split V3 — V2 forward-overwrite caused 12 new failures, fixed with per-row overlap (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-18，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
