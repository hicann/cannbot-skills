---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "VECIN-only pipeline cannot do GM→UB→GM pass-through"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only"
confidence: single_run
original_id: EC-21
timestamp_inferred: true
tags: [ascendc, ec-21]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Data corruption or sync hang when doing DataCopy(UB←GM) then DataCopy(GM←UB) through VECIN queue only
- **Root cause**: VECIN syncs MTE2→VEC, but MTE3 store needs VEC→MTE3 sync. Without a VEC op and VECOUT queue, the pipeline has a sync gap.
- **Fix**: Split-queue pattern: VECIN for load + VECOUT for store + VEC identity op (Adds 0.0f) between them
- **Evidence**: Pad V2, Cat, Split all use this pattern (P-CAT-1)

<!-- 迁移自 porter kb/target/ascendc/（EC-21，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
