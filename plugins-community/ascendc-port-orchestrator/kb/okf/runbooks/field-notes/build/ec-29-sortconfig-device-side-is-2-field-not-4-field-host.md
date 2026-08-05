---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "`SortConfig` device-side is 2-field, not 4-field (host tiling header mismatches)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Kernel compile error excess elements in struct initializer on line declaring constexpr SortConfig SORT_CFG = {...}."
confidence: single_run
original_id: EC-29
timestamp_inferred: true
tags: [sortconfig, sort, ascendc, ec-29]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Error pattern**: Kernel compile error `excess elements in struct initializer` on line declaring `constexpr SortConfig SORT_CFG = {...}`.
- **Root cause**: The hardware `Sort` API has **two** `SortConfig` definitions:
  - **Host-side tiling header** `sort_tiling_intf.h`: 4 fields `{type, isDescend, hasSrcIndex, hasDstIndex}`.
  - **Device-side impl** `sort_impl.h`: 2 fields `{type, isDescend}` — source/destination index options are fixed by overload choice, not config field.
- **Fix**: For device-side (the kernel), use 2-field initializer only: `constexpr SortConfig CFG = {SortType::RADIX_SORT, true};`. Use the simpler Sort overload `Sort<T, isReuse, cfg>(dst, dstIdx, src, tmp, count)` which auto-assigns default indices.
- **Detection**: First-line-of-kernel compile error, no other errors preceding.
- **Evidence**: 9_TopKTopP V2 iter 2 Phase C (2026-04-17). 4-field initializer copied from CANN host-side sample → device compile fails.

<!-- 迁移自 porter kb/target/ascendc/（EC-29，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
