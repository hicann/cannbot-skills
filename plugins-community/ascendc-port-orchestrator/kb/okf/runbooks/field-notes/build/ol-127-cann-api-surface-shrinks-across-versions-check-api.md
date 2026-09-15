---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "CANN API surface shrinks across versions — check API availability before using A5-KB patterns on older CANN [V220, CANN 9.0.0]"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-127
timestamp_inferred: true
tags: [ascendc, ol-127]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

**Principle**: CANN 9.0.0 on V220 has a smaller API surface than CANN 8.x on A5. Several APIs commonly used in A5 KB patterns and code templates do not exist on V220. Workers porting from A5 patterns MUST verify each non-trivial API against the V220 header set before committing to it in Phase B.

**Concrete anchor — APIs absent from V220 CANN 9.0.0**:

| Absent API | Replacement |
|---|---|
| `DataCopyPad(dst, src, DataCopyParams)` + `DataCopyParams` struct | `DataCopy(dst, src, count)` — simple 3-arg form only |
| `PadNone` / `padNone` enum | Does not exist. Align buffers to datablock size (fp32:8, fp16:16) |
| `aclrtLaunchKernelWithHostArgs` (bisheng auto-gen host stub) | Manual `extern "C" uint32_t aclrtlaunch_<name>(uint32_t blockDim, void* stream, ...)` stubs |
| `GM_ADDR` type alias | `__gm__ uint8_t*` (kernel entry), `void*` (host stub) |
| `Divs` (tensor / scalar division) | Pre-compute reciprocal in pybind, use `Muls` in kernel |

**Worker build checklist for V220**: before completing Phase B, grep kernel code for `DataCopyPad`, `DataCopyParams`, `PadNone`, `GM_ADDR`, `aclrtLaunchKernelWithHostArgs`, `Divs`.

**Evidence**: op#31 IOU DS cold-start kw-1 (2026-05-01): 12 build iterations stuck on DataCopyPad. op#17 AdamW (2026-05-02): Divs compile error.

**Other instances (predicted)**: any A5→V220 port using strided DataCopy, bisheng auto-gen host stubs, pad-mode control, or scalar division via Divs.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-127（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
