---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "V220 TBuf→MTE3 DataCopy reads stale data even after VEC write — use TQue<VECOUT> for output [V220]"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "paradigm: ascendc"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-128
timestamp_inferred: true
tags: [datacopy, ascendc, ol-128]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

**Principle**: On V220 (CANN 9.0.0), when VEC ops write to a `TBuf<VECCALC>` and a subsequent `DataCopy` (MTE3) reads from the same or a different TBuf, the MTE3 read may return **stale data from a different buffer region** — even when `SetFlag<HardEvent::V_MTE3>/WaitFlag` fences are in place.

**Fix**: Use `TQue<QuePosition::VECOUT, depth=1>` for output instead of bare `TBuf<VECCALC>` + `DataCopy`. The TQue's `EnQue/DeQue` rotation provides hardware-managed V→MTE3 sync.

**Evidence**: op#31 IOU DS kw-3 (2026-05-01): 5 precision iterations, DataCopy always read wrong buffer. kw-4 applied TQue<VECOUT> → MTE3 coherence fixed.

**Other instances (predicted)**: any V220 kernel where TBuf is used for output staging before DataCopy to GM.

**Cross-ref**: PB-22 (MTE2 DataCopy 32-byte limit), OL-127 (CANN API surface gaps V220), OL-94 (TQue vs TBuf decision).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-128（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
