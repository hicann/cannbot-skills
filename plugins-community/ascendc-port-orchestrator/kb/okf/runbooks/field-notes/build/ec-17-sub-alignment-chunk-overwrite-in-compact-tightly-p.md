---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Sub-alignment chunk overwrite in compact (tightly-packed) output"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)"
confidence: single_run
original_id: EC-17
timestamp_inferred: true
tags: [ascendc, ec-17]
created_at: 2026-07-09T16:00:00Z
updated_at: 2026-07-09T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```

- **Error pattern**: Precision failures when chunk_size < DataCopy alignment AND output elements are tightly packed (no gaps between chunks from different outer iterations)
- **Root cause**: DataCopy writes aligned count of elements. When chunk < align, excess elements overwrite the next chunk's data. In Cat's output (strided with gaps), overlapping tail write works. In Split's output (compact), there are no gaps — adjacent chunks are immediately adjacent.
- **Fix (host-side)**: Detect `chunk < align && outer > 1`. Use `nblk=1` (serial execution — overwrites self-correct within one block) + allocate padded output + narrow to exact size.
- **Applicability**: Any kernel writing to compact output with non-aligned chunk boundaries
- **Evidence**: Split V1 failed 4/57 cases, fixed in V2 (2026-04-09)

<!-- 迁移自 porter kb/target/ascendc/（EC-17，convert_family_to_okf.py，M1，整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
