---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Research official docs + CANN source before optimization attempts"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "before any non-trivial optimization or architecture change"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-26
timestamp_inferred: true
tags: [ascendc, ol-26]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process
- **Loaded by**: Builder, Optimizer, Researcher
- **Trigger**: before any non-trivial optimization or architecture change
- **Lesson**: E14 first attempt was a guess (TQue input + TBuf accum). If we had first verified TBuf sync semantics from official docs, we would have known it can't work. The correct pattern (accum in TQue<VECOUT>) was already in the forward code and confirmed by CANN MoE source.
  Research workflow:
  1. Check official docs (dev-browser for JS-rendered hiascend.com)
  2. Check CANN source code (~/workspace/cann/, git fetch first)
  3. Record findings as patterns/OL entries for future use
  4. THEN implement
- **Evidence**: E14 session (2026-04-06 → 2026-04-07), ASCENDC_LANGUAGE_REFERENCE.md creation

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-26（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
