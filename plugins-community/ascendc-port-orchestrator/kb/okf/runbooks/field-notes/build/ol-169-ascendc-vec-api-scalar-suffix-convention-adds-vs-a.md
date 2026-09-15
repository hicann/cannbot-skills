---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "AscendC VEC API scalar-suffix convention (`Adds` vs `Add`, `Muls` vs `Mul`, ...)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "writing any VEC arithmetic — Add, Mul, Sub, Div, Max, Min, etc."
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-169
timestamp_inferred: true
tags: [adds, add, muls, mul, sub, div, max, ascendc, ol-169]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: API_glossary
- **Loaded by**: All workers (Phase A KB load), aog-kernel-optimizer
- **Trigger**: writing any VEC arithmetic — `Add`, `Mul`, `Sub`, `Div`, `Max`, `Min`, etc.
- **Lesson**: AscendC VEC ops follow a strict naming convention:
  - **No `s` suffix** → vector ⊕ vector: `Add(dst, src1, src2, count)`. Both operands are LocalTensors of the same length.
  - **`s` suffix** → vector ⊕ scalar broadcast: `Adds(dst, src, scalar, count)`. The scalar is a single value (int/float, NOT a tensor) broadcast to all elements.
  - **Same convention for**: `Muls/Mul`, `Subs/Sub`, `Divs/Div`, `Mins/Min`, `Maxs/Max`, `Adds/Add`.
  - **Note `Cast` does NOT follow this** — `Cast(dst, src, mode, count)` is the only signature; the dtype-conversion is the operation.
- **Common mistake**: writing `Add(dst, src, 1, count)` when you mean `Adds(dst, src, 1, count)`. Compiler may accept but result is undefined or compile error depending on overload resolution.
- **Quick rule**: count of LocalTensor args = arity. `Adds` has 1 LocalTensor (the src), `Add` has 2.
- **Evidence**: CANN advance_step_spec.h uses `Adds(buf, buf, accepted_num_i, tokenEachReqs)` and `Add(buf, buf, slotMappingLocal, count)` in the same Compute() function; mixing them up at line 178 vs 250 would silently break index arithmetic.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-169（category=api_glossary，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
