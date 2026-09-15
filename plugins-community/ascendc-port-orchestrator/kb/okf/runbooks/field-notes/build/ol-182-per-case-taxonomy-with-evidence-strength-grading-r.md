---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Per-case taxonomy with evidence_strength grading required before claiming fix coverage"
description: "applies_to: soc=all; cann=all; op_class=all; workflow=verification"
phenomenon: build_failure
signal:
  - "applies_to: soc=all; cann=all; op_class=all; workflow=verification"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-182
timestamp_inferred: true
tags: [ascendc, ol-182]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=all; cann=all; op_class=all; workflow=verification`
`verified_on: multiple ops (26_AvgPool3d, 27_MaxPool3d)`

**Principle**: Derived from FA V220 decision discussion (2026-05-22). Claims like "X of Y cases solvable" without per-case evidence and strength grading (HIGH/MEDIUM/LOW) are the same shape as "0.014× CANN is real ceiling" — confident framing without per-instance evidence. LOW-confidence entries must be classified `OTHER`, not the predicted-fix bucket.

**Application**: Before declaring ko-2 fixes for residual failures, produce `fixture_eval_err_taxonomy.json` with per-case root cause classification + evidence_strength. This prevents scope substitution.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-182（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
