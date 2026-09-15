---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "README-driven doc updates"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "when creating or modifying any documentation file"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-10
timestamp_inferred: true
tags: [ascendc, ol-10]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when creating or modifying any documentation file
- **Lesson**: Every documentation change must use README.md as the index and verify that all related documents are updated in sync. New files must be added to the README directory tree and document hierarchy. Batch results go into OPTIMIZATION_PLAN.md; README "current status" keeps only the final snapshot. This rule exists because Batch 9 changes were recorded in REPORT.md and EXPERT_FEEDBACK.md but README.md and OPTIMIZATION_PLAN.md were forgotten, leaving the project index stale.
- **Evidence**: CLAUDE.md lines 39-44, README.md directory tree

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-10（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
