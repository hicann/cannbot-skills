---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Expert feedback must result in code changes, never just documentation"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "when processing expert review feedback"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-2
timestamp_inferred: true
tags: [ascendc, ol-2]
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
- **Trigger**: when processing expert review feedback
- **Lesson**: Marking expert-reported issues as "already safe" in documentation -- without modifying, compiling, and testing the code -- is forbidden. The int64 problem was annotated "confirmed safe" in docs 3 times while the truncating `static_cast<int>()` calls remained in the source. The rule is: modify code + compile + run precision tests, or the issue stays open.
- **Evidence**: CLAUDE.md lines 34-37, EXPERT_FEEDBACK.md E8-3 round-by-round history

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-2（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
