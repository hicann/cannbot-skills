---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Every code change MUST be followed by BOTH precision AND performance verification"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "after any kernel code modification"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-18
timestamp_inferred: true
tags: [ascendc, ol-18]
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
- **Trigger**: after any kernel code modification
- **Lesson**: CLAUDE.md said "every code update must verify precision" but did not mention performance. int64 fixes passed precision (61/61 PASS) but caused 6% performance regression that went undetected for 2 batches. The rule must be: verify precision AND performance after every change. A commit message claiming "no regression" without benchmark data is a lie.
- **Evidence**: E9-1 investigation, REPORT.md data was stale for 2 batches

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-18（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
