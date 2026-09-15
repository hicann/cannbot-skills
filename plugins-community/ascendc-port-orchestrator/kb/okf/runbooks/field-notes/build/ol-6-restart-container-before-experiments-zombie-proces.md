---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Restart container before experiments (zombie processes)"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "before launching any benchmark or experiment on the A5 container"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-6
timestamp_inferred: true
tags: [ascendc, ol-6]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: environment
- **Loaded by**: QA
- **Trigger**: before launching any benchmark or experiment on the A5 container
- **Lesson**: After long experiment sessions, zombie processes accumulate from previous NPU kernel launches. 2280 zombie processes were found after one extended run. These cause training hangs, kernel launch failures, resource exhaustion, and unreproducible timing results. Always restart the container (`docker restart can_torch_cann_device_1`) before every experiment session.
- **Evidence**: CLAUDE.md (global) "CRITICAL: ALWAYS restart container before EVERY experiment", MEMORY.md

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-6（category=environment，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
