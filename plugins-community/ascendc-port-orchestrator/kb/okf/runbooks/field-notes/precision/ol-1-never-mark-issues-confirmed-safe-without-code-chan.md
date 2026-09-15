---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Never mark issues \"confirmed safe\" without code change"
description: "paradigm: ascendc"
phenomenon: precision_issue
signal:
  - "when reviewing generated code for correctness"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-1
timestamp_inferred: true
tags: [ascendc, ol-1]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Precision-audit (CPU-truth, 2026-04-29)**: ✅ VALIDATED-CPU — co-occurs with CANN-pass op (20_FusedRopeWithQkNormAndKvCacheUpdate). Do not downgrade. See `docs/data/kb_audit_cpu_truth_2026_04_29.md`.
- **Category**: process
- **Loaded by**: Builder, QA
- **Trigger**: when reviewing generated code for correctness
- **Lesson**: int64 truncation was flagged 5 times by experts. The first 3 times, CC marked "confirmed safe" without changing code. The truncation was real -- `static_cast<int>(int64_var)` overflows when `dim * index > INT32_MAX`. It took 5 rounds and 72 code changes across 5 files to finally fix all instances.
- **Evidence**: EXPERT_FEEDBACK.md E8-3 (timeline of 5 rounds), CLAUDE.md lines 27-37

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-1（category=process，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
