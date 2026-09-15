---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Never permanently label a platform feature as \"broken\" without re-verification"
description: "paradigm: ascendc"
phenomenon: perf_regression
signal:
  - "when avoiding a platform feature (API, intrinsic, pattern) due to a past bug report"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-23
timestamp_inferred: true
tags: [ascendc, ol-23]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: trust_calibration
- **Loaded by**: Builder, all skills
- **Trigger**: when avoiding a platform feature (API, intrinsic, pattern) due to a past bug report
- **Lesson**: In Batch 5 (CANN 9.0.T501), TQue<VECIN,2> caused data corruption (OL-4). We labeled TQue as "broken" and used PipeBarrier<PIPE_ALL> for 6 days. During this time: (1) CANN was updated to 9.0.0 which fixed the bug; (2) The backward kernel successfully used TQue in the SAME session; (3) We still didn't try TQue on forward due to the "broken" label. When finally tested, TQue gave **1.6-2.3x speedup** over PipeBarrier. We left ~60% performance on the table for 6 days due to confirmation bias.
- **Anti-pattern**: "Feature X is broken" → avoid X forever, even when: environment changes (CANN version), other code paths succeed with X, or the expert's code uses X.
- **Correct pattern**: When a feature was previously broken:
  1. Check if the environment changed (CANN version, compiler version)
  2. Check if other code paths in the SAME codebase use the feature successfully
  3. If either is true, **re-test the feature** before assuming it's still broken
  4. Platform bugs are version-specific — always record the CANN version with the bug report
- **Evidence**: E13-P1 benchmark (2026-04-01): TQue 1.6-2.3x faster than PipeBarrier across all cases

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-23（category=trust_calibration，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
