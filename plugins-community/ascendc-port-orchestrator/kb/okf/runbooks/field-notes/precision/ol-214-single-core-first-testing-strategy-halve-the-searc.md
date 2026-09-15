---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Single-core-first testing strategy — halve the search space before multi-core debugging [V351, ALL_MODES, testing-methodology]"
description: "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-214
timestamp_inferred: true
tags: [ascendc, ol-214]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend950PR; cann=9.0.0; bisheng=n/a; op_class=all`
`verified_on: soc=Ascend950PR; cann=9.0.0 (LightningIndexerGrad P133, 2026-06-05)`

### Principle

When a kernel precision/hang bug appears, the FIRST diagnostic split should be **1-core vs multi-core**:

- **1-core PASS, multi-core FAIL** → bug is in synchronization/concurrency (SyncAll, events, atomic contention)
- **1-core FAIL** → bug is in algorithm/UB layout/alignment/compute logic

This simple split halves the search space immediately. It costs ~30 seconds (re-run with `usedCoreNum=1`) and can save hours of debugging the wrong hypothesis.

### Decision rule (MANDATORY for worker Phase D)

1. First precision test: run with **usedCoreNum=1** on 3 diverse configs (small/medium/large).
2. If 1-core PASS → proceed to multi-core sweep.
3. If 1-core FAIL → fix the single-core bug FIRST. Do NOT spawn multi-core until 1-core is clean.
4. Multi-core hang → immediately test with usedCoreNum=1 to confirm it's a sync issue.

### Anti-pattern

LightningIndexerGrad P128-P132: spent 4 iterations debugging multi-core hangs while never isolating to 1-core. The SyncAll root cause would have been obvious if we'd confirmed "1-core works, multi-core hangs → check SyncAll". Instead we chased event lifecycle (P131), UB overflow (P132), and zero-work core skip (P132 Fix 2) before finding the real cause at P133.

### Integration

Worker brief (kw_brief.py) Phase D now mandates: "Run precision with usedCoreNum=1 first. Only proceed to multi-core after 1-core PASS."

### Cross-references

- OL-213 (SyncAll vs PipeBarrier — the root cause this strategy catches fastest)
- OL-216 (diagnostic matrix — symptom→root cause for multi-core hangs)

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-214（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
