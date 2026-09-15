---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "TPipe must be a class member — local destructor frees TQue buffers before async MTE3 writes complete"
description: "applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all"
phenomenon: precision_issue
signal:
  - "applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-179
timestamp_inferred: true
tags: [ascendc, ol-179]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382; cann=9.0.0; op_class=all`
`verified_on: soc=Ascend910_9382 (V220); cann=9.0.0`

**Principle**: TPipe manages TQue buffer allocation. When declared as local variable, its destructor runs at scope exit, potentially freeing TQue buffers before MTE3 finishes writing to GM. When TPipe is a class member, it persists for the kernel object's lifetime.

**Evidence**: 12_Permute a3-ds kw-3 (2026-05-21). No precision failures attributable to TPipe lifecycle when member.

**Cross-ref**: OL-94 (TQue vs TBuf sync decision table); OL-128 (V220 TBuf→MTE3 stale data).

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-179（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
