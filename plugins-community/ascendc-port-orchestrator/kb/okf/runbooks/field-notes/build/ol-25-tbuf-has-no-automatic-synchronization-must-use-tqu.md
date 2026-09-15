---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "TBuf has NO automatic synchronization — must use TQue or explicit sync"
description: "paradigm: ascendc"
phenomenon: build_failure
signal:
  - "mixing TBuf with TQue in the same kernel, or using TBuf for accumulators alongside DMA operations"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-25
timestamp_inferred: true
tags: [ascendc, ol-25]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

```yaml
applies_to:
  paradigm: ascendc
```
- **Category**: platform_bug (actually: design feature)
- **Loaded by**: Builder, Optimizer
- **Trigger**: mixing TBuf with TQue in the same kernel, or using TBuf for accumulators alongside DMA operations
- **Lesson**: E14 TQue refactoring of backward sorted kernel FAILED (max_diff=0.76) because accum was in TBuf while input was in TQue. Root cause: TBuf.Get() has **no EnQue/DeQue → no hardware set/wait signals**. When VEC writes to TBuf accum and MTE2 writes to TQue input concurrently, there's no sync → UB bus contention → data corruption.
  Fix: move accum to TQue<VECOUT> (same pattern as working forward PingPong). Official docs confirm: "TBuf申请的内存空间只能参与计算，无法执行队列的入队出队操作" and "EnQue调用会发射同步指令set" — TBuf simply has no sync mechanism.
- **Rule**: When using TQue for DMA, ALL VEC-accessed buffers must EITHER be TQue-managed OR have explicit SyncFunc/SetFlag/WaitFlag. TBuf + TQue without explicit sync = data corruption.
- **Evidence**: E14 revert (2026-04-06), CANN 9.0.0-beta.2 official docs "编程模型设计原理", ASCENDC_LANGUAGE_REFERENCE.md

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-25（category=platform_bug (actually: design feature)，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
