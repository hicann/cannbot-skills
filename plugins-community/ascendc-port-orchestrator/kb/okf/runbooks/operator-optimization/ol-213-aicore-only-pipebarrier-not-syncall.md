---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "AiCore-only pipelines must use PipeBarrier<PIPE_ALL>, not SyncAll — cross-core barriers cause multi-core hangs"
description: "In AiCore-only pipelines each core has independent data, so use intra-core PipeBarrier<PIPE_ALL>; SyncAll is a cross-core barrier that deadlocks when zero-work cores skip it."
confidence: single_run
original_id: OL-213
classified_by: llm-assisted
timestamp_inferred: true
tags: [sync-correctness, optimization, ol-213, syncall, pipebarrier, multi-core-hang]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.0.0 / all op classes. Verified on LightningIndexerGrad P133, 2026-06-05.

**Principle**: `SyncAll()` is a **cross-core barrier** — ALL scheduled cores must reach it before any proceeds. In AiCore-only pipelines (no MIX_AIC), each core operates on **independent data** (different s1Idx/n2Idx slices), so cross-core synchronization is unnecessary AND harmful:
1. **Zero-work cores** (`split.length == 0`) that skip the batch loop never reach SyncAll, so working cores wait forever (deadlock).
2. Even with all cores working, SyncAll adds unnecessary cross-core coupling — any core progressing at a different rate blocks all others, turning an embarrassingly-parallel workload into a lockstep one.

`PipeBarrier<PIPE_ALL>()` is the **correct** primitive: it synchronizes V/MTE1/MTE2/MTE3 pipes **within a single core**, ensuring MTE3 writes are visible before the next pipeline stage reads them — no cross-core dependency.

**Decision rule** for AiCore-only pipelines:
- **DEFAULT**: `PipeBarrier<PIPE_ALL>()` — intra-core pipe sync.
- **SyncAll ALLOWED ONLY** in: (1) Pre pipeline — global init where all cores must finish before main compute starts (e.g. `LIGVectorPre::ZeroDkWorkspace`); (2) Post pipeline — global finalization where all cores must finish before results are consumed (e.g. `LIGVectorPost::CastAndWriteDk`).
- **SyncAll FORBIDDEN** in: (1) the main compute (batch) loop — per-iteration cross-core barriers are the #1 cause of multi-core hangs; (2) any code path reachable from a zero-work-core early-exit — if a core can skip work, it must not be required to reach a SyncAll.

**Concrete anchor**:
```cpp
// WRONG — cross-core barrier in AiCore-only batch loop:
for (int b = 0; b < batch; b++) {
    ProcessVec2(...);
    SyncAll();  // DEADLOCK: zero-work cores skip this, working cores wait forever
}
// CORRECT — intra-core pipe sync:
for (int b = 0; b < batch; b++) {
    ProcessVec2(...);
    PipeBarrier<PIPE_ALL>();  // synchronizes V/MTE1/MTE2/MTE3 within THIS core only
}
```

**Auto-detection**: `pre_build_check.py --checks sync` flags any `SyncAll()` in an AiCore-only kernel (no MIX_AIC in file) and any `SyncAll()` inside a loop body (the most dangerous pattern), reporting line number + a replace-with-PipeBarrier suggestion.

**Evidence**: LightningIndexerGrad (2026-06-05): P133 replaced all 9 `SyncAll()` calls with `PipeBarrier<PIPE_ALL>()`. Before: B≥2 multi-core hang 100% of the time. After: 20/20 stability tests, zero failures, 26+ diverse configs all PASS. The 9 SyncAll calls had been introduced over 3 prior iterations (P128–P132) as "safety" barriers that actually caused the hang.
