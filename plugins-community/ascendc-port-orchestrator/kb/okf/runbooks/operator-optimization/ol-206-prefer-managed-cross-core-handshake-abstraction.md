---
schema_version: okf.v1
kind: operator_optimization
type: optimization_runbook
source_family: curated
title: "Prefer the managed cross-core abstraction for a MIX cube↔vec result handshake — hand-rolling exposes uncovered sub-cases one at a time"
description: "A hand-rolled cube↔vec result handshake must get sync-mode, per-AIV flags, asymmetric pipe, managed id, and GM visibility all right at once — prefer the managed cross-core abstraction."
confidence: single_run
original_id: OL-206
classified_by: llm-assisted
timestamp_inferred: true
tags: [cross-core-sync, optimization, ol-206, flash-attention, cube-vec-mix, race]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 优化点 / 选型

**Applies to**: Ascend950PR / CANN 9.0.0 / MIX 1:1 cube↔vec software-pipelined result handshake (FlashAttention / fused norm+matmul / MoE finalize). Verified on FA-A5 graybox kw-gb4→kw-gb5, 2026-06-03.

**Principle**: a hand-rolled cube↔vec *result* handshake on arch35-AIV must get MANY independent dimensions right **simultaneously**:
- sync mode 4, not 2;
- one flag per consumer AIV (`id` + `id+16` for 1-AIC→2-AIV; a single flag races the 2nd AIV);
- an ASYMMETRIC pipe (producer `Set` on `PIPE_FIX`, consumer `Wait` on `PIPE_V` — not the producer's pipe);
- a managed non-colliding event-id (not a hand-picked literal);
- and for a GM-routed result, an explicit data-visibility ordering for the consumer's MTE2 GM-read (flag-pipe ordering alone is insufficient).

Fixing any one dimension exposes the next still-uncovered sub-case, so a recipe-driven hand-roll converges only after iterating through *all* of them. The managed BaseApi cross-core abstraction (`Buffer<SyncType=CROSS_CORE_SYNC_FORWARD>` with `.SetCrossCore()`/`.WaitCrossCore()`) encapsulates every dimension (picks the pipe, manages the id, chooses routing + fence) and is bit-exact + deadlock-free.

**Decision**: for a MIX cube↔vec result handshake, PREFER the abstraction; hand-roll only with a specific reason, and if you do, expect to walk the entire `cross_core_sync.md §4(C)` sub-case ladder.

**Concrete anchor (the ladder, each rung a separate iteration)**: FA-A5 graybox —
- (gb2) `<0x2>` symmetric → DEADLOCK → mode-4 + `+16` dual-flag;
- (gb3) result `nan` → softmax-stat broadcast (CAND-FA-SOFTMAX-STAT-1);
- (gb4/gb5) consumer `Wait` on `PIPE_FIX`→`PIPE_V` compiles but a GM-routed result (consumer first-touch = MTE2 GM-read) STILL races (`softmax_sum` 0↔64 across fresh-process runs, `507015` aivec OOB) → needs route-result-through-UB or an explicit GM visibility fence.

Five iterations, each closing one rung. The whole-port reference never hand-rolls it — it routes through the abstraction and is bit-exact 64/64 on the first build.

**Cross-ref**: `cross_core_sync.md §4` (full hand-roll recipe + GM-vs-UB sub-case ladder), CAND-FA-SOFTMAX-STAT-1, `feedback_fsm_orchestration_crutch_whitebox_first` (reading the working reference's mechanism beats N hand-roll iterations), `feedback_passcount_variance_first_hypothesis_is_nondeterminism`, P-P103 (FA-class template).

**Evidence**: FA-A5 `flash_attention_score` graybox kw-gb4 (case7 minimal repro: 1-AIC+2-AIV `softmax_sum` drift 55) + kw-gb5 (PIPE_V compiles, GM-routed still races), 2026-06-03; whole-port reference bit-exact 64/64 via the abstraction (`1f00bbdc`). Predicted to recur on any MIX 1:1 cube↔vec software-pipelined op with a producer→consumer result path.
