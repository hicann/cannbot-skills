---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "Cross-core group barrier — every paired sub-block must participate symmetrically, and each Set must follow that sub-block's own real GM write"
description: "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any MIX_AIC_1_2 cube↔vec kernel)"
phenomenon: build_failure
signal:
  - "applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any MIX_AIC_1_2 cube↔vec kernel)"
confidence: single_run
classified_by: fallback-whole-body
original_id: OL-190
timestamp_inferred: true
tags: [ascendc, ol-190]
created_at: 2026-08-30T16:00:00Z
updated_at: 2026-08-30T16:00:00Z
---
## 条目正文（忠实搬运，含全部更正/佐证 bullet）

`applies_to: soc=Ascend910_9382 (V220 / A3); cann=9.0.0; bisheng=n/a; op_class=all (any MIX_AIC_1_2 cube↔vec kernel)`
`verified_on: soc=Ascend910_9382; cann=9.0.0; kernel_type=KERNEL_TYPE_MIX_AIC_1_2`
`unverified_on: soc=Ascend950PR (V351 / A5)`

**Principle**: `CrossCoreSetFlag<0x2>` / `CrossCoreWaitFlag<0x2>` is a **group barrier** coupling one AIC with its paired AIV sub-blocks (2 per AIC under `MIX_AIC_1_2`). Two non-negotiable invariants:
1. **Symmetric participation** — the AIC's single `CrossCoreWaitFlag<0x2>` clears only after BOTH AIV sub-blocks have issued the matching `CrossCoreSetFlag<0x2>`. Gating the whole handshake on one sub-block (e.g. `if (subBlockIdx==0)`, or idling the second sub-block when `BLOCK_M` only fills one) → the group-set never completes → **AIC waits forever → deadlock** (kill at timeout, zero output).
2. **Set-after-own-write** — each sub-block's `Set` must follow that sub-block's OWN real GM write on the matching pipe (`PIPE_MTE3` after a DataCopy store, `PIPE_FIX` after a Fixpipe). An "empty" Set with no preceding GM traffic retires immediately; the group barrier then clears before the data-bearing sub-block's write lands → the AIC reads STALE/uninitialized workspace → **garbage output (not a hang)**.

**Resolution rule (applies to ANY single-producer vector stage)**: when the producer work is naturally single-sub-block (the data isn't row-splittable), still make BOTH sub-blocks participate by SPLITTING the producer by row range (topK rows, N1 tiles, etc.) so each sub-block writes its own disjoint GM slice then Sets — the union of slices is the full workspace, counts balance, and every Set certifies real retired data. This is the cv-agent FA pattern: both AIV sub-blocks run ComputeVec1, each writes `pSlot[rowStart_*BLOCK_N]` (its own range) then `ProducerReleaseMte3`.

Concrete anchor (the deadlock fix):
```cpp
// WRONG: only sub-0 sets -> AIC's wait never completes -> deadlock
if (subBlockIdx_ == 0) { CrossCoreSetFlag<0x2, PIPE_MTE3>(SIG); }
// WRONG: both set but sub-1's Set has no preceding write -> AIC reads stale -> garbage
if ASCEND_IS_AIV { CrossCoreSetFlag<0x2, PIPE_MTE3>(SIG); }   // sub-1 wrote nothing
// RIGHT: both sub-blocks write their own slice THEN set
if ASCEND_IS_AIV { vec_.GatherSlice(cid); CrossCoreSetFlag<0x2, PIPE_MTE3>(SIG); }
if ASCEND_IS_AIC { CrossCoreWaitFlag<0x2>(SIG); }
```

**Cross-ref**: `fa_class/cross_core_sync.md` §3 (single-flag-broadcast rule, "所有 AIV 子块共享同一 flag") — this OL adds the AIV→AIC direction (group-set-count) and the set-after-own-write ordering nuance.

## Evidence
- lightning_indexer_grad (generated AscendC kernel, A3, 2026-05-27): first instance. `aivActive=(subBlockIdx==0)` gate deadlocked (38-case verify killed at 400s, zero output). Making both sub-blocks set fixed the hang but an empty sub-1 Set then read stale workspace → garbage. Splitting GatherGk by topK rows + ComputeGrads by N1 tiles (each sub-block writes its own slice then Sets) fixed both.

## Other instances (predicted)
Any `MIX_AIC_1_2` cube↔vec op with an AIV→AIC handoff: FlashAttention (P ready), MoE finalize, fused norm+matmul, any two-stage producer/consumer across the AIC↔AIV boundary.

<!-- 迁移自 porter OPERATIONAL_KNOWLEDGE.md OL-190（category=none，convert_ol_to_okf.py --faithful-fallback，B2 整档忠实搬运）。confidence/severity/reproduce_count 未升格。 -->
