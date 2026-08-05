---
schema_version: okf.v1
kind: implementation_trap
type: implementation_trap
source_family: curated
title: "GM workspace slot rotation via modulo-(MAX_LAG+1) for cross-core stage decoupling"
description: "applies_to: any soc with cross-core sync (CAND-FA1); cann=9.0.0+; op_class=multi_stage_pipeline_with_GM_handoff derived-from: cann-source (FA-class workspace layout, 2026-05-10 revise-cl4) verified_on"
phenomenon: build_failure
signal:
  - "A multi-stage producer→consumer pipeline (CAND-FA1) uses GM-resident workspace tensors as the hand-off medium between cores (cube↔vec or peer-cube↔peer-cube), a"
confidence: inferred
status: stub
original_id: CAND-FA3
timestamp_inferred: true
tags: [candidate, inferred, max_lag, lag, cand-fa3]
created_at: 2026-07-12T16:00:00Z
updated_at: 2026-07-12T16:00:00Z
---
## 候选（未验证 —— 默认检索不返回,需 --status all 才可见）

`applies_to: any soc with cross-core sync (CAND-FA1); cann=9.0.0+; op_class=multi_stage_pipeline_with_GM_handoff`
`derived-from: cann-source (FA-class workspace layout, 2026-05-10 revise-cl4)`
`verified_on: cann-source (FA-class op_kernel main scheduling loop)`
`unverified_on: a5_ops`

**Trigger**: A multi-stage producer→consumer pipeline (CAND-FA1) uses GM-resident workspace tensors as the hand-off medium between cores (cube↔vec or peer-cube↔peer-cube), and adjacent stages must overlap without an iteration overwriting GM scratch that a downstream stage is still consuming.

**Parameter definition (corrected vs prior revision)**: `MAX_LAG` is the maximum iteration distance between a producer write of a slot and the last still-active consumer read of that same slot. It is **not** the count of stages in flight. Example: with three pipeline stages (cube-QK → vec-Softmax → cube-PV), the producer of stage-1 at iteration `k` must keep its slot intact until the stage-3 consumer at iteration `k` finishes — that consumer runs `MAX_LAG = 2` iterations after the producer in the outer loop (it is launched when the producer is on iteration `k+2`). Hence `MAX_LAG = 2` requires `MAX_LAG + 1 = 3` distinct slots, not 3 in-flight producers.

**Recommendation**: Allocate the workspace as `(MAX_LAG + 1)` parallel GM slots per core. At outer-loop iteration `k`, the producer writes:

```cpp
slotIdx = k % (MAX_LAG + 1);
```

A consumer that reads data produced `lag` iterations earlier (`1 <= lag <= MAX_LAG`) reads:

```cpp
readSlot = (k + (MAX_LAG + 1) - lag) % (MAX_LAG + 1);
```

The set `{k, k-1, ..., k-MAX_LAG}` is pairwise-distinct modulo `MAX_LAG + 1`, so the active slots never alias. The slot used at iteration `k - (MAX_LAG + 1)` is the one being overwritten now, which is collision-free if and only if every consumer that could still reference it has finished.

**Safety condition (must be structurally enforced)**: this scheme is collision-free if and only if `MAX_LAG` is an **upper bound** on the actual iteration retention of every consumer of the slot. In CAND-FA1-style pipelines this is enforced by the cross-core flag chain: stage `s+1` at iteration `k` blocks on stage `s`'s "produced" flag for iteration `k`, and the producer at iteration `k + MAX_LAG + 1` blocks on the "consumed" event from the last downstream stage at iteration `k`. If the actual lag is bounded above by `MAX_LAG` for the entire schedule, modulo rotation is sound; otherwise it is unsafe.

**Per-core GM offset**:

```cpp
slotOffset = uint64_t(coreIdx) * SLOT_BYTES * (MAX_LAG + 1)
           + uint64_t(slotIdx) * SLOT_BYTES;
```

Typical `MAX_LAG = 2` gives three slots per core and supports a three-stage overlapped window: stage-A writes iteration `k`, stage-B reads iteration `k-1`, stage-C reads iteration `k-2`. `MAX_LAG = 1` reduces to classic double-buffer ping-pong. `MAX_LAG >= 3` is rarely useful because the cross-core flag chain and the slowest stage usually cap useful overlap; justify with profiling.

**Concrete anchor** (public-API surface):

```cpp
constexpr uint32_t MAX_LAG = 2;
constexpr uint32_t SLOT_COUNT = MAX_LAG + 1;

// Producer (e.g. cube core at stage 0):
uint32_t writeSlot = stageCount % SLOT_COUNT;
uint64_t writeOffset = uint64_t(coreIdx) * SLOT_BYTES * SLOT_COUNT
                     + uint64_t(writeSlot) * SLOT_BYTES;
auto gProducerView = gWorkspace[writeOffset / sizeof(T)];
// ... write gProducerView, then publish CAND-FA1 cross-core flag for this stage/iter ...

// Consumer at downstream stage, reading data produced `lag` iters earlier:
uint32_t readSlot = (stageCount + SLOT_COUNT - lag) % SLOT_COUNT;
uint64_t readOffset = uint64_t(coreIdx) * SLOT_BYTES * SLOT_COUNT
                    + uint64_t(readSlot) * SLOT_BYTES;
auto gConsumerView = gWorkspace[readOffset / sizeof(T)];
// ... wait on CAND-FA1 flag for the corresponding (stage, iter) before reading ...
```

**Cross-reference to P-P77 (no conflict, different tier)**: P-P77 rotates UB-resident `TQue` slots for **same-core** inter-iteration synchronization between MTE2 and V via TQue/EnQue/DeQue semantics. CAND-FA3 rotates **GM-resident workspace slots** for **cross-core** inter-stage hand-off with explicit cross-core flags. The two tiers are orthogonal: a kernel can use UB ping-pong (P-P77) inside each stage and GM slot rotation (CAND-FA3) between stages. UB ping-pong is bounded by UB capacity and queue depth (typically 2); GM slot rotation is bounded by workspace allocation and the cross-core flag protocol, which makes `MAX_LAG > 1` cheap.

**Determinism**: deterministic when (a) each `(coreIdx, stageCount, slotIdx)` write region has exactly one producer; (b) every consumer read is gated by the CAND-FA1 producer-complete flag for the matching `(stage, iteration)`; (c) the scheduler structurally guarantees that no consumer retains a slot beyond `MAX_LAG` iterations after its producer (e.g., by having the producer at iteration `k + MAX_LAG + 1` block on the last consumer's completion flag for iteration `k`). It does **not** make concurrent multi-producer writes to the same GM region deterministic; those remain an A-P61-style race unless the workspace is partitioned per producer.

**Hard do-not-apply (delayed-consumer case)**:
- Do not use bare modulo rotation when consumer retention is not structurally bounded by `MAX_LAG`. Examples that violate this:
  - Producer can race ahead of the slowest consumer because of data-dependent skip logic (e.g., sparse-block attention where some stage-B iterations are no-ops while stage-A continues at full speed).
  - Cross-core flag is only set after producer completes but never signals consumer release, so the producer at `k+MAX_LAG+1` can wrap and overwrite slot before the consumer at `k` reads it.
  - Backpressure from a downstream sink (e.g., HBM-bandwidth-stalled WriteBack stage) makes the latest consumer fall behind by more than `MAX_LAG` iterations.
- In these cases, choose one of:
  1. Grow the slot count to a hard upper bound on actual retention (acceptable when retention is bounded but variable).
  2. Add a per-slot release/ack: each consumer signals "slot released" after read; the producer at iteration `k + MAX_LAG + 1` waits on the release event for iteration `k` before reusing the slot.
  3. Switch to an explicit queue (e.g., TQue at GM tier via a circular ring buffer with head/tail flags) — this is a generalization of the modulo scheme with explicit credit accounting.

**Other instances predicted**:
- FA-class forward: multiple GM scratch tensors (S, P, OTmp, OUpdate in the CANN reference) all use the same slot-rotation scheme with shared `MAX_LAG`.
- Prefill+decode hybrid attention where cube prefill produces GM scratch consumed by vec decode at a fixed iteration lag.
- Multi-stage scan/reduction with cube partials produced N iterations before vec finalize consumes them.
- Fused norm-then-GEMM where vec normalization produces normalized GM scratch consumed by cube GEMM at the next iteration.
- MoE finalize where expert-output scratch is produced by one stage and combined by a routing stage at fixed lag.

**Promote when**: a5_ops ships a multi-stage cube↔vec kernel using this rotation AND msprof confirms at least two stages in flight simultaneously, visible as cube and vec utilization both greater than 0% in the same wall-clock window AND the safety condition is structurally argued (not just empirically observed) in `knowledge_update.md`.

**Risks before promotion**:
- `SLOT_BYTES` must include alignment padding so adjacent slots do not share an HBM line / cache sector (false-share serializes through the LLC). Use `SLOT_BYTES = AlignUp(payload_bytes, 512)` unless a tighter architecture-specific bound is proven.
- Workspace size grows linearly with `(MAX_LAG + 1) * num_cores`. Example: 24 cores × 3 slots × 128 KB each = 9 MB of GM workspace. Host launch code must pre-check required workspace size against device free memory before dispatch.
- Modulo rotation is a layout convention, not a synchronization primitive. It prevents address collision only under a bounded-lag schedule; the cross-core flag protocol (CAND-FA1) or an equivalent ack chain is still required for read-after-write and write-after-read correctness.
- If the kernel later adds a stage with data-dependent skip behaviour (e.g., sparse-block attention paths that nop-out some stage-B iterations), revisit whether the lag is still bounded by `MAX_LAG`; the modulo scheme may silently violate the safety condition without crashing.
- `MAX_LAG` is a compile-time constant in the CANN reference. Templating it makes per-shape tuning possible but multiplies dispatch table size; combine with CAND-FA1 dispatch budget before promotion.

<!-- 迁移自 porter kb/target/ascendc/patterns/unverified/candidates.md（CAND-FA3，convert_cand_to_okf.py）。status=stub 未验证,待复现后 promote。 -->
