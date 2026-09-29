---
title: Regbase Sync Patterns
purpose: Choose synchronization by layer so UB handoff, VF lane control, and overlap logic are not mixed into one model.
read_when:
  - You need to reason about queue handoff, VF lane control, or explicit overlap flags.
  - Failures look like synchronization bugs but the layer is still unclear.
not_for:
  - Operator classification
  - Pure build-time problems
keywords:
  - synchronization
  - queue handoff
  - VF lane control
  - SetFlag
next_reads:
  - ../api/regbase_api_sync.md
  - regbase_kernel_dataflow_patterns.md
  - regbase_buffer_partitioning.md
depth: intermediate
topic_type: pattern
---

# Regbase Sync Patterns

This note explains synchronization by layer. The most common mistake is to use one sync model for every layer of the kernel.

## 1. Default Rule

Start with the lightest synchronization model that matches the layer you are working in.

In many regbase operators:

- UB stage handoff is handled by queue lifecycle or straightforward stage ordering
- VF internals need lane control, not event synchronization
- explicit `SetFlag` / `WaitFlag` only appear in more advanced overlap or ping-pong designs

## 2. UB-Level Stage Handoff

When the outer shell uses queue-based staging, the local readiness model is:

- `EnQue`: a UB tile is ready for the next stage
- `DeQue`: the next stage can consume the UB tile

This is stage coordination for local pipeline flow. It is not the same thing as VF lane control.

Use this level to reason about:

- whether `CopyIn` has finished preparing the UB tile
- whether `Compute` is done with the current UB tile
- whether `CopyOut` can consume the finished UB tile

## 3. VF-Level “Synchronization” Is Usually Not Event Sync

Inside `__VEC_SCOPE__`:

- `MaskReg` controls active lanes
- `LoadDist` controls how UB data is loaded into registers
- `StoreDist` controls how register results are written back

None of those are event synchronization primitives.

That means:

- do not describe `MaskReg` as a sync mechanism
- do not use `LoadDist` / `StoreDist` as if they were barriers
- do not add `PipeBarrier` just because a VF loop has multiple instructions

## 4. Explicit Event Flags Belong To Advanced Overlap Patterns

Use `SetFlag` / `WaitFlag` when the design truly overlaps movement and compute manually and the reference implementation already uses that pattern.

Representative pattern:

```cpp
SetFlag<HardEvent::V_MTE2>(pingPongID);
WaitFlag<HardEvent::V_MTE2>(pingPongID);
SetFlag<HardEvent::MTE2_V>(pingPongID);
WaitFlag<HardEvent::MTE2_V>(pingPongID);
```

This style appears in advanced transformer kernels where multiple UB slices, ping-pong buffers, and MTE/vector overlap are orchestrated explicitly.

Reference shape:

- a multi-slice or ping-pong implementation that already proves explicit overlap is required

Do not import this pattern into a light unary or short fused chain unless the operator really needs it.

## 5. Cross-Core Flags Are A Separate Layer

`CrossCoreSetFlag` and `CrossCoreWaitFlag` belong to true inter-core coordination.

Use them only when:

- one core depends on another core’s progress
- the kernel design explicitly spans cross-core producer/consumer relationships

Do not use them to patch a local UB ordering issue.

## 6. Practical Decision Path

1. Ask whether the issue is local stage readiness, VF lane control, explicit overlap, or cross-core coordination.
2. For local stage readiness, start from queue lifecycle or stable stage ordering.
3. For VF internals, use `MaskReg` and correct `LoadDist` / `StoreDist` semantics, not barriers.
4. For explicit overlap, verify an existing regbase reference before introducing `SetFlag` / `WaitFlag`.
5. For cross-core logic, only then consider cross-core flags.

## Related Documents

- [[../regbase_development_guide]]
- [[regbase_kernel_dataflow_patterns]]
- [[regbase_buffer_partitioning]]
- [[../api/regbase_api_sync]]
