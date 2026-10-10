---
title: Regbase API Sync
purpose: Distinguish queue handoff, VF lane control, overlap events, and cross-core coordination so sync choices stay on the right layer.
read_when:
  - The task needs a sync decision tree rather than a single generic sync answer.
  - You are debugging or designing handoff boundaries around regbase kernels.
not_for:
  - Top-level operator classification
  - Pure build failures
keywords:
  - sync
  - queue handoff
  - VF
  - overlap
next_reads:
  - ../patterns/regbase_sync_patterns.md
  - ../pitfalls/symptom_to_cause.md
  - regbase_api_whitelist.md
depth: foundation
topic_type: api
---

# Regbase API Sync

This card records synchronization decisions that commonly appear around regbase kernels. Use it to distinguish:

- UB-stage handoff
- VF-internal lane control
- explicit event-driven overlap
- cross-core coordination

Do not collapse those into one “sync” concept.

## 1. Default Reading

For most regbase operators:

- the outer kernel shell may still use queue-based stage handoff
- the VF body usually does not need explicit event synchronization
- advanced `SetFlag` / `WaitFlag` patterns are a special case, not the default

## 2. Queue-Based Stage Handoff

When the kernel uses queue-backed UB staging:

- `EnQue` marks a UB tile ready for the next stage
- `DeQue` makes the next stage wait until that tile is ready

That is valid local stage ordering in the kernel shell. It is not a sign that the task has automatically left the regbase path.

Use this model when the question is:

- has `CopyIn` finished preparing the current UB tile?
- can `Compute` consume it yet?
- can `CopyOut` now write the finished UB tile?

## 3. VF Lane Control Is Not Event Sync

Inside `__VEC_SCOPE__`:

- `MaskReg` controls active lanes
- `LoadDist` controls how UB data is loaded into register objects
- `StoreDist` controls how register results are written back to UB

These are part of the VF execution model. They do not replace:

- queue lifecycle
- event flags
- cross-core coordination

## 4. Explicit Event Flags

Use explicit flags only when the design truly overlaps movement and compute manually.

Typical families:

- `SetFlag`
- `WaitFlag`
- `PipeBarrier`

Typical use:

- ping-pong overlap between vector work and MTE work
- proving that a bug is caused by missing stage ordering
- advanced kernels with multiple concurrent UB slices

Do not add explicit flags to a simple regbase kernel unless a real reference operator proves the need.

## 5. Cross-Core Coordination

Use cross-core flags only for real inter-core dependencies:

- `CrossCoreSetFlag`
- `CrossCoreWaitFlag`

These are not substitutes for local UB stage ordering.

## 6. Practical Decision Path

1. Ask whether the problem is UB-stage readiness, VF lane semantics, explicit overlap, or cross-core dependency.
2. For UB-stage readiness, start with queue lifecycle or stable stage ordering.
3. For VF bodies, keep `MaskReg` and `LoadDist` / `StoreDist` in the “register semantics” bucket, not the “sync primitive” bucket.
4. Introduce explicit flags only after checking a real reference operator that uses them.
5. Use cross-core flags only when the design truly crosses core boundaries.

## Related Documents

- [[../regbase_development_guide]]
- [[../patterns/regbase_sync_patterns]]
- [[../patterns/regbase_kernel_dataflow_patterns]]
- [[pipeline_and_buffer]]
- [[precision_and_runtime]]
