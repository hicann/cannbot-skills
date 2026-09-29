---
title: Pipeline And Buffer
purpose: Explain queued buffering and handoff only as compatibility material that can be contrasted with the regbase-authoritative path.
read_when:
  - You need to explain or compare a membase-like queued pipeline.
  - A fallback path still uses TQue/TBuf semantics and needs clear limits.
not_for:
  - Default regbase shell semantics
  - VF compute-body design
keywords:
  - pipeline
  - buffer
  - TQue
  - compatibility
next_reads:
  - datacopy_best_practices.md
  - regbase_api_sync.md
  - ../pitfalls/regbase_vs_membase_confusions.md
depth: intermediate
topic_type: api
---

# Pipeline and Buffer

This card records UB buffering and queue-style handoff as compatibility material for regbase direct-invoke work. Use it when you need to contrast a membase-style pipeline with the regbase-authoritative path or document a fallback implementation.

## Buffer Choice

When documenting a compatibility path, use `TQue` when the buffer participates in movement plus compute handoff.

Use `TBuf` when the buffer is only scratch space for vector computation in that compatibility path.

## Queue Depth And Double Buffer

Do not confuse queue depth with double buffering in the compatibility path:

- `TQue<..., depth>` controls the queue abstraction.
- `InitBuffer(que, num, size)` controls how many physical buffers are provisioned.
- `num = 2` is what enables double buffering for the queue.

In practice, keep the template depth small and let the `InitBuffer` count do the work when you are comparing against or falling back to a queued model.

## Pipeline Sync Model

The compatibility flow is:

1. allocate a queue buffer
2. copy into it
3. `EnQue` it
4. `DeQue` it in the next stage
5. compute
6. enqueue the output
7. `DeQue` the output when it is ready to leave UB

This is what makes movement and compute overlap safely in the compatibility path.

## When To Use `PipeBarrier`

Use a barrier only when you need a coarse fence in the compatibility path or are diagnosing a sync defect.

- good for debugging whether the bug is ordering-related
- poor as a default performance strategy
- not a substitute for proper queue handoff

## Batch Movement Pattern

For row-oriented work, a common pattern is:

- batch copy multiple rows into UB
- process rows one by one in place
- batch copy the results back out

This pattern is often better than issuing one copy call per row in a queue-based fallback.

## Common Mistakes

- Treating `AllocTensor` as if it also waits for data readiness.
- Assuming `depth` is what turns on double buffering.
- Using `TBuf` for a movement path that actually needs queue semantics.
- Serializing the pipeline with barriers when queue handoff would keep the stages overlapped.

## Related Documents

- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[arithmetic_and_reduce]]
- [[precision_and_runtime]]
