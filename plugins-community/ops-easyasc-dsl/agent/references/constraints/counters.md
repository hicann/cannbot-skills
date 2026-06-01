# Counter and Buffering Constraints

Read this file when a kernel uses `DBuff`, `TBuff`, nested loops, delayed reuse, or autosync-sensitive slot ownership.

## Goal

Keep buffer lifetimes explicit enough that:
- ownership stays understandable
- slot lineage stays stable
- autosync grouping stays correct
- different stages do not silently fight over one counter

## 1. Core rule

Different buffer lifetimes must use different counters.

This is not an optional cleanup preference.
It is a required authoring rule in this repository.

Symmetric rule:
- same-lifetime paired buffers may share one counter

## 2. What counts as the same lifetime

Buffers may share one counter when they are:
- loaded together
- consumed together
- retired together
- logically one stage pair

Typical example:
- `l1x` and `l1y` for the same `k` tile may share one `l1_cnt`

Buffers should not share a counter when one of them lives across a different loop boundary or stage boundary.

Typical example:
- `l1` streaming inside the `k` loop should not share the same counter with an outer `L0C` or vec postprocess stage

Concrete nested matmul example:
- if `l1x` / `l1y` advance on every `k` tile and are consumed as one operand pair, they may share one `l1_cnt`
- if `l0c` is owned by the outer `n` tile, keep its `l0c_cnt` separate from the `k`-loop `l1_cnt`
- reusing one counter across those different loop-owned lifetimes can blur autosync slot lineage and break the pipeline rhythm

## 3. Why this matters

Reusing one counter across different lifetimes can break reasoning in several ways:
- DBuff parity stops matching the real stage lifetime
- autosync grouping sees a misleading slot lineage
- nested loops become harder to reason about
- a later refactor silently changes which stage owns a slot

A kernel may still "look" fine while the lifetime model is already broken.

## 4. Preferred counter layout

Name counters by stage ownership, not by generic sequence order.
Good examples:
- `l1_cnt`
- `l0c_cnt`
- `tile_cnt`
- `stage1_cnt`
- `stage2_cnt`

Bad pattern:
- one global counter reused everywhere because it was convenient once

## 5. Nested-loop rule

Across different loop levels, do not reuse one counter for different `DBuff` / `TBuff` lifetimes.
If two loop levels each own their own buffering rhythm, give them separate counters.

This matters especially when:
- one loop streams operands
- a parent loop owns an output tile
- a delayed consumer runs one iteration later than the producer

## 6. Delayed-stage rule

If a stage consumes data one iteration later, treat that as a distinct lifetime unless the entire slot family truly remains one coherent delayed pipeline.

Stable pattern in this repository:
- one counter for immediate score/probability production
- a different counter for delayed `p @ value` accumulation

Do not hide a delayed lifetime behind the producer counter.

### Buffer depth must match the delayed overlap window

Do not assume "`one-tile delay` means `DBuff` is enough".

Counterexample:
- tile `j` is produced into slot `0`
- tile `j+1` is produced into slot `1`
- the delayed stage is still consuming tile `j`
- the producer is already free to start tile `j+2`

If tile `j+2` also maps to slot `0`, a `DBuff` family can overwrite the still-live data for tile `j`.

Stable rule:
- when an operand family is reused by a delayed stage, size the local buffer family from the
  real producer/consumer overlap, not from the nominal delay count alone
- for the common "`produce j`, `consume j-1`, producer may begin `j+2` early" pattern,
  the reused operand family often needs `TBuff`

Concrete example:
- stage 1 uses `k_j` for `q @ k_j^T`
- delayed stage later reuses that same `k_j` for `dqk_j @ k_j`
- `k` may need `TBuff` even though `v` can still remain `DBuff`

## 6a. Chunked vec stages usually need separate input and output counters

When one vec stage is split into row chunks such as `32 x 128` or `16 x 128`,
do not assume one counter should drive every local family just because all
chunks advance in the same loop.

Stable pattern:
- one counter for the `MTE2 -> V` input families such as `qkbuf` / `dpbuf`
- one counter for the `V -> MTE3` output families such as `pbuf` / `dqkbuf`

Even if both counters increment once per chunk today, keeping them separate is
still the safer authoring rule because:
- the slot families belong to different autosync pairings
- later refactors often change one side's lifetime first
- reusing one counter can blur which family actually owns a problematic slot

Practical a2 lesson:
- `vec_in_cnt` and `vec_out_cnt` made the per-chunk `MTE2 -> V -> MTE3` story explicit
- trying to treat a live input family as scratch for output preparation made the lineage much harder to reason about

## 6b. Scratch tensors must shrink with the chunk, not just the loops

If you squeeze a stage from `32 x 128` down to `16 x 128`, update the scratch
tensors and helper assumptions too.

Do not assume a larger scratch tensor can always be safely reused through a
smaller sliced view.

Practical failure modes:
- simulator-v2 storage/view validation failures
- `ub_to_ub` size mismatches such as `need=16384 bytes, have=8192 bytes`
- hidden helper assumptions that still operate on the old full-chunk shape

Stable rule:
- shape dedicated scratch tensors from the current chunk size
- keep helper-local metadata tensors such as `quant_meta` / `quant_scale` on the same `[chunk_m, TILE_N]` shape as the real chunk
- revalidate UB usage after every chunk-size change; the safe chunk is an implementation result, not a constant you can cargo-cult

Practical a2 lesson:
- `32 x 128` chunking worked for the hif8 bring-up but left UB tight
- `16 x 128` plus dedicated chunk-sized quant scratch reduced pressure and kept the pipeline warning-free

## 7. Same-pipe grouping still matters

Counter correctness is easier to preserve when same-pipe instructions are grouped together.
Within one loop iteration, prefer:
- grouped MTE loads
- grouped compute
- grouped writeback

This makes ownership and stage boundaries much more obvious.

## 8. Mutex depth must match the real slot count

`CvMutex` / `VcMutex` `depth` is not a vague pipeline-tuning hint.
In this repository it directly controls how many initial ready tokens the kernel gets for that
intra-core handoff.

Concrete implementation fact:
- kernel build injects one initial `vec_ready` / `cube_ready` per unit of `depth`
- so `depth=N` means the producer may have up to `N` payloads in flight before the consumer has
  to return capacity

Stable rule:
- plain `Tensor` guarded by a mutex means `depth=1`
- `DBuff` may justify `depth=2`
- `TBuff` may justify `depth=3`
- `QBuff` may justify `depth=4`

Important boundary:
- those `2/3/4` values are upper bounds from the physical slot family, not defaults
- only use them when the producer/consumer really rotate across those distinct slots
- if the code keeps reusing one fixed view or one effective slot, `depth` must stay `1` even if
  the object type is `DBuff` / `TBuff` / `QBuff`

Why this matters:
- setting `depth=2` on a single-buffer `Tensor` tells the runtime there are two free slots
- the producer can then publish two in-flight payloads even though both writes land on the same
  storage
- that silently models overlap the kernel does not actually have and makes overwrite races look
  legal

Practical test:
- count the actual distinct storage slots that can hold independent unconsumed payloads
- set `depth` to that number, not to the overlap you wish you had

## 9. `TBuff` indexing rule in fused side-split kernels

In the Python DSL, the triple-buffer class is `TBuff`.
Its `__getitem__` already carries the buffer object and the raw counter to lowering.

For fused cube/vec kernels, prefer:
- `buf[stage_cnt]`

Stable pattern:
- keep the delayed-stage lineage in one counter such as `stage2_cnt`
- index `TBuff` directly with that counter
- let the buffer family, not the kernel body, own the `% 3` slot mapping

You can still spell an explicit modulo slot when you really need to reuse that slot value:
- `slot = var_mod(stage_cnt, 3)`
- `buf[slot]`

Current parser behavior:
- side pruning now removes dead tmp `var_mod` chains that no longer feed the retained side
- so inline `buf[var_mod(stage_cnt, 3)]` is no longer expected to fail by itself

Even so, direct indexing remains the preferred style because:
- the buffer family already knows it is triple-buffered
- `buf[stage_cnt]` keeps the lineage clearer
- it avoids redundant scalar instructions when the slot value is not reused anywhere else

## 10. Quick checklist

Before accepting the counter design, verify:
- each counter has one clear stage owner
- different loop-owned lifetimes do not share one counter
- same-lifetime pairs share only when that pairing is intentional
- delayed consumers use a counter that matches their own lifetime
- delayed reused operands have enough physical slots for the actual overlap window
- each `CvMutex` / `VcMutex` depth matches the real number of simultaneously reusable slots
- autosync grouping still matches the logical slot story

## Files to study

- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/matmul_rowwise_norm_large_nk.py`
- `agent/example/kernels/a5/vec_cube_vec_scale2_abs_add1_matmul.py`
- `agent/example/kernels/a5/test_mla_entire.py`
