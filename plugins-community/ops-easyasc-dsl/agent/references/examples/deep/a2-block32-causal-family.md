# Deep Note: A2 block32 causal attention family

Open this file only after `kernel-catalog.md` has narrowed the candidate to one
of the `agent/example/kernels/a2/attention/block32_causal/` entries.

## Shared Contract

All entries in this family preserve the same public math:
- `score_j = q.float() @ k_j.float().t() * scale`
- score tiles obey blockwise causal masking: `floor(k_pos / 32) <= floor(q_pos / 32)`
- online softmax keeps `rowmax` and `rowsum` in float
- the delayed value path rounds only `p_j` through `p_j.half().float()`
- final outputs are `out`, `rowmax`, and `rowsum`

The variants are scheduling and buffering experiments around that contract. Do
not treat them as different attention formulas.

## Family Map

### `flash_attn_full_pj_half_block32_causal.py`
- simplest correct block32-causal reference
- one `128x128` N tile per stage-1 / stage-2 step
- two-slot score/P/PV workspace pattern
- future fully-invalid score tiles are skipped with `active_tiles_n = Min(tiles_n, lmt + 1)`
- best starting point when the formula, masking, or tail behavior is still unclear

### `flash_attn_full_pj_half_block32_causal_v3.py`
- same math, but with causal-work-balanced M-tile scheduling
- distributes M tiles round-robin across cube cores and reverses M-tile order on odd BH blocks
- hoists Q GM->L1 so one Q tile can feed all active N tiles for that M tile
- uses lookahead-3 with four workspace slots for score/P/PV/expdiff
- good for studying scheduling depth without cross-M row-state prefetch

### `flash_attn_full_pj_half_block32_causal_v2.py`
- same math, but adds cross-M prefetch on top of a deeper queue
- score/PV handoffs use depth 3, P handoff uses depth 4, and delayed `p @ v` reads `ni - 3`
- prefetches the next same-core M tile's first two score/softmax/P tokens during the current M drain
- rowmax/rowsum prefetch state is handed through GM workspaces before the next M tile becomes current
- useful for understanding why cross-M prefetch needs explicit row-state ownership

### `flash_attn_full_pj_half_block32_causal_v4.py`
- first grouped-N experiment with `GROUP_N=4`
- groups up to four N tiles into one softmax/PV step, with `[TILE_M, GROUP_N*TILE_N]` score/P workspaces
- processes vector rows in `ROW_CHUNK=32` chunks to keep UB pressure bounded
- useful as a traffic-study artifact, not as the fastest current implementation
- the checked-in version avoids the rejected wide `[ROW_CHUNK, 512]` UB score/P path

### `flash_attn_full_pj_half_block32_causal_v5.py`
- production grouped path before v6
- keeps `GROUP_N=4`, `GROUP_LOOKAHEAD=3`, and `GROUP_STAGE_SLOTS=5`
- stores grouped score/P as `[slot, group, TILE_M, TILE_N]` instead of one wide 512-column plane
- pairs two QK matmuls under one `qk_mutex.lock()/.ready()`
- preloads V into an L1 ring (`V_PRELOAD_SLOTS=8`) before `p_mutex.wait()`
- replaces `QBuff` expdiff with explicit slot-indexed UB storage because the grouped pipeline needs more live slots than QBuff can hold
- still restarts the grouped pipeline for every M tile, so every M tile pays fill and drain

### `flash_attn_full_pj_half_block32_causal_v6.py`
- same grouped math and per-group operation order as v5
- flattens this core's `(M tile, group)` work into one continuous stream
- fills the grouped pipeline once at the first group and drains it once at the last group
- keeps `accum_ub` single-buffered, but puts loop-carried rowmax/rowsum into an `MTILE_SLOTS` ring
- uses separate cube-side and vec-side cursors because the side splitter does not safely infer those loop-carried scalar dependencies
- current fastest block32-causal path in the catalog, but also the hardest one to modify safely

## Choosing A Variant

- Start from the baseline when validating the formula, block32 mask, tails, or public outputs.
- Use v3 to study round-robin/snake M scheduling and Q hoisting without cross-M prefetch.
- Use v2 only when the question is specifically about cross-M prefetch and row-state handoff.
- Use v4 to study grouped-N traffic, not as the production target.
- Use v5 when you want the grouped pipeline with simpler per-M control flow.
- Use v6 when you need the fastest current grouped path and can preserve the dual-cursor stream structure.

## Pitfalls

- The block32 rule is not the same as left-up causal masking. It is
  `floor(k_pos / 32) <= floor(q_pos / 32)`.
- `active_tiles_n = Min(tiles_n, lmt + 1)` is still valid because a future
  `128x128` tile is fully invalid under the block32 rule.
- `row_sum` is updated from float probability values. The half cast belongs only
  to the delayed `p @ v` value path.
- Workspace ring depth must be strictly larger than lookahead. The simulator can
  miss real hardware aliasing when mutex depth is under-provisioned.
- In v6, do not collapse the private cube and vec cursors into one shared cursor.
  That leaks loop-carried scalar dependencies across the cube/vec side split.
