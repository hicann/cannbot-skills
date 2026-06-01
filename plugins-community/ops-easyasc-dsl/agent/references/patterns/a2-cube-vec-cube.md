# a2 Cube-to-Vec-to-Cube Pattern (Double GM Bridge, One-Tile Lookahead)

Read this file when writing an a2 (`easyasc.a2`, device `b3`) kernel with:
- one cube stage that produces a tile
- vec logic that transforms that tile
- a later cube stage that consumes the vec result

Do not use this file for a5 kernels. The a5 path is materially different because it can
publish vec output directly to `L1`.

## When to use

- the formula is structurally `cube -> vec -> cube`
- the vec result must be consumed by a later cube matmul
- the later cube stage naturally runs one iteration behind the producer stage
- the vec output is large enough that pretending it stays purely on chip would be misleading

Typical example:
- `score_j = q @ k_j^T`
- vec computes `p_j = exp(score_j - running_max).half()`
- delayed cube stage computes `pv_j = p_j.float() @ v_j.float()`

## Why a2 needs a special pattern

Two a2 hardware/software constraints dominate the design:

1. `l0c_to_ub` is unavailable.
   Cube output cannot go directly from `L0C` to `UB`.

2. `ub_to_l1_nd2nz` / `ub_to_l1_nz` are a5-only.
   Vec output cannot go directly from `UB` to `L1`.

Practical consequence:
- cube -> vec must bridge through GM workspace
- vec -> cube must also bridge through GM workspace

So the real a2 flow is:

`GM(q,k) -> L1 -> L0 -> L0C -> GM(score_ws) -> UB -> vec -> GM(p_ws) -> L1 -> L0 -> L0C -> GM(pv)`

This is the core difference from a5.

## Stable schedule: warmup, steady state, drain

The clean control structure is a one-tile lookahead loop:

```python
for ni in range(0, tiles_n + 1):
    if ni < tiles_n:
        # stage 1: produce current tile p_j
    if ni > 0:
        # stage 2: consume previous tile p_{j-1}
```

Meaning:
- `ni < tiles_n`: produce tile `j = ni`
- `ni > 0`: consume tile `j = ni - 1`

This creates:
- warmup: first iteration only produces
- steady state: middle iterations produce `j` while consuming `j-1`
- drain: last iteration only consumes the final tile

Do not force both stages into the same tile index inside one iteration.
The delayed consumer is the point of the pattern.

## Workspace layout

Use two separate GM workspaces:

1. `score_ws`
   - dtype: `float`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: bridge `L0C(score)` -> `UB`

2. `p_ws`
   - dtype: `half`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: bridge `UB(p_j)` -> `L1`

Why two workspaces:
- stage-1 score is naturally `float`
- stage-2 cube input should consume `half` if the target contract is `p_j.half().float() @ v_j.float()`
- keeping them separate makes dtype intent explicit and avoids hidden casts

## Buffer ownership and reuse

### 1. Reuse one `L0C` family across both cube stages

On a2, `TILE_M = TILE_N = 128` with float accumulation already fills the entire `128 KB` `L0C`.
That leaves no room for a second full-size `L0C` family.

Stable rule for this specific pattern:
- reuse one physical `l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)`
- let both cube stages write into that same family
- advance one shared `l0c_cnt`

Why this is safe here:
- stage 1 and stage 2 do not need `L0C` simultaneously
- stage 1 publishes `L0C -> score_ws` before stage 2 reuses the slot
- stage 2 publishes `L0C -> output` before the next stage-1 reuse

Do not generalize this into "all delayed stages should share counters".
This is a targeted capacity-driven exception for one serially reused `L0C` family.

### 2. Keep other lifetimes separate

Even though `l0c_cnt` is shared, other stage-owned lifetimes should stay separate:
- `l1q/l1k` and `l1p/l1v` should not share one counter
- delayed slot ownership should use `stage1_cnt` and `stage2_cnt`

Recommended split:
- `l1qk_cnt`: stage-1 operand loads
- `l1pv_cnt`: stage-2 operand loads
- `l0c_cnt`: shared physical `L0C` family
- `stage1_cnt`: `score_ws` / `p_ws` producer slot rhythm
- `stage2_cnt`: delayed consumer slot rhythm

### 3. If a delayed consumer reuses a producer operand, match buffer depth to the overlap

Sometimes the delayed cube stage needs not only the vec result, but also one of the
original stage-1 operands again.

Concrete example from dense attention backward:
- stage 1 loads `k_j` and computes `qk_j = q @ k_j^T`
- vec computes `dqk_j`
- delayed cube stage later computes `gq += dqk_j @ k_j`

If you want to avoid reloading `k_j` from GM, keep that operand family on chip and reuse it
from the delayed stage.

Important overlap rule:
- for a one-tile lookahead loop, `DBuff` is often **not** enough for a reused producer operand
- while the delayed stage is still consuming tile `j`, the producer may already be starting tile `j+2`
- with only two slots, tile `j+2` can overwrite slot `j` before the delayed consumer is done

Stable rule for this case:
- promote only the reused delayed operand family to `TBuff`
- keep unrelated families such as `v` on `DBuff` if they are not reused by the delayed stage
- let the delayed consumer index that `TBuff` by its own delayed-stage lineage, not by the
  immediate producer slot

Practical outcome:
- `k` may need `TBuff`
- `v` may still stay `DBuff`
- the extra on-chip slot can be cheaper than a second GM read on every tile

This is a lifetime decision, not a micro-optimization accident.
Choose the buffer depth from the real overlap window.

## Cross-side synchronization

This pattern has two ownership edges.

### Edge 1: cube -> vec (`score`)

Use:

```python
CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
```

Reason:
- producer ends with `l0c_to_gm_nz2nd` on `FIX`
- vec consumer starts with `gm_to_ub_pad` on `MTE2`

### Edge 2: vec -> cube (`p_j`)

Use:

```python
VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
```

Reason:
- vec producer ends with `ub_to_gm_pad` on `MTE3`
- cube consumer eventually finishes the delayed use after `gm_to_l1 -> l1_to_l0 -> mmad -> writeback`
- for this pattern, conservative release is safer: free only after the cube stage finishes the tile

This conservative `dst_end_pipe=Pipe.FIX` matches the "do not release early" rule for delayed reuse.

## Two-sub-block publication rule

Each a2 cube core has 2 vec sub-blocks.
Each vec sub-block owns only `HALF_M` rows in `UB`.

So stage 1 should:
- read `HALF_M` rows from `score_ws`
- compute `p_j` for only those rows
- write those rows into the shared `p_ws` slot

Typical write pattern:

```python
sb = GetSubBlockIdx()
sb_row = Var(sb * HALF_M)
p_ws[cube_idx, slot, sb_row:sb_row + HALF_M, 0:TILE_N] <<= ub_p
```

Then stage 2 cube waits on the `VcMutex` and reads the full tile:

```python
l1p[...] <<= p_ws[cube_idx, slot, 0:TILE_M, 0:TILE_N]
```

Important simulator/runtime fact:
- cube-side `wait_vec()` completes only after both vec lanes have produced their tokens
- this makes the full-tile read safe without an extra manual barrier

## Row-max state rules

If the vec stage uses running row max across tiles:
- keep the running state in `[HALF_M, 1]` scalar format
- initialize with `dup(neg_large)` where `neg_large` is a sufficiently large finite negative sentinel
- update with `vmax(ub_rmax_s, ub_rmax_s, ub_max_s)`
- broadcast only after the scalar update using `brcb`

For `TILE_N = 128`, the stable sequence is:
1. `vmax` between the two 64-column halves
2. `cmax` to `[HALF_M, 1]`
3. `vmax` with running state in `[HALF_M, 1]`
4. `brcb` to `[HALF_M, 8]`
5. sliced `sub` on `[0:64]` and `[64:128]`
6. `exp`
7. `cast` to `half`

Do not:
- update running max in `[HALF_M, 8]` broadcast format
- subtract a narrow max buffer from an unsliced `[HALF_M, 128]` tile

## Stage ordering inside one loop iteration

For this a2 pattern, a stable order is:

1. stage 1 cube computes `score_j`
2. stage 1 vec computes `p_j` and writes `p_ws`
3. stage 2 cube consumes delayed `p_{j-1}`

In other words: "produce current tile first, then consume previous tile".

Why this order is helpful:
- the reused `L0C` family is naturally free after `score_j -> score_ws`
- the delayed cube stage can then reuse that same `L0C` family safely
- one shared `l0c_cnt` remains easy to reason about

If the delayed stage also reuses stage-1 `k_j` on chip:
- the schedule is still "produce `j`, then consume `j-1`"
- but the `k` buffer family now lives longer than the immediate `v` family
- reflect that longer lifetime in the buffer depth (`TBuff`) and in the counter choice

## Output layout rule

For flattened GM output that preserves `[B, H, tiles_n, S1, D]`, a stable write index is:

```python
out_row = Var((bh * tiles_n + tile_n) * S1 + local_row)
```

That corresponds to the physical layout:

`[(bh * n_tiles + tile_n) * S1 + row, D]`

Use this when the user wants to preserve the logical `[B, H, tile_n, S1, D]` grouping
while still flattening `BH` in the kernel contract.

## Validation checklist

For the first runnable version, keep the contract narrow and explicit:
- `S1 % 128 == 0`
- `S2 % 128 == 0`
- `D == 128`
- `scale` passed in as an explicit kernel scalar

Reference formula to compare against:

```python
for j in range(0, S2, 128):
    score_j = q.float() @ k_j.float().t() * scale
    m = maximum(m, rowmax(score_j))
    p_j = exp(score_j - m).half()
    pv_j = p_j.float() @ v_j.float()
```

Good first validation order:
1. `(B,H,S1,S2,D) = (1,1,256,512,128)`
2. multi-head small case
3. full aligned case such as `(1,3,2048,4096,128)`

## Common mistakes

- trying to use `UB -> L1` directly on a2
- allocating separate full-size `L0C` families for both cube stages
- sharing every counter just because `l0c_cnt` is shared
- forgetting the `tiles_n + 1` warmup/drain loop
- consuming tile `j` in the same iteration that is supposed to produce tile `j`
- writing only one vec sub-block's rows into `p_ws`
- releasing the vec -> cube mutex before the delayed cube stage really finishes
- documenting the kernel as "online softmax" when it only keeps running max and does not maintain running sum

## Files to study

- `agent/example/kernels/a2/flash_attn_score_pv.py`
- `agent/example/kernels/a2/flash_attn_score_iter.py`
- `agent/references/patterns/a2-cube-vec.md`
- `agent/references/constraints/a2-device.md`
- `agent/references/constraints/vec-reduction-a2.md`
- `agent/references/constraints/vec-stride.md`
