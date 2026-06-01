# a2 Cube-to-Vec-to-Cube-to-Vec Pattern (Triple Bridge, Delayed Numerator Accumulation)

Read this file when writing an a2 (`easyasc.a2`, device `b3`) kernel with:
- one cube stage that produces a score tile
- vec logic that updates running row state and emits a delayed cube input
- a later cube stage that consumes that delayed tile
- a final vec stage that accumulates the delayed cube output

Typical target formula:
- `score_j = q.float() @ k_j.float().t() * scale`
- `curr_m = maximum(prev_m, rowmax(score_j))`
- `expdiff_j = exp(prev_m - curr_m)`
- `p_j = exp(score_j - curr_m).half()`
- `pv_j = p_j.float() @ v_j.float()`
- `out = out * expdiff_j + pv_j`

This is **not** normalized online softmax.
It keeps running max and a rescaled numerator only.
There is no running sum or final divide.
If you need running `row_sum` and a final `out / row_sum`, switch to
`agent/references/patterns/a2-cube-vec-cube-vec-softmax.md`.

## Why this needs its own a2 pattern

This topology combines all a2 bridge constraints in one kernel:
- cube -> vec cannot use `l0c_to_ub`
- vec -> cube cannot use `ub_to_l1_*`
- the delayed cube output must return to vec for the final accumulation

So the stable data path is:

`GM(q,k,v) -> L1 -> L0 -> L0C(score) -> GM(score_ws) -> UB(score)`
`-> GM(p_ws) -> L1 -> L0 -> L0C(pv) -> GM(pv_ws) -> UB(pv) -> UB(accum) -> GM(out)`

Use explicit workspaces instead of pretending this can stay on chip end-to-end.

## Workspaces and ownership edges

Use three GM workspaces:

1. `score_ws`
   - dtype: `float`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: `L0C(score)` -> `UB(score)`

2. `p_ws`
   - dtype: `half`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: `UB(p_j)` -> `L1(p_j)`

3. `pv_ws`
   - dtype: `float`
   - shape: `[GetCubeNum(), 2, TILE_M, D]`
   - purpose: `L0C(pv_j)` -> `UB(pv_j)`

Ownership edges:
- stage 1 cube -> vec: `CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`
- stage 1 vec -> stage 2 cube: `VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`
- stage 2 cube -> stage 3 vec: `CvMutex(2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`

## Stable schedule

Use one-tile lookahead:

```python
for ni in range(0, tiles_n + 1):
    if ni < tiles_n:
        # stage 1: produce tile j = ni
    if ni > 0:
        # stage 2 + stage 3: consume tile j = ni - 1
```

This gives:
- warmup: first iteration only produces
- steady state: produce `j` while consuming `j - 1`
- drain: final iteration only consumes the last delayed tile

## Shared `L0C` rule

Reuse one physical `L0C` family across the two cube stages.

Why this is the stable a2 choice here:
- stage 1 writes a full float `[TILE_M, TILE_N]` score tile
- stage 2 writes a full float `[TILE_M, D]` `pv_j` tile with the same validated `D == 128`
- a2 only has `128 KB` `L0C`, so a second full float family would be a misleading design target

Stable ownership story:
- keep one `l0c = DBuff(DT.float, [TILE_M, TILE_N], Position.L0C)`
- let stage 1 publish `score_ws` before stage 2 reuses that slot
- let stage 2 publish `pv_ws` before the next stage-1 reuse
- advance one shared `l0c_cnt`

This is a capacity-driven exception, not a general license to merge unrelated counters.
Only the physical `L0C` family is shared. Other stage-owned lifetimes stay separate.

## Counter layout

Keep these lifetimes separate:
- `l1qk_cnt`: stage-1 `q/k` loads
- `l1pv_cnt`: stage-2 `p/v` loads
- `l0c_cnt`: shared physical `L0C` family across the two cube stages
- `stage1_cnt`: delayed slot rhythm for `score_ws`, `p_ws`, and `expdiff`
- `stage2_cnt`: delayed slot rhythm for `p_ws` consumption and `pv_ws`

Do not hide the delayed accumulator lifetime behind `stage1_cnt`.

## Vec-resident persistent state

Keep these values in per-subblock UB across the whole inner loop:
- running row max: `[HALF_M, 1]`
- delayed `expdiff` slots: `DBuff(DT.float, [HALF_M, 1], Position.UB)`
- final numerator accumulation: `[HALF_M, D]`

Use `GetSubBlockIdx()` so each vec lane owns only its own `HALF_M` rows.

## Critical scalar-state rule on a2

Do **not** copy `[HALF_M, 1]` scalar-format state with `ub_to_ub`.

Reason:
- `ub_to_ub` infers burst length in units of `C0` blocks
- for `[64, 1]` float views, that means copying 8 elements per row
- this silently miscopies row-scalar state such as `prev_m`

Stable fix:
- keep scalar state in `[HALF_M, 1]`
- copy it with a vec binary op that respects the `[M,1]` stride model, for example:

```python
dup(ub_zero_s, 0.0)
add(expdiff_buf[slot], ub_rmax_s, ub_zero_s)
```

Then update or transform that copied buffer with more vec ops.

## Delayed `expdiff` handling

`expdiff_j` belongs to the delayed consumer lifetime, not only to stage 1.

Stable pattern:
1. stage 1 copies `prev_m` into the delayed `expdiff` slot
2. stage 1 updates running max
3. stage 1 overwrites the delayed slot with `exp(prev_m - curr_m)`
4. stage 3 later reads that same slot and broadcasts it before scaling `accum`

Use `stage1_cnt` parity for the write slot and `stage2_cnt` parity for the read slot.

## Final vec accumulation

After loading `pv_j` back into UB:
1. `brcb` the delayed `expdiff` slot to `[HALF_M, 8]`
2. scale `accum[:, 0:64]`
3. scale `accum[:, 64:128]`
4. `add(accum, accum, pv_j)`

Why sliced scaling is required:
- `accum` is wide (`[HALF_M, 128]`)
- `expdiff` broadcast is narrow (`[HALF_M, 8]`)
- follow the same sliced-row rule used for row-max subtraction

## Validation target

Keep the first validated contract narrow:
- `D == 128`
- `S1 % 128 == 0`
- `S2 % 128 == 0`
- input `q/k/v` are `float16`
- output is `float32`

Suggested cases:
1. `(1, 1, 256, 512, 128)`
2. `(1, 3, 256, 512, 128)`
3. `(1, 3, 2048, 4096, 128)`

## Files to study

- `agent/example/kernels/a2/flash_attn_score_iter.py`
- `agent/example/kernels/a2/flash_attn_score_pv.py`
- `agent/example/kernels/a2/flash_attn_unnorm.py`
- `agent/references/patterns/a2-cube-vec.md`
- `agent/references/patterns/a2-cube-vec-cube.md`
- `agent/references/constraints/a2-device.md`
- `agent/references/constraints/vec-reduction-a2.md`
- `agent/references/constraints/vec-stride.md`
