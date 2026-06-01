# a2 Cube-to-Vec-to-Cube-to-Vec Pattern (Triple Bridge, Normalized Online Softmax)

Read this file when writing an a2 (`easyasc.a2`, device `b3`) kernel with:
- one cube stage that produces a score tile
- vec logic that updates running row max and running row sum
- a later cube stage that consumes the delayed probability tile
- a final vec stage that accumulates the delayed cube output
- one final vec-only divide by the accumulated row sum

Typical target formula:
- `score_j = q.float() @ k_j.float().t() * scale`
- `curr_m = maximum(prev_m, rowmax(score_j))`
- `expdiff_j = exp(prev_m - curr_m)`
- `p_j = exp(score_j - curr_m)`
- `row_sum = row_sum * expdiff_j + p_j.sum(-1)`
- `pv_j = p_j.half().float() @ v_j.float()`
- `out = out * expdiff_j + pv_j`
- `out = out / row_sum`

This is the normalized counterpart to `a2-cube-vec-cube-vec.md`.
Use that older pattern only when the kernel stops at the unnormalized numerator.

## One-page route for the common case

If this file matches your contract, do **not** preload all of:
- `agent/references/constraints/reduction.md`
- `agent/references/constraints/vec-reduction-a2.md`
- `agent/references/constraints/vec-stride.md`
- `agent/references/constraints/online-softmax-tail.md`

This page now owns the common normalized-online-softmax authoring rules.
Open the smaller constraint pages only when a specific failure mode still remains unclear after this file.

## Why this needs its own a2 pattern

The a2 hardware constraints are the same as the unnormalized case:
- cube -> vec cannot use `l0c_to_ub`
- vec -> cube cannot use `ub_to_l1_*`
- delayed cube output must come back to vec for final accumulation

But normalized online softmax adds two stability-sensitive requirements:
- running `row_sum` must be updated from the float `exp(...)` tile before any cast to half
- the final divide must happen only once, after all delayed numerator tiles have been accumulated

So the stable a2 flow is:

`GM(q,k,v) -> L1 -> L0 -> L0C(score) -> GM(score_ws) -> UB(score)`
`-> vec(max, expdiff, exp, row_sum, cast p) -> GM(p_ws) -> L1 -> L0 -> L0C(pv)`
`-> GM(pv_ws) -> UB(pv) -> UB(accum) -> final UB divide by row_sum -> GM(out)`

## Workspaces and ownership edges

Use the same three GM workspaces as the unnormalized pattern:

1. `score_ws`
   - dtype: `float`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: `L0C(score)` -> `UB(score)`

2. `p_ws`
   - dtype: `half`
   - shape: `[GetCubeNum(), 2, TILE_M, TILE_N]`
   - purpose: `UB(p_j.half())` -> `L1(p_j)`

3. `pv_ws`
   - dtype: `float`
   - shape: `[GetCubeNum(), 2, TILE_M, D]`
   - purpose: `L0C(pv_j)` -> `UB(pv_j)`

Ownership edges:
- stage 1 cube -> vec: `CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`
- stage 1 vec -> stage 2 cube: `VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`
- stage 2 cube -> stage 3 vec: `CvMutex(2, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)`

## Stable schedule

Use the same one-tile lookahead loop as the unnormalized pattern:

```python
for ni in range(0, tiles_n + 1):
    if ni < tiles_n:
        # stage 1: produce tile j = ni
    if ni > 0:
        # stage 2 + stage 3: consume tile j = ni - 1
```

That gives:
- warmup: first iteration only produces
- steady state: produce `j` while consuming `j - 1`
- drain: final iteration only consumes the last delayed tile

## Shared `L0C` rule

Reuse one physical `L0C` family across the two cube stages.

This is the same capacity-driven choice as the unnormalized pattern:
- stage 1 needs float `[TILE_M, TILE_N]`
- stage 2 needs float `[TILE_M, D]` with validated `D == 128`
- a2 still has only `128 KB` `L0C`

Keep one shared `l0c_cnt`, but do not merge unrelated counters just because `L0C` is shared.

## Counter layout

Keep these lifetimes separate:
- `l1qk_cnt`: stage-1 `q/k` loads
- `l1pv_cnt`: stage-2 `p/v` loads
- `l0c_cnt`: shared physical `L0C` family across the two cube stages
- `stage1_cnt`: delayed slot rhythm for `score_ws`, `p_ws`, and `expdiff`
- `stage2_cnt`: delayed slot rhythm for `p_ws` consumption and `pv_ws`

Running `row_sum` does not need its own delayed counter.
It stays vec-resident for the whole inner loop and updates immediately in stage 1.

## Vec-resident persistent state

Keep these values in per-subblock UB across the whole inner loop:
- running row max: `[HALF_M, 1]`
- running row sum: `[HALF_M, 1]`
- delayed `expdiff` slots: `DBuff(DT.float, [HALF_M, 1], Position.UB)`
- final numerator accumulation: `[HALF_M, D]`

Use `GetSubBlockIdx()` so each vec lane owns only its own `HALF_M` rows.

## Stable stage-1 update order

The normalized online update order matters:

1. compute `rowmax(score_j)` in `[HALF_M, 1]`
2. snapshot `prev_m` into the delayed `expdiff` slot with `add(..., zero)`
3. update `running_max = maximum(running_max, tile_max)`
4. turn the delayed slot into `exp(prev_m - curr_m)`
5. broadcast `running_max` and subtract from the score tile
6. compute the float probability tile `p_j = exp(score_j - curr_m)`
7. reduce `sum_j` from that float tile with `add` + `cadd`
8. update `running_sum = running_sum * expdiff_j + sum_j` in `[HALF_M, 1]`
9. cast `p_j` to `half` only now, because stage 2 wants the exact `p_j.half().float()` contract

Do not move the row-sum update after the cast.
That would silently change the reference contract.

## Vec rules you usually need without extra docs

For the common `TILE_N = 128`, `D = 128` path, the usual extra questions are already answered here:

1. keep `running_max`, `running_sum`, and delayed `expdiff` in scalar format `[HALF_M, 1]`
2. snapshot scalar state with `add(dst, src, zero)`, not `ub_to_ub`
3. `cmax` / `cadd` output dense scalars, so broadcast them with:
   - `brcb(dst, src, dst_blk_stride=1, dst_rep_stride=8)`
4. when a wide `[HALF_M, 128]` buffer is paired with a narrow `[HALF_M, 8]` broadcast row, operate on:
   - `buf[:, 0:64]`
   - `buf[:, 64:128]`
   rather than on the full 128-column view in one vec call
5. update `running_sum` from the float `p_j` tile before any cast to `half` or `hif8`
6. for non-aligned `S2`, invalidate score columns before `cmax` with a sufficiently negative finite sentinel; `valid_n` on the GM load alone is not enough

These six rules cover the usual reasons people would otherwise open the separate reduction, vec-reduction, vec-stride, and tail files.

## Critical scalar-state rule on a2

Do **not** copy `[HALF_M, 1]` scalar-format state with `ub_to_ub`.

That applies to both:
- `prev_m`
- any temporary scalar snapshot you might be tempted to use for `row_sum`

Use `add(dst, src, zero)` for scalar-format copies, and keep both `running_max`
and `running_sum` in `[M,1]` format until you explicitly need a broadcast.

## Final vec accumulation and divide

Stage 3 still matches the unnormalized pattern:
1. load delayed `pv_j` back into UB
2. `brcb` the delayed `expdiff` slot to `[HALF_M, 8]`
3. scale the two 64-column halves of `accum`
4. `add(accum, accum, pv_j)`

After the inner loop finishes:
1. `brcb` the final `running_sum` to `[HALF_M, 8]`
2. `div(accum[:, 0:64], accum[:, 0:64], row_sum_broadcast)`
3. `div(accum[:, 64:128], accum[:, 64:128], row_sum_broadcast)`
4. write the normalized result to GM

Why the divide happens at the end:
- `accum` must finish all delayed `pv_j` contributions first
- `row_sum` is the denominator for the whole streamed softmax, not one tile

## Extending the pattern to non-aligned `S2`

The initial validated contract for this pattern kept `S2 % 128 == 0` so the
first implementation could ignore score-tail masking.

When `S2` is not aligned, do **not** stop at GM-boundary `valid_n` slicing.
For normalized online softmax, padded score columns can still corrupt:
- `rowmax(score_j)`
- `curr_m`
- delayed `expdiff`
- `row_sum`

Stable rule:
- load `k` / `v` through `valid_n`
- keep local score buffers full-sized
- before `cmax`, force invalid score columns to behave like `-inf`
- when materializing that mask, use a sufficiently large finite negative fill value instead of literal `-inf`
- after `exp`, those same columns naturally behave like `0`

For the current `TILE_N = 128` layout, the simplest a2 implementation is:
- split the score tile into two `[HALF_M, 64]` halves
- use vec mask + finite-negative `dup(...)` on the affected half
- recompute `prev_valid_n` for the delayed `v` load in stage 2

Read next for the exact rule and mask-construction trick:
- `agent/references/constraints/online-softmax-tail.md`

## Validation target

Keep the first validated contract narrow:
- `D == 128`
- `S1 % 128 == 0`
- `S2 % 128 == 0`
- input `q/k/v` are `float16`
- output is `float32`

Suggested cases:
1. `(1, 3, 256, 256, 128)` for the smallest two-tile online update
2. `(1, 1, 256, 512, 128)`
3. `(1, 3, 256, 512, 128)`
4. `(1, 3, 2048, 4096, 128)`

For non-aligned `S2` extensions, add at least:
1. one aligned baseline: `S2 % 128 == 0`
2. one left-half tail: `S2 % 128 == 10`
3. one cross-boundary case: `S2 % 128 == 65`
4. one mid-right-half case: `S2 % 128 == 96`
5. one last-column case: `S2 % 128 == 127`

## Files to study / deeper fallbacks

- `agent/example/kernels/a2/flash_attn_full.py`
- `agent/example/kernels/a2/flash_attn_unnorm.py`
- `agent/example/kernels/a2/flash_attn_score_pv.py`
- `agent/references/patterns/a2-cube-vec-cube-vec.md`
- `agent/references/constraints/reduction.md` — fallback only when the online update order is still unclear
- `agent/references/constraints/vec-reduction-a2.md` — fallback only when the `cmax/cadd -> brcb` detail is still unclear
- `agent/references/constraints/vec-stride.md` — fallback only when a sliced wide/narrow vec op is still unclear
- `agent/references/constraints/online-softmax-tail.md` — fallback only when the non-aligned `S2` mask construction itself is the question
