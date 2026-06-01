# Online Softmax Tail Handling on A2 Triple-Bridge Kernels

Read this file when debugging or extending an a2 (`easyasc.a2`, device `b3`)
normalized online softmax kernel with delayed `p` / `pv` stages and a non-aligned
`S2` tail or `S1` tail.

Typical targets:
- `agent/example/kernels/a2/flash_attn_full.py`
- `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py`

Do not use this file as the first reference for generic tail bugs.
For generic GM-boundary tail rules, read `agent/references/constraints/tail-safety.md` first.
This file only covers the extra rule that appears once the kernel has:
- running `row_max`
- running `row_sum`
- delayed `expdiff`
- delayed `p @ v`

## Goal

Handle a non-aligned `S2` tail or `S1` tail without breaking the normalized
online softmax math.

The two axes are **not** symmetric:
- `S2` tail means invalid **columns** inside otherwise valid score rows
- `S1` tail means invalid **rows** inside an otherwise full local score tile

The stable rules are therefore different:
- `S2` still uses `valid_n` at GM boundaries, but also needs score-domain
  `-inf` masking before `rowmax`
- `S1` uses `valid_m` at GM boundaries, then masks only the local invalid rows
  after `score - rowmax` and before `exp`

## 1. Why GM-boundary slicing alone is not enough

The generic tail rule still applies:
- local tensors stay full-tile sized
- only GM loads/stores use `valid_n`

That prevents out-of-bounds reads, but it is **not** enough for online softmax.

If the last `k` / `v` tile is loaded with `valid_n < TILE_N`, the padded columns
look like zeros in the staged full tile.
That creates a second problem:
- `rowmax(score_j)` can see the padded columns
- `curr_m = maximum(prev_m, rowmax(score_j))` can become too large
- `expdiff_j = exp(prev_m - curr_m)` then rescales previous accumulated state incorrectly
- `row_sum` and `out` are both corrupted even if later `p_j` is masked to zero

So for normalized online softmax:
- padded tail columns must behave like `-inf` **before** `rowmax`
- the same padded columns then naturally become `0` after `exp`

## 2. Do not start from a `p`-domain-only fix

A `p`-domain-only tail mask is insufficient for normalized online softmax.

It can fix:
- delayed `p @ v`
- any later use of `p_j`

It cannot fix:
- `rowmax(score_j)`
- `curr_m`
- delayed `expdiff_j`
- `row_sum`

If the kernel has running `row_max` / `row_sum`, fix the score tile first.

## 3. Stable semantic rule for invalid tail columns

For the last `S2` tile:
- before `cmax`: invalid columns must look like `-inf`
- after `exp`: invalid columns must behave like `0`

This rule preserves the exact reference update:
- `curr_m = maximum(prev_m, rowmax(score_j_valid_only))`
- `p_j = exp(score_j_valid_only - curr_m)`
- `row_sum = row_sum * expdiff_j + p_j.sum(-1)`

You do **not** need a separate `p`-domain tail mask if the score tile already
uses this `-inf` rule and the delayed `v` load also uses `valid_n`.

## 4. Stable a2 implementation shape

For the current validated flash-attention kernels:
- `TILE_N = 128`
- score is processed in vec as `[HALF_M, TILE_N]`
- the practical split is two `[HALF_M, 64]` halves

That gives a stable rule:
- left half handles columns `[0:64)`
- right half handles columns `[64:128)`

Tail cases:
- `valid_n == 128`
  - both halves fully valid
- `64 < valid_n < 128`
  - left half fully valid
  - right half needs a suffix invalid mask
- `valid_n == 64`
  - left half fully valid
  - right half fully invalid
- `0 < valid_n < 64`
  - left half needs a suffix invalid mask
  - right half fully invalid
- `valid_n == 0`
  - both halves fully invalid

## 5. Why vec mask + finite negative sentinel is the simplest score-domain fix

For `float` vec ops on a2:
- the active mask prefix length is `64`
- the same `64`-lane mask prefix is reused for each repeat

That matches a `[HALF_M, 64]` score half perfectly:
- one row uses one repeat
- each row wants the same tail-column mask

So the stable suffix invalidation pattern is:
1. compute a 64-bit suffix-invalid mask
2. `set_mask(0, low_mask)`
3. `dup(score_half, neg_large)`
4. `reset_mask()`

This is usually simpler than materializing a `[HALF_M, 64]` flag tensor and then
doing `select(...)` on the score half. The intent is still `-inf` behavior, but
the concrete fill should stay finite on hardware paths.

Read next for exact vec mask semantics:
- `agent/references/constraints/mask.md`

## 6. Bit order and mask meaning

Instruction semantics:
- `low` writes `mask[0:64]`
- bit `0` maps to the lowest logical lane in that prefix
- bit `63` maps to the highest logical lane in that prefix

Stub call note:
- the current a2 stub is called as `set_mask(mask_high, mask_low)`
- so a low-only score-half mask is written with `set_mask(0, low_mask)`

So for a suffix invalid mask on one 64-column score half:
- columns `[0:valid_cols)` should be `0`
- columns `[valid_cols:64)` should be `1`

Examples:
- `valid_cols = 64` -> no invalid bits
- `valid_cols = 63` -> only bit `63` is `1`
- `valid_cols = 10` -> bits `[10:63]` are `1`
- `valid_cols = 0` -> all bits are `1`

Validated repository tests:
- `testcases/simulator/micro/test_simulator_v2_muladddst_mask.py`
- `testcases/simulator/micro/test_simulator_v2_vec_ops_extended.py`

## 7. Stable scalar-mask construction trick

The obvious unsigned construction:
- build a huge `uint64` value like `18446744073709550592`

can trip the simulator's scalar cast path because the current runtime first
creates a Python/Torch signed integer before converting to `uint64`.

The stable workaround is:
- start from signed `-1`
- left-shift it `valid_cols` times
- then assign the signed result into a `uint64` `Var`

For one 64-lane score half this builds the same suffix-invalid bit pattern:

```python
@func()
def build_suffix_invalid_mask(valid_cols: Var, out_mask: Var):
    signed_mask = Var(-1, DT.int64)
    two_i64 = Var(2, DT.int64)
    for _ in range(0, valid_cols):
        signed_mask <<= signed_mask * two_i64
    out_mask <<= signed_mask
```

Why this works:
- `-1 << valid_cols` equals the desired suffix-invalid mask in two's-complement
- the intermediate signed values stay representable in `int64`
- the final `uint64` assignment preserves the bit pattern

## 8. Minimal integration recipe

For a normalized online softmax stage-1 score tile:

1. load `k` with `valid_n`
2. stage the full `[HALF_M, TILE_N]` score tile
3. apply score-tail masking only when `valid_n < TILE_N`
4. only then run:
   - `vmax(...)`
   - `cmax(...)`
   - delayed `expdiff`
   - `exp(...)`
   - `cadd(...)`
5. stage delayed `p`
6. later load `v` with the recomputed previous-tile `valid_n`

The score-tail masking point should be:
- after scale is applied
- before any `rowmax` / `cmax`

## 9. Minimal validation set

Do not validate only aligned cases.

For `TILE_N = 128`, keep at least:
- one aligned baseline: `S2 % 128 == 0`
- one small left-half tail: `S2 % 128 == 10`
- one first-right-half case: `S2 % 128 == 65`
- one mid-right-half case: `S2 % 128 == 96`
- one last-column case: `S2 % 128 == 127`

For `flash_attn_full_pj_hif8.py`, the validated runnable regression lives in the kernel self-check:
- `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`

## 10. Why `S1` tail is a different problem

Do not try to solve `S1` tail by reusing the `S2` column-tail mental model.

For `S1` tail:
- the invalid region is a suffix of **rows**, not columns
- `q` must use `valid_m` at the GM boundary
- final `out` must also use `valid_m` at the GM boundary
- the vec side still sees a fixed physical `[HALF_M, TILE_N]` score tile

Current validated a2 flash-attention shape:
- the two vec subblocks read fixed physical row ranges
  - subblock `0` reads rows `[0:64)`
  - subblock `1` reads rows `[64:128)`
- this is **not** the a5-style `CeilDiv(valid_m, 2)` compact half split

So the stable local quantity is:
- `local_valid_m = clamp(valid_m - sb_row, 0, HALF_M)`

where:
- `valid_m` is the tile-level valid query-row count
- `sb_row` is the fixed physical subblock row origin (`0` or `64`)

## 11. Stable `S1` implementation rule

For a normalized online softmax stage-1 score tile with `S1` tail:

1. load `q` with `valid_m`
2. rely on the current `gm_to_l1_nd2nz` zero-fill behavior for the local tail rows
3. run the normal score tile, `rowmax`, `curr_m`, and `expdiff` flow on the
   full local score tile
4. after `score_j - curr_m`, but before `exp(score_j)`, overwrite the local
   invalid row suffix with a sufficiently negative finite sentinel
5. keep the delayed `p` / `pv` path full-tile sized
6. write back only `local_valid_m` rows to GM

Why this point is stable:
- masking invalid rows **before** `cmax` can create invalid-row sentinel `rowmax`
  and unstable invalid-row subtraction behavior analogous to `-inf - (-inf)`
- masking them **after** subtracting `curr_m` preserves the valid-row online
  softmax math
- the invalid local rows then become `0` after `exp`, so they contribute
  nothing to delayed `p @ v`

Current repository tolerance:
- invalid `S1` tail rows may still become `NaN` after the final `out / row_sum`
  on local UB rows
- this is acceptable because those rows are not written back to GM

## 12. Minimal `S1` validation set

Do not validate only one row-tail case.

For `TILE_M = 128`, keep at least:
- one aligned baseline: `S1 % 128 == 0`
- one one-row tail: `S1 % 128 == 1`
- one last-row-in-first-half case: `S1 % 128 == 63`
- one exact half case: `S1 % 128 == 64`
- one first-row-in-second-half case: `S1 % 128 == 65`
- one last-row case: `S1 % 128 == 127`
- one larger shape beyond two tiles, for example `S1 == 257`
- one multi-head shape

Keep `S2` aligned while validating the new `S1` path first, so failures are
easier to attribute.

## 13. Causal diagonal tiles on a2

Read this when extending the same normalized online-softmax pipeline from plain
tail handling to left-up causal masking (`k_pos <= q_pos`).

The stable tile classification is:
- `nt < lmt`: the tile is fully valid
- `nt == lmt`: the tile is diagonal and contains mixed valid/invalid columns
- `nt > lmt`: the tile is fully invalid and should be skipped

For the current validated causal kernel, the stable scheduling rule is:
- clamp the stage-1/stage-2 loop to `active_tiles_n = Min(tiles_n, lmt + 1)`
- still keep the outer `n_loops + 1` style drain shape by iterating to `active_tiles_n + 1`
- this preserves the delayed `p @ v` flush while removing future fully-invalid tiles

The diagonal tile is **not** a plain `valid_n` tail:
- invalid columns vary by row
- the stable local quantity is `valid_cols = sb_row + row + 1`
- `sb_row` is the fixed subblock row origin (`0` or `64`)

Stable implementation rule for the diagonal tile:
1. load and scale the full `[HALF_M, TILE_N]` score tile
2. prebuild reusable packed-bit masks once per subblock before the main tile loop:
   - `causal_mask_left: Tensor(DT.uint8, [HALF_M, HALF_N // 8], Position.UB)`
   - `causal_mask_right: Tensor(DT.uint8, [HALF_M, HALF_N // 8], Position.UB)`
3. initialize one reusable integer column-index row for `[0, 1, ..., 63]`; the current validated kernel writes two `int32` entries at a time through an `int64` reinterpret to keep `SetValueTo(...)` count low
4. use a Python-unrolled row loop (`py_range(HALF_M)`) only for the per-row threshold, and synthesize packed mask bytes with `compare_scalar(...)`:
   - if `sb_row == 0`, build only `causal_mask_left[row]` with threshold `row + 1`
   - if `sb_row == 64`, fill `causal_mask_left` to all ones and build only `causal_mask_right[row]` with threshold `row + 1`
5. apply the packed masks with `select(..., SelectMode.TENSOR_SCALAR)` before `cmax` / `rowmax`
6. if the same tile is also the final `S2` tail tile, apply the diagonal causal mask first and `valid_n` tail masking second

Why this is the stable path:
- it matches the current hardware / simulator rule that `compare_scalar(...)` and `select(...)` use packed-bit `uint8` control
- it keeps the control path cheap by building the causal masks once per subblock instead of reconstructing them inside every diagonal-tile visit
- it avoids the large simulator overhead of byte-by-byte `SetValueTo(...)` loops for mask construction
- it avoids trying to repair causal semantics later in the `p` or `pv` path
- it preserves the exact online-softmax updates for `row_max`, `expdiff`, and `row_sum`

Minimal causal validation set:
- one `S1 == S2` aligned case
- one `S1 == S2` unaligned case
- one `S1 < S2` case
- one `S1 > S2` case
- one multi-head case

Validated runnable example:
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py`

## 14. Files to study

- `agent/example/kernels/a2/flash_attn_full_pj_hif8.py`
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py`
- `testcases/simulator/micro/test_simulator_v2_muladddst_mask.py`
- `testcases/simulator/micro/test_simulator_v2_vec_ops_extended.py`
- `agent/references/constraints/tail-safety.md`
- `agent/references/constraints/mask.md`
- `agent/references/patterns/a2-cube-vec-cube-vec-softmax.md`
