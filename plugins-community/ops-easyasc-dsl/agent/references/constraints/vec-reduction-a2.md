# Vec Reduction on a2 (cmax + brcb Pattern)

Read this file when implementing per-row reductions (max, sum) on a2 using the vec pipeline.
On a2 there are no `Reg`/`RegList`, so reductions use UB-to-UB `cmax`/`cadd` + `brcb`.

## Goal

Get per-row max (or sum) correct on a2, including the broadcast step
that is easy to forget.

## 1. The cmax output format

`cmax(dst, src)` reduces one repeat (64 float elements = 8 blocks of 8) to a **single scalar**.
The scalar is stored at `dst[rep * dst_rep_stride]` — one float element per repeat.

With the default `dst_rep_stride=1`, the scalars are packed densely:
```
dst[0] = max of row 0
dst[1] = max of row 1
...
dst[63] = max of row 63
```

This is **not** a C0 block layout. The 8-element block structure that `sub`/`vmax` expect
is not satisfied.

## 2. The bug: using cmax output directly in sub

If you pass the cmax output to `sub` with `blk_stride=0`:
- `sub` reads a C0 block (8 elements) and broadcasts it across all 8 blocks of each repeat
- But the 8 elements in that block are maxes of **8 different rows**, not 8 copies of one row's max
- Result: each row gets subtracted by the wrong max → `exp` produces huge or wrong values

**Symptom**: output values > 1.0 from `exp(score - max)` where max should be the row max.

## 3. The fix: brcb broadcast between cmax and sub

After cmax, use `brcb` to expand each scalar to fill a full C0 block:

```python
ub_max_s = Tensor(DT.float, [HALF_M, 1], Position.UB)   # cmax scalars
ub_max   = Tensor(DT.float, [HALF_M, 8], Position.UB)    # broadcast result

cmax(ub_max_s, ub_tmp)
brcb(ub_max, ub_max_s, dst_blk_stride=1, dst_rep_stride=8)
```

How brcb works:
- `repeat = infer_repeat_brcb(src) = HALF_M * 1 // 8 = 8`
- For each repeat: reads 8 scalars from `src[rep*8 : rep*8+8]`
- For each of 8 blocks: fills `dst[block_begin : block_begin + C0]` with one scalar
- With `dst_blk_stride=1, dst_rep_stride=8`: blocks are contiguous, repeats advance by 8 blocks

Result: `ub_max[n*8 : n*8+8]` all contain `max_of_row_n` for n in 0..63.

### 3a. Dense row `[1, 64]` -> broadcast `[64, 8]` also needs explicit `brcb` params

When the scalar statistics arrive as one dense row such as:

- `qkmaxbuf = Tensor(DT.float, [1, 64], Position.UB)`
- `qksumbuf = Tensor(DT.float, [1, 64], Position.UB)`

and the destination is the usual broadcast format:

- `qkmaxbrcb = Tensor(DT.float, [64, 8], Position.UB)`

do **not** rely on default `brcb(...)` parameter inference.

Validated pattern:

```python
qkmaxbuf <<= qkmax[bh:bh + 1, row0:row0 + 64]
brcb(qkmaxbrcb, qkmaxbuf, repeat=64 // 8, dst_blk_stride=1, dst_rep_stride=8)
```

Why this matters:
- the source load into `[1,64]` is fine
- the failure comes from the broadcast configuration, not from the GM -> UB read itself
- with the validated explicit parameters, row `r` is expanded to `qkmaxbrcb[r, 0:8]`

Concrete reproducer:
- `tmp/validate_row64_brcb.py`

Practical rule:
- for row-stat broadcasts on a2, treat `brcb(..., dst_blk_stride=1, dst_rep_stride=8)` as mandatory
- when the source is `[1,64]`, also pin `repeat=64 // 8` explicitly in validated kernels instead of trusting defaults

## 4. Complete row-max pattern for [HALF_M, 128] float data

```python
HALF_M = 64
HALF_N = 64

ub_data  = Tensor(DT.float, [HALF_M, 128], Position.UB)
ub_tmp   = Tensor(DT.float, [HALF_M, HALF_N], Position.UB)
ub_max_s = Tensor(DT.float, [HALF_M, 1], Position.UB)
ub_max   = Tensor(DT.float, [HALF_M, 8], Position.UB)

# Step 1: element-wise max of two 64-col halves → 64 values per row
vmax(ub_tmp, ub_data[0:HALF_M, 0:HALF_N], ub_data[0:HALF_M, HALF_N:128])

# Step 2: reduce 64 → 1 scalar per row
cmax(ub_max_s, ub_tmp)

# Step 3: broadcast each scalar to fill a C0 block (8 identical elements)
brcb(ub_max, ub_max_s, dst_blk_stride=1, dst_rep_stride=8)

# Step 4: subtract (sliced to align repeat with narrow max buf)
sub(ub_data[0:HALF_M, 0:HALF_N], ub_data[0:HALF_M, 0:HALF_N], ub_max)
sub(ub_data[0:HALF_M, HALF_N:128], ub_data[0:HALF_M, HALF_N:128], ub_max)
```

Why each step is needed:
- **vmax**: 128 columns exceed one repeat (64 elements). Must merge to 64 first.
- **cmax**: reduces 64 → 1 scalar per row. Output is dense, not block-aligned.
- **brcb**: fills C0 blocks so that `sub` with `blk_stride=0` broadcasts correctly.
- **sub with slicing**: see `agent/references/constraints/vec-stride.md` for why.

## 5. Why `[M, 8]` broadcast format fails for binary ops between two narrow buffers

After `brcb`, the result tensor has shape `[HALF_M, 8]` with `span[1]=8=C0`.
Stride inference for `[64, 8]` float gives: `blk_stride=0, rep_stride=1, repeat=8`.

With `blk_stride=0`, all 8 blocks within one repeat address the **same** 8 elements.
So each repeat touches 8 unique elements, and 8 repeats touch 8×8=64 elements.
But the buffer contains 64×8=512 elements. The remaining 448 are **never reached**.

This means `vmax(buf_a[64,8], buf_a[64,8], buf_b[64,8])` only computes the max
for the first 8 rows. Rows 8–63 are left unchanged.

**Root cause**: `blk_stride=0` is the broadcast stride designed for `sub(wide, wide, narrow)`,
where the wide destination's repeat cadence drives iteration and the narrow source stays
per-row. It was never intended for element-wise operations between two identically-shaped
narrow buffers.

**Diagnostic method**: before choosing a tensor format for any vec binary operation,
manually trace:
1. `infer_repeat(dst)` = `span[0] * span[1] / (256 // dtype.size)`
2. `infer_strides(tensor)` — check if `blk_stride=0` or `1`
3. total unique elements = `repeat × (8 if blk_stride==1 else 1) × elements_per_block`
4. compare against the actual element count (`shape[0] * shape[1]`)

If the totals disagree, the operation will silently skip elements.

Reference implementation: `easyasc/stub_functions/vec/vecutils.py` (`infer_strides`, `infer_repeat`).

## 6. Using `[M, 1]` scalar format for binary ops between reduction outputs

The `cmax` output `[HALF_M, 1]` has `span[1]=1`.
Stride inference for `[64, 1]` float: `span[1]=1` matches neither `64` nor `8`,
so defaults apply: `blk_stride=1, rep_stride=8, repeat=1`.

With `blk_stride=1` and 8 blocks per repeat:
- Block 0: elements `[0:8]`
- Block 1: elements `[8:16]`
- …
- Block 7: elements `[56:64]`
- Total: 1 repeat × 8 blocks × 8 elements = **64 elements = all rows** ✓

So `vmax(dst[64,1], src1[64,1], src2[64,1])` correctly computes per-row element-wise max
over all 64 dense scalars from `cmax` output. No rows are skipped.

**Key insight**: operate on the dense scalar `[M, 1]` format BEFORE `brcb` broadcast.
Only `brcb` to `[M, 8]` after the scalar-level operation is complete.

Validated pattern for running max across tiles:

```python
ub_max_s    = Tensor(DT.float, [HALF_M, 1], Position.UB)  # per-tile cmax output
ub_rmax_s   = Tensor(DT.float, [HALF_M, 1], Position.UB)  # running max (persistent)
ub_max      = Tensor(DT.float, [HALF_M, 8], Position.UB)   # broadcast for sub

# before inner loop: initialize running max
dup(ub_rmax_s, neg_large)

# inside each tile:
cmax(ub_max_s, ub_tmp)                                     # per-tile row max
vmax(ub_rmax_s, ub_rmax_s, ub_max_s)                       # update in [M,1] format
brcb(ub_max, ub_rmax_s, dst_blk_stride=1, dst_rep_stride=8)  # broadcast AFTER update
sub(ub_data[0:M, 0:64], ub_data[0:M, 0:64], ub_max)
sub(ub_data[0:M, 64:128], ub_data[0:M, 64:128], ub_max)
```

Here `neg_large` is a sufficiently large finite negative sentinel, not literal
`float("-inf")`.

UB overhead for running max: one extra `[64, 1]` float tensor = 0.25 KB.

## 6a. Copying `[M,1]` scalar state across iterations

The validated running-max pattern often needs a snapshot of the previous scalar state
before updating it, for example to compute `exp(prev_m - curr_m)` in streamed attention.

Do **not** snapshot `[M,1]` buffers with `ub_to_ub`.

Why this fails:
- `ub_to_ub` works in `C0`-sized blocks
- for float `[64,1]`, that means an 8-element block copy per row
- the operation does not mean "copy one scalar per row"

Stable fix:
- allocate a zero buffer in the same `[M,1]` format
- use a vec binary op such as `add(dst, src, zero)` to make the copy

Example:

```python
ub_prev_s = DBuff(DT.float, [HALF_M, 1], Position.UB)
ub_rmax_s = Tensor(DT.float, [HALF_M, 1], Position.UB)
ub_zero_s = Tensor(DT.float, [HALF_M, 1], Position.UB)

dup(ub_zero_s, 0.0)
add(ub_prev_s[slot], ub_rmax_s, ub_zero_s)  # safe scalar-format copy
vmax(ub_rmax_s, ub_rmax_s, ub_max_s)
sub(ub_prev_s[slot], ub_prev_s[slot], ub_rmax_s)
exp(ub_prev_s[slot], ub_prev_s[slot])
```

Study:
- `agent/example/kernels/a2/flash_attn_unnorm.py`
- `agent/references/patterns/a2-cube-vec-cube-vec.md`

## 7. Adapting for row sum (cadd)

Same pattern, replace `vmax` → `add`, `cmax` → `cadd`:

```python
add(ub_tmp, ub_data[0:M, 0:64], ub_data[0:M, 64:128])
cadd(ub_sum_s, ub_tmp)
brcb(ub_sum, ub_sum_s, dst_blk_stride=1, dst_rep_stride=8)
div(ub_data[0:M, 0:64], ub_data[0:M, 0:64], ub_sum)
div(ub_data[0:M, 64:128], ub_data[0:M, 64:128], ub_sum)
```

For streamed normalized attention on a2, the stable update order is:
1. compute `expdiff = exp(prev_max - curr_max)` in `[M,1]`
2. compute the float probability tile `p = exp(score - curr_max)`
3. reduce `sum_j` from that float tile with `add` + `cadd`
4. update `row_sum = row_sum * expdiff + sum_j` in `[M,1]`
5. cast `p` to half only after the sum update if the downstream cube stage needs `p.half().float()`

## 8. UB cost

| Buffer | Shape | Bytes (float) |
|--------|-------|--------------|
| ub_tmp | [64, 64] | 16 KB |
| ub_max_s | [64, 1] | 0.25 KB |
| ub_max | [64, 8] | 2 KB |
| **Total reduction overhead** | | **~18.25 KB** |

## Files to study

- `agent/example/kernels/a2/flash_attn_score.py` — per-tile independent row max
- `agent/example/kernels/a2/flash_attn_score_iter.py` — running max across tiles using `[M,1]` scalar `vmax`
- `agent/example/kernels/a2/flash_attn_unnorm.py` — delayed `expdiff` computed from copied `[M,1]` running state
- `agent/example/kernels/a2/flash_attn_full.py` — running sum + final sliced `div` on top of the delayed numerator pipeline
- `easyasc/simulator_v2/ops/vec/v.py` and `easyasc/simulator_v2/ops/vec/_legacy_vpipe.py` — current vec runtime path for `cmax`, `brcb`, and `dup`
- `easyasc/stub_functions/vec/group.py` — cmax stub with dst_rep_stride default
- `easyasc/stub_functions/vec/dupbrcb.py` — dup and brcb stubs
- `easyasc/stub_functions/vec/vecutils.py` — `infer_strides` and `infer_repeat` logic
