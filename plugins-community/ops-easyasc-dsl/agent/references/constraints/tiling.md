# Tiling, Datamove, and Tensor Alignment

Read this file when a kernel needs non-trivial tile selection, core split selection,
local-buffer capacity reasoning, datamove pattern selection, or tensor layout decisions.
Do not read it for tiny untiled examples.

## 1. Capacity budgets

Use `agent/references/facts-device-runtime.md` for the current device caps
and core counts. This file owns how to apply those caps to tile shape, core
split, local tensor shape, and datamove decisions.

### L0A/L0B element budget

The estimator enforces: `TILE_M * TILE_K + TILE_N * TILE_K <= MAX_TOTAL_TILE_ELEMENTS`
where `MAX_TOTAL_TILE_ELEMENTS = 128 * 1024`.
Source: `agent/scripts/estimate_matmul_datamove.py:9`.

DBuff halves the effective per-slot capacity by doubling the allocation:
- `apply_dbuf_size(size, use_dbuf)` returns `size` when `use_dbuf=True`, `size/2` when `False`.
  Source: `agent/scripts/estimate_matmul_datamove.py:74-77`.

### L0C element budget

The estimator's L0C tile limit is an element-count guard for a float/int32
accumulator tile:

| L0C object | Estimator limit (elements) | fp32/int32 allocation |
|------------|----------------------------|-----------------------|
| `DBuff`    | 32 * 1024                  | 256 KB total (2 slots)|
| `Tensor`   | 64 * 1024                  | 256 KB total (1 slot) |

Source: `agent/scripts/estimate_matmul_datamove.py:10-11, 80-83`.
Runtime caps come from `globvars.l0c_cap`: a2 defaults to 128 KB
(`easyasc/globvars.py:21`), while a5 sets 256 KB (`easyasc/a5.py:105`).

Practical rule: with fp32 L0C DBuff, `TILE_M=128, TILE_N=256` allocates
`2 * 128 * 256 * 4 = 256 KB`. This fits a5 exactly and overflows a2.
Reducing the tile size is the normal fix. `TBuff` / `QBuff` increase the number
of physical slots, so they are not a capacity escape hatch; use them only when
the overlap lifetime truly requires more slots and the smaller tile still fits.

Mandatory authoring check: `slot_count * TILE_M * TILE_N * sizeof(dtype) <= l0c_cap`.
Check `L0C` first before diagnosing simulator errors.

Keep matmul destinations at row offset `0`. Solve oversized M with a higher-level
`TILE_M` decision; N-side subdivision is fine when the destination remains anchored.
**Why:** real cube HW does not place a matmul partial at a nonzero L0C *row*
sub-offset — the result silently fails to land there, leaving that L0C region
uninitialized (which then corrupts downstream `rowmax` / `out`). A column / N
sub-offset is fine; only a nonzero row sub-offset breaks. This is **sim-invisible**:
the Python simulator models `mmad` functionally and honors the offset, so a
row-offset dst passes the sim bit-identically and fails only on hardware.

## 2. Split legality and format constraints

The cube-only Pattern owns the scheduling choice among `split_m`, `split_n`,
and `mix`, plus output-tile ownership. This file owns whether the resulting
tile/split values are legal for the selected layout, dtype, and device.

### L1 fractal height must equal the mmad dimension it feeds

An L1 tile is written in NZ with a fractal z-stride of `M_dst * C0`, and L1 → L0
is a straight fractal copy. So the mmad dimension reading that operand has to
equal the `M_dst` it was written with:

| operand | requirement |
|---------|-------------|
| `l1a` (A) | `M_dst == m` |
| `l1b` (B) | `M_dst == n` |

`gm_to_l1_nd2nz` / `ub_to_l1_nd2nz` default `M_dst` to `dst.shape[0]`, so a plain
`l1tile <<= gm[...]` binds it to the *tile* height. Passing a smaller `n=valid_n`
to `matmul()` then shifts every fractal and the L0C tile comes back **entirely
NaN** — not merely inaccurate, which makes it easy to misread as a data or
synchronisation bug.

The two tail directions are handled differently:

- **M tail**: keep `m = TILE_M`. The rows past `valid_m` compute garbage; slice
  them off at `l0c_to_ub` / `l0c_to_gm`.
- **N tail**: pass `M_dst=valid_n` explicitly on the B load and `n=valid_n` to
  the mmad.

`M_dst` is rounded up to 16 rows internally, so an L1 tensor declared exactly
`[hidden, K]` for a non-multiple-of-16 `hidden` raises "dst footprint exceeds
tensor storage". Declare it `[CeilDiv(hidden, 16) * 16, K]` and pass that value
as `n`; zero the pad rows if their contribution must not be garbage.

The same rule forbids handing `matmul()` a **slice** of that L1 tile. A slice
addresses the tensor with its declared z-stride (`shape[0] * C0`), not the
`M_dst` the ND2NZ actually wrote, so

```python
matmul(dst, l1a, l1w[cnt][0:valid_n, 0:valid_k], n=valid_n, k=valid_k)  # wrong
matmul(dst, l1a, l1w[cnt],                       n=valid_n, k=valid_k)  # right
```

differ whenever `valid_n < shape[0]`. Pass the whole tile and let `n=` / `k=`
bound the mmad. This is easy to miss because the two agree exactly when the tile
is full: a streaming-weight GRU kernel written this way was correct at
`hidden = 127 / 128 / 256` and wrong at `31 / 32 / 64`.

When comparing a suspect matmul result, use NaN-aware comparisons:
`max(0.0, float('nan'))` is `0.0` in Python, so an all-NaN tile reads as a
perfect match through the usual running-max idiom.

### `splitk` vs `splitn`

- Use `splitk` when K-side staging into L0A/L0B is too large, or to legalize the
  inner cube load size while keeping a large outer `TILE_K`.
- Use `splitn` when N-side output-tile width pushes the buffer budget over the cap.
- Normal matmul has no generic hard minimum for `splitk` or `splitn`. Start a
  tuning search at 32 or larger as a conservative heuristic, then use smaller
  values only when the concrete layout, capacity checks, and validation support them.
- MX rule: `matmul_mx` requires `splitk` to be a multiple of 64 and `splitn`
  to be a multiple of 16. For transposed-B MX shortcut paths, static `splitn`
  is also checked for a multiple of 32. FP4 transposed MX loads have stricter
  physical-source alignment: pad or choose a tile whose transposed source axes
  are 64-element aligned.
- Normal-matmul fallback: if the initial 32-element search range does not fit,
  retile `TILE_M` / `TILE_N` or evaluate a smaller split against the actual
  format constraints. For MX / FP4, follow the hard format-specific alignment first.

## 3. Tile size candidates

The estimator searches: `TILE_CANDIDATES = (32, 64, 128, 256, 512)`.
Source: `agent/scripts/estimate_matmul_datamove.py:15`.

### Stable large-K pattern (a5 MKNK 2D-grid splitk)

```
TILE_M  = 128
TILE_N  = 256
TILE_K  = 256
SPLIT_K = 64
```

Source: `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk.py:11-14`.

This uses `select_mknk_2dgrid_splitk` (line 23) to find valid `m_split`/`n_split`
at runtime. The pattern fits a5's 256 KB L0C with DBuff.

### Ownership pattern

For standard cube-major matmuls, split by tile index:
```python
tile_m = CeilDiv(M, TILE_M)
tile_m_per_core = CeilDiv(tile_m, GetCubeNum())
# iterate mt in [tile_m_begin, tile_m_end)
```

For batched small matmuls, flatten and split over the batch axis:
```python
BHN = B * H * N
bhn_per_core = CeilDiv(BHN, GetCubeNum())
```

### Device cross-check

- a5 tile `TILE_M=128, TILE_N=256` at DBuff L0C = 256 KB: fits a5, overflows a2 (128 KB cap).
- a5/950 uses `GetCubeNum()=32`, a5pr/950pr uses `28`, and both a2/b3 and
  a3/`Ascend910_9362` use `20`.
  Verify load balance against the selected facade.
- a5 DBuff UB allocations up to 256 KB must stay within 192 KB on a2; when the
  a5 kernel contains SIMT, use the reduced 216 KB UB cap because the remaining
  space is reserved for SIMT functionality.

## 4. Datamove model

### TILEK alignment

When `TILEK < K`, the effective K span used in datamove estimates is aligned to 256:

```python
align_tile_k(TILEK, k):
    if TILEK != k:
        return CeilDiv(TILEK, 256) * 256
    return TILEK
```

Source: `agent/scripts/estimate_matmul_datamove.py:90-93`.

The operand visited through the TILEK loop pays the aligned cost;
the operand that loads the full K once does not.
Source: `agent/scripts/estimate_matmul_datamove.py:181-188`.

### Per-core datamove

`estimate_percore_datamove(m, n, k, TILEM, TILEN, TILEK, mode, ...)` computes total
elements moved per core, using one of three modes:
- `left_first`: left operand loaded once per M tile; right operand reloaded per TILEK step
- `right_first`: right operand loaded once per N tile; left operand reloaded per TILEK step
- `balanced`: both operands reloaded per TILEK step

Source: `agent/scripts/estimate_matmul_datamove.py:140-196`.

`estimate_multi_core(...)` tries all three modes and returns the best.
Source: `agent/scripts/estimate_matmul_datamove.py:234-272`.

### ND publish (`ub_to_l1_nd2nz`)

Best for straightforward vec preprocess + cube consume.
- Write subblock rows into UB, then publish with explicit `m_dst/n_dst/m_src/n_src`.
- Keep row mapping consistent with `GetSubBlockIdx()`.
- Source UB views must be compact for the logical operand. Do not publish a
  `[m, n]` subview sliced out of a wider UB matrix and expect the column offset
  or row stride to become a compact operand; first copy or produce the block
  into a dedicated compact UB tile.
- For general vec preprocess, split into two half ranges per vector side:
  - `half_rows = CeilDiv(total_rows, 2)`; sides 0 and 1 handle their half independently.

Physical NZ backing for `gm_to_l1_nd2nz` destinations:
`CeilDiv(N, C0) * Align16(M_dst) * C0`.
A half L1 tile needs 16-column physical backing even when logical N is 2 or 8.
Source: `easyasc/stub_functions/vec/datamove.py` (simulator validates NZ destination size).

L0C NZ rule: `mmad` and `l0c_to_*` treat L0C as NZ with fixed `c0=16`.
Required backing: `CeilDiv(N, 16) * Align16(M) * 16`.

For compact small-M outputs, the physical L0C source stride must stay compact
too. If `matmul(..., m=5, n=64, ...)` writes into a physical `[16,64]` L0C
tile, follow-up `l0c_to_gm_nz2nd` / `l0c_to_ub` calls should use
`M_src=16`, not the reduction K or some larger staging height such as `256`.
Oversized `M_src` reads later 16-column blocks from the wrong L0C addresses on
A5/C310.

Debugging: if a validator says an `nd2nz` destination is too small, check the physical
padded backing first before changing logical dimensions.

### NZ publish (`ub_to_l1_nz`)

Use when input is already packed for the NZ path:
- Do vec compute in ND register form.
- Pack to NZ-friendly UB layout (`deinterleave`, `reg_to_ub`).
- Publish with `l1 <<= ub.nz()`.

### Sub-block assembly for matmul operands

When two vec sub-blocks cooperatively build an L1 operand for cube matmul, split
along a dimension that keeps the matmul K dimension physically coherent. For an
operand logically consumed as `[D, L]`, it is safer to store physical `[L, D]`
row ranges and pass `.T` to matmul than to let sub-blocks write column slices of
`[D, L]`.

```python
# OK: sub-blocks own row ranges, then matmul sees [D, L] through .T.
l1_v = Tensor(DT.bfloat16, [L, D], Position.L1)
if GetSubBlockIdx() == 0:
    l1_v[0:half_l, 0:D] <<= ub0
else:
    l1_v[half_l:L, 0:D] <<= ub1
matmul(out, a, l1_v.T)

# Risky: sub-blocks assemble the K dimension through disjoint column slices.
l1_v = Tensor(DT.bfloat16, [D, L], Position.L1)
if GetSubBlockIdx() == 0:
    l1_v[0:D, 0:half_l] <<= ub0
else:
    l1_v[0:D, half_l:L] <<= ub1
matmul(out, a, l1_v)
```

If a matmul is correct for one producer but fails only after sub-block column
assembly, inspect the physical L1 layout before changing the math.

### Unaligned GM width

For unaligned GM widths, allocate the local second dimension to an aligned
width; do not shrink to the logical width. Use
`gm_to_ub_pad(..., burst_len_element=logical_width, dst_stride=(aligned_width - logical_width) / C0)`
to zero-pad each row on load, and mirror with `ub_to_gm_pad` on writeback.

### Strided GM gather

Use `gm_to_ub_pad` directly with `n_burst`, `burst_len_element`, and
`src_stride_element` to gather non-contiguous rows without host-side `permute`.

### Internal workspace bridge for single-kernel fusion

When stage-1 produces data on `MTE3` and stage-2 must reread it through `MTE2`,
materialize the intermediate in GM workspace — do not try to keep it purely local.

Stable attention fusion pattern:
- keep `qk_tmp:[BH,S]` as a float workspace for the three-pass softmax
- store `p.half()` into `prob_tmp:[BH,S]` workspace before the final PV matmul
- add an explicit stage boundary before reloading `prob_tmp`
- perform the final value scaling from that half workspace so the `p.half().float()`
  contract stays exact

For the final vec-only `prob_tmp -> value -> out` stage:
- keep the nested reload/compute/writeback chain inside one outer `auto_sync()`
- make DBuff slot ownership explicit via the ready/valid handshake
- verify both simulator execution and generated C++ declarations before removing
  any manual barriers

If the delayed reuse fits in one tile of on-chip lifetime, prefer an on-chip
lookahead bridge instead of GM workspace:
- keep the stage-1 operand needed by stage-2 resident in `L1` / `TBuff`
- publish the vec-produced fp8 probability tile directly into an `L1` slot
- buffer per-tile rescale state in the same delayed slot family as the consumer

Caveat: do not republish a freshly packed fp8 UB tile straight to L1 when exact
downstream reuse matters; the packed UB layout can differ from the ND view the
later cube path expects.

Reference: `agent/example/kernels/a5/attention/test_mla_entire.py` (streamed MLA with workspace-mediated
reuse).

## 5. Tensor alignment and layout

### Second-dimension 32-byte rule

Local UB/L1/L0 tensor allocations must keep the second dimension 32-byte aligned:

```
shape[1] * dtype.size % 32 == 0
```

Equivalently, `shape[1]` must be a multiple of `C0 = 32 // dtype.size`:

| dtype | elem size | C0 | minimum legal cols |
|-------|-----------|----|--------------------|
| float / int / uint32 | 4 | 8 | 8 |
| half / bfloat16 / int16 / uint16 | 2 | 16 | 16 |
| int8 / uint8 / fp8 / hif8 | 1 | 32 | 32 |
| uint64 / int64 | 8 | 4 | 4 |

Packed compute-view dtypes are exceptions to the `dtype.size` formula:
`DT.int4` and `DT.fp4_*` cannot be allocated directly as `Tensor` / `DBuff` /
`TBuff` / `QBuff`; use their carrier dtype and reinterpret only for the compute
op. Logical int4 and FP4 report `C0 == 64`, while FP4 payload storage uses
`DT.uint8` carriers.

A consequence of `int8` NZ `C0 == 32`: a *compact* `[M, K]` int8 L1 operand needs
`K` a multiple of 32, or the NZ fractal pads `K` up to 32 and overflows the
tensor's logical footprint.

Apply this to `Tensor`, `DBuff`, `TBuff`, and `QBuff` shapes even when the Python
constructor does not reject a narrow second dimension. Treat non-aligned shapes such as
`[M, 1]` float UB tensors as legacy patterns to migrate, not as templates for new code.

### 2D shape constraint

`Tensor.__init__` and `DBuff.__init__` both enforce `len(shape) == 2` and raise
`ValueError` on violation. Source: `easyasc/utils/Tensor.py:74-75, 617-618`.

### Reduction scalar buffers

Choose the scalar layout from the reduction primitive:
- `cmax` / `cadd` row scalars on a2: use `[1, M]`, then `brcb(..., repeat=M//8,
  dst_blk_stride=1, dst_rep_stride=8)` to expand to `[M, 8]` for float broadcast.
- `cgadd` examples that reduce a compact row tile also use an aligned second dimension,
  for example `[1, M]`.

### Reduction layout handoff

This file owns the aligned physical layouts above. Complete row/group
reduction-to-broadcast compositions live in
`agent/references/patterns/vec-row-reduce-broadcast.md` and
`agent/references/patterns/vec-group-reduce-broadcast.md`.

## 6. Tool reference

`agent/scripts/estimate_matmul_datamove.py` is the canonical calculator. Use it when:
- The matmul is large enough that core split matters.
- There are multiple legal tile candidates.
- Downstream vec work constrains which axis may be split.

Key exported functions: `estimate_strategy`, `estimate_multi_core`,
`estimate_percore_datamove`, `align_tile_k`, `estimate_total_tile_elements`.

Read the result as: tile shape, core split, loop mode, and candidate tie set.

## 7. Quick checklist

Before accepting a tiled kernel:
- target device identified (a2/a5) and device-specific budgets used
- tile shape chosen explicitly
- core split chosen explicitly with the correct core count
- split mode matches downstream dependency
- L0A/L0B byte budgets checked
- L0C DBuff total checked against device-specific `l0c_cap`
- UB DBuff total checked against device-specific `ub_cap` (when vec stages exist);
  for SIMT kernels on a5, the effective UB cap is 216 KB
- `splitk` / `splitn` follow the shortcut's format-specific alignment rule
- L0C destination remains row-offset 0
- `shape[1] * dtype.size` is 32-byte aligned for every local tensor/buffer
- Reduction scalar buffers use `[1, M]` before `brcb`, not `[M, 1]`
- Ownership and counters make sense for the chosen loop structure

## 8. Files to study

- `agent/scripts/estimate_matmul_datamove.py` — canonical datamove calculator
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk.py` — stable large-K pattern
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul.py` — ND publish example
- `agent/example/kernels/a5/pipeline_patterns/vec_cube_abs_sqrt_matmul_nz.py` — NZ publish example
- `easyasc/utils/Tensor.py` — shape enforcement
