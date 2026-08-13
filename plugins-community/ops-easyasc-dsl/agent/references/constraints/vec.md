# Vec Compute Semantics: Stride, Reduction, and Masking

Read this file when writing or debugging A2 public vec-stub compute in a kernel
body, an a2 pure-vec kernel, or an a2 vec sub-block. It covers stride
alignment, row-wise reductions, broadcast, masking, flag/select helpers, and
zero-copy UB views.

Boundary: a2 public surface, GM bridges, mutexes, sub-block layout, and UB
allocation gotchas live in `constraints/a2.md`. Full-tile tail policy lives in
`constraints/tail.md`; online-softmax mask/update timing lives in
`patterns/online-softmax-tail.md`. A5 `@vf()` / `micro` uses different
register-level semantics; see `constraints/a5.md` and `constraints/sync.md`.

Cross-references:
- `constraints/a2.md` — a2 device surface, bridge, sub-block, and UB layout boundaries
- `constraints/tail.md` — full local shapes and generic tail legality
- `patterns/online-softmax-tail.md` — online-softmax S1/S2/causal mask timing
- `constraints/precision.md` — exact cast / hif8 rounding requirements
- `constraints/a5.md` — A5 `@vf()` / `micro` boundary; do not apply this A2 vec model there

---

## 1. Vec-stride basics

Most A2 vec ops infer `repeat` and strides from each tensor's `span` (the slice
extent) and `shape` (the full allocation width). `cast` has a separate
wider-dtype repeat/stride rule; see §5.3 and `easyasc/stub_functions/vec/cast.py`.

`vecutils.infer_strides(tensor)` — `easyasc/stub_functions/vec/vecutils.py` lines 109–132:

| `span[1]` (float, C0=8) | `blk_stride` | `rep_stride` |
|-------------------------|-------------|-------------|
| `64` (= 8×C0)           | 1           | `shape[1] // C0` |
| `8` (= C0)              | 0           | `shape[1] // C0` |
| other                   | 1 (default) | 8 (default) |

| `span[1]` (half, C0=16) | `blk_stride` | `rep_stride` |
|-------------------------|-------------|-------------|
| `128` (= 8×C0)          | 1           | `shape[1] // C0` |
| `16` (= C0)             | 0           | `shape[1] // C0` |
| other                   | 1 (default) | 8 (default) |

Override: if `span[0] == 1` and a match occurred, `rep_stride` is forced to `0`.

`infer_repeat(tensor)` — `easyasc/stub_functions/vec/vecutils.py` lines 67–82 —
uses:

```
CeilDiv(span[0] * span[1], 256 // dtype.size)
```

This only applies to addressable vec dtypes. Packed compute-view dtypes such as
`DT.int4` and `DT.fp4_*` do not have `dtype.size` and are not ordinary A2 vec
operands.

Count controls are different modes, not aliases:

- `count=N`: process `N` total elements in counter mode; `repeat`, strides, and
  the normal mask do not define traversal.
- `count_per_rep=N`: keep repeat mode and activate the first `N` lanes inside
  every repeat; `repeat` and `*_rep_stride` still define how many chunks run and
  where they start. `N` cannot exceed one repeat for the operation dtype.

They are mutually exclusive. Use `repeat + rep_stride` for multiple chunks;
never pass a total tile length as `count_per_rep`.

### 1.1 Narrow `[M, C0]` views are broadcast-layout aliases

When `span[1] == C0`, `infer_strides()` returns `blk_stride=0`, not `1`. That
means the 8 blocks inside one repeat alias the same physical C0 block, and the
view advances only when `rep_stride` advances to the next repeat.

Example: float `[64, 8]` has `C0 = 8`, so default inference gives:

- `repeat = CeilDiv(64 * 8, 64) = 8`
- `blk_stride = 0`
- `rep_stride = 8 // 8 = 1`

Each repeat still issues 8 blocks, but those 8 blocks all land on the same 8
physical floats. Across 8 repeats, the op touches only the first 64 physical
elements, not 512 unique elements.

This is a general A2 vec auto-infer rule, not a `dup()` special case. It
affects any vec helper whose omitted stride parameters are derived from that
view. Treat `[M, 8]` (float) and `[M, 16]` (half) as broadcast layouts, not as
dense storage. Keep row-scalar state in `[1, M]`, then materialize the
broadcast view with `brcb(...)` when needed. If you truly need to fill or walk
the full physical storage of the broadcast buffer, pass explicit count/stride
control instead of relying on default inference.

---

## 2. Continuous vs sliced

**Purely element-wise ops on same-shaped operands** (e.g. `muls`, `exp`, `cast`)
can run over the full buffer without slicing.

**Mixed-width ops** (e.g. `sub(wide, wide, narrow)`) need sliced views so that
every operand advances by exactly one row per repeat.

Classic misalignment scenario for float:

- `wide = [M, 128]`, `narrow = [M, 8]`
- `repeat = M * 128 / 64 = 2M` (from dst)
- narrow advances 1 repeat per repeat → row 0's second half gets row 1's max

**Fix**: slice the wide operand to `[M, 64]` views before the op:

```python
sub(ub[0:M, 0:64],   ub[0:M, 0:64],   max_buf)   # first half
sub(ub[0:M, 64:128], ub[0:M, 64:128], max_buf)    # second half
```

Tensor slicing creates a view with updated `span` and `offset` but the same
`shape`, so `rep_stride = shape[1] // C0` still skips the full row width between
repeats while `repeat` covers only the sliced region.

**Quick decision table:**

| Op | Needs slicing? |
|----|----------------|
| `muls(wide, wide, scalar)` | No — scalar broadcasts uniformly |
| `exp(wide, wide)` | No — same-shape in-place |
| `cast(half_out, float_in)` | No for matching logical shape, but repeat/mask use the wider dtype |
| `sub(wide, wide, narrow)` | **Yes** — narrow advances faster |
| `vmax(dst64, wide_half1, wide_half2)` | **Yes** — column views of wider buffer |
| `brcb(wide, narrow)` | Explicit strides — see §4 |

Verified in: `agent/example/kernels/a2/attention/flash_attn_score_pv.py` lines 84–85 (sliced sub).

---

## 3. Reduction output format (`cmax` / `cadd`)

`cmax(dst, src)` and `cadd(dst, src)` reduce **one full repeat** (64 float or 128 half
elements = 8 blocks) to **one scalar** per repeat.

Default `dst_rep_stride=1` → scalars packed densely:

```
dst[0] = max/sum of row 0
dst[1] = max/sum of row 1
...
```

Verified in: `easyasc/stub_functions/vec/group.py` lines 105–127 (`cmax`) and
lines 205–227 (`cadd`); simulator execution in `easyasc/simulator/pipe_vec.py`
lines 634–685.

This dense scalar layout is **not** a C0-block layout — you cannot pass it
directly to `sub`/`div` as the narrow operand. See §4 for the required broadcast.

Use an aligned `[1, M]` tensor for row-scalar state when accumulating multiple
`cmax` / `cadd` outputs with `vmax` / `add` before broadcasting.
Do not update running row-scalar state in `[M, 8]` broadcast format:
binary ops infer `blk_stride=0` for that narrow view, so the same C0 block is
reused across repeats instead of advancing one scalar per row.
For tail rows, keep the physical scalar width aligned and gate the GM boundary;
do not allocate `[valid_m, 1]` or `[M, 1]`.

The **group** variants `cgmax` / `cgadd` (also `cgmin` / `cpadd`, all a2-only)
reduce **one block** (8 elements) to one scalar — i.e. they emit **one scalar per
source block**, not one per repeat. A direct `cgadd(ub_sum_s, src)` into a
`[1, ROW_CHUNK]` row-scalar buffer therefore overruns it (one scalar per source
block is far more than `ROW_CHUNK` scalars) and silently corrupts neighbouring
scratch. Use a **two-stage block reduction**: `cgadd(wide_tmp, src)` first (one
scalar per block), then a strided second pass
`cgadd(ub_sum_s, wide_tmp, repeat=ROW_CHUNK // 8, dst_rep_stride=1, src_blk_stride=1, src_rep_stride=8)`.
The A2 block32 flash-attn row max/sum reductions use exactly this two-stage
`cgmax` / `cgadd` shape.

### 3.1 Narrow group reductions: `count_per_rep` limits width, `rep_stride` moves to the next group

`count_per_rep` sets the live prefix for the current reduction and the stub
restores the full vec mask after the op. It does not select the next group:
`*_rep_stride` does that in 32-byte data blocks. For fp32 group32,
`rep_stride=4` advances 32 elements.

Keep narrow logical groups in physically aligned storage unless every later op
has been checked against the repeat/mask model. The complete multi-primitive
composition, failure signatures, and runnable references now live in
`agent/references/patterns/vec-group-reduce-broadcast.md`.

### 3.2 Buffering long vec conversion chains

For long pure-vec low-precision conversion chains, double-buffer the GM-facing
UB staging buffers first (`GM -> UB` input and `UB -> GM` output). Keep large
fp32 numeric scratch single-buffered unless it is genuinely live across adjacent
MTE2/V/MTE3 beats. This is the pattern used by the grouped bf16 fp4-emulation
kernels: `x_bf16` / `y_bf16` rotate as `DBuff` slots to overlap GM traffic,
while the fp32 absmax/scale/quantization scratch stays single-slot so group64
variants do not blow the UB budget. Stress with a shape that makes each vec lane
reuse the same physical DBuff slot, for example a non-round row count such as
`rows=321`.

---

## 4. `brcb` broadcast

`brcb(dst, src)` expands each scalar in `src` to fill one C0 block in `dst`.

Required pattern between `cmax`/`cadd` and any `sub`/`div` that consumes the
per-row statistic:

```python
ub_max_s = Tensor(DT.float, [1, HALF_M], Position.UB)   # cmax output
ub_max   = Tensor(DT.float, [HALF_M, 8], Position.UB)   # broadcast result

cmax(ub_max_s, ub_tmp)
brcb(ub_max, ub_max_s, repeat=HALF_M // 8, dst_blk_stride=1, dst_rep_stride=8)
sub(ub_data[0:M, 0:64],   ub_data[0:M, 0:64],   ub_max)
sub(ub_data[0:M, 64:128], ub_data[0:M, 64:128], ub_max)
```

Always pass `repeat=M//8`, `dst_blk_stride=1`, and `dst_rep_stride=8` explicitly
for row-stat broadcasts on a2. Defaults are not guaranteed to be correct for this
scalar-to-row broadcast.

`brcb` ignores the vec mask (confirmed: `easyasc/simulator/pipe_vec.py` lines 714–727;
`easyasc/stub_functions/vec/dupbrcb.py` line 93 calls `ensure_repeat_mode()`
which exits count mode but does not apply mask).

Full running-max pattern with `brcb` after each per-tile update:

```python
dup(ub_rmax_s, neg_large)            # reset per M-tile
for nt in range(...):
    cmax(ub_max_s, ub_tmp)
    vmax(ub_rmax_s, ub_rmax_s, ub_max_s)    # accumulate in [1,M]
    brcb(ub_max, ub_rmax_s, repeat=M // 8, dst_blk_stride=1, dst_rep_stride=8)
    sub(ub_data[0:M, 0:64],   ub_data[0:M, 0:64],   ub_max)
    sub(ub_data[0:M, 64:128], ub_data[0:M, 64:128], ub_max)
```

Aligned scalar-buffer reference: `agent/example/kernels/a2/attention/attn_backward_dense_total_tail_stage1_prob_dqk_gq_gk_gv_hif8_output_cast.py`
lines 207–208 and 330–341.

### 4.1 One row scalar reused by every 64-lane group

The reusable two-level `cadd` plus two-level `brcb` composition moved to
`agent/references/patterns/vec-row-reduce-broadcast.md`. This constraint file
owns the individual reduction, mask, repeat, stride, and `brcb` semantics; the
pattern owns their full-row composition and physical-layout invariant.

Row-sum-before-cast rule: update `row_sum` from the float probability tile
**before** casting to half for the downstream cube path.
For online-softmax S1/S2 tail mask placement around this update sequence, see
`patterns/online-softmax-tail.md`.

---

## 5. Mask semantics

Each vec lane owns one `current_mask: uint8[256]`, initialized to all ones
(`easyasc/simulator/pipe_vec.py` lines 401–408).

### 5.1 `set_mask` / `reset_mask`

Stub signature: `set_mask(mask_high, mask_low)`
(`easyasc/stub_functions/vec/vecmask.py` lines 10–20)

- `mask_low` (uint64): bit `i` → `mask[i]` for `i` in `[0, 64)`
- `mask_high` (uint64): bit `i` → `mask[64+i]` for `i` in `[0, 64)`
- `mask[128:256]` is **not** touched by `set_mask`

Low-prefix-only usage (most common): `set_mask(0, low_mask)`

`reset_mask()` restores all 256 slots to 1.

### 5.2 Active prefix by dtype

| dtype | active prefix |
|-------|---------------|
| `int64` / `uint64` | `mask[0:32]` |
| `float` / `int32` / `uint32` | `mask[0:64]` |
| `half` / `bfloat16` / `int16` / `uint16` | `mask[0:128]` |
| `int8` / `uint8` / `DT.e4m3` / `DT.e5m2` / `hif8` | `mask[0:256]` |

The active prefix is `256 // dtype.size` for vec ops that reach `VPipe`.
Current A2 public stubs support only a subset of these dtypes for each op.
Packed compute-view dtypes (`DT.int4`, `DT.fp4_*`) have no addressable byte size
and must not be reasoned about with this table.

The same prefix is reused for every repeat — the mask does **not** advance per repeat.
`set_mask_by_count(count)` replaces the whole 256-entry mask with a prefix mask
(`pipe_vec.py` lines 427–433).

### 5.3 Ops that gate writeback on mask

`mask == 1` → write computed result; `mask == 0` → keep old `dst` value.

Applies to (verified in `easyasc/simulator/pipe_vec.py` `_masked_write` lines 886–895):

- Unary: `exp`, `ln`, `abs`, `rec`, `sqrt`, `rsqrt`, `vnot`, `relu`
- Binary: `add`, `sub`, `mul`, `div`, `vmax`, `vmin`, `vand`, `vor`, `muladddst`
- Unaryscalar: `adds`, `muls`, `vmaxs`, `vmins`, `lrelu`, `axpy`
- `dup`
- `cast` (handled separately at `pipe_vec.py` lines 568–632, but with the same
  preserve-old-dst behavior for masked lanes)

For `cast`: the active domain is determined by the **wider** of src/dst dtypes
(e.g. `float ↔ half` uses 64 mask slots per repeat, not 128).

### 5.4 Additive reductions: mask zeroes contribution

`mask == 0` → src slot contributes `0` (not preserved dst).

Applies to: `cadd`, `cgadd`, `cpadd`
(`easyasc/simulator/pipe_vec.py` lines 656–685 and 706–712)

For `cadd` / `cgadd`, if all lanes for that scalar/block are masked off, dst is
**not overwritten**. `cpadd` is different: it always writes the pair sums, so a
fully masked pair writes zero.

### 5.5 Max/min reductions: masked slots → sentinel

`mask == 0` → slot replaced by sentinel before reduction:

- `cmax`, `cgmax`: masked slots act like `-inf` (`pipe_vec.py` lines 660–661)
- `cmin`, `cgmin`: masked slots act like `+inf` (`pipe_vec.py` lines 662–663)

For these max/min reductions, if all lanes for that scalar/block are masked off,
dst is **not overwritten**.

### 5.6 Ops that ignore the vec mask

These ops follow their own semantics only:

`select`, `compare`, `compare_scalar`, `gather`, `gather_block`, `scatter`, `sort32`,
`mergesort4`, `mergesort_2seq`, `brcb`

`compare`/`compare_scalar` and `select` use an explicit packed-bit `uint8`
control tensor (shape `[..., N // 8]`, not `[..., N]`). Simulator execution is
in `easyasc/simulator/pipe_vec.py` lines 901–1039.

---

## 6. Compare/select and reinterpret helpers

### 6.1 `compare_scalar` + `select` control flow

Both `compare_scalar` and `select` ignore the current vec mask; selection is
controlled only by the explicit `uint8` flag tensor. This makes them the stable
control-flow primitives for pure vec kernels.

Non-finite guarding pattern (condensed from
`agent/example/kernels/a2/vec_only/to_hif8_torch.py` lines 70–124):
```python
absub <<= x.abs()
compare_scalar(finiteflag, absub, FLOAT32_FINITE_MAX, CompareMode.LE)
select(workub, finiteflag, xub_t, zero_buf, SelectMode.TENSOR_SCALAR)
# ... finite-safe computation ...
select(finite_part, finiteflag, outub_t, zero_buf, SelectMode.TENSOR_SCALAR)
absub <<= xub_t.abs()
compare_scalar(infflag, absub, FLOAT32_FINITE_MAX, CompareMode.GT)
select(inf_part, infflag, xub_t, zero_buf, SelectMode.TENSOR_SCALAR)
finiteflag_u16 = finiteflag.reinterpret(DT.uint16)
infflag_u16 = infflag.reinterpret(DT.uint16)
nanflag_u16 = nanflag.reinterpret(DT.uint16)
tmpflag_u16 = tmpflag.reinterpret(DT.uint16)
vnot(nanflag_u16, finiteflag_u16)
vnot(tmpflag_u16, infflag_u16)
vand(nanflag_u16, nanflag_u16, tmpflag_u16)
select(nan_part, nanflag, xub_t, zero_buf, SelectMode.TENSOR_SCALAR)
outub_t <<= finite_part + inf_part
outub_t <<= outub_t + nan_part
```

Do not use `compare(x, x, CompareMode.NE)` as an A2 public-vec NaN detector.
The simulator originally treated `NaN != NaN` as `true`, but real 910B3 vec
compare reported unordered-false, so the stable contract is to derive `nanflag`
from packed `finiteflag` / overflow flags instead.

`SelectMode.TENSOR_TENSOR` is supported only when `dst` starts at a different
UB address from `src1` and `src2`. Parser lowering emits `PipeBarrier<PIPE_V>();`
between `SetCmpMask(...)` and the backend `Select(...)`. If `dst` would alias a
source, rewrite the expression as scalar-select pieces plus arithmetic merge when
the dtype has a suitable identity value.

### 6.2 Bit-level float analysis with `reinterpret`

For the UB bit-analysis pattern here, `reinterpret` is a zero-copy view change
that rescales the second dimension by dtype-width ratio. The helper is blocked
on L0C, and packed compute-view reinterprets (`DT.int4`, `DT.fp4_*`) have
stricter carrier/location rules; see `easyasc/stub_functions/misc.py` lines
61–168.

Pattern from `agent/example/kernels/a2/vec_only/to_hif8_torch.py:75-82`:

```python
x_u16     = workub.reinterpret(DT.uint16)
exp_u16   = expub.reinterpret(DT.uint16)
mask_u16  = expmask.reinterpret(DT.uint16)
vand(exp_u16, x_u16, mask_u16)          # isolate exponent bits
expabs_u16 = expabsub.reinterpret(DT.uint16)
vnot(expabs_u16, exp_u16)
vand(expabs_u16, expabs_u16, mask_u16)
```

Exact hif8 rounding is not a vec control-flow rule; see
`constraints/precision.md` for the contract and cast sequence.

---

## 7. a2 `Div<float>` 1-ulp behavior

A2 hardware `Div<float>` (the vec `div` op on fp32) is **not** IEEE
round-to-nearest bit-for-bit. It carries a deterministic 1-ulp correction that
depends only on the normalized numerator mantissa family (`1.0*2^e` vs
`1.5*2^e`) and the divisor's low mantissa bits — independent of sign and
exponent. A simulator or PyTorch fp32 divide will therefore differ from the
board by up to 1 ulp even on otherwise bit-exact operands.

This was confirmed bringing up `agent/example/kernels/a2/vec_only/mbs_mxfp4_fp32.py`: the
final per-128 macro divide stayed 1 ulp off real HW even after forcing the
numerator through `bf16` (ruling out extra numerator mantissa), and a board scan
over all `factor = 1 + k/256` (e0m8) divisors showed the drift is a pure
function of the `(prediv mantissa family, k)` pair. The fp32 reference only
matches HW bit-for-bit through the table-driven `_a2_mbs_div_fp32_sim()` helper
in that kernel.

Implication: when an a2 vec kernel's correctness depends on an fp32 `div`
matching the board exactly, model the HW `Div<float>` correction (e.g. a small
table keyed on the normalized mantissa family and divisor low bits) instead of
trusting a plain fp32 divide. Localize such drift with a stage-truncated probe
like `agent/example/kernels/a2/vec_only/mbs_mxfp4_fp32_line_probe.py`, which exports the
pre-divide stage and runs a standalone `div` on exact reference operands to
separate "earlier stage hidden by quantization" from "the divide itself is the
first visible HW divergence".

---

## 8. Files to study

- `easyasc/stub_functions/vec/vecutils.py` — `infer_strides`, `infer_repeat`
- `easyasc/stub_functions/vec/group.py` — `cmax`, `cadd` stubs; default `dst_rep_stride=1`
- `easyasc/stub_functions/vec/cast.py` — wider-dtype repeat inference and cast rep-stride rules
- `easyasc/stub_functions/vec/dupbrcb.py` — `dup` and `brcb` stubs
- `easyasc/stub_functions/vec/vecmask.py` — `set_mask`, `reset_mask` stubs
- `easyasc/simulator/pipe_vec.py` — simulator execution model for all vec ops
- `agent/example/kernels/a2/attention/attn_backward_dense_total_tail_stage1_prob_dqk_gq_gk_gv_hif8_output_cast.py` — aligned `[1, HALF_M]` scalar buffers and `brcb(..., repeat=HALF_M//8)`
- `agent/example/kernels/a2/attention/flash_attn_score_pv.py` — sliced sub, cmax→brcb, running max; aligned `[1, HALF_M]` scalar buffers
- `agent/example/kernels/a2/attention/flash_attn_full.py` — running sum, delayed expdiff, sliced div; aligned `[1, HALF_M]` scalar buffers
- `constraints/a2.md` — a2 bridge, sub-block, and UB allocation boundaries that call into these vec semantics
