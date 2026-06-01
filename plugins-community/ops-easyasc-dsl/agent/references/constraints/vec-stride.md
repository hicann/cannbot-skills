# Vec Stride and Slicing Constraints

Read this file when a vec operation needs to access part of a wider buffer, or when a "narrow" source
(e.g. row-max buffer) must align with a "wide" destination row by row.

## Goal

Decide correctly when a vec operation can run continuously over a full buffer
versus when it requires sliced views or explicit stride configuration.

## 1. The alignment problem

Vec operations infer `repeat` from the destination tensor and strides from each tensor's `span`/`shape`.
When a wide buffer (e.g. `[M, 128]`) is paired with a narrow buffer (e.g. `[M, 8]`), the repeat counts
may not align row-by-row.

For float (`C0=8`):
- `[M, 128]` → `span1=128` does not match `8*C0=64` or `C0=8` → default strides (`blk=1, rep=8`)
- Each row takes **2 repeats** (128 / 64 = 2)
- `[M, 8]` → `span1=8 == C0` → `blk=0, rep=1`
- Each row takes **1 repeat** from the narrow buffer

If `sub(wide[M,128], wide[M,128], narrow[M,8])` is called directly:
- `repeat = M * 128 / 64 = 2M` (from dst)
- narrow advances 1 per repeat → after repeat 0 (row 0 first half), narrow moves to row 1
- **row 0's second half gets row 1's value** → misaligned!

## 2. Fix: slice the wide buffer to 64-column views

Slicing to `[M, 64]` creates a view where `span1=64 == 8*C0`:
- `blk=1, rep=shape[1]//C0` (e.g. `128//8=16` for a 128-wide parent)
- Each row takes **1 repeat** → aligns with the narrow buffer's `rep=1`

```python
# Correct: sliced views ensure 1 repeat per row
sub(ub[0:M, 0:64], ub[0:M, 0:64], max_buf)   # first half
sub(ub[0:M, 64:128], ub[0:M, 64:128], max_buf) # second half
```

The slice syntax creates a Tensor view with updated `span` and `offset` while keeping the
original `shape`. The stride auto-inference uses `span` for stride selection and `shape` for
`rep_stride` calculation, which correctly skips the full row width between repeats.

## 3. When slicing is NOT needed

Purely element-wise operations (no narrow source) can run continuously over the full buffer:

| Operation | Needs slicing? | Reason |
|-----------|---------------|--------|
| `muls(wide, wide, scalar)` | No | Scalar broadcasts uniformly |
| `exp(wide, wide)` | No | Same-shape in-place, no alignment issue |
| `cast(half_out, float_in)` | No | Same-shape element-wise conversion |
| `sub(wide, wide, narrow)` | **Yes** | Narrow source advances 1 row/repeat |
| `vmax(dst64, wide_half1, wide_half2)` | **Yes** | Need column views of a wider buffer |
| `brcb(wide, narrow)` | Explicit strides | See brcb section |

**Rule**: if all source and destination tensors have the same `span` and are operated element-wise,
no slicing is needed. If any operand has a different width (narrower), slice the wider operands to match
the narrow operand's per-row repeat cadence.

## 4. Stride auto-inference rules

From `vecutils.infer_strides(tensor)` for float (`C0=8`):

| `span[1]` | Matches | `blk_stride` | `rep_stride` |
|-----------|---------|-------------|-------------|
| `64` (= 8×C0) | Yes | 1 | `shape[1] // C0` |
| `8` (= C0) | Yes | 0 | `shape[1] // C0` |
| other | No | 1 (default) | 8 (default) |

For half (`C0=16`):

| `span[1]` | Matches | `blk_stride` | `rep_stride` |
|-----------|---------|-------------|-------------|
| `128` (= 8×C0) | Yes | 1 | `shape[1] // C0` |
| `16` (= C0) | Yes | 0 | `shape[1] // C0` |
| other | No | 1 (default) | 8 (default) |

When `span[0] == 1` and a match occurred, `rep_stride` is overridden to `0`.

`infer_repeat(tensor)` always uses: `span[0] * span[1] / (256 // dtype.size)`

## 5. Column slicing via Tensor views

DSL tensor slicing (`tensor[row_start:row_end, col_start:col_end]`) creates a view with:
- `offset` adjusted to the slice start
- `span` set to the slice extent
- `shape` inherited from the parent (full allocation width)

This means `rep_stride = shape[1] // C0` correctly accounts for the full row width,
while `repeat = span[0] * span[1] // (256 // dtype_size)` only covers the sliced region.

Example for `ub_data[0:64, 64:128]` where `ub_data` is `Tensor(float, [64, 128])`:
- `span = [64, 64]`, `shape = [64, 128]`, `offset = [0, 64]`
- `blk=1, rep=128//8=16` (skips full 128-wide row)
- `repeat = 64*64/64 = 64` (one repeat per row)

## Files to study

- `easyasc/stub_functions/vec/vecutils.py` — stride inference logic
- `easyasc/utils/Tensor.py` — slice/view creation
- `agent/example/kernels/a2/flash_attn_score.py` — practical use of sliced sub + continuous exp/cast
