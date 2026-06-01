# Vector Mask Constraints

Read this file when implementing or debugging A2 vec mask behavior in the simulator.

## Goal

Keep mask semantics explicit so that:
- every vec op either clearly consumes the current mask or clearly ignores it
- `set_mask` / `reset_mask` state is reproducible across tests
- masked arithmetic matches the repository's agreed behavior instead of ad-hoc guesses

## 1. Current mask state

Each vec lane/runtime owns one `vector_mask: uint8[256]`.

Default state:
- all 256 entries are `1`

Interpretation:
- one mask slot corresponds to one logical vec element
- the active prefix depends on dtype / logical repeat payload size

Typical active prefix lengths:
- `float` / `int32`: `mask[0:64]`
- `half` / `int16`: `mask[0:128]`
- `int8` / `uint8`: `mask[0:256]`

Current simulator behavior:
- each repeat reuses the same mask prefix for the dtype
- the mask does **not** advance per repeat

## 2. `set_mask` / `reset_mask`

### Logical mask parts

Instruction semantics:
- `low` and `high` are treated as `uint64`
- bit `i` of `low` writes `mask[i]`
- bit `i` of `high` writes `mask[64 + i]`
- `mask[128:256]` stays unchanged by `set_mask`

Current bit order:
- bit `0` -> lowest mask slot in the covered range
- bit `63` -> highest slot in the covered range

### Stub call shape

Current a2 stub API is:
- `set_mask(mask_high, mask_low)`

So the emitted instruction swaps the two call arguments:
- `set_mask(hi, lo)` emits instruction fields `high=hi`, `low=lo`

This is why many call sites that only want the low 64-bit prefix use:
- `set_mask(0, low_mask)`

Validated repository test:
- `testcases/simulator/micro/test_simulator_v2_muladddst_mask.py`
- `testcases/simulator/micro/test_simulator_v2_vec_ops_extended.py`

### `reset_mask()`

- resets the full `mask[0:256]` back to `1`

## 3. Ops that consume the current mask

### 3.1 Unary / binary / unaryscalar

These ops consume mask at **dst writeback**.

Rule:
- `mask == 1` -> write the newly computed result
- `mask == 0` -> keep the old `dst` value unchanged

This currently applies to:

#### Unary
- `exp`
- `ln`
- `abs`
- `rec`
- `sqrt`
- `rsqrt`
- `vnot`
- `relu`

#### Binary
- `add`
- `sub`
- `mul`
- `div`
- `vmax`
- `vmin`
- `vand`
- `vor`
- `muladddst`

#### Unaryscalar
- `adds`
- `muls`
- `vmaxs`
- `vmins`
- `lrelu`
- `axpy`

#### Dup
- `dup`

#### Cast
- `cast`

Additional cast rule:
- the active cast-element domain is determined by the **wider** of `src` / `dst` dtypes
- for example, `float <-> half` uses `64` mask slots per full repeat, not `128`

### 3.2 Additive group reductions

For these ops, mask does **not** preserve old `dst`.
Instead, `mask == 0` means the corresponding **src slot contributes `0`**.

This currently applies to:
- `cadd`
- `cgadd`
- `cpadd`

Detailed behavior:
- `cadd`: masked-off elements do not contribute to the full-repeat sum; if **all** lanes in the active prefix are masked off, the destination scalar is **not overwritten**
- `cgadd`: masked-off elements do not contribute to their block-local sum; if one block's mask prefix is entirely zero, that block's destination scalar is **not overwritten**
- `cpadd`: masked-off elements are zeroed before flat-stream pairwise add

### 3.3 Max/min reductions with masked infinities

For these ops, masked-off source slots do not preserve old `dst` directly.
Instead they are replaced with an extreme sentinel before reduction.

#### Max family
Masked-off slots act like `-inf`.

This currently applies to:
- `cmax`
- `cgmax`

Detailed behavior:
- `cmax`: masked-off elements are replaced by `-inf`; if **all** lanes in the active prefix are masked off, the destination scalar is **not overwritten**
- `cgmax`: masked-off elements are replaced by `-inf`; if one block's mask prefix is entirely zero, that block's destination scalar is **not overwritten**

#### Min family
Masked-off slots act like `+inf`.

This currently applies to:
- `cmin`
- `cgmin`

Detailed behavior:
- `cmin`: masked-off elements are replaced by `+inf`; if **all** lanes in the active prefix are masked off, the destination scalar is **not overwritten**
- `cgmin`: masked-off elements are replaced by `+inf`; if one block's mask prefix is entirely zero, that block's destination scalar is **not overwritten**

## 4. Ops that explicitly do NOT consume the current mask

These ops currently ignore the vec mask entirely and follow only their own normal semantics:

- `select`
- `compare`
- `compare_scalar`
- `gather`
- `scatter`
- `sort32`
- `mergesort4`
- `mergesort_2seq`
- `brcb`

For the current V2 path, `compare(...)` / `compare_scalar(...)` and `select(...)`
also use their own explicit control tensor semantics:
- the current vec mask from `set_mask(...)` is irrelevant to them
- when the control tensor is a packed-bit `uint8` mask, the shape is `[..., N // 8]`,
  not an expanded `[..., N]` byte-per-element mask
- practical example: packed causal control for a `[64, 64]` half-tile is
  `Tensor(DT.uint8, [64, 8], Position.UB)`

## 5. Deprecated / pending pieces

### Deprecated
- `set_cmpmask`
  - deprecated because its role overlaps with `compare()` / `compare_scalar()` result generation
  - kept only for compatibility, not for simulator feature growth

### Pending / not yet decided
Mask semantics are still not fixed for:
- `cast`

If you touch new undecided ops later, document the decision here first.

## 6. Implementation rules

When adding mask support to a new vec op, decide which of these two patterns it belongs to:

### Pattern A: mask gates writeback
Use for elementwise ops.

Meaning:
- compute the full result
- write only where mask is `1`
- preserve old `dst` where mask is `0`

### Pattern B: mask zeroes src contribution
Use for additive reductions.

Meaning:
- masked-off `src` slots act like `0`
- reduction result is still written normally

Do not mix the two patterns casually.

## 7. Validation checklist

When testing mask behavior:
- verify default mask is all ones
- verify `set_mask` changes only `mask[0:128]`
- verify `reset_mask` restores full ones
- for writeback-gated ops, confirm masked-off lanes keep old `dst`
- for additive reductions, confirm masked-off lanes contribute zero instead of preserving old `dst`
