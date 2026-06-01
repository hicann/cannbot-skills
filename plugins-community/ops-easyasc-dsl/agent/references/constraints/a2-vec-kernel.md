# A2 Vec-Only Kernel Authoring

Read this file when writing or debugging a pure vec kernel on a2 (`easyasc.a2`) with no cube stage.
Typical targets are elementwise transforms, bit-level float analysis, scalar-threshold gating, and quantization-style postprocess.

Do not use this file as the main guide for mixed cube/vec kernels.
If cube is involved, start from `agent/references/constraints/a2-device.md` and the matching pattern file instead.

## Goal

Capture the stable authoring rules for a2 vec-only kernels so that:
- the kernel body starts from the right minimal structure
- UB buffers are chosen intentionally
- `compare_scalar`, `select`, `reinterpret`, and `cast` are used with the repository's real semantics
- exact numeric contracts are not delegated to simulator rounding by accident

## 1. Use this layer when

This file is the right first read when:
- the whole kernel is `GM -> UB vec ops -> GM`
- there is no `@vf`, no `Reg`, and no cube handoff
- the logic is mostly elementwise, flag-driven, or bit-driven
- the output contract depends on thresholding, saturation, or explicit rounding

Read another file first when:
- you need row-wise reductions or narrow-broadcast arithmetic
  - then also read `agent/references/constraints/vec-reduction-a2.md`
  - and `agent/references/constraints/vec-stride.md`
- you need explicit vec mask behavior
  - then read `agent/references/constraints/mask.md`
- you need cube -> vec or vec -> cube ownership
  - then read `agent/references/constraints/a2-device.md`
  - and the matching file under `agent/references/patterns/`

## 2. Minimal kernel skeleton

Stable pure-vec structure on a2:

```python
@kernel()
def vec_kernel(x: GMTensor, y: GMTensor, total: Var):
    data = Tensor(DT.float, [1, TILE], Position.UB)
    work = Tensor(DT.float, [1, TILE], Position.UB)
    flag = Tensor(DT.uint8, [1, TILE], Position.UB)

    with vec_scope():
        n_tiles = CeilDiv(total, TILE)
        tile_per_core = CeilDiv(n_tiles, GetVecNum())
        tile_start = Var(tile_per_core * GetVecIdx())
        tile_end = Min(tile_start + tile_per_core, n_tiles)

    dup(...)

    with auto_sync():
        for t in range(tile_start, tile_end):
            n1 = Var(t * TILE)
            n_valid = Min(total - n1, TILE)
            data <<= x[n1:n1 + n_valid]
            # vec compute on UB
            y[n1:n1 + n_valid] <<= work
```

What this skeleton gets right:
- `vec_scope()` decides tile ownership across vec lanes before the loop
- constants are initialized once with `dup(...)`
- the inner loop keeps all work in UB
- tail handling stays local through `n_valid`

## 3. UB buffer selection rules

For pure vec kernels, prefer plain `Tensor(..., Position.UB)` by default.
Do not start from `DBuff` unless you truly need staged overlap or lookahead.

Useful buffer categories:
- data tiles: `Tensor(DT.float, [1, TILE], Position.UB)`
- temporary compute buffers: same dtype and shape as the data tile
- compare/select flags: `Tensor(DT.uint8, [1, TILE], Position.UB)`
- bit masks for reinterpret paths: `Tensor(DT.uint32, [1, TILE], Position.UB)` or another width-matched integer view
- final integer staging for exact rounding: `Tensor(DT.int, [1, TILE], Position.UB)`

Practical rule:
- if the whole tile is consumed and produced once per loop iteration, `Tensor` is usually enough
- if a buffer lifetime crosses iterations or producer/consumer stages, reconsider the topology before adding double buffering

## 4. Stable vec control idioms

### 4.1 `compare_scalar` + `select`

Use `compare_scalar` to build `uint8` flag tensors, then use `select` to route values.

Important repository behavior:
- `compare_scalar(...)` ignores the current vec mask
- `select(...)` also ignores the current vec mask
- selection is controlled only by the explicit `uint8` flag tensor
- on current a2 hardware/runtime, do not rely on `uint8 -> float` casts for compare flags;
  keep mask-controlled float paths in `compare_scalar(...) + select(...)`

This makes them the stable control-flow building blocks for pure vec kernels.

Typical uses:
- finite vs non-finite split
- underflow / overflow gating
- sign-dependent bias selection
- replacing invalid values before a bit reinterpret path

### 4.2 Non-finite guarding

If the later path assumes finite floats, sanitize first:

```python
absub <<= x.abs()
compare_scalar(finiteflag, absub, FLOAT32_FINITE_MAX, CompareMode.LE)
select(workub, finiteflag, x, 0.0)
```

Then restore original non-finite values at the end:

```python
select(outub, finiteflag, outub, x)
```

This avoids pushing `NaN`/`Inf` through exponent extraction or scale math while
keeping the control constant finite.

## 5. Bit-level float analysis with `reinterpret`

For exponent/mantissa logic, use `reinterpret(...)` instead of float arithmetic guesses.

Stable pattern from `agent/example/kernels/a2/to_hif8_torch.py`:

```python
x_u16 = workub.reinterpret(DT.uint16)
exp_u16 = expub.reinterpret(DT.uint16)
mask_u16 = expmask.reinterpret(DT.uint16)
vand(exp_u16, x_u16, mask_u16)
```

Useful rules:
- `reinterpret` is a view change, not a numeric cast
- it rescales the second dimension by dtype-width ratio
- it is legal on UB here
- it does not support `L0C`

When extracting absolute exponent-style metadata:
- use `vnot` + `vand` on the reinterpreted integer view
- then reinterpret to `DT.int` or another arithmetic dtype only after the bit pattern is where you want it

## 6. Exact rounding: do not over-trust default vec `cast`

Default `cast(...)` is convenient for ordinary dtype conversion, but it should not be treated as a proof of a higher-level numeric contract.

When the formula explicitly requires a rounding rule such as:
- `sign(x) * floor(abs(x) + 0.5)`
- round-half-away-from-zero
- quantization followed by scale restore

prefer an explicit sequence.

Stable sequence:

```python
outub <<= x / scale
compare_scalar(nonnegflag, outub, 0.0, CompareMode.GE)
select(biasub, nonnegflag, plus_halfub, -0.5)
outub <<= outub + biasub
cast(intub, outub, round_mode=RoundMode.TRUNC)
cast(outub, intub)
outub <<= outub * scale
```

Why this is safer:
- the sign-dependent `+0.5 / -0.5` encodes the formula directly
- `RoundMode.TRUNC` is only used for the final integer drop
- the result no longer depends on the simulator's interpretation of a more implicit rounding mode

Practical rule:
- use direct `cast(dst, src)` when the formula only needs a normal dtype conversion
- use an explicit bias + `TRUNC` path when the rounding rule itself is part of the contract
- if the decision came from a `uint8` compare flag, materialize the float branch with `select(...)`;
  do not plan a follow-up `uint8 -> float` cast

## 7. Tile-size and tail heuristics

For float vec-only kernels, `TILE = 512` is a good default starting point:
- simple to reason about
- comfortably small for a2 UB
- large enough to amortize fixed per-tile work

For tail handling:
- keep `n_valid = Min(total - n1, TILE)`
- load/store through GM slices using that tail width
- avoid adding a separate tail kernel unless the contract truly needs special handling

Do not optimize tile size first.
Get the contract right with one simple tile size, then revisit only if UB pressure or runtime suggests it.

## 8. When a vec-only kernel stops being "simple"

Escalate to another focused file when you hit one of these signs:
- wide `[M, 128]` buffers interacting with narrow `[M, 8]` buffers
  - read `agent/references/constraints/vec-stride.md`
- row max / row sum / online normalization
  - read `agent/references/constraints/vec-reduction-a2.md`
- temporary partial masks or masked writeback behavior
  - read `agent/references/constraints/mask.md`
- cross-stage workspace reuse or delayed consumer logic
  - read `agent/references/constraints/a2-device.md`
  - and the matching pattern under `agent/references/patterns/`

## 9. Concrete examples

Study first:
- `agent/example/kernels/a2/to_hif8_torch.py`

Study carefully but do not copy blindly:
- `agent/example/demo/a2/a2_hif8.py`

Why the demo is not enough for exact-contract work:
- it is useful for exponent extraction and threshold structure
- but it relies on a simpler cast/store path
- and it is not the best source when the exact PyTorch rounding contract must be preserved
