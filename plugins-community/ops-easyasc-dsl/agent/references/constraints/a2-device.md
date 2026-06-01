# a2 Device Constraints

Read this file when writing a kernel targeting a2 (`easyasc.a2`, device `b3`).
Do not read it for a5 kernels — the two architectures differ significantly.

## Goal

Capture all a2-specific differences from a5 so that:
- a5 patterns are not blindly reused on a2
- the correct data path, buffer, and vec model is chosen from the start

## 1. Hardware budgets and missing features

For exact per-device capacities (`L0A`, `L0B`, `L0C`, `UB`, `L1`, cube core count, vec sub-blocks per core) see `agent/references/facts-device-runtime.md`. For the a2-missing a5 features (`@vf`, `Reg` / `RegList` / `MaskReg`, `l0c_to_ub`, `ub_to_l1_nd2nz`, `ub_to_l1_nz`, `micro`, `l0c_to_l1(float)`), see `agent/references/facts-authoring.md`.

Key consequence for tile strategy: a5 tile strategies that fit L0C at 256 KB overflow on a2. Always verify `TILE_M * TILE_N * 4 * 2 <= 128 KB` for float L0C DBuff.

## 2. Cube → vec data path on a2

Since `l0c_to_ub` is absent and `l0c_to_l1(float)` is blocked:

**Mandatory path**: `L0C → GM workspace → UB`
- Cube: `l0c_to_gm_nz2nd` writes float L0C to a GM workspace buffer (FIX pipe)
- Vec: `gm_to_ub_pad` reads from GM workspace into UB (MTE2 pipe)

**CvMutex configuration**:
```
CvMutex(0, src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.MTE2)
```

This differs from the a5 standard (`dst_end_pipe=Pipe.V`) because the vec side's
first consumer operation is `gm_to_ub_pad` on MTE2, not a V-pipe compute.

**GM workspace design**:
- Use `split_workspace(DT.float, [GetCubeNum(), 2, TILE_M, TILE_N])` for pingpong
- Dimension `2` provides double-buffering slots
- Index as `ws[GetCubeIdx(), slot, row_slice, col_slice]`
- Cube writes full `TILE_M` rows; each vec sub-block reads `TILE_M // 2` rows

## 3. Vec → cube data path on a2

Since `ub_to_l1_nd2nz` and `ub_to_l1_nz` are a5-only, a2 cannot publish vec output
directly from `UB` to `L1`.

**Mandatory path for delayed vec -> cube reuse**: `UB -> GM workspace -> L1`
- Vec: `ub_to_gm_pad` writes the vec result into a GM workspace buffer (MTE3 pipe)
- Cube: `gm_to_l1_nd2nz` reloads that workspace tile into `L1` (MTE2 pipe)
- Cube then continues with the normal `l1_to_l0 -> mmad` path

This is the stable bridge for patterns such as:
- stage 1 cube computes score
- vec computes `p_j`
- stage 2 cube consumes delayed `p_j @ v_j`

Recommended synchronization:
```python
VcMutex(1, src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)
```

Why `dst_end_pipe=Pipe.FIX`:
- vec producer truly ends on `ub_to_gm_pad` (`MTE3`)
- for delayed-consumer kernels, conservative release is simpler and safer
- the cube side may reload from GM on `MTE2`, then continue through `L1 -> L0 -> MMAD -> FIX`
- freeing only after the cube stage finishes avoids premature workspace reuse

Workspace design mirrors the cube -> vec bridge:
- use `split_workspace(dtype, [GetCubeNum(), 2, TILE_M, TILE_N])`
- vec sub-block 0 writes rows `[0:HALF_M]`
- vec sub-block 1 writes rows `[HALF_M:TILE_M]`
- cube waits on the `VcMutex`, then reloads the full tile from the same slot

Important synchronization fact from the simulator/runtime model:
- cube-side `wait_vec()` succeeds only after **both** vec lanes have produced their tokens
- this makes a full-tile cube reload safe after the two half-row vec writes complete

## 3a. Cube-side matmul dependency reuse rule on a2

The same high-level rule as a5 still applies on a2:
- if one cube matmul only feeds a later cube matmul, prefer keeping that dependency on the
  cube-side path instead of inventing a UB detour

On a2 the practical boundary is stricter:
- `l0c_to_l1` exists
- but `l0c_to_l1(float)` is blocked on `b*` devices
- and `l0c_to_ub` is absent

So the stable rule is:
- if the intermediate dependency can be republished to `L1` in a supported dtype, prefer
  `L0C -> L1 -> L0 -> mmad`
- do **not** route a pure cube-side dependency through `UB`
- only fall back to GM workspace when the dependency truly needs a vec-side stage or when the
  required `L1` destination dtype is unsupported

Practical implication:
- a2 still benefits from the same "avoid unnecessary `L0C -> UB -> L1` thinking" rule
- the difference from a5 is not the existence of `l0c_to_l1`, but the narrower dtype surface

## 4. Sub-block execution model

Each cube core has 2 vec sub-blocks. On a2:
- Each sub-block has its own **independent 192 KB UB**
- Vec instructions in the kernel body execute on both sub-blocks simultaneously
- Use `GetSubBlockIdx()` to compute different GM offsets for each sub-block
- Each sub-block reads its own `TILE_M // 2` rows from the workspace

Typical pattern:
```python
sb = GetSubBlockIdx()
sb_row = Var(sb * HALF_M)
ub_data <<= workspace[cube_idx, slot, sb_row:sb_row + HALF_M, 0:TILE_N]
# ... vec processing ...
out_row = Var(q_row + sb_row)
output[out_row:out_row + HALF_M, col:col + TILE_N] <<= ub_out
```

## 5. Vec operations available on a2

All vec operations work on UB tensors directly (no Reg intermediate):

| Category | Operations |
|----------|-----------|
| Unary | `exp`, `ln`, `abs`, `rec`, `sqrt`, `rsqrt`, `relu` |
| Binary | `add`, `sub`, `mul`, `div`, `vmax`, `vmin` |
| Scalar | `adds`, `muls`, `vmaxs`, `vmins`, `axpy` |
| Reduction | `cmax`, `cgmax`, `cmin`, `cgmin`, `cadd`, `cgadd`, `cpadd` |
| Broadcast | `dup`, `brcb` |
| Datamove | `gm_to_ub_pad`, `ub_to_gm_pad`, `ub_to_ub` |
| Cast | `cast` |
| Compare | `compare`, `compare_scalar`, `set_cmpmask` |
| Select | `select` |
| Mask | `set_mask`, `reset_mask` |

### 5.1 Bitpacked `uint8` masks and scalar-fill `select`

For a2 vec-side masking that is naturally bitpacked by column prefix, a practical
shape is `Tensor(DT.uint8, [HALF_M, TILE_N // 8], Position.UB)`.

Stable rules:
- `compare(...)` / `compare_scalar(...)` produce packed-bit `uint8` masks for this path, and `select(...)` consumes the same packed form
- do **not** treat `select` control as an expanded `[HALF_M, TILE_N]` byte-per-element `0/1` tensor when the hardware contract is bitpacked
- for rowwise prefix or suffix masks, prefer synthesizing mask bytes with `compare_scalar(...)` against a reusable integer column-index tensor instead of populating every mask byte with `SetValueTo(...)`
- `dup` does **not** support `uint8`, but whole-mask initialization can often be done efficiently through a width-compatible integer reinterpret such as `mask.reinterpret(DT.int)`
- `SetValueTo` still writes only the first element of the destination view, so per-byte loops are correct as a fallback but expensive in simulator control flow
- `select(..., SelectMode.TENSOR_SCALAR)` only uses the first scalar value from `src2`
- for score-domain invalidation, a small scalar-fill source such as `Tensor(DT.float, [1, 64], Position.UB)` initialized with `dup(..., neg_large)` is sufficient; `src2` does not need to match the full score tile shape

Use this when:
- the control comes from an explicit packed `uint8` mask tensor
- the fallback branch is a single scalar fill value like `neg_large` or `0.0`
- you want to avoid allocating a full-size fallback tensor just for `select`
- you want to amortize mask construction by building it once and reusing it across many vec operations

Diagonal causal reuse rule:
- for repeated diagonal masking on `[HALF_M, TILE_N]` tiles, it is valid to build one packed mask `Tensor(DT.uint8, [HALF_M, TILE_N // 8], Position.UB)` and apply one full-tile `select(...)`
- if the diagonal geometry is stable across iterations, prebuild that packed mask once and reuse it
- for a2 dense backward causal tails, full `128x128` q-tiles have stable subblock geometry and can reuse one static packed diagonal mask per `GetSubBlockIdx()`
- tail `M` tiles cannot blindly reuse that full-tile mask when the kernel splits rows with `half_rows = CeilDiv(valid_m, 2)`, because the second subblock `row_begin` shifts away from the fixed `64`
- in that case, keep the same packed-mask + one-`select(...)` pattern, but rebuild the packed mask dynamically only for the tail tile

Column-index template trick:
- when the packed mask compares against contiguous column ids, build a reusable integer template once, e.g. `Tensor(DT.int, [1, TILE_N], Position.UB)`
- a compact a2 pattern is to reinterpret that tensor as `int64` and write two `int32` column ids per store while initializing `[0, 1, 2, ..., TILE_N - 1]`
- later `compare_scalar(mask[row:row + 1, 0:TILE_N // 8], col_idx, valid_cols, CompareMode.LT)` synthesizes the whole row mask without scalar-per-byte loops

## 6. UB initialization with `dup`

On a2, UB contents are undefined at kernel entry. There is no zero-initialization guarantee.
Operations like `muls(ub, ub, 0.0)` are unreliable on uninitialized buffers because
`0.0 × NaN = NaN`.

**Use `dup(tensor, scalar_value)` to fill a UB tensor with a known value.**

For the current validated a2 softmax kernels, a sufficiently negative finite
sentinel such as `neg_large = -1.0e30` is the default way to materialize
score-domain invalidation and running-max identity state. Treat literal
`float("-inf")` as simulator-convenient but hardware-fragile.

`dup` signature: `dup(dst: Tensor, value: Union[int, float, Var], ...)`

The `dup` operation uses the same stride inference as other vec operations. It fills
blocks and repeats according to `infer_repeat(dst)` and the auto-inferred strides.

Coverage analysis for common shapes (float, C0=8):

| Shape | repeat | blk_stride | Elements covered | Buffer size | Complete? |
|-------|--------|------------|-----------------|-------------|-----------|
| `[64, 1]` | 1 | 1 | 1×8×8 = 64 | 64 | ✓ |
| `[64, 8]` | 8 | 0 | 8×1×8 = 64 | 512 | ✗ (only 1/8) |
| `[64, 64]` | 64 | 1 | 64×8×8 = 4096 | 4096 | ✓ |
| `[64, 128]` | 128 | 1 | 128×8×8 = 8192 | 8192 | ✓ |

**Warning**: `dup` on `[M, 8]` (broadcast-format) only fills 64 out of 512 elements.
This is acceptable if the tensor is only consumed via `blk_stride=0` operations (like `sub`),
which also only read those 64 positions. But it is incorrect if you later attempt a full
element-wise operation over the entire 512-element buffer.

**Practical rule**: initialize in the natural computation format (`[M, 1]` for scalars,
`[M, N]` for data), not in the broadcast format.

### Placement within `auto_sync`

A `dup` placed inside the outer loop but before the inner loop (e.g. for per-M-tile
reinitialization) is safe. It generates an extra V→MTE3 autosync event pair because
auto_sync sees the V-pipe `dup` and the inner loop's MTE3 `ub_to_gm_pad` as a
producer-consumer pair within the same scope.

This extra event is harmless: `dup` completes before the inner loop starts, so the
MTE3 event is already satisfied when the first store executes. The inner loop's own
V→MTE3 events for the actual `exp → cast → store` flow are managed separately with
different sync-key groups.

```python
with auto_sync():
    for gmt in range(mt_begin, mt_end):
        # ... variable declarations ...
        dup(ub_rmax_s, neg_large)  # V-pipe, safe here
        for nt in range(0, tiles_n):
            # ... cube + vec tile processing ...
```

## 7. Scalar computation

- `scalar_sqrt(Var)` requires `Var.dtype == Datatype.float`; integer Vars will TypeError
- For `1/sqrt(D)`, prefer passing the precomputed float value as a kernel parameter
- `muls(ub, ub, scale_var)` accepts Var(float) as the scalar argument

## 8. Copying scalar-format UB state

For row-scalar buffers on a2, the natural vec format is `[M, 1]`.
This format is safe for vec binary ops like `vmax`, `add`, `sub`, and `exp`,
but it is **not** a safe format for `ub_to_ub` snapshots.

Why:
- `ub_to_ub` infers burst length in units of `C0` blocks
- for float `[64, 1]`, that implies copying one full 8-element block per row
- this silently corrupts or repeats scalar row state instead of copying one scalar per row

Practical rule:
- if you need to snapshot `[M,1]` running state such as `prev_row_max`,
  copy it with a vec binary op and an explicit zero buffer:

```python
dup(ub_zero_s, 0.0)
add(tmp_scalar_buf, ub_rmax_s, ub_zero_s)
```

- then update or transform `tmp_scalar_buf` with more vec ops
- do not use `ub_to_ub` as a generic copy for `[M,1]` scalar state

## Files to study

- `agent/example/kernels/a2/qk_matmul_batched.py` — cube-only a2 baseline
- `agent/example/kernels/a2/flash_attn_score.py` — cube → vec with GM workspace bridge
- `agent/example/kernels/a2/flash_attn_score_iter.py` — running max with `dup` initialization and `[M,1]` vmax
- `agent/example/kernels/a2/flash_attn_unnorm.py` — delayed `expdiff` + final numerator accumulation on a2
- `agent/example/kernels/a2/flash_attn_full.py` — delayed numerator accumulation plus running sum/final divide on a2
- `agent/example/kernels/a2/flash_attn_full_pj_hif8_causal.py` — rowwise diagonal-tile causal masking plus `active_tiles_n` skip on a2, now with shared vec-side `DBuff` scratch for stage-1/2 overlap
- `agent/example/kernels/a2/flash_attn_full_pj_half_block32_causal.py` — block-32 causal masking plus `active_tiles_n` skip on a2, also using the shared vec-side `DBuff` scratch lineage
- `easyasc/stub_functions/vec/dupbrcb.py` — dup stub (validates dst is UB, infers repeat)
- `easyasc/simulator_v2/ops/vec/v.py` and `easyasc/simulator_v2/ops/vec/_legacy_vpipe.py` — current vec runtime path for dup-style execution
