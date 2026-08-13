# Simulator Datamove Footprint Guards

This note tracks the boundary-checking model for simulator datamove and cube
layout operations.

## Goal

Every simulator datamove, layout decode, and layout encode path should reject an
access that would read or write beyond the real tensor storage it is about to
touch.

## Burst Footprint Rule

Burst copies are not dense rectangles. For `n_burst` bursts, each burst touches
exactly `burst_len` bytes. If the inter-burst step is `step`, the required byte
footprint is:

```text
(n_burst - 1) * step + burst_len
```

The skipped tail between the end of a burst and the next burst start is not
accessed. It is legal for that skipped tail to extend beyond tensor storage.

For `gm_to_ub_pad`, destination padding is actually zeroed up to the 32-byte
aligned burst size, so the source footprint uses `burst_len` while the
destination footprint uses `align32(burst_len)`.

On A2, `gm_to_ub_pad` and `ub_to_gm_pad` additionally require the resolved
`n_burst` in `[0, 4095]`. Validate this before computing a footprint or entering
the transfer loop. `n_burst=0` is an empty transfer; 4095 is valid and 4096 is
rejected. A5/A5PR retain their existing behavior.

## Layout Footprint Rule

NZ/ZZ helpers should check the physical panel footprint implied by the layout
decoder or encoder, not the logical `M * N` shape:

- NZ full-panel decode/encode: `ceil(N / C0) * stride_m * C0`.
- ZZ full-panel decode/encode: `ceil(M / row_block) * align_N * row_block`.
- Sparse tail writers, such as `ub_to_l1_nd2nz`, should check the last actually
  written element. The unused `C0 - tail` columns in the final block are not
  touched.

On A5, the C310 `InitConstValue` implementation only supports 16-bit and wider
matrix element types (`half`, `bfloat16`, `float`, signed/unsigned 16-bit, and
signed/unsigned 32-bit). The simulator must reject byte-typed L1 constant fills
instead of modeling them as successful flat writes. Reinterpret byte storage as
a supported 16-bit type when a real constant fill is required.

A5 byte L1-to-L0 transfers also require a C0-aligned extent when the source
window starts at a nonzero row or column offset. A partial K tail is valid at
offset zero, but a split-K tail such as `src_col0=256, n_dst=1` is not. Enforce
this resolved-runtime restriction before decoding the L1 source window.

## Runtime GM Slice View Rule

`slice_gm_tensor` materializes a runtime view before later pipe tasks consume the
GM slice. The view must be large enough for strided row/plane access, so its
linear coverage is:

```text
1 + sum((span_i - 1) * stride_i)
```

not `product(spans)`.

The runtime checks every dimension first:

```text
0 <= offset_i
0 <= span_i
offset_i + span_i <= src_dim_i
```

Then it checks `flat_offset + view_footprint <= parent.numel()`. Do not truncate
the view to the parent storage size with `min(...)`; that hides the real source
of an out-of-bounds slice and can later surface as a confusing pipe footprint
failure.

Runtime view identity must include the source tensor, flat offset, logical
offsets, source dims, and spans. A tail slice such as `8 x 128` and a full slice
such as `128 x 128` may have the same flat offset in different loop iterations;
they must become distinct runtime views.

## Coverage

Runtime simulator guards should cover:

- byte burst moves: `gm_to_ub_pad`, `ub_to_gm_pad`, `ub_to_l1`, `ub_to_ub`
- UB/L1 layout moves: `ub_to_l1_nd2nz`, `ub_to_l1_nz`
- GM/L1/L0 cube moves: `gm_to_l1_nd2nz`, `l1_to_l0`
- L0C reads/writes: `mmad`, `l0c_to_gm_nz2nd`, `l0c_to_l1`, `l0c_to_ub`
- constant writes: `set_constant_to_l1`
- runtime GM slice materialization: `slice_gm_tensor`

Static DSL guards are useful when all dimensions and tensor shapes are known,
but simulator runtime checks are the required fallback for dynamic `Var` values.
