# Datamove Constraints

Read this file when a kernel needs to move data between GM, UB, L1, or L0 using non-trivial transfer patterns.

## Goal

Choose the right datamove recipe so that:
- the publish path matches the downstream consumer's expected layout
- unaligned widths are handled by padding rather than by shrinking local tensors
- strided gathers avoid unnecessary host-side `permute` or `expand`
- internal workspace bridges stay explicit when on-chip reuse does not fit

## 1. ND publish (`ub_to_l1_nd2nz`)

Best for straightforward vec preprocess + cube consume.
- write subblock rows into UB, then publish with explicit `m_dst/n_dst/m_src/n_src`
- keep row mapping consistent with `GetSubBlockIdx()`
- in general vec preprocess, split into two half ranges for the two vector sides:
  - `half_rows = CeilDiv(total_rows, 2)`
  - vector side 0 handles `[0:half_rows]`
  - vector side 1 handles `[half_rows:total_rows]`
  - publish each half independently to the matching L1 row slice

Files to study:
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`

## 2. NZ publish (`ub.nz()`)

Use when input is already packed for NZ path.
Common flow:
- do vec compute in ND register form
- pack to NZ-friendly UB layout (`deinterleave`, `reg_to_ub`)
- publish with `l1 <<= ub.nz()`

Files to study:
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul_nz.py`

## 3. Unaligned width handling

For unaligned GM widths, allocate UB second dim to aligned width and rely on padded transfer behavior.
Do not shrink the UB tensor shape to the logical width.

For narrow a5 vec-only row kernels, a useful specialization is:
- keep the logical host contract as `[rows, H]`
- when `H < 64`, still stage the chunk in UB as `[rows, 64]`
- use `gm_to_ub_pad(..., burst_len_element=H, dst_stride=(64 - H) / C0)` to zero-pad each row on load
- run the same `@vf()` row logic against `row_stride = 64`
- write back with `ub_to_gm_pad(..., burst_len_element=H, src_stride=(64 - H) / C0)` so only the logical columns return to GM
- this is a good fit when the vec math is row-recursive and you want one shared `@vf()` body for both wide rows and narrow `H < 64` rows

Practical limit:
- for float32, this padding shortcut is cleanest when the row-width gap is expressible in `C0=8` units
- it does not solve the wider-column tail case by itself when `H >= 64` but `H % 64 != 0`

Files to study:
- `agent/example/kernels/a5/vec_unaligned_gm_to_ub_pad.py`
- `agent/example/kernels/a5/chunk_row_cumsum.py`

## 4. Strided GM gather without host `permute`

When logical rows are separated by a fixed stride in flattened GM, use `gm_to_ub_pad` directly:
- set `n_burst` to the number of logical rows
- set `burst_len_element` to the contiguous row width
- set `src_stride_element` to `full_row_step - burst_len_element`
- keep `dst_stride=0` when the UB row shape already matches the aligned burst footprint

This is the main way to preserve a reshape-only host contract for attention-style layouts such as `key:[B,S,H,D]` and `prob:[BH,S]`.

## 5. Internal workspace bridge for single-kernel fusion

If one kernel stage produces data on `MTE3` and a later stage must reread it through `MTE2`, materialize that intermediate in GM workspace instead of trying to keep it purely local.

Stable attention pattern:
- keep `qk_tmp:[BH,S]` as float workspace for the three-pass softmax
- store `p.half()` into `prob_tmp:[BH,S]` workspace
- add an explicit stage boundary before reloading `prob_tmp`
- perform the final value scaling from that half workspace so the `p.half().float()` contract stays exact

For the final vec-only `prob_tmp -> value -> out` stage:
- keep the whole nested reload/compute/writeback chain inside one outer `auto_sync()`
- make DBuff slot ownership explicit through the ready/valid handshake rule
- verify both simulator execution and generated C++ declarations before removing manual barriers

If the delayed reuse fits in one tile of on-chip lifetime, prefer an on-chip lookahead bridge:
- keep the stage-1 operand needed again by stage-2 resident in `L1` / `TBuff`
- publish the vec-produced fp8 probability tile directly into an `L1` slot for the second cube matmul
- buffer per-tile rescale state in the same delayed slot family as the later consumer

Do not republish a freshly packed fp8 UB tile straight to L1 when exact downstream reuse matters; the packed UB layout can differ from the ND view expected by the later cube path.

Files to study:
- `agent/example/kernels/a5/test_mla_entire.py`
