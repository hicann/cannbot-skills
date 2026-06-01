# Authoring Facts

Use this file for repository-wide hard rules, DBuff quick-models, and a2-specific bridge reminders.
Use constraint files for the why; use this page for the exact rule.

## Authoring hard rules

- `splitk`, `splitn` must be `>= 32`
- matmul `L0C` destination must use row offset `0`
- matmul accumulation stays in `float` unless there is a strong, documented reason
- vec writeback half-row split:
  ```text
  half_rows  = CeilDiv(valid_m, 2)
  row_begin  = GetSubBlockIdx() * half_rows
  row_end    = Min(row_begin + half_rows, valid_m)
  row_count  = row_end - row_begin
  ```
- local buffers stay full-tile sized; only GM read/write uses `valid_m` / `valid_n`
- different loop-owned lifetimes must use different counters; same-pair operands may share one
- `CvMutex` / `VcMutex` `depth` must match the real reusable slot count of the guarded buffer family; plain `Tensor` means `depth=1`, while `DBuff` / `TBuff` / `QBuff` only justify `2` / `3` / `4` when the code really rotates across those physical slots
- `with atomic_add():` narrows the GM-store critical section when accumulating
- a2: no `@vf`, no `Reg` / `RegList` / `MaskReg`, no `l0c_to_ub`, no `ub_to_l1_nd2nz`, no `ub_to_l1_nz`, no `micro`
- a2: `l0c_to_l1` does not support `float` destination
- a2: `l0c_to_l1` still exists for supported destination dtypes, so pure cube-side matmul dependencies should still prefer `L0C -> L1` reuse over inventing a UB detour
- a5: vec-side math belongs in `@vf()` / `micro` / `ub_to_ub` / sort-family ops, not raw a2-style vec code in the kernel body
- use a sufficiently large finite sentinel (for example `-1.0e30`) instead of literal `float("-inf")` for score-domain invalidation

## DBuff capacity quick-models

- `splitk` L0A elements: `TILE_M * SPLIT_K * 2`
- `splitk` L0B elements: `TILE_N * SPLIT_K * 2`
- `splitn` L0A elements: `TILE_M * TILE_K * 2`
- `splitn` L0B elements: `SPLIT_N * TILE_K * 2`
- `L0C` DBuff bytes: `2 * TILE_M * TILE_N * sizeof(accum_dtype)`
- byte budget: `elements * dtype_size_bytes <= local_cap_bytes`

Require:
- `L0A <= 64 KB`
- `L0B <= 64 KB`
- `L0C <= device cap`
- `UB <= device cap`

Stable large-`K` aligned MKNK pattern:
- `TILE_M=128, TILE_N=256, TILE_K=256, SPLIT_K=64`
- L0A `32 KB`, L0B `64 KB`

## a2 data-path reminders

- cube -> vec mandatory bridge: `l0c_to_gm_nz2nd` -> GM workspace -> `gm_to_ub_pad`
- vec -> cube mandatory bridge: `ub_to_gm_pad` -> GM workspace -> `gm_to_l1_nd2nz`
- typical workspace: `split_workspace(dtype, [GetCubeNum(), 2, TILE_M, TILE_N])`
- UB is undefined at kernel entry; initialize with `dup(...)`
- do not rely on `muls(*, 0.0)` to clear uninitialized UB
- `dup` on broadcast-format `[M, 8]` fills only 64 of 512 elements; initialize in the natural compute format
- vec reduction `cmax` outputs dense scalars; apply `brcb` before using them in `sub` / `div`
- each sub-block has independent 192 KB UB; use `GetSubBlockIdx()` to split work
- `ub_to_ub` is not a safe generic copy for `[M, 1]` scalar state; use `add(tmp, src, ub_zero_s)` instead

## Deeper references

- `agent/references/constraints/tiling.md`
- `agent/references/constraints/counters.md`
- `agent/references/constraints/a2-device.md`
- `agent/references/constraints/a5-device.md`
- `agent/references/patterns/a2-cube-vec.md`
