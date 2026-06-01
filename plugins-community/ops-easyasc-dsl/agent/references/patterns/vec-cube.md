# Vec-to-Cube Pattern

> Generic baseline only. For a2 (b3) kernels, prefer the a2-specific patterns under `agent/references/patterns/` (e.g., `a2-cube-vec.md`) and read `agent/references/constraints/a2-device.md` for device-side rules.

Read this file when vec work preprocesses data before cube consumes it in a later matmul stage.

## Use this pattern when

- the formula needs elementwise or row-wise preprocessing first
- the cube stage should consume the transformed result
- the host-side contract should stay reshape-only instead of doing a heavy layout transform outside the kernel

## Minimal flow

`GM -> UB -> @vf -> UB -> L1 -> L0 -> L0C -> GM`

## Ownership rule

The vec-to-cube publish is a cross-side ownership edge.
Use explicit `VcMutex`.
Do not expect `auto_sync()` to replace it.

Stable repository mapping:
- `VcMutex(..., src_end_pipe=Pipe.MTE3, dst_end_pipe=Pipe.FIX)`

## What usually matters most

- whether the publish path is ND or NZ
- whether the host-side layout stays reshape-only
- how subblock rows are split between vec sides
- whether the preprocessed value must remain in half or float before cube consume

## Typical files to study

- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul_nz.py`
- `agent/example/kernels/a5/recompute_wu_cube_vec.py`
