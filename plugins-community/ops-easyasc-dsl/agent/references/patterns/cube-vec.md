# Cube-to-Vec Pattern

> Generic baseline only. For a2 (b3) kernels, prefer `agent/references/patterns/a2-cube-vec.md`, which adds the mandatory a2 GM-workspace bridge and device-specific constraints.

Read this file when cube work produces an intermediate that vec work must postprocess before final writeback.

## Use this pattern when

- cube computes the main tile
- vec applies elementwise or row-wise postprocess
- the result is finally written out from the vec side

## Minimal flow

`GM -> L1 -> L0 -> L0C -> UB -> @vf -> GM`

## Ownership rule

The cube-to-vec handoff is a cross-side ownership edge.
Use explicit `CvMutex`.
Do not expect `auto_sync()` to replace it.

Stable repository mapping:
- `CvMutex(..., src_end_pipe=Pipe.FIX, dst_end_pipe=Pipe.V)`

## What usually matters most

- where the cube tile becomes a vec-visible UB tile
- whether the postprocess runs in `float` or after a cast
- standard half-row vec writeback for tails
- separate counters for the longer postprocess lifetime if needed

## a2 variant

On a2, `l0c_to_ub` is absent. The cube → vec handoff goes through GM workspace instead.
Read `agent/references/patterns/a2-cube-vec.md` for the full pattern.

Key differences from a5:
- `CvMutex(FIX → MTE2)` instead of `CvMutex(FIX → V)`
- GM workspace with `split_workspace` instead of UB DBuff from `l0c_to_ub`
- No `@vf` — vec operations are directly in the kernel body
- Each sub-block reads its own `TILE_M//2` rows via `GetSubBlockIdx()`

## Typical files to study

- `agent/example/kernels/a5/basic_cube_vec_mix.py` — a5 baseline
- `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- `agent/example/kernels/a5/matmul_rowwise_norm.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk_add1.py`
- `agent/example/kernels/a2/flash_attn_score.py` — a2 variant with GM bridge
