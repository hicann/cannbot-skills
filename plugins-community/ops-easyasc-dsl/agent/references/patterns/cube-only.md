# Cube-Only Pattern

Read this file when the formula can stay entirely on the cube side without vec or micro postprocessing.

## Use this pattern when

- the formula is basically a matmul or cube-local reduction path
- no vec-side nonlinear postprocess is required
- no vec-side publish/consume stage is needed

## Minimal flow

`GM -> L1 -> L0 -> L0C -> GM`

Typical steps:
1. load operand tiles into `L1`
2. move to `L0A` / `L0B`
3. run `matmul` / `mmad`
4. write result back from `L0C`

## What usually matters most

- exact layout choice at matmul call site
- tile/core split legality
- `shape_bindings` when scalar dims repeat
- keeping the output tile ownership stable

## Typical files to study

- `agent/example/kernels/a5/matmul_float_mmad.py`
- `agent/example/kernels/a5/matmul_kmkn_fp32_out.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitn.py`
- `agent/example/kernels/a5/matmul_mknk_2dgrid_splitk.py`
