# Tail-Safety Constraints

Read this file when a kernel has tile tails, odd row splits, or partial GM boundaries.

## Goal

Keep tail handling correct without corrupting the stable local-tensor shape assumptions used by lowering and simulation.

## 1. Core rule

Apply `valid_m`, `valid_n`, and `valid_k` at GM read and write boundaries.
Do not shrink local tensor shapes for every tail tile.

Repository expectation:
- local buffers remain full tile-sized
- only the GM boundary slices use the tail sizes

## 2. Why this rule exists

Stable local tensor shapes make lowering and simulator behavior predictable.
If you start shrinking local buffers for tails, it becomes much easier to create shape drift between the intended logical tile and the actual staged buffer.

## 3. Standard cube -> vec half-row writeback pattern

For many cube -> vec kernels, use the standard half-row split:
- `half_rows = CeilDiv(valid_m, 2)`
- `row_begin = GetSubBlockIdx() * half_rows`
- `row_end = Min(row_begin + half_rows, valid_m)`
- `row_count = row_end - row_begin`

This keeps odd-row tails stable across the two vec subblocks.

## 4. Common symptoms of tail bugs

Tail issues often look like this:
- aligned shapes pass but odd shapes fail
- only the last tile is wrong
- one vec subblock is correct and the other is garbage
- output shape looks right but the boundary rows or columns are corrupted

When this happens, inspect the GM boundary slices first.
Do not start by changing the local buffer shape.

Special case:
- if the kernel is a normalized online softmax with running `row_max` / `row_sum`,
  GM-boundary slicing alone is not enough; invalid score columns must behave like
  `-inf` before `rowmax`
- read `agent/references/constraints/online-softmax-tail.md`

## 5. Quick checklist

Before accepting tail logic, verify:
- `valid_m`, `valid_n`, `valid_k` come from the current tile boundary
- local buffers still use full `TILE_*` shapes
- GM load boundaries use the valid sizes
- GM store boundaries use the valid sizes
- vec half-row split uses `CeilDiv` and clamps with `Min`
- at least one odd-size case has been tested
- for normalized online softmax, verify score-domain invalid columns are masked
  before `cmax`

## Files to study

- `agent/example/kernels/a5/basic_cube_vec_mix.py`
- `agent/example/kernels/a5/matmul_half_splitn_bias10p2_vf.py`
- `agent/example/kernels/a5/matmul_rowwise_norm.py`
- `agent/example/kernels/a5/vec_cube_abs_sqrt_matmul.py`
- `agent/example/kernels/a5/vec_unaligned_gm_to_ub_pad.py`
