# Vec Fixed-Group Reduction and Broadcast

## Applies when

Use this A2 vec pattern when a row is partitioned into independent fixed-size
groups and each group produces its own scalar that is reused only inside that
group. Common widths are 64, 32, or 16 fp32 lanes.

## Logical dataflow

```text
row -> fixed groups -> one scalar per group -> per-group scalar transform
    -> broadcast inside the same group -> elementwise group use
```

This is different from a full-row reduction: do not add the group scalars
together unless the contract explicitly asks for one row statistic.

## Physical invariants

- Keep narrow logical groups in 64-lane-aligned physical fp32 storage when
  later vec operations use normal repeat mode.
- `count_per_rep` selects the live prefix inside one repeat; it does not choose
  where the next group starts.
- `*_rep_stride` is measured in 32-byte data blocks and moves to the next group.
- Every later operation walking the same in-place segments needs the matching
  explicit repeat stride.
- Scalar scratch used by default `dup`, `select`, or broadcast behavior remains
  physically 64 lanes wide even when the logical group count is smaller.

## Minimal skeleton

For four consecutive group32 segments in one physical fp32 row of width 128:

```python
cmax(scale_s, x_ub[0:1, 0:32], repeat=4,
     src_rep_stride=4, count_per_rep=32)
brcb(scale_brcb, scale_s, repeat=1,
     dst_blk_stride=1, dst_rep_stride=8)
div(
    x_ub[0:1, 0:32],
    x_ub[0:1, 0:32],
    scale_brcb[0:4, 0:8],
    repeat=4,
    dst_rep_stride=4,
    src1_rep_stride=4,
    src2_rep_stride=1,
    count_per_rep=32,
)
```

For fp32, `rep_stride=4` advances `4 * 8 = 32` elements. The four repeats
therefore start at columns 0, 32, 64, and 96.

## Failure signatures

- every second group reads the wrong segment: count and repeat stride were
  treated as the same control;
- a narrow `[groups, 32]` or `[groups, 16]` buffer overruns under a later normal
  vec op: logical width was used as physical storage without checking the full
  instruction chain;
- reduction is correct but normalize/mul/div is wrong: a later helper omitted
  the explicit stride used by the reduction;
- the second half of a tile is corrupted: repeated prefix-mask behavior or tile
  group count crossed an instruction mask boundary.

## Runnable references

- `agent/example/kernels/a2/vec_only/group32_stride4_wide128_probe.py`: minimal in-place
  group32 stride/count chain, validated in simulator and on 910B3.
- `agent/example/kernels/a2/vec_only/group64_bf16_fp4_e2m1.py`: padded group64 reduction and
  broadcast inside a longer conversion chain.
- `agent/example/kernels/a2/vec_only/group32_bf16_fp4_e2m1.py`: group32 count behavior with
  physically aligned scalar scratch.
- `agent/references/constraints/vec.md`: primitive semantics and mask reset.

## Do not use when

- the statistic spans the complete row; use `vec-row-reduce-broadcast`;
- groups are not contiguous or require a gather layout;
- the group has a dynamic tail not covered by the mask/stride chain;
- the target is A5 and register/micro grouping is the intended path.

## Source escape

For a new group width, dtype, or helper chain, inspect the exact stubs and
simulator footprint through `agent/references/evidence-escalation.md`. Validate
one group and multiple consecutive groups before integrating the full kernel.
