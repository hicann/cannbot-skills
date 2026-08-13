# Vec Row Reduction to Runtime Broadcast

## Applies when

Use this pattern when one complete A2 fp32 row produces one tensor-resident
scalar and later vec operations reuse that scalar across every 64-lane group in
the row. Typical scalar transforms include normalization, reciprocal, or
scaling performed on tensor data inside the kernel.

## Logical dataflow

```text
row
-> one partial per 64-lane repeat
-> one row scalar
-> scalar tensor transform
-> one reusable 64-lane block
-> elementwise row use
```

The reduction result stays in a `Tensor`. This pattern does not require a
Tensor-to-`Var` bridge or host-side scalar math.

## Physical invariants

| Layer | Requirement |
| --- | --- |
| logical partial count | `groups = width // 64` |
| first reduction | one scalar per 64-lane fp32 repeat |
| partial scratch | physically at least `[1, 64]`, even when `groups < 64` |
| row-scalar scratch | physically at least `[1, 64]` |
| scalar expansion | lane 0 -> one 8-lane block -> one 64-lane block |
| row reuse | apply the 64-lane block once per source group |

`count_per_rep` controls live lanes for the current instruction. It does not
shrink or realign the backing UB allocation. See
`agent/references/constraints/vec.md` for the primitive mask, repeat, and stride
semantics.

## Minimal skeleton

```python
groups = width // 64
partial_s = Tensor(DT.float, [1, 64], Position.UB)
total_s = Tensor(DT.float, [1, 64], Position.UB)
scalar_blocks = Tensor(DT.float, [8, 8], Position.UB)
scale64 = Tensor(DT.float, [8, 8], Position.UB)

cadd(partial_s, row, repeat=groups, src_rep_stride=8, count_per_rep=64)
cadd(total_s[0:1, 0:1], partial_s, repeat=1, count_per_rep=groups)

# Apply the required tensor scalar transform to total_s[0:1, 0:64].
# Lane 0 is the semantic result; the remaining aligned lanes are scratch.

brcb(scalar_blocks, total_s, repeat=1, dst_blk_stride=1, dst_rep_stride=8)
brcb(scale64, scalar_blocks[0:1, 0:8], repeat=1,
     dst_blk_stride=1, dst_rep_stride=8)
```

The first `brcb` materializes lane 0 as one equal 8-lane block. The second
consumes those eight equal values and materializes the reusable 64-lane block.
Keep intermediate vec transforms on the aligned 64-lane view; do not shrink
back to a `[1, 1]` operation. Only the first 8-lane block from `scalar_blocks`
feeds the second broadcast.

## Failure signatures

- `addresses 64 elems but only N left`: logical partial count was used as the
  physical allocation; widen the scratch before changing strides.
- second-level `cadd` alignment failure: keep the source at a 32-byte-aligned
  base and use an aligned physical row.
- host computes a normalizer from a tensor result: the candidate left the
  single-kernel contract instead of retaining the scalar in tensor storage.
- a report claims `brcb` cannot broadcast one scalar because it consumes eight
  source values: the first 8-lane materialization stage is missing.

## Runnable references

- `agent/example/kernels/a2/vec_only/rowwise_reduce_broadcast.py`: minimal full-row sum and
  two-stage runtime broadcast, simulator- and 910B3-validated at widths 64,
  256, 1024, and 1536.
- `agent/references/constraints/vec.md`: `cadd`, `brcb`, count, repeat, mask,
  and stride semantics.
- `agent/references/constraints/tiling.md`: UB alignment and capacity rules.

## Do not use when

- the statistic belongs to each fixed-size group rather than the complete row;
  use `vec-group-reduce-broadcast`;
- width has an unhandled tail or exceeds the example's tile capacity;
- the target is A5 and a register/micro reduction is the intended execution
  model;
- the scalar must cross runtime kernels, which requires an explicit workspace
  contract rather than this on-chip pattern.

## Source escape

If a scalar transform or dtype is not covered, do not force this skeleton.
Follow `agent/references/evidence-escalation.md` for the specific primitive and
return with a minimal aligned probe before changing the overall dataflow.
