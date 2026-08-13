# Tail Constraints

Read this file for generic tile-tail legality: stable local shapes, GM-boundary
gating, sub-block ownership, and the rule that padded lanes must be excluded
before a reduction consumes them. Complete online-softmax mask/update ordering
lives in `agent/references/patterns/online-softmax-tail.md`.

## 1. Stable local shapes

Keep L1, L0C, UB, and internal workspace tensors at their declared full tile
shape. Carry `valid_m`, `valid_n`, and `valid_k` separately.

- gate GM reads and writes with valid extents;
- zero-fill or explicitly mask padded local lanes when they can affect math;
- do not shrink local tensor shapes on the last tile;
- `l0c_to_ub` does not accept a sliced L0C source, so keep the full L0C tile and
  gate later boundaries.

Stable physical shapes keep lowering, stride inference, and simulator behavior
consistent between aligned and tail tiles.

## 2. Sub-block row ownership

A5 compact vec splits may divide `valid_m` with `CeilDiv(valid_m, 2)`. A2
workspace-bridge kernels use a fixed physical split: sub-block 0 owns the first
half-tile and sub-block 1 owns the second, then each computes its own
`local_valid_m`.

Do not mix compact and fixed physical splits. Device topology and sub-block
layout are owned by `constraints/a2.md`, `constraints/a5.md`, and the selected
mixed-pipeline pattern.

## 3. Workspace tails

Declare internal workspaces at a stable full tile shape. Choose slot count from
the producer/consumer lifetime, not from the final tail length. Crop only GM
input/output boundaries; explicitly mask padded values before local reductions
or nonlinear operations that would observe them.

## 4. Reduction-mask placement

GM slicing alone is insufficient when padded local values reach a reduction.
Apply the semantic identity before the first affected reduction:

- max/min domains: use the appropriate finite sentinel;
- additive domains: zero invalid contributions;
- output-only tails: gate the final GM write when no earlier math observes the
  padded lanes.

For online softmax, S2/causal score masks must precede row max, while invalid S1
rows are masked after the max shift and before `exp`. Read
`agent/references/patterns/online-softmax-tail.md` for the complete ordering.

## 5. Failure signatures

- aligned shapes pass but odd shapes fail;
- only the final tile is wrong;
- one vec sub-block is correct and the other is corrupt;
- output shape is correct but boundary rows/columns are wrong;
- row max or row sum changes when only padded columns were added.

Inspect GM boundary extents and the first reduction/nonlinearity that can see a
padded lane before changing tile sizes or local allocation.

## 6. Pattern routes

- online-softmax S1/S2/causal ordering:
  `agent/references/patterns/online-softmax-tail.md`
- A2 fixed sub-block workspace bridge:
  `agent/references/patterns/a2-mixed-pipeline.md`
- A5 compact/direct handoff:
  `agent/references/patterns/a5-mixed-pipeline.md`
- vec count, mask, and reduction semantics:
  `agent/references/constraints/vec.md`
