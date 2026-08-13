# Cube-Only Matmul Pattern

## Applies when

Use this pattern when the formula stays on the cube side without vec/micro
preprocessing, nonlinear postprocessing, or a cross-side publish/consume stage.

## Logical dataflow

```text
GM operands -> tiled matmul/cube reduction -> GM output
```

## Physical invariants

- keep the matmul destination anchored at L0C row offset 0;
- choose tile shape from local-memory/layout legality before choosing core split;
- keep output ownership stable: one core owns each output tile unless the
  selected split explicitly defines a reduction/merge contract;
- keep format-specific split alignment and capacity rules in
  `agent/references/constraints/tiling.md`.

## Minimal skeleton

```text
GM -> L1 -> L0A/L0B -> MMAD -> L0C -> GM
```

Typical implementation steps are load operand tiles, publish them to L0A/L0B,
run `matmul`/`mmad`, and write the owned L0C tile to GM.

## Core split and ownership

Treat tile shape and core split as separate decisions:

| Mode | Use when |
| --- | --- |
| `split_m` | one core should keep all N tiles for the same M rows |
| `split_n` | one core should keep all M tiles for the same N columns |
| `mix` | both output axes may split and no downstream ownership requires one axis |

For ordinary cube-major matmuls, split by output-tile index. For batched small
matmuls, flatten the independent batch/head/output-tile ownership domain before
distributing it across cube cores. Choose the mode from downstream ownership,
then use `agent/scripts/estimate_matmul_datamove.py` to compare legal candidates.

## Failure signatures

- simulator passes but hardware leaves part of L0C uninitialized: a matmul
  destination used a nonzero L0C row sub-offset;
- output tiles overlap or disappear only under multicore: ownership and split
  axes disagree;
- a split value is rejected or silently misaligned: format-specific legality
  was treated as a scheduling preference;
- a pure matmul gained GM workspace and vec synchronization: a mixed-pipeline
  pattern was applied without a formula stage that needs it.

## Runnable references

- `agent/example/kernels/a5/matmul/matmul_float_mmad.py`: shortest cube baseline.
- `agent/example/kernels/a5/matmul/matmul_f32_tailsafe.py`: full-tile locals with GM tails.
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitn.py`: split-N ownership.
- `agent/example/kernels/a5/matmul/matmul_mknk_2dgrid_splitk.py`: legal split-K staging.
- `agent/example/kernels/a2/matmul/qk_matmul_batched.py`: flattened batched ownership.
- `agent/references/constraints/tiling.md`: capacity, layout, alignment, and
  format-specific split constraints.

## Do not use when

- vec/micro must transform an operand before cube consumes it;
- cube output requires vec/micro postprocessing before the public output;
- a later cube stage consumes a vec-produced intermediate;
- the output reduction is distributed across runtime kernels without an
  explicit merge contract.

## Source escape

For a new layout, shortcut, or format-specific split, follow
`agent/references/evidence-escalation.md`. Inspect the public facade, lowering,
and one minimal aligned generated kernel before generalizing the pattern.
