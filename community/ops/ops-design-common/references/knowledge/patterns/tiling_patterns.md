---
name: tiling_patterns
description: Regbase-first 的 tiling 设计模式，覆盖线性切分、全载/分载判定、UB 切分、split-axis 选择与设计记录字段。
title: Tiling Patterns
purpose: Choose block split, UB split, and recorded tiling fields that keep regbase design and implementation aligned.
read_when:
  - The task needs a tiling plan before coding.
  - You must decide split axis, UB unit, or full-load vs segmented work.
not_for:
  - Runtime debugging after the design is already fixed
  - API-family lookup
keywords:
  - tiling
  - block split
  - UB split
  - tiling key
next_reads:
  - regbase_operator_patterns.md
  - reduction_patterns.md
  - ../dev-experience/tiling_review_notes.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: common
verified: false
patterns: [tiling, block_split, ub_split, regbase, direct_invoke]
---

# Tiling Patterns

This note collects the tiling patterns that remain valid for regbase-first work even though many source rules come from general Ascend C design references.

## What Tiling Must Decide

Every usable tiling design answers five questions:

1. What is the shape abstraction after collapse: linear, `AR`, `ARA`, or broadcasted multi-axis?
2. How many cores can do meaningful work without creating tiny blocks?
3. Which unit is the first stable split: block, row group, tile, or broadcast slice?
4. What is the UB processing unit and how is its tail handled?
5. Which fields must be recorded so implementation and review use the same branch assumptions?

## Pattern 1: Linearized 1D Tiling

Use this when inputs and outputs are shape-equal or collapse into one dimension.

Key ideas from the elewise and one-dim broadcast routes:

- keep each core above a minimum useful work size instead of forcing maximal fan-out
- align block work to stable vector-friendly boundaries
- distinguish `blockFormer` from the tail block rather than pretending every core is symmetric
- choose `ubFormer` independently from the block size so the UB loop stays aligned

Typical defaults from the sources:

- per-core work should stay above the small-work threshold
- linear block splits often align to 512-element style boundaries
- UB tiles align to 256B, while one-dim broadcast can use a 128B-oriented inner split

## Pattern 2: Full-Load Versus Split-Load

For reduction or broadcast work, the first branch is usually:

- can one full logical unit fit into UB with all required temporary state?
- if yes, choose full-load
- if no, choose split-load and make merge behavior explicit

Examples:

- `AR` reduction: can one full row fit?
- `ARA` reduction: can one `[R, tileA0Len]` block fit?
- broadcast: can one collapsed multi-axis output slice fit after accounting for all live inputs and outputs?

Never describe this as “fits roughly”. The design should name the logical unit that fits or does not fit.

## Pattern 3: Split the Most Stable Axis

When full-load is impossible, split the axis whose merge semantics are easiest to reason about:

- split `R` for streaming reductions when chunk merge is associative enough
- split the output tile axis for broadcast when stride-zero reconstruction stays simple
- keep the retained `A` axes as the identity side of the computation whenever possible

This matters more than raw tile size. A slightly smaller but semantically stable split is safer than a larger tile with unclear merge rules.

## Pattern 4: Collapse Before You Tile

Before computing any tile numbers:

- collapse same-shape elementwise work to 1D
- collapse reduction axes into `A/R` form
- collapse broadcast flags into a smaller stride-zero representation

Do not compute tiling over the original high-rank shape if the branch route is determined by the collapsed representation.

## Pattern 5: Record the Contract, Not Just the Numbers

A good regbase tiling section records:

- collapsed shape form
- branch key: full-load vs split-load, `AR` vs `ARA`, one-dim vs multi-dim broadcast
- core split fields: core count, block size, tail block behavior
- UB split fields: `ubFormer`, loop count, tail tile behavior
- alignment-sensitive lengths
- any workspace requirement

These fields are what walkthrough, implementation, and review need. Raw formulas alone are not enough.

## Tiling Red Flags

- The design gives formulas but never states the logical unit being loaded.
- Tail handling is omitted for either block or UB loops.
- The split axis changes between design and code comments.
- UB capacity is computed without counting live buffers or temporary storage.
- The design mixes regbase execution assumptions with membase queue choreography.

## Related Documents

- [[regbase_operator_patterns]]
- [[reduction_patterns]]
- [[broadcast_patterns]]
- [[kernel_design_patterns]]
- [[../api/index]]
- [Precision Guide](../pitfalls/precision_guide.md)
