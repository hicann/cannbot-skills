---
name: broadcast_patterns
description: Regbase-first 广播模式选择，覆盖维度折叠、OneDim、UB Broadcast、NDDMA Broadcast 以及标量快速路径。
title: Broadcast Patterns
purpose: Choose the correct broadcast route after dimensions are collapsed and the operand reuse pattern is clear.
read_when:
  - The task contains broadcasting and you need a concrete route such as OneDim, UB broadcast, or NDDMA broadcast.
  - Broadcast shape handling affects the overall regbase kernel design.
not_for:
  - Reduction-only design
  - Detailed synchronization debugging
keywords:
  - broadcast
  - onedim
  - nddma
  - dimension collapse
next_reads:
  - regbase_operator_patterns.md
  - regbase_kernel_dataflow_patterns.md
  - ../pitfalls/common_traps.md
depth: intermediate
topic_type: pattern
type: knowledge_card
platform: common
verified: false
patterns: [broadcast, dimension_collapse, onedim, nddma, ub_broadcast]
---

# Broadcast Patterns

This note captures the broadcast routes that are most useful for regbase-first direct-invoke work on Ascend 950 and adjacent compatibility contexts.

## Start With Dimension Collapse

Before branch selection:

- left-pad smaller-rank inputs with `1`
- compute per-axis broadcast flags
- merge adjacent axes that share the same broadcast pattern
- compute stride-zero axes explicitly

After collapse, most tasks become one of two categories:

- one-dimensional linear work
- multi-dimensional broadcast work

## Pattern 1: OneDim Fast Path

Choose OneDim when collapse leaves one logical dimension.

Key traits:

- all data is processed linearly
- scalar inputs are a first-class case
- UB and block tiling can be computed from a simple linear length

Fast path rule:

- prefer TensorScalar forms such as `Adds` / `Muls` when one side is scalar
- use `Duplicate` only when no scalar form exists

## Pattern 2: UB Broadcast

Use UB Broadcast when the task is multi-dimensional and the chosen platform path cannot or should not rely on NDDMA broadcast.

Shape:

1. copy the un-broadcast source tile into UB
2. expand it in UB
3. run the arithmetic step on the broadcasted tile

Best fit:

- compatibility-oriented routes
- shapes where explicit UB-side expansion is easier to reason about than a multi-copy setup
- cases where tmp-buffer cost is acceptable

## Pattern 3: NDDMA Broadcast on Ascend 950

For `ascend950 / regbase`, NDDMA broadcast is often the preferred multi-dimensional route when the shape fits its constraints.

Why:

- broadcast happens during GM-to-UB movement
- no extra UB-side broadcast instruction is needed
- destination tiles arrive already expanded

Checkpoints:

- only use it when the broadcasted axes can be encoded as stride-zero movement
- respect the limited multi-dimensional copy rank
- when an input does not actually broadcast on the split axis, let it degenerate to a normal copy

## Pattern 4: Split-Axis and Loop Selection

Multi-dimensional broadcast design still needs a stable split:

- find the innermost product that fits UB
- define `ubSplitAxis`, `ubFormer`, `ubOuter`, and tail behavior
- fuse the outer product for multi-core scheduling

The split axis is not just a size calculation. It determines:

- the loop structure
- the GM offset logic
- whether the broadcast can stay inside a single copy primitive

## Pattern 5: Broadcast Is Still a Dataflow Contract

Even when the math is elementwise, broadcast work must record:

- which input axes are broadcasted
- whether the path is OneDim, UB Broadcast, or NDDMA Broadcast
- whether any scalar fast path is available
- what shape the compute stage actually sees after expansion

Without this, implementation and review will silently disagree on what the kernel is doing.

## Regbase-First Checks

- On Ascend 950, prefer NDDMA broadcast only when its rank and stride-zero assumptions actually fit.
- Keep scalar fast paths explicit; they often remove one buffer and one copy.
- Use `DataCopyPad`-style safe copy behavior whenever alignment is uncertain.
- Avoid mixing a broadcast route with an unrelated membase execution explanation.

## Related Documents

- [[tiling_patterns]]
- [[regbase_operator_patterns]]
- [[kernel_design_patterns]]
- [[../api/index]]
- [Precision Guide](../pitfalls/precision_guide.md)
