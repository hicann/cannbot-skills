---
name: reduction_patterns
description: Regbase-first 归约模式选择，覆盖 AR/ARA、全载与分载、多轴变换、Welford、Group Reduce、二分累加和索引跟踪。
title: Reduction Patterns
purpose: Route regbase reduction work by shape form, load strategy, and algorithm family before writing the compute chain.
read_when:
  - The task contains reduction, norm, variance, softmax-style rows, or with-index accumulation.
  - You need to decide AR vs ARA, full-load vs split-load, or extra reduction algorithms.
not_for:
  - Pure broadcast-only operators
  - Build or packaging issues
keywords:
  - reduction
  - AR
  - ARA
  - Welford
next_reads:
  - regbase_operator_patterns.md
  - ../pitfalls/precision_guide.md
  - ../api/arithmetic_and_reduce.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: common
verified: false
patterns: [reduction, ar, ara, welford, group_reduce, with_index]
---

# Reduction Patterns

This note condenses the general Ascend C reduction routes into a regbase-first selection guide.

## Route the Reduction Before Choosing APIs

The stable order is:

1. Collapse dimensions into `A/R` form.
2. Decide whether the reduced shape is single-axis or multi-axis.
3. For single-axis cases, decide `AR` vs `ARA`.
4. Decide full-load vs split-load.
5. Add orthogonal algorithm choices such as Welford, Group Reduce, dichotomy addition, or with-index tracking.

## Pattern 1: `AR` Route

Use `AR` when the reduced axis is the tail axis after collapse and each row of `R` elements is logically continuous.

Branch split:

- full-load: a whole row fits, so copy-in happens once and the row can be reduced locally
- split-load: `R` must be chunked and partial results must be merged across chunks

Typical fit:

- scalar-per-row outputs
- tail-axis `sum` / `max` / `mean`
- the final reduction step of a multi-stage design

## Pattern 2: `ARA` Route

Use `ARA` when the reduced axis is not the tail axis after collapse and each working block is logically `[R, A0]`.

Branch split:

- full-load: one `[R, tileA0Len]` block fits and can be reduced as a local 2D unit
- split-load: rows or row groups must be streamed and merged

Typical fit:

- channel-preserving reductions
- norm-like work where the kept axis is vector-valued
- post-reduction work that still needs the kept `A0` lanes

## Pattern 3: Multi-Axis Reduction

When collapse still leaves alternating `A/R` segments:

- do not invent a brand-new branch type
- decompose the route into nested `AR` and `ARA` decisions
- keep each reduced segment explicit in the design packet

This is the right place to document whether the kernel is truly multi-axis or just a collapsed single-axis special case.

## Pattern 4: Welford for Streaming Dual Statistics

Use Welford when split-load reduction must produce two coupled statistics such as mean plus variance.

Choose it when:

- two related statistics would otherwise require two passes
- streaming precision matters
- the merge rule must support chunked or grouped partials

Design reminders:

- keep mean, `M2`, and count merge rules explicit
- prefer fp32 intermediate state
- document whether the design uses plain Welford or grouped Welford

## Pattern 5: Group Reduce for Large `R`

Use Group Reduce when:

- `R` is too large for one core to traverse economically
- the kept `A` side is too small to expose enough ordinary parallelism

The shape becomes two phases:

1. each core computes a partial
2. partials are written to workspace and merged after synchronization

If the design uses Group Reduce, it should name:

- the partial shape
- workspace layout and alignment
- who performs the final merge

## Pattern 6: Dichotomy Addition for Precision-Sensitive Sum

Use dichotomy or half-interval addition when the problem is specifically sum precision, not generic reduction routing.

Good fit:

- large `sum`
- high dynamic range
- chunked accumulation where linear order would swallow small values

Do not apply this pattern to `max` / `min` where the precision problem is of a different kind.

## Pattern 7: With-Index Is an Orthogonal Variant

With-index tracking does not replace the normal branch route.

Instead:

- reuse the same `AR` / `ARA` / full-load / split-load decision
- layer index tracking on top
- make the extra constraints explicit: index dtype, compare/select alignment, and cross-chunk offset merge

## Regbase-First Checks

- Verify the chosen reduction APIs in `../api/index.md` before freezing the design.
- Keep precision-sensitive summaries in fp32 long enough for post-processing.
- If no useful existing reference implementation exists, stay greenfield; do not import a reduction shape by analogy alone.
- Do not slip back into membase queue reasoning to explain a regbase reduction route.

## Related Documents

- [[regbase_operator_patterns]]
- [[tiling_patterns]]
- [[broadcast_patterns]]
- [Precision Guide](../pitfalls/precision_guide.md)
- [[../api/index]]
