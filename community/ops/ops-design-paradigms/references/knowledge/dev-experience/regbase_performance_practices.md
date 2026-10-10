---
title: Regbase Performance Practices
purpose: Capture performance-minded implementation habits that should influence design before profiling starts.
read_when:
  - The operator is already functionally clear and you need high-value performance heuristics.
  - You are deciding between full-load vs split-load, vectorized reduction strategy, or scope placement.
not_for:
  - First-pass functional correctness
  - Exact API availability lookup
keywords:
  - performance
  - full load
  - split load
  - vector reduction
next_reads:
  - ../patterns/regbase_kernel_dataflow_patterns.md
  - ../patterns/regbase_buffer_partitioning.md
  - tiling_review_notes.md
depth: advanced
topic_type: dev-experience
---

# Regbase Performance Practices

This note extracts the most reusable engineering lessons from the 950 performance cards. It is meant for design and implementation decisions, not for a full optimization manual.

## 1. Full-Load vs Non-Full-Load Is A Design Split

Do not treat this as a late optimization detail.

The early question is:

- can the real working set stay resident locally?

If yes, a full-load route can collapse several passes and avoid workspace.
If not, the implementation must accept segmented execution and the extra bookkeeping that comes with it.

When scale is unknown, the safe experience-based default is to design with both routes in mind.

## 2. Vectorized Reduction Is Non-Negotiable

Repeated failure pattern:

- scalar loops get written because they are easy to reason about
- then reduction performance collapses

The practical rule is:

- if the problem is a reduction, start from vectorized reduction assumptions
- scalar accumulation is only acceptable for the tiny cross-segment summary layer, not the main body

## 3. Double Buffering Must Earn Its Space

Double-buffering helps when:

- data movement is dominating
- there is enough local memory for two useful stages
- the implementation has enough steady work to hide the transfer

Do not add it by reflex. It is valuable when it hides latency, not when it just complicates the buffer story.

## 4. MTE Efficiency Is About Shape, Not Just API Choice

Performance often depends less on “which API was used” and more on:

- whether movement happens in large enough blocks
- whether addresses are aligned cleanly
- whether the design fragmented transfers into too many tiny pieces

If the route is copy-bound, revisit transfer shape before revisiting math.

## 5. `__VEC_SCOPE__` Placement Affects Real Performance

On 950, `__VEC_SCOPE__` is not just syntax. Poor placement can:

- repeatedly re-enter vector context
- fragment a stable compute chain
- increase register and scheduling overhead

The working habit is to keep the hot path contiguous unless a real dataflow break forces a split.

## 6. Multi-Row Batching Is Often The Real Optimization

Some kernels speed up more from batching the right rows together than from inventing a fancier instruction sequence. When rows are naturally small, batching can be the difference between a plausible route and a slow one.

## Related Documents

- [[regbase_programming_notes]]
- [[working_vs_failed_950_cases]]
- [[tiling_review_notes]]
- [[../patterns/reduction_patterns]]
- [[../patterns/tiling_patterns]]
