---
title: Requirements Analysis Patterns
purpose: Lock the requirements that repeatedly decide regbase route, tiling shape, precision strategy, and validation plan.
read_when:
  - The task definition is still ambiguous before design.
  - Dtype, precision, scale, or shape questions are unresolved.
not_for:
  - Final API signature lookup
  - Repairing an already isolated runtime bug
keywords:
  - requirements
  - dtype
  - shape
  - precision route
next_reads:
  - ../regbase_development_guide.md
  - complexity_and_route_selection.md
  - ../patterns/regbase_operator_patterns.md
depth: foundation
topic_type: dev-experience
---

# Requirements Analysis Patterns

This note adapts the project’s requirement-question checklist into a smaller set of questions that repeatedly prevent rework in regbase tasks. Read it before freezing a design packet.

## Always Lock These First

### 1. Dtypes And Precision Route

Before design, clarify:

- input dtypes
- output dtypes
- whether accumulation or post-process must stay in fp32
- whether dtype conversion is part of the operator contract or only an implementation detail

If this is vague, the API route, cast placement, and precision validation plan all stay unstable.

### 2. Data Scale And UB Fit

Ask whether the important working set:

- always fits a comfortable local tile
- sometimes exceeds UB and needs split execution
- is unknown and should therefore default to a conservative split-capable route

Many “simple” kernels become structurally different once the answer changes from full-load to non-full-load.

### 3. Shape Semantics

Clarify early whether the task is:

- pure elementwise
- reduction
- reduction plus post-process
- broadcasted binary op
- shape-transform plus compute

This decides the pattern family before any API discussion starts.

### 4. Mixed Precision, Quantization, Or Special Numeric Contracts

If the task touches quantization, fp8, bf16, or asymmetric dtype mappings, ask for the missing contract explicitly:

- target numeric format
- scale or zero-point behavior
- block size or layout requirements
- whether host attributes or metadata participate in the computation

Do not let the implementation discover these by accident.

### 5. Invalid Input And Edge Behavior

For numerically sensitive operators, ask what should happen on:

- divide-by-zero
- invalid domains such as negative sqrt
- empty or degenerate dimensions
- tail tiles and alignment boundaries

These answers belong in the design packet, not only in tests.

## High-Value Defaulting Rules

- If scale is unknown, assume the implementation must survive the non-full-load path.
- If reduction axes are unclear, stop and clarify instead of assuming “last axis”.
- If mixed precision is implied but not specified, make the cast and accumulation route explicit before coding.
- If the task description mentions a similar open-source operator, use `[[../../reference-ops/open_source_operator_table]]` only as a hint for choosing a real reference implementation to inspect, not as proof of reuse.

## Related Documents

- [[process_memory_cards]]
- [[regbase_programming_notes]]
- [[../patterns/regbase_operator_patterns]]
- [[../patterns/reduction_patterns]]
- [[../pitfalls/precision_guide]]
