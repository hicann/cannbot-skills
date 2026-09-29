---
title: Complexity And Route Selection
purpose: Classify the operator and choose the right regbase development route before coding.
read_when:
  - The operator shape is still unclear before design.
  - You need to choose between unary, fused, reduction, broadcast, transpose, or heavier framework routes.
not_for:
  - API signature lookup
  - Low-level sync or buffer partition details
keywords:
  - route selection
  - operator shape
  - complexity
  - branch choice
next_reads:
  - ../regbase_development_guide.md
  - ../patterns/regbase_operator_patterns.md
  - regbase_kernel_case_notes.md
depth: foundation
topic_type: dev-experience
---

# Complexity And Route Selection

This note distills the project’s operator-classification experience into one practical question: what kind of problem are we actually building, and what route should that force before code appears?

## 1. Start With Operator Shape, Not With Favorite APIs

The first useful split is not “DAG vs regbase” or “template vs hand-written”. It is:

- single-step elementwise
- short fused chain
- matrix-heavy route
- performance-sensitive large-data route
- global-aggregation or normalization route

If this classification is wrong, the rest of the design usually drifts.

## 2. A Good Complexity Label Changes Expectations

### Level 0: Single-Step Elementwise

Examples: `abs`, `exp`, `sqrt`, direct unary/binary vector math.

Experience rule:

- do not over-design
- use the smallest viable dataflow
- only bring in heavier tiling or buffering when data scale forces it

### Level 1: Short Fused Chain

Examples: `silu`, `swish`, simple fused post-process chains.

Experience rule:

- keep the chain local
- count intermediate buffers before choosing a route
- the task is still “small” mathematically, but it can already be expensive structurally

### Level 2: Matrix-Oriented Or Framework-Heavy Work

Examples: matmul or routes that fundamentally depend on cube-side abstractions.

Experience rule:

- stop pretending this is just a dressed-up elementwise op
- elevate framework and tiling decisions early

### Level 3: Performance-Sensitive Scale

Examples: otherwise simple operators that become large enough to require double-buffering, split paths, or stronger memory planning.

Experience rule:

- the math may stay simple
- the implementation no longer is
- performance planning becomes part of the design, not a later tuning pass

### Level 4: Global Aggregation / Normalization / Attention-Like Flow

Examples: `softmax`, `layer_norm`, `rms_norm`, multi-pass reductions.

Experience rule:

- assume multiple passes until proven otherwise
- global statistics and intermediate reuse become primary design objects

## 3. Route Selection Heuristics

- If one vector API can express the core operator, stay simple.
- If a short chain fits a stable local compute path, treat it as fusion, not as a framework exercise.
- If the operator contains global statistics, do not reduce it to “just another fused elementwise op”.
- If data scale is unclear, design a route that can survive both full-load and non-full-load instead of assuming the small path.

## 4. Complexity Is About Failure Risk, Not Just Code Length

The useful question is:

- what can silently go wrong here?

For example:

- a tiny elementwise op can still become Level 3 if the real challenge is throughput and GM traffic
- a seemingly moderate fused op can be Level 4 if it hides a true global aggregation dependency

## Related Documents

- [[requirements_analysis_patterns]]
- [[regbase_programming_notes]]
- [[working_vs_failed_950_cases]]
- [[../patterns/regbase_operator_patterns]]
- [[../patterns/kernel_design_patterns]]
