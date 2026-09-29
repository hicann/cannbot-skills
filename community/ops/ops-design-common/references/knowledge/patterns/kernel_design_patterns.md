---
name: kernel_design_patterns
description: Regbase-first kernel 设计模式，覆盖设计包、API 语义校验、骨架优先实现顺序、串讲审查和修复闭环。
title: Kernel Design Patterns
purpose: Keep design packets, walkthroughs, implementation order, and repair loops coherent for regbase work.
read_when:
  - The task is moving from design into implementation or review.
  - You need to keep branch packets, walkthrough artifacts, and repair loops aligned.
not_for:
  - Exact API whitelist questions
  - Low-level VF math selection
keywords:
  - design packet
  - walkthrough
  - repair loop
  - semantic verification
next_reads:
  - ../regbase_development_guide.md
  - ../dev-experience/regbase_kernel_case_notes.md
  - ../pitfalls/common_traps.md
depth: foundation
topic_type: pattern
type: knowledge_card
platform: common
verified: false
patterns: [kernel_design, workflow, semantic_verification, walkthrough, repair]
---

# Kernel Design Patterns

This note captures the engineering patterns that turn regbase-first design ideas into code and review artifacts that stay coherent through walkthrough, implementation, and repair.

## Pattern 1: Branch Packet Before Code

Before implementation, the design packet should make the branch explicit:

- target platform and branch
- mathematical route
- shape abstraction and tiling route
- reference-operator status: inspected reference vs greenfield
- precision-sensitive stages
- expected API families

If any of these are missing, later stages tend to drift back toward default-template assumptions.

## Pattern 2: Skeleton Before Logic

Implementation order should be:

1. start from the routed project skeleton and lock the branch packet first
2. make the empty project compile
3. add host tiling contract
4. add kernel compute path
5. add verification and edge-case checks

This keeps structural errors separate from math errors and makes repair loops smaller.

## Pattern 3: API Semantic Verification Before Commitment

A candidate API is not “chosen” until it has passed all of these checks:

- the API family matches the branch and dataflow
- the name appears in the regbase-facing API material
- the expected semantics match alignment, repeat, or synchronization needs
- the design does not assume a membase primitive under a regbase label

Use `../api/index.md` as the local constraint-check and regbase-facing lookup path, and reconcile any design mismatch before the API is treated as chosen.

## Pattern 4: Walkthrough Uses Pattern Evidence

In walkthrough:

- review the shape abstraction
- challenge the branch route
- verify reduction or broadcast selection
- question precision boundaries
- question reuse claims

A good walkthrough comment is not “this feels wrong”. It names the pattern mismatch:

- wrong branch
- wrong split axis
- unsupported reuse claim
- precision-sensitive state dropped too early

## Pattern 5: Repair Preserves Branch Context

When a review fails:

- do not switch execution families casually
- fix inside the current regbase route unless the branch assumption itself is proven wrong
- re-check API whitelist and pattern choice before rewriting compute code
- for fusion tasks, separate “keep reused” from “rewrite for regbase”

This is how repair stays surgical instead of devolving into a hidden redesign.

## Pattern 6: Minimal but Complete Test Boundaries

At minimum, design and implementation should cover:

- compile viability
- tail block and tail tile behavior
- non-aligned copy boundaries
- precision-sensitive shapes
- branch-specific edge cases such as split-load reduction or broadcast degenerating to scalar

These are pattern tests, not just example tests. They prove the route, not only the sample input.

## Related Documents

- [[regbase_operator_patterns]]
- [[tiling_patterns]]
- [[reduction_patterns]]
- [[broadcast_patterns]]
- [[../api/index]]
- [Precision Guide](../pitfalls/precision_guide.md)
- [[../../reference-ops/open_source_operator_table]]
