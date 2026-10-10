---
title: Working Vs Failed 950 Cases
purpose: Compare stable and broken 950 implementation patterns so route choice and code review start from real success and failure evidence.
read_when:
  - You need concrete 950 lessons before coding.
  - The current design feels close to a known failure mode but the exact difference is unclear.
not_for:
  - Exact API or header lookup
  - Detailed runtime synchronization design
keywords:
  - 950 cases
  - working vs failed
  - route choice
  - failure evidence
next_reads:
  - regbase_kernel_case_notes.md
  - regbase_programming_notes.md
  - ../pitfalls/common_traps.md
depth: case
topic_type: dev-experience
---

# Working Vs Failed 950 Cases

This note turns the `knowledge_cards/kernel_code_patterns_950.md` material into a smaller set of engineering lessons. The goal is not to catalog every operator, but to show what repeatedly separates stable 950 implementations from broken ones.

## 1. Simple DAG Routes Succeed When The Problem Really Is Simple

Successful elementwise operators on `ascend950` tend to share these traits:

- the operator is genuinely short and local
- the route stays inside the DAG abstraction cleanly
- dtype handling is explicit
- tiling keys, config, and kernel branches line up

The lesson is not “always use DAG”. The lesson is that DAG works well when the operator shape really matches it.

## 2. Regbase Works Best When You Commit To It

Successful regbase paths usually look like this:

- explicit class-based kernel organization
- direct regbase compute chain
- deliberate buffer and dtype handling
- no attempt to hide imperative logic under a half-DAG abstraction

The common success signal is commitment. The implementation chooses regbase and then stays regbase.

## 3. Mixing DAG And Regbase Is A High-Probability Failure Mode

One of the most expensive 950 mistakes is hybrid code that:

- borrows DAG structure
- then injects raw MicroAPI assumptions inside it
- or assumes a Vec wrapper and a MicroAPI primitive are interchangeable

This produces code that is neither a clean DAG path nor a clean regbase path. The result is usually API mismatch, configuration mismatch, or both.

## 4. Many “Kernel” Failures Are Actually Configuration Failures

A large share of broken 950 work comes from:

- `simplified_key.ini` not matching the route
- TilingKey branches not matching runtime registration
- missing or inconsistent config updates
- operator support declarations drifting away from the code path

When the math seems fine but the route still breaks, inspect config before redesigning computation.

## 5. A Practical 950 Review Habit

Before trusting a new 950 implementation, ask:

- Is this really a DAG-shaped problem or a regbase-shaped problem?
- Did we keep one execution family all the way through?
- Does TilingKey routing agree with config and dtype coverage?
- Is any “framework mixing” happening under a different name?

## Related Documents

- [[complexity_and_route_selection]]
- [[regbase_programming_notes]]
- [[regbase_performance_practices]]
- [[tiling_review_notes]]
- [[../patterns/regbase_operator_patterns]]
