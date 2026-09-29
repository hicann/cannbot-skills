---
title: Tiling Review Notes
purpose: Turn high-frequency tiling mistakes into a review checklist for already-designed or already-written regbase kernels.
read_when:
  - Tiling code already exists and needs review.
  - A design packet is close to implementation and you want to catch high-yield tiling risks.
not_for:
  - First operator classification
  - Exact API whitelist questions
keywords:
  - tiling review
  - storage shape
  - tiling key
  - host kernel contract
next_reads:
  - ../patterns/tiling_patterns.md
  - regbase_performance_practices.md
  - ../pitfalls/common_traps.md
depth: intermediate
topic_type: dev-experience
---

# Tiling Review Notes

This note turns the 950 tiling checklist into review-oriented engineering experience. It is for catching the high-frequency mistakes that make tiling code look almost right while still being broken.

## 1. Treat Includes As Part Of The Contract

Many tiling failures are not algorithmic. They come from file-role confusion:

- host tiling header
- host tiling implementation
- kernel entry
- kernel-side struct
- kernel implementation header

The lesson is to review includes by file role, not by memory.

## 2. `StorageShape` Errors Are Usually Logic Smells

If a tiling implementation is confused about `StorageShape`, `OriginShape`, or where dimensions are read from, the underlying problem is often that the author has not clearly separated:

- storage-level layout
- logical operator semantics
- runtime tiling data

Fix the conceptual split, not just the method call.

## 3. TilingKey Drift Breaks Otherwise Valid Routes

A correct tiling decision is still broken if:

- host sets one key
- kernel branches on another
- binary metadata advertises a third set

Review TilingKey as a three-part contract: generation, dispatch, registration.

## 4. Namespace Hygiene Matters More Than It Looks

Tiling code often spans host and kernel boundaries. Namespace mismatches in `_struct.h`, `_tiling_arch35.*`, and `_apt.cpp` are easy to dismiss as cosmetic, but they are usually a sign that the route was assembled by copy-paste rather than by a stable mental model.

## 5. `TilingPrepare` And `Tiling` Are Different Jobs

Repeated failure pattern:

- platform-only logic leaks into runtime shape logic
- or runtime assumptions are incorrectly pushed into prepare-time code

The stable habit is:

- `TilingPrepare` captures platform facts
- `Tiling` consumes real shape/dtype facts

Review any code that blurs that boundary.

## 6. Serialization Rules Need Their Own Review Pass

The host side may use macros for tiling-data serialization while kernel-side code may need plain structs. When these are “almost aligned” rather than exactly aligned, the bug can hide until much later.

## Related Documents

- [[regbase_build_notes]]
- [[working_vs_failed_950_cases]]
- [[../patterns/tiling_patterns]]
- [[../patterns/kernel_design_patterns]]
- [[../pitfalls/common_traps]]
