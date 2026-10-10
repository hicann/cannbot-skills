---
title: Small Channel Transpose Notes
purpose: Preserve a proven small-channel transpose compatibility case so layout-change tasks can learn from it without mistaking it for the default regbase route.
read_when:
  - The task involves transpose or layout conversion with very small channel counts.
  - You need a compatibility-side case note before deciding whether a specialized path is worth it.
not_for:
  - Default regbase kernel skeleton design
  - General broadcast or reduction routing
keywords:
  - transpose
  - small channel
  - compatibility case
  - 910b
next_reads:
  - complexity_and_route_selection.md
  - ../patterns/regbase_operator_patterns.md
  - ../pitfalls/regbase_vs_membase_confusions.md
depth: case
topic_type: dev-experience
---

# Small Channel Transpose Notes

This note preserves a high-value compatibility case from the 910b knowledge base. The filename is neutral because the lesson is about the engineering pattern, but the original evidence and performance numbers come from a `910b / membase` implementation context.

## 1. When This Case Is Useful

Read this when the task involves:

- NCHW to NHWC transpose
- very small channel count
- a question about whether a specialized path is worth it
- evaluating a compatibility-side reference before designing a new route

This is not a regbase default pattern. It is a proven compatibility experience note.

## 2. The Winning Trick For Very Small `C`

For very small channel counts, the practical winner in the source case was:

- `vnchwconv` to interleave channels into a 16-element block
- then `Gather` to extract only the valid channels

The key lesson is that a specialized vector-friendly route can beat more “generic” transpose or per-pixel extraction paths by a large margin.

## 3. What Lost And Why

The losing patterns in the source case were valuable because they failed for understandable reasons:

- scalar packing introduced `GetValue` / `SetValue` style overhead
- tiny `DataCopyPad` blocks paid DMA setup costs too often
- generic transpose APIs carried internal overhead that was too large for the shape

The experience lesson is to distrust generic-looking APIs for micro-shape transpose work until the actual shape has been tested.

## 4. Practical Pitfalls

- `Gather` path may force an intermediate cast route depending on dtype
- small transfer sizes can make DMA-heavy ideas look elegant but perform terribly
- alignment behavior in UB and GM still matters even when the logical tensor is tiny

## 5. How To Use This In Regbase Work

Use this note as:

- a compatibility reference
- a performance intuition source
- a reminder that small-shape transpose work may justify a special path

Do not treat it as direct proof that the same implementation should be copied into the regbase branch.

## Related Documents

- [[complexity_and_route_selection]]
- [[regbase_performance_practices]]
- [[../../reference-ops/open_source_operator_table]]
- [[../pitfalls/regbase_vs_membase_confusions]]
