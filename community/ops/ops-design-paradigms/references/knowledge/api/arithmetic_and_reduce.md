---
title: Arithmetic And Reduce
purpose: Choose arithmetic, broadcast-style, and reduction API families with the right repeat and shape constraints.
read_when:
  - You are deciding between scalar helpers, repeated broadcast forms, or reduction families.
  - The operator math is clear but the arithmetic or reduce API family is not.
not_for:
  - Build or packaging questions
  - High-level kernel structuring
keywords:
  - arithmetic
  - reduce
  - repeat limits
  - scalar helpers
next_reads:
  - regbase_api_whitelist.md
  - ../patterns/reduction_patterns.md
  - precision_and_runtime.md
depth: intermediate
topic_type: api
---

# Arithmetic And Reduce

This card combines the two API families that usually decide the shape of a regbase kernel: scalar/broadcast arithmetic and local or patterned reduction.

## Arithmetic Selection

For single-row or scalar-like work, prefer scalar helpers:

- use `Adds` instead of `Duplicate + Add`
- use `Muls` instead of `Duplicate + Mul`

For repeated broadcast across rows, use the repeat form with `BinaryRepeatParams` and set `src1RepStride = 0` so one source operand is reused without materializing a broadcast buffer.

## Repeat Limits

Many vector repeat-style APIs take `repeatTime` as `uint8_t`, which means the practical limit is 255.

If a tile can exceed that limit, split the work before you enter the kernel loop rather than letting the repeat counter overflow.

## Reduce Selection

Choose the reduce family by shape and alignment:

- level-2 reduce for row-local reductions such as softmax or layernorm style rows
- pattern reduce for cross-row or batch-oriented reduction when the input is aligned and the pattern is a good fit

## Level-2 Reduce Rules

- the temporary buffer type must match the data type being reduced
- `count` is the number of effective elements, not the padded row length
- row offsets should use the padded row stride, not the logical row length

## Pattern Reduce Rules

- the source shape must use aligned columns
- prefer the explicit shared temporary buffer form
- reserve the temporary space before the call if the interface expects it
- use pattern reduce only when the shape and alignment are already under control

## Compare Constraint

`Compare` is not a free-form fallback for arbitrary shapes. When it is used in a compatibility path, the working region must be padded so the compared `count` occupies a 256-byte aligned span.

- pad the working buffer to a 256-byte boundary before calling `Compare`
- keep the compare span aligned to the padded size, not the logical unpadded count
- copy out only the effective elements after the compare result is produced
- use explicit extreme values for padding when the compare semantics depend on the neutral element

Rule of thumb:

- ArgMax-style compare: pad with the minimum representable value for the active dtype
- ArgMin-style compare: pad with the maximum representable value for the active dtype

## Practical Decision Tree

1. Is this just a scalar adjustment on one row? Use `Adds` or `Muls`.
2. Is this a repeated row-wise broadcast? Use the repeat form and keep `src1RepStride = 0`.
3. Is this a row-local reduction? Use the level-2 reduce API.
4. Is this a cross-row or patterned batch reduction? Use the pattern reduce API with aligned shapes and an explicit temporary buffer.

## Common Mistakes

- Using `Duplicate` when a scalar helper would do the job in one instruction.
- Forgetting that `repeatTime` can overflow.
- Passing padded counts into reduction APIs that expect effective counts.
- Using pattern reduce on an unaligned shape and hoping the API will fix it.
- Reusing the same buffer for `dst` and `tmpBuffer` when the API requires separation.

## Related Documents

- [[regbase_api_reference]]
- [[regbase_api_whitelist]]
- [[pipeline_and_buffer]]
- [[precision_and_runtime]]
