---
title: Precision And Runtime
purpose: Combine cast, precision, host runtime, and launch-time guardrails that usually decide whether a regbase implementation is viable.
read_when:
  - Host launch order, cast placement, or runtime guardrails are part of the question.
  - Design and debugging need both precision rules and runtime sequencing together.
not_for:
  - Full kernel-structure design
  - Existing-source reference selection
keywords:
  - precision
  - runtime
  - cast
  - host launch
next_reads:
  - ../pitfalls/precision_guide.md
  - regbase_api_sync.md
  - ../regbase_development_guide.md
depth: foundation
topic_type: api
---

# Precision And Runtime

This card combines precision rules with the host-side runtime sequence and the most important launch-time guardrails for regbase direct-invoke work.

## Precision Rules

Keep numerically sensitive chains in higher precision until the final cast:

- use fp32 for reduce, `exp`, `log`, reciprocal-style, and normalization intermediates
- cast back to fp16 or bf16 only when the output boundary requires it
- avoid chaining low-precision intermediates through several nonlinear operations

## Cast Rules

- `half -> float`: use `CAST_NONE`
- `float -> half`: use `CAST_ROUND`
- `int32_t -> float`: use `CAST_NONE`
- `half -> int32_t`: choose the rounding mode that matches the quantization rule

The safe rule is simple: raise precision early, lower precision late.

## Host Runtime Order

On the host side, the launch sequence should be deterministic:

1. initialize ACL
2. call `aclrtSetDevice`
3. call `aclrtGetDeviceInfo`
4. choose the core count attribute that matches the operator family
5. allocate GM
6. compute tiling
7. launch the kernel
8. clean up

`aclrtGetDeviceInfo` must come after `aclrtSetDevice`.

## Core Count Selection

Pick the resource query by operator type:

- pure vector work: `ACL_DEV_ATTR_VECTOR_CORE_NUM`
- matrix work: `ACL_DEV_ATTR_CUBE_CORE_NUM`
- mixed work: `ACL_DEV_ATTR_AICORE_CORE_NUM`

Do not hardcode a core count when the runtime can query it for you.

## Kernel-Side Guardrails

The kernel side should avoid host-style facilities that are not available in AI Core code:

- no `std::` math functions in kernel code
- no dynamic memory allocation
- keep host and kernel headers separated

These restrictions often surface as runtime or build failures, so treat them as part of the launch contract.

## Parameter Limits

- `repeatTime`-style repeat counters can overflow when they are stored in `uint8_t`
- compare-style operations still need their documented alignment constraints
- if a row count or repeat count can exceed the limit, split the work in the host tiling or in the kernel loop

## Practical Precision Checklist

- Is the critical intermediate in fp32?
- Is the final cast happening at the boundary, not in the middle of the nonlinear chain?
- Is the host querying the device after `aclrtSetDevice`?
- Is the selected core count attribute correct for the operator family?
- Are repeat and alignment limits satisfied before launch?

## Related Documents

- [[regbase_api_reference]]
- [[regbase_api_whitelist]]
- [[arithmetic_and_reduce]]
- [[datacopy_best_practices]]
