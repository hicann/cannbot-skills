---
title: Regbase Buffer Partitioning
purpose: Decide what should live in GM, UB, and registers so regbase kernels stage data cleanly across shell, UB, and VF layers.
read_when:
  - You are planning tile-local buffers, scratch storage, or register residency.
  - The design needs clear GM to UB to register boundaries before coding.
not_for:
  - High-level operator classification
  - Exact API signature lookup
keywords:
  - buffer partitioning
  - GM
  - UB
  - RegTensor
next_reads:
  - regbase_kernel_dataflow_patterns.md
  - regbase_sync_patterns.md
  - ../api/regbase_api_whitelist.md
depth: intermediate
topic_type: pattern
---

# Regbase Buffer Partitioning

This note explains where data should live during regbase execution. The main rule is to separate GM state, UB-stage buffers, and register-state temporaries instead of treating them as one pool.

## 1. The Three Storage Domains

| Domain | Typical Objects | What Belongs Here |
|---|---|---|
| GM | `GlobalTensor<T>` | full inputs, full outputs, persistent tensor state |
| UB | `LocalTensor<T>`, `TQue`, raw `__ubuf__` pointers | tile-sized staged data, ping-pong buffers, reusable tile-local scratch |
| Registers | `RegTensor<T>`, `MaskReg` | VF-local operands, masks, cast temporaries, compare results, fused arithmetic intermediates |

## 2. Unary And Binary Operator Baseline

For a simple unary or binary regbase operator:

- input tiles enter UB first
- `Compute` receives UB addresses or UB tensors
- VF loads pieces from UB into register objects
- register results are stored back into UB
- `CopyOut` writes final UB tiles back to GM

Baseline examples:

- one input / one output:
  - one input UB tile
  - one output UB tile
- two inputs / one output:
  - two input UB tiles
  - one output UB tile

Only materialize an extra UB temp if the same tile-local result must be reused across multiple VF chains or multiple output writes.

## 3. Register Temp Versus UB Temp

Prefer **register temps** when the value:

- only exists inside one VF loop
- is consumed immediately by the next register instruction
- does not need to survive across queue handoff or tile boundary

Prefer **UB temps** when the value:

- is reused by more than one VF sub-step
- must be shared across different local stages
- needs to be revisited after another asynchronous or staged action

Bad habit:

- materializing every intermediate into UB because the VF chain feels unfamiliar

Better habit:

- keep short-lived math in `RegTensor`
- promote to UB only when reuse or staging actually requires it

## 4. Dtype-Sensitive Buffer Planning

For fp16 / bf16 routes, separate:

- the UB storage type
- the compute-register type

Typical pattern:

1. UB keeps packed fp16/bf16 tiles
2. VF loads packed narrow values into a narrow `RegTensor`
3. VF casts into fp32 `RegTensor`
4. compute happens in fp32
5. result is cast back to narrow `RegTensor`
6. VF stores packed narrow result back to UB

This means extra **register** state is often required even when extra **UB** state is not.

## 5. Buffer Checklist Before Coding

Before coding, answer these:

1. Which tensors live in GM for the full kernel lifetime?
2. Which tile slices need UB residency?
3. Which intermediates can stay in registers?
4. Do any results need a second UB pass before `CopyOut`?
5. Does the dtype route need separate narrow and wide register objects?
6. Is there a ping-pong requirement, or is one active tile enough?

If these answers are unclear, the operator is not ready for implementation.

## 6. Reference Cases

- a compact unary regbase implementation
  - clear split between UB staging and register math
- a fused register-chain implementation
  - more register intermediates, but not a general outer-shell template

## Related Documents

- [[../regbase_development_guide]]
- [[regbase_kernel_dataflow_patterns]]
- [[regbase_sync_patterns]]
- [[../api/regbase_api_whitelist]]
