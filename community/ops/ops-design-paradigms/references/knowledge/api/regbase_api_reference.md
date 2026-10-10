---
title: Regbase API Reference
purpose: Provide a compact regbase signature map after API availability and VF boundary have already been narrowed down.
read_when:
  - You already know the task is in regbase and need a family-level signature overview.
  - You need to map names, scopes, and call families before checking headers in detail.
not_for:
  - Exhaustive header verification
  - High-level route selection
keywords:
  - api reference
  - signatures
  - RegTensor
  - LoadDist
next_reads:
  - regbase_api_whitelist.md
  - arithmetic_and_reduce.md
  - ../patterns/regbase_kernel_dataflow_patterns.md
depth: foundation
topic_type: api
---

# Ascend C Regbase API Reference

This card is the local regbase signature map for direct-invoke work. It is intentionally compact: use it to orient yourself, then verify the exact availability and call form against the whitelist and the installed SDK headers.

## Scope

- Covers regbase-first kernel-side APIs used in ascend950 direct-invoke work.
- Separates ordinary kernel-side APIs from VF-function-safe `AscendC::Reg::*` APIs.
- Notes that open-source kernels often spell the same VF-safe surface as `AscendC::MicroAPI::*`, because `MicroAPI` is an SDK alias of `Reg`.
- Keeps membase compatibility as context only; it is not the source of truth for LocalTensor / TQue-based workflows.
- Focuses on the core types and call families that typically appear in design and implementation reviews.

## Core Types And Scope Markers

- `RegTensor<T>`: register-level tensor storage.
- `MaskReg`: mask register for vector operations.
- `__VEC_SCOPE__ { ... }`: regbase compute scope.
- `LoadDist` / `StoreDist`: distributed load and store between GM and registers.
- `CastTrait`: cast behavior control, including `RegLayout`, `SatMode`, `MaskMergeMode`, and `RoundMode`.
- VF-function-safe callable families live under `$ASCEND_HOME_PATH/include/ascendc/basic_api/interface/reg_compute/`.
- Ordinary kernel-side callable families live under `$ASCEND_HOME_PATH/include/ascendc/basic_api/interface/`.
- In local SDKs, many real kernels include `kernel_operator.h` and write these VF-safe calls as `AscendC::MicroAPI::*`; the alias is defined in `$ASCEND_HOME_PATH/asc/impl/basic_api/kernel_macros.h`.

## Compute Family

Typical regbase compute calls include:

- unary math: `Abs`, `Exp`, `Relu`, `Sqrt`, `Reciprocal`
- binary math: `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`
- scalar helpers: `Adds`, `Muls`
- compare and select: `Compare`, `Select`
- type conversion and packing: `Cast`, `Pack`, `UnPack`
- reduction-related helpers: `ReduceSum`, `WholeReduceSum`, `BlockReduceSum`

Use the whitelist first, then confirm the exact signature shape here before writing guidance. In particular, use the whitelist to decide whether the current code path needs an ordinary kernel-side API or a VF-function-safe `AscendC::Reg::*` API, which may appear in source code through the `AscendC::MicroAPI::*` alias.

## Data Movement Family

The regbase direct-invoke path also relies on a small data movement set:

- `LoadDist`
- `StoreDist`
- `DataCopy<T>`
- `DataCopyGather`
- `DataCopyUnAlignPre`
- `DataCopyUnAlign`
- `DataCopyUnAlignPost`

For GM/UB movement details, prefer [[datacopy_best_practices]] rather than trying to infer behavior from the raw signature list.

## Decision Rules

1. Check [[regbase_api_whitelist]] first.
2. Use this file to confirm the broad regbase call family and the right mental model.
3. Use [[regbase_api_sync]] when the question is about readiness, stalls, or cross-core coordination.
4. Use [[precision_and_runtime]] when the task is likely to change precision, cast timing, or host-side launch behavior.
5. If the task actually depends on a membase-only workflow, treat that as a scope change and do not force regbase semantics onto it.

## Related Documents

- [[regbase_api_whitelist]]
- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[pipeline_and_buffer]]
- [[arithmetic_and_reduce]]
- [[precision_and_runtime]]
