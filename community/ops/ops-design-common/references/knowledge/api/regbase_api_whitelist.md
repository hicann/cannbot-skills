---
title: Regbase API Whitelist
purpose: Act as the launch gate for allowed regbase APIs and the boundary between ordinary kernel-side APIs and VF-function-safe AscendC::Reg::* calls.
read_when:
  - You must verify whether an API family is actually allowed before coding or review.
  - The main question is whether a call belongs to the kernel shell, UB staging, or VF function body.
not_for:
  - Performance tuning
  - Existing-source case selection
keywords:
  - whitelist
  - vf function
  - AscendC::Reg
  - AscendC::MicroAPI
  - allowed api
next_reads:
  - regbase_api_reference.md
  - ../patterns/regbase_kernel_dataflow_patterns.md
  - ../pitfalls/api_misuse.md
depth: foundation
topic_type: api
---

# Regbase API Whitelist

This file is the local launch gate for regbase direct-invoke work. It separates ordinary regbase kernel-side APIs from VF-function-safe APIs so design and review do not mix `LocalTensor`-style assumptions with `AscendC::Reg::*` reg-compute semantics. In open-source kernels, the same VF-safe surface often appears through the alias `AscendC::MicroAPI::*`. If an API is absent from the lists below, treat it as unavailable until you verify it in the installed SDK headers and update the design instead of guessing.

## Header Reference Roots

Use these SDK roots as the primary verification source. Do not replace them with environment-specific absolute paths in design or review notes.

- `$ASCEND_HOME_PATH/include/ascendc/basic_api/interface/`
  Ordinary kernel-side AscendC interfaces such as `DataCopy`, vector `LocalTensor` compute, barriers, and event sync.
- `$ASCEND_HOME_PATH/include/ascendc/basic_api/interface/reg_compute/`
  `AscendC::Reg::*` interfaces and support types that are valid call targets inside VF functions.
- `$ASCEND_HOME_PATH/include/ascendc/include/adv_api/`
  Task-specific higher-level regbase helpers such as norm, pad, transpose, select, and quantization families.
- `$ASCEND_HOME_PATH/asc/impl/basic_api/kernel_macros.h`
  Defines `namespace MicroAPI = Reg;`, which is why many open-source kernels call the same VF-safe surface as `AscendC::MicroAPI::*`.

Existing knowledge cards may record the same SDK sources with the shorthand `$INCLUDE_DIR/...`. Treat that as the same logical header root, not as a second API surface.

## Scope Rules

- Ordinary regbase kernel-side APIs and VF-function APIs are not interchangeable, even when names look similar.
- When a design enters a VF function, prefer `AscendC::Reg::*` and other `reg_compute/` interfaces as the callable surface.
- When reading open-source code, treat `AscendC::MicroAPI::*` as the same VF-safe API family unless the local code adds its own wrapper layer on top.
- A VF function may call `__simd_callee__` interfaces from `reg_compute/`, but it must not treat another VF helper as a general callable utility.
- Pointer arguments passed into VF helpers must stay in `__ubuf__` address space.
- If the same capability exists in both a kernel-side and a VF-function-safe form, verify the namespace, parameter style, and mask/control contract before coding.

## Ordinary Regbase Kernel-Side APIs

These families are used from ordinary regbase kernel code. They are not the default whitelist for VF functions.

| Family | Representative APIs | VF Function? | Primary Header Root |
|---|---|---|---|
| Vector compute | `Abs`, `Exp`, `Relu`, `Sqrt`, `Rsqrt`, `Reciprocal`, `Ln`, `Log`, `Sigmoid`, `Tanh`, `LeakyRelu`, `Ceil`, `Floor`, `Round` | No | `basic_api/interface/` |
| Binary compute | `Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`, `And`, `Or`, `Xor`, `Axpy`, `FusedMulAdd`, `AddRelu`, `SubRelu` | No | `basic_api/interface/` |
| Scalar helpers | `Adds`, `Muls`, `Divs`, `Maxs`, `Mins`, `ShiftLefts`, `ShiftRights` | No | `basic_api/interface/` |
| Compare and select | `Compare`, `Compares`, `Select` | No | `basic_api/interface/` |
| Reduction | `ReduceSum`, `WholeReduceSum`, `BlockReduceSum`, `ReduceMax`, `WholeReduceMax`, `BlockReduceMax`, `ReduceMin`, `WholeReduceMin`, `BlockReduceMin`, `ReduceMean`, `ReduceProd`, `ReduceAll`, `ReduceAny` | No | `basic_api/interface/` |
| Conversion and utility | `Cast`, `ReinterpretCast`, `Duplicate`, `Arange`, `CreateVecIndex`, `Interleave`, `DeInterleave`, `Move` | No | `basic_api/interface/` |
| Data movement | `DataCopy`, `DataCopyPad`, `DataCopyGather`, `DataCopyUnAlignPre`, `DataCopyUnAlign`, `DataCopyUnAlignPost`, `Copy` | No | `basic_api/interface/` |
| Sync and event | `PipeBarrier`, `SetFlag`, `WaitFlag`, `CrossCoreSetFlag`, `CrossCoreWaitFlag`, `SyncAll` | No | `basic_api/interface/` |

## VF-Function-Safe APIs (`AscendC::Reg::*`)

These are the primary callable families when implementation enters a VF function or another reg-compute helper path. The authoritative source is `reg_compute/`.

Open-source regbase kernels often spell the same API family as `AscendC::MicroAPI::*`. Treat that as an aliasing style difference, not as a second whitelist.

| Family | Representative APIs | VF Function? | Primary Header Root |
|---|---|---|---|
| Unary reg compute | `Abs`, `Relu`, `Exp`, `Sqrt`, `Ln`, `Log`, `Log2`, `Log10`, `Neg`, `Not` | Yes | `basic_api/interface/reg_compute/` |
| Binary reg compute | `Add`, `Sub`, `Mul`, `Div`†, `Max`, `Min`, `And`, `Or`, `Xor`, `MulAddDst`, `Mull` | Yes | `basic_api/interface/reg_compute/` |
| Scalar reg compute | `Adds`, `Muls`, `Divs`, `Maxs`, `Mins`, `ShiftLefts`, `ShiftRights` | Yes | `basic_api/interface/reg_compute/` |
| Compare and select | `Compare`, `Compares`, `Select` | Yes | `basic_api/interface/reg_compute/` |
| Conversion and utility | `Cast`, `Duplicate` | Yes | `basic_api/interface/reg_compute/` |
| Reg reduction | `Reduce`, `ReduceDataBlock`, `PairReduceElem` | Yes | `basic_api/interface/reg_compute/` |
| Reg load and store | `LoadAlign`, `StoreAlign`, `LoadUnAlignPre`, `LoadUnAlign`, `StoreUnAlign`, `StoreUnAlignPost`, `Load`, `Store`, `Gather`, `Scatter` | Yes | `basic_api/interface/reg_compute/` |
| Mask and copy helpers | `MaskReg`, `CreateMask`, `UpdateMask`, `Pack`, `UnPack`, `Move`, `LocalMemBar` | Yes | `basic_api/interface/reg_compute/` |
| Traits and enums used in VF paths | `LoadDist`, `StoreDist`, `CastTrait`, `CMPMODE`, `RoundMode`, `MaskMergeMode`, `PostLiteral`, `DataCopyMode`, `ReduceType`, `PairReduce` | Yes | `basic_api/interface/reg_compute/` |

> † `Reg::Div` 默认 `INTRINSIC` 模式最大 1 ULP 误差（非 IEEE-754 correctly-rounded）。高精度场景传 `DivSpecificMode{precisionMode=true, algo=PRECISION_0ULP_FTZ_FALSE}`（差值补偿算法，0 ULP，仍为 `__simd_callee__` VF-safe）。

## Advanced Task-Specific Regbase APIs

These families exist in the SDK and are useful in regbase work, but they are not "default primitives". Verify task fit, tiling assumptions, and exact signatures before using them in generated guidance.

| Family | Representative APIs | VF Function? | Primary Header Root |
|---|---|---|---|
| Norm and mean | `LayerNorm`, `RmsNorm`, `GroupNorm`, `Normalize`, `Mean` | No | `include/adv_api/` |
| Pad and broadcast | `Pad`, `Broadcast`, `Brcb` | No | `include/adv_api/` |
| Structured select | `SelectWithBytesMask` | No | `include/adv_api/` |
| Transpose and layout change | `TransData`, `ConfusionTranspose` | No | `include/adv_api/` |
| Quantization | `AscendQuant`, `AscendDequant`, `AscendAntiQuant`, `Quantize`, `Dequantize`, `Antiquantize` | No | `include/adv_api/` |

## VF-Specific Guardrails

- Treat `AscendC::Reg::*` as the default API surface once you are inside a VF function. When source code uses `AscendC::MicroAPI::*`, read it as the same VF-safe family.
- Do not assume that a kernel-side `LocalTensor` API has the same signature, mask contract, or address assumptions as its VF-side counterpart.
- Check the VF-capable header first when the code path uses `RegTensor`, `MaskReg`, `LoadDist`, or `StoreDist`.
- Re-check mask creation and update rules before using `CreateMask`, `UpdateMask`, `Pack`, or `UnPack` inside a VF helper.
- If a helper would need to call another VF helper, stop and re-check the design. The safe callable surface inside VF functions is the `__simd_callee__` family from `reg_compute/`.

## Guardrails

- Check this file before drafting code or generated guidance.
- Use [[regbase_api_reference]] to confirm the broad regbase call family after the whitelist check.
- Use [[regbase_api_sync]] when the question is about barriers, flags, or cross-core coordination rather than pure API availability.
- Prefer `DataCopyPad` when alignment is uncertain; do not assume `DataCopy` can absorb a misaligned GM transfer.
- If the task needs an API that is not listed here, verify the SDK header family first and update the design rather than filling the gap with speculation.

## Related Documents

- [[regbase_api_reference]]
- [[regbase_api_sync]]
- [[datacopy_best_practices]]
- [[arithmetic_and_reduce]]
- [[precision_and_runtime]]
