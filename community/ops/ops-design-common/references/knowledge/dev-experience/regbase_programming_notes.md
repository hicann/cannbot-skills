---
title: Regbase Programming Notes
purpose: Keep the regbase mental model stable by separating the outer shell, UB staging, and VF/reg-compute body.
read_when:
  - The implementation is drifting toward membase or template-default thinking.
  - You need to restate what "regbase" means before choosing APIs or writing kernel structure.
not_for:
  - Exact API signatures
  - Build, package, or install questions
keywords:
  - mental model
  - spmd
  - vec scope
  - regtensor
next_reads:
  - ../regbase_development_guide.md
  - ../patterns/regbase_kernel_dataflow_patterns.md
  - ../pitfalls/regbase_vs_membase_confusions.md
depth: foundation
topic_type: dev-experience
---

# Regbase Programming Notes

This note distills the project’s common programming-model reminders for `ascend950 / regbase`. It is meant to stabilize design judgment, especially when code starts drifting toward the wrong layer model.

## 1. Think In SPMD First

Regbase kernel work still runs in the Ascend C SPMD model:

- one kernel body executes on many cores
- each core identifies its slice by `GetBlockIdx()`
- host-side tiling and block routing decide what each core sees

If the design cannot explain the block split clearly, it is not ready for detailed API selection.

## 2. Regbase Is Register-Centric At The Compute Core

At the compute core, regbase centers on:

- `__VEC_SCOPE__`
- `RegTensor<T>`
- `MaskReg`
- `LoadDist` / `StoreDist`
- VF-safe reg-compute chains such as `Cast`, `Add`, `Mul`, `Abs`, `Div`, usually written as `AscendC::MicroAPI::*` in open-source code and exposed as `AscendC::Reg::*` in the SDK headers

Use regbase when the design is naturally expressed as a tight local compute chain over register-shaped state.

## 3. Do Not Collapse The Outer Shell Into The Compute Core

Real regbase kernels often still have an outer shell with:

- `TPipe`
- `TQue`
- `GlobalTensor`
- `LocalTensor`
- UB-level `CopyIn -> Compute -> CopyOut`

Those constructs do not automatically mean the operator has left the regbase path.

The real question is: **where does the actual math happen?**

- if the core math is expressed through `__VEC_SCOPE__`, `RegTensor`, `MaskReg`, and VF-safe `Reg` / `MicroAPI` calls, the design is still aligned with regbase
- if the core math is written as a `LocalTensor`-centric compute pipeline and the VF layer disappears, the design is drifting away from the regbase writing style

## 4. Know The Boundary Against Membase

Re-check the branch assumption when the implementation starts depending on:

- `LocalTensor` as the main compute state instead of UB staging
- arithmetic written directly over `LocalTensor` without a VF/reg-compute body
- queue handoff being treated as the whole compute model
- `DataCopyPad` and queue movement becoming the center of the design instead of a staging layer

That does not automatically make the design wrong, but it does mean the branch assumption should be re-checked instead of silently mixed.

## 5. Half Precision Usually Implies A Wider Compute Stage

For fp16 or bf16 heavy paths, the practical rule is:

- load or unpack the narrow type
- promote into a wider compute representation when the math is sensitive
- keep summaries local in fp32 when the post-process depends on them
- only cast back once the stable result is ready

This is a design choice, not a cosmetic optimization.

## 6. TilingData And TilingKey Are Part Of The Programming Model

Treat tiling structures as a first-class contract:

- host computes the tiling packet
- kernel reads it explicitly
- `TILING_KEY_IS()` selects the route

Do not leave dtype route, branch route, or reduction strategy implicit if they can be expressed in `TilingKey` and a stable tiling struct.

## Related Documents

- [[process_memory_cards]]
- [[requirements_analysis_patterns]]
- [[../regbase_development_guide]]
- [[regbase_kernel_case_notes]]
- [[../patterns/regbase_operator_patterns]]
- [[../patterns/regbase_kernel_dataflow_patterns]]
- [[../patterns/kernel_design_patterns]]
- [[../api/regbase_api_reference]]
- [[../pitfalls/regbase_vs_membase_confusions]]
